"""Пакетное детектирование иерархией самоорганизующихся детекторов (§6.1, §6.5).

ActivationEngine считает реакцию кодового пространства на один набор стимулов и пишет её в журнал — так
читается первая раскладка картинкой. Уровню иерархии нужно то же для сотен тысяч кодов по одному: какие
детекторы зажигает каждый код. Здесь те же формулы считаются пакетами на устройстве:

    a_j = sim_λa(s, v_j)                         — пороговая активация точки стимулом (§6.1);
    E(d) = Σ a_j · e_j / e_d по точкам поля d    — уровень активации детектора, точки с e_j ≥ μ_e (§6.5);
    детектор активен при E(d) ≥ μ_d.

На одном стимуле результат совпадает с ActivationEngine.detect.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from damp.activation.activation import Detector
from damp.article_refs import OUTSIDE_ARTICLE, STIMULUS_DETECTION
from damp.layout.damp_layout import Layout
from damp.logging import LOGGER


class HierarchyDetection:
    """Какие детекторы иерархии зажигает каждый код-стимул (§6.5), пакетами.

    Детекторы упорядочены по слоям: слой — порог λ_d, с которым детектор построен (§6.4), слой 0 — самый
    низкий порог (самые широкие области, §7.1.10). Номера детекторов в ответе — по этому порядку.
    """

    BLOCK = 2048

    def __init__(
        self,
        layout: Layout,
        detectors: Sequence[Detector],
        energies: Sequence[float],
        *,
        similarity: str,
        lambda_threshold: float,
        eta: float | None,
        energy_threshold: float,
        detector_threshold: float,
        activate_best: bool = False,
        use_gpu: bool = True,
        name: str = "layout",
    ) -> None:
        import torch

        if not detectors:
            raise ValueError("at least one detector is required")
        if similarity not in ("cosine", "jaccard"):
            raise ValueError("similarity must be cosine or jaccard")
        table = layout.code_table
        self._length = int(table.code_length)
        self._row_bytes = -(-self._length // 8)
        if len(energies) != int(table.packed.shape[0]):
            raise ValueError("energies size must match layout points")
        thresholds = sorted({float(detector.lambda_threshold) for detector in detectors})
        layer = np.asarray([thresholds.index(float(detector.lambda_threshold)) for detector in detectors], dtype=np.int64)
        order = np.argsort(layer, kind="stable")
        self.detectors = tuple(detectors[int(index)] for index in order)
        self.layer_index = layer[order]
        self.layer_thresholds = tuple(thresholds)
        self._torch = torch
        self._device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self._similarity = similarity
        self._lambda = float(lambda_threshold)
        self._eta = None if eta is None else float(eta)
        self._detector_threshold = float(detector_threshold)
        # Самая похожая точка зажигается всегда, даже если её сходство ниже λ_a: код без единой зажжённой точки
        # не оставил бы следа на карте (расширение статьи).
        self._activate_best = bool(activate_best)
        self._name = str(name)
        self._points = self._to_device(np.ascontiguousarray(table.packed))
        self._ones = self._points.sum(dim=1)
        energy = np.asarray(energies, dtype=np.float32)
        # Точки с энергией ниже μ_e в уровень активации детектора не входят (§6.5).
        weights = np.where(energy >= float(energy_threshold), energy, 0.0).astype(np.float32)
        self._weights = torch.as_tensor(weights, device=self._device)
        cells = layout.cells_array().astype(np.float64)
        centre = np.asarray([detector.center for detector in self.detectors], dtype=np.float64)
        radius = np.asarray([detector.radius for detector in self.detectors], dtype=np.float64)
        # Поле детектора — точки не дальше r_d от его центра (§6.5), как в ActivationEngine.
        inside = np.hypot(cells[:, None, 0] - centre[None, :, 0], cells[:, None, 1] - centre[None, :, 1]) <= radius[None, :]
        self._coverage = torch.as_tensor(inside.astype(np.float32), device=self._device)
        self._detector_energy = torch.as_tensor(
            np.asarray([detector.energy for detector in self.detectors], dtype=np.float32), device=self._device
        )
        LOGGER.event(
            "activation.hierarchy.init",
            section=STIMULUS_DETECTION,
            data={
                "name": self._name,
                "points": int(cells.shape[0]),
                "detectors": len(self.detectors),
                "layers": len(thresholds),
                "layer_thresholds": list(thresholds),
                "detectors_per_layer": np.bincount(self.layer_index, minlength=len(thresholds)).tolist(),
                "covered_points_mean": float(inside.sum(axis=0).mean()),
                "lambda_threshold": self._lambda,
                "eta": 0.0 if self._eta is None else self._eta,
                "energy_threshold": float(energy_threshold),
                "detector_threshold": self._detector_threshold,
                "activate_best": self._activate_best,
                "note": "детекторы упорядочены по слоям: слой 0 — самый низкий порог λ_d (самые широкие области)",
            },
        )

    def _to_device(self, packed: "np.ndarray"):  # noqa: ANN202
        bits = np.unpackbits(packed, axis=1, count=self._length)
        return self._torch.as_tensor(bits, device=self._device).to(self._torch.float32)

    def active(self, packed: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
        """Пары (номер кода, номер детектора) для всех сработавших детекторов; коды — строки × ⌈L/8⌉ байт."""
        torch = self._torch
        packed = np.ascontiguousarray(packed, dtype=np.uint8)
        if packed.ndim != 2 or packed.shape[1] != self._row_bytes:
            raise ValueError("codes must be rows × ceil(code_length / 8) packed bytes")
        rows, columns, lit = [], [], 0
        with torch.no_grad():
            for start in range(0, packed.shape[0], self.BLOCK):
                block = self._to_device(packed[start : start + self.BLOCK])
                ones = block.sum(dim=1)
                common = block @ self._points.T
                if self._similarity == "cosine":
                    denominator = torch.sqrt(ones[:, None] * self._ones[None, :])
                else:
                    denominator = ones[:, None] + self._ones[None, :] - common
                similarity = torch.where(denominator > 0, common / denominator, torch.zeros_like(common))
                if self._eta is None:
                    activation = torch.where(similarity >= self._lambda, similarity, torch.zeros_like(similarity))
                else:
                    activation = similarity / (1.0 + torch.exp(-self._eta * (similarity - self._lambda)))
                if self._activate_best:
                    best = similarity.max(dim=1, keepdim=True).values
                    activation = torch.maximum(activation, torch.where(similarity >= best, similarity, torch.zeros_like(similarity)))
                lit += int((activation > 0).sum().item())
                level = ((activation * self._weights[None, :]) @ self._coverage) / self._detector_energy[None, :]
                found = torch.nonzero(level >= self._detector_threshold)
                rows.append(found[:, 0].cpu().numpy() + start)
                columns.append(found[:, 1].cpu().numpy())
        rows = np.concatenate(rows).astype(np.int64) if rows else np.zeros(0, dtype=np.int64)
        columns = np.concatenate(columns).astype(np.int64) if columns else np.zeros(0, dtype=np.int64)
        per_code = np.bincount(rows, minlength=packed.shape[0])
        LOGGER.event(
            "activation.hierarchy.detect",
            section=STIMULUS_DETECTION,
            data={
                "name": self._name,
                "codes": int(packed.shape[0]),
                "points_lit_per_code": lit / max(1, packed.shape[0]),
                "detectors_per_code": float(per_code.mean()) if per_code.size else 0.0,
                "detectors_per_code_by_layer": [
                    float(np.count_nonzero(self.layer_index[columns] == number)) / max(1, packed.shape[0])
                    for number in range(len(self.layer_thresholds))
                ],
                "codes_without_detectors": float(np.mean(per_code == 0)) if per_code.size else 0.0,
                "note": "код зажигает все точки со сходством не ниже λ_a (§6.1); детектор активен, если энергия "
                "зажжённых точек его поля не меньше доли μ_d его собственной энергии (§6.5)",
            },
        )
        return rows, columns


class DetectedCodes:
    """Сработавшие детекторы иерархии как координата «что» (расширение статьи).

    Отвечает на те же вопросы, что Encoder у решётки кругов: слой каждого детектора и активные детекторы
    значений. Значение — номер кода в наборе, для которого детекторы уже найдены (HierarchyDetection.active).
    """

    def __init__(self, title: str, rows: "np.ndarray", detectors: "np.ndarray", layer_index: "np.ndarray", count: int) -> None:
        self.title = str(title)
        self.dimensions = (self,)
        self._layers = np.asarray(layer_index, dtype=np.int64)
        order = np.argsort(rows, kind="stable")
        self._detectors = np.asarray(detectors, dtype=np.int64)[order]
        per_code = np.bincount(np.asarray(rows, dtype=np.int64), minlength=int(count))
        self._starts = np.concatenate([[0], np.cumsum(per_code)]).astype(np.int64)
        LOGGER.event(
            "window_level.detected_codes",
            section=OUTSIDE_ARTICLE,
            data={
                "title": self.title,
                "codes": int(count),
                "detectors": int(self._layers.shape[0]),
                "pairs": int(self._detectors.shape[0]),
                "note": "координата «что» уровня — детекторы иерархии, сработавшие на код куска",
            },
        )

    def detector_layer_indices(self) -> "np.ndarray":
        return self._layers

    def active_detectors(self, dimension: int | str, values: "np.ndarray", *, nested: bool = False) -> tuple["np.ndarray", "np.ndarray"]:
        """Пары (номер значения, номер детектора); значения — номера кодов."""
        codes = np.asarray(values, dtype=np.float64).reshape(-1).astype(np.int64)
        counts = self._starts[codes + 1] - self._starts[codes]
        rows = np.repeat(np.arange(codes.shape[0], dtype=np.int64), counts)
        within = np.arange(rows.shape[0]) - np.repeat(np.cumsum(counts) - counts, counts)
        return rows, self._detectors[np.repeat(self._starts[codes], counts) + within]
