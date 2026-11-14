"""Код следующего уровня на каждую точку раскладки: код её места на карте (§4.3, §6.2; расширение статьи).

Вариант второго уровня «код на углопозицию». Набор стимулов — один стимул точки первой раскладки: он
зажигает точки с тем же кодом (BestMatchActivation), зажжённые точки активируют детекторы-круги
(PlaneDetectorSpace, §4.3), биты кругов дают код (PlaneDetection). Для кода, который встречается на карте один
раз, это ровно код ячейки точки (двумерная позиция, §4.3); для кода, который встречается несколько раз, —
объединение кодов мест всех его точек (§2.2.1). Кодов столько же, сколько точек первой раскладки; они несут
только топологию карты (§6.2: «коды, унаследовавшие топологию стимулов, раскладываются на следующем уровне»).

ThresholdActivation — другое правило зажигания (§6.1, §6.5): стимул зажигает все точки со сходством не ниже
λ_a, а не только самые похожие. Код точки тогда объединяет места всех похожих на неё кодов.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from damp.activation.plane_detection import BestMatch, PlaneDetection
from damp.article_refs import (
    CODE_SPACE_ACTIVATION,
    DETECTOR_HIERARCHY,
    LONG_RANGE_IMPORTANCE,
    SPARSE_BIT_VECTORS,
    STIMULUS_DETECTION,
)
from damp.encoding.bitarray import BitArray
from damp.layout.code_table import LayoutCodeTable
from damp.logging import LOGGER


def _similarity_name(similarity: str) -> str:
    """Мера сходства кодов (§2.2.4): косинус или Жаккар."""
    name = (similarity or "jaccard").lower()
    return "cosine" if name in ("cosine", "cos") else "jaccard"


class ThresholdActivation:
    """Активация кодового пространства стимулом по порогу сходства (§6.1, §6.5).

    a_ji = sim_λa(s, v_ji) с жёстким порогом; активны все точки с a > 0, то есть со сходством не ниже λ_a
    (§6.5). В отличие от BestMatchActivation, стимул зажигает не только самые похожие точки: код следующего
    уровня объединяет места всех кодов, похожих на стимул. Интерфейс тот же, что у BestMatchActivation.
    """

    def __init__(
        self,
        table: LayoutCodeTable,
        *,
        similarity: str,
        lambda_threshold: float,
        use_gpu: bool = True,
    ) -> None:
        if table.count <= 0:
            raise ValueError("layout code table must be non-empty")
        if not 0.0 < lambda_threshold <= 1.0:
            raise ValueError("lambda_threshold must be in (0, 1]: с нулевым порогом стимул зажигает всю раскладку")
        self._code_length = int(table.code_length)
        self._similarity = _similarity_name(similarity)
        self._lambda = float(lambda_threshold)
        self._row_bytes = int(table.packed.shape[1])
        self._point_count = int(table.count)
        self._unique_codes = int(np.unique(table.packed, axis=0).shape[0])
        self._ones = np.asarray(table.ones, dtype=np.float32)
        bits = table.bits()
        self._backend = "numpy"
        self._torch = None
        self._bits: object = np.ascontiguousarray(bits, dtype=np.float32)
        reason = "gpu_disabled"
        if use_gpu:
            try:
                import torch

                if torch.cuda.is_available():
                    device = torch.device("cuda")
                    self._torch = torch
                    self._bits = torch.as_tensor(np.ascontiguousarray(bits), device=device).to(torch.float32)
                    self._ones_device = torch.as_tensor(self._ones, device=device)
                    self._backend = "torch"
                else:
                    reason = "no_cuda"
            except Exception as exc:  # noqa: BLE001 — без torch или без памяти устройства считаем на хосте
                reason = repr(exc)
                self._torch = None
                self._bits = np.ascontiguousarray(bits, dtype=np.float32)
        LOGGER.event(
            "point_codes.activation.init",
            section=CODE_SPACE_ACTIVATION,
            data={
                "points": self._point_count,
                "unique_codes": self._unique_codes,
                "code_length": self._code_length,
                "similarity": self._similarity,
                "lambda_threshold": self._lambda,
                "rule": "threshold",
                "backend": self._backend,
                "fallback_reason": None if self._backend == "torch" else reason,
                "note": "a = sim_λa(s, v) (§6.1), порог жёсткий; стимул зажигает все точки со сходством ≥ λ_a (§6.5)",
            },
        )

    @property
    def code_length(self) -> int:
        return self._code_length

    @property
    def point_count(self) -> int:
        return self._point_count

    @property
    def unique_codes(self) -> int:
        return self._unique_codes

    def pack(self, stimuli: Sequence[BitArray]) -> "np.ndarray":
        """Коды стимулов BitArray → упакованные биты (k × ⌈L/8⌉), как в таблице кодов раскладки."""
        bits = np.zeros((len(stimuli), self._code_length), dtype=np.uint8)
        for row, code in enumerate(stimuli):
            if len(code) != self._code_length:
                raise ValueError("stimulus code length must match layout code length")
            bits[row] = np.frombuffer(code._bits, dtype=np.uint8, count=self._code_length)
        return LayoutCodeTable.pack_bits(bits)

    def similarities(self, packed: "np.ndarray") -> "np.ndarray":
        """Сходство стимулов (k × ⌈L/8⌉) со всеми точками раскладки (k × N, float32) (§2.2.4)."""
        stimuli = np.ascontiguousarray(packed, dtype=np.uint8)
        if stimuli.ndim != 2 or stimuli.shape[1] != self._row_bytes:
            raise ValueError("stimuli must be k × ceil(code_length / 8) packed bytes")
        bits = np.unpackbits(stimuli, axis=1, count=self._code_length)
        if self._torch is not None:
            torch = self._torch
            with torch.no_grad():
                matrix = torch.as_tensor(np.ascontiguousarray(bits), device=self._bits.device).to(torch.float32)
                common = matrix @ self._bits.T
                stimulus_ones = matrix.sum(dim=1)
                if self._similarity == "cosine":
                    denominator = torch.sqrt(stimulus_ones[:, None] * self._ones_device[None, :])
                else:
                    denominator = stimulus_ones[:, None] + self._ones_device[None, :] - common
                similarity = torch.where(denominator > 0, common / denominator, torch.zeros_like(common))
                return similarity.detach().cpu().numpy()
        matrix = bits.astype(np.float32)
        common = matrix @ self._bits.T
        stimulus_ones = matrix.sum(axis=1)
        if self._similarity == "cosine":
            denominator = np.sqrt(stimulus_ones[:, None] * self._ones[None, :])
        else:
            denominator = stimulus_ones[:, None] + self._ones[None, :] - common
        return np.divide(common, denominator, out=np.zeros_like(common), where=denominator > 0)

    def lit_points(self, packed: "np.ndarray") -> BestMatch:
        """Точки раскладки, зажжённые набором стимулов: все со сходством ≥ λ_a хотя бы с одним из них (§6.5)."""
        stimuli = np.ascontiguousarray(packed, dtype=np.uint8)
        if stimuli.ndim != 2 or stimuli.shape[1] != self._row_bytes:
            raise ValueError("stimuli must be k × ceil(code_length / 8) packed bytes")
        distinct = np.unique(stimuli, axis=0) if stimuli.shape[0] > 1 else stimuli
        similarity = self.similarities(distinct)
        above = similarity >= self._lambda
        best = similarity.max(axis=1) if similarity.size else np.zeros(0, dtype=np.float32)
        lit_stimuli = above.any(axis=1)
        exact = int(np.count_nonzero(best >= 1.0 - 1e-6))
        points = np.flatnonzero(above.any(axis=0)).astype(np.int64)
        LOGGER.event(
            "point_codes.activation",
            section=STIMULUS_DETECTION,
            data={
                "stimuli": int(distinct.shape[0]),
                "exact": exact,
                "silent": int(distinct.shape[0] - np.count_nonzero(lit_stimuli)),
                "lit_points": int(points.size),
                "weakest_similarity": float(best[lit_stimuli].min()) if lit_stimuli.any() else 0.0,
                "lambda_threshold": self._lambda,
            },
        )
        return BestMatch(
            points=points,
            stimuli=int(distinct.shape[0]),
            exact=exact,
            nearest=int(np.count_nonzero(lit_stimuli)) - exact,
            silent=int(distinct.shape[0] - np.count_nonzero(lit_stimuli)),
            weakest=float(best[lit_stimuli].min()) if lit_stimuli.any() else 0.0,
        )


@dataclass(frozen=True)
class PointCodeSummary:
    """Сводка кодов следующего уровня точек раскладки (§2, §4.5, §5.13.1)."""

    codes: int
    code_length: int
    density_mean: float
    density_min: float
    density_max: float
    source_unique_codes: int  # различных кодов первой раскладки
    unique_codes: int  # различных кодов следующего уровня
    lit_points_mean: float
    lit_points_max: int
    pairs_above_lambda: float  # доля пар точек со сходством кодов следующего уровня ≥ λ
    ordered: bool  # сходство кодов убывает с расстоянием на карте первой раскладки
    profile: tuple[tuple[float, float], ...]  # (расстояние на карте, среднее сходство кодов)


class LayoutPointCodes:
    """Коды следующего уровня всех точек раскладки: точка → код её места на карте (§4.3, §6.2).

    Каждая точка подаётся как набор из одного стимула (PlaneDetection.encode_packed). Точки с одним и тем же
    кодом получают один и тот же код следующего уровня (стимул один и тот же), поэтому он считается один раз
    на различный код. Таблица кодов следующего уровня хранит точки в том же порядке, с тем же hue (известная
    компонента стимула) и той же подписью, что таблица первой раскладки: i-я точка второй раскладки — это
    i-я точка первой.
    """

    # Точек в выборке профиля сходства (матрица выборка × все точки).
    PROFILE_SAMPLE = 1000
    # Границы интервалов расстояния на карте (в ячейках) для профиля сходства; дальше — удвоением.
    PROFILE_EDGES = (0.5, 1.5, 2.5, 3.5, 4.5, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)
    # Допуск немонотонности профиля — как в Encoder.long_range_order.
    PROFILE_TOLERANCE = 0.02
    # Интервал расстояний входит в профиль, если в нём не меньше этой доли всех пар выборки: на самых больших
    # расстояниях остаются единичные пары с точками у края карты, по ним среднее сходство не оценить.
    PROFILE_MIN_PAIR_SHARE = 1e-4
    # Точек за одну распаковку бит при расчёте профиля (матрица «все точки × биты» целиком не создаётся).
    PROFILE_CHUNK_POINTS = 2048

    def __init__(self, detection: PlaneDetection, table: LayoutCodeTable) -> None:
        if table.count <= 0:
            raise ValueError("layout code table must be non-empty")
        self._detection = detection
        self._source = table
        self._code_length = int(detection.code_length)
        count = int(table.count)
        packed = np.zeros((count, -(-self._code_length // 8)), dtype=np.uint8)
        ones = np.zeros(count, dtype=np.int32)
        lit = np.zeros(count, dtype=np.int32)
        detectors = np.zeros(count, dtype=np.int32)
        cache: dict[bytes, tuple["np.ndarray", int, int, int]] = {}
        source = table.packed
        silent = 0
        for index in range(count):
            key = source[index].tobytes()
            entry = cache.get(key)
            if entry is None:
                result = detection.encode_packed(source[index : index + 1], label=table.label(index))
                entry = (
                    np.packbits(result.bits != 0),
                    int(np.count_nonzero(result.bits)),
                    int(result.lit_points.size),
                    int(result.active_detectors.size),
                )
                cache[key] = entry
                silent += result.match.silent
            packed[index] = entry[0]
            ones[index] = entry[1]
            lit[index] = entry[2]
            detectors[index] = entry[3]
        self._packed = packed
        self._ones = ones
        self._lit_points = lit
        self._active_detectors = detectors
        self._source_unique = len(cache)
        self._silent = silent
        LOGGER.event(
            "point_codes.init",
            section=DETECTOR_HIERARCHY,
            data={
                "points": count,
                "source_unique_codes": self._source_unique,
                "source_code_length": int(table.code_length),
                "code_length": self._code_length,
                "silent_stimuli": silent,
                "empty_codes": int(np.count_nonzero(ones == 0)),
                "note": "расширение статьи: набор стимулов — один стимул точки; код следующего уровня несёт место "
                "точки на карте (§4.3), а не сам стимул; такие коды раскладываются на следующем уровне (§6.2)",
            },
        )

    def __len__(self) -> int:
        return int(self._packed.shape[0])

    @property
    def code_length(self) -> int:
        return self._code_length

    @property
    def packed(self) -> "np.ndarray":
        """Коды следующего уровня упакованными битами (точки × ⌈L/8⌉) в порядке точек первой раскладки."""
        return self._packed

    @property
    def lit_points(self) -> "np.ndarray":
        """Сколько точек первой раскладки зажёг стимул каждой точки."""
        return self._lit_points

    def table(self) -> LayoutCodeTable:
        """Таблица кодов для раскладки следующего уровня: порядок, hue и подписи — точек первой раскладки."""
        return LayoutCodeTable(
            packed=self._packed,
            code_length=self._code_length,
            hues=self._source.hues,
            label_index=self._source.label_index,
            label_values=self._source.label_values,
            ones=self._ones,
        )

    def log_summary(
        self,
        cells: "np.ndarray",
        *,
        similarity: str,
        lambda_threshold: float,
        seed: int = 0,
    ) -> PointCodeSummary:
        """Плотность кодов (§2) и профиль их сходства от расстояния между точками на карте (§4.5, §5.13.1)."""
        cells = np.asarray(cells, dtype=np.float64)
        count = len(self)
        if cells.shape != (count, 2):
            raise ValueError("cells must contain one (y, x) cell per point")
        ones = self._ones.astype(np.float64)
        unique_codes = int(np.unique(self._packed, axis=0).shape[0])
        LOGGER.event(
            "point_codes.codes",
            section=SPARSE_BIT_VECTORS,
            data={
                "codes": count,
                "code_length": self._code_length,
                "bits_mean": float(ones.mean()),
                "bits_min": int(ones.min()),
                "bits_max": int(ones.max()),
                "density_mean": float(ones.mean()) / self._code_length,
                "density_min": float(ones.min()) / self._code_length,
                "density_max": float(ones.max()) / self._code_length,
                "source_unique_codes": self._source_unique,
                "unique_codes": unique_codes,
                "lit_points_mean": float(self._lit_points.mean()),
                "lit_points_max": int(self._lit_points.max()),
                "active_detectors_mean": float(self._active_detectors.mean()),
                "note": "§2: плотность кода задаёт его длина; lit_points — сколько точек первой раскладки зажёг стимул "
                "точки (1 — код встречается на карте один раз и код следующего уровня — код его ячейки)",
            },
        )
        profile, above, ordered = self._profile(cells, similarity, lambda_threshold, seed)
        LOGGER.event(
            "point_codes.order",
            section=LONG_RANGE_IMPORTANCE,
            data={
                "sample": min(count, self.PROFILE_SAMPLE),
                "similarity": _similarity_name(similarity),
                "lambda_threshold": lambda_threshold,
                "pairs_above_lambda": above,
                "ordered": ordered,
                "profile": [(round(distance, 1), round(value, 3)) for distance, value in profile],
                "note": "среднее сходство кодов следующего уровня от расстояния между точками на карте первой "
                "раскладки; для дальнего порядка оно должно убывать на всём диапазоне (§4.5, §5.13.1)",
            },
        )
        return PointCodeSummary(
            codes=count,
            code_length=self._code_length,
            density_mean=float(ones.mean()) / self._code_length,
            density_min=float(ones.min()) / self._code_length,
            density_max=float(ones.max()) / self._code_length,
            source_unique_codes=self._source_unique,
            unique_codes=unique_codes,
            lit_points_mean=float(self._lit_points.mean()),
            lit_points_max=int(self._lit_points.max()),
            pairs_above_lambda=above,
            ordered=ordered,
            profile=tuple(profile),
        )

    def _profile(
        self, cells: "np.ndarray", similarity: str, lambda_threshold: float, seed: int
    ) -> tuple[list[tuple[float, float]], float, bool]:
        """Профиль «расстояние на карте → среднее сходство кодов» по выборке точек против всех точек."""
        count = len(self)
        if count < 2:
            return [], 0.0, False
        if count <= self.PROFILE_SAMPLE:
            rows = np.arange(count, dtype=np.int64)
        else:
            rows = np.sort(np.random.default_rng(seed).choice(count, size=self.PROFILE_SAMPLE, replace=False))
        ones = self._ones.astype(np.float32)
        sample = np.unpackbits(self._packed[rows], axis=1, count=self._code_length).astype(np.float32)
        common = np.zeros((rows.shape[0], count), dtype=np.float32)
        for start in range(0, count, self.PROFILE_CHUNK_POINTS):
            stop = min(count, start + self.PROFILE_CHUNK_POINTS)
            block = np.unpackbits(self._packed[start:stop], axis=1, count=self._code_length).astype(np.float32)
            common[:, start:stop] = sample @ block.T
        if _similarity_name(similarity) == "cosine":
            denominator = np.sqrt(ones[rows][:, None] * ones[None, :])
        else:
            denominator = ones[rows][:, None] + ones[None, :] - common
        matrix = np.divide(common, denominator, out=np.zeros_like(common), where=denominator > 0)
        distance = np.hypot(
            cells[rows, 0][:, None] - cells[None, :, 0], cells[rows, 1][:, None] - cells[None, :, 1]
        )
        # Пара точки с самой собой в профиль не входит.
        other = np.ones_like(matrix, dtype=bool)
        other[np.arange(rows.shape[0]), rows] = False
        edges = list(self.PROFILE_EDGES)
        longest = float(distance.max())
        while edges[-1] < longest:
            edges.append(edges[-1] * 2.0)
        profile: list[tuple[float, float]] = []
        least = max(1, int(self.PROFILE_MIN_PAIR_SHARE * int(other.sum())))
        for low, high in zip(edges[:-1], edges[1:]):
            inside = other & (distance >= low) & (distance < high)
            if int(inside.sum()) >= least:
                profile.append((float(distance[inside].mean()), float(matrix[inside].mean())))
        above = float(np.mean(matrix[other] >= lambda_threshold))
        values = [value for _, value in profile]
        ordered = bool(
            len(values) >= 2
            and values[0] > values[-1]
            and all(values[k + 1] <= values[k] + self.PROFILE_TOLERANCE for k in range(len(values) - 1))
        )
        return profile, above, ordered


__all__ = ["LayoutPointCodes", "PointCodeSummary", "ThresholdActivation"]
