"""Детекторы-круги над раскладкой и коды следующего уровня (§4.3, §6.1, §6.2, §6.5).

Расширение статьи. В §6.4 детекторы строятся по кластерам раскладки и расставлены неравномерно; здесь
пространство раскладки заполнено кругами всех слоёв равномерно — так же, как детекторы двумерной позиции
(§4.3: «детекторы — круги на плоскости, всё пространство заполнено детекторами всех уровней иерархии»;
гл. 6: её детекторы «принципиально не отличаются» от детекторов главы 4). Правило срабатывания — правило
кодера позиции (§4.1, §4.3): круг активен, если зажжённая точка раскладки лежит в его поле.

Путь набора стимулов (например, всех углопозиций одной картинки) к коду следующего уровня:

1. каждый стимул зажигает точки раскладки с самым похожим кодом (BestMatchActivation; §6.1, §6.5);
2. зажжённые точки активируют круги, в поле которых лежат (PlaneDetectorSpace; §4.3);
3. код набора — вектор фиксированной длины: каждому кругу назначен один случайный бит цветовой полосы его
   слоя, активный круг зажигает свой бит (§4.2.1); длина кода подбирается по плотности (PlaneCodeSizer; §6.6).

Полученные коды наследуют топологию стимулов и раскладываются на следующем уровне (§6.2, §6.6).
"""
from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from damp.article_refs import (
    CODE_SPACE_ACTIVATION,
    DETECTOR_HIERARCHY,
    MAPPING_CONTINUITY,
    NEURAL_ANALOGY,
    OUTSIDE_ARTICLE,
    SIMILARITY_MEASURES,
    SPARSE_BIT_VECTORS,
    STIMULUS_DETECTION,
    TWO_DIM_CODES,
)
from damp.encoding.bitarray import BitArray
from damp.encoding.damp_encoder import DetectorCircle, Encoder, PlaneDimension
from damp.layout.code_table import LayoutCodeTable
from damp.layout.damp_layout import Layout
from damp.logging import LOGGER, RerunViewerBlueprint


def _similarity_name(similarity: str) -> str:
    """Мера сходства кодов (§2.2.4): косинус или Жаккар (для битовых векторов J2 = J, §2.2.4.2)."""
    name = (similarity or "jaccard").lower()
    return "cosine" if name in ("cosine", "cos") else "jaccard"


class _NumpyBestMatchBackend:
    """Сходство стимулов с различными кодами раскладки на хосте (§2.2.4)."""

    name = "numpy"

    def __init__(self, bits: "np.ndarray", ones: "np.ndarray", similarity: str) -> None:
        self._bits = np.ascontiguousarray(bits, dtype=np.float32)
        self._ones = np.asarray(ones, dtype=np.float32)
        self._similarity = similarity

    @property
    def count(self) -> int:
        return int(self._bits.shape[0])

    def best(self, stimuli: "np.ndarray", tie_eps: float) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
        """Для каждого стимула: наибольшее сходство и пары (стимул, код) с этим сходством (с допуском)."""
        stimuli = np.asarray(stimuli, dtype=np.float32)
        common = stimuli @ self._bits.T
        stimulus_ones = stimuli.sum(axis=1)
        if self._similarity == "cosine":
            denominator = np.sqrt(stimulus_ones[:, None] * self._ones[None, :])
        else:
            denominator = stimulus_ones[:, None] + self._ones[None, :] - common
        similarity = np.divide(common, denominator, out=np.zeros_like(common), where=denominator > 0)
        best = similarity.max(axis=1)
        rows, codes = np.nonzero(similarity >= best[:, None] - tie_eps)
        return best.astype(np.float64), rows.astype(np.int64), codes.astype(np.int64)


class _TorchBestMatchBackend:
    """То же сходство одним матричным произведением на устройстве (§5.12)."""

    name = "torch"

    def __init__(self, torch_module, device, bits: "np.ndarray", ones: "np.ndarray", similarity: str) -> None:  # noqa: ANN001
        self._torch = torch_module
        self._device = device
        self._bits = torch_module.as_tensor(np.ascontiguousarray(bits), device=device).to(torch_module.float32)
        self._ones = torch_module.as_tensor(np.asarray(ones, dtype=np.float32), device=device)
        self._similarity = similarity

    @property
    def count(self) -> int:
        return int(self._bits.shape[0])

    def best(self, stimuli: "np.ndarray", tie_eps: float) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
        torch = self._torch
        with torch.no_grad():
            matrix = torch.as_tensor(np.ascontiguousarray(stimuli), device=self._device).to(torch.float32)
            common = matrix @ self._bits.T
            stimulus_ones = matrix.sum(dim=1)
            if self._similarity == "cosine":
                denominator = torch.sqrt(stimulus_ones[:, None] * self._ones[None, :])
            else:
                denominator = stimulus_ones[:, None] + self._ones[None, :] - common
            similarity = torch.where(denominator > 0, common / denominator, torch.zeros_like(common))
            best = similarity.max(dim=1).values
            rows, codes = torch.nonzero(similarity >= best[:, None] - tie_eps, as_tuple=True)
            return (
                best.detach().cpu().numpy().astype(np.float64),
                rows.detach().cpu().numpy().astype(np.int64),
                codes.detach().cpu().numpy().astype(np.int64),
            )


@dataclass(frozen=True)
class BestMatch:
    """Точки раскладки, зажжённые набором стимулов, и как они найдены (§6.5)."""

    points: "np.ndarray"  # номера зажжённых точек раскладки по возрастанию
    stimuli: int  # различных кодов стимулов в наборе
    exact: int  # стимулов, код которых есть в раскладке (сходство 1)
    nearest: int  # стимулов, зажёгших самые похожие коды (λ_a ≤ сходство < 1)
    silent: int  # стимулов без кода со сходством ≥ λ_a: не зажгли ничего
    weakest: float  # наименьшее из наибольших сходств зажёгших стимулов


class BestMatchActivation:
    """Активация кодового пространства стимулами по самым похожим кодам (§6.1, §6.5).

    Активация точки — a_ji = sim_λa(s, v_ji) (§6.1). В §6.5 активны все точки с a > 0; здесь (расширение)
    стимул зажигает только точки, на которых его сходство наибольшее по всему пространству V, если оно
    не ниже λ_a. Для стимула, код которого есть в раскладке, это все точки с тем же кодом; для нового
    стимула — точки с самым похожим кодом (при равенстве — все такие).
    """

    # Допуск равенства сходств: точки с одинаковым наибольшим сходством зажигаются вместе.
    TIE_EPS = 1e-6
    # Предел элементов матрицы «стимулы × различные коды» за одно произведение.
    BLOCK_ELEMENTS = 48_000_000

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
        if not 0.0 <= lambda_threshold <= 1.0:
            raise ValueError("lambda_threshold must be in [0, 1]")
        self._code_length = int(table.code_length)
        self._similarity = _similarity_name(similarity)
        self._lambda = float(lambda_threshold)
        self._use_gpu = bool(use_gpu)
        packed = np.ascontiguousarray(table.packed)
        self._row_bytes = int(packed.shape[1])
        # Различные коды раскладки и точки каждого из них: одинаковые коды зажигаются вместе.
        rows = packed.view(np.dtype((np.void, self._row_bytes))).reshape(-1)
        _, first, inverse = np.unique(rows, return_index=True, return_inverse=True)
        inverse = inverse.reshape(-1)
        self._unique_packed = packed[first]
        self._unique_ones = np.asarray(table.ones)[first]
        counts = np.bincount(inverse, minlength=first.shape[0])
        self._group_order = np.argsort(inverse, kind="stable").astype(np.int64)
        self._group_starts = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        self._lookup = {self._unique_packed[index].tobytes(): index for index in range(first.shape[0])}
        self._point_count = int(table.count)
        self._backend: _NumpyBestMatchBackend | _TorchBestMatchBackend | None = None
        LOGGER.event(
            "plane_detection.activation.init",
            section=CODE_SPACE_ACTIVATION,
            data={
                "points": self._point_count,
                "unique_codes": int(first.shape[0]),
                "points_per_code_max": int(counts.max()),
                "code_length": self._code_length,
                "similarity": self._similarity,
                "lambda_threshold": self._lambda,
                "rule": "best_match",
                "note": "a = sim_λa(s, v) (§6.1); стимул зажигает точки с наибольшим сходством по V, если оно ≥ λ_a "
                "(расширение: в §6.5 активны все точки с a > 0)",
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
        return int(self._unique_packed.shape[0])

    def pack(self, stimuli: Sequence[BitArray]) -> "np.ndarray":
        """Коды стимулов BitArray → упакованные биты (k × ⌈L/8⌉), как в таблице кодов раскладки."""
        bits = np.zeros((len(stimuli), self._code_length), dtype=np.uint8)
        for row, code in enumerate(stimuli):
            if len(code) != self._code_length:
                raise ValueError("stimulus code length must match layout code length")
            bits[row] = np.frombuffer(code._bits, dtype=np.uint8, count=self._code_length)
        return LayoutCodeTable.pack_bits(bits)

    def lit_points(self, packed: "np.ndarray") -> BestMatch:
        """Точки раскладки, зажжённые набором стимулов (упакованные коды k × ⌈L/8⌉) (§6.5)."""
        stimuli = np.ascontiguousarray(packed, dtype=np.uint8)
        if stimuli.ndim != 2 or stimuli.shape[1] != self._row_bytes:
            raise ValueError("stimuli must be k × ceil(code_length / 8) packed bytes")
        # Одинаковые коды стимулов зажигают одни и те же точки: достаточно различных.
        distinct: dict[bytes, int] = {}
        for row in range(stimuli.shape[0]):
            distinct.setdefault(stimuli[row].tobytes(), row)
        exact_codes: list[int] = []
        pending: list[int] = []
        for key, row in distinct.items():
            code = self._lookup.get(key)
            if code is None:
                pending.append(row)
            else:
                exact_codes.append(code)
        nearest_codes = np.zeros(0, dtype=np.int64)
        nearest = silent = 0
        weakest = 1.0 if exact_codes else 0.0
        if pending:
            best, rows, codes = self._best(stimuli[pending])
            lit = best >= self._lambda
            nearest = int(np.count_nonzero(lit))
            silent = int(best.shape[0]) - nearest
            nearest_codes = codes[lit[rows]]
            if nearest:
                weakest = float(best[lit].min())
        code_ids = np.unique(np.concatenate([np.asarray(exact_codes, dtype=np.int64), nearest_codes]))
        if code_ids.size:
            points = np.sort(
                np.concatenate(
                    [
                        self._group_order[self._group_starts[code] : self._group_starts[code + 1]]
                        for code in code_ids.tolist()
                    ]
                )
            )
        else:
            points = np.zeros(0, dtype=np.int64)
        LOGGER.event(
            "plane_detection.activation",
            section=STIMULUS_DETECTION,
            data={
                "stimuli": len(distinct),
                "exact": len(exact_codes),
                "nearest": nearest,
                "silent": silent,
                "lit_codes": int(code_ids.size),
                "lit_points": int(points.size),
                "weakest_similarity": weakest,
                "lambda_threshold": self._lambda,
            },
        )
        return BestMatch(
            points=points,
            stimuli=len(distinct),
            exact=len(exact_codes),
            nearest=nearest,
            silent=silent,
            weakest=weakest,
        )

    def _best(self, packed: "np.ndarray") -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
        """Наибольшее сходство каждого стимула с кодами раскладки и пары (стимул, код) с этим сходством."""
        backend = self._ensure_backend()
        bits = np.unpackbits(packed, axis=1, count=self._code_length)
        block = max(1, self.BLOCK_ELEMENTS // max(1, backend.count))
        best_parts: list[np.ndarray] = []
        row_parts: list[np.ndarray] = []
        code_parts: list[np.ndarray] = []
        for start in range(0, bits.shape[0], block):
            best, rows, codes = backend.best(bits[start : start + block], self.TIE_EPS)
            best_parts.append(best)
            row_parts.append(rows + start)
            code_parts.append(codes)
        return np.concatenate(best_parts), np.concatenate(row_parts), np.concatenate(code_parts)

    def _ensure_backend(self) -> "_NumpyBestMatchBackend | _TorchBestMatchBackend":
        """Матрица различных кодов для поиска самых похожих: строится при первом стимуле не из раскладки."""
        if self._backend is not None:
            return self._backend
        bits = np.unpackbits(self._unique_packed, axis=1, count=self._code_length)
        reason = "gpu_disabled"
        if self._use_gpu:
            try:
                import torch

                if torch.cuda.is_available():
                    self._backend = _TorchBestMatchBackend(
                        torch, torch.device("cuda"), bits, self._unique_ones, self._similarity
                    )
                else:
                    reason = "no_cuda"
            except Exception as exc:  # noqa: BLE001 — без torch или без памяти устройства считаем на хосте
                reason = repr(exc)
        if self._backend is None:
            self._backend = _NumpyBestMatchBackend(bits, self._unique_ones, self._similarity)
        LOGGER.event(
            "plane_detection.activation.backend",
            section=OUTSIDE_ARTICLE,
            data={
                "backend": self._backend.name,
                "unique_codes": self._backend.count,
                "fallback_reason": None if self._backend.name == "torch" else reason,
            },
        )
        return self._backend


@dataclass(frozen=True)
class PlaneCode:
    """Код набора зажжённых точек в пространстве детекторов-кругов (§4.3, §6.5)."""

    bits: "np.ndarray"  # биты кода, uint8, длина кода
    active_detectors: "np.ndarray"  # номера активных кругов (по детекторам кодера)
    positions: int  # различных зажжённых ячеек


class PlaneDetectorSpace:
    """Пространство детекторов-кругов над раскладкой (§4.3, гл. 6).

    Круги всех слоёв заполняют плоскость раскладки; их геометрию, цветовые полосы и случайные биты задаёт
    Encoder с двумерной координатой (§4.3) — тем же способом, что слои детекторов одномерных координат
    стимула (§4.2). Позиция точки раскладки — центр её ячейки (x + 0.5, y + 0.5).
    """

    # Наибольшее число окружностей в одной записи Rerun (окно не тянет тысячи ломаных на кадр).
    VISUAL_CIRCLES_MAX = 400
    VISUAL_CIRCLE_SEGMENTS = 24

    def __init__(self, layout: Layout, encoder: Encoder) -> None:
        if len(encoder.dimensions) != 1 or not isinstance(encoder.dimensions[0], PlaneDimension):
            raise ValueError("plane detector space needs an Encoder with a single PlaneDimension (§4.3)")
        plane = encoder.dimensions[0]
        self._height = int(layout.height)
        self._width = int(layout.width)
        if plane.y_min > 0.0 or plane.x_min > 0.0 or plane.y_max < self._height or plane.x_max < self._width:
            raise ValueError("the plane of the encoder must cover the layout grid")
        self._encoder = encoder
        self._plane = plane
        self._cells = layout.cells_array()
        self._layers = encoder.detector_layer_indices()
        self._layer_count = len(plane.detector_layers)
        self._layer_sizes = np.bincount(self._layers, minlength=self._layer_count)
        LOGGER.event(
            "plane_detection.space.init",
            section=TWO_DIM_CODES,
            data={
                "height": self._height,
                "width": self._width,
                "points": int(self._cells.shape[0]),
                "detectors": len(encoder.detectors),
                "layers": self._layer_count,
                "layer_detectors": self._per_layer(self._layer_sizes),
                "code_length": encoder.code_length,
                "note": "расширение статьи: круги расставлены равномерно, как детекторы двумерной позиции (§4.3), "
                "а не по кластерам раскладки (§6.4); круг активен, если в его поле есть зажжённая точка, "
                "и зажигает свой бит кода",
            },
        )

    @staticmethod
    def plane_size(layout: Layout) -> tuple[tuple[float, float], tuple[float, float]]:
        """Плоскость двумерной координаты для раскладки: вся её сетка ячеек."""
        return (0.0, float(layout.height)), (0.0, float(layout.width))

    @staticmethod
    def _per_layer(values: "np.ndarray") -> str:
        """Значения по слоям одной строкой (от самого широкого слоя к самому узкому)."""
        return " ".join(str(int(value)) for value in values)

    @property
    def encoder(self) -> Encoder:
        return self._encoder

    @property
    def code_length(self) -> int:
        return int(self._encoder.code_length)

    @property
    def layer_count(self) -> int:
        return self._layer_count

    def layer_band(self, layer_index: int) -> tuple[int, int]:
        """Цветовая полоса слоя в коде: начало и длина (§3.2)."""
        return self._plane.layer_band(layer_index)

    def positions(self, points: "np.ndarray") -> "np.ndarray":
        """Позиции зажжённых точек на плоскости: центры их ячеек (y + 0.5, x + 0.5), без повторов."""
        points = np.asarray(points, dtype=np.int64)
        cells = np.unique(self._cells[points], axis=0) if points.size else np.zeros((0, 2), dtype=np.int64)
        return cells.astype(np.float64) + 0.5

    def detect(self, points: "np.ndarray") -> PlaneCode:
        """Код набора зажжённых точек: свой бит зажигает каждый круг, накрывающий хотя бы одну из них (§4.3)."""
        points = np.asarray(points, dtype=np.int64)
        positions = self.positions(points)
        bits, active = self._encoder.encode_set(positions)
        active_bits = int(np.count_nonzero(bits))
        layer_active = np.bincount(self._layers[active], minlength=self._layer_count)
        LOGGER.event(
            "plane_detection.detect",
            section=TWO_DIM_CODES,
            data={
                "lit_points": int(points.size),
                "lit_cells": int(positions.shape[0]),
                "active_detectors": int(active.size),
                "layer_active": self._per_layer(layer_active),
                "layer_detectors": self._per_layer(self._layer_sizes),
                "active_bits": active_bits,
                # Активных кругов больше, чем единиц: несколько кругов могут делить бит (коллизии §4.2.1).
                "shared": int(active.size) - active_bits,
                "density": active_bits / self.code_length,
            },
        )
        return PlaneCode(bits=bits, active_detectors=active, positions=int(positions.shape[0]))

    def log_visual(self, points: "np.ndarray", code: PlaneCode, *, label: object | None = None) -> None:
        """Зажжённые точки раскладки и активные круги в Rerun (ср. рис. 19, 20)."""
        event = "plane_detection.visual"
        if not LOGGER.should_log(event) or not LOGGER.visuals_enabled():
            return
        image = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        image[self._cells[:, 0], self._cells[:, 1]] = (48, 48, 48)
        lit = self._cells[np.asarray(points, dtype=np.int64)]
        image[lit[:, 0], lit[:, 1]] = (255, 255, 255)
        active = code.active_detectors
        if active.size > self.VISUAL_CIRCLES_MAX:
            active = active[np.linspace(0, active.size - 1, self.VISUAL_CIRCLES_MAX).astype(np.int64)]
        angles = np.linspace(0.0, 2.0 * math.pi, self.VISUAL_CIRCLE_SEGMENTS + 1)
        strips: list[list[tuple[float, float]]] = []
        colors: list[tuple[int, int, int]] = []
        for detector_id in active.tolist():
            circle = self._encoder.detectors[detector_id]
            if not isinstance(circle, DetectorCircle):
                continue
            strips.append(
                [
                    (circle.center_x + circle.radius * math.cos(a), circle.center_y + circle.radius * math.sin(a))
                    for a in angles
                ]
            )
            # Цвет слоя: широкий — красный, узкий — фиолетовый (§3.2, §5.14).
            hue = 270.0 * circle.layer_index / max(1, self._layer_count - 1) / 360.0
            red, green, blue = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            colors.append((int(red * 255), int(green * 255), int(blue * 255)))
        LOGGER.event(
            event,
            section=DETECTOR_HIERARCHY,
            data={
                "label": label,
                "lit_points": int(np.asarray(points).size),
                "active_detectors": int(code.active_detectors.size),
                "circles_shown": len(strips),
            },
            visuals=[
                LOGGER.visual_image(RerunViewerBlueprint.PLANE_DETECTION_MAP, image),
                LOGGER.visual_line_strips2d(
                    RerunViewerBlueprint.PLANE_DETECTION_CIRCLES, strips, colors=colors, radii=0.08
                ),
            ],
            force=True,
        )


class PlaneCodeSizer:
    """Длина кода набора стимулов по заданной плотности (§2, §6.6).

    §6.6, шаг 3: размер выходного слоя вычисляется по среднему числу детекторов, активируемых стимулами.
    Код имеет фиксированную длину, каждому кругу назначен один случайный бит цветовой полосы его слоя
    (§4.2.1), активный круг зажигает свой бит. Длина — наименьшая, при которой средняя плотность кодов
    калибровочных наборов не выше заданной (§2: не выше 25 %). Какие круги активны, от длины кода не
    зависит — от неё зависят только биты кругов, поэтому наборы активируются один раз, а пробные кодеры
    разной длины строятся без записи в журнал.
    """

    # Длины перебираются кратными этому числу: код хранится упакованным по 8 бит в байт.
    LENGTH_STEP = 8

    def __init__(self, *, target_density: float, min_length: int, max_length: int) -> None:
        if not 0.0 < target_density < 1.0:
            raise ValueError("target_density must be in (0, 1)")
        if min_length <= 0 or max_length < min_length:
            raise ValueError("code length bounds must satisfy 0 < min_length <= max_length")
        self._target = float(target_density)
        self._min_steps = -(-int(min_length) // self.LENGTH_STEP)
        self._max_steps = int(max_length) // self.LENGTH_STEP
        if self._max_steps < self._min_steps:
            raise ValueError("code length bounds leave no length that is a multiple of LENGTH_STEP")

    def fit(
        self,
        layout: Layout,
        activation: BestMatchActivation,
        stimulus_sets: Sequence["np.ndarray"],
        build_encoder: Callable[[int], Encoder],
    ) -> int:
        """Длина кода для раскладки: stimulus_sets — упакованные коды стимулов калибровочных наборов,
        build_encoder(length) — кодер двумерной позиции с кодом длины length."""
        if not stimulus_sets:
            raise ValueError("at least one calibration stimulus set is required")
        with LOGGER.quiet():
            probe = PlaneDetectorSpace(layout, build_encoder(self._max_steps * self.LENGTH_STEP))
            active_sets = [
                probe.encoder.encode_set(probe.positions(activation.lit_points(packed).points))[1]
                for packed in stimulus_sets
            ]
        trials: dict[int, float] = {}

        def density(steps: int) -> float:
            if steps not in trials:
                length = steps * self.LENGTH_STEP
                with LOGGER.quiet():
                    bits = build_encoder(length).detector_bits()
                trials[steps] = float(np.mean([np.unique(bits[active]).size for active in active_sets])) / length
            return trials[steps]

        low, high = self._min_steps, self._max_steps
        if density(low) <= self._target:
            high = low
        elif density(high) > self._target:
            low = high
        else:
            # Плотность убывает с длиной кода: ищется наименьшая длина с плотностью не выше заданной.
            while high - low > 1:
                middle = (low + high) // 2
                if density(middle) > self._target:
                    low = middle
                else:
                    high = middle
        length = high * self.LENGTH_STEP
        LOGGER.event(
            "plane_detection.code_size",
            section=NEURAL_ANALOGY,
            data={
                "samples": len(active_sets),
                "mean_active_detectors": float(np.mean([active.size for active in active_sets])),
                "min_active_detectors": int(min(active.size for active in active_sets)),
                "max_active_detectors": int(max(active.size for active in active_sets)),
                "target_density": self._target,
                "code_length": length,
                "density": density(high),
                "reached": density(high) <= self._target,
                "trials": len(trials),
                "min_length": self._min_steps * self.LENGTH_STEP,
                "max_length": self._max_steps * self.LENGTH_STEP,
                "note": "§6.6, шаг 3: размер выходного слоя — по среднему числу активных детекторов; длина кода "
                "наименьшая, при которой средняя плотность кодов калибровочных наборов не выше заданной (§2)",
            },
        )
        return length


@dataclass(frozen=True)
class PlaneDetectionResult:
    """Реакция пространства на набор стимулов и его код следующего уровня (§6.2, §6.5)."""

    bits: "np.ndarray"  # код следующего уровня: биты, uint8
    lit_points: "np.ndarray"  # зажжённые точки раскладки
    active_detectors: "np.ndarray"  # активные круги
    match: BestMatch

    @property
    def embedding(self) -> BitArray:
        """Код следующего уровня вектором BitArray (структурный эмбеддинг C(A), §6.5)."""
        return BitArray.from_bits(self.bits.tobytes(), np.flatnonzero(self.bits).tolist())


class PlaneDetection:
    """Код следующего уровня для набора стимулов (§6.2, §6.5).

    Стимулы зажигают самые похожие точки раскладки, зажжённые точки активируют детекторы-круги, биты
    активных кругов объединяются в код. Набор — всё, что подаётся вместе: все углопозиции одной картинки
    или один стимул.
    """

    def __init__(self, activation: BestMatchActivation, space: PlaneDetectorSpace) -> None:
        self._activation = activation
        self._space = space
        self._code_length = space.code_length
        LOGGER.event(
            "plane_detection.init",
            section=DETECTOR_HIERARCHY,
            data={
                "points": activation.point_count,
                "unique_codes": activation.unique_codes,
                "stimulus_code_length": activation.code_length,
                "output_bits": self._code_length,
                "note": "§6.2: код активных детекторов несёт отпечаток картины активации и наследует топологию "
                "стимулов; такие коды раскладываются на следующем уровне",
            },
        )

    @property
    def space(self) -> PlaneDetectorSpace:
        return self._space

    @property
    def code_length(self) -> int:
        return self._code_length

    def encode_packed(self, packed: "np.ndarray", *, label: object | None = None) -> PlaneDetectionResult:
        """Код следующего уровня набора стимулов, заданных упакованными кодами (k × ⌈L/8⌉)."""
        match = self._activation.lit_points(packed)
        code = self._space.detect(match.points)
        self._space.log_visual(match.points, code, label=label)
        LOGGER.event(
            "plane_detection.code",
            section=STIMULUS_DETECTION,
            data={
                "label": label,
                "stimuli": match.stimuli,
                "lit_points": int(match.points.size),
                "active_detectors": int(code.active_detectors.size),
                "code_bits": int(np.count_nonzero(code.bits)),
                "density": int(np.count_nonzero(code.bits)) / self._code_length,
            },
        )
        return PlaneDetectionResult(
            bits=code.bits,
            lit_points=match.points,
            active_detectors=code.active_detectors,
            match=match,
        )

    def activate(self, stimuli: Sequence[BitArray]) -> PlaneDetectionResult:
        """То же для кодов BitArray: интерфейс движка активации, которым пользуется декодер."""
        if not stimuli:
            raise ValueError("stimuli must be non-empty")
        return self.encode_packed(self._activation.pack(stimuli))


class StimulusSetCodes:
    """Коды следующего уровня набора картинок: упакованные биты, метки и сводка по набору (§6.2).

    Коды хранятся упакованными по 8 бит в байт, как в таблице кодов раскладки: набор сразу превращается
    в таблицу для раскладки следующего уровня (§6.6).
    """

    # Наибольшее число кодов в выборке для оценки сходства по классам (матрица выборка × выборка).
    SIMILARITY_SAMPLE = 2000

    def __init__(self, code_length: int) -> None:
        if code_length <= 0:
            raise ValueError("code_length must be positive")
        self._code_length = int(code_length)
        self._packed: list[np.ndarray] = []
        self._ones: list[int] = []
        self._labels: list[int] = []
        self._lit_points: list[int] = []
        self._active_detectors: list[int] = []
        self._silent = 0
        self._stimuli = 0

    def __len__(self) -> int:
        return len(self._labels)

    @property
    def code_length(self) -> int:
        return self._code_length

    @property
    def labels(self) -> list[int]:
        return list(self._labels)

    def add(self, label: int, result: PlaneDetectionResult) -> None:
        if result.bits.shape[0] != self._code_length:
            raise ValueError("code length mismatch")
        self._packed.append(np.packbits(result.bits != 0))
        self._ones.append(int(np.count_nonzero(result.bits)))
        self._labels.append(int(label))
        self._lit_points.append(int(result.lit_points.size))
        self._active_detectors.append(int(result.active_detectors.size))
        self._silent += result.match.silent
        self._stimuli += result.match.stimuli

    def packed(self) -> "np.ndarray":
        """Коды набора упакованными битами (картинки × ⌈L/8⌉)."""
        if not self._packed:
            return np.zeros((0, -(-self._code_length // 8)), dtype=np.uint8)
        return np.stack(self._packed, axis=0)

    def bits(self, rows: "np.ndarray | None" = None) -> "np.ndarray":
        """Биты кодов (всех или выбранных строк) матрицей uint8."""
        packed = self.packed() if rows is None else self.packed()[np.asarray(rows, dtype=np.int64)]
        return np.unpackbits(packed, axis=1, count=self._code_length)

    def embeddings(self) -> list[BitArray]:
        """Коды набора векторами BitArray — база нечёткого поиска (§2.2.5)."""
        bits = self.bits()
        return [BitArray.from_bits(bits[row].tobytes(), np.flatnonzero(bits[row]).tolist()) for row in range(len(self))]

    def table(self, *, hue_step: float) -> LayoutCodeTable:
        """Таблица кодов для раскладки следующего уровня: подпись точки — класс, hue = класс · hue_step."""
        if not self._labels:
            raise ValueError("no codes to lay out")
        values = sorted(set(self._labels))
        slots = {value: slot for slot, value in enumerate(values)}
        return LayoutCodeTable(
            packed=self.packed(),
            code_length=self._code_length,
            hues=np.asarray([label * float(hue_step) for label in self._labels], dtype=np.float64),
            label_index=np.asarray([slots[label] for label in self._labels], dtype=np.int32),
            label_values=[str(value) for value in values],
            ones=np.asarray(self._ones, dtype=np.int32),
        )

    def log_summary(
        self,
        space: PlaneDetectorSpace,
        *,
        similarity: str,
        seed: int = 0,
        quantiles: Sequence[float] = (),
    ) -> dict[str, float]:
        """Сводка набора: плотность кодов (§2), заполнение цветовых полос слоёв (§3.2), сходство по классам (§2.2.4).

        quantiles — квантили попарных сходств кодов, которые нужно вернуть (ключи similarity_q<квантиль>):
        по ним задаётся порог λ раскладки следующего уровня.
        """
        count = len(self)
        if count == 0:
            raise ValueError("no codes to summarise")
        ones = np.asarray(self._ones, dtype=np.float64)
        # Полосы и сходство считаются по выборке кодов: матрица «все коды × биты» большого набора не строится.
        rows = self._sample_rows(seed)
        bits = self.bits(rows)
        band_fill: list[str] = []
        band_constant: list[str] = []
        # Полосы слоёв; если бит круга берётся из всего кода, полоса одна — весь код.
        bands = list(dict.fromkeys(space.layer_band(layer) for layer in range(space.layer_count)))
        for start, length in bands:
            band = bits[:, start : start + length]
            band_fill.append(f"{band.mean():.2f}")
            # Биты полосы, одинаковые у всех кодов набора, не различают картинки.
            column_mean = band.mean(axis=0)
            band_constant.append(f"{float(np.mean((column_mean == 0.0) | (column_mean == 1.0))):.2f}")
        LOGGER.event(
            "plane_detection.codes",
            section=SPARSE_BIT_VECTORS,
            data={
                "codes": count,
                "code_length": self._code_length,
                "bits_mean": float(ones.mean()),
                "bits_min": int(ones.min()),
                "bits_max": int(ones.max()),
                "density_mean": float(ones.mean()) / self._code_length,
                "empty_codes": int(np.count_nonzero(ones == 0)),
                "lit_points_mean": float(np.mean(self._lit_points)),
                "active_detectors_mean": float(np.mean(self._active_detectors)),
                "silent_stimuli": self._silent,
                "stimuli": self._stimuli,
                "sample": int(rows.shape[0]),
                "bands": len(bands),
                "band_fill": " ".join(band_fill),
                "band_constant": " ".join(band_constant),
                "note": "band_fill — доля единиц в цветовой полосе слоя (от широкого слоя к узкому; одна полоса — "
                "бит круга берётся из всего кода); band_constant — доля битов полосы, одинаковых у всех кодов",
            },
        )
        stats = self._similarity_stats(similarity, rows, bits, quantiles)
        LOGGER.event(
            "plane_detection.codes.similarity",
            section=SIMILARITY_MEASURES,
            data={
                **stats,
                "similarity": _similarity_name(similarity),
                "note": "сходство кодов следующего уровня внутри класса и между классами; nearest_same_class — "
                "доля кодов, у которых самый похожий код набора того же класса",
            },
        )
        return stats

    def _sample_rows(self, seed: int) -> "np.ndarray":
        """Номера кодов выборки для сводки: весь набор, пока он не больше SIMILARITY_SAMPLE."""
        count = len(self)
        if count <= self.SIMILARITY_SAMPLE:
            return np.arange(count, dtype=np.int64)
        return np.sort(np.random.default_rng(seed).choice(count, size=self.SIMILARITY_SAMPLE, replace=False))

    @staticmethod
    def quantile_key(quantile: float) -> str:
        """Ключ квантиля попарных сходств в сводке набора."""
        return f"similarity_q{float(quantile):g}"

    def _similarity_stats(
        self,
        similarity: str,
        rows: "np.ndarray",
        bits: "np.ndarray",
        quantiles: Sequence[float] = (),
    ) -> dict[str, float]:
        bits = bits.astype(np.float32)
        labels = np.asarray(self._labels, dtype=np.int64)[rows]
        ones = bits.sum(axis=1)
        common = bits @ bits.T
        if _similarity_name(similarity) == "cosine":
            denominator = np.sqrt(ones[:, None] * ones[None, :])
        else:
            denominator = ones[:, None] + ones[None, :] - common
        matrix = np.divide(common, denominator, out=np.zeros_like(common), where=denominator > 0)
        same = labels[:, None] == labels[None, :]
        off_diagonal = ~np.eye(rows.shape[0], dtype=bool)
        same_pairs = same & off_diagonal
        other_pairs = ~same
        pairwise = matrix[off_diagonal]
        requested = {
            self.quantile_key(quantile): float(np.quantile(pairwise, float(quantile))) if pairwise.size else 0.0
            for quantile in quantiles
        }
        np.fill_diagonal(matrix, -1.0)
        nearest = matrix.argmax(axis=1)
        return {
            **requested,
            "sample": int(rows.shape[0]),
            "same_class_mean": float(matrix[same_pairs].mean()) if same_pairs.any() else 0.0,
            "other_class_mean": float(matrix[other_pairs].mean()) if other_pairs.any() else 0.0,
            "similarity_p05": float(np.quantile(pairwise, 0.05)) if pairwise.size else 0.0,
            "similarity_p50": float(np.quantile(pairwise, 0.50)) if pairwise.size else 0.0,
            "similarity_p95": float(np.quantile(pairwise, 0.95)) if pairwise.size else 0.0,
            "nearest_same_class": float(np.mean(labels[nearest] == labels)) if rows.shape[0] > 1 else 0.0,
        }


class LabelMapQuality:
    """Согласие подписей соседних точек на карте раскладки (диагностика; ср. рис. 16, §5.13.2).

    Подпись в раскладке не участвует: это известное свойство стимула (класс картинки), по которому видно,
    собрала ли раскладка похожие коды рядом. Мера — доля пар соседних занятых ячеек (по горизонтали и
    вертикали) с одинаковой подписью; для сравнения даётся её ожидание при случайной расстановке, Σ p_c².
    """

    def measure(self, layout: Layout) -> dict[str, float]:
        cells = layout.cells_array()
        labels = np.asarray(layout.code_table.label_index, dtype=np.int64)
        grid = np.full((int(layout.height), int(layout.width)), -1, dtype=np.int64)
        grid[cells[:, 0], cells[:, 1]] = labels
        pairs = same = 0
        for first, second in ((grid[:, :-1], grid[:, 1:]), (grid[:-1, :], grid[1:, :])):
            occupied = (first >= 0) & (second >= 0)
            pairs += int(np.count_nonzero(occupied))
            same += int(np.count_nonzero(occupied & (first == second)))
        shares = np.bincount(labels).astype(np.float64) / max(1, labels.shape[0])
        quality = {
            "neighbour_pairs": float(pairs),
            "same_label": same / pairs if pairs else 0.0,
            "same_label_random": float(np.sum(shares * shares)),
        }
        LOGGER.event(
            "plane_detection.map",
            section=MAPPING_CONTINUITY,
            data={
                "points": int(labels.shape[0]),
                "labels": int(np.count_nonzero(shares)),
                "neighbour_pairs": pairs,
                "same_label": quality["same_label"],
                "same_label_random": quality["same_label_random"],
                "note": "доля пар соседних занятых ячеек с одинаковой подписью; same_label_random — её ожидание "
                "при случайной расстановке (подпись в раскладке не участвует)",
            },
        )
        return quality
