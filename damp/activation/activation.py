from __future__ import annotations

import math
import concurrent.futures
import copy
import random
from dataclasses import dataclass
from typing import Any, Sequence

try:
    import numpy as np
except Exception:
    np = None  # type: ignore[assignment]

from damp.article_refs import (
    CODE_SPACE_ACTIVATION,
    COLOR_MERGE,
    DETECTOR_CENTER,
    DETECTOR_CLUSTERING,
    DETECTOR_CONSTRUCTION,
    DETECTOR_LEVELS,
    DETECTOR_PARAMS,
    DETECTOR_RADIUS,
    MORPHOLOGICAL_EMBEDDINGS,
    NEURAL_ANALOGY,
    OUTSIDE_ARTICLE,
    POINT_ENERGY,
    STIMULUS_DETECTION,
)
from damp.encoding.bitarray import BitArray
from damp.layout.damp_layout import Layout
from damp.logging import LOGGER


@dataclass(frozen=True)
class Detector:
    """Параметры детектора по §6.3: центр, радиус, λ_d, e_d, n_d и выходной код b_d."""

    center: tuple[float, float]
    radius: float
    lambda_threshold: float
    energy: float
    points_count: int
    output_code: BitArray
    label: str | None = None

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("detector radius must be positive")
        if self.energy <= 0:
            raise ValueError("detector energy must be positive")
        if self.points_count <= 0:
            raise ValueError("detector points_count must be positive")
        if len(self.output_code) == 0:
            raise ValueError("detector output_code must be non-empty")
        LOGGER.event(
            "activation.detector.init",
            section=DETECTOR_PARAMS,
            data={
                "center": self.center,
                "receptive_radius": self.radius,
                "lambda_threshold": self.lambda_threshold,
                "energy": self.energy,
                "points_count": self.points_count,
                "output_bits": len(self.output_code),
                "output_bit": self.output_code.active_indices(),
                "detector_label": "" if self.label is None else str(self.label),
            },
        )


@dataclass(frozen=True)
class ActivatedDetector:
    detector: Detector
    activation_level: float
    active_points: int
    index: int = -1  # номер детектора в иерархии движка; -1 — неизвестен


@dataclass(frozen=True)
class ActivationResult:
    """Реакция пространства на стимулы (§6.5): a_ji по точкам, активные детекторы D_μd и эмбеддинг C(A)."""

    activation_values: Sequence[float]
    active_detectors: list[ActivatedDetector]
    embedding: BitArray
    positions: Sequence[tuple[int, int]] | None = None
    grid_shape: tuple[int, int] | None = None

    @property
    def activation_matrix(self) -> list[list[float]]:
        """Карта активации по ячейкам сетки H×W; собирается по требованию, а не на каждую активацию."""
        if self.positions is None or self.grid_shape is None:
            raise ValueError("activation result has no layout geometry")
        height, width = self.grid_shape
        matrix = [[0.0 for _ in range(width)] for _ in range(height)]
        for idx, value in enumerate(self.activation_values):
            y, x = self.positions[idx]
            if 0 <= y < height and 0 <= x < width:
                matrix[y][x] = float(value)
        return matrix


def _similarity_name(similarity: str) -> str:
    name = (similarity or "jaccard").lower()
    if name in ("cosine", "cos"):
        return "cosine"
    # §2.2.4.2: J2(a, b) = Σ a_i·b_i / Σ max(a_i², b_i²); для битовых векторов a_i² = a_i, и J2 = |a∧b| / |a∨b| = J.
    return "jaccard"


class _ActivationBackend:
    name: str = "base"
    uses_gpu: bool = False
    device: str | None = None

    def compute(self, stimuli: Sequence[BitArray]) -> list[float]:
        raise NotImplementedError


class _PythonActivationBackend(_ActivationBackend):
    name = "python"
    uses_gpu = False

    def __init__(
        self,
        codes: Sequence[BitArray],
        ones: Sequence[int],
        similarity: str,
        lambda_threshold: float,
        eta: float | None,
    ) -> None:
        self._codes = list(codes)
        self._ones = list(ones)
        self._similarity = _similarity_name(similarity)
        self._lambda = lambda_threshold
        self._eta = eta
        self._code_length = len(self._codes[0]) if self._codes else 0

    def compute(self, stimuli: Sequence[BitArray]) -> list[float]:
        if not stimuli:
            raise ValueError("stimuli must be non-empty")
        if self._code_length <= 0:
            return []
        prepared = [(code, code.count()) for code in stimuli]
        for code, _ in prepared:
            if len(code) != self._code_length:
                raise ValueError("stimulus code length must match layout code length")
        activation: list[float] = []
        for idx, code_b in enumerate(self._codes):
            best = 0.0
            ones_b = self._ones[idx]
            for code_a, ones_a in prepared:
                sim = self._similarity_value(code_a, ones_a, code_b, ones_b)
                value = self._apply_threshold(sim)
                if value > best:
                    best = value
            activation.append(best)
        return activation

    def _similarity_value(self, code_a: BitArray, ones_a: int, code_b: BitArray, ones_b: int) -> float:
        if len(code_a) != len(code_b):
            raise ValueError("code length mismatch for similarity calculation")
        if ones_a <= 0 or ones_b <= 0:
            return 0.0
        common = code_a.common(code_b)
        if self._similarity == "cosine":
            denom = math.sqrt(ones_a * ones_b)
            return 0.0 if denom == 0 else common / denom
        union = ones_a + ones_b - common
        if union == 0:
            return 0.0
        return common / union

    def _apply_threshold(self, similarity: float) -> float:
        if similarity <= 0.0:
            return 0.0
        if self._eta is None:
            return similarity if similarity >= self._lambda else 0.0
        scaled = 1.0 / (1.0 + math.exp(-self._eta * (similarity - self._lambda)))
        return similarity * scaled


class _NumpyActivationBackend(_ActivationBackend):
    name = "numpy"
    uses_gpu = False

    def __init__(
        self,
        codes_matrix: "np.ndarray",
        ones: "np.ndarray",
        similarity: str,
        lambda_threshold: float,
        eta: float | None,
    ) -> None:
        if np is None:
            raise RuntimeError("numpy backend requested but numpy is unavailable")
        self._np = np
        self._codes = codes_matrix.astype(self._np.float32, copy=False)
        self._ones = ones.astype(self._np.float32, copy=False)
        self._similarity = _similarity_name(similarity)
        self._lambda = lambda_threshold
        self._eta = eta
        self._code_length = codes_matrix.shape[1] if codes_matrix.size > 0 else 0

    def compute(self, stimuli: Sequence[BitArray]) -> list[float]:
        if not stimuli:
            raise ValueError("stimuli must be non-empty")
        if self._code_length <= 0:
            return []
        stim_matrix = self._stack_stimuli(stimuli)
        if stim_matrix.shape[1] != self._code_length:
            raise ValueError("stimulus code length must match layout code length")
        stim_ones = stim_matrix.sum(axis=1)
        intersection = self._codes @ stim_matrix.T
        if self._similarity == "cosine":
            denom = self._np.sqrt(self._ones[:, None] * stim_ones[None, :])
            sim = self._np.divide(
                intersection,
                denom,
                out=self._np.zeros_like(intersection),
                where=denom > 0,
            )
        else:
            union = self._ones[:, None] + stim_ones[None, :] - intersection
            sim = self._np.divide(
                intersection,
                union,
                out=self._np.zeros_like(intersection),
                where=union > 0,
            )
        sim = self._apply_threshold(sim)
        if sim.size == 0:
            return self._np.zeros(self._codes.shape[0], dtype=self._np.float32)
        return sim.max(axis=1).astype(self._np.float32, copy=False)

    def _stack_stimuli(self, stimuli: Sequence[BitArray]) -> "np.ndarray":
        rows = []
        for code in stimuli:
            if len(code) != self._code_length:
                raise ValueError("stimulus code length must match layout code length")
            rows.append(self._np.frombuffer(code._bits, dtype=self._np.uint8, count=self._code_length))
        if not rows:
            return self._np.zeros((0, self._code_length), dtype=self._np.float32)
        return self._np.stack(rows, axis=0).astype(self._np.float32, copy=False)

    def _apply_threshold(self, similarity: "np.ndarray") -> "np.ndarray":
        sim = self._np.where(similarity > 0.0, similarity, self._np.zeros_like(similarity))
        if self._eta is None:
            if self._lambda <= 0.0:
                return sim
            return self._np.where(sim >= self._lambda, sim, self._np.zeros_like(sim))
        scaled = 1.0 / (1.0 + self._np.exp(-self._eta * (sim - self._lambda)))
        return sim * scaled


class _TorchActivationBackend(_ActivationBackend):
    name = "torch"
    uses_gpu = True

    def __init__(
        self,
        *,
        torch_mod: "Any",
        device: "Any",
        codes_matrix: "np.ndarray",
        ones: "np.ndarray",
        similarity: str,
        lambda_threshold: float,
        eta: float | None,
    ) -> None:
        self._torch = torch_mod
        self.device = str(device)
        self._similarity = _similarity_name(similarity)
        self._lambda = lambda_threshold
        self._eta = eta
        self._codes = self._torch.as_tensor(codes_matrix, dtype=self._torch.float32, device=device)
        self._ones = self._torch.as_tensor(ones, dtype=self._torch.float32, device=device)
        self._code_length = codes_matrix.shape[1] if codes_matrix.size > 0 else 0
        self._np = np

    def compute(self, stimuli: Sequence[BitArray]) -> list[float]:
        if not stimuli:
            raise ValueError("stimuli must be non-empty")
        torch = self._torch
        if self._code_length <= 0:
            return []
        with torch.no_grad():
            stim_matrix = self._stack_stimuli(stimuli)
            if stim_matrix.numel() == 0:
                return [0.0 for _ in range(self._codes.shape[0])]
            stim_ones = torch.sum(stim_matrix, dim=1)
            intersection = self._codes @ stim_matrix.T
            if self._similarity == "cosine":
                denom = torch.sqrt(self._ones[:, None] * stim_ones[None, :])
                sim = torch.where(denom > 0, intersection / denom, torch.zeros_like(intersection))
            else:
                union = self._ones[:, None] + stim_ones[None, :] - intersection
                sim = torch.where(union > 0, intersection / union, torch.zeros_like(intersection))
            sim = self._apply_threshold(sim)
            if sim.numel() == 0:
                return self._np.zeros(self._codes.shape[0], dtype=self._np.float32)
            best = torch.max(sim, dim=1).values
            return best.detach().cpu().numpy()

    def _stack_stimuli(self, stimuli: Sequence[BitArray]):
        torch = self._torch
        for code in stimuli:
            if len(code) != self._code_length:
                raise ValueError("stimulus code length must match layout code length")
        if not stimuli:
            return torch.zeros((0, self._code_length), dtype=torch.float32, device=self._codes.device)
        if self._np is not None:
            # Все коды стимула одной матрицей и одной загрузкой на устройство, а не по коду.
            matrix = self._np.stack(
                [self._np.frombuffer(code._bits, dtype=self._np.uint8, count=self._code_length) for code in stimuli],
                axis=0,
            ).astype(self._np.float32, copy=False)
            return torch.as_tensor(matrix, dtype=torch.float32, device=self._codes.device)
        rows = [torch.as_tensor(list(code), dtype=torch.float32, device=self._codes.device) for code in stimuli]
        return torch.stack(rows, dim=0)

    def _apply_threshold(self, similarity):
        torch = self._torch
        sim = torch.where(similarity > 0.0, similarity, torch.zeros_like(similarity))
        if self._eta is None:
            if self._lambda <= 0.0:
                return sim
            return torch.where(sim >= self._lambda, sim, torch.zeros_like(sim))
        scaled = 1.0 / (1.0 + torch.exp(-self._eta * (sim - self._lambda)))
        return sim * scaled


class ActivationMap:
    """Активация кодового пространства стимулами: a_ji = max_s sim_λ(s, v_ji) (§6.1, §6.5)."""

    def __init__(
        self,
        layout: Layout,
        *,
        lambda_threshold: float,
        eta: float | None,
        use_gpu: bool | None = None,
    ) -> None:
        if lambda_threshold < 0 or lambda_threshold > 1:
            raise ValueError("lambda_threshold must be in [0, 1]")
        if eta is not None and eta <= 0:
            raise ValueError("eta must be positive or None for a hard cutoff (§5.4)")
        self._lambda_threshold = float(lambda_threshold)
        self._eta = None if eta is None else float(eta)
        self._layout = layout
        self._points = list(getattr(layout, "_points", []))
        self._positions: list[tuple[int, int]] = list(getattr(layout, "_positions", []))
        self._similarity_mode: str = getattr(layout, "_similarity", "jaccard")
        self._point_ones = [point.ones for point in self._points]
        self._codes = [point.code for point in self._points]
        self.height = layout.height
        self.width = layout.width
        self._code_length = len(self._codes[0]) if self._codes else 0
        if len(self._codes) != len(self._positions):
            raise ValueError("layout points and positions are inconsistent")
        self._backend, backend_reason = self._select_backend(use_gpu)
        self._log_backend(self._backend, requested_gpu=use_gpu, reason=backend_reason)
        LOGGER.event(
            "activation.map.init",
            section=CODE_SPACE_ACTIVATION,
            data={
                "points": len(self._codes),
                "lambda_threshold": self._lambda_threshold,
                "eta": 0.0 if self._eta is None else self._eta,
                "eta_mode": "hard_cutoff" if self._eta is None else "sigmoid",
                "similarity": self._similarity_mode,
            },
        )

    @property
    def lambda_threshold(self) -> float:
        return self._lambda_threshold

    @property
    def eta(self) -> float | None:
        return self._eta

    @property
    def point_count(self) -> int:
        return len(self._codes)

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def backend_uses_gpu(self) -> bool:
        return bool(self._backend.uses_gpu)

    @property
    def positions(self) -> list[tuple[int, int]]:
        return self._positions

    def compute(self, stimuli: Sequence[BitArray]) -> Sequence[float]:
        """a_ji = max_s sim_λ(s, v_ji) по всем точкам V (§6.5); numpy-вектор, если numpy доступен."""
        activation = self._backend.compute(stimuli)
        if np is not None and not isinstance(activation, np.ndarray):
            activation = np.asarray(activation, dtype=np.float32)
        if len(activation) != self.point_count:
            raise ValueError("activation size must match number of layout points")
        LOGGER.event(
            "activation.map",
            section=STIMULUS_DETECTION,
            data={
                "stimuli": len(stimuli),
                "points": len(activation),
                "lambda_threshold": self._lambda_threshold,
                "eta": 0.0 if self._eta is None else self._eta,
            },
        )
        return activation

    def to_matrix(self, activation: Sequence[float]) -> list[list[float]]:
        if len(activation) != self.point_count:
            raise ValueError("activation size must match number of layout points")
        matrix = [[0.0 for _ in range(self.width)] for _ in range(self.height)]
        for idx, value in enumerate(activation):
            y, x = self._positions[idx]
            if 0 <= y < self.height and 0 <= x < self.width:
                matrix[y][x] = value
        return matrix

    def _resolve_use_gpu(self, use_gpu: bool | None) -> bool:
        if use_gpu is not None:
            return bool(use_gpu)
        return bool(getattr(self._layout, "_use_gpu", False))

    def _resolve_dense_codes(self) -> tuple["np.ndarray | None", "np.ndarray | None"]:
        if np is None or self._code_length <= 0:
            return None, None
        matrix = getattr(self._layout, "_code_matrix", None)
        ones_array = getattr(self._layout, "_ones_array", None)
        if matrix is not None and ones_array is not None:
            return matrix, ones_array
        code_matrix = np.zeros((len(self._codes), self._code_length), dtype=np.uint8)
        ones_values = np.zeros(len(self._codes), dtype=np.float32)
        for idx, code in enumerate(self._codes):
            code_matrix[idx] = np.frombuffer(code._bits, dtype=np.uint8, count=self._code_length)
            ones_values[idx] = float(self._point_ones[idx])
        return code_matrix, ones_values

    def _try_build_torch_backend(
        self,
        codes_matrix: "np.ndarray",
        ones_array: "np.ndarray",
    ) -> _ActivationBackend | None:
        try:
            import torch
        except Exception as exc:  # noqa: BLE001
            LOGGER.event(
                "activation.backend.gpu_unavailable",
                section=OUTSIDE_ARTICLE,
                data={"error": str(exc)},
            )
            return None
        device = None
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        if device is None:
            LOGGER.event(
                "activation.backend.gpu_unavailable",
                section=OUTSIDE_ARTICLE,
                data={"error": "no gpu device"},
            )
            return None
        try:
            return _TorchActivationBackend(
                torch_mod=torch,
                device=device,
                codes_matrix=codes_matrix,
                ones=ones_array,
                similarity=self._similarity_mode,
                lambda_threshold=self._lambda_threshold,
                eta=self._eta,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.event(
                "activation.backend.gpu_fallback",
                section=OUTSIDE_ARTICLE,
                data={
                    "error": str(exc),
                    "device": str(device),
                },
            )
            return None

    def _select_backend(self, use_gpu: bool | None) -> tuple[_ActivationBackend, str | None]:
        prefer_gpu = self._resolve_use_gpu(use_gpu)
        dense_codes, dense_ones = self._resolve_dense_codes()
        missing_gpu_input = prefer_gpu and (dense_codes is None or dense_ones is None)
        gpu_attempted = False
        gpu_backend: _ActivationBackend | None = None
        if prefer_gpu and dense_codes is not None and dense_ones is not None:
            gpu_attempted = True
            gpu_backend = self._try_build_torch_backend(dense_codes, dense_ones)
            if gpu_backend is not None:
                return gpu_backend, "gpu"
        if dense_codes is not None and dense_ones is not None and np is not None:
            numpy_backend = _NumpyActivationBackend(
                dense_codes,
                dense_ones,
                self._similarity_mode,
                self._lambda_threshold,
                self._eta,
            )
            reason = "gpu_fallback" if gpu_attempted else "numpy"
            return numpy_backend, reason
        python_backend = _PythonActivationBackend(
            self._codes,
            self._point_ones,
            self._similarity_mode,
            self._lambda_threshold,
            self._eta,
        )
        if missing_gpu_input:
            reason = "gpu_input_missing"
        else:
            reason = "gpu_fallback" if gpu_attempted else "python"
        return python_backend, reason

    def _log_backend(self, backend: _ActivationBackend, *, requested_gpu: bool | None, reason: str | None) -> None:
        LOGGER.event(
            "activation.backend",
            section=OUTSIDE_ARTICLE,
            data={
                "backend": backend.name,
                "device": backend.device or "cpu",
                "points": len(self._codes),
                "code_length": self._code_length,
                "similarity": self._similarity_mode,
                "requested_gpu": requested_gpu,
                "reason": "" if reason is None else reason,
            },
        )


# ----------------------------------------------------------------------------
# Построение детекторной иерархии (§6.3–§6.4.5)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectorLayerConfig:
    """Слой детекторной иерархии: порог λ_d, радиус активации r_a ≥ r_d и минимум точек (§6.4)."""

    lambda_threshold: float
    activation_radius: int
    min_points: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.lambda_threshold <= 1.0:
            raise ValueError("lambda_threshold must be in [0, 1]")
        if self.activation_radius <= 0:
            raise ValueError("activation_radius must be positive")
        if self.min_points <= 0:
            raise ValueError("min_points must be positive")


@dataclass(frozen=True)
class DetectorCandidate:
    center: tuple[float, float]
    radius: float
    energy: float
    points_count: int
    fill_factor: float
    insertion_density: float


class DbscanClusterer:
    """DBSCAN по координатам активированных точек; выбирается кластер с центром (§6.4.1)."""

    def __init__(self, *, eps: float, min_samples: int) -> None:
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if min_samples <= 0:
            raise ValueError("min_samples must be positive")
        self._eps = float(eps)
        self._min_samples = int(min_samples)

    @property
    def eps(self) -> float:
        return self._eps

    @property
    def min_samples(self) -> int:
        return self._min_samples

    def _offsets(self) -> list[tuple[int, int]]:
        """Смещения соседей в радиусе eps по сетке (включая саму точку), в порядке dy, dx по возрастанию."""
        eps_sq = self._eps * self._eps
        reach = int(math.ceil(self._eps))
        return [
            (dy, dx)
            for dy in range(-reach, reach + 1)
            for dx in range(-reach, reach + 1)
            if dy * dy + dx * dx <= eps_sq
        ]

    def cluster_of(
        self,
        positions: Sequence[tuple[int, int]],
        seed: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """Точки кластера, содержащего ``seed``; пустой список, если seed является шумом."""
        if not positions:
            return []
        if np is None:
            return self._cluster_of_python(positions, seed)
        ys = np.asarray([p[0] for p in positions], dtype=np.int64)
        xs = np.asarray([p[1] for p in positions], dtype=np.int64)
        mask = self.cluster_mask(ys, xs, (int(seed[0]), int(seed[1])))
        return sorted((int(y), int(x)) for y, x in zip(ys[mask], xs[mask]))

    def cluster_mask(
        self,
        ys: "np.ndarray",
        xs: "np.ndarray",
        seed: tuple[int, int],
    ) -> "np.ndarray":
        """Маска точек кластера, содержащего seed (все False, если seed — шум); порядок — как у входа."""
        if ys.size == 0:
            return np.zeros(0, dtype=bool)
        positions = list(zip(ys.tolist(), xs.tolist()))
        cluster = self._cluster_of_python(positions, (int(seed[0]), int(seed[1])))
        if not cluster:
            return np.zeros(ys.shape, dtype=bool)
        members = set(cluster)
        return np.fromiter((position in members for position in positions), dtype=bool, count=len(positions))

    def _cluster_of_python(
        self,
        positions: Sequence[tuple[int, int]],
        seed: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """DBSCAN на множествах без numpy (эталонная реализация)."""
        index_of = {position: index for index, position in enumerate(positions)}
        if seed not in index_of:
            return []
        offsets = self._offsets()
        position_set = set(index_of)

        def neighbours(point: tuple[int, int]) -> list[tuple[int, int]]:
            return [
                (point[0] + dy, point[1] + dx)
                for dy, dx in offsets
                if (point[0] + dy, point[1] + dx) in position_set
            ]

        seed_neighbours = neighbours(seed)
        if len(seed_neighbours) < self._min_samples:
            core_nearby = [
                point
                for point in seed_neighbours
                if point != seed and len(neighbours(point)) >= self._min_samples
            ]
            if not core_nearby:
                LOGGER.event(
                    "activation.detector.cluster.noise",
                    section=DETECTOR_CLUSTERING,
                    data={"seed": seed, "eps": self._eps, "min_samples": self._min_samples},
                )
                return []
            start = core_nearby[0]
        else:
            start = seed
        cluster: set[tuple[int, int]] = {start}
        pending = [start]
        while pending:
            point = pending.pop()
            point_neighbours = neighbours(point)
            if len(point_neighbours) < self._min_samples:
                continue
            for neighbour in point_neighbours:
                if neighbour not in cluster:
                    cluster.add(neighbour)
                    pending.append(neighbour)
        cluster.add(seed)
        return sorted(cluster)


class DetectorHierarchyBuilder:
    """Стохастическое построение слоёв детекторов по §6.4–§6.4.5.

    Для каждого слоя повторяется: случайный центр c̃ ∈ V, активация
    A_λd(c̃, r_a), кластеризация DBSCAN, взвешенный центроид, оптимальный радиус
    по максимуму заполнения, проверка перекрытий и вставка с заменой. Слой
    закончен, когда подряд ``attempts_without_change`` попыток не изменили его.
    """

    def __init__(
        self,
        *,
        layout: Layout,
        energies: Sequence[float],
        layer_configs: Sequence[DetectorLayerConfig],
        radius_min: int,
        output_bits: int,
        energy_threshold: float,
        eta: float | None,
        attempts_without_change: int,
        seed: int,
        clusterer: DbscanClusterer,
    ) -> None:
        if radius_min <= 0:
            raise ValueError("radius_min must be positive")
        if output_bits <= 0:
            raise ValueError("output_bits must be positive")
        if len(energies) != len(layout.positions()):
            raise ValueError("energies size must match layout points")
        if not layer_configs:
            raise ValueError("layer_configs must be non-empty")
        if any(layer.activation_radius < radius_min for layer in layer_configs):
            raise ValueError("activation radius must be at least radius_min")
        if attempts_without_change <= 0:
            raise ValueError("attempts_without_change must be positive")
        if eta is not None and eta <= 0:
            raise ValueError("eta must be positive or None for a hard cutoff (§5.4)")
        self._layout = layout
        self._energies = [float(value) for value in energies]
        self._layer_configs = tuple(layer_configs)
        self._radius_min = int(radius_min)
        self._output_bits = int(output_bits)
        self._energy_threshold = float(energy_threshold)
        self._eta = None if eta is None else float(eta)
        self._attempts_without_change = int(attempts_without_change)
        self._seed = int(seed)
        self._clusterer = clusterer
        self._output_rng = random.Random(seed)
        if np is None:
            raise RuntimeError("DetectorHierarchyBuilder requires numpy")
        # Координаты и энергии точек массивами: поиск точек в диске и сходство центра с диском — векторно.
        self._pos_y_np = np.asarray(layout._pos_y, dtype=np.float64)
        self._pos_x_np = np.asarray(layout._pos_x, dtype=np.float64)
        self._energies_np = np.asarray(self._energies, dtype=np.float64)
        # Точки, отсортированные по y: диск радиуса r ищется в полосе строк, а не по всем N точкам.
        self._y_order = np.argsort(self._pos_y_np, kind="stable")
        self._y_sorted = self._pos_y_np[self._y_order]
        # Коды упакованы по 8 бит в байт: общие биты центра с точками диска — AND и подсчёт единиц
        # (np.bitwise_count), в 32 раза меньше памяти, чем строки float32-матрицы кодов.
        code_matrix = getattr(layout, "_code_matrix", None)
        self._packed_codes = None
        if code_matrix is not None and hasattr(np, "bitwise_count"):
            self._packed_codes = np.packbits(np.asarray(code_matrix, dtype=np.uint8), axis=1)
        LOGGER.event(
            "activation.detector.builder.init",
            section=DETECTOR_CONSTRUCTION,
            data={
                "layer_specs": [
                    (layer.lambda_threshold, layer.activation_radius, layer.min_points)
                    for layer in self._layer_configs
                ],
                "radius_min": self._radius_min,
                "output_bits": self._output_bits,
                "energy_threshold": self._energy_threshold,
                "eta": 0.0 if self._eta is None else self._eta,
                "eta_mode": "hard_cutoff" if self._eta is None else "sigmoid",
                "attempts_without_change": self._attempts_without_change,
                "seed": seed,
                "dbscan_eps": clusterer.eps,
                "dbscan_min_samples": clusterer.min_samples,
                "points": len(self._energies),
            },
        )

    def build(self) -> list[Detector]:
        detectors: list[Detector] = []
        LOGGER.event(
            "activation.detector.levels",
            section=DETECTOR_LEVELS,
            data={"lambda_levels": [layer.lambda_threshold for layer in self._layer_configs]},
        )
        for layer_index, layer in enumerate(self._layer_configs):
            rng = random.Random(self._seed + layer_index * 1_000_003)
            candidates = self._build_layer(layer, rng)
            for candidate in candidates:
                # §6.3: выходной код b_d — вектор с одним случайным установленным битом.
                output_code = BitArray(self._output_bits)
                output_code.set(self._output_rng.randrange(self._output_bits), 1)
                detectors.append(
                    Detector(
                        center=candidate.center,
                        radius=candidate.radius,
                        lambda_threshold=layer.lambda_threshold,
                        energy=candidate.energy,
                        points_count=candidate.points_count,
                        output_code=output_code,
                        label=f"λ={layer.lambda_threshold:.2f}",
                    )
                )
            LOGGER.event(
                "activation.detector.layer",
                section=DETECTOR_CONSTRUCTION,
                data={
                    "lambda": layer.lambda_threshold,
                    "layer_detectors": len(candidates),
                    "activation_radius": layer.activation_radius,
                    "min_points": layer.min_points,
                    "fill_min": min((item.fill_factor for item in candidates), default=0.0),
                    "fill_max": max((item.fill_factor for item in candidates), default=0.0),
                    "radius_min": min((item.radius for item in candidates), default=0.0),
                    "radius_max": max((item.radius for item in candidates), default=0.0),
                },
            )
        if not detectors:
            raise RuntimeError("не удалось построить ни одного детектора для активации")
        shared_output_bits = len(detectors) - len(
            {detector.output_code.active_indices() for detector in detectors}
        )
        LOGGER.event(
            "activation.detector.hierarchy",
            section=DETECTOR_CONSTRUCTION,
            data={
                "detectors": len(detectors),
                "layers": len(self._layer_configs),
                "output_bits": self._output_bits,
                "shared_output_bits": shared_output_bits,
            },
        )
        return detectors

    def _build_layer(
        self,
        layer: DetectorLayerConfig,
        rng: random.Random,
    ) -> list[DetectorCandidate]:
        selected: list[DetectorCandidate] = []
        # Центры, радиусы и плотности выбранных детекторов массивами — проверка перекрытий (§6.4.5) одним
        # вычислением расстояний вместо перебора; пересобираются при изменении состава слоя.
        selected_arrays: tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"] | None = None
        candidate_cache: dict[int, DetectorCandidate | None] = {}
        unchanged = 0
        attempts = 0
        replacements = 0
        point_count = len(self._energies)
        while unchanged < self._attempts_without_change:
            attempts += 1
            # §6.4: случайная точка c̃ ∈ V — приближённый центр детектора.
            center_index = rng.randrange(point_count)
            if center_index in candidate_cache:
                candidate = candidate_cache[center_index]
            else:
                candidate = self._make_candidate(center_index, layer)
                candidate_cache[center_index] = candidate
            if candidate is None:
                unchanged += 1
                continue
            if selected:
                if selected_arrays is None:
                    selected_arrays = (
                        np.asarray([item.center[0] for item in selected], dtype=np.float64),
                        np.asarray([item.center[1] for item in selected], dtype=np.float64),
                        np.asarray([item.radius for item in selected], dtype=np.float64),
                        np.asarray([item.insertion_density for item in selected], dtype=np.float64),
                    )
                sel_y, sel_x, sel_r, sel_density = selected_arrays
                distance = np.hypot(sel_y - candidate.center[0], sel_x - candidate.center[1])
                overlap_mask = (distance <= sel_r) | (distance <= candidate.radius)
            else:
                overlap_mask = None
            if overlap_mask is not None and overlap_mask.any():
                # §6.4.5: при перекрытии новый детектор должен быть плотнее
                # каждого перекрываемого, иначе он отбрасывается.
                if np.any(sel_density[overlap_mask] >= candidate.insertion_density):
                    unchanged += 1
                    continue
                keep = ~overlap_mask
                selected = [item for item, kept in zip(selected, keep) if kept]
                selected.append(candidate)
                selected_arrays = None
                replacements += 1
                unchanged = 0
                continue
            selected.append(candidate)
            selected_arrays = None
            unchanged = 0
        LOGGER.event(
            "activation.detector.layer.filled",
            section=DETECTOR_CONSTRUCTION,
            data={
                "lambda": layer.lambda_threshold,
                "attempts": attempts,
                "unique_centers": len(candidate_cache),
                "replacements": replacements,
                "selected": len(selected),
                "attempts_without_change": self._attempts_without_change,
            },
        )
        return selected

    def _make_candidate(
        self,
        center_index: int,
        layer: DetectorLayerConfig,
    ) -> DetectorCandidate | None:
        lambda_value = layer.lambda_threshold
        if self._energies[center_index] < self._energy_threshold:
            return None
        center_y = self._layout._pos_y[center_index]
        center_x = self._layout._pos_x[center_index]
        # §6.4.1: активация A_λd(c̃, r_a) кодом центра в радиусе r_a; точки с e < μ не участвуют.
        disc = self._indices_in_radius_np(center_y, center_x, layer.activation_radius)
        disc = disc[self._energies_np[disc] >= self._energy_threshold]
        activation = self._threshold_similarity_np(self._similarity_row(center_index, disc), lambda_value)
        active_mask = activation > 0.0
        active_indices = disc[active_mask]
        # §6.4.2: вес точки — произведение активации и энергии.
        weights = activation[active_mask] * self._energies_np[active_indices]
        active_y = self._pos_y_np[active_indices].astype(np.int64)
        active_x = self._pos_x_np[active_indices].astype(np.int64)
        seed = (center_y, center_x)
        if not np.any((active_y == center_y) & (active_x == center_x)):
            return None
        cluster_mask = self._clusterer.cluster_mask(active_y, active_x, seed)
        cluster_count = int(cluster_mask.sum())
        if cluster_count < layer.min_points:
            return None
        LOGGER.event(
            "activation.detector.cluster",
            section=DETECTOR_CLUSTERING,
            data={
                "points": cluster_count,
                "activated": int(active_indices.size),
                "lambda": lambda_value,
                "seed": seed,
            },
        )
        cluster_w = weights[cluster_mask]
        weight_sum = float(cluster_w.sum())
        if weight_sum <= 0.0:
            return None
        cluster_y = active_y[cluster_mask].astype(np.float64)
        cluster_x = active_x[cluster_mask].astype(np.float64)
        centroid_y = float((cluster_y * cluster_w).sum() / weight_sum)
        centroid_x = float((cluster_x * cluster_w).sum() / weight_sum)
        LOGGER.event(
            "activation.detector.center",
            section=DETECTOR_CENTER,
            data={"center_y": centroid_y, "center_x": centroid_x, "weight_sum": weight_sum},
        )
        # §6.4.3: радиус по максимуму f(p) = |{q: r(q) ≤ r(p)}| / (π r(p)²).
        distances = np.sort(np.hypot(cluster_y - centroid_y, cluster_x - centroid_x))
        counts = np.arange(1, distances.size + 1, dtype=np.float64)
        candidate_radii = np.maximum(float(self._radius_min), distances)
        allowed = (candidate_radii <= layer.activation_radius) & (counts >= layer.min_points)
        if not allowed.any():
            return None
        fill = np.where(allowed, counts / (math.pi * candidate_radii * candidate_radii), -np.inf)
        best_index = int(np.argmax(fill))
        best_fill = float(fill[best_index])
        best_radius = float(candidate_radii[best_index])
        if best_radius <= 0.0:
            return None
        # §6.3, §6.4.5: после c_d и r_d поле A_λd(c_d, r_d) активируется кодом из центра детектора
        # (§6.1: все точки R(c_d, r_d), включая нулевую активацию); A_μ — точки поля с e ≥ μ,
        # n_d = |A_μ|, e_d = Σ a_ji · e_ji по A_μ.
        field_points = self._indices_in_radius_np(centroid_y, centroid_x, best_radius)
        if field_points.size == 0:
            return None
        center_code = self._center_code_index_np(field_points, centroid_y, centroid_x)
        field_energy = self._energies_np[field_points]
        in_a_mu = field_energy >= self._energy_threshold
        a_mu = field_points[in_a_mu]
        field_activation = self._threshold_similarity_np(self._similarity_row(center_code, a_mu), lambda_value)
        points_count = int(a_mu.size)
        activated_points = int(np.count_nonzero(field_activation > 0.0))
        detector_energy = float((field_activation * field_energy[in_a_mu]).sum())
        LOGGER.event(
            "activation.detector.params",
            section=DETECTOR_PARAMS,
            data={
                "center_code": int(center_code),
                "field_points": int(field_points.size),
                "n_d": points_count,
                "activated_points": activated_points,
                "e_d": detector_energy,
                "lambda": lambda_value,
                "mu": self._energy_threshold,
            },
        )
        if detector_energy <= 0.0 or points_count < layer.min_points:
            return None
        LOGGER.event(
            "activation.detector.radius",
            section=DETECTOR_RADIUS,
            data={
                "radius": best_radius,
                "fill_factor": best_fill,
                "points_count": points_count,
                "cluster_points": cluster_count,
                "energy": detector_energy,
            },
        )
        return DetectorCandidate(
            center=(centroid_y, centroid_x),
            radius=best_radius,
            energy=detector_energy,
            points_count=points_count,
            fill_factor=best_fill,
            insertion_density=points_count / best_radius,
        )

    def _similarity_row(self, center_index: int, indices: "np.ndarray") -> "np.ndarray":
        """Сходство кода центра с кодами точек indices (§2.2.4) одним матричным произведением."""
        if indices.size == 0:
            return np.zeros(0, dtype=np.float64)
        layout = self._layout
        code_matrix = layout._code_matrix
        ones = layout._ones_array
        if self._packed_codes is not None and ones is not None:
            common = np.bitwise_count(self._packed_codes[indices] & self._packed_codes[center_index]).sum(axis=1)
            ones_center = float(ones[center_index])
            ones_points = ones[indices].astype(np.float64)
            if layout._similarity == "cosine":
                denominator = np.sqrt(ones_center * ones_points)
            else:
                denominator = ones_center + ones_points - common
            row = np.divide(
                common.astype(np.float64),
                denominator,
                out=np.zeros(indices.size, dtype=np.float64),
                where=(denominator > 0) & (ones_points > 0) & (ones_center > 0),
            )
        elif code_matrix is None or ones is None:
            sim_base = getattr(layout, "_sim_base", None)
            if sim_base is not None:
                row = np.asarray(sim_base[center_index], dtype=np.float64)[indices]
            else:
                row = np.asarray([layout._similarity_value(center_index, int(i)) for i in indices], dtype=np.float64)
        else:
            if True:
                # float32 — BLAS; значения целые и < 2^24, поэтому произведение точное.
                common = code_matrix[indices].astype(np.float32) @ code_matrix[center_index].astype(np.float32)
                ones_center = float(ones[center_index])
                ones_points = ones[indices].astype(np.float64)
                if layout._similarity == "cosine":
                    denominator = np.sqrt(ones_center * ones_points)
                else:
                    denominator = ones_center + ones_points - common
                row = np.divide(
                    common.astype(np.float64),
                    denominator,
                    out=np.zeros(indices.size, dtype=np.float64),
                    where=(denominator > 0) & (ones_points > 0) & (ones_center > 0),
                )
        row = row.copy()
        row[indices == center_index] = 1.0
        return row

    def _threshold_similarity_np(self, similarity: "np.ndarray", lambda_value: float) -> "np.ndarray":
        """Порог λ_d или сигмоида η для вектора сходств (§5.4), как _threshold_similarity поэлементно."""
        positive = similarity > 0.0
        if self._eta is None:
            return np.where(positive & (similarity >= lambda_value), similarity, 0.0)
        scaled = similarity / (1.0 + np.exp(-self._eta * (similarity - lambda_value)))
        return np.where(positive, scaled, 0.0)

    def _threshold_similarity(self, similarity: float, lambda_value: float) -> float:
        if similarity <= 0.0:
            return 0.0
        if self._eta is None:
            return similarity if similarity >= lambda_value else 0.0
        return similarity / (1.0 + math.exp(-self._eta * (similarity - lambda_value)))

    def _indices_in_radius(self, center_y: float, center_x: float, radius: float) -> list[int]:
        """Точки окрестности R(c, r) = {v : ‖v − c‖ ≤ r} (§5.7, §6.1); центр и радиус могут быть вещественными."""
        indices: list[int] = []
        y_min = max(0, math.ceil(center_y - radius))
        y_max = min(self._layout.height - 1, math.floor(center_y + radius))
        for y in range(y_min, y_max + 1):
            for x in self._layout._row_occupied[y]:
                if x < center_x - radius or x > center_x + radius:
                    continue
                # Расстояние считается так же, как r(p) в §6.4.3: точка, задающая r_d, входит в поле.
                if math.hypot(y - center_y, x - center_x) > radius:
                    continue
                point_index = self._layout.grid[y][x]
                if point_index is not None:
                    indices.append(point_index)
        return indices

    def _indices_in_radius_np(self, center_y: float, center_x: float, radius: float) -> "np.ndarray":
        """R(c, r) = {v : ‖v − c‖ ≤ r} по массивам координат (§5.7, §6.1); тот же критерий, что и построчно."""
        lo = int(np.searchsorted(self._y_sorted, center_y - radius, side="left"))
        hi = int(np.searchsorted(self._y_sorted, center_y + radius, side="right"))
        band = self._y_order[lo:hi]
        band = band[np.abs(self._pos_x_np[band] - center_x) <= radius]
        inside = np.hypot(self._pos_y_np[band] - center_y, self._pos_x_np[band] - center_x) <= radius
        return np.sort(band[inside])

    def _center_code_index_np(self, candidates: "np.ndarray", center_y: float, center_x: float) -> int:
        """Код из центра детектора (§6.3): ближайшая к c_d точка поля, при равенстве — с меньшим индексом."""
        distance_sq = (self._pos_y_np[candidates] - center_y) ** 2 + (self._pos_x_np[candidates] - center_x) ** 2
        order = np.lexsort((candidates, distance_sq))
        return int(candidates[order[0]])

    def _center_code_index(self, center_y: float, center_x: float, radius: float) -> int | None:
        """Код из центра детектора (§6.3): точка, занимающая ячейку, ближайшую к c_d."""
        candidates = self._indices_in_radius(center_y, center_x, radius)
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda index: (
                (self._layout._pos_y[index] - center_y) ** 2 + (self._layout._pos_x[index] - center_x) ** 2,
                index,
            ),
        )

    @staticmethod
    def _overlaps(left: DetectorCandidate, right: DetectorCandidate) -> bool:
        """§6.4.5: центры не должны попадать в рецептивное поле друг друга."""
        distance = math.hypot(left.center[0] - right.center[0], left.center[1] - right.center[1])
        return distance <= left.radius or distance <= right.radius


# ----------------------------------------------------------------------------
# Размер выходного слоя по среднему числу активных детекторов (§6.6)
# ----------------------------------------------------------------------------


def reassign_output_bits(
    detectors: Sequence[Detector],
    *,
    output_bits: int,
    seed: int,
) -> list[Detector]:
    """Перевыдаёт детекторам случайные биты кода новой длины (§6.3)."""
    if output_bits <= 0:
        raise ValueError("output_bits must be positive")
    rng = random.Random(seed)
    resized: list[Detector] = []
    for detector in detectors:
        output_code = BitArray(output_bits)
        output_code.set(rng.randrange(output_bits), 1)
        resized.append(
            Detector(
                center=detector.center,
                radius=detector.radius,
                lambda_threshold=detector.lambda_threshold,
                energy=detector.energy,
                points_count=detector.points_count,
                output_code=output_code,
                label=detector.label,
            )
        )
    shared = len(resized) - len({item.output_code.active_indices() for item in resized})
    LOGGER.event(
        "activation.detector.output_bits.reassign",
        section=DETECTOR_PARAMS,
        data={"detectors": len(resized), "output_bits": output_bits, "shared_output_bits": shared, "seed": seed},
    )
    return resized


@dataclass(frozen=True)
class OutputCodeSize:
    output_bits: int
    saturation_limit: int
    mean_active_detectors: float
    samples: int


class DetectorOutputSizer:
    """§6.6, шаг 3: размер и насыщенность выходного слоя задаются средним числом
    детекторов, активируемых стимулами, а плотность кода не превышает заданной (§2)."""

    def __init__(
        self,
        *,
        max_density: float,
        min_output_bits: int,
        min_saturation: int,
        max_saturation: int,
    ) -> None:
        if not 0.0 < max_density <= 1.0:
            raise ValueError("max_density must be in (0, 1]")
        if min_output_bits <= 0 or min_saturation <= 0 or max_saturation < min_saturation:
            raise ValueError("invalid output code size bounds")
        self._max_density = float(max_density)
        self._min_output_bits = int(min_output_bits)
        self._min_saturation = int(min_saturation)
        self._max_saturation = int(max_saturation)

    def measure(
        self,
        engine: "ActivationEngine",
        stimuli_batches: Sequence[Sequence[BitArray]],
    ) -> OutputCodeSize:
        size, _ = self.measure_with_results(engine, stimuli_batches)
        return size

    def measure_with_results(
        self,
        engine: "ActivationEngine",
        stimuli_batches: Sequence[Sequence[BitArray]],
    ) -> tuple[OutputCodeSize, list[list[ActivatedDetector]]]:
        """Размер выхода и активные детекторы каждого калибровочного стимула (их уровни можно переиспользовать)."""
        batches = [batch for batch in stimuli_batches if batch]
        if not batches:
            raise ValueError("at least one non-empty stimuli batch is required")
        active_lists = [engine.detect(batch)[1] for batch in batches]
        counts = [len(active) for active in active_lists]
        mean_active = sum(counts) / len(counts)
        saturation = max(self._min_saturation, min(self._max_saturation, int(round(mean_active))))
        required_bits = int(math.ceil(saturation / self._max_density))
        output_bits = self._min_output_bits
        while output_bits < required_bits:
            output_bits *= 2
        size = OutputCodeSize(
            output_bits=output_bits,
            saturation_limit=saturation,
            mean_active_detectors=mean_active,
            samples=len(counts),
        )
        LOGGER.event(
            "activation.output_size",
            section=NEURAL_ANALOGY,
            data={
                "samples": len(counts),
                "mean_active_detectors": mean_active,
                "min_active_detectors": min(counts),
                "max_active_detectors": max(counts),
                "saturation_limit": saturation,
                "output_bits": output_bits,
                "max_density": self._max_density,
                "detectors": len(engine.detectors),
            },
        )
        return size, active_lists


# ----------------------------------------------------------------------------
# Детекция стимулов и структурный эмбеддинг (§6.5, §7.1.12)
# ----------------------------------------------------------------------------


class ActivationEngine:
    def __init__(
        self,
        layout: Layout,
        detectors: Sequence[Detector],
        *,
        lambda_threshold: float,
        eta: float | None,
        energy_threshold: float,
        detector_threshold: float,
        saturation_limit: int,
        energy_map: Sequence[float] | None = None,
        use_gpu: bool | None = None,
    ) -> None:
        if not detectors:
            raise ValueError("at least one detector is required to build embeddings")
        if saturation_limit <= 0:
            raise ValueError("saturation_limit must be positive")
        self._layout = layout
        self._detectors = list(detectors)
        self._activation_map = ActivationMap(
            layout,
            lambda_threshold=lambda_threshold,
            eta=eta,
            use_gpu=use_gpu,
        )
        self._energy_threshold = float(energy_threshold)
        self._detector_threshold = float(detector_threshold)
        self._saturation_limit = int(saturation_limit)
        self._energies = self._prepare_energy_map(energy_map)
        self._energies_array = np.asarray(self._energies, dtype=np.float32) if np is not None else None
        self._coverage = self._precompute_coverage()
        self._coverage_rows, self._coverage_cols = self._flatten_coverage(self._coverage)
        self._detector_energy = self._detector_energy_array()
        self._code_length = self._resolve_code_length()
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        LOGGER.event(
            "activation.engine.init",
            section=STIMULUS_DETECTION,
            data={
                "detectors": len(self._detectors),
                "points": self._activation_map.point_count,
                "energy_threshold": self._energy_threshold,
                "detector_threshold": self._detector_threshold,
                "saturation_limit": self._saturation_limit,
                "output_bits": self._code_length,
            },
        )

    @property
    def layout(self) -> Layout:
        return self._layout

    @property
    def detectors(self) -> tuple[Detector, ...]:
        return tuple(self._detectors)

    def detect(self, stimuli: Sequence[BitArray]) -> tuple[Sequence[float], list[ActivatedDetector]]:
        """Активация A(S) и активные детекторы D_μd(A) без построения эмбеддинга (§6.5)."""
        activation_values = self._activation_map.compute(stimuli)
        self._log_map(
            activation_values,
            event="activation.map.visual",
            section=STIMULUS_DETECTION,
            path="activation/map",
        )
        self._log_map(
            self._energies,
            event="activation.energy.visual",
            section=POINT_ENERGY,
            path="activation/energy",
        )
        return activation_values, self._evaluate_detectors(activation_values)

    def activate(self, stimuli: Sequence[BitArray]) -> ActivationResult:
        activation_values, active_detectors = self.detect(stimuli)
        embedding = self._build_embedding(active_detectors)
        LOGGER.event(
            "activation.embedding",
            section=STIMULUS_DETECTION,
            data={
                "active_detectors": len(active_detectors),
                "embedding_bits": embedding.count(),
                "saturation_limit": self._saturation_limit,
            },
        )
        return ActivationResult(
            activation_values=activation_values,
            active_detectors=active_detectors,
            embedding=embedding,
            positions=self._activation_map.positions,
            grid_shape=(self._activation_map.height, self._activation_map.width),
        )

    def embedding_from_active(self, active_detectors: Sequence[ActivatedDetector]) -> BitArray:
        """Эмбеддинг по уже найденным активным детекторам (их номера — в поле index).

        Уровни активации не зависят от выходных кодов и насыщенности (§6.5), поэтому после перевыдачи
        битов (§6.3) те же уровни дают эмбеддинг без повторной активации пространства.
        """
        remapped = [
            ActivatedDetector(
                detector=self._detectors[item.index],
                activation_level=item.activation_level,
                active_points=item.active_points,
                index=item.index,
            )
            for item in active_detectors
        ]
        return self._build_embedding(remapped)

    def rebind(self, detectors: Sequence[Detector], *, saturation_limit: int) -> "ActivationEngine":
        """Движок с теми же детекторами по геометрии и энергии, но новыми выходными кодами и σ.

        Карта активации (коды V на устройстве), покрытие детекторов и энергии переиспользуются.
        """
        detectors = list(detectors)
        if saturation_limit <= 0:
            raise ValueError("saturation_limit must be positive")
        if len(detectors) != len(self._detectors):
            raise ValueError("rebind requires the same number of detectors")
        for old, new in zip(self._detectors, detectors):
            if old.center != new.center or old.radius != new.radius or old.energy != new.energy:
                raise ValueError("rebind requires detectors with the same geometry and energy")
        clone = copy.copy(self)
        clone._detectors = detectors
        clone._saturation_limit = int(saturation_limit)
        clone._code_length = clone._resolve_code_length()
        clone._executor = None
        LOGGER.event(
            "activation.engine.rebind",
            section=STIMULUS_DETECTION,
            data={
                "detectors": len(detectors),
                "saturation_limit": clone._saturation_limit,
                "output_bits": clone._code_length,
                "note": "покрытие и карта активации общие: уровни E(d, A) не зависят от b_d и σ (§6.5)",
            },
        )
        return clone

    def activate_many(
        self,
        stimuli_batch: Sequence[Sequence[BitArray]],
        *,
        max_workers: int,
    ) -> list[ActivationResult]:
        """Активирует независимые стимулы параллельно, сохраняя порядок входа (§5.10)."""
        workers = int(max_workers)
        if workers <= 0:
            raise ValueError("max_workers must be positive")
        batches = list(stimuli_batch)
        if not batches:
            return []
        effective_workers = min(workers, len(batches))
        LOGGER.event(
            "activation.parallel.start",
            section=OUTSIDE_ARTICLE,
            data={
                "images": len(batches),
                "workers": effective_workers,
                "backend": self._activation_map.backend_name,
            },
        )
        if effective_workers == 1 or self._activation_map.backend_uses_gpu:
            # GPU-бэкенд: активация — несколько матричных операций, потоки под GIL только мешают.
            results = [self.activate(stimuli) for stimuli in batches]
        else:
            if self._executor is None or self._executor._max_workers < effective_workers:
                self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers)
            results = list(self._executor.map(self.activate, batches))
        LOGGER.event(
            "activation.parallel.done",
            section=OUTSIDE_ARTICLE,
            data={"images": len(results), "workers": effective_workers},
        )
        return results

    def _prepare_energy_map(self, energy_map: Sequence[float] | None) -> list[float]:
        if energy_map is None:
            return [1.0 for _ in range(self._activation_map.point_count)]
        energies = [float(value) for value in energy_map]
        if len(energies) != self._activation_map.point_count:
            raise ValueError("energy_map length must match number of layout points")
        return energies

    def _precompute_coverage(self) -> list[tuple[int, ...]]:
        """Точки рецептивного поля каждого детектора: ‖a − c_d‖ ≤ r_d (§6.5), как r(p) в §6.4.3."""
        coverage: list[tuple[int, ...] | "np.ndarray"] = []
        positions = self._activation_map.positions
        if np is not None:
            pos = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
            pos_y, pos_x = pos[:, 0], pos[:, 1]
        covered_counts: list[int] = []
        for detector in self._detectors:
            if np is not None:
                inside = np.hypot(pos_y - float(detector.center[0]), pos_x - float(detector.center[1])) <= detector.radius
                covered = np.flatnonzero(inside).astype(np.int32)
                coverage.append(covered)
            else:
                covered_list = [
                    point_idx
                    for point_idx, (y, x) in enumerate(positions)
                    if math.hypot(float(y) - float(detector.center[0]), float(x) - float(detector.center[1]))
                    <= detector.radius
                ]
                coverage.append(tuple(covered_list))
                covered = covered_list
            covered_counts.append(len(covered))
        LOGGER.event(
            "activation.detector.coverage",
            section=STIMULUS_DETECTION,
            data={
                "detectors": len(self._detectors),
                "covered_points_min": min(covered_counts, default=0),
                "covered_points_mean": sum(covered_counts) / len(covered_counts) if covered_counts else 0.0,
                "covered_points_max": max(covered_counts, default=0),
                "receptive_radius_min": min((d.radius for d in self._detectors), default=0.0),
                "receptive_radius_max": max((d.radius for d in self._detectors), default=0.0),
            },
        )
        return coverage

    @staticmethod
    def _flatten_coverage(coverage: Sequence[Sequence[int]]) -> tuple["np.ndarray | None", "np.ndarray | None"]:
        """Покрытие как разреженная матрица D×N в координатном виде (строка — детектор, столбец — точка)."""
        if np is None:
            return None, None
        rows = np.concatenate(
            [np.full(len(indices), det_idx, dtype=np.int64) for det_idx, indices in enumerate(coverage)]
            or [np.zeros(0, dtype=np.int64)]
        )
        cols = np.concatenate(
            [np.asarray(indices, dtype=np.int64) for indices in coverage] or [np.zeros(0, dtype=np.int64)]
        )
        return rows, cols

    def _detector_energy_array(self) -> "np.ndarray | None":
        if np is None:
            return None
        return np.asarray([detector.energy for detector in self._detectors], dtype=np.float64)

    def _evaluate_detectors(self, activation_values: Sequence[float]) -> list[ActivatedDetector]:
        """Уровень активации E(d, A) = Σ a·e / e_d без ограничения сверху (§6.5).

        A_μe(d, A) — точки поля детектора с e ≥ μ_e и a > 0; сумма по всем детекторам считается одним
        bincount по разреженной матрице покрытия, затем порог μ_d (§6.5).
        """
        if len(activation_values) != self._activation_map.point_count:
            raise ValueError("activation size must match number of layout points")
        if np is not None and self._coverage_rows is not None and self._energies_array is not None:
            activation_np = np.asarray(activation_values, dtype=np.float64)
            energies_np = self._energies_array.astype(np.float64, copy=False)
            valid = (energies_np >= self._energy_threshold) & (activation_np > 0.0)
            weights = np.where(valid, activation_np * energies_np, 0.0)
            detector_count = len(self._detectors)
            energy_sum = np.bincount(
                self._coverage_rows, weights=weights[self._coverage_cols], minlength=detector_count
            )
            points_used = np.bincount(
                self._coverage_rows, weights=valid[self._coverage_cols].astype(np.float64), minlength=detector_count
            )
            levels = np.divide(
                energy_sum,
                self._detector_energy,
                out=np.zeros(detector_count, dtype=np.float64),
                where=self._detector_energy > 0.0,
            )
            active_indices = np.flatnonzero(levels >= self._detector_threshold)
            active = [
                ActivatedDetector(
                    detector=self._detectors[int(idx)],
                    activation_level=float(levels[idx]),
                    active_points=int(round(points_used[idx])),
                    index=int(idx),
                )
                for idx in active_indices
            ]
            top = int(np.argmax(levels)) if detector_count else -1
            LOGGER.event(
                "activation.detector.level",
                section=STIMULUS_DETECTION,
                data={
                    "detectors": detector_count,
                    "active": len(active),
                    "detector_threshold": self._detector_threshold,
                    "level_max": float(levels[top]) if top >= 0 else 0.0,
                    "level_mean": float(levels.mean()) if detector_count else 0.0,
                    "top_detector": top,
                    "top_detector_label": "" if top < 0 or self._detectors[top].label is None else str(self._detectors[top].label),
                    "top_points_used": int(round(points_used[top])) if top >= 0 else 0,
                    "top_energy_sum": float(energy_sum[top]) if top >= 0 else 0.0,
                },
            )
            if active:
                strongest = max(active, key=lambda item: item.activation_level)
                LOGGER.event(
                    "activation.detector.active",
                    section=STIMULUS_DETECTION,
                    data={
                        "active": len(active),
                        "detector": strongest.index,
                        "detector_label": "" if strongest.detector.label is None else str(strongest.detector.label),
                        "activation_level": strongest.activation_level,
                        "points_used": strongest.active_points,
                    },
                )
            return active
        return self._evaluate_detectors_python(activation_values)

    def _evaluate_detectors_python(self, activation_values: Sequence[float]) -> list[ActivatedDetector]:
        """Тот же расчёт E(d, A) без numpy (§6.5)."""
        active: list[ActivatedDetector] = []
        for idx, detector in enumerate(self._detectors):
            energy_sum = 0.0
            points_used = 0
            for point_idx in self._coverage[idx]:
                energy = self._energies[int(point_idx)]
                if energy < self._energy_threshold:
                    continue
                act_value = float(activation_values[int(point_idx)])
                if act_value <= 0.0:
                    continue
                energy_sum += act_value * energy
                points_used += 1
            activation_level = 0.0 if detector.energy <= 0 else energy_sum / detector.energy
            if activation_level >= self._detector_threshold:
                active.append(
                    ActivatedDetector(
                        detector=detector,
                        activation_level=activation_level,
                        active_points=points_used,
                        index=idx,
                    )
                )
        LOGGER.event(
            "activation.detector.level",
            section=STIMULUS_DETECTION,
            data={"detectors": len(self._detectors), "active": len(active), "detector_threshold": self._detector_threshold},
        )
        return active

    def _build_embedding(self, active_detectors: Sequence[ActivatedDetector]) -> BitArray:
        """Colour merge кодов активных детекторов с насыщенностью ≤ σ (§3.3, §6.5, §7.1.12)."""
        merged = BitArray(self._code_length)
        if not active_detectors:
            return merged

        # §7.1.12: динамический порог μ_c выбирается так, чтобы |D_μ| ~ σ.
        selected_detectors = sorted(
            active_detectors,
            key=lambda item: item.activation_level,
            reverse=True,
        )[: self._saturation_limit]
        mu_c = selected_detectors[-1].activation_level if selected_detectors else 0.0

        # §6.5: «цвет» бита — λ_d детектора; при нехватке бюджета сохраняются
        # биты коротковолновой части (детекторы с большим λ_d), §3.3.2.
        ranked = sorted(
            selected_detectors,
            key=lambda item: (item.detector.lambda_threshold, item.activation_level),
            reverse=True,
        )

        bits_count = 0
        used_detectors = 0
        skipped_detectors = 0
        for activated in ranked:
            if bits_count >= self._saturation_limit:
                skipped_detectors += 1
                continue
            new_bits = [
                index
                for index in activated.detector.output_code.active_indices()
                if not merged[index]
            ]
            if not new_bits:
                continue
            if bits_count + len(new_bits) > self._saturation_limit:
                skipped_detectors += 1
                continue
            for index in new_bits:
                merged.set(index, 1)
            bits_count += len(new_bits)
            used_detectors += 1

        LOGGER.event(
            "activation.color_merge",
            section=COLOR_MERGE,
            data={
                "candidate_detectors": len(ranked),
                "active_detectors": len(active_detectors),
                "used_detectors": used_detectors,
                "skipped_detectors": skipped_detectors,
                "selected_bits": bits_count,
                "saturation_limit": self._saturation_limit,
                "mu_c": mu_c,
                "lambda_max": ranked[0].detector.lambda_threshold if ranked else 0.0,
                "lambda_min": ranked[-1].detector.lambda_threshold if ranked else 0.0,
            },
        )
        LOGGER.event(
            "activation.embedding.saturation",
            section=MORPHOLOGICAL_EMBEDDINGS,
            data={
                "output_bits": self._code_length,
                "active_bits": bits_count,
                "density": bits_count / self._code_length,
                "mu_c": mu_c,
            },
        )
        return merged

    def _resolve_code_length(self) -> int:
        lengths = {len(detector.output_code) for detector in self._detectors}
        if len(lengths) != 1:
            raise ValueError("all detector output codes must have the same length")
        return lengths.pop()

    @staticmethod
    def _value_to_rgba(value: float) -> tuple[int, int, int, int]:
        v = max(0.0, min(1.0, value))
        alpha = int(v * 255)
        return 255, 0, 0, alpha

    def _log_map(
        self,
        values: Sequence[float],
        *,
        event: str,
        section: str,
        path: str,
    ) -> None:
        if len(values) == 0:
            return
        if len(values) != self._activation_map.point_count:
            raise ValueError("map values length must match points count")
        if not LOGGER.should_log(event):
            return
        vmax = float(max(values))
        scale = 1.0 / vmax if vmax > 0 else 0.0
        colors = [self._value_to_rgba(float(val) * scale) for val in values]
        positions = self._activation_map.positions
        point_positions = [(x + 0.5, y + 0.5) for y, x in positions]
        LOGGER.event(
            event,
            section=section,
            data={
                "points": len(values),
                "value_max": vmax,
                "value_min": float(min(values)),
            },
            visuals=[
                LOGGER.visual_points2d(
                    path,
                    point_positions,
                    colors=colors,
                    radii=0.45,
                )
            ],
            force=True,
        )
