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


# Журнал построения детекторов: событие activation.detector.init — раз в LOG_INTERVAL_ACTIVATION_INIT построенных
# детекторов; детекторы, распакованные из кеша, не пишутся вовсе (счёт не меняется).
_DETECTOR_INIT_COUNT = 0
_DETECTOR_INIT_QUIET = 0


class quiet_detector_init:
    """Контекст: детекторы, создаваемые внутри, не пишут activation.detector.init (распаковка из кеша)."""

    def __enter__(self) -> None:
        global _DETECTOR_INIT_QUIET
        _DETECTOR_INIT_QUIET += 1

    def __exit__(self, *exc: object) -> None:
        global _DETECTOR_INIT_QUIET
        _DETECTOR_INIT_QUIET -= 1


def _detector_init_due() -> bool:
    global _DETECTOR_INIT_COUNT
    if _DETECTOR_INIT_QUIET:
        return False
    from damp import settings

    interval = max(1, int(getattr(settings, "LOG_INTERVAL_ACTIVATION_INIT", 1)))
    count = _DETECTOR_INIT_COUNT
    _DETECTOR_INIT_COUNT = count + 1
    return count % interval == 0


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
        if not _detector_init_due():
            return
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

try:  # numba — необязательная зависимость: без неё кандидатов в детекторы строит numpy по одному центру.
    import numba
except Exception:  # noqa: BLE001
    numba = None

# Числа кандидата в детекторы (вне статьи — техника счёта): кортеж (стадия, n кластера, код центра, точек поля, n_d,
# зажжённых точек поля, Σw, c_y, c_x, e_d, r_d, f). Стадия — докуда дошёл счёт и какие события пишутся (§6.4.1–§6.4.5):
# 0 — энергия центра ниже μ (ничего); 1 — центр — шум DBSCAN; 2 — в кластере меньше точек слоя; 3 — Σw ≤ 0 (событие
# кластера); 4 — центр найден, радиуса или поля нет (+ событие центра); 5 — поле посчитано, e_d ≤ 0 или n_d мало
# (+ событие поля); 6 — кандидат (+ событие радиуса). Поля, до которых счёт не дошёл, — нули.
_CANDIDATE_INTS = 5
_CANDIDATE_FLOATS = 6
_NO_NUMBERS = (0,) * _CANDIDATE_INTS + (0.0,) * _CANDIDATE_FLOATS


def _same_numbers(left: tuple, right: tuple) -> bool:
    """Числа двух кандидатов совпадают до бита: целые — равенством, float — по битам (0.0 и −0.0 различаются)."""
    if len(left) != len(right):
        return False
    whole = 1 + _CANDIDATE_INTS
    if tuple(int(value) for value in left[:whole]) != tuple(int(value) for value in right[:whole]):
        return False
    return all(np.float64(a).tobytes() == np.float64(b).tobytes() for a, b in zip(left[whole:], right[whole:]))


def _block_sum_py(values, start, count):  # noqa: ANN001, ANN201
    """Лист попарной суммы numpy (count ≤ 128): до 8 чисел — подряд от −0.0, иначе восемь накопителей и остаток."""
    if count < 8:
        total = -0.0
        for offset in range(count):
            total += values[start + offset]
        return total
    if count <= 128:
        r0 = values[start]
        r1 = values[start + 1]
        r2 = values[start + 2]
        r3 = values[start + 3]
        r4 = values[start + 4]
        r5 = values[start + 5]
        r6 = values[start + 6]
        r7 = values[start + 7]
        offset = 8
        stop = count - count % 8
        while offset < stop:
            r0 += values[start + offset]
            r1 += values[start + offset + 1]
            r2 += values[start + offset + 2]
            r3 += values[start + offset + 3]
            r4 += values[start + offset + 4]
            r5 += values[start + offset + 5]
            r6 += values[start + offset + 6]
            r7 += values[start + offset + 7]
            offset += 8
        total = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while offset < count:
            total += values[start + offset]
            offset += 1
        return total
    return 0.0


def _pairwise_sum_py(values, start, count):  # noqa: ANN001, ANN201
    """Сумма values[start : start + count] в float64 тем же порядком сложений, что у numpy (add.reduce по непрерывному
    массиву: кусок больше 128 делится пополам по кратному 8 — левая половина + правая, лист — _block_sum_py).

    Без рекурсии (numba не кеширует рекурсивные функции): то же дерево обходится явным стеком кадров «кусок, шаг»."""
    if count <= 128:
        return _block_sum(values, start, count)
    first = np.empty(64, dtype=np.int64)
    size = np.empty(64, dtype=np.int64)
    step = np.empty(64, dtype=np.int64)
    left = np.empty(64, dtype=np.float64)
    first[0] = start
    size[0] = count
    step[0] = 0
    depth = 1
    value = 0.0
    returning = False
    while depth > 0:
        top = depth - 1
        if returning:
            if step[top] == 1:
                # Левая половина посчитана — считается правая.
                left[top] = value
                step[top] = 2
                half = size[top] // 2
                half -= half % 8
                first[depth] = first[top] + half
                size[depth] = size[top] - half
                step[depth] = 0
                depth += 1
                returning = False
            else:
                value = left[top] + value
                depth -= 1
            continue
        if size[top] <= 128:
            value = _block_sum(values, first[top], size[top])
            depth -= 1
            returning = True
            continue
        half = size[top] // 2
        half -= half % 8
        step[top] = 1
        first[depth] = first[top]
        size[depth] = half
        step[depth] = 0
        depth += 1
    return value


def _array_sum_py(values, count):  # noqa: ANN001, ANN201
    """float(values[:count].sum()) numpy: начальное значение суммы +0.0 плюс попарная сумма."""
    return 0.0 + _pairwise_sum(values, 0, count)


def _table_similarity_py(common, ones_center, ones_point, cosine):  # noqa: ANN001, ANN201
    """Сходство кодов по числу общих бит (§2.2.4) — те же действия float64, что в _similarity_row."""
    if cosine:
        denominator = math.sqrt(ones_center * ones_point)
    else:
        denominator = ones_center + ones_point - float(common)
    if denominator > 0.0 and ones_point > 0.0 and ones_center > 0.0:
        return float(common) / denominator
    return 0.0


def _block_candidates_py(  # noqa: ANN201, PLR0913, PLR0912, PLR0915
    rows, window, windowed, inside, energies, pos_y, pos_x, grid, table, ones, cosine, lambda_value, energy_threshold,
    min_points, activation_radius, radius_min, fill_cell, gyration, stage, ints, floats,
):  # noqa: ANN001
    """Числа кандидатов блока центров (§6.4.1–§6.4.5) — те же действия float64 в том же порядке, что у
    DetectorHierarchyBuilder._cluster_numbers / _finish_numbers (порог λ без η, таблица общих бит, сетка карты):
    точки кластера — по возрастанию номера, суммы — попарные, как у numpy. Ответ — в stage, ints, floats (нули на входе)."""
    cells = inside.shape[1]
    count_all = energies.shape[0]
    height = grid.shape[0]
    width = grid.shape[1]
    points = np.empty(cells, dtype=np.int64)
    weights = np.empty(cells, dtype=np.float64)
    cluster_y = np.empty(cells, dtype=np.float64)
    cluster_x = np.empty(cells, dtype=np.float64)
    spare = np.empty(cells, dtype=np.float64)
    field = np.empty(count_all, dtype=np.int64)
    terms = np.empty(count_all, dtype=np.float64)
    for line in range(rows.shape[0]):
        centre = rows[line]
        if energies[centre] < energy_threshold:
            continue
        found = 0
        for cell in range(cells):
            if inside[line, cell]:
                points[found] = window[line, cell] if windowed else cell
                found += 1
        if found == 0:
            stage[line] = 1
            continue
        ints[line, 0] = found
        if found < min_points:
            stage[line] = 2
            continue
        chosen = np.sort(points[:found])
        ones_centre = ones[centre]
        # §6.4.2: вес точки — активация (порог λ) × энергия.
        for k in range(found):
            point = chosen[k]
            if point == centre:
                similarity = 1.0
            else:
                similarity = _table_similarity(table[centre, point], ones_centre, ones[point], cosine)
            activation = similarity if (similarity > 0.0 and similarity >= lambda_value) else 0.0
            weights[k] = activation * energies[point]
            cluster_y[k] = float(np.int64(pos_y[point]))
            cluster_x[k] = float(np.int64(pos_x[point]))
        weight_sum = _array_sum(weights, found)
        floats[line, 0] = weight_sum
        if weight_sum <= 0.0:
            stage[line] = 3
            continue
        for k in range(found):
            spare[k] = cluster_y[k] * weights[k]
        centroid_y = _array_sum(spare, found) / weight_sum
        for k in range(found):
            spare[k] = cluster_x[k] * weights[k]
        centroid_x = _array_sum(spare, found) / weight_sum
        floats[line, 1] = centroid_y
        floats[line, 2] = centroid_x
        stage[line] = 4
        # §6.4.3: радиус по максимуму f(p) = |{q: r(q) ≤ r(p)}| / (π (r(p) + клетка)²), первый наибольший.
        for k in range(found):
            spare[k] = math.hypot(cluster_y[k] - centroid_y, cluster_x[k] - centroid_x)
        distances = np.sort(spare[:found])
        best_index = -1
        best_fill = -np.inf
        for k in range(found):
            counted = float(k + 1)
            radius = distances[k] if distances[k] > radius_min else radius_min
            if radius <= activation_radius and counted >= min_points:
                cell_radius = radius + fill_cell
                fill = counted / (math.pi * cell_radius * cell_radius)
                if best_index < 0 or fill > best_fill:
                    best_index = k
                    best_fill = fill
        if best_index < 0:
            continue
        best_radius = distances[best_index] if distances[best_index] > radius_min else radius_min
        if gyration:
            # §6.4.4: круг того же второго момента, r_d = √2 · r_g, r_g² = Σ w·d² / Σ w.
            for k in range(found):
                dy = cluster_y[k] - centroid_y
                dx = cluster_x[k] - centroid_x
                spare[k] = weights[k] * (dy * dy + dx * dx)
            gyration_radius = math.sqrt(_array_sum(spare, found) / weight_sum)
            # min(r_a, max(r_min, √2·r_g)) как у встроенных min / max Python: первый, если второй не меньше / не больше.
            spread = math.sqrt(2.0) * gyration_radius
            widest = spread if spread > radius_min else radius_min
            best_radius = widest if widest < activation_radius else activation_radius
            covered = 0
            for k in range(found):
                if distances[k] <= best_radius:
                    covered += 1
            if covered < min_points:
                continue
            best_fill = covered / (math.pi * best_radius * best_radius)
        if best_radius <= 0.0:
            continue
        # §6.3, §6.4.5: поле R(c_d, r_d) по сетке карты — те же проверки, что в _indices_in_radius_np.
        y_low = max(0, int(math.floor(centroid_y - best_radius)) - 1)
        y_high = min(height - 1, int(math.ceil(centroid_y + best_radius)) + 1)
        x_low = max(0, int(math.floor(centroid_x - best_radius)) - 1)
        x_high = min(width - 1, int(math.ceil(centroid_x + best_radius)) + 1)
        in_field = 0
        for y in range(y_low, y_high + 1):
            for x in range(x_low, x_high + 1):
                point = grid[y, x]
                if point < 0:
                    continue
                if not (abs(pos_x[point] - centroid_x) <= best_radius):
                    continue
                if math.hypot(pos_y[point] - centroid_y, pos_x[point] - centroid_x) <= best_radius:
                    field[in_field] = point
                    in_field += 1
        if in_field == 0:
            continue
        field_sorted = np.sort(field[:in_field])
        # Код из центра детектора: ближайшая к c_d точка поля, при равенстве — с меньшим номером.
        code = -1
        nearest = np.inf
        for k in range(in_field):
            point = field_sorted[k]
            dy = pos_y[point] - centroid_y
            dx = pos_x[point] - centroid_x
            distance_sq = dy * dy + dx * dx
            if code < 0 or distance_sq < nearest:
                nearest = distance_sq
                code = point
        # A_μ — точки поля с e ≥ μ; n_d = |A_μ|, e_d = Σ a·e по A_μ (сходство с кодом центра детектора).
        ones_code = ones[code]
        points_count = 0
        lit = 0
        for k in range(in_field):
            point = field_sorted[k]
            energy = energies[point]
            if energy >= energy_threshold:
                if point == code:
                    similarity = 1.0
                else:
                    similarity = _table_similarity(table[code, point], ones_code, ones[point], cosine)
                activation = similarity if (similarity > 0.0 and similarity >= lambda_value) else 0.0
                if activation > 0.0:
                    lit += 1
                terms[points_count] = activation * energy
                points_count += 1
        detector_energy = _array_sum(terms, points_count)
        ints[line, 1] = code
        ints[line, 2] = in_field
        ints[line, 3] = points_count
        ints[line, 4] = lit
        floats[line, 3] = detector_energy
        floats[line, 4] = best_radius
        floats[line, 5] = best_fill
        stage[line] = 5 if (detector_energy <= 0.0 or points_count < min_points) else 6


if numba is not None:
    _block_sum = numba.njit(cache=True, nogil=True)(_block_sum_py)
    _pairwise_sum = numba.njit(cache=True, nogil=True)(_pairwise_sum_py)
    _array_sum = numba.njit(cache=True, nogil=True)(_array_sum_py)
    _table_similarity = numba.njit(cache=True, nogil=True)(_table_similarity_py)
    _block_candidates_compiled = numba.njit(cache=True, nogil=True)(_block_candidates_py)
else:  # pragma: no cover - без numba кандидатов считает numpy
    _block_sum = _block_sum_py
    _pairwise_sum = _pairwise_sum_py
    _array_sum = _array_sum_py
    _table_similarity = _table_similarity_py
    _block_candidates_compiled = None


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
        """Маска точек кластера, содержащего seed (все False, если seed — шум); порядок — как у входа.

        Тот же кластер, что у эталонного обхода _cluster_of_python, но на сетке: точки ставятся в ячейки
        прямоугольника, ядра находятся подсчётом соседей сдвигами, кластер — связная область ядер вместе с
        соседями этих ядер.
        """
        if ys.size == 0:
            return np.zeros(0, dtype=bool)
        ys = np.asarray(ys, dtype=np.int64)
        xs = np.asarray(xs, dtype=np.int64)
        offsets = self._offsets()
        reach = int(math.ceil(self._eps))
        # Поле вокруг точек шириной reach: сдвиги соседей не выходят за сетку.
        top, left = int(ys.min()) - reach, int(xs.min()) - reach
        height, width = int(ys.max()) - top + reach + 1, int(xs.max()) - left + reach + 1
        seed_y, seed_x = int(seed[0]) - top, int(seed[1]) - left
        grid = np.zeros((height, width), dtype=bool)
        grid[ys - top, xs - left] = True
        if not (0 <= seed_y < height and 0 <= seed_x < width) or not grid[seed_y, seed_x]:
            return np.zeros(ys.shape, dtype=bool)
        inner = (slice(reach, height - reach), slice(reach, width - reach))

        def shifted(source: "np.ndarray", dy: int, dx: int) -> "np.ndarray":
            """Значение source в ячейке (y + dy, x + dx) для ячеек внутренней части сетки."""
            return source[reach + dy : height - reach + dy, reach + dx : width - reach + dx]

        # Ядро — точка, у которой не меньше min_samples соседей (считая её саму).
        neighbours = np.zeros((height - 2 * reach, width - 2 * reach), dtype=np.int32)
        for dy, dx in offsets:
            neighbours += shifted(grid, dy, dx)
        core = np.zeros_like(grid)
        core[inner] = grid[inner] & (neighbours >= self._min_samples)
        if core[seed_y, seed_x]:
            start = (seed_y, seed_x)
        else:
            # Центр не ядро: кластер начинается с первого соседнего ядра (в порядке сдвигов), иначе центр — шум.
            start = next(
                (
                    (seed_y + dy, seed_x + dx)
                    for dy, dx in offsets
                    if (dy, dx) != (0, 0) and core[seed_y + dy, seed_x + dx]
                ),
                None,
            )
            if start is None:
                LOGGER.event(
                    "activation.detector.cluster.noise",
                    section=DETECTOR_CLUSTERING,
                    data={"seed": seed, "eps": self._eps, "min_samples": self._min_samples},
                )
                return np.zeros(ys.shape, dtype=bool)
        component = self._core_component(core, start, offsets, reach)
        # Кластер — ядра связной области и все точки, соседние с ними; центр входит в свой кластер.
        member = np.zeros_like(grid)
        for dy, dx in offsets:
            member[inner] |= shifted(component, dy, dx)
        member &= grid
        member[seed_y, seed_x] = True
        return member[ys - top, xs - left]

    @staticmethod
    def _core_component(
        core: "np.ndarray", start: tuple[int, int], offsets: list[tuple[int, int]], reach: int
    ) -> "np.ndarray":
        """Связная область ядер, содержащая start: ядра связаны, если стоят друг от друга не дальше eps."""
        # Соседи — восемь ячеек вокруг точки (eps от √2 до 2): связные области ядер находит готовая разметка SciPy.
        if len(offsets) == 9 and all(abs(dy) <= 1 and abs(dx) <= 1 for dy, dx in offsets):
            try:
                from scipy import ndimage
            except ImportError:  # SciPy необязателен: ниже — тот же ответ наращиванием области
                ndimage = None
            if ndimage is not None:
                labels, _ = ndimage.label(core, structure=np.ones((3, 3), dtype=bool))
                return labels == labels[start]
        height, width = core.shape
        component = np.zeros_like(core)
        component[start] = True
        while True:
            grown = component.copy()
            for dy, dx in offsets:
                grown[reach : height - reach, reach : width - reach] |= component[
                    reach + dy : height - reach + dy, reach + dx : width - reach + dx
                ]
            grown &= core
            if np.array_equal(grown, component):
                return component
            component = grown

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
        fill_cell: float = 0.0,
        radius_mode: str = "fill",
    ) -> None:
        if radius_min <= 0:
            raise ValueError("radius_min must be positive")
        if fill_cell < 0:
            raise ValueError("fill_cell must be >= 0")
        if radius_mode not in ("fill", "gyration"):
            raise ValueError("radius_mode must be 'fill' (§6.4.3) or 'gyration' (§6.4.4)")
        self._radius_mode = str(radius_mode)
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
        # Поправка заполнения на клетку сетки (вне статьи): f = n / (π (r + fill_cell)²); 0 — формула §6.4.3 как есть.
        self._fill_cell = float(fill_cell)
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
        # Таблица общих бит всех пар кодов карты: считается один раз при первом обращении (_common_table).
        self._common: "np.ndarray | None" = None
        self._common_ready = False
        # Кандидаты в детекторы всех центров каждого слоя, построенные по кластерам, посчитанным разом
        # (_prepare_clusters): по слою — список длиной в число точек, у центра — кандидат либо None.
        self._layer_candidates: dict[int, list["DetectorCandidate | None"]] = {}
        self._clusters_ready = False
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
            candidates = self._build_layer(layer, rng, index=layer_index, built=detectors)
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
        *,
        index: int = 0,
        built: Sequence[Detector] = (),
    ) -> list[DetectorCandidate]:
        """Слой детекторов (§6.4): случайные центры, кластер похожих точек, замена при перекрытии (§6.4.5).

        index и built — номер слоя и детекторы прошлых слоёв: только для показа слоя по ходу счёта (вне статьи).
        """
        selected: list[DetectorCandidate] = []
        watched = LOGGER.watched("activation.detectors.layer")
        # Центры, радиусы и плотности выбранных детекторов массивами — проверка перекрытий (§6.4.5) одним
        # вычислением расстояний вместо перебора; пересобираются при изменении состава слоя.
        selected_arrays: tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"] | None = None
        candidate_cache: dict[int, DetectorCandidate | None] = {}
        unchanged = 0
        attempts = 0
        replacements = 0
        point_count = len(self._energies)
        self._prepare_clusters()
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
                if watched:
                    LOGGER.progress("activation.detectors.layer", layer=index, selected=selected, built=built, done=False)
                continue
            selected.append(candidate)
            selected_arrays = None
            unchanged = 0
            if watched:
                LOGGER.progress("activation.detectors.layer", layer=index, selected=selected, built=built, done=False)
        if watched:
            LOGGER.progress("activation.detectors.layer", layer=index, selected=selected, built=built, done=True)
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
        prepared = self._layer_candidates.get(id(layer))
        if prepared is not None:
            # Кандидат построен по кластеру, найденному вместе с кластерами всех центров слоя (§6.4.1): тот же,
            # что дал бы поштучный счёт ниже.
            return prepared[center_index]
        lambda_value = layer.lambda_threshold
        if self._energies[center_index] < self._energy_threshold:
            return None
        center_y = self._layout._pos_y[center_index]
        center_x = self._layout._pos_x[center_index]
        seed = (center_y, center_x)
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
        if not np.any((active_y == center_y) & (active_x == center_x)):
            return None
        cluster_mask = self._clusterer.cluster_mask(active_y, active_x, seed)
        cluster_count = int(cluster_mask.sum())
        if cluster_count < layer.min_points:
            return None
        activated_count = int(active_indices.size)
        cluster_w = weights[cluster_mask]
        cluster_y = active_y[cluster_mask].astype(np.float64)
        cluster_x = active_x[cluster_mask].astype(np.float64)
        return self._finish_candidate(layer, seed, cluster_count, activated_count, cluster_w, cluster_y, cluster_x)

    def _cluster_candidate(
        self,
        center_index: int,
        layer: DetectorLayerConfig,
        cluster_indices: "np.ndarray",
        activated_count: int,
        similarity: "np.ndarray | None" = None,
    ) -> DetectorCandidate | None:
        """Кандидат по кластеру центра, найденному пакетным счётом (_prepare_clusters) — те же действия, что у
        поштучного счёта в _make_candidate: точки кластера — по возрастанию номера, поэтому веса и суммы те же.
        similarity — уже посчитанное сходство центра с точками кластера (в том же порядке): те же числа, что даёт
        _similarity_row, но без повторного счёта."""
        numbers = self._cluster_numbers(center_index, layer, cluster_indices, similarity)
        if numbers[0] == 0:
            return None
        seed = (self._layout._pos_y[center_index], self._layout._pos_x[center_index])
        return self._emit_candidate(layer, seed, int(activated_count), numbers)

    def _cluster_numbers(
        self,
        center_index: int,
        layer: DetectorLayerConfig,
        cluster_indices: "np.ndarray",
        similarity: "np.ndarray | None" = None,
    ) -> tuple:
        """Числа кандидата по кластеру центра (без журнала; стадии — у _CANDIDATE_INTS): счёт _cluster_candidate."""
        if self._energies[center_index] < self._energy_threshold:
            return (0,) + _NO_NUMBERS
        if cluster_indices.size == 0:
            return (1,) + _NO_NUMBERS
        cluster_count = int(cluster_indices.size)
        if cluster_count < layer.min_points:
            return (2, cluster_count) + _NO_NUMBERS[1:]
        if similarity is None:
            similarity = self._similarity_row(center_index, cluster_indices)
        cluster_activation = self._threshold_similarity_np(similarity, layer.lambda_threshold)
        # §6.4.2: вес точки — произведение активации и энергии.
        cluster_w = cluster_activation * self._energies_np[cluster_indices]
        cluster_y = self._pos_y_np[cluster_indices].astype(np.int64).astype(np.float64)
        cluster_x = self._pos_x_np[cluster_indices].astype(np.int64).astype(np.float64)
        return self._finish_numbers(layer, cluster_count, cluster_w, cluster_y, cluster_x)

    def _finish_candidate(
        self,
        layer: DetectorLayerConfig,
        seed: tuple,
        cluster_count: int,
        activated_count: int,
        cluster_w: "np.ndarray",
        cluster_y: "np.ndarray",
        cluster_x: "np.ndarray",
    ) -> DetectorCandidate | None:
        """Кандидат по кластеру (§6.4.2–§6.4.5): взвешенный центр, радиус по наибольшему заполнению, поле и энергия."""
        return self._emit_candidate(
            layer, seed, activated_count, self._finish_numbers(layer, cluster_count, cluster_w, cluster_y, cluster_x)
        )

    def _emit_candidate(
        self, layer: DetectorLayerConfig, seed: tuple, activated_count: int, numbers: tuple
    ) -> DetectorCandidate | None:
        """События кандидата в журнал (§6.4.1–§6.4.5) по его числам и сам кандидат — те же записи в том же порядке,
        что при счёте вперемешку с записью: между записями одного кандидата других событий нет."""
        stage = numbers[0]
        if stage == 0 or stage == 2:
            return None
        if stage == 1:
            LOGGER.event(
                "activation.detector.cluster.noise",
                section=DETECTOR_CLUSTERING,
                data={"seed": seed, "eps": self._clusterer.eps, "min_samples": self._clusterer.min_samples},
            )
            return None
        (_, cluster_count, center_code, field_size, points_count, activated_points, weight_sum, centroid_y, centroid_x,
         detector_energy, best_radius, best_fill) = numbers
        lambda_value = layer.lambda_threshold
        LOGGER.event(
            "activation.detector.cluster",
            section=DETECTOR_CLUSTERING,
            data={
                "points": cluster_count,
                "activated": activated_count,
                "lambda": lambda_value,
                "seed": seed,
            },
        )
        if stage == 3:
            return None
        LOGGER.event(
            "activation.detector.center",
            section=DETECTOR_CENTER,
            data={"center_y": centroid_y, "center_x": centroid_x, "weight_sum": weight_sum},
        )
        if stage == 4:
            return None
        LOGGER.event(
            "activation.detector.params",
            section=DETECTOR_PARAMS,
            data={
                "center_code": int(center_code),
                "field_points": int(field_size),
                "n_d": points_count,
                "activated_points": activated_points,
                "e_d": detector_energy,
                "lambda": lambda_value,
                "mu": self._energy_threshold,
            },
        )
        if stage == 5:
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

    def _finish_numbers(
        self,
        layer: DetectorLayerConfig,
        cluster_count: int,
        cluster_w: "np.ndarray",
        cluster_y: "np.ndarray",
        cluster_x: "np.ndarray",
    ) -> tuple:
        """Числа кандидата по кластеру (§6.4.2–§6.4.5), без журнала: взвешенный центр, радиус по наибольшему
        заполнению, поле и энергия. Кортеж и стадии — у _CANDIDATE_INTS."""
        lambda_value = layer.lambda_threshold
        weight_sum = float(cluster_w.sum())
        if weight_sum <= 0.0:
            return (3, cluster_count, 0, 0, 0, 0, weight_sum, 0.0, 0.0, 0.0, 0.0, 0.0)
        centroid_y = float((cluster_y * cluster_w).sum() / weight_sum)
        centroid_x = float((cluster_x * cluster_w).sum() / weight_sum)
        centred = (4, cluster_count, 0, 0, 0, 0, weight_sum, centroid_y, centroid_x, 0.0, 0.0, 0.0)
        # §6.4.3: радиус по максимуму f(p) = |{q: r(q) ≤ r(p)}| / (π r(p)²).
        distances = np.sort(np.hypot(cluster_y - centroid_y, cluster_x - centroid_x))
        counts = np.arange(1, distances.size + 1, dtype=np.float64)
        candidate_radii = np.maximum(float(self._radius_min), distances)
        allowed = (candidate_radii <= layer.activation_radius) & (counts >= layer.min_points)
        if not allowed.any():
            return centred
        # На сетке точка занимает клетку: n(r) / (π r²) при малых r всегда выше плотности (r = 1 — 5 / π ≈ 1.6),
        # и максимум уходит к самому малому радиусу, где набран минимум точек слоя. С поправкой на клетку
        # (fill_cell = 0.5) у сплошного круга f ≈ плотность при любом r и падает за краем кластера (§6.4.3).
        cell_radii = candidate_radii + self._fill_cell
        fill = np.where(allowed, counts / (math.pi * cell_radii * cell_radii), -np.inf)
        best_index = int(np.argmax(fill))
        best_fill = float(fill[best_index])
        best_radius = float(candidate_radii[best_index])
        if self._radius_mode == "gyration":
            # §6.4.4 (статистический метод вместо круга §6.4.3): «in complex cases, the principal component analysis
            # and the Mahalanobis statistical distance can be used to calculate the parameters of a circumscribed
            # ellipse». Здесь — круг того же второго момента: r_g² = Σ w·d² / Σ w (след ковариации, веса — активация ×
            # энергия, §6.4.2), r_d = √2 · r_g — радиус сплошного круга с тем же моментом; выбросы с малым весом
            # радиус почти не раздувают. Предел — радиус активации слоя r_a.
            squared = (cluster_y - centroid_y) ** 2 + (cluster_x - centroid_x) ** 2
            gyration = math.sqrt(float((cluster_w * squared).sum() / weight_sum))
            best_radius = min(float(layer.activation_radius), max(float(self._radius_min), math.sqrt(2.0) * gyration))
            inside = int(np.count_nonzero(distances <= best_radius))
            if inside < layer.min_points:
                return centred
            best_fill = inside / (math.pi * best_radius * best_radius)
        if best_radius <= 0.0:
            return centred
        # §6.3, §6.4.5: после c_d и r_d поле A_λd(c_d, r_d) активируется кодом из центра детектора
        # (§6.1: все точки R(c_d, r_d), включая нулевую активацию); A_μ — точки поля с e ≥ μ,
        # n_d = |A_μ|, e_d = Σ a_ji · e_ji по A_μ.
        field_points = self._indices_in_radius_np(centroid_y, centroid_x, best_radius)
        if field_points.size == 0:
            return centred
        center_code = self._center_code_index_np(field_points, centroid_y, centroid_x)
        field_energy = self._energies_np[field_points]
        in_a_mu = field_energy >= self._energy_threshold
        a_mu = field_points[in_a_mu]
        field_activation = self._threshold_similarity_np(self._similarity_row(center_code, a_mu), lambda_value)
        points_count = int(a_mu.size)
        activated_points = int(np.count_nonzero(field_activation > 0.0))
        detector_energy = float((field_activation * field_energy[in_a_mu]).sum())
        stage = 5 if (detector_energy <= 0.0 or points_count < layer.min_points) else 6
        return (stage, cluster_count, int(center_code), int(field_points.size), points_count, activated_points, weight_sum,
                centroid_y, centroid_x, detector_energy, best_radius, best_fill)

    # Центров за один проход пакетной кластеризации слоя — не больше CLUSTER_ROWS и не больше CLUSTER_CELLS пар
    # «центр × точка» (память блока линейна по числу точек карты); на результат размер блока не влияет.
    CLUSTER_ROWS = 1024
    CLUSTER_CELLS = 40_000_000
    # Пакетный счёт кластеров (видеокарта); False — только поштучный счёт (эталон в тестах).
    BATCHED = True
    # Пакетный счёт в окне вокруг диска центра (с таблицей общих бит): активируются только точки диска r_a (§6.4.1),
    # поэтому сходство и сетка DBSCAN нужны лишь в квадрате со стороной 2·(r_a + 1) + 1, а не на всей карте (на 250
    # цифрах в диске 11–177 точек из 13 тыс.). Результат тот же; False — прежний счёт по всей карте (эталон в тестах).
    WINDOWED = True
    # Центров за один проход счёта в окне — не больше CLUSTER_ROWS и не больше CLUSTER_CELLS пар «центр × клетка окна»:
    # память блока не больше, чем у счёта по всей карте. 2026-11-21: блок 4096 центров при окне 83 × 83 (r_a = 40 на
    # уровнях 1–3) переполнил видеопамять двух прогонов, и счёт встал.
    WINDOW_ROWS = 1024
    # Окно берётся, если в нём меньше этой доли клеток карты; иначе счёт по всей карте. 1.0 (анализ 2026-11-21): окно
    # (2·r_a + 3)² — 0.49–0.52 карты на всех уровнях, при 0.5 оно включалось то на одном уровне, то ни на одном, а с ним
    # детекторы быстрее и при такой доле (82 против 103 с, прогоны 596/597 против 588/589, числа те же).
    WINDOW_MAX_SHARE = 1.0
    # Точек кодов за одно матричное произведение общих бит блока на видеокарте (без таблицы общих бит).
    COMMON_CHUNK = 16384
    # Числа кандидатов блока — скомпилированным циклом numba (_block_candidates_py): те же действия float64 в том же
    # порядке, суммы — попарные, как у numpy. Нужны numba, порог λ без η, таблица общих бит и сетка карты; иначе — numpy
    # по одному центру. False — только numpy (эталон в тестах). Первые COMPILED_CHECK центров первого блока каждого слоя
    # сверяются с numpy; при расхождении — предупреждение в журнал и numpy до конца процесса.
    COMPILED = True
    COMPILED_CHECK = 64
    _compiled_broken = False

    GROW_STEPS = 8

    @classmethod
    def _grow_region(cls, component, core, pool, torch):  # noqa: ANN001, ANN202
        """Связная область ядер: рост от центра шагами «соседи ядер» до неподвижной точки. Сходимость проверяется раз
        в GROW_STEPS шагов (одна синхронизация с видеокартой вместо одной на шаг); лишние шаги после сходимости
        область не меняют — тот же результат, что при остановке на первом шаге без изменений."""
        while True:
            for _ in range(cls.GROW_STEPS - 1):
                component = pool(component, 3, stride=1, padding=1) * core
            grown = pool(component, 3, stride=1, padding=1) * core
            if torch.equal(grown, component):
                return grown
            component = grown

    def _prepare_clusters(self) -> None:
        """Кластеры §6.4.1 всех центров всех слоёв разом; результат тот же, что у поштучного счёта в _make_candidate.

        Для центра c активируются точки диска r_a с энергией не ниже μ и сходством с кодом центра по порогу слоя;
        кластер — связная область ядер DBSCAN, содержащая центр, с соседями этих ядер. Сходства блока центров со
        всеми точками считаются один раз на все слои теми же действиями, что в _similarity_row, принадлежность
        диску — той же функцией расстояния. Связные области ищутся на видеокарте: активированные точки каждого
        центра кладутся в свою сетку карты, ядра находятся свёрткой 3×3, область растёт от центра шагами «соседи
        ядер». Нужны: видеокарта, соседство DBSCAN из восьми ячеек (eps от √2 до 2) и целые координаты точек; иначе
        кластеры считаются поштучно, как раньше.

        Общие биты блока — строки таблицы общих бит, а без неё (карта больше COMMON_TABLE_POINTS) — матричное
        произведение кодов на видеокарте: те же целые числа. Из кластера центра сразу строится кандидат в детекторы
        (_cluster_candidate) — хранятся только кандидаты, поэтому память не растёт как квадрат числа точек.
        """
        if self._clusters_ready:
            return
        self._clusters_ready = True
        if not self.BATCHED:
            return
        layout = self._layout
        table = self._common_table()
        code_matrix = getattr(layout, "_code_matrix", None)
        ones_array = getattr(layout, "_ones_array", None)
        offsets = self._clusterer._offsets()
        eight = len(offsets) == 9 and all(abs(dy) <= 1 and abs(dx) <= 1 for dy, dx in offsets)
        try:
            import torch
        except ImportError:  # pragma: no cover
            return
        whole = bool(np.all(self._pos_y_np == np.floor(self._pos_y_np)) and np.all(self._pos_x_np == np.floor(self._pos_x_np)))
        if (table is None and code_matrix is None) or ones_array is None or not eight or not whole or not torch.cuda.is_available():
            return
        count = len(self._energies)
        pos_y = self._pos_y_np.astype(np.int64)
        pos_x = self._pos_x_np.astype(np.int64)
        if count == 0 or int(pos_y.min()) < 0 or int(pos_x.min()) < 0:
            return
        height, width = int(pos_y.max()) + 1, int(pos_x.max()) + 1
        # Сетка карты с полем в одну ячейку: у каждой точки есть все восемь соседних ячеек.
        cell = (pos_y + 1) * (width + 2) + (pos_x + 1)
        cells = (height + 2) * (width + 2)
        alive = self._energies_np >= self._energy_threshold
        ones = np.asarray(ones_array).astype(np.float64)
        # Принадлежность диску: то же сравнение hypot(dy, dx) ≤ r_a, что в _indices_in_radius_np, таблицей по |dy|, |dx|.
        reach = np.hypot(np.arange(height, dtype=np.float64)[:, None], np.arange(width, dtype=np.float64)[None, :])
        layers = list(self._layer_configs)
        within = [reach <= float(layer.activation_radius) for layer in layers]
        candidates: list[list["DetectorCandidate | None"]] = [[None] * count for _ in layers]
        activated = [np.zeros(count, dtype=np.int64) for _ in layers]
        device = torch.device("cuda")
        # Окно вокруг диска выгодно, только если оно меньше карты: при r_a ≈ 40 (уровни 1–3) окно 83 × 83 — почти вся карта.
        side = 2 * (int(math.ceil(max(float(layer.activation_radius) for layer in layers))) + 1) + 1
        windowed = table is not None and self.WINDOWED and side * side < cells * self.WINDOW_MAX_SHARE
        if windowed:
            block_rows = self._windowed_clusters(
                table, ones, pos_y, pos_x, height, width, alive, layers, offsets, candidates, activated, torch, device
            )
        # Без таблицы общие биты считаются по кодам на видеокарте: коды — байтами, в float32 — кусками по COMMON_CHUNK
        # точек (значения целые и меньше 2^24 — произведение точное).
        codes_gpu = None if table is not None else torch.as_tensor(np.asarray(code_matrix, dtype=np.uint8), device=device)
        if not windowed:
            block_rows = max(1, min(self.CLUSTER_ROWS, int(self.CLUSTER_CELLS // max(1, count))))
        cell_t = torch.as_tensor(cell, device=device)
        kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=device)
        steps = [dy * (width + 2) + dx for dy, dx in offsets if (dy, dx) != (0, 0)]
        pool = torch.nn.functional.max_pool2d
        pending: list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as workers, torch.no_grad():
            # Счёт в окне уже сделан (_windowed_clusters) — по всей карте не считается.
            for first in range(0, 0 if windowed else count, block_rows):
                rows = np.arange(first, min(count, first + block_rows))
                block = rows.size
                # Сходство центров блока со всеми точками — те же действия, что в _similarity_row.
                if table is not None:
                    common = table[rows]
                else:
                    centres = codes_gpu[torch.as_tensor(rows, device=device)].to(torch.float32)
                    common = np.empty((block, count), dtype=np.int64)
                    for start in range(0, count, self.COMMON_CHUNK):
                        part = codes_gpu[start : start + self.COMMON_CHUNK].to(torch.float32)
                        common[:, start : start + part.shape[0]] = (centres @ part.T).to(torch.int64).cpu().numpy()
                ones_center = ones[rows][:, None]
                ones_points = ones[None, :]
                if layout._similarity == "cosine":
                    denominator = np.sqrt(ones_center * ones_points)
                else:
                    denominator = ones_center + ones_points - common
                similarity = np.divide(
                    common.astype(np.float64),
                    denominator,
                    out=np.zeros(common.shape, dtype=np.float64),
                    where=(denominator > 0) & (ones_points > 0) & (ones_center > 0),
                )
                similarity[np.arange(block), rows] = 1.0
                gap_y = np.abs(pos_y[rows][:, None] - pos_y[None, :])
                gap_x = np.abs(pos_x[rows][:, None] - pos_x[None, :])
                living = alive[None, :] & alive[rows][:, None]
                lines = torch.arange(block, device=device)
                centre = cell_t[torch.as_tensor(rows, device=device)]
                for number, layer in enumerate(layers):
                    lit = self._lit(similarity, layer.lambda_threshold)
                    active = lit & within[number][gap_y, gap_x] & living
                    activated[number][rows] = active.sum(axis=1)
                    grid = torch.zeros((block, cells), dtype=torch.float32, device=device)
                    grid[:, cell_t] = torch.as_tensor(active, device=device).to(torch.float32)
                    grid = grid.view(block, 1, height + 2, width + 2)
                    # Ядро — активированная точка, у которой не меньше min_samples соседей (считая её саму).
                    neighbours = torch.nn.functional.conv2d(grid, kernel, padding=1)
                    core = grid * (neighbours >= float(self._clusterer.min_samples)).to(torch.float32)
                    flat_core = core.view(block, cells)
                    start = torch.where(flat_core[lines, centre] > 0, centre, torch.full_like(centre, -1))
                    # Центр не ядро: кластер начинается с первого соседнего ядра (в порядке сдвигов), иначе шум.
                    for step in steps:
                        neighbour = centre + step
                        start = torch.where((start < 0) & (flat_core[lines, neighbour] > 0), neighbour, start)
                    valid = (start >= 0) & (grid.view(block, cells)[lines, centre] > 0)
                    component = torch.zeros((block, cells), dtype=torch.float32, device=device)
                    component[lines[valid], start[valid]] = 1.0
                    component = component.view(block, 1, height + 2, width + 2)
                    component = self._grow_region(component, core, pool, torch)
                    # Кластер — ядра связной области и активированные точки, соседние с ними; центр входит в свой кластер.
                    member = (pool(component, 3, stride=1, padding=1) * grid).view(block, cells)
                    member[lines[valid], centre[valid]] = 1.0
                    inside = (member[:, cell_t] > 0).cpu().numpy()
                    pending.append(
                        workers.submit(
                            self._block_candidates, layer, rows, None, inside, similarity, activated[number], candidates[number]
                        )
                    )
                while len(pending) > 2 * len(layers):
                    pending.pop(0).result()
            for future in pending:
                future.result()
        for number, layer in enumerate(layers):
            self._layer_candidates[id(layer)] = candidates[number]
        LOGGER.event(
            "activation.detector.clusters",
            section=DETECTOR_CLUSTERING,
            data={
                "layers": len(layers),
                "centres": count,
                "block_rows": int(block_rows),
                "common_table": table is not None,
                "windowed": bool(windowed),
                "compiled_blocks": int(self.__dict__.get("_compiled_blocks", 0)),
                "compiled_checked": int(sum(self.__dict__.get("_compiled_checked", {}).values())),
                "activated_mean": [float(item.mean()) for item in activated],
                "note": "кластеры DBSCAN всех центров всех слоёв посчитаны разом (§6.4.1): ядра — свёрткой по сетке "
                "карты, связная область — ростом от центра; по кластеру сразу строится кандидат в детекторы; "
                "результат тот же, что при счёте по одному центру",
            },
        )

    def _block_candidates(
        self, layer: DetectorLayerConfig, rows: "np.ndarray", window: "np.ndarray | None", inside: "np.ndarray",
        similarity: "np.ndarray", activated: "np.ndarray", out: list,
    ) -> None:
        """Кандидаты слоя по кластерам блока центров (§6.4.1): точки кластера — по возрастанию номера, сходство берётся
        из уже посчитанного сходства блока (те же числа, что у _similarity_row). Числа кандидатов блока — одним
        скомпилированным циклом (COMPILED), иначе numpy по центру; события журнала и кандидаты — те же."""
        numbers = self._compiled_numbers(layer, rows, window, inside, similarity)
        if numbers is None:
            for line, centre_index in enumerate(rows.tolist()):
                points, weights = self._block_cluster(line, window, inside, similarity)
                out[centre_index] = self._cluster_candidate(centre_index, layer, points, int(activated[centre_index]), weights)
            return
        pos_y, pos_x = self._layout._pos_y, self._layout._pos_x
        for line, centre_index in enumerate(rows.tolist()):
            found = numbers[line]
            if found[0] == 0:
                out[centre_index] = None
                continue
            seed = (pos_y[centre_index], pos_x[centre_index])
            out[centre_index] = self._emit_candidate(layer, seed, int(activated[centre_index]), found)

    @staticmethod
    def _block_cluster(
        line: int, window: "np.ndarray | None", inside: "np.ndarray", similarity: "np.ndarray"
    ) -> tuple["np.ndarray", "np.ndarray"]:
        """Точки кластера центра строки line блока по возрастанию номера и их сходство с центром из сходства блока."""
        chosen = inside[line]
        points = np.flatnonzero(chosen) if window is None else window[line][chosen]
        order = np.argsort(points, kind="stable")
        return points[order], similarity[line][chosen][order]

    def _compiled_numbers(
        self, layer: DetectorLayerConfig, rows: "np.ndarray", window: "np.ndarray | None", inside: "np.ndarray",
        similarity: "np.ndarray",
    ) -> "list[tuple] | None":
        """Числа кандидатов блока скомпилированным циклом (_block_candidates_py); None — считать numpy по центру.

        Первые COMPILED_CHECK центров первого блока слоя сверяются с numpy (_cluster_numbers) до бита: при расхождении
        — событие activation.detector.compiled.warning и numpy до конца процесса (числа этого блока — тоже numpy)."""
        table = self._common
        ones_array = getattr(self._layout, "_ones_array", None)
        if (
            _block_candidates_compiled is None
            or not self.COMPILED
            or DetectorHierarchyBuilder._compiled_broken
            or self._eta is not None
            or table is None
            or ones_array is None
        ):
            return None
        grid = self._point_grid()
        if grid is None:
            return None
        ones = self.__dict__.get("_ones_float64")
        if ones is None:
            ones = self._ones_float64 = np.asarray(ones_array).astype(np.float64)
        block = int(rows.shape[0])
        stage = np.zeros(block, dtype=np.int64)
        ints = np.zeros((block, _CANDIDATE_INTS), dtype=np.int64)
        floats = np.zeros((block, _CANDIDATE_FLOATS), dtype=np.float64)
        windowed = window is not None
        _block_candidates_compiled(
            np.ascontiguousarray(rows, dtype=np.int64),
            window if windowed else np.zeros((1, 1), dtype=np.int64),
            windowed,
            inside,
            self._energies_np,
            self._pos_y_np,
            self._pos_x_np,
            grid,
            table,
            ones,
            self._layout._similarity == "cosine",
            float(layer.lambda_threshold),
            self._energy_threshold,
            int(layer.min_points),
            float(layer.activation_radius),
            float(self._radius_min),
            self._fill_cell,
            self._radius_mode == "gyration",
            stage,
            ints,
            floats,
        )
        numbers = [
            (found,) + tuple(whole) + tuple(real)
            for found, whole, real in zip(stage.tolist(), ints.tolist(), floats.tolist())
        ]
        checked = self.__dict__.setdefault("_compiled_checked", {})
        if id(layer) not in checked:
            lines = min(block, int(self.COMPILED_CHECK))
            expected = []
            for line in range(lines):
                points, weights = self._block_cluster(line, window, inside, similarity)
                expected.append(self._cluster_numbers(int(rows[line]), layer, points, weights))
            identical = all(_same_numbers(left, right) for left, right in zip(expected, numbers))
            checked[id(layer)] = lines
            if not identical:
                DetectorHierarchyBuilder._compiled_broken = True
                LOGGER.event(
                    "activation.detector.compiled.warning",
                    section=DETECTOR_CONSTRUCTION,
                    data={
                        "lambda": layer.lambda_threshold,
                        "checked": lines,
                        "note": "числа кандидатов (§6.4.1–§6.4.5) скомпилированным циклом расходятся с numpy: дальше — "
                        "numpy по одному центру",
                    },
                )
                return None
        self._compiled_blocks = self.__dict__.get("_compiled_blocks", 0) + 1
        return numbers

    def _windowed_clusters(
        self, table, ones, pos_y, pos_x, height, width, alive, layers, offsets, candidates, activated, torch, device
    ) -> int:  # noqa: ANN001
        """Кластеры §6.4.1 всех центров в окне вокруг диска центра — тот же счёт, что в _prepare_clusters по всей карте.

        Активироваться могут только точки диска r_a (§6.4.1), поэтому сходство, ядра DBSCAN и рост связной области
        считаются в квадрате со стороной 2·half + 1, half = ⌈наибольший r_a⌉ + 1: клетка поля за диском пуста на
        любом слое, и свёртка 3×3 и рост у края окна дают то же, что на всей карте. Сходство — те же действия, что в
        _similarity_row; принадлежность диску — та же таблица hypot(|dy|, |dx|); точки кластера — по возрастанию номера,
        как flatnonzero по всей карте. Возвращает число центров в блоке."""
        count = int(pos_y.shape[0])
        half = int(math.ceil(max(float(layer.activation_radius) for layer in layers))) + 1
        side = 2 * half + 1
        # Номер точки в каждой клетке карты с полем half со всех сторон: −1 — пустая клетка.
        index = np.full((height + 2 * half, width + 2 * half), -1, dtype=np.int64)
        index[pos_y + half, pos_x + half] = np.arange(count)
        shift_y, shift_x = (grid.reshape(-1) for grid in np.mgrid[-half : half + 1, -half : half + 1])
        reach = np.hypot(np.abs(shift_y).astype(np.float64), np.abs(shift_x).astype(np.float64))
        within = [reach <= float(layer.activation_radius) for layer in layers]
        middle = half * side + half  # центр окна — сам центр детектора
        steps = [dy * side + dx for dy, dx in offsets if (dy, dx) != (0, 0)]
        kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=device)
        pool = torch.nn.functional.max_pool2d
        block_rows = max(1, min(int(self.WINDOW_ROWS), int(self.CLUSTER_CELLS // (side * side))))
        pending: list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as workers, torch.no_grad():
            for first in range(0, count, block_rows):
                rows = np.arange(first, min(count, first + block_rows))
                block = rows.size
                window = index[(pos_y[rows] + half)[:, None] + shift_y[None, :], (pos_x[rows] + half)[:, None] + shift_x[None, :]]
                present = window >= 0
                safe = np.where(present, window, 0)
                # Сходство центров блока с точками окна — те же действия, что в _similarity_row.
                common = table[rows[:, None], safe]
                ones_center = ones[rows][:, None]
                ones_points = ones[safe]
                if self._layout._similarity == "cosine":
                    denominator = np.sqrt(ones_center * ones_points)
                else:
                    denominator = ones_center + ones_points - common
                similarity = np.divide(
                    common.astype(np.float64),
                    denominator,
                    out=np.zeros(common.shape, dtype=np.float64),
                    where=(denominator > 0) & (ones_points > 0) & (ones_center > 0),
                )
                similarity[:, middle] = 1.0
                living = present & alive[safe] & alive[rows][:, None]
                lines = torch.arange(block, device=device)
                for number, layer in enumerate(layers):
                    lit = self._lit(similarity, layer.lambda_threshold)
                    active = lit & within[number][None, :] & living
                    activated[number][rows] = active.sum(axis=1)
                    grid = torch.as_tensor(active, device=device).to(torch.float32).view(block, 1, side, side)
                    # Ядро — активированная точка, у которой не меньше min_samples соседей (считая её саму).
                    neighbours = torch.nn.functional.conv2d(grid, kernel, padding=1)
                    core = grid * (neighbours >= float(self._clusterer.min_samples)).to(torch.float32)
                    flat_core = core.view(block, side * side)
                    centre = torch.full((block,), middle, dtype=torch.int64, device=device)
                    start = torch.where(flat_core[:, middle] > 0, centre, torch.full_like(centre, -1))
                    # Центр не ядро: кластер начинается с первого соседнего ядра (в порядке сдвигов), иначе шум.
                    for step in steps:
                        start = torch.where((start < 0) & (flat_core[:, middle + step] > 0), centre + step, start)
                    valid = (start >= 0) & (grid.view(block, side * side)[:, middle] > 0)
                    component = torch.zeros((block, side * side), dtype=torch.float32, device=device)
                    component[lines[valid], start[valid]] = 1.0
                    component = component.view(block, 1, side, side)
                    component = self._grow_region(component, core, pool, torch)
                    member = (pool(component, 3, stride=1, padding=1) * grid).view(block, side * side)
                    member[lines[valid], middle] = 1.0
                    inside = (member > 0).cpu().numpy()
                    # Кандидатов строит рабочий поток, пока видеокарта считает следующий слой и блок; порядок и числа те
                    # же (каждый кандидат пишется в свой номер центра).
                    pending.append(
                        workers.submit(
                            self._block_candidates, layer, rows, window, inside, similarity, activated[number], candidates[number]
                        )
                    )
                while len(pending) > 2 * len(layers):
                    pending.pop(0).result()
            for future in pending:
                future.result()
        return block_rows

    # Карта не больше стольких точек получает таблицу общих бит всех пар кодов (int16: N² · 2 байта, 3.2 ГБ на
    # 40 000 точек); COMMON_TABLE_ROWS — строк таблицы за одно матричное произведение. Карта больше — общие биты блоками
    # на видеокарте (_prepare_clusters) и по месту (_similarity_row): тот же результат, но в 2–3 раза дольше. Было
    # 20 000; поднято 2026-11-18: карта уровня 1 на 500 цифрах с уровнем 0 — 22 734 точки.
    COMMON_TABLE_POINTS = 40000
    COMMON_TABLE_ROWS = 2048

    def _common_table(self) -> "np.ndarray | None":
        """Число общих бит у всех пар кодов карты (§2.2.4), посчитанное один раз.

        Построение слоя спрашивает сходство кода центра с точками диска десятки тысяч раз; каждый раз считать
        общие биты заново — половина времени построения. Таблица N × N считается одним матричным произведением
        (на видеокарте, если она есть) и даёт те же целые числа, что и побитовый счёт. None — карта слишком
        велика для таблицы либо у раскладки нет матрицы кодов: тогда общие биты считаются по месту, как раньше.
        """
        if self._common_ready:
            return self._common
        self._common_ready = True
        code_matrix = getattr(self._layout, "_code_matrix", None)
        count = len(self._energies)
        if code_matrix is None or count > self.COMMON_TABLE_POINTS or int(code_matrix.shape[1]) > 32767:
            return None
        table = np.empty((count, count), dtype=np.int16)
        device = "cpu"
        try:
            import torch
        except ImportError:  # pragma: no cover - torch нужен раскладке, без него таблица считается на numpy
            torch = None
        if torch is not None and torch.cuda.is_available():
            device = "cuda"
            with torch.no_grad():
                codes = torch.as_tensor(np.asarray(code_matrix, dtype=np.uint8), device=device).to(torch.float32)
                for start in range(0, count, self.COMMON_TABLE_ROWS):
                    # float32: значения целые и меньше 2^24, произведение точное.
                    block = codes[start : start + self.COMMON_TABLE_ROWS] @ codes.T
                    table[start : start + self.COMMON_TABLE_ROWS] = block.to(torch.int16).cpu().numpy()
        else:
            codes = np.asarray(code_matrix, dtype=np.float32)
            for start in range(0, count, self.COMMON_TABLE_ROWS):
                table[start : start + self.COMMON_TABLE_ROWS] = (codes[start : start + self.COMMON_TABLE_ROWS] @ codes.T).astype(np.int16)
        self._common = table
        LOGGER.event(
            "activation.detector.common_bits",
            section=DETECTOR_CONSTRUCTION,
            data={
                "points": count,
                "code_length": int(code_matrix.shape[1]),
                "device": device,
                "megabytes": float(table.nbytes) / 2**20,
                "note": "таблица общих бит всех пар кодов карты считается один раз; сходство центра с точками диска "
                "(§6.4.1, §2.2.4) берётся из неё — результат тот же, что при побитовом счёте",
            },
        )
        return table

    def _similarity_row(self, center_index: int, indices: "np.ndarray") -> "np.ndarray":
        """Сходство кода центра с кодами точек indices (§2.2.4) одним матричным произведением."""
        if indices.size == 0:
            return np.zeros(0, dtype=np.float64)
        layout = self._layout
        code_matrix = layout._code_matrix
        ones = layout._ones_array
        table = self._common_table() if ones is not None else None
        if table is not None:
            common = table[center_index, indices]
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
        elif self._packed_codes is not None and ones is not None:
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

    def _lit(self, similarity: "np.ndarray", lambda_value: float) -> "np.ndarray":
        """Зажжена ли точка: _threshold_similarity_np(similarity, λ) > 0 — без η одним сравнением (тот же результат:
        при пороге λ значение остаётся, только если сходство положительно и не ниже λ)."""
        if self._eta is None:
            return (similarity > 0.0) & (similarity >= lambda_value)
        return self._threshold_similarity_np(similarity, lambda_value) > 0.0

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

    _GRID_MAX_CELLS = 50_000_000

    def _point_grid(self) -> "np.ndarray | None":
        """Номер точки в каждой клетке карты (−1 — пусто); None — координаты не целые или не неотрицательные."""
        if self.__dict__.get("_grid_ready"):
            return self._grid
        self._grid_ready = True
        self._grid = None
        pos_y, pos_x = self._pos_y_np, self._pos_x_np
        whole = pos_y.size and bool(np.all(pos_y == np.floor(pos_y)) and np.all(pos_x == np.floor(pos_x)))
        if whole and pos_y.min() >= 0 and pos_x.min() >= 0:
            height, width = int(pos_y.max()) + 1, int(pos_x.max()) + 1
            if height * width <= self._GRID_MAX_CELLS:
                grid = np.full((height, width), -1, dtype=np.int64)
                grid[pos_y.astype(np.int64), pos_x.astype(np.int64)] = np.arange(pos_y.size)
                self._grid = grid
        return self._grid

    def _indices_in_radius_np(self, center_y: float, center_x: float, radius: float) -> "np.ndarray":
        """R(c, r) = {v : ‖v − c‖ ≤ r} по массивам координат (§5.7, §6.1); тот же критерий, что и построчно."""
        grid = self._point_grid()
        if grid is not None:
            # Точки прямоугольника радиуса (с запасом в клетку) по сетке карты; дальше те же проверки, что ниже.
            height, width = grid.shape
            y0, y1 = max(0, int(math.floor(center_y - radius)) - 1), min(height - 1, int(math.ceil(center_y + radius)) + 1)
            x0, x1 = max(0, int(math.floor(center_x - radius)) - 1), min(width - 1, int(math.ceil(center_x + radius)) + 1)
            if y1 < y0 or x1 < x0:
                return np.zeros(0, dtype=np.int64)
            patch = grid[y0 : y1 + 1, x0 : x1 + 1]
            band = patch[patch >= 0]
            band = band[np.abs(self._pos_x_np[band] - center_x) <= radius]
            inside = np.hypot(self._pos_y_np[band] - center_y, self._pos_x_np[band] - center_x) <= radius
            return np.sort(band[inside])
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
