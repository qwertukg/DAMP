from __future__ import annotations

import math
import concurrent.futures
import random
import threading
from dataclasses import dataclass
from typing import Any, Sequence

try:
    import numpy as np
except Exception:
    np = None  # type: ignore[assignment]

from damp.article_refs import (
    CODE_SPACE_ACTIVATION,
    COLOR_MERGE,
    DETECTOR_HIERARCHY,
    DETECTOR_CLUSTERING,
    DETECTOR_CENTER,
    DETECTOR_CONSTRUCTION,
    DETECTOR_INSERTION,
    DETECTOR_PARAMS,
    DETECTOR_RADIUS,
    ENERGY_CALC,
    STIMULUS_DETECTION,
    GPU_IMPLEMENTATION,
    PARALLEL_PROCESSING,
)
from damp.encoding.bitarray import BitArray
from damp.layout.damp_layout import Layout
from damp.logging import LOGGER


@dataclass(frozen=True)
class Detector:
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
                "detector_label": "" if self.label is None else str(self.label),
            },
        )


@dataclass(frozen=True)
class ActivatedDetector:
    detector: Detector
    activation_level: float
    active_points: int


@dataclass(frozen=True)
class ActivationResult:
    activation_values: list[float]
    activation_matrix: list[list[float]]
    active_detectors: list[ActivatedDetector]
    embedding: BitArray


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
        self._similarity = similarity
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
        common = sum(1 for a, b in zip(code_a, code_b) if a and b)
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
        self._similarity = similarity
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
            return [0.0 for _ in range(self._codes.shape[0])]
        best = sim.max(axis=1)
        return best.tolist()

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
        self._similarity = similarity
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
                return [0.0 for _ in range(self._codes.shape[0])]
            best = torch.max(sim, dim=1).values
            return best.detach().cpu().tolist()

    def _stack_stimuli(self, stimuli: Sequence[BitArray]):
        torch = self._torch
        rows = []
        for code in stimuli:
            if len(code) != self._code_length:
                raise ValueError("stimulus code length must match layout code length")
            if self._np is not None:
                raw = self._np.frombuffer(code._bits, dtype=self._np.uint8, count=self._code_length)
                tensor_row = torch.as_tensor(raw, dtype=torch.float32, device=self._codes.device)
            else:
                tensor_row = torch.as_tensor(list(code), dtype=torch.float32, device=self._codes.device)
            rows.append(tensor_row)
        if not rows:
            return torch.zeros((0, self._code_length), dtype=torch.float32, device=self._codes.device)
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
            "activation.engine.init",
            section=CODE_SPACE_ACTIVATION,
            data={
                "points": len(self._codes),
                "lambda_threshold": self._lambda_threshold,
                "eta": 0.0 if self._eta is None else self._eta,
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
    def positions(self) -> list[tuple[int, int]]:
        return self._positions

    def compute(self, stimuli: Sequence[BitArray]) -> list[float]:
        activation = self._backend.compute(stimuli)
        if len(activation) != self.point_count:
            raise ValueError("activation size must match number of layout points")
        LOGGER.event(
            "activation.map",
            section=CODE_SPACE_ACTIVATION,
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
                section=GPU_IMPLEMENTATION,
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
                section=GPU_IMPLEMENTATION,
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
                section=GPU_IMPLEMENTATION,
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
            section=GPU_IMPLEMENTATION if backend.uses_gpu else PARALLEL_PROCESSING,
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


class _ActivationEngineBase:
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
        self._code_length = self._resolve_code_length()
        self._activation_sequence = 0
        self._activation_sequence_lock = threading.Lock()
        LOGGER.event(
            "activation.engine.init",
            section=DETECTOR_HIERARCHY,
            data={
                "detectors": len(self._detectors),
                "points": self._activation_map.point_count,
                "energy_threshold": self._energy_threshold,
                "detector_threshold": self._detector_threshold,
                "saturation_limit": self._saturation_limit,
            },
        )
        self._log_structure_overview()

    def activate(self, stimuli: Sequence[BitArray]) -> ActivationResult:
        activation_values = self._activation_map.compute(stimuli)
        activation_matrix = self._activation_map.to_matrix(activation_values)
        self._log_map(
            activation_values,
            event="activation.map.visual",
            section=CODE_SPACE_ACTIVATION,
            path="activation/latest",
        )
        active_detectors = self._evaluate_detectors(activation_values)
        embedding = self._build_embedding(active_detectors)
        with self._activation_sequence_lock:
            self._activation_sequence += 1
            activation_sequence = self._activation_sequence
        activation_max = max(activation_values, default=0.0)
        activation_mean = (
            sum(activation_values) / len(activation_values)
            if activation_values
            else 0.0
        )
        active_points = sum(value > 0.0 for value in activation_values)
        LOGGER.event(
            "activation.embedding",
            section=DETECTOR_INSERTION,
            data={
                "active_detectors": len(active_detectors),
                "embedding_bits": embedding.count(),
                "saturation_limit": self._saturation_limit,
            },
        )
        LOGGER.event(
            "activation.progress",
            section=CODE_SPACE_ACTIVATION,
            data={
                "sample": activation_sequence,
                "activation_max": activation_max,
                "activation_mean": activation_mean,
                "active_points": active_points,
                "active_detectors": len(active_detectors),
                "embedding_bits": embedding.count(),
            },
            visuals=[
                *LOGGER.visual_scalars(
                    "activation/metrics/map",
                    {
                        "maximum": activation_max,
                        "mean": activation_mean,
                        "active_points": active_points,
                    },
                    step=activation_sequence,
                    timeline="activation_sample",
                ),
                *LOGGER.visual_scalars(
                    "activation/metrics/embedding",
                    {
                        "active_detectors": len(active_detectors),
                        "active_bits": embedding.count(),
                    },
                    step=activation_sequence,
                    timeline="activation_sample",
                ),
            ],
        )
        return ActivationResult(
            activation_values=activation_values,
            activation_matrix=activation_matrix,
            active_detectors=active_detectors,
            embedding=embedding,
        )

    def log_activation_overview(self, values: Sequence[float]) -> None:
        """Фиксирует последнюю карту независимо от интервала (§6.1)."""
        self._log_map(
            values,
            event="activation.map.visual",
            section=CODE_SPACE_ACTIVATION,
            path="activation/latest",
            force=True,
        )


@dataclass(frozen=True)
class DetectorCandidate:
    center: tuple[float, float]
    radius: float
    energy: float
    points_count: int
    fill_factor: float
    insertion_density: float


class DetectorHierarchyBuilder:
    """Стохастически строит слои детекторов по §6.3–6.4.5."""

    def __init__(
        self,
        *,
        layout: Layout,
        energies: Sequence[float],
        radius: int,
        radius_min: int,
        min_points: int,
        max_detectors: int,
        output_bits: int,
        lambda_levels: Sequence[float],
        energy_threshold: float,
        eta: float | None,
        attempts_without_change: int,
        seed: int,
    ) -> None:
        if radius <= 0 or radius_min <= 0 or radius_min > radius:
            raise ValueError("invalid detector radius range")
        if min_points <= 0:
            raise ValueError("min_points must be positive")
        if max_detectors <= 0 or output_bits <= 0:
            raise ValueError("detector and output sizes must be positive")
        if len(energies) != len(layout.positions()):
            raise ValueError("energies size must match layout points")
        if not lambda_levels:
            raise ValueError("lambda_levels must be non-empty")
        if attempts_without_change <= 0:
            raise ValueError("attempts_without_change must be positive")
        self._layout = layout
        self._energies = [float(value) for value in energies]
        self._radius = int(radius)
        self._radius_min = int(radius_min)
        self._min_points = int(min_points)
        self._max_detectors = int(max_detectors)
        self._output_bits = int(output_bits)
        self._lambda_levels = tuple(float(value) for value in lambda_levels)
        self._energy_threshold = float(energy_threshold)
        self._eta = None if eta is None else float(eta)
        self._attempts_without_change = int(attempts_without_change)
        self._rng = random.Random(seed)
        LOGGER.event(
            "activation.detector.builder.init",
            section=DETECTOR_CONSTRUCTION,
            data={
                "layers": len(self._lambda_levels),
                "max_detectors": self._max_detectors,
                "output_bits": self._output_bits,
                "attempts_without_change": self._attempts_without_change,
                "seed": seed,
            },
        )

    def build(self) -> list[Detector]:
        per_layer_limit = max(1, self._max_detectors // len(self._lambda_levels))
        detectors: list[Detector] = []
        for lambda_value in self._lambda_levels:
            candidates = self._build_layer(lambda_value, per_layer_limit)
            for candidate in candidates:
                output_code = BitArray(self._output_bits)
                output_code.set(self._rng.randrange(self._output_bits), 1)
                detectors.append(
                    Detector(
                        center=candidate.center,
                        radius=candidate.radius,
                        lambda_threshold=lambda_value,
                        energy=candidate.energy,
                        points_count=candidate.points_count,
                        output_code=output_code,
                        label=f"λ={lambda_value:.2f}",
                    )
                )
            LOGGER.event(
                "activation.detector.layer",
                section=DETECTOR_CONSTRUCTION,
                data={
                    "lambda": lambda_value,
                    "layer_detectors": len(candidates),
                    "per_layer_limit": per_layer_limit,
                    "fill_min": min((item.fill_factor for item in candidates), default=0.0),
                    "fill_max": max((item.fill_factor for item in candidates), default=0.0),
                    "radius_min": min((item.radius for item in candidates), default=0.0),
                    "radius_max": max((item.radius for item in candidates), default=0.0),
                },
            )
        if not detectors:
            raise RuntimeError("не удалось построить ни одного детектора для активации")
        return detectors

    def _build_layer(self, lambda_value: float, limit: int) -> list[DetectorCandidate]:
        selected: list[DetectorCandidate] = []
        unchanged = 0
        point_count = len(self._energies)
        while unchanged < self._attempts_without_change:
            center_index = self._rng.randrange(point_count)
            candidate = self._make_candidate(center_index, lambda_value)
            if candidate is None:
                unchanged += 1
                continue
            overlaps = [item for item in selected if self._overlaps(candidate, item)]
            if overlaps:
                if any(item.insertion_density >= candidate.insertion_density for item in overlaps):
                    unchanged += 1
                    continue
                selected = [item for item in selected if item not in overlaps]
                selected.append(candidate)
                unchanged = 0
                continue
            if len(selected) < limit:
                selected.append(candidate)
                unchanged = 0
            else:
                unchanged += 1
        selected.sort(key=lambda item: item.insertion_density, reverse=True)
        return selected[:limit]

    def _make_candidate(self, center_index: int, lambda_value: float) -> DetectorCandidate | None:
        if self._energies[center_index] < self._energy_threshold:
            return None
        center_y = self._layout._pos_y[center_index]
        center_x = self._layout._pos_x[center_index]
        active: dict[tuple[int, int], tuple[int, float]] = {}
        for point_index in self._indices_in_radius(center_y, center_x, self._radius):
            point_energy = self._energies[point_index]
            if point_energy < self._energy_threshold:
                continue
            similarity = self._layout._similarity_value(center_index, point_index)
            activation = self._threshold_similarity(similarity, lambda_value)
            if activation <= 0.0:
                continue
            position = (self._layout._pos_y[point_index], self._layout._pos_x[point_index])
            active[position] = (point_index, activation * point_energy)
        seed = (center_y, center_x)
        if seed not in active:
            return None
        cluster_positions = self._connected_component(seed, active)
        if len(cluster_positions) < self._min_points:
            return None
        LOGGER.event(
            "activation.detector.cluster",
            section=DETECTOR_CLUSTERING,
            data={"points": len(cluster_positions), "lambda": lambda_value},
        )
        weight_sum = sum(active[position][1] for position in cluster_positions)
        if weight_sum <= 0.0:
            return None
        centroid_y = sum(position[0] * active[position][1] for position in cluster_positions) / weight_sum
        centroid_x = sum(position[1] * active[position][1] for position in cluster_positions) / weight_sum
        LOGGER.event(
            "activation.detector.center",
            section=DETECTOR_CENTER,
            data={"center_y": centroid_y, "center_x": centroid_x, "weight_sum": weight_sum},
        )
        distances = sorted(
            math.hypot(position[0] - centroid_y, position[1] - centroid_x)
            for position in cluster_positions
        )
        best_radius = 0.0
        best_fill = -1.0
        for distance in distances:
            candidate_radius = max(float(self._radius_min), distance)
            if candidate_radius > self._radius:
                continue
            points_count = sum(value <= candidate_radius for value in distances)
            if points_count < self._min_points:
                continue
            fill_factor = points_count / (math.pi * candidate_radius * candidate_radius)
            if fill_factor > best_fill:
                best_fill = fill_factor
                best_radius = candidate_radius
        if best_radius <= 0.0:
            return None
        included = [
            position
            for position in cluster_positions
            if math.hypot(position[0] - centroid_y, position[1] - centroid_x) <= best_radius
        ]
        detector_energy = sum(active[position][1] for position in included)
        points_count = len(included)
        if detector_energy <= 0.0 or points_count < self._min_points:
            return None
        return DetectorCandidate(
            center=(centroid_y, centroid_x),
            radius=best_radius,
            energy=detector_energy,
            points_count=points_count,
            fill_factor=best_fill,
            insertion_density=points_count / best_radius,
        )

    def _threshold_similarity(self, similarity: float, lambda_value: float) -> float:
        if similarity <= 0.0:
            return 0.0
        if self._eta is None:
            return similarity if similarity >= lambda_value else 0.0
        return similarity / (1.0 + math.exp(-self._eta * (similarity - lambda_value)))

    def _indices_in_radius(self, center_y: int, center_x: int, radius: int) -> list[int]:
        indices: list[int] = []
        radius_sq = radius * radius
        for y in range(max(0, center_y - radius), min(self._layout.height - 1, center_y + radius) + 1):
            dy_sq = (center_y - y) ** 2
            for x in self._layout._row_occupied[y]:
                if x < center_x - radius or x > center_x + radius:
                    continue
                if dy_sq + (center_x - x) ** 2 > radius_sq:
                    continue
                point_index = self._layout.grid[y][x]
                if point_index is not None:
                    indices.append(point_index)
        return indices

    @staticmethod
    def _connected_component(
        seed: tuple[int, int],
        active: dict[tuple[int, int], tuple[int, float]],
    ) -> list[tuple[int, int]]:
        pending = [seed]
        visited = {seed}
        while pending:
            y, x = pending.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    neighbour = (y + dy, x + dx)
                    if neighbour in active and neighbour not in visited:
                        visited.add(neighbour)
                        pending.append(neighbour)
        return list(visited)

    @staticmethod
    def _overlaps(left: DetectorCandidate, right: DetectorCandidate) -> bool:
        distance = math.hypot(left.center[0] - right.center[0], left.center[1] - right.center[1])
        return distance <= left.radius or distance <= right.radius


class ActivationEngine(_ActivationEngineBase):
    def activate_many(
        self,
        stimuli_batch: Sequence[Sequence[BitArray]],
        *,
        max_workers: int,
    ) -> list[ActivationResult]:
        """Активирует независимые стимулы параллельно, сохраняя порядок входа."""
        workers = int(max_workers)
        if workers <= 0:
            raise ValueError("max_workers must be positive")
        batches = list(stimuli_batch)
        if not batches:
            return []
        effective_workers = min(workers, len(batches))
        LOGGER.event(
            "activation.parallel.start",
            section=PARALLEL_PROCESSING,
            data={
                "images": len(batches),
                "workers": effective_workers,
                "backend": self._activation_map.backend_name,
            },
        )
        if effective_workers == 1:
            results = [self.activate(stimuli) for stimuli in batches]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
                results = list(executor.map(self.activate, batches))
        if results:
            self.log_activation_overview(results[-1].activation_values)
        LOGGER.event(
            "activation.parallel.done",
            section=PARALLEL_PROCESSING,
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
        coverage: list[tuple[int, ...] | "np.ndarray"] = []
        positions = self._activation_map.positions
        for idx, detector in enumerate(self._detectors):
            radius_sq = detector.radius * detector.radius
            covered: list[int] = []
            for point_idx, (y, x) in enumerate(positions):
                dy = float(y) - float(detector.center[0])
                dx = float(x) - float(detector.center[1])
                if dy * dy + dx * dx <= radius_sq:
                    covered.append(point_idx)
            if np is not None:
                coverage.append(np.asarray(covered, dtype=np.int32))
            else:
                coverage.append(tuple(covered))
            LOGGER.event(
                "activation.detector.coverage",
                section=DETECTOR_RADIUS,
                data={
                    "detector": idx,
                    "detector_label": "" if detector.label is None else str(detector.label),
                    "receptive_radius": detector.radius,
                    "covered_points": len(covered),
                },
            )
        return coverage

    def _evaluate_detectors(self, activation_values: Sequence[float]) -> list[ActivatedDetector]:
        if len(activation_values) != self._activation_map.point_count:
            raise ValueError("activation size must match number of layout points")
        active: list[ActivatedDetector] = []
        activation_np = np.asarray(activation_values, dtype=np.float32) if np is not None else None
        energies_np = self._energies_array if np is not None else None
        for idx, detector in enumerate(self._detectors):
            indices_obj = self._coverage[idx]
            index_array: "np.ndarray | None"
            indices_iterable: Sequence[int]
            if isinstance(indices_obj, tuple):
                if not indices_obj:
                    LOGGER.event(
                        "activation.detector.level",
                        section=STIMULUS_DETECTION,
                        data={
                            "detector": idx,
                            "detector_label": "" if detector.label is None else str(detector.label),
                            "activation_level": 0.0,
                            "points_used": 0,
                            "energy_sum": 0.0,
                        },
                    )
                    continue
                index_array = np.fromiter(indices_obj, dtype=np.int32) if np is not None else None
                indices_iterable = indices_obj
            else:
                index_array = indices_obj
                if index_array.size == 0:
                    LOGGER.event(
                        "activation.detector.level",
                        section=STIMULUS_DETECTION,
                        data={
                            "detector": idx,
                            "detector_label": "" if detector.label is None else str(detector.label),
                            "activation_level": 0.0,
                            "points_used": 0,
                            "energy_sum": 0.0,
                        },
                    )
                    continue
                indices_iterable = ()
            energy_sum = 0.0
            points_used = 0
            if (
                np is not None
                and activation_np is not None
                and energies_np is not None
                and index_array is not None
            ):
                act_slice = activation_np[index_array]
                energy_slice = energies_np[index_array]
                valid_mask = (energy_slice >= self._energy_threshold) & (act_slice > 0.0)
                if valid_mask.any():
                    weighted = act_slice[valid_mask] * energy_slice[valid_mask]
                    energy_sum = float(weighted.sum())
                    points_used = int(valid_mask.sum())
            else:
                iterable_indices = (
                    indices_iterable
                    if indices_iterable
                    else tuple(int(v) for v in index_array.tolist()) if index_array is not None else ()
                )
                for point_idx in iterable_indices:
                    energy = self._energies[point_idx]
                    if energy < self._energy_threshold:
                        continue
                    act_value = activation_values[point_idx]
                    if act_value <= 0.0:
                        continue
                    energy_sum += act_value * energy
                    points_used += 1
            activation_level = 0.0 if detector.energy <= 0 else energy_sum / detector.energy
            LOGGER.event(
                "activation.detector.level",
                section=STIMULUS_DETECTION,
                data={
                    "detector": idx,
                    "detector_label": "" if detector.label is None else str(detector.label),
                    "activation_level": activation_level,
                    "points_used": points_used,
                    "energy_sum": energy_sum,
                },
            )
            if activation_level >= self._detector_threshold:
                activated = ActivatedDetector(
                    detector=detector,
                    activation_level=activation_level,
                    active_points=points_used,
                )
                active.append(activated)
                LOGGER.event(
                    "activation.detector.active",
                    section=STIMULUS_DETECTION,
                    data={
                        "detector": idx,
                        "detector_label": "" if detector.label is None else str(detector.label),
                        "activation_level": activation_level,
                        "points_used": points_used,
                    },
                )
        return active

    def _build_embedding(self, active_detectors: Sequence[ActivatedDetector]) -> BitArray:
        merged = BitArray(self._code_length)
        if not active_detectors:
            return merged

        # §7.1.12: динамический порог mu_c выбирается так, чтобы |D_mu| ~ sigma.
        selected_detectors = sorted(
            active_detectors,
            key=lambda item: item.activation_level,
            reverse=True,
        )[: self._saturation_limit]
        mu_c = selected_detectors[-1].activation_level if selected_detectors else 0.0

        ranked: list[tuple[float, ActivatedDetector]] = []
        for activated in selected_detectors:
            color = activated.detector.lambda_threshold
            priority = color
            if priority > 0.0:
                ranked.append((priority, activated))

        if not ranked:
            return merged

        ranked.sort(key=lambda item: item[0], reverse=True)

        bits_count = 0
        used_detectors = 0
        skipped_detectors = 0

        for priority, activated in ranked:
            if bits_count >= self._saturation_limit:
                skipped_detectors += 1
                continue

            code = activated.detector.output_code

            # Count how many *new* bits this detector would add.
            new_bits = 0
            for idx, bit in enumerate(code):
                if bit and not merged[idx]:
                    new_bits += 1

            if new_bits == 0:
                continue

            # To preserve detector structure (as in detector insertion / color merge),
            # we either insert the detector code as a whole or skip it.
            if bits_count + new_bits > self._saturation_limit:
                skipped_detectors += 1
                continue

            for idx, bit in enumerate(code):
                if bit and not merged[idx]:
                    merged.set(idx, 1)
                    bits_count += 1

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
                "priority_max": ranked[0][0],
                "priority_min": ranked[-1][0],
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

    @staticmethod
    def _detector_color(value: float) -> tuple[int, int, int, int]:
        v = max(0.0, min(1.0, value))
        return (
            int(255.0 * v),
            int(255.0 * (1.0 - abs(2.0 * v - 1.0))),
            int(255.0 * (1.0 - v)),
            255,
        )

    def _log_structure_overview(self) -> None:
        positions = self._activation_map.positions
        point_positions = [(x + 0.5, y + 0.5) for y, x in positions]
        energy_max = max(self._energies, default=0.0)
        energy_scale = 1.0 / energy_max if energy_max > 0.0 else 0.0
        energy_colors = [
            self._value_to_rgba(value * energy_scale) for value in self._energies
        ]
        detector_mins = [
            (detector.center[1] - detector.radius, detector.center[0] - detector.radius)
            for detector in self._detectors
        ]
        detector_sizes = [
            (detector.radius * 2.0, detector.radius * 2.0)
            for detector in self._detectors
        ]
        detector_positions = [
            (detector.center[1], detector.center[0]) for detector in self._detectors
        ]
        detector_colors = [
            self._detector_color(detector.lambda_threshold)
            for detector in self._detectors
        ]
        visuals = [
            LOGGER.visual_scalar_map(
                "layout/diagnostics/local_energy/map",
                self._energies,
                positions,
                height=self._activation_map.height,
                width=self._activation_map.width,
                static=True,
            ),
            LOGGER.visual_points2d(
                "layout/diagnostics/local_energy/points",
                point_positions,
                colors=energy_colors,
                radii=0.45,
                static=True,
            ),
            LOGGER.visual_boxes2d(
                "layout/diagnostics/detectors/receptive_fields",
                detector_mins,
                detector_sizes,
                colors=detector_colors,
                static=True,
            ),
            LOGGER.visual_points2d(
                "layout/diagnostics/detectors/centers",
                detector_positions,
                colors=detector_colors,
                radii=0.65,
                static=True,
            ),
        ]
        LOGGER.event(
            "activation.structure.visual",
            section=DETECTOR_HIERARCHY,
            data={
                "points": len(positions),
                "detectors": len(self._detectors),
                "energy_min": min(self._energies, default=0.0),
                "energy_max": energy_max,
                "radius_min": min(
                    (detector.radius for detector in self._detectors), default=0.0
                ),
                "radius_max": max(
                    (detector.radius for detector in self._detectors), default=0.0
                ),
            },
            visuals=visuals,
        )

    def _log_map(
        self,
        values: Sequence[float],
        *,
        event: str,
        section: str,
        path: str,
        force: bool = False,
    ) -> None:
        if not values:
            return
        if len(values) != self._activation_map.point_count:
            raise ValueError("map values length must match points count")
        log_visual = force or LOGGER.should_log(event)
        if not log_visual:
            return
        vmax = max(values)
        scale = 1.0 / vmax if vmax > 0 else 0.0
        colors = [self._value_to_rgba(val * scale) for val in values]
        positions = self._activation_map.positions
        point_positions = [(x + 0.5, y + 0.5) for y, x in positions]
        LOGGER.event(
            event,
            section=section,
            data={
                "points": len(values),
                "value_max": vmax,
                "value_min": min(values),
            },
            visuals=[
                LOGGER.visual_scalar_map(
                    f"{path}/map",
                    values,
                    positions,
                    height=self._activation_map.height,
                    width=self._activation_map.width,
                ),
                LOGGER.visual_points2d(
                    f"{path}/points",
                    point_positions,
                    colors=colors,
                    radii=0.45,
                )
            ],
            force=True,
        )
