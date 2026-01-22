from __future__ import annotations

import math
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
    center: tuple[int, int]
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
                "radius": self.radius,
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
        return 0.0 if union == 0 else common / union

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
        self._code_length = self._resolve_code_length()
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

    def activate(self, stimuli: Sequence[BitArray]) -> ActivationResult:
        activation_values = self._activation_map.compute(stimuli)
        activation_matrix = self._activation_map.to_matrix(activation_values)
        self._log_map(
            activation_values,
            event="activation.map.visual",
            section=CODE_SPACE_ACTIVATION,
            path="activation/map",
        )
        self._log_map(
            self._energies,
            event="activation.energy.visual",
            section=ENERGY_CALC,
            path="activation/energy",
        )
        active_detectors = self._evaluate_detectors(activation_values)
        embedding = self._build_embedding(active_detectors)
        LOGGER.event(
            "activation.embedding",
            section=DETECTOR_INSERTION,
            data={
                "active_detectors": len(active_detectors),
                "embedding_bits": embedding.count(),
                "saturation_limit": self._saturation_limit,
            },
        )
        return ActivationResult(
            activation_values=activation_values,
            activation_matrix=activation_matrix,
            active_detectors=active_detectors,
            embedding=embedding,
        )

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
                    "radius": detector.radius,
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
        bit_priority: dict[int, float] = {}
        for activated in active_detectors:
            code = activated.detector.output_code
            color = activated.detector.lambda_threshold
            weight = activated.activation_level * color
            if weight <= 0.0:
                continue
            for idx, bit in enumerate(code):
                if not bit:
                    continue
                current = bit_priority.get(idx)
                if current is None or weight > current:
                    bit_priority[idx] = weight
        sorted_bits = sorted(bit_priority.items(), key=lambda item: item[1], reverse=True)
        for idx, _ in sorted_bits[: self._saturation_limit]:
            merged.set(idx, 1)
        LOGGER.event(
            "activation.color_merge",
            section=COLOR_MERGE,
            data={
                "candidate_bits": len(bit_priority),
                "selected_bits": merged.count(),
                "saturation_limit": self._saturation_limit,
                "priority_max": 0.0 if not sorted_bits else sorted_bits[0][1],
                "priority_min": 0.0 if not sorted_bits else sorted_bits[-1][1],
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
        if not values:
            return
        if len(values) != self._activation_map.point_count:
            raise ValueError("map values length must match points count")
        vmax = max(values)
        scale = 1.0 / vmax if vmax > 0 else 0.0
        colors = [self._value_to_rgba(val * scale) for val in values]
        positions = self._activation_map.positions
        LOGGER.event(
            event,
            section=section,
            data={
                "points": len(values),
                "value_max": vmax,
                "value_min": min(values),
            },
            visuals=[
                LOGGER.visual_points2d(
                    path,
                    positions,
                    colors=colors,
                    radii=0.45,
                )
            ],
        )
