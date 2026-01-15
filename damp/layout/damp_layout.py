from __future__ import annotations

import bisect
import concurrent.futures
from dataclasses import dataclass, field
from collections import deque
import json
import math
import os
import random
from typing import Any, Iterable, Mapping, Sequence

from damp.article_refs import (
    ENERGY_CALC,
    ENERGIES,
    ENERGY_LONG,
    ENERGY_SHORT,
    GPU_IMPLEMENTATION,
    LAYOUT_ALGORITHM,
    LAYOUT_COMPACTNESS,
    LAYOUT_PARAMETERS,
    OPTIM_ENERGY,
    OPTIM_FIRST_POINT,
    OPTIM_PAIR_SELECTION,
    OPTIM_SIM_MATRIX,
    OPTIM_SUBSET,
    LAID_OUT_STRUCTURE,
    PAIR_SELECTION,
    PARALLEL_PROCESSING,
    QUALITY_ASSESS,
    SIMILARITY_DEFINITION,
    SIMILARITY_MEASURES,
    SPARSE_BIT_VECTORS,
)
from damp.encoding.bitarray import BitArray
from damp.logging import LOGGER


@dataclass(frozen=True)
class LayoutPoint:
    code: BitArray
    label: float
    hue: float
    ones: int


@dataclass(frozen=True)
class AdaptiveLayoutConfig:
    start_radius: int
    end_radius: int = 1
    swap_ratio_trigger: float = 0.01
    lambda_step: float | None = None


@dataclass(frozen=True)
class ShortLayoutOptimConfig:
    energy_radius: int
    energy_max_points: int | None = None
    energy_recalc_every: int = 50
    energy_eps: float = 1e-6
    use_weighted_first_point: bool = True
    similarity_cutoff: float | None = None
    partitions: int = 1


@dataclass(frozen=True)
class _ShortPair:
    y1: int
    x1: int
    y2: int
    x2: int
    idx_a: int | None
    idx_b: int | None
    radius: float

_UNSET = object()
HUE = 360.0
_PARALLEL_SIM_MIN_PAIRS = 50_000
_SIM_POINTS: Sequence[LayoutPoint] | None = None
_SIM_MODE: str | None = None
_VISUAL_ENERGY_POINT_LIMIT = 5000
_VISUAL_POINT_LIMIT = 200000
_AUTO_ENERGY_RADIUS_MAX = 15
_LAYOUT_STEP_TIMELINE = "layout/step"


def _init_similarity_worker(points: Sequence[LayoutPoint], similarity: str) -> None:
    global _SIM_POINTS, _SIM_MODE
    _SIM_POINTS = points
    _SIM_MODE = similarity


def _similarity_base_points(a: LayoutPoint, b: LayoutPoint, similarity: str) -> float:
    if a.ones == 0 or b.ones == 0:
        return 0.0
    common = a.code.common(b.code)
    if similarity == "cosine":
        denom = math.sqrt(a.ones * b.ones)
        return 0.0 if denom == 0 else common / denom
    union = a.ones + b.ones - common
    return 0.0 if union == 0 else common / union


def _compute_similarity_stride(
    args: tuple[int, int, int]
) -> list[tuple[int, list[float]]]:
    start, step, point_count = args
    points = _SIM_POINTS
    similarity = _SIM_MODE
    if points is None or similarity is None:
        raise RuntimeError("Similarity worker not initialized")
    results: list[tuple[int, list[float]]] = []
    for i in range(start, point_count, step):
        a = points[i]
        sims = []
        for j in range(i + 1, point_count):
            sims.append(_similarity_base_points(a, points[j], similarity))
        results.append((i, sims))
    return results


class _GpuLayoutEngine:
    _VERTEX_SHADER = """
        #version 410
        const vec2 verts[3] = vec2[3](
            vec2(-1.0, -1.0),
            vec2(3.0, -1.0),
            vec2(-1.0, 3.0)
        );
        void main() {
            gl_Position = vec4(verts[gl_VertexID], 0.0, 1.0);
        }
    """
    _SIM_SHADER = """
        #version 410
        uniform sampler2D codes;
        uniform sampler2D ones;
        uniform int code_len;
        uniform int point_count;
        uniform int use_cosine;
        out float frag;

        void main() {
            ivec2 coord = ivec2(gl_FragCoord.xy);
            int i = coord.y;
            int j = coord.x;
            if (i >= point_count || j >= point_count) {
                frag = 0.0;
                return;
            }
            if (i == j) {
                frag = 1.0;
                return;
            }

            float ones_a = texelFetch(ones, ivec2(i, 0), 0).r;
            float ones_b = texelFetch(ones, ivec2(j, 0), 0).r;
            if (ones_a <= 0.0 || ones_b <= 0.0) {
                frag = 0.0;
                return;
            }

            float common_bits = 0.0;
            for (int k = 0; k < code_len; k++) {
                float a = texelFetch(codes, ivec2(k, i), 0).r;
                float b = texelFetch(codes, ivec2(k, j), 0).r;
                common_bits += a * b;
            }

            if (use_cosine == 1) {
                float denom = sqrt(ones_a * ones_b);
                frag = denom == 0.0 ? 0.0 : common_bits / denom;
                return;
            }
            float uni = ones_a + ones_b - common_bits;
            frag = uni == 0.0 ? 0.0 : common_bits / uni;
        }
    """
    _PAIR_SHADER = """
        #version 410
        uniform sampler2D sim_tex;
        uniform sampler2D pos_tex;
        uniform sampler2D pair_idx_tex;
        uniform sampler2D pair_pos_tex;
        uniform int point_count;
        uniform int mode;
        uniform float radius;
        uniform int use_radius_override;
        uniform float lambda_threshold;
        uniform float eta;
        uniform int eta_enabled;
        uniform float distance_eps;
        out vec4 frag;

        float apply_lambda(float sim) {
            if (sim <= 0.0) {
                return 0.0;
            }
            if (eta_enabled == 0) {
                return sim >= lambda_threshold ? sim : 0.0;
            }
            float scaled = 1.0 / (1.0 + exp(-eta * (sim - lambda_threshold)));
            return sim * scaled;
        }

        void main() {
            int pair_idx = int(gl_FragCoord.x);
            vec2 idxs = texelFetch(pair_idx_tex, ivec2(pair_idx, 0), 0).rg;
            int idx_a = int(idxs.r);
            int idx_b = int(idxs.g);
            vec4 pos = texelFetch(pair_pos_tex, ivec2(pair_idx, 0), 0);
            float pos_a_y = pos.r;
            float pos_a_x = pos.g;
            float pos_b_y = pos.b;
            float pos_b_x = pos.a;

            float energy_c = 0.0;
            float energy_s = 0.0;
            float cy = 0.0;
            float cx = 0.0;
            float radius_sq = 0.0;
            if (mode == 1) {
                cy = (pos_a_y + pos_b_y) * 0.5;
                cx = (pos_a_x + pos_b_x) * 0.5;
                float dy = pos_a_y - pos_b_y;
                float dx = pos_a_x - pos_b_x;
                float dist = sqrt(dy * dy + dx * dx);
                float base_rad = max(1.0, ceil(dist));
                float rad = base_rad;
                if (use_radius_override == 1) {
                    rad = max(rad, radius);
                }
                radius_sq = rad * rad;
            }

            for (int idx = 0; idx < point_count; idx++) {
                if (idx == idx_a || idx == idx_b) {
                    continue;
                }
                vec2 p = texelFetch(pos_tex, ivec2(idx, 0), 0).rg;
                float oy = p.r;
                float ox = p.g;
                if (mode == 1) {
                    float dyc = oy - cy;
                    float dxc = ox - cx;
                    if (dyc * dyc + dxc * dxc > radius_sq) {
                        continue;
                    }
                }
                float s1 = apply_lambda(texelFetch(sim_tex, ivec2(idx_a, idx), 0).r);
                float s2 = 0.0;
                if (idx_b >= 0) {
                    s2 = apply_lambda(texelFetch(sim_tex, ivec2(idx_b, idx), 0).r);
                }
                if (s1 == 0.0 && s2 == 0.0) {
                    continue;
                }
                float dy1 = pos_a_y - oy;
                float dx1 = pos_a_x - ox;
                float dy2 = pos_b_y - oy;
                float dx2 = pos_b_x - ox;
                float d1 = dy1 * dy1 + dx1 * dx1;
                float d2 = dy2 * dy2 + dx2 * dx2;
                if (mode == 0) {
                    energy_c += s1 * d1 + s2 * d2;
                    energy_s += s2 * d1 + s1 * d2;
                } else {
                    energy_c += s1 / (d1 + distance_eps) + s2 / (d2 + distance_eps);
                    energy_s += s2 / (d1 + distance_eps) + s1 / (d2 + distance_eps);
                }
            }

            frag = vec4(energy_c, energy_s, 0.0, 0.0);
        }
    """
    _AVG_SHADER = """
        #version 410
        uniform sampler2D sim_tex;
        uniform sampler2D pos_tex;
        uniform int point_count;
        uniform float radius;
        uniform float lambda_threshold;
        uniform float eta;
        uniform int eta_enabled;
        uniform float distance_eps;
        out float frag;

        float apply_lambda(float sim) {
            if (sim <= 0.0) {
                return 0.0;
            }
            if (eta_enabled == 0) {
                return sim >= lambda_threshold ? sim : 0.0;
            }
            float scaled = 1.0 / (1.0 + exp(-eta * (sim - lambda_threshold)));
            return sim * scaled;
        }

        void main() {
            int idx = int(gl_FragCoord.x);
            if (idx >= point_count) {
                frag = 0.0;
                return;
            }
            vec2 cpos = texelFetch(pos_tex, ivec2(idx, 0), 0).rg;
            float energy = 0.0;
            float radius_sq = radius * radius;
            for (int j = 0; j < point_count; j++) {
                if (j == idx) {
                    continue;
                }
                vec2 p = texelFetch(pos_tex, ivec2(j, 0), 0).rg;
                float dy = cpos.r - p.r;
                float dx = cpos.g - p.g;
                float dist_sq = dy * dy + dx * dx;
                if (dist_sq > radius_sq) {
                    continue;
                }
                float sim = apply_lambda(texelFetch(sim_tex, ivec2(j, idx), 0).r);
                if (sim <= 0.0) {
                    continue;
                }
                energy += sim / (dist_sq + distance_eps);
            }
            frag = energy;
        }
    """

    def __init__(
        self,
        ctx: "moderngl.Context",
        np: "numpy",
        gl: "moderngl",
        points: Sequence[LayoutPoint],
        similarity: str,
    ) -> None:
        self._ctx = ctx
        self._np = np
        self._gl = gl
        self._point_count = len(points)
        if self._point_count <= 0:
            raise RuntimeError("GPU layout requires at least one point")
        self._code_len = len(points[0].code)
        self._similarity = similarity

        max_size = getattr(self._ctx, "max_texture_size", 0) or 0
        if max_size and (self._point_count > max_size or self._code_len > max_size):
            raise RuntimeError("GPU texture size too small for layout")

        codes = self._np.zeros((self._point_count, self._code_len), dtype=self._np.float32)
        ones = self._np.zeros(self._point_count, dtype=self._np.float32)
        for idx, point in enumerate(points):
            ones[idx] = point.ones
            bits = self._np.frombuffer(
                point.code._bits, dtype=self._np.uint8, count=self._code_len
            )
            codes[idx] = bits.astype(self._np.float32)

        self._codes_data = codes
        self._ones_data = ones

        self._pos_tex = self._ctx.texture((self._point_count, 1), 2, dtype="f4")
        self._pos_tex.filter = (self._gl.NEAREST, self._gl.NEAREST)
        self._sim_tex = None
        self._sim_fbo = None

        self._pair_capacity = 0
        self._pair_idx_tex = None
        self._pair_pos_tex = None
        self._pair_out_tex = None
        self._pair_fbo = None

        self._avg_out_tex = None
        self._avg_fbo = None

        self._sim_prog = self._ctx.program(
            vertex_shader=self._VERTEX_SHADER, fragment_shader=self._SIM_SHADER
        )
        self._sim_vao = self._ctx.vertex_array(self._sim_prog, [])

        self._pair_prog = self._ctx.program(
            vertex_shader=self._VERTEX_SHADER, fragment_shader=self._PAIR_SHADER
        )
        self._pair_vao = self._ctx.vertex_array(self._pair_prog, [])

        self._avg_prog = self._ctx.program(
            vertex_shader=self._VERTEX_SHADER, fragment_shader=self._AVG_SHADER
        )
        self._avg_vao = self._ctx.vertex_array(self._avg_prog, [])

    @classmethod
    def create(
        cls, points: Sequence[LayoutPoint], similarity: str
    ) -> "_GpuLayoutEngine | None":
        try:
            import numpy as np
            import moderngl
        except Exception as exc:
            LOGGER.event(
                "layout.gpu.import_failed",
                section=GPU_IMPLEMENTATION,
                data={"error": str(exc)},
            )
            return None

        try:
            ctx = moderngl.create_standalone_context(require=410)
        except Exception:
            try:
                ctx = moderngl.create_standalone_context()
            except Exception as exc:
                LOGGER.event(
                    "layout.gpu.context_failed",
                    section=GPU_IMPLEMENTATION,
                    data={"error": str(exc)},
                )
                return None
            if getattr(ctx, "version_code", 0) < 410:
                LOGGER.event(
                    "layout.gpu.version_too_low",
                    section=GPU_IMPLEMENTATION,
                    data={"version_code": getattr(ctx, "version_code", 0)},
                )
                return None

        try:
            return cls(ctx, np, moderngl, points, similarity)
        except Exception as exc:
            LOGGER.event(
                "layout.gpu.init_failed",
                section=GPU_IMPLEMENTATION,
                data={"error": str(exc)},
            )
            return None

    @property
    def has_similarity(self) -> bool:
        return self._sim_tex is not None

    def build_similarity_matrix(self) -> list[list[float]] | None:
        point_count = self._point_count
        if point_count == 0:
            return []

        codes_tex = self._ctx.texture(
            (self._code_len, point_count),
            1,
            data=self._codes_data.tobytes(),
            dtype="f4",
        )
        codes_tex.filter = (self._gl.NEAREST, self._gl.NEAREST)
        ones_tex = self._ctx.texture(
            (point_count, 1), 1, data=self._ones_data.tobytes(), dtype="f4"
        )
        ones_tex.filter = (self._gl.NEAREST, self._gl.NEAREST)

        if self._sim_tex is None:
            self._sim_tex = self._ctx.texture((point_count, point_count), 1, dtype="f4")
            self._sim_tex.filter = (self._gl.NEAREST, self._gl.NEAREST)
            self._sim_fbo = self._ctx.framebuffer(color_attachments=[self._sim_tex])

        codes_tex.use(location=0)
        ones_tex.use(location=1)
        self._sim_prog["codes"].value = 0
        self._sim_prog["ones"].value = 1
        self._sim_prog["code_len"].value = self._code_len
        self._sim_prog["point_count"].value = point_count
        self._sim_prog["use_cosine"].value = 1 if self._similarity == "cosine" else 0

        assert self._sim_fbo is not None
        self._sim_fbo.use()
        self._ctx.viewport = (0, 0, point_count, point_count)
        self._sim_vao.render(mode=self._gl.TRIANGLES, vertices=3)

        data = self._sim_tex.read()
        sim_matrix = self._np.frombuffer(data, dtype=self._np.float32).reshape(
            (point_count, point_count)
        )
        return sim_matrix.tolist()

    def upload_similarity(self, sim_base: Sequence[Sequence[float]]) -> bool:
        sim_matrix = self._np.asarray(sim_base, dtype=self._np.float32)
        if sim_matrix.shape != (self._point_count, self._point_count):
            return False

        if self._sim_tex is None:
            self._sim_tex = self._ctx.texture(
                (self._point_count, self._point_count),
                1,
                data=sim_matrix.tobytes(),
                dtype="f4",
            )
            self._sim_tex.filter = (self._gl.NEAREST, self._gl.NEAREST)
            self._sim_fbo = self._ctx.framebuffer(color_attachments=[self._sim_tex])
        else:
            self._sim_tex.write(sim_matrix.tobytes())
        return True

    def update_positions(self, pos_y: Sequence[int], pos_x: Sequence[int]) -> bool:
        if len(pos_y) != self._point_count or len(pos_x) != self._point_count:
            return False
        pos = self._np.empty((self._point_count, 2), dtype=self._np.float32)
        pos[:, 0] = pos_y
        pos[:, 1] = pos_x
        self._pos_tex.write(pos.tobytes())
        return True

    def _ensure_pair_buffers(self, pair_count: int) -> None:
        if pair_count <= self._pair_capacity:
            return
        capacity = max(pair_count, self._pair_capacity * 2 or 1)
        self._pair_capacity = capacity
        self._pair_idx_tex = self._ctx.texture((capacity, 1), 2, dtype="f4")
        self._pair_idx_tex.filter = (self._gl.NEAREST, self._gl.NEAREST)
        self._pair_pos_tex = self._ctx.texture((capacity, 1), 4, dtype="f4")
        self._pair_pos_tex.filter = (self._gl.NEAREST, self._gl.NEAREST)
        self._pair_out_tex = self._ctx.texture((capacity, 1), 4, dtype="f4")
        self._pair_out_tex.filter = (self._gl.NEAREST, self._gl.NEAREST)
        self._pair_fbo = self._ctx.framebuffer(color_attachments=[self._pair_out_tex])

    def compute_pair_energies(
        self,
        pairs: Sequence[tuple[int, int, int, int, int, int]],
        *,
        mode: str,
        local_radius: int | None,
        lambda_threshold: float,
        eta: float | None,
        distance_eps: float,
    ) -> list[tuple[float, float]] | None:
        if self._sim_tex is None:
            return None
        pair_count = len(pairs)
        if pair_count == 0:
            return []

        self._ensure_pair_buffers(pair_count)
        assert self._pair_idx_tex is not None
        assert self._pair_pos_tex is not None
        assert self._pair_out_tex is not None
        assert self._pair_fbo is not None

        idx_data = self._np.empty((pair_count, 2), dtype=self._np.float32)
        pos_data = self._np.empty((pair_count, 4), dtype=self._np.float32)
        for i, (idx_a, idx_b, y1, x1, y2, x2) in enumerate(pairs):
            idx_data[i, 0] = float(idx_a)
            idx_data[i, 1] = float(idx_b)
            pos_data[i, 0] = float(y1)
            pos_data[i, 1] = float(x1)
            pos_data[i, 2] = float(y2)
            pos_data[i, 3] = float(x2)

        self._pair_idx_tex.write(
            idx_data.tobytes(), viewport=(0, 0, pair_count, 1)
        )
        self._pair_pos_tex.write(
            pos_data.tobytes(), viewport=(0, 0, pair_count, 1)
        )

        self._sim_tex.use(location=0)
        self._pos_tex.use(location=1)
        self._pair_idx_tex.use(location=2)
        self._pair_pos_tex.use(location=3)
        self._pair_prog["sim_tex"].value = 0
        self._pair_prog["pos_tex"].value = 1
        self._pair_prog["pair_idx_tex"].value = 2
        self._pair_prog["pair_pos_tex"].value = 3
        self._pair_prog["point_count"].value = self._point_count
        self._pair_prog["mode"].value = 0 if mode == "long" else 1
        self._pair_prog["use_radius_override"].value = (
            1 if local_radius is not None else 0
        )
        self._pair_prog["radius"].value = float(local_radius or 0)
        self._pair_prog["lambda_threshold"].value = float(lambda_threshold)
        self._pair_prog["eta_enabled"].value = 0 if eta is None else 1
        self._pair_prog["eta"].value = 0.0 if eta is None else float(eta)
        self._pair_prog["distance_eps"].value = float(distance_eps)

        self._pair_fbo.use()
        self._ctx.viewport = (0, 0, pair_count, 1)
        self._pair_vao.render(mode=self._gl.TRIANGLES, vertices=3)

        data = self._pair_out_tex.read()
        raw = self._np.frombuffer(data, dtype=self._np.float32).reshape(
            (self._pair_capacity, 4)
        )
        energies = raw[:pair_count]
        return [(float(row[0]), float(row[1])) for row in energies]

    def average_local_energy(
        self,
        radius: int,
        *,
        lambda_threshold: float,
        eta: float | None,
        distance_eps: float,
    ) -> float | None:
        if self._sim_tex is None:
            return None

        if self._avg_out_tex is None:
            self._avg_out_tex = self._ctx.texture(
                (self._point_count, 1), 1, dtype="f4"
            )
            self._avg_out_tex.filter = (self._gl.NEAREST, self._gl.NEAREST)
            self._avg_fbo = self._ctx.framebuffer(color_attachments=[self._avg_out_tex])

        self._sim_tex.use(location=0)
        self._pos_tex.use(location=1)
        self._avg_prog["sim_tex"].value = 0
        self._avg_prog["pos_tex"].value = 1
        self._avg_prog["point_count"].value = self._point_count
        self._avg_prog["radius"].value = float(radius)
        self._avg_prog["lambda_threshold"].value = float(lambda_threshold)
        self._avg_prog["eta_enabled"].value = 0 if eta is None else 1
        self._avg_prog["eta"].value = 0.0 if eta is None else float(eta)
        self._avg_prog["distance_eps"].value = float(distance_eps)

        assert self._avg_fbo is not None
        self._avg_fbo.use()
        self._ctx.viewport = (0, 0, self._point_count, 1)
        self._avg_vao.render(mode=self._gl.TRIANGLES, vertices=3)

        data = self._avg_out_tex.read()
        energies = self._np.frombuffer(data, dtype=self._np.float32)
        if energies.size == 0:
            return 0.0
        emax = float(energies.max())
        if emax <= 0.0:
            return 0.0
        return float((energies / emax).mean())


@dataclass(frozen=True)
class _TensorEnergyBatch:
    energies: list[tuple[float, float]]
    contrib_points: list[int]
    s1_min: list[float]
    s1_max: list[float]
    s2_min: list[float]
    s2_max: list[float]
    s1_sum: list[float]
    s2_sum: list[float]
    d1_min: list[float]
    d1_max: list[float]
    d2_min: list[float]
    d2_max: list[float]


@dataclass(frozen=True)
class _TensorShortEnergyBatch:
    energies: list[tuple[float, float]]
    contrib_points: list[int]
    subset_size: int


class _TensorLayoutEngine:
    def __init__(
        self,
        *,
        torch_mod: "Any",
        device: "Any",
        code_matrix: "numpy.ndarray",
        ones: "numpy.ndarray",
        similarity: str,
    ) -> None:
        self._torch = torch_mod
        self._device = device
        self._similarity = similarity
        self._codes = self._torch.as_tensor(code_matrix, dtype=self._torch.float32, device=device)
        self._ones = self._torch.as_tensor(ones, dtype=self._torch.float32, device=device)
        self._point_count = code_matrix.shape[0]
        self._subset_indices: tuple[int, ...] | None = None
        self._subset_tensor: "Any" | None = None
        self._zero_row: "Any" | None = None
        self._sim_cache: dict[int, "Any"] = {}
        self._lambda: float | None = None
        self._eta: float | None = None

    @classmethod
    def create(
        cls,
        *,
        code_matrix: "numpy.ndarray | None",
        ones: "numpy.ndarray | None",
        similarity: str,
    ) -> "_TensorLayoutEngine | None":
        try:
            import torch
        except Exception as exc:
            LOGGER.event(
                "layout.gpu.tensor.unavailable",
                section=GPU_IMPLEMENTATION,
                data={"error": str(exc)},
            )
            return None
        if code_matrix is None or ones is None:
            LOGGER.event(
                "layout.gpu.tensor.unavailable",
                section=GPU_IMPLEMENTATION,
                data={"error": "code matrix missing"},
            )
            return None
        device: "torch.device | None" = None
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        if device is None:
            LOGGER.event(
                "layout.gpu.tensor.unavailable",
                section=GPU_IMPLEMENTATION,
                data={"error": "no gpu backend"},
            )
            return None
        try:
            engine = cls(
                torch_mod=torch,
                device=device,
                code_matrix=code_matrix,
                ones=ones,
                similarity=similarity,
            )
        except Exception as exc:
            LOGGER.event(
                "layout.gpu.tensor.unavailable",
                section=GPU_IMPLEMENTATION,
                data={"error": str(exc)},
            )
            return None
        LOGGER.event(
            "layout.gpu.tensor.enabled",
            section=GPU_IMPLEMENTATION,
            data={"device": str(device), "similarity": similarity},
        )
        return engine

    def reset_subset(self) -> None:
        self._subset_indices = None
        self._subset_tensor = None
        self._zero_row = None
        self._sim_cache.clear()

    @property
    def device(self) -> str:
        return str(self._device)

    @property
    def point_count(self) -> int:
        return self._point_count

    def _apply_lambda_tensor(
        self, sim: "Any", lambda_threshold: float, eta: float | None
    ) -> "Any":
        if sim.numel() == 0:
            return sim
        if eta is None:
            if lambda_threshold <= 0.0:
                return self._torch.where(sim > 0.0, sim, self._torch.zeros_like(sim))
            return self._torch.where(sim >= lambda_threshold, sim, self._torch.zeros_like(sim))
        scaled = 1.0 / (1.0 + self._torch.exp(-eta * (sim - lambda_threshold)))
        return sim * scaled

    def ensure_subset(
        self, subset_indices: Sequence[int], lambda_threshold: float, eta: float | None
    ) -> bool:
        subset_key = tuple(subset_indices)
        if not subset_key:
            return False
        if (
            self._subset_indices != subset_key
            or self._lambda != lambda_threshold
            or self._eta != eta
        ):
            self._subset_indices = subset_key
            self._subset_tensor = self._torch.as_tensor(
                subset_indices, dtype=self._torch.long, device=self._device
            )
            self._zero_row = self._torch.zeros(
                (len(subset_indices),), dtype=self._torch.float32, device=self._device
            )
            self._sim_cache.clear()
            self._lambda = lambda_threshold
            self._eta = eta
        return True

    def _similarity_block_tensor(
        self, row_indices: Sequence[int]
    ) -> "Any":
        if self._subset_tensor is None:
            raise RuntimeError("subset is not prepared")
        rows = self._codes[self._torch.as_tensor(row_indices, device=self._device)]
        cols = self._codes[self._subset_tensor].T
        common = rows @ cols
        ones_rows = self._ones[self._torch.as_tensor(row_indices, device=self._device)]
        ones_cols = self._ones[self._subset_tensor]
        if self._similarity == "cosine":
            denom = self._torch.sqrt(self._torch.outer(ones_rows, ones_cols))
            safe = self._torch.where(denom != 0, denom, self._torch.ones_like(denom))
            sim = self._torch.where(denom != 0, common / safe, self._torch.zeros_like(common))
        else:
            union = ones_rows[:, None] + ones_cols[None, :] - common
            safe = self._torch.where(union != 0, union, self._torch.ones_like(union))
            sim = self._torch.where(union != 0, common / safe, self._torch.zeros_like(common))
        sim = self._apply_lambda_tensor(sim, self._lambda or 0.0, self._eta)
        return sim

    def similarity_block(
        self,
        row_indices: Sequence[int],
        subset_indices: Sequence[int],
        lambda_threshold: float,
        eta: float | None,
    ) -> "numpy.ndarray | None":
        if not self.ensure_subset(subset_indices, lambda_threshold, eta):
            return None
        with self._torch.no_grad():
            sim = self._similarity_block_tensor(row_indices)
            for idx, row in zip(row_indices, sim):
                self._sim_cache[idx] = row
            return sim.cpu().numpy().astype("float32", copy=False)

    def _get_sim_row(
        self, idx: int | None, sim_lookup: Mapping[int, "numpy.ndarray"]
    ) -> "Any":
        if idx is None:
            assert self._zero_row is not None
            return self._zero_row
        cached = self._sim_cache.get(idx)
        if cached is not None:
            return cached
        values = sim_lookup.get(idx)
        if values is None:
            assert self._zero_row is not None
            return self._zero_row
        row = self._torch.as_tensor(values, dtype=self._torch.float32, device=self._device)
        self._sim_cache[idx] = row
        return row

    def register_cache_row(self, idx: int, values: Sequence[float]) -> None:
        if self._subset_tensor is None:
            return
        row = self._torch.as_tensor(values, dtype=self._torch.float32, device=self._device)
        self._sim_cache[idx] = row

    def pair_energies(
        self,
        pairs: Sequence[tuple[int, int, int, int, int | None, int | None]],
        eval_indices: Sequence[int],
        pos_y: Sequence[int],
        pos_x: Sequence[int],
        sim_lookup: Mapping[int, "numpy.ndarray"],
        distance_eps: float,
    ) -> _TensorEnergyBatch | None:
        if self._subset_tensor is None or not pairs:
            return None
        if not self.ensure_subset(eval_indices, self._lambda or 0.0, self._eta):
            return None
        _ = distance_eps
        torch = self._torch
        device = self._device
        subset = self._subset_tensor
        zero_row = self._zero_row
        if zero_row is None:
            return None

        with torch.no_grad():
            eval_y = torch.as_tensor(
                [pos_y[idx] for idx in eval_indices],
                dtype=torch.float32,
                device=device,
            )
            eval_x = torch.as_tensor(
                [pos_x[idx] for idx in eval_indices],
                dtype=torch.float32,
                device=device,
            )

            idx_a_tensor = torch.as_tensor(
                [idx_a if idx_a is not None else -1 for _, _, _, _, idx_a, _ in pairs],
                dtype=torch.long,
                device=device,
            )
            idx_b_tensor = torch.as_tensor(
                [idx_b if idx_b is not None else -1 for _, _, _, _, _, idx_b in pairs],
                dtype=torch.long,
                device=device,
            )

            s1_rows = [
                self._get_sim_row(idx_a, sim_lookup)
                for (_, _, _, _, idx_a, _) in pairs
            ]
            s2_rows = [
                self._get_sim_row(idx_b, sim_lookup)
                for (_, _, _, _, _, idx_b) in pairs
            ]
            s1 = torch.stack(s1_rows, dim=0)
            s2 = torch.stack(s2_rows, dim=0)

            mask_a = subset[None, :] == idx_a_tensor[:, None]
            mask_b = subset[None, :] == idx_b_tensor[:, None]
            mask_self = mask_a | mask_b
            s1 = torch.where(mask_self, torch.zeros_like(s1), s1)
            s2 = torch.where(mask_self, torch.zeros_like(s2), s2)

            pair_y1 = torch.as_tensor(
                [y1 for (y1, _, _, _, _, _) in pairs],
                dtype=torch.float32,
                device=device,
            )
            pair_x1 = torch.as_tensor(
                [x1 for (_, x1, _, _, _, _) in pairs],
                dtype=torch.float32,
                device=device,
            )
            pair_y2 = torch.as_tensor(
                [y2 for (_, _, y2, _, _, _) in pairs],
                dtype=torch.float32,
                device=device,
            )
            pair_x2 = torch.as_tensor(
                [x2 for (_, _, _, x2, _, _) in pairs],
                dtype=torch.float32,
                device=device,
            )

            dy1 = pair_y1[:, None] - eval_y[None, :]
            dx1 = pair_x1[:, None] - eval_x[None, :]
            dy2 = pair_y2[:, None] - eval_y[None, :]
            dx2 = pair_x2[:, None] - eval_x[None, :]
            dist1 = dy1 * dy1 + dx1 * dx1
            dist2 = dy2 * dy2 + dx2 * dx2

            energy_c = torch.sum(s1 * dist1 + s2 * dist2, dim=1)
            energy_s = torch.sum(s2 * dist1 + s1 * dist2, dim=1)

            contrib_mask = (s1 > 0.0) | (s2 > 0.0)
            contrib_counts = torch.sum(contrib_mask, dim=1)

            def _masked_min(values: "Any") -> "Any":
                inf = torch.finfo(values.dtype).max
                masked = torch.where(contrib_mask, values, torch.full_like(values, inf))
                mins = torch.min(masked, dim=1).values
                return torch.where(contrib_counts > 0, mins, torch.zeros_like(mins))

            def _masked_max(values: "Any") -> "Any":
                ninf = torch.tensor(-torch.finfo(values.dtype).max, device=device, dtype=values.dtype)
                masked = torch.where(contrib_mask, values, ninf)
                maxs = torch.max(masked, dim=1).values
                return torch.where(contrib_counts > 0, maxs, torch.zeros_like(maxs))

            s1_sum = torch.sum(torch.where(contrib_mask, s1, torch.zeros_like(s1)), dim=1)
            s2_sum = torch.sum(torch.where(contrib_mask, s2, torch.zeros_like(s2)), dim=1)
            s1_min = _masked_min(s1)
            s1_max = _masked_max(s1)
            s2_min = _masked_min(s2)
            s2_max = _masked_max(s2)
            d1_min = _masked_min(dist1)
            d1_max = _masked_max(dist1)
            d2_min = _masked_min(dist2)
            d2_max = _masked_max(dist2)

            energies = list(
                zip(
                    energy_c.detach().cpu().tolist(),
                    energy_s.detach().cpu().tolist(),
                )
            )
            return _TensorEnergyBatch(
                energies=energies,
                contrib_points=[int(v) for v in contrib_counts.detach().cpu().tolist()],
                s1_min=[float(v) for v in s1_min.detach().cpu().tolist()],
                s1_max=[float(v) for v in s1_max.detach().cpu().tolist()],
                s2_min=[float(v) for v in s2_min.detach().cpu().tolist()],
                s2_max=[float(v) for v in s2_max.detach().cpu().tolist()],
                s1_sum=[float(v) for v in s1_sum.detach().cpu().tolist()],
                s2_sum=[float(v) for v in s2_sum.detach().cpu().tolist()],
                d1_min=[float(v) for v in d1_min.detach().cpu().tolist()],
                d1_max=[float(v) for v in d1_max.detach().cpu().tolist()],
                d2_min=[float(v) for v in d2_min.detach().cpu().tolist()],
                d2_max=[float(v) for v in d2_max.detach().cpu().tolist()],
            )

    def short_pair_energies(
        self,
        pairs: Sequence[tuple[int, int, int, int, int | None, int | None, float]],
        pos_y: Sequence[int],
        pos_x: Sequence[int],
        sim_lookup: Mapping[int, "numpy.ndarray"],
        distance_eps: float,
    ) -> _TensorShortEnergyBatch | None:
        if self._subset_tensor is None:
            return None
        pair_count = len(pairs)
        if pair_count == 0:
            return _TensorShortEnergyBatch(energies=[], contrib_points=[], subset_size=len(self._subset_tensor))
        subset = self._subset_tensor
        if subset.numel() == 0:
            energies = [(0.0, 0.0) for _ in pairs]
            return _TensorShortEnergyBatch(
                energies=energies,
                contrib_points=[0 for _ in pairs],
                subset_size=0,
            )
        torch = self._torch
        device = self._device
        with torch.no_grad():
            subset_indices = subset.tolist()
            subset_y = torch.as_tensor(
                [pos_y[idx] for idx in subset_indices],
                dtype=torch.float32,
                device=device,
            )
            subset_x = torch.as_tensor(
                [pos_x[idx] for idx in subset_indices],
                dtype=torch.float32,
                device=device,
            )
            idx_a_tensor = torch.as_tensor(
                [idx_a if idx_a is not None else -1 for (_, _, _, _, idx_a, _, _) in pairs],
                dtype=torch.long,
                device=device,
            )
            idx_b_tensor = torch.as_tensor(
                [idx_b if idx_b is not None else -1 for (_, _, _, _, _, idx_b, _) in pairs],
                dtype=torch.long,
                device=device,
            )
            center_y = torch.as_tensor(
                [((y1 + y2) / 2.0) for (y1, _, y2, _, _, _, _) in pairs],
                dtype=torch.float32,
                device=device,
            )
            center_x = torch.as_tensor(
                [((x1 + x2) / 2.0) for (_, x1, _, x2, _, _, _) in pairs],
                dtype=torch.float32,
                device=device,
            )
            radius = torch.as_tensor(
                [radius for (_, _, _, _, _, _, radius) in pairs],
                dtype=torch.float32,
                device=device,
            )
            pair_y1 = torch.as_tensor(
                [y1 for (y1, _, _, _, _, _, _) in pairs],
                dtype=torch.float32,
                device=device,
            )
            pair_x1 = torch.as_tensor(
                [x1 for (_, x1, _, _, _, _, _) in pairs],
                dtype=torch.float32,
                device=device,
            )
            pair_y2 = torch.as_tensor(
                [y2 for (_, _, y2, _, _, _, _) in pairs],
                dtype=torch.float32,
                device=device,
            )
            pair_x2 = torch.as_tensor(
                [x2 for (_, _, _, x2, _, _, _) in pairs],
                dtype=torch.float32,
                device=device,
            )

            s1_rows = [
                self._get_sim_row(idx_a, sim_lookup)
                for (_, _, _, _, idx_a, _, _) in pairs
            ]
            s2_rows = [
                self._get_sim_row(idx_b, sim_lookup)
                for (_, _, _, _, _, idx_b, _) in pairs
            ]
            s1 = torch.stack(s1_rows, dim=0)
            s2 = torch.stack(s2_rows, dim=0)

            mask_a = subset[None, :] == idx_a_tensor[:, None]
            mask_b = subset[None, :] == idx_b_tensor[:, None]
            mask_self = mask_a | mask_b
            s1 = torch.where(mask_self, torch.zeros_like(s1), s1)
            s2 = torch.where(mask_self, torch.zeros_like(s2), s2)

            dyc = center_y[:, None] - subset_y[None, :]
            dxc = center_x[:, None] - subset_x[None, :]
            radius_sq = radius[:, None] * radius[:, None]
            mask_radius = (dyc * dyc + dxc * dxc) <= radius_sq

            dist1 = (pair_y1[:, None] - subset_y[None, :]) ** 2 + (pair_x1[:, None] - subset_x[None, :]) ** 2
            dist2 = (pair_y2[:, None] - subset_y[None, :]) ** 2 + (pair_x2[:, None] - subset_x[None, :]) ** 2
            denom1 = dist1 + distance_eps
            denom2 = dist2 + distance_eps

            s1_masked = torch.where(mask_radius, s1, torch.zeros_like(s1))
            s2_masked = torch.where(mask_radius, s2, torch.zeros_like(s2))

            energy_c = torch.sum(s1_masked / denom1 + s2_masked / denom2, dim=1)
            energy_s = torch.sum(s2_masked / denom1 + s1_masked / denom2, dim=1)
            contrib_mask = mask_radius & ((s1_masked > 0.0) | (s2_masked > 0.0))
            contrib_counts = torch.sum(contrib_mask, dim=1)

            energies = list(
                zip(
                    energy_c.detach().cpu().tolist(),
                    energy_s.detach().cpu().tolist(),
                )
            )
            return _TensorShortEnergyBatch(
                energies=energies,
                contrib_points=[int(v) for v in contrib_counts.detach().cpu().tolist()],
                subset_size=len(subset_indices),
            )


class _AdaptiveLayoutState:
    def __init__(self, config: AdaptiveLayoutConfig, total_steps: int) -> None:
        self._config = config
        self._total_steps = max(1, total_steps)
        self._initial_swaps: int | None = None
        self._radius_cap = max(config.start_radius, config.end_radius)
        self._trigger_count = 0

    def _base_radius_for_step(self, step: int) -> int:
        if self._total_steps <= 1:
            return self._config.end_radius
        progress = step / max(1, self._total_steps - 1)
        decayed = self._config.start_radius - (
            (self._config.start_radius - self._config.end_radius) * progress
        )
        return max(self._config.end_radius, int(round(decayed)))

    def radius_for_step(self, step: int) -> int:
        radius = self._base_radius_for_step(step)
        return min(radius, self._radius_cap)

    def update_after_step(
        self, *, step: int, swaps: int, current_lambda: float
    ) -> tuple[float | None, float]:
        if self._initial_swaps is None:
            self._initial_swaps = max(1, swaps)
        swap_ratio = 0.0 if self._initial_swaps == 0 else swaps / self._initial_swaps
        new_lambda: float | None = None
        if swap_ratio <= self._config.swap_ratio_trigger:
            self._trigger_count += 1
            self._initial_swaps = max(1, swaps)
            base_radius = self._base_radius_for_step(step)
            reduced_cap = max(self._config.end_radius, int(base_radius * 0.5))
            if reduced_cap < self._radius_cap:
                self._radius_cap = reduced_cap
            if self._config.lambda_step is not None:
                new_lambda = min(1.0, current_lambda + self._config.lambda_step)
        return new_lambda, swap_ratio


@dataclass
class _EnergyStabilityMonitor:
    window: int
    threshold: float
    history: deque[float] = field(init=False)
    last_energy: float | None = None

    def __post_init__(self) -> None:
        self.history = deque(maxlen=self.window)

    def update(self, energy: float) -> tuple[bool, float | None, float]:
        diff = None
        if self.last_energy is not None:
            diff = abs(energy - self.last_energy)
            self.history.append(diff)
        self.last_energy = energy
        if len(self.history) < self.history.maxlen:
            max_delta = max(self.history) if self.history else 0.0
            return False, diff, max_delta
        max_delta = max(self.history) if self.history else 0.0
        return max_delta < self.threshold, diff, max_delta


@dataclass
class _SwapStallMonitor:
    min_ratio: float
    window: int
    history: deque[float] = field(init=False)

    def __post_init__(self) -> None:
        self.history = deque(maxlen=self.window)

    def update(self, swaps: int, pairs_per_step: int) -> tuple[bool, float, float]:
        if pairs_per_step <= 0:
            ratio = 0.0
        else:
            ratio = swaps / pairs_per_step
        self.history.append(ratio)
        average = sum(self.history) / len(self.history)
        if len(self.history) < self.history.maxlen:
            return False, ratio, average
        return average <= self.min_ratio, ratio, average


class Layout:
    """Discrete layout of code vectors on a 2D grid using the DAMP algorithm."""

    def __init__(
        self,
        codes: Mapping[float, Sequence[BitArray]],
        *,
        grid_size: int | None = None,
        empty_ratio: float = 0.15,
        similarity: str = "jaccard",
        lambda_threshold: float = 0.65,
        eta: float | None = 10.0,
        precompute_similarity: bool = True,
        max_precompute: int = 2000,
        seed: int = 0,
        parallel_workers: int | None = None,
        use_gpu: bool = True,
        long_subset_size: int | None = 512,
        long_subset_refresh: int = 50,
    ) -> None:
        if not codes:
            raise ValueError("codes must be non-empty")
        if empty_ratio < 0:
            raise ValueError("empty_ratio must be >= 0")
        if similarity not in ("cosine", "jaccard"):
            raise ValueError("similarity must be 'cosine' or 'jaccard'")
        if not 0 <= lambda_threshold <= 1:
            raise ValueError("lambda_threshold must be in [0, 1]")
        if max_precompute <= 0:
            raise ValueError("max_precompute must be positive")
        if parallel_workers is not None and parallel_workers < 0:
            raise ValueError("parallel_workers must be >= 0")

        self._points = []
        self._codes: dict[float, list[BitArray]] = {}
        for label, label_codes in codes.items():
            label_value = float(label)
            bucket = self._codes.setdefault(label_value, [])
            for code in label_codes:
                layout_code, ones = self._coerce_code(code)
                bucket.append(layout_code)
                self._points.append(
                    LayoutPoint(
                        code=layout_code,
                        label=label_value,
                        hue=label_value,
                        ones=ones,
                    )
                )
        self._labels = [self._format_label(point.label) for point in self._points]
        self._values = [point.code.to01() for point in self._points]
        self._point_count = len(self._points)
        self._similarity = similarity
        self._lambda = lambda_threshold
        self._eta = eta
        self._rng = random.Random(seed)
        self._parallel_workers = parallel_workers
        self._use_gpu = use_gpu
        self._long_subset_size = (
            None if long_subset_size is None else int(long_subset_size)
        )
        if self._long_subset_size is not None and self._long_subset_size <= 0:
            raise ValueError("long_subset_size must be positive when set")
        self._long_subset_refresh = max(1, int(long_subset_refresh))
        self._long_subset_indices: list[int] = []
        self._long_subset_last_step: int = 0
        self._long_step_counter: int = 0
        self._subset_similarity_cache: dict[int, "numpy.ndarray"] = {}
        self._short_subset_indices: tuple[int, ...] | None = None
        self._short_similarity_cache: dict[int, "numpy.ndarray"] = {}
        self._code_matrix, self._ones_array = self._build_code_matrix()

        if grid_size is None:
            target = len(self._points) * (1.0 + empty_ratio)
            grid_size = math.ceil(math.sqrt(target))
        if grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if grid_size * grid_size < len(self._points):
            raise ValueError("grid_size too small for number of points")

        self.height = grid_size
        self.width = grid_size
        LOGGER.event(
            "layout.init",
            section=LAYOUT_ALGORITHM,
            data={
                "points": self._point_count,
                "labels": len(self._codes),
                "seed": seed,
            },
        )
        LOGGER.event(
            "layout.grid",
            section=LAYOUT_COMPACTNESS,
            data={
                "grid_size": grid_size,
                "empty_ratio": empty_ratio,
            },
        )
        LOGGER.event(
            "layout.similarity",
            section=SIMILARITY_MEASURES,
            data={"similarity": self._similarity},
        )
        LOGGER.event(
            "layout.thresholds",
            section=ENERGIES,
            data={"lambda_threshold": self._lambda, "eta": self._eta},
        )
        LOGGER.event(
            "layout.precompute",
            section=OPTIM_SIM_MATRIX,
            data={"precompute_similarity": precompute_similarity},
        )
        LOGGER.event(
            "layout.precompute_limit",
            section=OPTIM_SUBSET,
            data={"max_precompute": max_precompute},
        )
        LOGGER.event(
            "layout.energy.subset.config",
            section=OPTIM_SUBSET,
            data={
                "long_subset_size": -1 if self._long_subset_size is None else self._long_subset_size,
                "long_subset_refresh": self._long_subset_refresh,
                "points": self._point_count,
            },
        )
        LOGGER.event(
            "layout.parallel",
            section=PARALLEL_PROCESSING,
            data={"parallel_workers": parallel_workers},
        )
        LOGGER.event(
            "layout.gpu.config",
            section=GPU_IMPLEMENTATION,
            data={"use_gpu": use_gpu},
        )
        if self._points:
            ones_values = [point.ones for point in self._points]
            LOGGER.event(
                "layout.codes",
                section=SPARSE_BIT_VECTORS,
                data={
                    "code_length": len(self._points[0].code),
                    "ones_min": min(ones_values),
                    "ones_max": max(ones_values),
                    "ones_avg": sum(ones_values) / len(ones_values),
                },
            )
        else:
            LOGGER.event(
                "layout.codes",
                section=SPARSE_BIT_VECTORS,
                data={
                    "code_length": 0,
                    "ones_min": 0,
                    "ones_max": 0,
                    "ones_avg": 0,
                },
            )
        self.grid: list[list[int | None]] = [
            [None for _ in range(self.width)] for _ in range(self.height)
        ]
        self._positions: list[tuple[int, int]] = [(-1, -1)] * len(self._points)
        self._pos_y = [0] * len(self._points)
        self._pos_x = [0] * len(self._points)
        self._row_occupied: list[set[int]] = [set() for _ in range(self.height)]
        self._radius_offsets_cache: dict[int, list[tuple[int, int]]] = {}
        self._short_energy_weights: list[float] | None = None
        self._short_energy_last_step: int = -1
        self._short_energy_radius: int | None = None
        self._short_energy_eps: float = 1e-6
        self._place_points()

        self._long_swap_radius_sum = 0.0
        self._long_swap_count = 0
        self._pair_radius: int | None = None
        self._distance_eps = 1e-6
        LOGGER.event(
            "layout.distance_eps",
            section=ENERGIES,
            data={"distance_eps": self._distance_eps},
        )
        self._last_visual_energy: float | None = None
        self._last_visual_energy_step: int | None = None
        self._last_energy_value: float | None = None
        self._last_energy_step: int | None = None
        self._gpu_engine: _GpuLayoutEngine | None = None
        self._tensor_engine: _TensorLayoutEngine | None = None
        self._gpu_positions_dirty = True
        if self._use_gpu:
            self._gpu_engine = _GpuLayoutEngine.create(self._points, self._similarity)
            self._tensor_engine = _TensorLayoutEngine.create(
                code_matrix=self._code_matrix,
                ones=self._ones_array,
                similarity=self._similarity,
            )
        if self._use_gpu and (self._gpu_engine is not None or self._tensor_engine is not None):
            status = "OpenGL 4.1" if self._gpu_engine is not None else f"tensor:{self._tensor_engine.device}"
            LOGGER.event(
                "layout.gpu.enabled",
                section=GPU_IMPLEMENTATION,
                data={"status": status},
            )
        else:
            LOGGER.event(
                "layout.gpu.disabled",
                section=GPU_IMPLEMENTATION,
                data={"status": "cpu"},
            )
        self.last_steps = 0
        self._sim_base: list[list[float]] | None = None
        if precompute_similarity and self._point_count <= max_precompute:
            self._build_similarity_cache()

    def _long_average_swap_radius(self) -> float | None:
        if self._long_swap_count == 0:
            return None
        return self._long_swap_radius_sum / self._long_swap_count

    def _resolve_pair_radius(self, pair_radius: int | None, mode: str) -> int | None:
        resolved_radius = pair_radius
        if mode == "short" and pair_radius is None:
            average_radius = self._long_average_swap_radius()
            if average_radius is None:
                LOGGER.event(
                    "layout.pair_radius.short_unset",
                    section=PAIR_SELECTION,
                    data={
                        "pair_radius": "auto",
                        "long_swaps": self._long_swap_count,
                    },
                )
                self.pair_radius = None
                return None
            resolved_radius = max(1, int(round(average_radius)))
            LOGGER.event(
                "layout.pair_radius.short_resolved",
                section=PAIR_SELECTION,
                data={
                    "pair_radius": resolved_radius,
                    "long_swap_average_radius": average_radius,
                    "long_swaps": self._long_swap_count,
                },
            )
        self.pair_radius = resolved_radius
        return resolved_radius

    def run(
        self,
        *,
        steps: int,
        pairs_per_step: int,
        pair_radius: int | None,
        mode: str = "long",
        local_radius: int | None = None,
        min_swap_ratio: float = 0.01,
        min_swap_window: int | None = None,
        log_every: int | None = None,
        log_path: str = "layout",
        log_visuals: bool = True,
        step_offset: int = 0,
        energy_radius: int | None = None,
        energy_stability_window: int | None = None,
        energy_stability_delta: float = 1e-3,
        energy_stability_every: int = 1,
        energy_stability_max_points: int | None = None,
        adaptive_params: AdaptiveLayoutConfig | None = None,
        short_optim: ShortLayoutOptimConfig | None = None,
    ) -> int:
        if steps <= 0:
            raise ValueError("steps must be positive")
        if pairs_per_step <= 0:
            raise ValueError("pairs_per_step must be positive")
        if pair_radius is not None and pair_radius <= 0:
            raise ValueError("pair_radius must be positive when set")
        if mode not in ("long", "short"):
            raise ValueError("mode must be 'long' or 'short'")
        if local_radius is not None and local_radius <= 0:
            raise ValueError("local_radius must be positive when set")
        if not 0 <= min_swap_ratio <= 1:
            raise ValueError("min_swap_ratio must be in [0, 1]")
        if min_swap_window is not None and min_swap_window <= 0:
            raise ValueError("min_swap_window must be positive when set")
        if step_offset < 0:
            raise ValueError("step_offset must be >= 0")
        if energy_radius is not None and energy_radius <= 0:
            raise ValueError("energy_radius must be positive when set")
        if energy_stability_window is not None and energy_stability_window <= 0:
            raise ValueError("energy_stability_window must be positive when set")
        if energy_stability_delta < 0:
            raise ValueError("energy_stability_delta must be >= 0")
        if energy_stability_every <= 0:
            raise ValueError("energy_stability_every must be positive")
        if energy_stability_max_points is not None and energy_stability_max_points <= 0:
            raise ValueError("energy_stability_max_points must be positive when set")
        if adaptive_params is not None:
            if adaptive_params.start_radius <= 0:
                raise ValueError("adaptive start_radius must be positive")
            if adaptive_params.end_radius <= 0:
                raise ValueError("adaptive end_radius must be positive")
            if adaptive_params.start_radius < adaptive_params.end_radius:
                raise ValueError("adaptive start_radius must be >= end_radius")
            if not 0 <= adaptive_params.swap_ratio_trigger <= 1:
                raise ValueError("adaptive swap_ratio_trigger must be in [0, 1]")
        if mode == "short" and short_optim is not None:
            if short_optim.energy_radius <= 0:
                raise ValueError("short energy_radius must be positive")
            if short_optim.energy_recalc_every <= 0:
                raise ValueError("short energy_recalc_every must be positive")
            if short_optim.energy_eps <= 0:
                raise ValueError("short energy_eps must be positive")
            if short_optim.energy_max_points is not None and short_optim.energy_max_points <= 0:
                raise ValueError("short energy_max_points must be positive when set")
            if short_optim.partitions <= 0:
                raise ValueError("short partitions must be positive")

        effective_energy_radius = energy_radius
        if effective_energy_radius is None:
            auto_radius = max(
                1, min(min(self.height, self.width) // 2, _AUTO_ENERGY_RADIUS_MAX)
            )
            effective_energy_radius = auto_radius
            LOGGER.event(
                "layout.energy.auto_radius",
                section=QUALITY_ASSESS,
                data={"energy_radius": effective_energy_radius},
            )

        resolved_pair_radius = self._resolve_pair_radius(pair_radius, mode)
        LOGGER.event(
            "layout.run.selection",
            section=PAIR_SELECTION,
            data={
                "pairs_per_step": pairs_per_step,
                "pair_radius": resolved_pair_radius,
                "pair_radius_requested": pair_radius,
            },
        )
        LOGGER.event(
            "layout.run.mode",
            section=ENERGY_LONG if mode == "long" else ENERGY_SHORT,
            data={
                "mode": mode,
                "local_radius": local_radius,
            },
        )
        LOGGER.event(
            "layout.run.settings",
            section=LAYOUT_ALGORITHM,
            data={
                "steps": steps,
                "min_swap_ratio": min_swap_ratio,
                "min_swap_window": min_swap_window,
                "log_every": log_every,
                "log_visuals": log_visuals,
                "step_offset": step_offset,
            },
        )
        LOGGER.event(
            "layout.run.energy_monitor",
            section=QUALITY_ASSESS,
            data={
                "energy_radius": effective_energy_radius,
            },
        )
        energy_monitor: _EnergyStabilityMonitor | None = None
        if energy_stability_window is not None:
            energy_monitor = _EnergyStabilityMonitor(
                window=energy_stability_window,
                threshold=energy_stability_delta,
            )
            LOGGER.event(
                "layout.run.energy_stability",
                section=QUALITY_ASSESS,
                data={
                    "window": energy_stability_window,
                    "delta": energy_stability_delta,
                    "every": energy_stability_every,
                    "max_points": energy_stability_max_points,
                },
            )
        swap_monitor: _SwapStallMonitor | None = None
        if min_swap_window and min_swap_ratio > 0:
            swap_monitor = _SwapStallMonitor(min_swap_ratio, min_swap_window)
            LOGGER.event(
                "layout.run.swap_monitor",
                section=OPTIM_PAIR_SELECTION,
                data={
                    "min_swap_ratio": min_swap_ratio,
                    "window": min_swap_window,
                    "pairs_per_step": pairs_per_step,
                },
            )
        adaptive_state: _AdaptiveLayoutState | None = None
        if adaptive_params is not None:
            adaptive_state = _AdaptiveLayoutState(adaptive_params, steps)
            LOGGER.event(
                "layout.adaptive.config",
                section=LAYOUT_PARAMETERS,
                data={
                    "start_radius": adaptive_params.start_radius,
                    "end_radius": adaptive_params.end_radius,
                    "swap_ratio_trigger": adaptive_params.swap_ratio_trigger,
                    "lambda_step": adaptive_params.lambda_step,
                    "steps": steps,
                },
            )
        short_config: ShortLayoutOptimConfig | None = short_optim
        if mode == "short":
            if short_config is None:
                base_energy_radius = local_radius if local_radius is not None else 3
                short_config = ShortLayoutOptimConfig(
                    energy_radius=base_energy_radius,
                    energy_max_points=None,
                    energy_recalc_every=max(1, steps // 20),
                    energy_eps=1e-6,
                    use_weighted_first_point=True,
                    similarity_cutoff=self._lambda,
                    partitions=1,
                )
            elif short_config.similarity_cutoff is None:
                short_config = ShortLayoutOptimConfig(
                    energy_radius=short_config.energy_radius,
                    energy_max_points=short_config.energy_max_points,
                    energy_recalc_every=short_config.energy_recalc_every,
                    energy_eps=short_config.energy_eps,
                    use_weighted_first_point=short_config.use_weighted_first_point,
                    similarity_cutoff=self._lambda,
                    partitions=short_config.partitions,
                )
            LOGGER.event(
                "layout.short.optim_config",
                section=OPTIM_FIRST_POINT,
                data={
                    "energy_radius": short_config.energy_radius,
                    "energy_recalc_every": short_config.energy_recalc_every,
                    "energy_max_points": -1
                    if short_config.energy_max_points is None
                    else short_config.energy_max_points,
                    "energy_eps": short_config.energy_eps,
                    "similarity_cutoff": short_config.similarity_cutoff,
                    "weighted_first": short_config.use_weighted_first_point,
                    "partitions": short_config.partitions,
                },
            )
        total_swaps = 0
        steps_executed = 0

        for step in range(steps):
            effective_pair_radius = (
                adaptive_state.radius_for_step(step)
                if adaptive_state is not None
                else resolved_pair_radius
            )
            swaps = self.step(
                pairs_per_step=pairs_per_step,
                pair_radius=effective_pair_radius,
                mode=mode,
                local_radius=local_radius,
                step_index=step,
                short_config=short_config,
            )
            total_swaps += swaps
            steps_executed = step + 1
            current_step = step + step_offset
            if swap_monitor is not None:
                should_stop_swaps, swap_ratio, avg_ratio = swap_monitor.update(
                    swaps, pairs_per_step
                )
                avg_swaps_window = avg_ratio * pairs_per_step
                log_swap_ratio = LOGGER.should_log("layout.run.swap_ratio")
                swap_visuals = (
                    LOGGER.visual_scalars(
                        f"{log_path}/swap_ratio",
                        {
                            "swap_ratio": swap_ratio,
                            "avg_ratio": avg_ratio,
                        },
                        step=current_step,
                        timeline=_LAYOUT_STEP_TIMELINE,
                    )
                    if log_swap_ratio
                    else None
                )
                LOGGER.event(
                    "layout.run.swap_ratio",
                    section=OPTIM_PAIR_SELECTION,
                    data={
                        "step": current_step,
                        "swap_ratio": swap_ratio,
                        "avg_ratio": avg_ratio,
                        "avg_swaps": avg_swaps_window,
                        "window": swap_monitor.window,
                        "min_swap_ratio": swap_monitor.min_ratio,
                    },
                    visuals=swap_visuals,
                    force=log_swap_ratio,
                )
                if should_stop_swaps:
                    LOGGER.event(
                        "layout.run.swap_stability.stop",
                        section=OPTIM_PAIR_SELECTION,
                        data={
                            "step": current_step,
                            "window": swap_monitor.window,
                            "min_swap_ratio": swap_monitor.min_ratio,
                            "avg_ratio": avg_ratio,
                            "avg_swaps_window": avg_swaps_window,
                            "swap_ratio": swap_ratio,
                            "total_swaps": total_swaps,
                        },
                    )
                    break
            if log_every is not None and step % log_every == 0:
                self._log_layout_state(
                    path=log_path, step=step + step_offset, log_visuals=log_visuals
                )
            energy_for_stability: float | None = None
            should_eval_energy = (
                energy_monitor is not None
                and step % energy_stability_every == 0
            )
            if effective_energy_radius is not None and should_eval_energy:
                energy = self.average_local_energy(
                    effective_energy_radius,
                    max_points=energy_stability_max_points,
                )
                energy_for_stability = energy
                self._last_energy_value = energy
                self._last_energy_step = step + step_offset
            if adaptive_state is not None:
                new_lambda, swap_ratio = adaptive_state.update_after_step(
                    step=step, swaps=swaps, current_lambda=self._lambda
                )
                LOGGER.event(
                    "layout.adaptive.radius",
                    section=LAYOUT_PARAMETERS,
                    data={
                        "step": step + step_offset,
                        "pair_radius": effective_pair_radius,
                        "swap_ratio": swap_ratio,
                        "lambda_threshold": self._lambda,
                    },
                )
                if new_lambda is not None and new_lambda != self._lambda:
                    self._lambda = new_lambda
                    LOGGER.event(
                        "layout.adaptive.lambda",
                        section=LAYOUT_PARAMETERS,
                        data={
                            "step": step + step_offset,
                            "lambda_threshold": self._lambda,
                            "swap_ratio": swap_ratio,
                        },
                    )
            if energy_monitor is not None and energy_for_stability is not None:
                should_stop, energy_diff, max_delta = energy_monitor.update(
                    energy_for_stability
                )
                if should_stop:
                    LOGGER.event(
                        "layout.run.energy_stability.stop",
                        section=QUALITY_ASSESS,
                        data={
                            "step": step + step_offset,
                            "window": energy_monitor.window,
                            "delta": energy_monitor.threshold,
                            "every": energy_stability_every,
                            "max_delta": max_delta,
                            "last_energy_diff": energy_diff,
                            "total_swaps": total_swaps,
                        },
                    )
                    break

        self.last_steps = steps_executed
        avg_swaps = (total_swaps / steps_executed) if steps_executed else 0.0
        LOGGER.event(
            "layout.run.done",
            section=LAYOUT_ALGORITHM,
            data={
                "steps": steps_executed,
                "total_swaps": total_swaps,
                "avg_swaps": avg_swaps,
            },
        )
        return total_swaps

    def _log_layout_state(self, path: str, step: int, *, log_visuals: bool) -> None:
        if not LOGGER.should_log("layout.visual"):
            return
        visuals_skipped = self._point_count > _VISUAL_POINT_LIMIT or not log_visuals
        visuals: list = []
        if log_visuals and not visuals_skipped:
            positions = [(x + 0.5, y + 0.5) for y, x in self._positions]
            colors = self.colors_rgb()
            image = self._build_image(colors)
            visuals = [
                LOGGER.visual_image(
                    f"{path}/image",
                    image,
                ),
                LOGGER.visual_points2d(
                    f"{path}/image",
                    positions,
                    colors=colors,
                    radii=0.5,
                ),
            ]
        energy_radius: int | None = None
        energy_value: float | None = None
        energy_source: str | None = None
        if self._last_energy_value is not None:
            energy_value = self._last_energy_value
            energy_radius = None
            energy_source = "cached"
        elif log_visuals and self._point_count <= _VISUAL_ENERGY_POINT_LIMIT:
            energy_radius = max(1, min(self.height, self.width) // 2)
            energy_value = self.average_local_energy(energy_radius)
            energy_source = "visual"
        elif not log_visuals:
            energy_source = "disabled"
        energy_step = self._last_energy_step if energy_source == "cached" else step
        energy_diff = None
        if (
            energy_value is not None
            and self._last_visual_energy is not None
            and energy_step != self._last_visual_energy_step
        ):
            energy_diff = energy_value - self._last_visual_energy
        if energy_value is not None:
            self._last_visual_energy = energy_value
            self._last_visual_energy_step = energy_step
        LOGGER.event(
            "layout.visual",
            section=LAYOUT_ALGORITHM,
            data={
                "step": step,
                "points": self._point_count,
                "grid_size": self.width,
                "energy": energy_value,
                "energy_diff": energy_diff,
                "energy_radius": energy_radius,
                "energy_source": energy_source,
                "visuals_enabled": log_visuals,
                "visuals_skipped": visuals_skipped,
            },
            visuals=visuals,
            force=True,
        )

    def set_similarity_params(
        self,
        *,
        lambda_threshold: float | object = _UNSET,
        eta: float | None | object = _UNSET,
    ) -> None:
        if lambda_threshold is not _UNSET:
            if not 0 <= lambda_threshold <= 1:
                raise ValueError("lambda_threshold must be in [0, 1]")
            self._lambda = lambda_threshold
        if eta is not _UNSET:
            self._eta = eta
        LOGGER.event(
            "layout.similarity_params",
            section=ENERGIES,
            data={"lambda_threshold": self._lambda, "eta": self._eta},
        )

    @property
    def pair_radius(self) -> int | None:
        return self._pair_radius

    @pair_radius.setter
    def pair_radius(self, value: int | None) -> None:
        if value is not None and value <= 0:
            raise ValueError("pair_radius must be positive when set")
        self._pair_radius = value
        LOGGER.event(
            "layout.pair_radius.update",
            section=PAIR_SELECTION,
            data={"pair_radius": "auto" if value is None else value},
        )

    def step(
        self,
        *,
        pairs_per_step: int,
        pair_radius: int | None,
        mode: str,
        local_radius: int | None,
        step_index: int | None = None,
        short_config: ShortLayoutOptimConfig | None = None,
    ) -> int:
        swaps_with_delta: list[tuple[tuple[int, int, int, int], float]] = []
        if mode == "short":
            similarity_cutoff = (
                self._lambda
                if short_config is None or short_config.similarity_cutoff is None
                else short_config.similarity_cutoff
            )
            weighted_first = bool(short_config.use_weighted_first_point) if short_config else False
            if short_config is not None:
                self._ensure_short_energy_weights(
                    step_index=step_index, config=short_config
                )
            partitions = 1 if short_config is None else max(1, short_config.partitions)
            if partitions > 1:
                pairs = self._sample_pairs_short_partitioned(
                    pairs_per_step,
                    pair_radius=pair_radius,
                    similarity_cutoff=similarity_cutoff,
                    weighted_first=weighted_first,
                    step_index=step_index,
                    partitions=partitions,
                )
            else:
                pairs = self._sample_pairs_short(
                    pairs_per_step,
                    pair_radius=pair_radius,
                    similarity_cutoff=similarity_cutoff,
                    weighted_first=weighted_first,
                    step_index=step_index,
                )
        else:
            pairs = [self._sample_pair(pair_radius) for _ in range(pairs_per_step)]

        if mode == "short":
            tensor_swaps = self._evaluate_pairs_short_tensor(
                pairs,
                local_radius=local_radius,
            )
            if tensor_swaps is not None:
                return self._apply_tracked_swaps(tensor_swaps, mode=mode)

        if (
            self._gpu_engine is not None
            and self._ensure_gpu_similarity()
            and self._ensure_gpu_positions()
        ):
            pair_data: list[tuple[int, int, int, int, int, int]] = []
            swap_coords: list[tuple[int, int, int, int]] = []
            for y1, x1, y2, x2 in pairs:
                idx_a = self.grid[y1][x1]
                idx_b = self.grid[y2][x2]
                if idx_a is None and idx_b is None:
                    continue
                if idx_a is None:
                    idx_a, idx_b = idx_b, idx_a
                    pos_a = (y2, x2)
                    pos_b = (y1, x1)
                else:
                    pos_a = (y1, x1)
                    pos_b = (y2, x2)
                pair_data.append(
                    (
                        idx_a,
                        -1 if idx_b is None else idx_b,
                        pos_a[0],
                        pos_a[1],
                        pos_b[0],
                        pos_b[1],
                    )
                )
                swap_coords.append((y1, x1, y2, x2))

            try:
                if pair_data:
                    energies = self._gpu_engine.compute_pair_energies(
                        pair_data,
                        mode=mode,
                        local_radius=local_radius,
                        lambda_threshold=self._lambda,
                        eta=self._eta,
                        distance_eps=self._distance_eps,
                    )
                else:
                    energies = []
            except Exception:
                self._gpu_engine = None
                energies = None

            if energies is not None:
                for (y1, x1, y2, x2), (energy_c, energy_s) in zip(
                    swap_coords, energies
                ):
                    if mode == "long":
                        delta = energy_c - energy_s
                        if energy_s < energy_c:
                            swaps_with_delta.append(((y1, x1, y2, x2), delta))
                    else:
                        delta = energy_s - energy_c
                        if energy_s > energy_c:
                            swaps_with_delta.append(((y1, x1, y2, x2), delta))
                return self._apply_tracked_swaps(swaps_with_delta, mode=mode)
            self._gpu_engine = None

        if mode == "long":
            pair_data: list[tuple[int, int, int, int, int | None, int | None]] = []
            swap_coords = []
            for y1, x1, y2, x2 in pairs:
                idx_a = self.grid[y1][x1]
                idx_b = self.grid[y2][x2]
                if idx_a is None and idx_b is None:
                    continue
                pair_data.append((y1, x1, y2, x2, idx_a, idx_b))
                swap_coords.append((y1, x1, y2, x2))
            if pair_data:
                energies = self._calculate_far_energies(pair_data)
            else:
                energies = []
            for (y1, x1, y2, x2), (energy_c, energy_s) in zip(
                swap_coords, energies
            ):
                delta = energy_c - energy_s
                if energy_s < energy_c:
                    swaps_with_delta.append(((y1, x1, y2, x2), delta))
            return self._apply_tracked_swaps(swaps_with_delta, mode=mode)

        partitions = 1 if short_config is None else max(1, short_config.partitions)
        if partitions > 1:
            workers = min(partitions, os.cpu_count() or partitions)
            swaps_with_delta = self._evaluate_pairs_short_parallel(
                pairs,
                local_radius=local_radius,
                workers=workers,
                partitions=partitions,
            )
        else:
            for y1, x1, y2, x2 in pairs:
                should_swap, delta = self._should_swap(
                    y1, x1, y2, x2, mode=mode, local_radius=local_radius
                )
                if should_swap:
                    swaps_with_delta.append(((y1, x1, y2, x2), delta))
        return self._apply_tracked_swaps(swaps_with_delta, mode=mode)

    def positions(self) -> list[tuple[int, int]]:
        return list(self._positions)

    def labels(self) -> list[str]:
        return list(self._labels)

    def values(self) -> list[str]:
        return list(self._values)

    def save_json(self, path: str) -> None:
        if not path:
            raise ValueError("path must be provided")
        points_payload: list[dict[str, Any]] = []
        for idx, (y, x) in enumerate(self._positions):
            point = self._points[idx]
            points_payload.append(
                {
                    "index": idx,
                    "y": y,
                    "x": x,
                    "label": self._labels[idx],
                    "value": self._values[idx],
                    "hue": point.hue,
                    "ones": point.ones,
                }
            )
        payload = {
            "width": self.width,
            "height": self.height,
            "points": self._point_count,
            "similarity": self._similarity,
            "lambda_threshold": self._lambda,
            "eta": self._eta,
            "layout": points_payload,
        }
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, ensure_ascii=True, indent=2)
        except Exception as exc:
            LOGGER.event(
                "layout.export.json.error",
                section=LAID_OUT_STRUCTURE,
                data={"path": path, "error": str(exc)},
            )
            raise
        LOGGER.event(
            "layout.export.json",
            section=LAID_OUT_STRUCTURE,
            data={
                "path": path,
                "points": self._point_count,
                "width": self.width,
                "height": self.height,
            },
        )

    def colors_rgb(self) -> list[tuple[int, int, int]]:
        hues = self._normalized_hues()
        return [self._hue_to_rgb(hue) for hue in hues]

    def render_image(self, *, log: bool = True) -> "numpy.ndarray":
        colors = self.colors_rgb()
        image = self._build_image(colors)
        if log:
            positions = [(x + 0.5, y + 0.5) for y, x in self._positions]
            LOGGER.event(
                "layout.render_image",
                section=LAYOUT_ALGORITHM,
                data={"height": self.height, "width": self.width},
                visuals=[
                    LOGGER.visual_image(
                        "layout/render_image",
                        image,
                    ),
                    LOGGER.visual_points2d(
                        "layout/render_image",
                        positions,
                        colors=colors,
                        radii=0.5,
                    ),
                ],
            )
        return image

    def _build_image(
        self,
        colors: Sequence[tuple[int, int, int]] | None = None,
    ) -> "numpy.ndarray":
        try:
            import numpy as np
        except Exception as exc:
            LOGGER.event(
                "layout.energy.long.numpy_unavailable",
                section=OPTIM_ENERGY,
                data={"error": str(exc)},
            )
            return []

        if colors is None:
            colors = self.colors_rgb()
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for idx, (y, x) in enumerate(self._positions):
            if 0 <= y < self.height and 0 <= x < self.width:
                image[y, x] = colors[idx]
        return image

    def average_local_energy(
        self,
        radius: int,
        *,
        max_points: int | None = None,
    ) -> float:
        if radius <= 0:
            raise ValueError("radius must be positive")
        if self._point_count == 0:
            return 0.0
        use_subset = (
            max_points is not None and max_points > 0 and max_points < self._point_count
        )
        if not use_subset:
            if (
                self._gpu_engine is not None
                and self._ensure_gpu_similarity()
                and self._ensure_gpu_positions()
            ):
                try:
                    energy = self._gpu_engine.average_local_energy(
                        radius,
                        lambda_threshold=self._lambda,
                        eta=self._eta,
                        distance_eps=self._distance_eps,
                    )
                except Exception:
                    self._gpu_engine = None
                    energy = None
                if energy is not None:
                    LOGGER.event(
                        "layout.energy.average",
                        section=QUALITY_ASSESS,
                        data={"radius": radius, "energy": energy, "mode": "gpu"},
                    )
                    return energy
                self._gpu_engine = None

        radius_sq = radius * radius
        energies: list[float] = []
        pos_y = self._pos_y
        pos_x = self._pos_x
        sim_lambda = self._sim_lambda_idx
        if use_subset:
            indices = self._rng.sample(range(self._point_count), max_points)  # type: ignore[arg-type]
        else:
            indices = range(self._point_count)

        for idx in indices:
            cy = pos_y[idx]
            cx = pos_x[idx]
            energy = 0.0
            y_min = max(0, cy - radius)
            y_max = min(self.height - 1, cy + radius)
            x_min = max(0, cx - radius)
            x_max = min(self.width - 1, cx + radius)
            for y in range(y_min, y_max + 1):
                dy = cy - y
                dy_sq = dy * dy
                occupied = self._row_occupied[y]
                if not occupied:
                    continue
                for x in occupied:
                    if x < x_min or x > x_max:
                        continue
                    dx = cx - x
                    dist_sq = dy_sq + dx * dx
                    if dist_sq > radius_sq:
                        continue
                    other_idx = self.grid[y][x]
                    if other_idx is None or other_idx == idx:
                        continue
                    sim = sim_lambda(idx, other_idx)
                    if sim <= 0.0:
                        continue
                    energy += sim / (dist_sq + self._distance_eps)
            energies.append(energy)

        emax = max(energies)
        if emax <= 0.0:
            return 0.0
        avg_energy = sum(e / emax for e in energies) / len(energies)
        LOGGER.event(
            "layout.energy.average",
            section=QUALITY_ASSESS,
            data={
                "radius": radius,
                "energy": avg_energy,
                "mode": "cpu_sample" if use_subset else "cpu",
                "sample_size": len(energies),
                "points": self._point_count,
            },
        )
        return avg_energy

    def _place_points(self) -> None:
        cells = [(y, x) for y in range(self.height) for x in range(self.width)]
        self._rng.shuffle(cells)
        for idx, (y, x) in enumerate(cells[: len(self._points)]):
            self.grid[y][x] = idx
            self._positions[idx] = (y, x)
            self._pos_y[idx] = y
            self._pos_x[idx] = x
            self._row_occupied[y].add(x)

    def _sample_pair(self, radius: int | None) -> tuple[int, int, int, int]:
        idx_a = self._rng.randrange(self._point_count)
        y1, x1 = self._positions[idx_a]

        if radius is None:
            while True:
                y2 = self._rng.randrange(self.height)
                x2 = self._rng.randrange(self.width)
                if (y2, x2) != (y1, x1):
                    return y1, x1, y2, x2

        radius_sq = radius * radius
        for _ in range(50):
            dy = self._rng.randint(-radius, radius)
            dx = self._rng.randint(-radius, radius)
            if dy * dy + dx * dx > radius_sq:
                continue
            y2 = y1 + dy
            x2 = x1 + dx
            if 0 <= y2 < self.height and 0 <= x2 < self.width:
                if (y2, x2) != (y1, x1):
                    return y1, x1, y2, x2

        while True:
            y2 = self._rng.randrange(self.height)
            x2 = self._rng.randrange(self.width)
            if (y2, x2) != (y1, x1):
                return y1, x1, y2, x2

    def _sample_nearby_position(
        self,
        y1: int,
        x1: int,
        radius: int | None,
        bounds: tuple[int, int, int, int] | None = None,
    ) -> tuple[int, int]:
        y_min_bound = 0 if bounds is None else bounds[0]
        y_max_bound = (self.height - 1) if bounds is None else bounds[1]
        x_min_bound = 0 if bounds is None else bounds[2]
        x_max_bound = (self.width - 1) if bounds is None else bounds[3]
        if radius is None:
            while True:
                y2 = self._rng.randint(y_min_bound, y_max_bound)
                x2 = self._rng.randint(x_min_bound, x_max_bound)
                if (y2, x2) != (y1, x1):
                    return y2, x2
        radius_sq = radius * radius
        for _ in range(50):
            dy = self._rng.randint(-radius, radius)
            dx = self._rng.randint(-radius, radius)
            if dy * dy + dx * dx > radius_sq:
                continue
            y2 = y1 + dy
            x2 = x1 + dx
            if (
                y_min_bound <= y2 <= y_max_bound
                and x_min_bound <= x2 <= x_max_bound
                and (y2, x2) != (y1, x1)
            ):
                return y2, x2
        while True:
            y2 = self._rng.randint(y_min_bound, y_max_bound)
            x2 = self._rng.randint(x_min_bound, x_max_bound)
            if (y2, x2) != (y1, x1):
                return y2, x2

    def _compute_point_energies(
        self,
        radius: int,
        *,
        max_points: int | None = None,
    ) -> list[float]:
        if radius <= 0:
            raise ValueError("radius must be positive")
        if self._point_count == 0:
            return []
        radius_sq = radius * radius
        indices: Sequence[int]
        if max_points is not None and 0 < max_points < self._point_count:
            indices = self._rng.sample(range(self._point_count), max_points)  # type: ignore[arg-type]
        else:
            indices = range(self._point_count)

        energies = [0.0 for _ in range(self._point_count)]
        for idx in indices:
            cy = self._pos_y[idx]
            cx = self._pos_x[idx]
            y_min = max(0, cy - radius)
            y_max = min(self.height - 1, cy + radius)
            x_min = max(0, cx - radius)
            x_max = min(self.width - 1, cx + radius)
            energy = 0.0
            for y in range(y_min, y_max + 1):
                occupied = self._row_occupied[y]
                if not occupied:
                    continue
                dy = cy - y
                dy_sq = dy * dy
                for x in occupied:
                    if x < x_min or x > x_max:
                        continue
                    dx = cx - x
                    dist_sq = dy_sq + dx * dx
                    if dist_sq > radius_sq:
                        continue
                    other_idx = self.grid[y][x]
                    if other_idx is None or other_idx == idx:
                        continue
                    sim = self._sim_lambda_idx(idx, other_idx)
                    if sim <= 0.0:
                        continue
                    energy += sim / (dist_sq + self._distance_eps)
            energies[idx] = energy
        return energies

    def _ensure_short_energy_weights(
        self, *, step_index: int | None, config: ShortLayoutOptimConfig
    ) -> None:
        if not config.use_weighted_first_point:
            return
        if (
            self._short_energy_weights is not None
            and self._short_energy_last_step >= 0
            and step_index is not None
            and (step_index - self._short_energy_last_step) < config.energy_recalc_every
            and self._short_energy_radius == config.energy_radius
            and abs(self._short_energy_eps - config.energy_eps) < 1e-12
        ):
            return
        energies = self._compute_point_energies(
            config.energy_radius, max_points=config.energy_max_points
        )
        self._short_energy_radius = config.energy_radius
        self._short_energy_eps = config.energy_eps
        self._short_energy_last_step = 0 if step_index is None else step_index
        if not energies:
            self._short_energy_weights = None
            LOGGER.event(
                "layout.short.energy_weights.empty",
                section=OPTIM_FIRST_POINT,
                data={
                    "energy_radius": config.energy_radius,
                    "points": self._point_count,
                    "step": self._short_energy_last_step,
                },
            )
            return
        emax = max(energies)
        if emax <= 0.0:
            weights = [0.0 for _ in energies]
        else:
            weights = [
                max(0.0, -math.log(max(config.energy_eps, energy / emax)))
                for energy in energies
            ]
        self._short_energy_weights = weights
        LOGGER.event(
            "layout.short.energy_weights",
            section=OPTIM_FIRST_POINT,
            data={
                "energy_radius": config.energy_radius,
                "points": self._point_count,
                "step": self._short_energy_last_step,
                "recalc_every": config.energy_recalc_every,
                "max_points": -1 if config.energy_max_points is None else config.energy_max_points,
                "weighted": config.use_weighted_first_point,
            },
        )

    def _sample_first_point(self, *, weighted: bool) -> int:
        return self._sample_first_point_from_indices(
            list(range(self._point_count)), weighted=weighted
        )

    def _sample_first_point_from_indices(
        self, indices: Sequence[int], *, weighted: bool
    ) -> int:
        if not indices:
            raise ValueError("candidate indices must be non-empty")
        if weighted and self._short_energy_weights and sum(
            self._short_energy_weights[idx] for idx in indices
        ) > 0:
            try:
                weights = [self._short_energy_weights[idx] for idx in indices]
                return self._rng.choices(population=indices, weights=weights, k=1)[0]
            except Exception:
                pass
        return self._rng.choice(indices)

    def _sample_pairs_short(
        self,
        count: int,
        *,
        pair_radius: int | None,
        similarity_cutoff: float | None,
        weighted_first: bool,
        step_index: int | None,
        candidate_indices: Sequence[int] | None = None,
        bounds: tuple[int, int, int, int] | None = None,
    ) -> list[tuple[int, int, int, int]]:
        pairs: list[tuple[int, int, int, int]] = []
        rejected_similarity = 0
        attempts = 0
        max_attempts = max(count * 5, count)
        while len(pairs) < count and attempts < max_attempts:
            if candidate_indices is None:
                idx_a = self._sample_first_point(weighted=weighted_first)
            else:
                if not candidate_indices:
                    break
                idx_a = self._sample_first_point_from_indices(
                    candidate_indices, weighted=weighted_first
                )
            y1, x1 = self._positions[idx_a]
            y2, x2 = self._sample_nearby_position(y1, x1, pair_radius, bounds=bounds)
            idx_b = self.grid[y2][x2]
            if idx_b is not None:
                sim = self._similarity_value(idx_a, idx_b)
                if similarity_cutoff is not None and sim < similarity_cutoff:
                    rejected_similarity += 1
                    attempts += 1
                    continue
            pairs.append((y1, x1, y2, x2))
            attempts += 1
        while len(pairs) < count:
            pairs.append(self._sample_pair(pair_radius))
        LOGGER.event(
            "layout.short.pair_sampling",
            section=OPTIM_PAIR_SELECTION,
            data={
                "pairs_requested": count,
                "pairs_generated": len(pairs),
                "rejected_similarity": rejected_similarity,
                "attempts": attempts,
                "pair_radius": "auto" if pair_radius is None else pair_radius,
                "step": 0 if step_index is None else step_index,
                "weighted": weighted_first,
                "similarity_cutoff": similarity_cutoff,
                "candidate_indices": -1 if candidate_indices is None else len(candidate_indices),
                "bounds": bounds,
            },
        )
        return pairs

    def _random_segments(self, total: int, parts: int) -> list[tuple[int, int]]:
        parts = max(1, parts)
        if parts == 1:
            return [(0, total - 1)]
        cuts = sorted(self._rng.sample(range(1, total), min(parts - 1, max(0, total - 1))))
        segments = []
        prev = 0
        for cut in cuts:
            segments.append((prev, cut - 1))
            prev = cut
        segments.append((prev, total - 1))
        return segments

    def _build_short_tiles(
        self, partitions: int
    ) -> tuple[list[list[int]], list[tuple[int, int]], list[tuple[int, int]]]:
        row_parts = max(1, int(round(math.sqrt(partitions))))
        col_parts = max(1, math.ceil(partitions / row_parts))
        segments_y = self._random_segments(self.height, row_parts)
        segments_x = self._random_segments(self.width, col_parts)
        y_ends = [end for _, end in segments_y]
        x_ends = [end for _, end in segments_x]
        tiles = [[] for _ in range(row_parts * col_parts)]

        for y in range(self.height):
            occupied = self._row_occupied[y]
            if not occupied:
                continue
            y_idx = bisect.bisect_left(y_ends, y)
            for x in occupied:
                x_idx = bisect.bisect_left(x_ends, x)
                tile_idx = y_idx * col_parts + x_idx
                idx = self.grid[y][x]
                if idx is not None:
                    tiles[tile_idx].append(idx)
        LOGGER.event(
            "layout.short.tiles",
            section=PARALLEL_PROCESSING,
            data={
                "partitions": row_parts * col_parts,
                "row_parts": row_parts,
                "col_parts": col_parts,
                "segments_y": segments_y,
                "segments_x": segments_x,
                "points": self._point_count,
            },
        )
        return tiles, segments_y, segments_x

    def _sample_pairs_short_partitioned(
        self,
        count: int,
        *,
        pair_radius: int | None,
        similarity_cutoff: float | None,
        weighted_first: bool,
        step_index: int | None,
        partitions: int,
    ) -> list[tuple[int, int, int, int]]:
        tiles, segments_y, segments_x = self._build_short_tiles(partitions)
        tile_count = len(tiles)
        if tile_count == 0:
            return self._sample_pairs_short(
                count,
                pair_radius=pair_radius,
                similarity_cutoff=similarity_cutoff,
                weighted_first=weighted_first,
                step_index=step_index,
            )
        order = list(range(tile_count))
        self._rng.shuffle(order)
        base = count // tile_count
        extra = count % tile_count
        all_pairs: list[tuple[int, int, int, int]] = []
        for idx, tile_idx in enumerate(order):
            quota = base + (1 if idx < extra else 0)
            if quota <= 0:
                continue
            candidates = tiles[tile_idx]
            if not candidates:
                continue
            row_parts = len(segments_y)
            col_parts = len(segments_x)
            tile_row = tile_idx // col_parts
            tile_col = tile_idx % col_parts
            y_start, y_end = segments_y[tile_row]
            x_start, x_end = segments_x[tile_col]
            bounds = (y_start, y_end, x_start, x_end)
            pairs = self._sample_pairs_short(
                quota,
                pair_radius=pair_radius,
                similarity_cutoff=similarity_cutoff,
                weighted_first=weighted_first,
                step_index=step_index,
                candidate_indices=candidates,
                bounds=bounds,
            )
            all_pairs.extend(pairs)
        remaining = count - len(all_pairs)
        if remaining > 0:
            fallback = self._sample_pairs_short(
                remaining,
                pair_radius=pair_radius,
                similarity_cutoff=similarity_cutoff,
                weighted_first=weighted_first,
                step_index=step_index,
            )
            all_pairs.extend(fallback)
        LOGGER.event(
            "layout.short.partition_pairs",
            section=PARALLEL_PROCESSING,
            data={
                "requested": count,
                "produced": len(all_pairs),
                "partitions": tile_count,
                "remaining_fallback": remaining,
            },
        )
        return all_pairs

    def _evaluate_pairs_short_parallel(
        self,
        pairs: Sequence[tuple[int, int, int, int]],
        *,
        local_radius: int | None,
        workers: int,
        partitions: int,
    ) -> list[tuple[tuple[int, int, int, int], float]]:
        if not pairs:
            return []
        workers = max(1, workers)
        chunk_size = max(1, len(pairs) // workers)

        def _eval_chunk(chunk: Sequence[tuple[int, int, int, int]]) -> list[tuple[tuple[int, int, int, int], float]]:
            local_swaps: list[tuple[tuple[int, int, int, int], float]] = []
            for y1, x1, y2, x2 in chunk:
                should_swap, delta = self._should_swap(
                    y1, x1, y2, x2, mode="short", local_radius=local_radius
                )
                if should_swap:
                    local_swaps.append(((y1, x1, y2, x2), delta))
            return local_swaps

        LOGGER.event(
            "layout.short.parallel",
            section=PARALLEL_PROCESSING,
            data={
                "pairs": len(pairs),
                "workers": workers,
                "partitions": partitions,
                "chunk_size": chunk_size,
            },
        )

        results: list[tuple[tuple[int, int, int, int], float]] = []
        if workers == 1:
            return _eval_chunk(pairs)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for start in range(0, len(pairs), chunk_size):
                futures.append(
                    executor.submit(_eval_chunk, pairs[start : start + chunk_size])
                )
            for fut in futures:
                try:
                    results.extend(fut.result())
                except Exception as exc:
                    LOGGER.event(
                        "layout.short.parallel.error",
                        section=PARALLEL_PROCESSING,
                        data={"error": str(exc)},
                    )
        return results

    def _short_pair_radius(self, y1: int, x1: int, y2: int, x2: int, local_radius: int | None) -> int:
        dist_sq = (y1 - y2) ** 2 + (x1 - x2) ** 2
        base_radius = max(1, int(math.ceil(math.sqrt(dist_sq))))
        if local_radius is None:
            return base_radius
        return max(local_radius, base_radius)

    def _collect_short_subset(
        self, pairs: Sequence[_ShortPair]
    ) -> list[int]:
        if not pairs:
            return []
        subset: set[int] = set()
        for pair in pairs:
            cy = (pair.y1 + pair.y2) / 2.0
            cx = (pair.x1 + pair.x2) / 2.0
            radius_sq = pair.radius * pair.radius
            y_min = max(0, int(math.floor(cy - pair.radius)))
            y_max = min(self.height - 1, int(math.ceil(cy + pair.radius)))
            x_min = max(0, int(math.floor(cx - pair.radius)))
            x_max = min(self.width - 1, int(math.ceil(cx + pair.radius)))
            for y in range(y_min, y_max + 1):
                occupied = self._row_occupied[y]
                if not occupied:
                    continue
                dyc = y - cy
                dyc_sq = dyc * dyc
                for x in occupied:
                    if x < x_min or x > x_max:
                        continue
                    dxc = x - cx
                    if dyc_sq + dxc * dxc > radius_sq:
                        continue
                    idx = self.grid[y][x]
                    if idx is None or idx == pair.idx_a or idx == pair.idx_b:
                        continue
                    subset.add(idx)
        return sorted(subset)

    def _evaluate_pairs_short_tensor(
        self,
        pairs: Sequence[tuple[int, int, int, int]],
        *,
        local_radius: int | None,
    ) -> list[tuple[tuple[int, int, int, int], float]] | None:
        tensor_engine = self._tensor_engine if self._use_gpu else None
        if tensor_engine is None:
            return None
        prepared: list[_ShortPair] = []
        for y1, x1, y2, x2 in pairs:
            idx_a = self.grid[y1][x1]
            idx_b = self.grid[y2][x2]
            if idx_a is None and idx_b is None:
                continue
            if idx_a is None:
                idx_a, idx_b = idx_b, idx_a
                y1, x1, y2, x2 = y2, x2, y1, x1
            radius = self._short_pair_radius(y1, x1, y2, x2, local_radius)
            prepared.append(
                _ShortPair(
                    y1=y1,
                    x1=x1,
                    y2=y2,
                    x2=x2,
                    idx_a=idx_a,
                    idx_b=idx_b,
                    radius=float(radius),
                )
            )
        if not prepared:
            return []
        subset_indices = self._collect_short_subset(prepared)
        subset_key = tuple(subset_indices)
        if subset_key != self._short_subset_indices:
            self._short_subset_indices = subset_key
            self._short_similarity_cache.clear()
        if not subset_indices:
            LOGGER.event(
                "layout.energy.short.tensor.empty_subset",
                section=ENERGY_SHORT,
                data={
                    "pairs": len(prepared),
                    "points": self._point_count,
                },
            )
            return []
        try:
            if not tensor_engine.ensure_subset(subset_indices, self._lambda, self._eta):
                return None
        except Exception as exc:
            LOGGER.event(
                "layout.gpu.tensor.short.disabled",
                section=GPU_IMPLEMENTATION,
                data={"error": str(exc)},
            )
            self._tensor_engine = None
            return None

        unique_indices = sorted(
            {
                idx
                for pair in prepared
                for idx in (pair.idx_a, pair.idx_b)
                if idx is not None
            }
        )
        missing = [idx for idx in unique_indices if idx not in self._short_similarity_cache]
        if missing:
            missing_matrix: "numpy.ndarray | None"
            try:
                missing_matrix = tensor_engine.similarity_block(
                    missing, subset_indices, self._lambda, self._eta
                )
            except Exception as exc:
                LOGGER.event(
                    "layout.gpu.tensor.short.sim_failed",
                    section=GPU_IMPLEMENTATION,
                    data={"error": str(exc)},
                )
                missing_matrix = None
            if missing_matrix is None:
                missing_matrix = self._apply_lambda_array(
                    self._similarity_block(missing, subset_indices)
                )
            for row, idx in enumerate(missing):
                self._short_similarity_cache[idx] = missing_matrix[row]
                try:
                    tensor_engine.register_cache_row(idx, missing_matrix[row])
                except Exception:
                    pass

        pair_payload = [
            (
                pair.y1,
                pair.x1,
                pair.y2,
                pair.x2,
                pair.idx_a,
                pair.idx_b,
                pair.radius,
            )
            for pair in prepared
        ]

        try:
            tensor_batch = tensor_engine.short_pair_energies(
                pair_payload,
                self._pos_y,
                self._pos_x,
                self._short_similarity_cache,
                self._distance_eps,
            )
        except Exception as exc:
            LOGGER.event(
                "layout.gpu.tensor.short.disabled",
                section=GPU_IMPLEMENTATION,
                data={"error": str(exc)},
            )
            self._tensor_engine = None
            tensor_batch = None

        if tensor_batch is None:
            return None

        LOGGER.event(
            "layout.energy.short.tensor",
            section=GPU_IMPLEMENTATION,
            data={
                "pairs": len(pair_payload),
                "subset_points": tensor_batch.subset_size,
                "points": self._point_count,
                "device": tensor_engine.device,
            },
        )
        swaps_with_delta: list[tuple[tuple[int, int, int, int], float]] = []
        for (pair, (energy_c, energy_s)) in zip(prepared, tensor_batch.energies):
            delta = energy_s - energy_c
            if energy_s > energy_c:
                swaps_with_delta.append(((pair.y1, pair.x1, pair.y2, pair.x2), delta))
        return swaps_with_delta

    def _should_swap(
        self,
        y1: int,
        x1: int,
        y2: int,
        x2: int,
        *,
        mode: str,
        local_radius: int | None,
    ) -> tuple[bool, float]:
        idx_a = self.grid[y1][x1]
        idx_b = self.grid[y2][x2]

        if idx_a is None and idx_b is None:
            return False, 0.0
        if idx_a is None:
            idx_a, idx_b = idx_b, idx_a
            y1, x1, y2, x2 = y2, x2, y1, x1

        if mode == "long":
            energy_c, energy_s = self._pair_energy_long(
                idx_a, (y1, x1), idx_b, (y2, x2)
            )
            return energy_s < energy_c, energy_c - energy_s

        dist_sq = (y1 - y2) ** 2 + (x1 - x2) ** 2
        base_radius = max(1, int(math.ceil(math.sqrt(dist_sq))))
        radius = base_radius if local_radius is None else max(local_radius, base_radius)
        energy_c, energy_s = self._pair_energy_short(
            idx_a, (y1, x1), idx_b, (y2, x2), radius=radius
        )
        return energy_s > energy_c, energy_s - energy_c

    def _pair_energy_long(
        self,
        idx_a: int,
        pos_a: tuple[int, int],
        idx_b: int | None,
        pos_b: tuple[int, int],
    ) -> tuple[float, float]:
        energies = self._calculate_far_energies(
            [(pos_a[0], pos_a[1], pos_b[0], pos_b[1], idx_a, idx_b)],
            advance_step=False,
        )
        if not energies:
            return 0.0, 0.0
        return energies[0]

    def _calculate_far_energies(
        self,
        pairs: Sequence[tuple[int, int, int, int, int | None, int | None]],
        *,
        advance_step: bool = True,
    ) -> list[tuple[float, float]]:
        import numpy as np

        batch_size = len(pairs)
        if batch_size == 0:
            LOGGER.event(
                "layout.energy.long.empty",
                section=ENERGY_LONG,
                data={
                    "pairs": 0,
                    "points": self._point_count,
                    "space_size": (self.height, self.width),
                },
            )
            return []

        if advance_step:
            self._long_step_counter += 1
        step_no = self._long_step_counter if advance_step else self._long_step_counter
        eval_indices = self._ensure_long_subset(step_no)
        if not eval_indices:
            return []

        cutoff = self._lambda if self._eta is None else 0.0
        LOGGER.event(
            "layout.energy.long.similarity",
            section=SIMILARITY_MEASURES,
            data={
                "similarity": self._similarity,
                "pairs": batch_size,
                "points": self._point_count,
            },
        )
        LOGGER.event(
            "layout.energy.long.sim_lambda",
            section=SIMILARITY_DEFINITION,
            data={
                "lambda_threshold": self._lambda,
                "eta": self._eta,
                "cutoff": cutoff,
            },
        )
        LOGGER.event(
            "layout.energy.long.threshold",
            section=ENERGIES,
            data={
                "lambda_threshold": self._lambda,
                "eta": self._eta,
                "cutoff": cutoff,
            },
        )
        LOGGER.event(
            "layout.energy.pair.ignore_self",
            section=ENERGY_CALC,
            data={"skip_pair_points": True},
        )
        empty_cells = (self.height * self.width) - self._point_count
        LOGGER.event(
            "layout.energy.long.space",
            section=ENERGY_LONG,
            data={
                "space_size": (self.height, self.width),
                "points": self._point_count,
                "empty_cells": empty_cells,
                "eval_points": len(eval_indices),
            },
        )

        tensor_batch: _TensorEnergyBatch | None = None
        tensor_ready = False
        tensor_engine = self._tensor_engine if self._use_gpu else None
        if tensor_engine is not None:
            try:
                tensor_ready = tensor_engine.ensure_subset(
                    eval_indices, self._lambda, self._eta
                )
            except Exception as exc:
                LOGGER.event(
                    "layout.gpu.tensor.disabled",
                    section=GPU_IMPLEMENTATION,
                    data={"error": str(exc)},
                )
                tensor_engine = None
                self._tensor_engine = None

        unique_indices = sorted(
            {
                idx
                for (_, _, _, _, idx_a, idx_b) in pairs
                for idx in (idx_a, idx_b)
                if idx is not None
            }
        )
        if not unique_indices:
            return [(0.0, 0.0) for _ in pairs]

        missing = [idx for idx in unique_indices if idx not in self._subset_similarity_cache]
        if missing:
            missing_matrix: "np.ndarray | None" = None
            if tensor_ready and tensor_engine is not None:
                try:
                    missing_matrix = tensor_engine.similarity_block(
                        missing, eval_indices, self._lambda, self._eta
                    )
                except Exception as exc:
                    LOGGER.event(
                        "layout.gpu.tensor.disabled",
                        section=GPU_IMPLEMENTATION,
                        data={"error": str(exc)},
                    )
                    tensor_ready = False
                    tensor_engine = None
                    self._tensor_engine = None
            if missing_matrix is None:
                missing_matrix = self._apply_lambda_array(
                    self._similarity_block(missing, eval_indices)
                ).astype(np.float32, copy=False)
            for row, idx in enumerate(missing):
                self._subset_similarity_cache[idx] = missing_matrix[row]
                if tensor_ready and tensor_engine is not None:
                    tensor_engine.register_cache_row(idx, missing_matrix[row])
        sim_lookup = self._subset_similarity_cache

        if tensor_ready and tensor_engine is not None:
            try:
                tensor_batch = tensor_engine.pair_energies(
                    pairs,
                    eval_indices,
                    self._pos_y,
                    self._pos_x,
                    sim_lookup,
                    self._distance_eps,
                )
            except Exception as exc:
                LOGGER.event(
                    "layout.gpu.tensor.disabled",
                    section=GPU_IMPLEMENTATION,
                    data={"error": str(exc)},
                )
                tensor_batch = None
                tensor_engine = None
                self._tensor_engine = None
            if tensor_batch is not None:
                LOGGER.event(
                    "layout.energy.long.tensor",
                    section=GPU_IMPLEMENTATION,
                    data={
                        "pairs": batch_size,
                        "points": self._point_count,
                        "eval_points": len(eval_indices),
                    },
                )
                gpu_results: list[tuple[float, float]] = []
                for pair_idx, (y1, x1, y2, x2, idx_a, idx_b) in enumerate(pairs):
                    energy_c, energy_s = tensor_batch.energies[pair_idx]
                    gpu_results.append((float(energy_c), float(energy_s)))
                    if not LOGGER.should_log("layout.energy.long.pair"):
                        continue
                    ones_a = 0 if idx_a is None else self._points[idx_a].ones
                    ones_b = 0 if idx_b is None else self._points[idx_b].ones
                    LOGGER.event(
                        "layout.energy.long.pair",
                        section=ENERGY_LONG,
                        data={
                            "pair_index": pair_idx,
                            "y1": y1,
                            "x1": x1,
                            "y2": y2,
                            "x2": x2,
                            "idx_a": -1 if idx_a is None else idx_a,
                            "idx_b": -1 if idx_b is None else idx_b,
                            "ones_a": ones_a,
                            "ones_b": ones_b,
                            "phi_c": float(energy_c),
                            "phi_s": float(energy_s),
                            "s1_sum": tensor_batch.s1_sum[pair_idx],
                            "s2_sum": tensor_batch.s2_sum[pair_idx],
                            "s1_min": tensor_batch.s1_min[pair_idx],
                            "s1_max": tensor_batch.s1_max[pair_idx],
                            "s2_min": tensor_batch.s2_min[pair_idx],
                            "s2_max": tensor_batch.s2_max[pair_idx],
                            "d1_min": tensor_batch.d1_min[pair_idx],
                            "d1_max": tensor_batch.d1_max[pair_idx],
                            "d2_min": tensor_batch.d2_min[pair_idx],
                            "d2_max": tensor_batch.d2_max[pair_idx],
                            "contrib_points": tensor_batch.contrib_points[pair_idx],
                            "points_total": self._point_count,
                            "eval_points": len(eval_indices),
                        },
                        force=True,
                    )
                return gpu_results

        eval_idx_arr = np.asarray(eval_indices, dtype=np.int32)
        eval_y = np.take(self._pos_y, eval_idx_arr).astype(np.float64)
        eval_x = np.take(self._pos_x, eval_idx_arr).astype(np.float64)
        zero_vec = np.zeros(len(eval_indices), dtype=np.float32)

        results: list[tuple[float, float]] = []
        for pair_idx, (y1, x1, y2, x2, idx_a, idx_b) in enumerate(pairs):
            s1 = zero_vec
            s2 = zero_vec
            ones_a = 0
            ones_b = 0
            if idx_a is not None:
                s1 = sim_lookup.get(idx_a, zero_vec)
                ones_a = self._points[idx_a].ones
            if idx_b is not None:
                s2 = sim_lookup.get(idx_b, zero_vec)
                ones_b = self._points[idx_b].ones
            if idx_a is not None or idx_b is not None:
                mask = np.zeros_like(eval_idx_arr, dtype=bool)
                if idx_a is not None:
                    mask |= eval_idx_arr == idx_a
                if idx_b is not None:
                    mask |= eval_idx_arr == idx_b
                if mask.any():
                    if idx_a is not None:
                        s1 = np.where(mask, 0.0, s1)
                    if idx_b is not None:
                        s2 = np.where(mask, 0.0, s2)

            dy1 = y1 - eval_y
            dx1 = x1 - eval_x
            dy2 = y2 - eval_y
            dx2 = x2 - eval_x
            dist1 = dy1 * dy1 + dx1 * dx1
            dist2 = dy2 * dy2 + dx2 * dx2
            energy_c = float(np.sum(s1 * dist1 + s2 * dist2))
            energy_s = float(np.sum(s2 * dist1 + s1 * dist2))

            results.append((energy_c, energy_s))
            if LOGGER.should_log("layout.energy.long.pair"):
                contrib_mask = (s1 > 0.0) | (s2 > 0.0)
                count = int(np.count_nonzero(contrib_mask))
                if count:
                    s1_m = s1[contrib_mask]
                    s2_m = s2[contrib_mask]
                    d1_m = dist1[contrib_mask]
                    d2_m = dist2[contrib_mask]
                    sim1_min_val = float(s1_m.min())
                    sim1_max_val = float(s1_m.max())
                    sim2_min_val = float(s2_m.min())
                    sim2_max_val = float(s2_m.max())
                    d1_min_val = float(d1_m.min())
                    d1_max_val = float(d1_m.max())
                    d2_min_val = float(d2_m.min())
                    d2_max_val = float(d2_m.max())
                    sim1_sum_val = float(s1_m.sum())
                    sim2_sum_val = float(s2_m.sum())
                else:
                    sim1_min_val = 0.0
                    sim1_max_val = 0.0
                    sim2_min_val = 0.0
                    sim2_max_val = 0.0
                    d1_min_val = 0.0
                    d1_max_val = 0.0
                    d2_min_val = 0.0
                    d2_max_val = 0.0
                    sim1_sum_val = 0.0
                    sim2_sum_val = 0.0

                LOGGER.event(
                    "layout.energy.long.pair",
                    section=ENERGY_LONG,
                    data={
                        "pair_index": pair_idx,
                        "y1": y1,
                        "x1": x1,
                        "y2": y2,
                        "x2": x2,
                        "idx_a": -1 if idx_a is None else idx_a,
                        "idx_b": -1 if idx_b is None else idx_b,
                        "ones_a": ones_a,
                        "ones_b": ones_b,
                        "phi_c": energy_c,
                        "phi_s": energy_s,
                        "s1_sum": sim1_sum_val,
                        "s2_sum": sim2_sum_val,
                        "s1_min": sim1_min_val,
                        "s1_max": sim1_max_val,
                        "s2_min": sim2_min_val,
                        "s2_max": sim2_max_val,
                        "d1_min": d1_min_val,
                        "d1_max": d1_max_val,
                        "d2_min": d2_min_val,
                        "d2_max": d2_max_val,
                        "contrib_points": count,
                        "points_total": self._point_count,
                        "eval_points": len(eval_indices),
                    },
                    force=True,
                )

        return results

    def _pair_energy_short(
        self,
        idx_a: int,
        pos_a: tuple[int, int],
        idx_b: int | None,
        pos_b: tuple[int, int],
        *,
        radius: int,
    ) -> tuple[float, float]:
        cy = (pos_a[0] + pos_b[0]) / 2.0
        cx = (pos_a[1] + pos_b[1]) / 2.0
        radius_sq = radius * radius

        sim_lambda = self._sim_lambda_idx
        energy_c = 0.0
        energy_s = 0.0
        y_min = max(0, int(math.floor(cy - radius)))
        y_max = min(self.height - 1, int(math.ceil(cy + radius)))
        x_min = max(0, int(math.floor(cx - radius)))
        x_max = min(self.width - 1, int(math.ceil(cx + radius)))
        for y in range(y_min, y_max + 1):
            dyc = y - cy
            dyc_sq = dyc * dyc
            row = self.grid[y]
            occupied = self._row_occupied[y]
            if not occupied:
                continue
            for x in occupied:
                if x < x_min or x > x_max:
                    continue
                idx = row[x]
                if idx is None or idx == idx_a or idx == idx_b:
                    continue
                dxc = x - cx
                if dyc_sq + dxc * dxc > radius_sq:
                    continue
                s1 = sim_lambda(idx, idx_a)
                s2 = sim_lambda(idx, idx_b) if idx_b is not None else 0.0
                if s1 == 0.0 and s2 == 0.0:
                    continue
                dy1 = pos_a[0] - y
                dx1 = pos_a[1] - x
                dy2 = pos_b[0] - y
                dx2 = pos_b[1] - x
                d1 = dy1 * dy1 + dx1 * dx1
                d2 = dy2 * dy2 + dx2 * dx2
                energy_c += s1 / (d1 + self._distance_eps) + s2 / (
                    d2 + self._distance_eps
                )
                energy_s += s2 / (d1 + self._distance_eps) + s1 / (
                    d2 + self._distance_eps
                )
        return energy_c, energy_s

    def _apply_tracked_swaps(
        self, swaps_with_delta: Sequence[tuple[tuple[int, int, int, int], float]], *, mode: str
    ) -> int:
        used: set[tuple[int, int]] = set()
        swaps: list[tuple[int, int, int, int]] = []
        for (y1, x1, y2, x2), delta in swaps_with_delta:
            if (y1, x1) in used or (y2, x2) in used:
                continue
            swaps.append((y1, x1, y2, x2))
            used.add((y1, x1))
            used.add((y2, x2))
        applied = self._apply_swaps(swaps, mode=mode)
        return applied

    def _apply_swaps(self, swaps: Iterable[tuple[int, int, int, int]], *, mode: str) -> int:
        used: set[tuple[int, int]] = set()
        applied = 0
        for y1, x1, y2, x2 in swaps:
            if (y1, x1) in used or (y2, x2) in used:
                continue
            idx_a = self.grid[y1][x1]
            idx_b = self.grid[y2][x2]
            if idx_a is not None:
                self._row_occupied[y1].discard(x1)
            if idx_b is not None:
                self._row_occupied[y2].discard(x2)
            self.grid[y1][x1], self.grid[y2][x2] = idx_b, idx_a
            if idx_a is not None:
                self._positions[idx_a] = (y2, x2)
                self._pos_y[idx_a] = y2
                self._pos_x[idx_a] = x2
                self._row_occupied[y2].add(x2)
            if idx_b is not None:
                self._positions[idx_b] = (y1, x1)
                self._pos_y[idx_b] = y1
                self._pos_x[idx_b] = x1
                self._row_occupied[y1].add(x1)
            used.add((y1, x1))
            used.add((y2, x2))
            applied += 1
            if mode == "long":
                self._register_long_swap_radius(y1, x1, y2, x2)
        if applied:
            self._short_energy_weights = None
            self._short_energy_last_step = -1
            self._gpu_positions_dirty = True
        return applied

    def _register_long_swap_radius(self, y1: int, x1: int, y2: int, x2: int) -> None:
        radius = math.hypot(y1 - y2, x1 - x2)
        self._long_swap_radius_sum += radius
        self._long_swap_count += 1

    def _build_similarity_cache(self) -> None:
        point_count = self._point_count
        workers = self._similarity_worker_count()
        LOGGER.event(
            "layout.sim_cache.start",
            section=OPTIM_SIM_MATRIX,
            data={"point_count": point_count},
        )
        LOGGER.event(
            "layout.sim_cache.similarity",
            section=SIMILARITY_MEASURES,
            data={"similarity": self._similarity},
        )
        LOGGER.event(
            "layout.sim_cache.parallel",
            section=PARALLEL_PROCESSING,
            data={"workers": workers},
        )
        LOGGER.event(
            "layout.sim_cache.gpu",
            section=GPU_IMPLEMENTATION,
            data={"use_gpu": self._use_gpu},
        )
        used_gpu = False
        if self._use_gpu:
            sim_base = self._build_similarity_cache_gpu()
            if sim_base is not None:
                self._sim_base = sim_base
                if self._gpu_engine is not None and not self._gpu_engine.has_similarity:
                    try:
                        if not self._gpu_engine.upload_similarity(sim_base):
                            self._gpu_engine = None
                    except Exception:
                        self._gpu_engine = None
                used_gpu = True
                LOGGER.event(
                    "layout.sim_cache.done",
                    section=GPU_IMPLEMENTATION,
                    data={"mode": "gpu"},
                )
                return

        sim_base: list[list[float]] = [
            [0.0] * point_count for _ in range(point_count)
        ]
        for i in range(point_count):
            sim_base[i][i] = 1.0

        if workers < 2:
            for i in range(point_count):
                for j in range(i + 1, point_count):
                    sim = self._similarity_base(self._points[i], self._points[j])
                    sim_base[i][j] = sim
                    sim_base[j][i] = sim
        else:
            tasks = [(offset, workers, point_count) for offset in range(workers)]
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_similarity_worker,
                initargs=(self._points, self._similarity),
            ) as executor:
                for chunk in executor.map(_compute_similarity_stride, tasks):
                    for i, sims in chunk:
                        row = sim_base[i]
                        for j, sim in enumerate(sims, start=i + 1):
                            row[j] = sim
                            sim_base[j][i] = sim
        self._sim_base = sim_base
        if self._gpu_engine is not None:
            try:
                if not self._gpu_engine.upload_similarity(sim_base):
                    self._gpu_engine = None
            except Exception:
                self._gpu_engine = None
        if not used_gpu:
            LOGGER.event(
                "layout.sim_cache.done",
                section=OPTIM_SIM_MATRIX,
                data={"mode": "cpu"},
            )

    def _build_similarity_cache_gpu(self) -> list[list[float]] | None:
        if self._gpu_engine is not None:
            sim_base = self._gpu_engine.build_similarity_matrix()
            if sim_base is not None:
                return sim_base

        try:
            import numpy as np
            import moderngl
        except Exception:
            return None

        point_count = self._point_count
        if point_count == 0:
            return []

        code_len = len(self._points[0].code)
        codes = np.zeros((point_count, code_len), dtype=np.float32)
        ones = np.zeros(point_count, dtype=np.float32)
        for idx, point in enumerate(self._points):
            ones[idx] = point.ones
            bits = np.frombuffer(point.code._bits, dtype=np.uint8, count=code_len)
            codes[idx] = bits.astype(np.float32)

        try:
            ctx = moderngl.create_standalone_context(require=410)
        except Exception:
            try:
                ctx = moderngl.create_standalone_context()
            except Exception:
                return None
            if getattr(ctx, "version_code", 0) < 410:
                return None

        codes_tex = ctx.texture(
            (code_len, point_count), 1, data=codes.tobytes(), dtype="f4"
        )
        codes_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        ones_tex = ctx.texture(
            (point_count, 1), 1, data=ones.tobytes(), dtype="f4"
        )
        ones_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        out_tex = ctx.texture((point_count, point_count), 1, dtype="f4")
        fbo = ctx.framebuffer(color_attachments=[out_tex])

        prog = ctx.program(
            vertex_shader="""
                #version 410
                const vec2 verts[3] = vec2[3](
                    vec2(-1.0, -1.0),
                    vec2(3.0, -1.0),
                    vec2(-1.0, 3.0)
                );
                void main() {
                    gl_Position = vec4(verts[gl_VertexID], 0.0, 1.0);
                }
            """,
            fragment_shader="""
                #version 410
                uniform sampler2D codes;
                uniform sampler2D ones;
                uniform int code_len;
                uniform int point_count;
                uniform int use_cosine;
                out float frag;

                void main() {
                    ivec2 coord = ivec2(gl_FragCoord.xy);
                    int i = coord.y;
                    int j = coord.x;
                    if (i >= point_count || j >= point_count) {
                        frag = 0.0;
                        return;
                    }
                    if (i == j) {
                        frag = 1.0;
                        return;
                    }

                    float ones_a = texelFetch(ones, ivec2(i, 0), 0).r;
                    float ones_b = texelFetch(ones, ivec2(j, 0), 0).r;
                    if (ones_a <= 0.0 || ones_b <= 0.0) {
                        frag = 0.0;
                        return;
                    }

                    float common_bits = 0.0;
                    for (int k = 0; k < code_len; k++) {
                        float a = texelFetch(codes, ivec2(k, i), 0).r;
                        float b = texelFetch(codes, ivec2(k, j), 0).r;
                        common_bits += a * b;
                    }

                    if (use_cosine == 1) {
                        float denom = sqrt(ones_a * ones_b);
                        frag = denom == 0.0 ? 0.0 : common_bits / denom;
                        return;
                    }
                    float uni = ones_a + ones_b - common_bits;
                    frag = uni == 0.0 ? 0.0 : common_bits / uni;
                }
            """,
        )

        codes_tex.use(location=0)
        ones_tex.use(location=1)
        prog["codes"].value = 0
        prog["ones"].value = 1
        prog["code_len"].value = code_len
        prog["point_count"].value = point_count
        prog["use_cosine"].value = 1 if self._similarity == "cosine" else 0

        fbo.use()
        ctx.viewport = (0, 0, point_count, point_count)
        vao = ctx.vertex_array(prog, [])
        vao.render(mode=moderngl.TRIANGLES, vertices=3)

        data = out_tex.read()
        sim_matrix = np.frombuffer(data, dtype=np.float32).reshape(
            (point_count, point_count)
        )
        return sim_matrix.tolist()

    def _ensure_gpu_similarity(self) -> bool:
        if self._gpu_engine is None:
            return False
        if self._gpu_engine.has_similarity:
            return True
        if self._sim_base is not None:
            try:
                return self._gpu_engine.upload_similarity(self._sim_base)
            except Exception:
                self._gpu_engine = None
                return False
        try:
            sim_base = self._gpu_engine.build_similarity_matrix()
        except Exception:
            self._gpu_engine = None
            return False
        if sim_base is None:
            return False
        if self._sim_base is None:
            self._sim_base = sim_base
        return True

    def _ensure_gpu_positions(self) -> bool:
        if self._gpu_engine is None:
            return False
        if not self._gpu_positions_dirty:
            return True
        try:
            if not self._gpu_engine.update_positions(self._pos_y, self._pos_x):
                self._gpu_engine = None
                return False
        except Exception:
            self._gpu_engine = None
            return False
        self._gpu_positions_dirty = False
        return True

    def _build_code_matrix(self) -> tuple["numpy.ndarray | None", "numpy.ndarray | None"]:
        try:
            import numpy as np
        except Exception as exc:
            LOGGER.event(
                "layout.sim_matrix.numpy_unavailable",
                section=OPTIM_ENERGY,
                data={"error": str(exc)},
            )
            return None, None
        if self._point_count == 0:
            return None, None
        code_len = len(self._points[0].code)
        matrix = np.zeros((self._point_count, code_len), dtype=np.uint8)
        ones = np.zeros(self._point_count, dtype=np.float32)
        for idx, point in enumerate(self._points):
            matrix[idx] = np.frombuffer(point.code._bits, dtype=np.uint8, count=code_len)
            ones[idx] = float(point.ones)
        LOGGER.event(
            "layout.sim_matrix.ready",
            section=OPTIM_ENERGY,
            data={
                "points": self._point_count,
                "code_len": code_len,
            },
        )
        return matrix, ones

    def _similarity_worker_count(self) -> int:
        if self._parallel_workers is not None:
            if self._parallel_workers <= 1:
                return 1
            return min(self._parallel_workers, self._point_count)
        cpu_count = os.cpu_count() or 1
        total_pairs = self._point_count * (self._point_count - 1) // 2
        if cpu_count < 2 or total_pairs < _PARALLEL_SIM_MIN_PAIRS:
            return 1
        LOGGER.event(
            "layout.parallel.cpu_count",
            section=PARALLEL_PROCESSING,
            data={"cpu_count": cpu_count},
        )
        return min(cpu_count, self._point_count)

    def _ensure_long_subset(self, step: int | None) -> list[int]:
        if self._point_count == 0:
            self._long_subset_indices = []
            if self._tensor_engine is not None:
                self._tensor_engine.reset_subset()
            return self._long_subset_indices
        subset_changed = False
        if self._long_subset_size is None or self._point_count <= self._long_subset_size:
            if len(self._long_subset_indices) != self._point_count:
                self._long_subset_indices = list(range(self._point_count))
                self._subset_similarity_cache.clear()
                subset_changed = True
                LOGGER.event(
                    "layout.energy.subset.full",
                    section=OPTIM_SUBSET,
                    data={
                        "points": self._point_count,
                        "selected": len(self._long_subset_indices),
                    },
                )
            return self._long_subset_indices

        need_refresh = not self._long_subset_indices
        if step is not None:
            need_refresh = need_refresh or (step - self._long_subset_last_step) >= self._long_subset_refresh
        if need_refresh:
            self._long_subset_indices = self._rng.sample(
                range(self._point_count), self._long_subset_size
            )
            self._long_subset_last_step = 0 if step is None else step
            self._subset_similarity_cache.clear()
            subset_changed = True
            LOGGER.event(
                "layout.energy.subset.refresh",
                section=OPTIM_SUBSET,
                data={
                    "points": self._point_count,
                    "selected": len(self._long_subset_indices),
                    "step": self._long_subset_last_step,
                },
            )
        if subset_changed and self._tensor_engine is not None:
            self._tensor_engine.reset_subset()
        return self._long_subset_indices

    def _apply_lambda_array(self, sim: "numpy.ndarray") -> "numpy.ndarray":
        import numpy as np

        if sim.size == 0:
            return sim
        if self._eta is None:
            if self._lambda <= 0.0:
                return np.where(sim > 0.0, sim, 0.0)
            return np.where(sim >= self._lambda, sim, 0.0)
        scaled = 1.0 / (1.0 + np.exp(-self._eta * (sim - self._lambda)))
        return sim * scaled

    def _similarity_value(self, idx_a: int, idx_b: int) -> float:
        if idx_a == idx_b:
            return 1.0
        if self._sim_base is not None:
            return self._sim_base[idx_a][idx_b]
        if self._code_matrix is not None and self._ones_array is not None:
            vec_a = self._code_matrix[idx_a]
            vec_b = self._code_matrix[idx_b]
            common = float(vec_a @ vec_b)
            ones_a = float(self._ones_array[idx_a])
            ones_b = float(self._ones_array[idx_b])
            if ones_a <= 0.0 or ones_b <= 0.0:
                return 0.0
            if self._similarity == "cosine":
                denom = math.sqrt(ones_a * ones_b)
                return 0.0 if denom == 0 else common / denom
            union = ones_a + ones_b - common
            return 0.0 if union == 0.0 else common / union
        return self._similarity_base(self._points[idx_a], self._points[idx_b])

    def _similarity_block(
        self, row_indices: Sequence[int], col_indices: Sequence[int]
    ) -> "numpy.ndarray":
        import numpy as np

        if not row_indices or not col_indices:
            return np.zeros((len(row_indices), len(col_indices)), dtype=np.float32)
        if self._code_matrix is None or self._ones_array is None:
            block = np.zeros((len(row_indices), len(col_indices)), dtype=np.float32)
            for i, ridx in enumerate(row_indices):
                for j, cidx in enumerate(col_indices):
                    block[i, j] = self._similarity_value(ridx, cidx)
            return block
        rows = self._code_matrix[list(row_indices)].astype(np.float32, copy=False)
        cols = self._code_matrix[list(col_indices)].astype(np.float32, copy=False).T
        common = rows @ cols
        ones_rows = self._ones_array[list(row_indices)].astype(np.float32, copy=False)
        ones_cols = self._ones_array[list(col_indices)].astype(np.float32, copy=False)
        if self._similarity == "cosine":
            denom = np.sqrt(np.outer(ones_rows, ones_cols))
            sim = np.divide(common, denom, out=np.zeros_like(common, dtype=np.float32), where=denom != 0)
        else:
            union = ones_rows[:, None] + ones_cols[None, :] - common
            sim = np.divide(common, union, out=np.zeros_like(common, dtype=np.float32), where=union != 0)
        return sim.astype(np.float32)

    def _sim_lambda_idx(self, idx_a: int, idx_b: int | None) -> float:
        if idx_b is None:
            return 0.0
        sim = self._similarity_value(idx_a, idx_b)
        if sim <= 0.0:
            return 0.0
        if self._eta is None:
            return sim if sim >= self._lambda else 0.0
        return sim * (1.0 / (1.0 + math.exp(-self._eta * (sim - self._lambda))))

    def _similarity_base(self, a: LayoutPoint, b: LayoutPoint) -> float:
        return _similarity_base_points(a, b, self._similarity)

    @staticmethod
    def _format_label(value: float) -> str:
        if abs(value - round(value)) < 1e-6:
            return str(int(round(value)))
        return f"{value:.3f}"

    def _normalized_hues(self) -> list[float]:
        if not self._points:
            return []
        hues = [point.hue for point in self._points]
        min_hue = min(hues)
        max_hue = max(hues)
        if max_hue == min_hue:
            return [0.0 for _ in hues]
        scale = HUE / (max_hue - min_hue)
        normalized = []
        for hue in hues:
            value = (hue - min_hue) * scale
            if value >= HUE:
                value = HUE - 1e-9
            normalized.append(value)
        return normalized

    @staticmethod
    def _coerce_code(code: BitArray | Iterable[int]) -> tuple[BitArray, int]:
        if isinstance(code, BitArray):
            return code, code.count()
        try:
            length = len(code)  # type: ignore[arg-type]
            bits = code  # type: ignore[assignment]
        except TypeError:
            bits = list(code)
            length = len(bits)
        layout_code = BitArray(length)
        ones = 0
        for idx, bit in enumerate(bits):
            if bit:
                layout_code.set(idx, 1)
                ones += 1
        return layout_code, ones

    @staticmethod
    def _angle_similarity(angle_a: float, angle_b: float) -> float:
        diff = abs(angle_a - angle_b) % 360.0
        diff = min(diff, 360.0 - diff)
        return (math.cos(math.radians(diff)) + 1.0) / 2.0

    @staticmethod
    def _hue_to_rgb(hue: float) -> tuple[int, int, int]:
        import colorsys

        r, g, b = colorsys.hsv_to_rgb((hue % HUE) / HUE, 1.0, 1.0)
        return int(r * 255), int(g * 255), int(b * 255)
