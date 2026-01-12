from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
import math
import os
import random
from typing import Iterable, Mapping, Sequence

from damp.article_refs import (
    ENERGIES,
    ENERGY_LONG,
    ENERGY_SHORT,
    GPU_IMPLEMENTATION,
    LAYOUT_ALGORITHM,
    LAYOUT_COMPACTNESS,
    OPTIM_SIM_MATRIX,
    OPTIM_SUBSET,
    PAIR_SELECTION,
    PARALLEL_PROCESSING,
    QUALITY_ASSESS,
    SIMILARITY_MEASURES,
    SPARSE_BIT_VECTORS,
)
from damp.encoding.bitarray import BitArray
from damp.logging import log_event


@dataclass(frozen=True)
class LayoutPoint:
    code: BitArray
    label: float
    hue: float
    ones: int

_UNSET = object()
HUE = 360.0
_PARALLEL_SIM_MIN_PAIRS = 50_000
_SIM_POINTS: Sequence[LayoutPoint] | None = None
_SIM_MODE: str | None = None


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
                if (use_radius_override == 1) {
                    radius_sq = radius * radius;
                } else {
                    float dy = pos_a_y - pos_b_y;
                    float dx = pos_a_x - pos_b_x;
                    float dist = sqrt(dy * dy + dx * dx);
                    float rad = max(1.0, ceil(dist));
                    radius_sq = rad * rad;
                }
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
            log_event(
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
                log_event(
                    "layout.gpu.context_failed",
                    section=GPU_IMPLEMENTATION,
                    data={"error": str(exc)},
                )
                return None
            if getattr(ctx, "version_code", 0) < 410:
                log_event(
                    "layout.gpu.version_too_low",
                    section=GPU_IMPLEMENTATION,
                    data={"version_code": getattr(ctx, "version_code", 0)},
                )
                return None

        try:
            return cls(ctx, np, moderngl, points, similarity)
        except Exception as exc:
            log_event(
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

        if grid_size is None:
            target = len(self._points) * (1.0 + empty_ratio)
            grid_size = math.ceil(math.sqrt(target))
        if grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if grid_size * grid_size < len(self._points):
            raise ValueError("grid_size too small for number of points")

        self.height = grid_size
        self.width = grid_size
        log_event(
            "layout.init",
            section=LAYOUT_ALGORITHM,
            data={
                "points": self._point_count,
                "labels": len(self._codes),
                "seed": seed,
            },
        )
        log_event(
            "layout.grid",
            section=LAYOUT_COMPACTNESS,
            data={
                "grid_size": grid_size,
                "empty_ratio": empty_ratio,
            },
        )
        log_event(
            "layout.similarity",
            section=SIMILARITY_MEASURES,
            data={"similarity": self._similarity},
        )
        log_event(
            "layout.thresholds",
            section=ENERGIES,
            data={"lambda_threshold": self._lambda, "eta": self._eta},
        )
        log_event(
            "layout.precompute",
            section=OPTIM_SIM_MATRIX,
            data={"precompute_similarity": precompute_similarity},
        )
        log_event(
            "layout.precompute_limit",
            section=OPTIM_SUBSET,
            data={"max_precompute": max_precompute},
        )
        log_event(
            "layout.parallel",
            section=PARALLEL_PROCESSING,
            data={"parallel_workers": parallel_workers},
        )
        log_event(
            "layout.gpu.config",
            section=GPU_IMPLEMENTATION,
            data={"use_gpu": use_gpu},
        )
        if self._points:
            ones_values = [point.ones for point in self._points]
            log_event(
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
            log_event(
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
        self._place_points()

        self._distance_eps = 1e-6
        log_event(
            "layout.distance_eps",
            section=ENERGIES,
            data={"distance_eps": self._distance_eps},
        )
        self._gpu_engine: _GpuLayoutEngine | None = None
        self._gpu_positions_dirty = True
        if self._use_gpu:
            self._gpu_engine = _GpuLayoutEngine.create(self._points, self._similarity)
            if self._gpu_engine is None:
                self._use_gpu = False
        if self._use_gpu and self._gpu_engine is not None:
            log_event(
                "layout.gpu.enabled",
                section=GPU_IMPLEMENTATION,
                data={"status": "OpenGL 4.1"},
            )
        else:
            log_event(
                "layout.gpu.disabled",
                section=GPU_IMPLEMENTATION,
                data={"status": "cpu"},
            )
        self.last_steps = 0
        self._sim_base: list[list[float]] | None = None
        if precompute_similarity and self._point_count <= max_precompute:
            self._build_similarity_cache()

    def run(
        self,
        *,
        steps: int,
        pairs_per_step: int,
        pair_radius: int | None,
        mode: str = "long",
        local_radius: int | None = None,
        min_swap_ratio: float = 0.01,
        log_every: int | None = None,
        log_path: str = "layout",
        step_offset: int = 0,
        energy_radius: int | None = None,
        energy_check_every: int = 10,
        energy_delta: float = 1e-3,
        energy_patience: int = 3,
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
        if step_offset < 0:
            raise ValueError("step_offset must be >= 0")
        if energy_radius is not None and energy_radius <= 0:
            raise ValueError("energy_radius must be positive when set")
        if energy_check_every <= 0:
            raise ValueError("energy_check_every must be positive")
        if energy_delta < 0:
            raise ValueError("energy_delta must be >= 0")
        if energy_patience <= 0:
            raise ValueError("energy_patience must be positive")

        log_event(
            "layout.run.selection",
            section=PAIR_SELECTION,
            data={
                "pairs_per_step": pairs_per_step,
                "pair_radius": pair_radius,
            },
        )
        log_event(
            "layout.run.mode",
            section=ENERGY_LONG if mode == "long" else ENERGY_SHORT,
            data={
                "mode": mode,
                "local_radius": local_radius,
            },
        )
        log_event(
            "layout.run.settings",
            section=LAYOUT_ALGORITHM,
            data={
                "steps": steps,
                "min_swap_ratio": min_swap_ratio,
                "log_every": log_every,
                "step_offset": step_offset,
            },
        )
        log_event(
            "layout.run.energy_stop",
            section=QUALITY_ASSESS,
            data={
                "energy_radius": energy_radius,
                "energy_check_every": energy_check_every,
                "energy_delta": energy_delta,
                "energy_patience": energy_patience,
            },
        )
        total_swaps = 0
        min_swaps = 0
        if min_swap_ratio > 0:
            min_swaps = max(1, int(pairs_per_step * min_swap_ratio))
        steps_executed = 0
        prev_energy: float | None = None
        stable_checks = 0

        for step in range(steps):
            swaps = self.step(
                pairs_per_step=pairs_per_step,
                pair_radius=pair_radius,
                mode=mode,
                local_radius=local_radius,
            )
            total_swaps += swaps
            steps_executed = step + 1
            if log_every is not None and step % log_every == 0:
                from .visualize_layout import log_layout

                log_layout(self, path=log_path, step=step + step_offset)
            if energy_radius is not None and mode == "long":
                if step % energy_check_every == 0:
                    energy = self.average_local_energy(energy_radius)
                    if prev_energy is not None:
                        if abs(energy - prev_energy) <= energy_delta:
                            stable_checks += 1
                        else:
                            stable_checks = 0
                        if stable_checks >= energy_patience:
                            break
                    prev_energy = energy
            if min_swaps and swaps < min_swaps:
                break

        self.last_steps = steps_executed
        avg_swaps = (total_swaps / steps_executed) if steps_executed else 0.0
        log_event(
            "layout.run.done",
            section=LAYOUT_ALGORITHM,
            data={
                "steps": steps_executed,
                "total_swaps": total_swaps,
                "avg_swaps": avg_swaps,
            },
        )
        return total_swaps

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
        log_event(
            "layout.similarity_params",
            section=ENERGIES,
            data={"lambda_threshold": self._lambda, "eta": self._eta},
        )

    def step(
        self,
        *,
        pairs_per_step: int,
        pair_radius: int | None,
        mode: str,
        local_radius: int | None,
    ) -> int:
        swaps: list[tuple[int, int, int, int]] = []
        pairs = [self._sample_pair(pair_radius) for _ in range(pairs_per_step)]

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
                        if energy_s < energy_c:
                            swaps.append((y1, x1, y2, x2))
                    else:
                        if energy_s > energy_c:
                            swaps.append((y1, x1, y2, x2))
                return self._apply_swaps(swaps)
            self._gpu_engine = None

        for y1, x1, y2, x2 in pairs:
            if self._should_swap(y1, x1, y2, x2, mode=mode, local_radius=local_radius):
                swaps.append((y1, x1, y2, x2))
        return self._apply_swaps(swaps)

    def positions(self) -> list[tuple[int, int]]:
        return list(self._positions)

    def labels(self) -> list[str]:
        return list(self._labels)

    def values(self) -> list[str]:
        return list(self._values)

    def colors_rgb(self) -> list[tuple[int, int, int]]:
        hues = self._normalized_hues()
        return [self._hue_to_rgb(hue) for hue in hues]

    def render_image(self) -> "numpy.ndarray":
        import numpy as np

        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        hues = self._normalized_hues()
        for idx, (y, x) in enumerate(self._positions):
            image[y, x] = self._hue_to_rgb(hues[idx])
        return image

    def average_local_energy(self, radius: int) -> float:
        if radius <= 0:
            raise ValueError("radius must be positive")
        if self._point_count == 0:
            return 0.0
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
                log_event(
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

        for idx in range(self._point_count):
            cy = pos_y[idx]
            cx = pos_x[idx]
            energy = 0.0
            for j in range(self._point_count):
                if j == idx:
                    continue
                dy = cy - pos_y[j]
                dx = cx - pos_x[j]
                dist_sq = dy * dy + dx * dx
                if dist_sq > radius_sq:
                    continue
                sim = sim_lambda(idx, j)
                if sim <= 0.0:
                    continue
                energy += sim / (dist_sq + self._distance_eps)
            energies.append(energy)

        emax = max(energies)
        if emax <= 0.0:
            return 0.0
        avg_energy = sum(e / emax for e in energies) / len(energies)
        log_event(
            "layout.energy.average",
            section=QUALITY_ASSESS,
            data={"radius": radius, "energy": avg_energy, "mode": "cpu"},
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

    def _should_swap(
        self,
        y1: int,
        x1: int,
        y2: int,
        x2: int,
        *,
        mode: str,
        local_radius: int | None,
    ) -> bool:
        idx_a = self.grid[y1][x1]
        idx_b = self.grid[y2][x2]

        if idx_a is None and idx_b is None:
            return False
        if idx_a is None:
            idx_a, idx_b = idx_b, idx_a
            y1, x1, y2, x2 = y2, x2, y1, x1

        if mode == "long":
            energy_c, energy_s = self._pair_energy_long(
                idx_a, (y1, x1), idx_b, (y2, x2)
            )
            return energy_s < energy_c

        radius = local_radius
        if radius is None:
            dist_sq = (y1 - y2) ** 2 + (x1 - x2) ** 2
            radius = max(1, int(math.ceil(math.sqrt(dist_sq))))
        energy_c, energy_s = self._pair_energy_short(
            idx_a, (y1, x1), idx_b, (y2, x2), radius=radius
        )
        return energy_s > energy_c

    def _pair_energy_long(
        self,
        idx_a: int,
        pos_a: tuple[int, int],
        idx_b: int | None,
        pos_b: tuple[int, int],
    ) -> tuple[float, float]:
        pos_a_y, pos_a_x = pos_a
        pos_b_y, pos_b_x = pos_b
        pos_y = self._pos_y
        pos_x = self._pos_x
        sim_lambda = self._sim_lambda_idx
        energy_c = 0.0
        energy_s = 0.0
        for idx in range(self._point_count):
            if idx == idx_a or idx == idx_b:
                continue
            s1 = sim_lambda(idx, idx_a)
            s2 = sim_lambda(idx, idx_b) if idx_b is not None else 0.0
            if s1 == 0.0 and s2 == 0.0:
                continue
            dy1 = pos_a_y - pos_y[idx]
            dx1 = pos_a_x - pos_x[idx]
            dy2 = pos_b_y - pos_y[idx]
            dx2 = pos_b_x - pos_x[idx]
            d1 = dy1 * dy1 + dx1 * dx1
            d2 = dy2 * dy2 + dx2 * dx2
            energy_c += s1 * d1 + s2 * d2
            energy_s += s2 * d1 + s1 * d2
        return energy_c, energy_s

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

        pos_y = self._pos_y
        pos_x = self._pos_x
        sim_lambda = self._sim_lambda_idx
        energy_c = 0.0
        energy_s = 0.0
        for idx in range(self._point_count):
            if idx == idx_a or idx == idx_b:
                continue
            dyc = pos_y[idx] - cy
            dxc = pos_x[idx] - cx
            if dyc * dyc + dxc * dxc > radius_sq:
                continue
            s1 = sim_lambda(idx, idx_a)
            s2 = sim_lambda(idx, idx_b) if idx_b is not None else 0.0
            if s1 == 0.0 and s2 == 0.0:
                continue
            dy1 = pos_a[0] - pos_y[idx]
            dx1 = pos_a[1] - pos_x[idx]
            dy2 = pos_b[0] - pos_y[idx]
            dx2 = pos_b[1] - pos_x[idx]
            d1 = dy1 * dy1 + dx1 * dx1
            d2 = dy2 * dy2 + dx2 * dx2
            energy_c += s1 / (d1 + self._distance_eps) + s2 / (
                d2 + self._distance_eps
            )
            energy_s += s2 / (d1 + self._distance_eps) + s1 / (
                d2 + self._distance_eps
            )
        return energy_c, energy_s

    def _apply_swaps(self, swaps: Iterable[tuple[int, int, int, int]]) -> int:
        used: set[tuple[int, int]] = set()
        applied = 0
        for y1, x1, y2, x2 in swaps:
            if (y1, x1) in used or (y2, x2) in used:
                continue
            idx_a = self.grid[y1][x1]
            idx_b = self.grid[y2][x2]
            self.grid[y1][x1], self.grid[y2][x2] = idx_b, idx_a
            if idx_a is not None:
                self._positions[idx_a] = (y2, x2)
                self._pos_y[idx_a] = y2
                self._pos_x[idx_a] = x2
            if idx_b is not None:
                self._positions[idx_b] = (y1, x1)
                self._pos_y[idx_b] = y1
                self._pos_x[idx_b] = x1
            used.add((y1, x1))
            used.add((y2, x2))
            applied += 1
        if applied:
            self._gpu_positions_dirty = True
        return applied

    def _build_similarity_cache(self) -> None:
        point_count = self._point_count
        workers = self._similarity_worker_count()
        log_event(
            "layout.sim_cache.start",
            section=OPTIM_SIM_MATRIX,
            data={"point_count": point_count},
        )
        log_event(
            "layout.sim_cache.similarity",
            section=SIMILARITY_MEASURES,
            data={"similarity": self._similarity},
        )
        log_event(
            "layout.sim_cache.parallel",
            section=PARALLEL_PROCESSING,
            data={"workers": workers},
        )
        log_event(
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
                log_event(
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
            log_event(
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

    def _similarity_worker_count(self) -> int:
        if self._parallel_workers is not None:
            if self._parallel_workers <= 1:
                return 1
            return min(self._parallel_workers, self._point_count)
        cpu_count = os.cpu_count() or 1
        total_pairs = self._point_count * (self._point_count - 1) // 2
        if cpu_count < 2 or total_pairs < _PARALLEL_SIM_MIN_PAIRS:
            return 1
        log_event(
            "layout.parallel.cpu_count",
            section=PARALLEL_PROCESSING,
            data={"cpu_count": cpu_count},
        )
        return min(cpu_count, self._point_count)

    def _sim_lambda_idx(self, idx_a: int, idx_b: int | None) -> float:
        if idx_b is None:
            return 0.0
        if self._sim_base is not None:
            sim = self._sim_base[idx_a][idx_b]
        else:
            sim = self._similarity_base(self._points[idx_a], self._points[idx_b])
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
