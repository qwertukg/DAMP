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

import numpy as np

from damp.article_refs import (
    ENERGY_CALC,
    ENERGY_LONG,
    ENERGY_SHORT,
    GPU_IMPLEMENTATION,
    LAID_OUT_STRUCTURE,
    LAYOUT_ALGORITHM,
    LAYOUT_DISCRETE,
    LAYOUT_PARAMETERS,
    MAPPING_CONTINUITY,
    OPTIM_ENERGY,
    OPTIM_FIRST_POINT,
    OPTIM_PAIR_SELECTION,
    OPTIM_SIM_MATRIX,
    OPTIM_SUBSET,
    OUTSIDE_ARTICLE,
    PAIR_SELECTION,
    PARALLEL_PROCESSING,
    POINT_ENERGY,
    POINT_EXCHANGE,
    QUALITY_ASSESS,
    SIMILARITY_MEASURES,
    SIMILARITY_THRESHOLD,
    SPACE_FILLING,
    SPARSE_BIT_VECTORS,
)
from damp.encoding.bitarray import BitArray
from damp.encoding.damp_encoder import EncoderCodeColorizer
from damp.layout.pair_energy_gpu import TorchPairEnergyEngine
from damp.layout.pair_sampler import CompiledPairSampler
from damp.layout.pair_sign_kernel import PairSignKernel
from damp.layout.potential_bounds import PotentialBounds
from damp.layout.reduction_replica import ReductionReplica
from damp.layout.code_table import LayoutCodeTable
from damp.layout.device_step import DeviceLongStep, DeviceShortStep
from damp.layout.payload import LayoutEncodedCode, LayoutJsonWriter
from damp.layout.sparse_long_energy import SimilarityRows, SparseLongEnergy
from damp.layout.window_energy_kernel import WindowEnergyKernel
from damp.logging import LOGGER


@dataclass(frozen=True)
class LayoutPoint:
    code: BitArray
    label: str
    hue: float
    ones: int


_RADIUS_SCHEDULES = ("swaps", "linear", "geometric")
# Выше этого числа точек отказ расчёта энергий на torch — ошибка раскладки: запасные пути не справятся.
_TORCH_FALLBACK_POINT_LIMIT = 200_000
# Шаг на устройстве сверяет энергии первых пар с исходным проходом движка на первом шаге и раз в столько шагов.
_DEVICE_STEP_VERIFY_EVERY = 1000
# Строк таблицы кодов на один блок расчёта псевдоцветов (§5.14): биты блока — 512 байт на код.
_COLOR_BLOCK_ROWS = 262_144


@dataclass(frozen=True)
class AdaptiveLayoutConfig:
    """Радиус выбора пар убывает от d/2 к 1, порог λ растёт ступенями (§7.1.7).

    radius_schedule = "swaps" — по статье: λ и r меняются, когда число обменов за шаг падает
    до swap_ratio_trigger от начального значения этапа (§7.1.7); r умножается на radius_factor.
    "linear" и "geometric" — расширение статьи: r убывает по номеру шага независимо от обменов.
    """

    start_radius: int
    end_radius: int = 1
    swap_ratio_trigger: float = 0.01
    lambda_step: float | None = None
    lambda_max: float = 1.0
    radius_schedule: str = "swaps"
    # Во сколько раз уменьшается r при срабатывании триггера: статья говорит только «постепенно» (§7.1.7).
    radius_factor: float = 0.5
    # Число обменов за шаг усредняется по окну шагов («число обменов в единицу времени», §7.1.7);
    # начальное значение этапа — среднее по первому окну после смены λ и r.
    swap_window: int = 1
    # Отжиг (расширение статьи): пока радиус пар лежит в [temperature_radius_min, temperature_radius_max],
    # невыгодные обмены принимаются с вероятностью exp(Δ / (T·rms|Δ|)); вне окна правило жадное (§5.6).
    temperature: float = 0.0
    temperature_radius_max: int = 0
    temperature_radius_min: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.lambda_max <= 1.0:
            raise ValueError("lambda_max must be in (0, 1]")
        if self.temperature < 0.0:
            raise ValueError("temperature must be >= 0")
        if self.radius_schedule not in _RADIUS_SCHEDULES:
            raise ValueError("radius_schedule must be 'swaps', 'linear' or 'geometric'")
        if not 0.0 < self.radius_factor < 1.0:
            raise ValueError("radius_factor must be in (0, 1)")
        if self.swap_window <= 0:
            raise ValueError("swap_window must be positive")

    def temperature_for_radius(self, radius: int) -> float:
        if self.temperature <= 0.0:
            return 0.0
        return self.temperature if self.temperature_radius_min <= radius <= self.temperature_radius_max else 0.0


@dataclass(frozen=True)
class ShortLayoutOptimConfig:
    energy_radius: int
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
# Псевдоцвета по умолчанию: младший бит красный, старший фиолетовый (§5.14).
_DEFAULT_PSEUDO_COLOR_MIN_HUE = 0.0
_DEFAULT_PSEUDO_COLOR_MAX_HUE = 300.0
_PARALLEL_SIM_MIN_PAIRS = 50_000
_SIM_POINTS: Sequence[LayoutPoint] | None = None
_SIM_MODE: str | None = None
_VISUAL_ENERGY_POINT_LIMIT = 5000
_VISUAL_POINT_LIMIT = 200000
# §5.5: d евклидово; §5.9.2 допускает суммы квадратов без корня.
_DISTANCE_MODES = ("squared", "euclidean")
# Ядро дальней энергии: "distance" — φ = Σ s·d (§5.5.1, минимизация); "gauss" — расширение статьи:
# φ = Σ s·exp(−2d²/r²) по окрестности радиуса r(t) = радиусу выбора пар (максимизация),
# ограниченное ядро не даёт далёким парам навязывать одинаковую закрутку соседним кластерам.
_LONG_KERNELS = ("distance", "gauss")
# Ядро окрестности при вычислении по §5.5.2: inverse — s/d, gauss — s·exp(−2d²/r²).
_NEIGHBOURHOOD_KERNELS = ("inverse", "gauss")
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
        uniform int euclidean_distance;
        uniform int kernel_mode;
        uniform float kernel_width;
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
                // §5.5.2: r >= |p1 - p2|; +1 гарантирует, что окрестность
                // содержит точки помимо самой пары.
                float base_rad = ceil(dist) + 1.0;
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
                float d1sq = dy1 * dy1 + dx1 * dx1;
                float d2sq = dy2 * dy2 + dx2 * dx2;
                float d1 = d1sq;
                float d2 = d2sq;
                if (euclidean_distance == 1) {
                    // §5.5: d1, d2 — евклидовы расстояния; иначе квадраты (§5.9.2).
                    d1 = sqrt(d1sq);
                    d2 = sqrt(d2sq);
                }
                if (mode == 0) {
                    energy_c += s1 * d1 + s2 * d2;
                    energy_s += s2 * d1 + s1 * d2;
                } else {
                    float w1;
                    float w2;
                    if (kernel_mode == 1) {
                        // Гауссово ядро окрестности радиуса r, σ = kernel_width·r (расширение статьи).
                        float sig2 = 2.0 * kernel_width * kernel_width * radius_sq;
                        w1 = exp(-d1sq / sig2);
                        w2 = exp(-d2sq / sig2);
                    } else {
                        w1 = 1.0 / (d1 + distance_eps);
                        w2 = 1.0 / (d2 + distance_eps);
                    }
                    energy_c += s1 * w1 + s2 * w2;
                    energy_s += s2 * w1 + s1 * w2;
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
                // §5.7: E(c, r) = Σ sim_λ / d_c, d_c евклидово (как в CPU-пути).
                energy += sim / (sqrt(dist_sq) + distance_eps);
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
        euclidean: bool = False,
        kernel: str = "inverse",
        kernel_width: float = 0.5,
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
        self._pair_prog["euclidean_distance"].value = 1 if euclidean else 0
        self._pair_prog["kernel_mode"].value = {"inverse": 0, "gauss": 1}[kernel]
        self._pair_prog["kernel_width"].value = float(kernel_width)

        self._pair_fbo.use()
        self._ctx.viewport = (0, 0, pair_count, 1)
        self._pair_vao.render(mode=self._gl.TRIANGLES, vertices=3)

        data = self._pair_out_tex.read()
        raw = self._np.frombuffer(data, dtype=self._np.float32).reshape(
            (self._pair_capacity, 4)
        )
        energies = raw[:pair_count]
        return [(float(row[0]), float(row[1])) for row in energies]

    def point_energies(
        self,
        radius: int,
        *,
        lambda_threshold: float,
        eta: float | None,
        distance_eps: float,
    ) -> "numpy.ndarray | None":
        """Энергии E(c, r) всех точек V одним проходом шейдера (§5.7, §5.12)."""
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
        return self._np.frombuffer(data, dtype=self._np.float32).copy()


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

    def _is_oom(self, exc: Exception) -> bool:
        return "out of memory" in str(exc).lower()

    def _is_mps(self) -> bool:
        return getattr(self._device, "type", str(self._device)).lower() == "mps"

    def release_memory_on_oom(self, exc: Exception, *, stage: str) -> bool:
        if not self._is_oom(exc):
            return False
        if not self._is_mps():
            return False
        self.reset_subset()
        try:
            self._torch.mps.empty_cache()
        except Exception:
            return False
        LOGGER.event(
            "layout.gpu.mps.empty_cache",
            section=GPU_IMPLEMENTATION,
            data={
                "device": str(self._device),
                "stage": stage,
                "error": str(exc),
            },
        )
        return True

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
        euclidean: bool = False,
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
            dist1_sq = dy1 * dy1 + dx1 * dx1
            dist2_sq = dy2 * dy2 + dx2 * dx2
            if euclidean:
                # §5.5.1: φ = Σ(s1·d1 + s2·d2) с евклидовыми d; иначе d² (§5.9.2).
                dist1 = torch.sqrt(dist1_sq)
                dist2 = torch.sqrt(dist2_sq)
            else:
                dist1 = dist1_sq
                dist2 = dist2_sq

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
        euclidean: bool = False,
        kernel: str = "inverse",
        kernel_width: float = 0.5,
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
            if kernel == "gauss":
                # Гауссово ядро окрестности радиуса r, σ = kernel_width·r (расширение статьи).
                sig2 = 2.0 * kernel_width * kernel_width * radius_sq
                w1 = torch.exp(-dist1 / sig2)
                w2 = torch.exp(-dist2 / sig2)
            else:
                if euclidean:
                    # §5.5.2: φ = Σ(s1/d1 + s2/d2) с евклидовыми d; иначе d² (§5.9.2).
                    dist1 = torch.sqrt(dist1)
                    dist2 = torch.sqrt(dist2)
                w1 = 1.0 / (dist1 + distance_eps)
                w2 = 1.0 / (dist2 + distance_eps)

            s1_masked = torch.where(mask_radius, s1, torch.zeros_like(s1))
            s2_masked = torch.where(mask_radius, s2, torch.zeros_like(s2))

            energy_c = torch.sum(s1_masked * w1 + s2_masked * w2, dim=1)
            energy_s = torch.sum(s2_masked * w1 + s1_masked * w2, dim=1)
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


class _UniformDraws:
    """Числа random.Random.random() пачкой (отбор обменов при отжиге берёт по числу на невыгодную пару).

    random.Random и numpy.random.RandomState — один и тот же генератор MT19937 с одной формулой числа:
    два 32-битных слова a, b дают (⌊a/2⁵⌋·2²⁶ + ⌊b/2⁶⌋)/2⁵³. Состояние генератора Python передаётся
    генератору numpy, тот выдаёт count чисел массивом, и состояние возвращается обратно: числа и состояние
    те же, что после count вызовов random(). Свойство проверяется один раз; если в установленных версиях
    оно не выполняется, числа берутся по одному.
    """

    # С этого числа запросов пачка дешевле вызовов по одному (передача состояния стоит ≈ 80 мкс).
    MIN_COUNT = 512
    _VERIFIED: bool | None = None
    _LEGACY: "np.random.RandomState | None" = None

    @classmethod
    def _bulk(cls, rng: random.Random, count: int) -> "np.ndarray":
        if cls._LEGACY is None:
            cls._LEGACY = np.random.RandomState(0)
        legacy = cls._LEGACY
        version, state, gauss = rng.getstate()
        legacy.set_state(("MT19937", np.array(state[:-1], dtype=np.uint32), state[-1]))
        values = legacy.random_sample(count)
        key, position = legacy.get_state()[1:3]
        rng.setstate((version, tuple(key.tolist()) + (int(position),), gauss))
        return values

    @classmethod
    def _verified(cls) -> bool:
        if cls._VERIFIED is None:
            matched = True
            try:
                one_by_one = random.Random(20240229)
                in_bulk = random.Random(20240229)
                for count in (1, 7, 623, 624, 625, 1000, 4096):
                    expected = np.array([one_by_one.random() for _ in range(count)])
                    matched = matched and np.array_equal(expected, cls._bulk(in_bulk, count))
                matched = matched and one_by_one.getstate() == in_bulk.getstate()
            except Exception:  # noqa: BLE001
                matched = False
            cls._VERIFIED = bool(matched)
            LOGGER.event(
                "layout.swap.bulk_draws",
                section=OUTSIDE_ARTICLE,
                data={
                    "bulk": cls._VERIFIED,
                    "note": "числа генератора для отбора обменов берутся пачкой: те же числа и то же состояние "
                    "генератора, что при вызовах по одному",
                },
            )
        return cls._VERIFIED

    @classmethod
    def take(cls, rng: random.Random, count: int) -> "np.ndarray":
        """count чисел rng.random() массивом float64; состояние rng — как после count вызовов."""
        if count >= cls.MIN_COUNT and type(rng) is random.Random and cls._verified():
            return cls._bulk(rng, count)
        draw = rng.random
        return np.array([draw() for _ in range(count)])


@dataclass(frozen=True)
class _AdaptiveUpdate:
    """Итог шага расписания §7.1.7: доля обменов этапа и смена параметров по триггеру."""

    new_lambda: float | None
    swap_ratio: float | None
    swap_rate: float | None
    stage_initial: float | None
    triggered: bool
    radius: int


class _AdaptiveLayoutState:
    """Расписание радиуса выбора пар r и порога λ (§7.1.7)."""

    _LAMBDA_EPS = 1e-12

    def __init__(self, config: AdaptiveLayoutConfig, total_steps: int) -> None:
        self._config = config
        self._total_steps = max(1, total_steps)
        self._radius_cap = max(config.start_radius, config.end_radius)
        self._window: deque[int] = deque(maxlen=config.swap_window)
        self._stage_initial: float | None = None
        self._trigger_count = 0

    @property
    def trigger_count(self) -> int:
        return self._trigger_count

    @property
    def by_swaps(self) -> bool:
        return self._config.radius_schedule == "swaps"

    def _base_radius_for_step(self, step: int) -> int:
        if self.by_swaps:
            # §7.1.7: r меняется только по триггеру обменов, между срабатываниями постоянен.
            return self._config.start_radius
        if self._total_steps <= 1:
            return self._config.end_radius
        progress = step / max(1, self._total_steps - 1)
        if self._config.radius_schedule == "geometric":
            ratio = self._config.end_radius / self._config.start_radius
            decayed = self._config.start_radius * (ratio ** progress)
        else:
            decayed = self._config.start_radius - (
                (self._config.start_radius - self._config.end_radius) * progress
            )
        return max(self._config.end_radius, int(round(decayed)))

    def radius_for_step(self, step: int) -> int:
        radius = self._base_radius_for_step(step)
        return min(radius, self._radius_cap)

    def schedule_complete(self, current_lambda: float) -> bool:
        """По статье остановка по числу обменов (§5.6) имеет смысл после того, как r = 1 и λ = λ_max."""
        if not self.by_swaps:
            return True
        lambda_done = (
            self._config.lambda_step is None
            or current_lambda >= self._config.lambda_max - self._LAMBDA_EPS
        )
        return self._radius_cap <= self._config.end_radius and lambda_done

    def update_after_step(
        self, *, step: int, swaps: int, current_lambda: float
    ) -> _AdaptiveUpdate:
        self._window.append(swaps)
        if len(self._window) < self._window.maxlen:
            return _AdaptiveUpdate(None, None, None, self._stage_initial, False, self.radius_for_step(step))
        rate = sum(self._window) / len(self._window)
        if self._stage_initial is None:
            # Начальное число обменов этапа — первое окно после старта или смены λ и r (§7.1.7).
            self._stage_initial = max(1.0, rate)
        stage_initial = self._stage_initial
        swap_ratio = rate / stage_initial
        # После r = 1 и λ = λ_max менять нечего: дальше действует только остановка по обменам (§5.6).
        triggered = (
            swap_ratio <= self._config.swap_ratio_trigger
            and not (self.by_swaps and self.schedule_complete(current_lambda))
        )
        new_lambda = self._next_stage(step, current_lambda) if triggered else None
        return _AdaptiveUpdate(
            new_lambda=new_lambda,
            swap_ratio=swap_ratio,
            swap_rate=rate,
            stage_initial=stage_initial,
            triggered=triggered,
            radius=self.radius_for_step(step + 1),
        )

    def force_next_stage(self, *, step: int, current_lambda: float) -> _AdaptiveUpdate:
        """Обмены упали ниже порога остановки (§5.6) раньше, чем r = 1 и λ = λ_max: следующий этап §7.1.7."""
        new_lambda = self._next_stage(step, current_lambda)
        return _AdaptiveUpdate(
            new_lambda=new_lambda,
            swap_ratio=None,
            swap_rate=None,
            stage_initial=None,
            triggered=True,
            radius=self.radius_for_step(step + 1),
        )

    def _next_stage(self, step: int, current_lambda: float) -> float | None:
        self._trigger_count += 1
        # Новый этап: начальное число обменов берётся заново по следующему окну.
        self._stage_initial = None
        self._window.clear()
        current_radius = self._radius_cap if self.by_swaps else self._base_radius_for_step(step)
        reduced_cap = max(self._config.end_radius, int(current_radius * self._config.radius_factor))
        if reduced_cap < self._radius_cap:
            self._radius_cap = reduced_cap
        if self._config.lambda_step is None:
            return None
        # §7.1.7: λ повышается ступенями, но не выше заданного максимума.
        capped = min(self._config.lambda_max, current_lambda + self._config.lambda_step)
        return capped if capped > current_lambda else None


@dataclass
class _SwapStallMonitor:
    min_ratio: float
    window: int
    history: deque[float] = field(init=False)

    def __post_init__(self) -> None:
        self.history = deque(maxlen=self.window)

    def reset(self) -> None:
        self.history.clear()

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

    # Пока пар без второй ячейки не больше этого числа, попытка выбора проверяется поэлементно (§5.3):
    # на таких массивах вызовы numpy стоят больше самой проверки.
    _SAMPLE_SCALAR_LIMIT = 48
    # Скомпилированная выборка пар сверяется с выборкой numpy на стольких первых вызовах каждой фазы.
    _SAMPLER_VERIFY_FIRST = 3
    # Можно ли брать смещения dy и dx одной попытки одним запросом к генератору (проверяется один раз).
    _MERGED_OFFSET_DRAWS: bool | None = None

    @classmethod
    def _merged_offset_draws(cls) -> bool:
        """Один запрос 2k смещений даёт те же числа, что два запроса по k, и оставляет генератор в том же состоянии.

        Генератор выдаёт ограниченные целые по одному, поэтому разбиение запроса на части не меняет поток.
        Свойство проверяется на отдельном генераторе: если в установленной версии numpy оно не выполняется,
        смещения запрашиваются двумя вызовами, как раньше.
        """
        if cls._MERGED_OFFSET_DRAWS is None:
            matched = True
            for radius, count in ((1, 1), (1, 7), (2, 1000), (3, 64), (50, 333), (160, 5)):
                together = np.random.default_rng(987654321)
                apart = np.random.default_rng(987654321)
                merged = together.integers(-radius, radius + 1, 2 * count)
                first = apart.integers(-radius, radius + 1, count)
                second = apart.integers(-radius, radius + 1, count)
                matched = (
                    matched
                    and np.array_equal(merged[:count], first)
                    and np.array_equal(merged[count:], second)
                    and together.bit_generator.state == apart.bit_generator.state
                )
            cls._MERGED_OFFSET_DRAWS = bool(matched)
            LOGGER.event(
                "layout.pair_sampling.merged_draws",
                section=PAIR_SELECTION,
                data={
                    "merged": cls._MERGED_OFFSET_DRAWS,
                    "note": "смещения dy, dx второй точки пары (§5.3) одним запросом к генератору: числа и состояние "
                    "генератора те же, что при двух запросах",
                },
            )
        return cls._MERGED_OFFSET_DRAWS

    def __init__(
        self,
        codes: "Mapping[float, Sequence[BitArray]] | LayoutCodeTable",
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
        long_subset_size: int | None = None,
        long_subset_refresh: int = 50,
        colorizer: EncoderCodeColorizer | None = None,
        long_distance: str = "squared",
        short_distance: str = "squared",
        long_kernel: str = "distance",
        long_kernel_radius_min: int = 3,
        long_kernel_width: float = 0.5,
        component_period: float | None = None,
        long_subset_min_neighbours: int = 50,
        sparse_long: bool = True,
        sparse_max_edges: int = 400_000_000,
        sparse_memory_fraction: float = 0.5,
        sparse_buffer_bytes: int = 256 * 1024 * 1024,
        sparse_entry_budget: int = 4_000_000,
        sparse_screen_min_chunks: int = 1,
        sparse_verify_every: int = 2000,
        sparse_verify_chunks: int = 1,
        pair_kernel: bool = True,
        pair_kernel_segment: int = 16,
        compiled_sampler: bool = True,
        sampler_verify_every: int = 1000,
        potential_bounds: bool = True,
        potential_bounds_grid: int = 8,
        reduction_replica: bool = True,
        replica_max_bytes: int = 1 << 30,
        fused_energy: bool = True,
        fused_table_max_bytes: int = 6 * 1024 * 1024 * 1024,
        fused_table_memory_fraction: float = 0.7,
        fused_block_elements: int = 40_000_000,
        fused_verify_every: int = 2000,
        fused_similarity_table: bool = True,
        neighbourhood_sample_side: int | None = None,
        enforce_pair_radius: bool = False,
        neighbourhood_radius_min: int | None = None,
        initial_cells: "np.ndarray | None" = None,
        device_step: bool = False,
        device_short_step: bool = False,
    ) -> None:
        if codes is None or len(codes) == 0:
            raise ValueError("codes must be non-empty")
        if neighbourhood_sample_side is not None and neighbourhood_sample_side <= 0:
            raise ValueError("neighbourhood_sample_side must be positive when set")
        if neighbourhood_radius_min is not None and neighbourhood_radius_min <= 0:
            raise ValueError("neighbourhood_radius_min must be positive when set")
        if fused_table_max_bytes < 0 or not 0.0 < fused_table_memory_fraction <= 1.0:
            raise ValueError("fused_table_max_bytes must be >= 0 and fused_table_memory_fraction in (0, 1]")
        if fused_block_elements <= 0 or fused_verify_every < 0:
            raise ValueError("fused_block_elements must be positive and fused_verify_every >= 0")
        if sparse_max_edges < 0 or not 0.0 < sparse_memory_fraction <= 1.0:
            raise ValueError("sparse_max_edges must be >= 0 and sparse_memory_fraction in (0, 1]")
        if sparse_buffer_bytes <= 0 or sparse_entry_budget <= 0:
            raise ValueError("sparse_buffer_bytes and sparse_entry_budget must be positive")
        if sparse_screen_min_chunks <= 0 or sparse_verify_every < 0:
            raise ValueError("sparse_screen_min_chunks must be positive and sparse_verify_every >= 0")
        if sparse_verify_chunks < 0 or pair_kernel_segment <= 0 or sampler_verify_every < 0:
            raise ValueError(
                "sparse_verify_chunks and sampler_verify_every must be >= 0, pair_kernel_segment must be positive"
            )
        if potential_bounds_grid <= 0:
            raise ValueError("potential_bounds_grid must be positive")
        if component_period is not None and component_period <= 0.0:
            raise ValueError("component_period must be positive when set")
        if long_kernel not in _LONG_KERNELS:
            raise ValueError("long_kernel must be 'distance' or 'gauss'")
        if long_kernel_radius_min <= 0:
            raise ValueError("long_kernel_radius_min must be positive")
        if long_kernel_width <= 0.0:
            raise ValueError("long_kernel_width must be positive")
        if empty_ratio < 0:
            raise ValueError("empty_ratio must be >= 0")
        if similarity not in ("cosine", "jaccard"):
            raise ValueError("similarity must be 'cosine' or 'jaccard'")
        if long_distance not in _DISTANCE_MODES or short_distance not in _DISTANCE_MODES:
            raise ValueError("long_distance and short_distance must be 'squared' or 'euclidean'")
        if not 0 <= lambda_threshold <= 1:
            raise ValueError("lambda_threshold must be in [0, 1]")
        self._validate_eta(eta)
        if max_precompute <= 0:
            raise ValueError("max_precompute must be positive")
        if parallel_workers is not None and parallel_workers < 0:
            raise ValueError("parallel_workers must be >= 0")

        # Коды точек хранятся таблицей массивов (упакованные биты, hue, подписи): на ней работают расчёт энергий,
        # цвета и запись раскладки. Точки объектами (_points), подписи (_labels) и строки кодов (_values)
        # строятся из таблицы при первом обращении — большим раскладкам они не нужны и в память не помещаются.
        self._points_cache: list[LayoutPoint] | None = None
        self._labels_cache: list[str] | None = None
        self._values_cache: list[str] | None = None
        self._code_matrix_cache: "np.ndarray | None" = None
        if isinstance(codes, LayoutCodeTable):
            self._table = codes
        else:
            points: list[LayoutPoint] = []
            for hue, label_codes in codes.items():
                hue_value = float(hue)
                for code_entry in label_codes:
                    layout_code, ones, label_text = self._coerce_code_entry(
                        code_entry,
                        default_label=hue_value,
                    )
                    points.append(
                        LayoutPoint(
                            code=layout_code,
                            label=label_text,
                            hue=hue_value,
                            ones=ones,
                        )
                    )
            self._points_cache = points
            self._table = LayoutCodeTable.from_codes(
                [point.code for point in points],
                hues=[point.hue for point in points],
                labels=[point.label for point in points],
            )
        self._point_count = self._table.count
        self._similarity = similarity
        self._long_distance = long_distance
        self._short_distance = short_distance
        self._long_kernel = long_kernel
        self._long_kernel_radius_min = int(long_kernel_radius_min)
        self._long_kernel_width = float(long_kernel_width)
        # Период известной компоненты стимула (угол) для диагностической раскраски хода раскладки (рис. 16);
        # в самой раскладке компонента не участвует.
        self._component_period = None if component_period is None else float(component_period)
        self._lambda = lambda_threshold
        self._eta = eta
        self._colorizer = colorizer or EncoderCodeColorizer(
            min_hue=_DEFAULT_PSEUDO_COLOR_MIN_HUE,
            max_hue=_DEFAULT_PSEUDO_COLOR_MAX_HUE,
        )
        self._colors_rgb_cache: tuple[tuple[int, int, int], ...] | None = None
        self._rng = random.Random(seed)
        # Векторная выборка пар (§5.3, §5.9.1, §5.9.3) — отдельный генератор numpy с тем же seed.
        self._np_rng = np.random.default_rng(seed)
        self._parallel_workers = parallel_workers
        self._use_gpu = use_gpu
        self._long_subset_size = (
            None if long_subset_size is None else int(long_subset_size)
        )
        if self._long_subset_size is not None and self._long_subset_size <= 0:
            raise ValueError("long_subset_size must be positive when set")
        self._long_subset_refresh = max(1, int(long_subset_refresh))
        if long_subset_min_neighbours <= 0:
            raise ValueError("long_subset_min_neighbours must be positive")
        self._long_subset_min_neighbours = int(long_subset_min_neighbours)
        # Выборка ячеек окрестности гауссова ядра дальней фазы (§5.9.4): число слоёв выборки по стороне окна;
        # None — окрестность считается целиком. Свой генератор: остальные потоки случайных чисел не меняются.
        self._neighbourhood_sample_side = (
            None if neighbourhood_sample_side is None else int(neighbourhood_sample_side)
        )
        self._sample_rng = np.random.default_rng([int(seed), 1])
        self._disc_offsets_cache: dict[int, tuple["np.ndarray", "np.ndarray"]] = {}
        # Шаг дальней фазы целиком на устройстве (большие раскладки): расстановка живёт на устройстве,
        # массивы хоста догоняют её при первом обращении (_pull_device).
        self._device_step_enabled = bool(device_step)
        self._device_seed = int(seed)
        self._device_state: DeviceLongStep | None = None
        self._device_ahead = False
        self._device_steps = 0
        # Шаг ближней фазы режима статьи на устройстве (большие раскладки): расстановка и ядро окна
        # WindowEnergyKernel; им же считаются энергии точек (§5.7). _device_owner — чья расстановка новее хоста.
        self._device_short_enabled = bool(device_short_step)
        self._device_short: DeviceShortStep | None = None
        self._device_owner: DeviceLongStep | None = None
        self._device_weights_step: int | None = None
        self._device_weights_key: tuple[float, float] | None = None
        self._device_energies_verified = False
        # True — пары, вторая ячейка которых оказалась дальше радиуса выбора r (§5.3), в шаг не входят.
        self._enforce_pair_radius = bool(enforce_pair_radius)
        # Радиус окрестности ближней фазы (§5.5.2: r ≥ ‖p1 − p2‖) следует за радиусом выбора пар, но не ниже
        # этого значения; None — постоянный local_radius фазы.
        self._neighbourhood_radius_min = None if neighbourhood_radius_min is None else int(neighbourhood_radius_min)
        LOGGER.event(
            "layout.energy.neighbourhood_sample.config",
            section=OPTIM_SUBSET,
            data={
                "sample_side": -1 if self._neighbourhood_sample_side is None else self._neighbourhood_sample_side,
                "mode": "full_window" if self._neighbourhood_sample_side is None else "stratified_cells",
                "note": "энергия окрестности гауссова ядра по случайной выборке её ячеек: окно делится на слои "
                "side × side, из слоя берётся одна случайная ячейка (§5.9.4); None — по всем ячейкам",
            },
        )
        self._long_subset_indices: list[int] = []
        self._long_subset_last_step: int = 0
        self._long_step_counter: int = 0
        self._subset_similarity_cache: dict[int, "numpy.ndarray"] = {}
        self._short_subset_indices: tuple[int, ...] | None = None
        self._short_similarity_cache: dict[int, "numpy.ndarray"] = {}
        # Число единиц кодов и коды 64-битными словами: число общих бит пары — popcount(a & b) (§2.2.4).
        # Матрица бит N × L (_code_matrix) распаковывается из таблицы при первом обращении.
        self._ones_array = self._table.ones.astype(np.float32)
        self._packed_codes = self._table.words64() if hasattr(np, "bitwise_count") else None
        LOGGER.event(
            "layout.sim_matrix.ready",
            section=OPTIM_SIM_MATRIX,
            data={
                "points": self._point_count,
                "code_len": self._table.code_length,
            },
        )
        # Дальняя энергия по ненулевым sim_λ при жёстком пороге (§5.5.1, §5.9.5): строки сходства строятся
        # при первом шаге дальней фазы и прореживаются при росте λ (§7.1.7).
        self._sparse_long_enabled = bool(sparse_long)
        self._sparse_max_edges = int(sparse_max_edges)
        self._sparse_memory_fraction = float(sparse_memory_fraction)
        self._sparse_buffer_bytes = int(sparse_buffer_bytes)
        self._sparse_entry_budget = int(sparse_entry_budget)
        self._sparse_screen_min_chunks = int(sparse_screen_min_chunks)
        self._sparse_verify_every = int(sparse_verify_every)
        self._sparse_long: SparseLongEnergy | None = None
        # Порог, для которого строки сходства оказались слишком плотными (повторно не строятся).
        self._sparse_rejected_lambda: float | None = None
        # Сверка с исходным проходом — отдельно для дальней энергии, окрестности пар и энергий точек.
        self._sparse_steps_since_verify = {"long": 0, "neighbourhood": 0, "points": 0}
        self._sparse_verify_pending = {"long": True, "neighbourhood": True, "points": True}
        # Сверка пар идёт на первых порциях исходного прохода (0 — на всех парах шага).
        self._sparse_verify_chunks = int(sparse_verify_chunks)
        # Знак φ_c − φ_s пар одним проходом ядра по строкам сходства (§5.6, §5.12): создаётся вместе со строками.
        self._pair_kernel_enabled = bool(pair_kernel)
        self._pair_kernel_segment = int(pair_kernel_segment)
        self._pair_kernel: PairSignKernel | None = None
        # Сколько шагов и пар потребовали сумм редукции исходной формы (разность в пределах ошибки суммирования).
        self._undecided_steps = 0
        self._undecided_pairs = 0
        # Нижняя граница потенциала точки (§5.5.1): пары дальней фазы, которым обмен заведомо невыгоден,
        # не считаются по строкам сходства. Суммы границы ведутся, пока обменов на шаг мало.
        self._bounds_enabled = bool(potential_bounds)
        self._bounds_grid = int(potential_bounds_grid)
        self._bounds: PotentialBounds | None = None
        self._bounds_on = False
        self._bounds_move_limit = 0
        self._bounds_cooldown = 0
        self._bounds_window = [0, 0, 0]
        self._bounded_pairs = 0
        self._last_moved_points = 0
        # Суммы пар у нуля разности на хосте в порядке сложения редукции исходного прохода (ReductionReplica).
        self._replica_enabled = bool(reduction_replica)
        self._replica_max_bytes = int(replica_max_bytes)
        self._replica: ReductionReplica | None = None
        self._replica_pairs = 0
        # Скомпилированная выборка пар (§5.3, §5.9.1, §5.9.3): создаётся после расстановки точек.
        self._compiled_sampler_enabled = bool(compiled_sampler)
        self._compiled_sampler: CompiledPairSampler | None = None
        self._sampler_verify_every = int(sampler_verify_every)
        self._sampler_calls = {"long": 0, "short": 0}
        self._ones_float64 = None if self._ones_array is None else self._ones_array.astype(np.float64)
        # Класс одинаковых кодов каждой точки: у пары точек одного класса φ_c = φ_s точно (PairSignKernel).
        # Считается при первом расчёте по строкам сходства: большим раскладкам (без строк) он не нужен.
        self._code_class_cache: "np.ndarray | None" = None
        # Списки расстановки — источник истины (§5.1.2); массивы _pos_np, _grid_np повторяют их для векторных
        # выборки пар и расчёта энергий. True — списки могли измениться в обход обменов, массивы пересобираются.
        self._arrays_stale = True
        # True — обмены шага применены только к массивам (_apply_swaps_arrays): списки пересобираются из массивов
        # при первом обращении к ним (_sync_lists). Устаревшими бывают либо списки, либо массивы, но не оба.
        self._lists_stale = False
        self._pos_np: "np.ndarray | None" = None
        self._grid_np: "np.ndarray | None" = None
        self._short_weights_np: "np.ndarray | None" = None
        self._short_weights_source: list[float] | None = None

        if grid_size is None:
            target = self._point_count * (1.0 + empty_ratio)
            grid_size = math.ceil(math.sqrt(target))
        if grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if grid_size * grid_size < self._point_count:
            raise ValueError("grid_size too small for number of points")

        self.height = grid_size
        self.width = grid_size
        LOGGER.event(
            "layout.init",
            section=LAYOUT_ALGORITHM,
            data={
                "points": self._point_count,
                "labels": self._table.hue_count(),
                "seed": seed,
            },
        )
        LOGGER.event(
            "layout.grid",
            section=SPACE_FILLING,
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
            section=SIMILARITY_THRESHOLD,
            data={"lambda_threshold": self._lambda, "eta": self._eta},
        )
        LOGGER.event(
            "layout.energy.distance",
            section=OPTIM_ENERGY,
            data={
                "long_distance": self._long_distance,
                "short_distance": self._short_distance,
                "note": "euclidean = d из §5.5; squared = сумма квадратов без корня (§5.9.2)",
            },
        )
        LOGGER.event(
            "layout.energy.long.kernel",
            section=ENERGY_LONG,
            data={
                "kernel": self._long_kernel,
                "radius_min": self._long_kernel_radius_min,
                "width": self._long_kernel_width,
                "note": "distance — φ = Σ s·d по всему V (§5.5.1); gauss — расширение статьи: "
                "φ = Σ s·exp(−d²/2σ²) по окрестности r(t) = радиусу выбора пар (§7.1.7), σ = width·r, максимизация",
            },
        )
        LOGGER.event(
            "layout.similarity_threshold",
            section=SIMILARITY_THRESHOLD,
            data={
                "lambda_threshold": self._lambda,
                "eta": 0.0 if self._eta is None else float(self._eta),
                "eta_mode": "hard_cutoff" if self._eta is None else "sigmoid",
                "function": "sim * sigmoid(eta * (sim - lambda))" if self._eta is not None else "sim if sim >= lambda else 0",
            },
        )
        LOGGER.event(
            "layout.precompute",
            section=OPTIM_SIM_MATRIX,
            data={"precompute_similarity": precompute_similarity},
        )
        LOGGER.event(
            "layout.precompute_limit",
            section=OPTIM_SIM_MATRIX,
            data={"max_precompute": max_precompute},
        )
        LOGGER.event(
            "layout.energy.subset.config",
            section=OPTIM_SUBSET,
            data={
                "long_subset_size": -1 if self._long_subset_size is None else self._long_subset_size,
                "long_subset_refresh": self._long_subset_refresh,
                "long_subset_min_neighbours": self._long_subset_min_neighbours,
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
        self._table.log_summary()
        # Списки расстановки заполняются из массивов при первом обращении (_sync_lists): начальная расстановка
        # (_place_points) пишется сразу в массивы.
        self._grid_cells: list[list[int | None]] = [[] for _ in range(self.height)]
        self._positions_list: list[tuple[int, int]] = []
        self._pos_y_list: list[int] = []
        self._pos_x_list: list[int] = []
        self._row_occupied_sets: list[set[int]] = [set() for _ in range(self.height)]
        self._radius_offsets_cache: dict[int, list[tuple[int, int]]] = {}
        self._short_energy_weights: list[float] | None = None
        self._short_energy_last_step: int = -1
        self._short_energy_radius: int | None = None
        self._short_energy_eps: float = 1e-6
        # Версия расстановки растёт при каждом обмене; энергии точек E(c, r) кешируются по (радиус, версия),
        # чтобы раскладка, активация и экспорт не считали одно и то же (§5.7).
        self._positions_version: int = 0
        self._point_energy_cache: dict[int, tuple[int, list[float]]] = {}
        if initial_cells is None:
            self._place_points()
        else:
            # Начальная расстановка задана (раскладка от затравки, вне статьи) вместо случайной (§5.2).
            self.place_cells(initial_cells, log=False)

        self._long_swap_radius_sum = 0.0
        self._long_swap_count = 0
        self._pair_radius: int | None = None
        self._distance_eps = 1e-6
        # Температура принятия обменов: 0 — жадное правило §5.6; T > 0 — невыгодный обмен принимается
        # с вероятностью exp(Δ / (T · rms|Δ|)) (отжиг, расширение статьи; §5.2 упоминает SA как прообраз).
        self._swap_temperature = 0.0
        LOGGER.event(
            "layout.distance_eps",
            section=OUTSIDE_ARTICLE,
            data={"distance_eps": self._distance_eps},
        )
        self._last_visual_energy: float | None = None
        self._last_visual_energy_step: int | None = None
        self._last_energy_value: float | None = None
        self._quality_radius: int | None = None
        self._last_energy_step: int | None = None
        self._last_energy_progress: float | None = None
        self._gpu_engine: _GpuLayoutEngine | None = None
        self._tensor_engine: _TensorLayoutEngine | None = None
        self._torch_engine: TorchPairEnergyEngine | None = None
        self._torch_positions_version = -1
        # Точки и ячейки, изменившиеся с последней загрузки на устройство: после шага загружаются только они.
        self._torch_dirty_points: set[int] = set()
        self._torch_dirty_cells: dict[tuple[int, int], int] = {}
        # Обмены, применённые по массивам и ещё не загруженные на устройство: по набору массивов на шаг
        # (сдвинутые точки и их ячейки, изменившиеся ячейки и их точки). _torch_tracked_changes — сколько
        # изменений версии расстановки описано этими наборами или списками _torch_dirty_*.
        self._torch_pending_moves: list[tuple["np.ndarray", ...]] = []
        self._swap_radius_cache: "np.ndarray | None" = None
        self._torch_tracked_changes = 0
        # Цвета точек массивами uint8 для карт хода раскладки (§5.14, рис. 16): от расстановки не зависят.
        self._colors_rgb_array: "np.ndarray | None" = None
        self._component_colors_cache: dict[float | None, list[tuple[int, int, int]]] = {}
        self._component_colors_arrays: dict[float | None, "np.ndarray"] = {}
        self._gpu_positions_dirty = True
        if self._use_gpu:
            # Основной путь: энергии пар и точек на torch со сходством из кодов (§5.9.5, §5.12);
            # OpenGL-движок с матрицей N×N и прежний тензорный путь — только если torch недоступен.
            self._torch_engine = TorchPairEnergyEngine.create(
                code_table=self._table,
                ones=self._ones_array,
                similarity=self._similarity,
                grid_shape=(self.height, self.width),
            )
            if self._torch_engine is not None:
                # Слитый расчёт энергий: те же φ_c, φ_s одним ядром на слагаемое (§5.5, §5.9.5, §5.12).
                self._torch_engine.configure_fused(
                    enabled=bool(fused_energy),
                    block_elements=int(fused_block_elements),
                    table_max_bytes=int(fused_table_max_bytes),
                    table_memory_fraction=float(fused_table_memory_fraction),
                    verify_every=int(fused_verify_every),
                    similarity_table=bool(fused_similarity_table),
                )
            if self._torch_engine is None:
                self._gpu_engine = _GpuLayoutEngine.create(self._points, self._similarity)
                self._tensor_engine = _TensorLayoutEngine.create(
                    code_matrix=self._code_matrix,
                    ones=self._ones_array,
                    similarity=self._similarity,
                )
        if self._use_gpu and (
            self._torch_engine is not None or self._gpu_engine is not None or self._tensor_engine is not None
        ):
            if self._torch_engine is not None:
                status = f"torch:{self._torch_engine.device}"
            elif self._gpu_engine is not None:
                status = "OpenGL 4.1"
            else:
                status = f"tensor:{self._tensor_engine.device}"
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
        if self._compiled_sampler_enabled and self._merged_offset_draws():
            self._compiled_sampler = CompiledPairSampler.create(height=self.height, width=self.width)

    # ------------------------------------------------------------------ списки расстановки
    # grid, _positions, _pos_y, _pos_x, _row_occupied — расстановка раскладки (§5.1.2) списками; массивы
    # _pos_np, _grid_np — она же для векторных выборки пар и расчёта энергий. Шаг на torch применяет обмены
    # только к массивам (_apply_swaps_arrays) и помечает списки устаревшими; обращение к спискам по этим
    # именам сначала пересобирает их из массивов (_sync_lists, на месте: ссылки на списки остаются верными),
    # а затем помечает устаревшими массивы — список могут править на месте или заменить, и следующий шаг
    # пересоберёт массивы (_sync_arrays). _apply_swaps меняет списки и массивы вместе.
    @property
    def _points(self) -> list[LayoutPoint]:
        """Точки объектами (код BitArray, подпись, hue, число единиц): строятся из таблицы кодов по требованию."""
        if self._points_cache is None:
            table = self._table
            labels = table.labels()
            hues = table.hues.tolist()
            ones = table.ones.tolist()
            self._points_cache = [
                LayoutPoint(code=code, label=labels[index], hue=hues[index], ones=ones[index])
                for index, code in enumerate(table.codes())
            ]
        return self._points_cache

    @property
    def _labels(self) -> list[str]:
        if self._labels_cache is None:
            self._labels_cache = [self._format_label(label) for label in self._table.labels()]
        return self._labels_cache

    @property
    def _values(self) -> list[str]:
        """Коды точек строками «0101…» (как BitArray.to01())."""
        if self._values_cache is None:
            self._values_cache = self._table.values()
        return self._values_cache

    @property
    def _code_matrix(self) -> "np.ndarray":
        """Биты кодов матрицей uint8 N × L (§2.2): распаковывается из таблицы при первом обращении."""
        if self._code_matrix_cache is None:
            self._code_matrix_cache = self._table.bits()
        return self._code_matrix_cache

    @property
    def point_count(self) -> int:
        """Число точек раскладки N (§5.1.1)."""
        return self._point_count

    @property
    def code_table(self) -> LayoutCodeTable:
        """Коды точек раскладки таблицей массивов (в порядке точек)."""
        return self._table

    @property
    def grid(self) -> list[list[int | None]]:
        self._sync_lists()
        self._arrays_stale = True
        return self._grid_cells

    @grid.setter
    def grid(self, value: list[list[int | None]]) -> None:
        self._sync_lists()
        self._arrays_stale = True
        self._grid_cells = value

    @property
    def _positions(self) -> list[tuple[int, int]]:
        self._sync_lists()
        self._arrays_stale = True
        return self._positions_list

    @_positions.setter
    def _positions(self, value: list[tuple[int, int]]) -> None:
        self._sync_lists()
        self._arrays_stale = True
        self._positions_list = value

    @property
    def _pos_y(self) -> list[int]:
        self._sync_lists()
        self._arrays_stale = True
        return self._pos_y_list

    @_pos_y.setter
    def _pos_y(self, value: list[int]) -> None:
        self._sync_lists()
        self._arrays_stale = True
        self._pos_y_list = value

    @property
    def _pos_x(self) -> list[int]:
        self._sync_lists()
        self._arrays_stale = True
        return self._pos_x_list

    @_pos_x.setter
    def _pos_x(self, value: list[int]) -> None:
        self._sync_lists()
        self._arrays_stale = True
        self._pos_x_list = value

    @property
    def _row_occupied(self) -> list[set[int]]:
        self._sync_lists()
        self._arrays_stale = True
        return self._row_occupied_sets

    @_row_occupied.setter
    def _row_occupied(self, value: list[set[int]]) -> None:
        self._sync_lists()
        self._arrays_stale = True
        self._row_occupied_sets = value

    def _sync_lists(self) -> None:
        """Списки расстановки догоняют массивы после обменов, применённых по массивам (_apply_swaps_arrays).

        Списки пересобираются на месте: объекты grid, _positions, … остаются теми же.
        """
        self._pull_device()
        if not self._lists_stale:
            return
        self._lists_stale = False
        grid_np = self._grid_np
        for cells, row in zip(self._grid_cells, grid_np.tolist()):
            cells[:] = [None if point < 0 else point for point in row]
        pos_y = self._pos_np[:, 0].tolist()
        pos_x = self._pos_np[:, 1].tolist()
        self._positions_list[:] = list(zip(pos_y, pos_x))
        self._pos_y_list[:] = pos_y
        self._pos_x_list[:] = pos_x
        occupied_y, occupied_x = np.nonzero(grid_np >= 0)
        starts = np.searchsorted(occupied_y, np.arange(self.height + 1))
        columns = occupied_x.tolist()
        for row, occupied in enumerate(self._row_occupied_sets):
            occupied.clear()
            occupied.update(columns[starts[row] : starts[row + 1]])

    def _positions_array(self) -> "np.ndarray":
        """Ячейки (y, x) всех точек массивом N×2 — без пересборки списков и без сброса массивов."""
        self._pull_device()
        if self._lists_stale or (not self._arrays_stale and self._pos_np is not None):
            return self._pos_np
        return np.asarray(self._positions_list, dtype=np.int64).reshape(-1, 2)

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
        energy_every: int | None = None,
        energy_target: float | None = None,
        adaptive_params: AdaptiveLayoutConfig | None = None,
        short_optim: ShortLayoutOptimConfig | None = None,
        save_on_stop_path: str | None = None,
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
        if energy_every is not None and energy_every <= 0:
            raise ValueError("energy_every must be positive when set")
        if energy_every is not None and energy_radius is None:
            raise ValueError("energy_every requires energy_radius: r_e of Ē (§5.7, §5.8) must be given")
        if energy_target is not None and not 0.0 < energy_target <= 1.0:
            raise ValueError("energy_target must be in (0, 1]: Ē is the mean normalised energy (§5.8)")
        if energy_target is not None and energy_every is None:
            raise ValueError("energy_target requires energy_every: Ē must be evaluated to be compared")
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
            if short_optim.partitions <= 0:
                raise ValueError("short partitions must be positive")

        effective_energy_radius = energy_radius
        if energy_radius is not None:
            # §5.7: r_e — радиус энергии точки для матрицы Ê и качества Ē (§5.8); «r_e ≤ 5 достаточно».
            self._quality_radius = int(energy_radius)

        LOGGER.event(
            "layout.run.speculative",
            section=PARALLEL_PROCESSING,
            data={
                "mode": mode,
                "pairs_per_step": pairs_per_step,
                "points": self._point_count,
                "pairs_per_point": pairs_per_step / max(1, self._point_count),
                "note": "пары шага считаются по старым позициям; без влияния на результат, "
                "пока обменов за шаг много меньше числа точек (§5.10)",
            },
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
        if energy_every is not None:
            LOGGER.event(
                "layout.run.energy_target",
                section=POINT_EXCHANGE,
                data={
                    "every": energy_every,
                    "energy_radius": energy_radius,
                    "energy_target": energy_target,
                    "note": "шаги повторяются, пока число обменов не упадёт ниже порога или "
                    "качество Ē (§5.8) не достигнет заданного значения; None — только по обменам",
                },
            )
        swap_monitor: _SwapStallMonitor | None = None
        if min_swap_window and min_swap_ratio > 0:
            swap_monitor = _SwapStallMonitor(min_swap_ratio, min_swap_window)
            LOGGER.event(
                "layout.run.swap_monitor",
                section=POINT_EXCHANGE,
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
                    "mode": mode,
                    "start_radius": adaptive_params.start_radius,
                    "end_radius": adaptive_params.end_radius,
                    "radius_schedule": adaptive_params.radius_schedule,
                    "radius_factor": adaptive_params.radius_factor,
                    "swap_ratio_trigger": adaptive_params.swap_ratio_trigger,
                    "swap_window": adaptive_params.swap_window,
                    "lambda_step": adaptive_params.lambda_step,
                    "lambda_max": adaptive_params.lambda_max,
                    "temperature": adaptive_params.temperature,
                    "temperature_radius_min": adaptive_params.temperature_radius_min,
                    "temperature_radius_max": adaptive_params.temperature_radius_max,
                    "steps": steps,
                    "note": "swaps — по статье: λ и r меняются, когда обмены падают до доли trigger "
                    "от начального значения этапа (§7.1.7); linear/geometric и temperature > 0 — "
                    "расширение статьи (сглаживание между пинвилами)",
                },
            )
        short_config: ShortLayoutOptimConfig | None = short_optim
        if mode == "short":
            if short_config is None:
                base_energy_radius = local_radius if local_radius is not None else 3
                short_config = ShortLayoutOptimConfig(
                    energy_radius=base_energy_radius,
                    energy_recalc_every=max(1, steps // 20),
                    energy_eps=1e-6,
                    use_weighted_first_point=True,
                    similarity_cutoff=self._lambda,
                    partitions=1,
                )
            elif short_config.similarity_cutoff is None:
                short_config = ShortLayoutOptimConfig(
                    energy_radius=short_config.energy_radius,
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
            if adaptive_params is not None and mode == "long":
                step_temperature = adaptive_params.temperature_for_radius(
                    effective_pair_radius if effective_pair_radius is not None else 0
                )
                if step_temperature != self._swap_temperature:
                    self.set_swap_temperature(step_temperature)
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
                    section=POINT_EXCHANGE,
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
                if (
                    should_stop_swaps
                    and adaptive_state is not None
                    and not adaptive_state.schedule_complete(self._lambda)
                ):
                    # §5.6 + §7.1.7: при текущих λ и r обменов почти нет, но r ещё не 1 или λ не λ_max —
                    # вместо остановки фаза переходит к следующему этапу расписания.
                    should_stop_swaps = False
                    forced = adaptive_state.force_next_stage(step=step, current_lambda=self._lambda)
                    swap_monitor.reset()
                    LOGGER.event(
                        "layout.adaptive.stage.forced",
                        section=LAYOUT_PARAMETERS,
                        data={
                            "step": current_step,
                            "avg_swaps": avg_swaps_window,
                            "min_swap_ratio": swap_monitor.min_ratio,
                            "pair_radius": forced.radius,
                            "lambda_threshold": self._lambda
                            if forced.new_lambda is None
                            else forced.new_lambda,
                            "triggers": adaptive_state.trigger_count,
                        },
                    )
                    self._apply_adaptive_lambda(forced, step=current_step)
                if should_stop_swaps:
                    LOGGER.event(
                        "layout.run.swap_stability.stop",
                        section=POINT_EXCHANGE,
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
                    if mode == "long":
                        self._log_layout_state(
                            path=log_path,
                            step=current_step,
                            log_visuals=log_visuals,
                            force=True,
                        )
                        self._save_on_stop(
                            path=save_on_stop_path,
                            reason="swap_ratio",
                            step=current_step,
                            mode=mode,
                        )
                    break
            if log_every is not None and step % log_every == 0:
                self._log_layout_state(
                    path=log_path, step=step + step_offset, log_visuals=log_visuals
                )
            energy_value: float | None = None
            previous_energy: float | None = None
            if energy_every is not None and step % energy_every == 0:
                energy_value = self.average_local_energy(effective_energy_radius)
                previous_energy = self._last_energy_progress
                self._last_energy_value = energy_value
                self._last_energy_step = step + step_offset
                self._last_energy_progress = energy_value
            if mode == "long" and energy_value is not None:
                # §5.8: Ē — качество раскладки, при упорядочивании растёт; прирост за период оценки.
                energy_gain = (
                    0.0
                    if previous_energy is None
                    else energy_value - previous_energy
                )
                log_energy_progress = LOGGER.should_log("layout.energy.long.progress")
                energy_visuals = (
                    LOGGER.visual_scalars(
                        f"{log_path}/energy/long",
                        {"average_energy": energy_value, "energy_gain": energy_gain},
                        step=current_step,
                        timeline=_LAYOUT_STEP_TIMELINE,
                    )
                    if log_energy_progress
                    else None
                )
                LOGGER.event(
                    "layout.energy.long.progress",
                    section=QUALITY_ASSESS,
                    data={
                        "step": current_step,
                        "energy": energy_value,
                        "energy_prev": previous_energy,
                        "energy_gain": energy_gain,
                        "radius": effective_energy_radius,
                    },
                    visuals=energy_visuals,
                    force=log_energy_progress,
                )
            if adaptive_state is not None:
                update = adaptive_state.update_after_step(
                    step=step, swaps=swaps, current_lambda=self._lambda
                )
                LOGGER.event(
                    "layout.adaptive.radius",
                    section=LAYOUT_PARAMETERS,
                    data={
                        "step": current_step,
                        "pair_radius": effective_pair_radius,
                        "swap_rate": update.swap_rate,
                        "stage_initial_swaps": update.stage_initial,
                        "swap_ratio": update.swap_ratio,
                        "lambda_threshold": self._lambda,
                    },
                )
                if update.triggered:
                    if adaptive_state.by_swaps and swap_monitor is not None:
                        swap_monitor.reset()
                    LOGGER.event(
                        "layout.adaptive.trigger",
                        section=LAYOUT_PARAMETERS,
                        data={
                            "step": current_step,
                            "swap_rate": update.swap_rate,
                            "stage_initial_swaps": update.stage_initial,
                            "swap_ratio": update.swap_ratio,
                            "swap_ratio_trigger": adaptive_params.swap_ratio_trigger,
                            "pair_radius": update.radius,
                            "triggers": adaptive_state.trigger_count,
                        },
                    )
                self._apply_adaptive_lambda(update, step=current_step)
            if (
                energy_target is not None
                and energy_value is not None
                and energy_value >= energy_target
            ):
                LOGGER.event(
                    "layout.run.energy_target.stop",
                    section=POINT_EXCHANGE,
                    data={
                        "step": current_step,
                        "energy": energy_value,
                        "energy_target": energy_target,
                        "total_swaps": total_swaps,
                    },
                )
                if mode == "long":
                    self._log_layout_state(
                        path=log_path,
                        step=current_step,
                        log_visuals=log_visuals,
                        force=True,
                    )
                    self._save_on_stop(
                        path=save_on_stop_path,
                        reason="energy_target",
                        step=current_step,
                        mode=mode,
                    )
                break

        if adaptive_state is not None and not adaptive_state.schedule_complete(self._lambda):
            LOGGER.event(
                "layout.adaptive.incomplete",
                section=LAYOUT_PARAMETERS,
                data={
                    "mode": mode,
                    "steps": steps_executed,
                    "pair_radius": adaptive_state.radius_for_step(steps_executed),
                    "end_radius": adaptive_params.end_radius,
                    "lambda_threshold": self._lambda,
                    "lambda_max": adaptive_params.lambda_max,
                    "triggers": adaptive_state.trigger_count,
                    "note": "бюджет шагов кончился раньше, чем r дошёл до 1 и λ до λ_max (§7.1.7)",
                },
            )

        if self._swap_temperature != 0.0:
            self.set_swap_temperature(0.0)
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

    def _save_on_stop(self, *, path: str | None, reason: str, step: int, mode: str) -> None:
        if mode != "long" or path is None:
            return
        try:
            self.save_json(path)
            LOGGER.event(
                "layout.run.save_on_stop",
                section=OUTSIDE_ARTICLE,
                data={
                    "path": path,
                    "reason": reason,
                    "step": step,
                    "mode": mode,
                },
            )
        except Exception as exc:
            LOGGER.event(
                "layout.run.save_on_stop.error",
                section=OUTSIDE_ARTICLE,
                data={
                    "path": path,
                    "reason": reason,
                    "step": step,
                    "mode": mode,
                    "error": str(exc),
                },
            )
            raise

    def _log_layout_state(
        self,
        path: str,
        step: int,
        *,
        log_visuals: bool,
        force: bool = False,
    ) -> None:
        if not force and not LOGGER.should_log("layout.visual"):
            return
        # Карта большой раскладки — картинка в десятки мегабайт: без Rerun она не строится.
        visuals_skipped = not log_visuals or (
            self._point_count > _VISUAL_POINT_LIMIT and not LOGGER.visuals_enabled()
        )
        visuals: list = []
        if log_visuals and not visuals_skipped:
            # Позиции и цвета массивами: те же значения, что у списков, без обхода точек в Python.
            colors = self._colors_array()
            visuals = self._map_visuals(f"{path}/image", f"{path}/image", self._build_image(colors), colors)
            if self._component_period is not None:
                # Та же карта, раскрашенная по углу стимула, как в pinwheel-тесте (рис. 16, §5.13.2).
                component_colors = self._component_colors_array(self._component_period)
                visuals += self._map_visuals(
                    f"{path}/angle", f"{path}/angle", self._build_image(component_colors), component_colors
                )
        energy_radius: int | None = None
        energy_value: float | None = None
        energy_source: str | None = None
        if self._last_energy_value is not None:
            energy_value = self._last_energy_value
            energy_radius = None
            energy_source = "cached"
        elif (
            log_visuals
            and self._quality_radius is not None
            and self._point_count <= _VISUAL_ENERGY_POINT_LIMIT
        ):
            # §5.8: Ē с тем же радиусом r_e, что и критерий качества фазы (§5.7).
            energy_radius = self._quality_radius
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
            self._validate_eta(eta)
            self._eta = eta
        self._invalidate_threshold_caches(reason="similarity_params")
        LOGGER.event(
            "layout.similarity_params",
            section=SIMILARITY_THRESHOLD,
            data={"lambda_threshold": self._lambda, "eta": self._eta},
        )

    def _invalidate_threshold_caches(self, *, reason: str) -> None:
        """Кеши строк sim_λ хранят схожесть с уже применённым порогом (§5.4): при смене λ или η они устаревают."""
        dropped = len(self._subset_similarity_cache) + len(self._short_similarity_cache)
        self._subset_similarity_cache.clear()
        self._short_similarity_cache.clear()
        self._point_energy_cache.clear()
        if self._tensor_engine is not None:
            self._tensor_engine.reset_subset()
        LOGGER.event(
            "layout.similarity_threshold.cache_reset",
            section=SIMILARITY_THRESHOLD,
            data={"reason": reason, "lambda_threshold": self._lambda, "eta": self._eta, "dropped_rows": dropped},
        )

    @staticmethod
    def _validate_eta(eta: object) -> None:
        """η задаёт крутизну сигмоиды порога λ; η = 0 отключает порог (§5.4)."""
        if eta is None:
            return
        if not isinstance(eta, (int, float)) or isinstance(eta, bool) or float(eta) <= 0.0:
            raise ValueError(
                "eta must be positive or None for a hard cutoff; eta=0 would "
                "turn sim_lambda into a constant 0.5*sim and disable the threshold (§5.4)"
            )

    def cells_array(self) -> "np.ndarray":
        """Ячейки (y, x) всех точек массивом N × 2 (копия)."""
        return np.array(self._positions_array(), dtype=np.int64)

    def place_cells(self, cells: "np.ndarray", *, log: bool = True) -> None:
        """Расставить точки по ячейкам массива N × 2 (y, x): одна точка в ячейке дискретной матрицы (§5.1.2).

        То же, что restore_positions, но массивами — без списков на каждую точку.
        """
        cells = np.asarray(cells, dtype=np.int64)
        if cells.shape != (self._point_count, 2):
            raise ValueError("cells must contain one (y, x) cell per point")
        pos_y = cells[:, 0]
        pos_x = cells[:, 1]
        if bool(((pos_y < 0) | (pos_y >= self.height) | (pos_x < 0) | (pos_x >= self.width)).any()):
            raise ValueError(f"cells are outside the {self.height}x{self.width} grid")
        grid = np.full((self.height, self.width), -1, dtype=np.int64)
        grid[pos_y, pos_x] = np.arange(self._point_count, dtype=np.int64)
        if int((grid >= 0).sum()) != self._point_count:
            raise ValueError("a cell holds two points; the layout matrix is discrete (§5.1.2)")
        self._pos_np = np.stack([pos_y, pos_x], axis=1)
        self._grid_np = grid
        self._arrays_stale = False
        self._lists_stale = True
        self._gpu_positions_dirty = True
        self._positions_version += 1
        if log:
            LOGGER.event(
                "layout.place_cells",
                section=LAYOUT_DISCRETE,
                data={"points": self._point_count, "width": self.width, "height": self.height},
            )

    def release_device(self) -> None:
        """Освободить память устройства, занятую расчётом энергий этой раскладки (таблицы сходства, коды)."""
        self._pull_device()
        engine = self._torch_engine
        self._torch_engine = None
        self._sparse_long = None
        self._device_short = None
        self._device_state = None
        self._device_owner = None
        if engine is not None:
            engine.release()

    def restore_positions(self, positions: Sequence[tuple[int, int]]) -> None:
        """Восстановить уже разложенное пространство: одна точка в ячейке дискретной матрицы (§5.1.2)."""
        if len(positions) != self._point_count:
            raise ValueError("positions must contain one cell per point")
        grid: list[list[int | None]] = [[None for _ in range(self.width)] for _ in range(self.height)]
        row_occupied: list[set[int]] = [set() for _ in range(self.height)]
        for idx, (y, x) in enumerate(positions):
            if not (0 <= y < self.height and 0 <= x < self.width):
                raise ValueError(f"position {(y, x)} is outside the {self.height}x{self.width} grid")
            if grid[y][x] is not None:
                raise ValueError(f"cell {(y, x)} holds two points; the layout matrix is discrete (§5.1.2)")
            grid[y][x] = idx
            row_occupied[y].add(x)
        self.grid = grid
        self._row_occupied = row_occupied
        self._positions = [(int(y), int(x)) for y, x in positions]
        self._pos_y = [p[0] for p in self._positions]
        self._pos_x = [p[1] for p in self._positions]
        self._gpu_positions_dirty = True
        self._positions_version += 1
        LOGGER.event(
            "layout.restore_positions",
            section=LAYOUT_DISCRETE,
            data={"points": self._point_count, "width": self.width, "height": self.height},
        )

    @property
    def lambda_threshold(self) -> float:
        """Текущий порог сходства λ (§5.4); растёт ступенями по ходу раскладки (§7.1.7)."""
        return float(self._lambda)

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
        if mode == "long" and self._device_step_enabled and pair_radius is not None:
            device_swaps = self._device_long_step(pairs_per_step, int(pair_radius), step_index)
            if device_swaps is not None:
                return device_swaps
        if mode == "short" and self._device_short_enabled and pair_radius is not None and local_radius is not None:
            device_swaps = self._device_short_step(
                pairs_per_step, int(pair_radius), int(local_radius), step_index, short_config
            )
            if device_swaps is not None:
                return device_swaps
        self._sync_arrays()
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
            pairs_np = self._sample_pairs_short_np(
                pairs_per_step,
                pair_radius=pair_radius,
                similarity_cutoff=similarity_cutoff,
                weighted_first=weighted_first,
                step_index=step_index,
                partitions=partitions,
            )
        else:
            pairs_np = self._sample_pairs_long_np(pairs_per_step, pair_radius)
        if self._enforce_pair_radius and pair_radius is not None and pairs_np.shape[0]:
            pairs_np = self._pairs_within_radius(pairs_np, pair_radius, mode=mode, step_index=step_index)

        # Дальняя фаза с гауссовым ядром считается машинерией окрестности §5.5.2
        # (радиус r(t) = радиус пар, ядро gauss, обмен при φ_s > φ_c); выбор пар остаётся дальним (§5.3).
        gauss_long = mode == "long" and self._long_kernel == "gauss"
        eval_short = mode == "short" or gauss_long
        eval_kernel = "gauss" if gauss_long else "inverse"
        if mode == "short":
            eval_local_radius = self._short_neighbourhood_radius(local_radius, pair_radius)
        else:
            eval_local_radius = self._long_kernel_radius(pair_radius)

        if self._torch_engine is not None:
            torch_swaps = self._evaluate_pairs_torch(
                pairs_np,
                mode=mode,
                eval_short=eval_short,
                kernel=eval_kernel,
                local_radius=eval_local_radius,
                step_index=step_index,
            )
            if torch_swaps is not None:
                return self._apply_swaps_arrays(torch_swaps, mode=mode)
        pairs = [(int(row[0]), int(row[1]), int(row[2]), int(row[3])) for row in pairs_np]

        # Для дальней фазы с гауссовым ядром OpenGL-движок (все пары одним вызовом) быстрее
        # тензорного пути с подмножеством всего V, поэтому тензорный путь берётся только без OpenGL.
        if eval_short and not (gauss_long and self._gpu_engine is not None):
            tensor_swaps = self._evaluate_pairs_short_tensor(
                pairs,
                local_radius=eval_local_radius,
                kernel=eval_kernel,
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
                        mode="short" if eval_short else "long",
                        local_radius=eval_local_radius,
                        lambda_threshold=self._lambda,
                        eta=self._eta,
                        distance_eps=self._distance_eps,
                        euclidean=self._distance_mode("short" if eval_short else "long") == "euclidean",
                        kernel=eval_kernel,
                        kernel_width=self._long_kernel_width,
                    )
                else:
                    energies = []
            except Exception:
                self._gpu_engine = None
                energies = None

            if energies is not None:
                deltas = [
                    (energy_s - energy_c) if eval_short else (energy_c - energy_s)
                    for (energy_c, energy_s) in energies
                ]
                return self._apply_tracked_swaps(
                    self._select_swaps(swap_coords, deltas), mode=mode
                )
            self._gpu_engine = None

        if mode == "long" and not gauss_long:
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
            deltas = [energy_c - energy_s for (energy_c, energy_s) in energies]
            return self._apply_tracked_swaps(
                self._select_swaps(swap_coords, deltas), mode=mode
            )

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
            coords: list[tuple[int, int, int, int]] = []
            deltas: list[float] = []
            for y1, x1, y2, x2 in pairs:
                _, delta = self._should_swap(
                    y1, x1, y2, x2,
                    mode="short" if eval_short else mode,
                    local_radius=eval_local_radius,
                    kernel=eval_kernel,
                )
                coords.append((y1, x1, y2, x2))
                deltas.append(delta)
            swaps_with_delta = self._select_swaps(coords, deltas)
        return self._apply_tracked_swaps(swaps_with_delta, mode=mode)

    def _short_neighbourhood_radius(self, local_radius: int | None, pair_radius: int | None) -> int | None:
        """Радиус окрестности пар ближней фазы (§5.5.2): r ≥ ‖p1 − p2‖, «чем больше r, тем точнее раскладка».

        Обычно это постоянный local_radius фазы. В больших раскладках (neighbourhood_radius_min задан) у каждой
        пары свой радиус — наименьший допустимый: r = max(⌈‖p1 − p2‖⌉ + 1, neighbourhood_radius_min), но не больше
        local_radius. Расчёт энергий берёт наибольшее из возвращённого значения и ⌈‖p1 − p2‖⌉ + 1, поэтому
        условие r ≥ ‖p1 − p2‖ выполнено всегда, а число слагаемых суммы следует за расстоянием между точками пары.
        """
        if self._neighbourhood_radius_min is None or local_radius is None or pair_radius is None:
            return local_radius
        return min(int(local_radius), self._neighbourhood_radius_min)

    # ----------------------------------------------------------------- массивы и векторная выборка пар
    def _sync_arrays(self) -> None:
        """Массивы позиций и сетки догоняют списки (источник истины — grid, _pos_y, _pos_x).

        Обмены меняют списки и массивы вместе (_apply_swaps), поэтому пересборка O(N) нужна только после
        обращения к спискам в обход обменов: расстановка точек, восстановление раскладки, правка снаружи.
        """
        self._pull_device()
        if not self._arrays_stale and self._pos_np is not None and self._grid_np is not None:
            return
        pos_y = np.asarray(self._pos_y_list, dtype=np.int64)
        pos_x = np.asarray(self._pos_x_list, dtype=np.int64)
        self._pos_np = np.stack([pos_y, pos_x], axis=1)
        grid = np.full((self.height, self.width), -1, dtype=np.int64)
        valid = (pos_y >= 0) & (pos_x >= 0)
        grid[pos_y[valid], pos_x[valid]] = np.flatnonzero(valid)
        self._grid_np = grid
        self._arrays_stale = False

    def _ensure_torch_positions(self) -> None:
        if self._torch_engine is None:
            return
        if self._torch_positions_version == self._positions_version:
            return
        # Все изменения расстановки с последней загрузки описаны обменами — загружаются только они.
        tracked = (
            self._torch_positions_version >= 0
            and self._torch_positions_version + self._torch_tracked_changes == self._positions_version
        )
        pending = self._torch_pending_moves
        limit = max(1, self._point_count // 4)
        bounds = self._bounds
        moved_points = sum(len(moves[0]) for moves in pending)
        if tracked and pending and not self._torch_dirty_points and moved_points <= limit:
            # Обмены, применённые по массивам: наборы уходят на устройство в порядке шагов. Изменились только
            # ячейки обменов: на устройство уходят они и их нынешние жильцы (обмены шага не делят ячеек, §5.10).
            width = self.width
            keep_bounds = (
                bounds is not None
                and self._bounds_on
                and bounds.valid
                and len(pending) == 1
                and moved_points <= self._bounds_move_limit
            )
            if bounds is not None and not keep_bounds:
                # Обменов слишком много (их учёт дороже расчёта строк) либо суммы не ведутся.
                self._bounds_on = False
                bounds.invalidate()
            for moved, moved_y, moved_x, y1, x1, y2, x2, idx_a, idx_b in pending:
                cell_1 = y1 * width + x1
                cell_2 = y2 * width + x2
                cells = np.concatenate([cell_1, cell_2])
                occupants = np.concatenate([idx_b, idx_a])
                if keep_bounds:
                    # Суммы границы потенциала обновляются вместе с позициями: проход по строкам сдвинутых
                    # точек — тройки (точка, старая ячейка, новая ячейка).
                    has_a = idx_a >= 0
                    has_b = idx_b >= 0
                    triples = np.concatenate(
                        [
                            np.stack([idx_a[has_a], cell_1[has_a], cell_2[has_a]], axis=1),
                            np.stack([idx_b[has_b], cell_2[has_b], cell_1[has_b]], axis=1),
                        ]
                    )
                    bounds.apply_moves(cells, occupants, triples)
                else:
                    self._torch_engine.update_cells(cells, occupants)
        elif tracked and not pending and 0 < len(self._torch_dirty_points) <= limit:
            if bounds is not None:
                bounds.invalidate()
            self._sync_lists()
            pos_y = self._pos_y_list
            pos_x = self._pos_x_list
            indices = sorted(self._torch_dirty_points)
            self._torch_engine.update_points(
                indices,
                [pos_y[idx] for idx in indices],
                [pos_x[idx] for idx in indices],
                [(y, x, occupant) for (y, x), occupant in self._torch_dirty_cells.items()],
            )
        else:
            if bounds is not None:
                bounds.invalidate()
            cells = self._positions_array()
            self._torch_engine.update_positions(cells[:, 0], cells[:, 1])
        self._last_moved_points = moved_points
        pending.clear()
        self._torch_tracked_changes = 0
        self._torch_dirty_points.clear()
        self._torch_dirty_cells.clear()
        self._torch_positions_version = self._positions_version

    def _host_layout_state(self, *, device_needed: bool) -> dict[str, "np.ndarray"]:
        """Расстановка для расчёта энергий движком torch: массивы хоста для слитого ядра (или пусто).

        Слитое ядро берёт нужную часть расстановки с хоста вместе с парами шага. Если расстановка на устройстве
        шагу не нужна (device_needed = False: энергии считает только слитое ядро), обмены прошлых шагов на
        устройство не загружаются: она остаётся устаревшей, и следующее обращение к ней
        (_ensure_torch_positions) загрузит её целиком. Иначе она обновляется как обычно.
        """
        engine = self._torch_engine
        if engine is None or not engine.fused_active:
            self._ensure_torch_positions()
            return {}
        self._sync_arrays()
        if device_needed:
            self._ensure_torch_positions()
        else:
            self._torch_pending_moves.clear()
            self._torch_dirty_points.clear()
            self._torch_dirty_cells.clear()
            self._torch_tracked_changes = 0
        return {"positions": self._pos_np, "grid": self._grid_np}

    def _sample_nearby_np(
        self,
        y1: "np.ndarray",
        x1: "np.ndarray",
        radius: int | None,
        bounds: tuple[int, int, int, int] | None,
    ) -> tuple["np.ndarray", "np.ndarray"]:
        """Вторые ячейки пар: до 50 попыток в радиусе r (§5.3, §5.9.1), затем любая ячейка в границах.

        То же правило, что и у поштучной выборки, но для всех пар сразу: плитка из одной ячейки,
        совпадающей с первой точкой, расширяется до всей сетки.
        """
        count = int(y1.size)
        rng = self._np_rng
        y2 = np.empty(count, dtype=np.int64)
        x2 = np.empty(count, dtype=np.int64)
        if bounds is None:
            lo_y, hi_y, lo_x, hi_x = 0, self.height - 1, 0, self.width - 1
            single = None
        else:
            lo_y, hi_y, lo_x, hi_x = bounds
            single = None
            if lo_y == hi_y and lo_x == hi_x:
                # Плитка из одной ячейки: парам, чья первая точка в ней, границей служит вся сетка.
                single = (y1 == lo_y) & (x1 == lo_x)
                if not single.any():
                    single = None
        # Границы общие для всех пар, если нет таких пар: сравнение со скалярами.
        uniform = single is None
        limit_arrays: list["np.ndarray"] = []

        def limits_of(indices: "np.ndarray") -> tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:
            """Границы (y_min, y_max, x_min, x_max) пар indices; массивы по всем парам строятся при первой нужде."""
            if not limit_arrays:
                y_lo = np.full(count, lo_y, dtype=np.int64)
                y_hi = np.full(count, hi_y, dtype=np.int64)
                x_lo = np.full(count, lo_x, dtype=np.int64)
                x_hi = np.full(count, hi_x, dtype=np.int64)
                if single is not None:
                    y_lo[single] = 0
                    y_hi[single] = self.height - 1
                    x_lo[single] = 0
                    x_hi[single] = self.width - 1
                limit_arrays.extend([y_lo, y_hi, x_lo, x_hi])
            return (
                limit_arrays[0][indices], limit_arrays[1][indices], limit_arrays[2][indices], limit_arrays[3][indices]
            )

        # Пары, которым вторая ячейка ещё не выбрана, по возрастанию номера: в этом порядке им достаются числа
        # генератора — те же, что при обходе по маске (смещения dy и dx длиной в число оставшихся пар на попытку).
        idx = np.arange(count)
        if radius is not None and count:
            radius_sq = radius * radius
            merged = self._merged_offset_draws()
            first_y = y1
            first_x = x1
            attempts = 0
            # Общие границы: 0 ≤ c − lo ≤ hi − lo проверяется одним сравнением без знака.
            span_y = np.uint64(hi_y - lo_y) if hi_y >= lo_y else None
            span_x = np.uint64(hi_x - lo_x) if hi_x >= lo_x else None
            unsigned_bounds = uniform and span_y is not None and span_x is not None
            while attempts < 50 and idx.size > self._SAMPLE_SCALAR_LIMIT:
                remaining = int(idx.size)
                if merged:
                    offsets = rng.integers(-radius, radius + 1, 2 * remaining)
                    dy = offsets[:remaining]
                    dx = offsets[remaining:]
                else:
                    dy = rng.integers(-radius, radius + 1, remaining)
                    dx = rng.integers(-radius, radius + 1, remaining)
                attempts += 1
                cy = first_y + dy
                cx = first_x + dx
                # Смещение в круге радиуса r и не нулевое: 0 < dy² + dx² ≤ r².
                distance_sq = dy * dy
                distance_sq += dx * dx
                ok = distance_sq <= radius_sq
                ok &= distance_sq != 0
                if unsigned_bounds:
                    ok &= (cy - lo_y if lo_y else cy).view(np.uint64) <= span_y
                    ok &= (cx - lo_x if lo_x else cx).view(np.uint64) <= span_x
                elif uniform:
                    ok &= (cy >= lo_y) & (cy <= hi_y) & (cx >= lo_x) & (cx <= hi_x)
                else:
                    y_lo, y_hi, x_lo, x_hi = limits_of(idx)
                    ok &= (cy >= y_lo) & (cy <= y_hi) & (cx >= x_lo) & (cx <= x_hi)
                chosen = np.flatnonzero(ok)
                hit = idx[chosen]
                y2[hit] = cy[chosen]
                x2[hit] = cx[chosen]
                np.logical_not(ok, out=ok)
                rest = np.flatnonzero(ok)
                idx = idx[rest]
                first_y = first_y[rest]
                first_x = first_x[rest]
            if attempts < 50 and idx.size:
                # Оставшихся пар мало: проверка поэлементно дешевле десятка вызовов numpy на попытку.
                left = idx.tolist()
                left_y = first_y.tolist()
                left_x = first_x.tolist()
                limits = None if uniform else list(zip(*(array.tolist() for array in limits_of(idx))))
                while attempts < 50 and left:
                    remaining = len(left)
                    if merged:
                        offsets = rng.integers(-radius, radius + 1, 2 * remaining).tolist()
                        dy_list = offsets[:remaining]
                        dx_list = offsets[remaining:]
                    else:
                        dy_list = rng.integers(-radius, radius + 1, remaining).tolist()
                        dx_list = rng.integers(-radius, radius + 1, remaining).tolist()
                    attempts += 1
                    keep: list[int] = []
                    for position in range(remaining):
                        step_y = dy_list[position]
                        step_x = dx_list[position]
                        if step_y * step_y + step_x * step_x <= radius_sq and (step_y or step_x):
                            cell_y = left_y[position] + step_y
                            cell_x = left_x[position] + step_x
                            if limits is not None:
                                lo_y, hi_y, lo_x, hi_x = limits[position]
                            if lo_y <= cell_y <= hi_y and lo_x <= cell_x <= hi_x:
                                pair = left[position]
                                y2[pair] = cell_y
                                x2[pair] = cell_x
                                continue
                        keep.append(position)
                    if len(keep) != remaining:
                        left = [left[position] for position in keep]
                        left_y = [left_y[position] for position in keep]
                        left_x = [left_x[position] for position in keep]
                        if limits is not None:
                            limits = [limits[position] for position in keep]
                idx = np.asarray(left, dtype=np.int64)
        while idx.size:
            y_lo, y_hi, x_lo, x_hi = limits_of(idx)
            cy = rng.integers(y_lo, y_hi + 1)
            cx = rng.integers(x_lo, x_hi + 1)
            ok = ~((cy == y1[idx]) & (cx == x1[idx]))
            hit = idx[ok]
            y2[hit] = cy[ok]
            x2[hit] = cx[ok]
            idx = idx[~ok]
        return y2, x2

    def _sampler_check_due(self, kind: str) -> bool:
        """Пора ли сверить скомпилированную выборку пар с выборкой numpy: первые вызовы и затем периодически."""
        calls = self._sampler_calls[kind]
        self._sampler_calls[kind] = calls + 1
        if calls < self._SAMPLER_VERIFY_FIRST:
            return True
        return self._sampler_verify_every > 0 and calls % self._sampler_verify_every == 0

    def _sampler_checked(self, kind: str, matched: bool) -> None:
        """Итог сверки выборок (§5.3): при расхождении скомпилированная выборка отключается до конца работы."""
        LOGGER.event(
            "layout.pair_sampling.compiled.verify",
            section=PAIR_SELECTION,
            data={
                "kind": kind,
                "call": self._sampler_calls[kind] - 1,
                "identical": matched,
                "note": "скомпилированная выборка пар сверена с выборкой numpy: те же пары и то же состояние "
                "генератора (§5.3, §5.9.1, §5.9.3)",
            },
        )
        if not matched and self._compiled_sampler is not None:
            self._compiled_sampler.log_mismatch(kind=kind, step=None)
            self._compiled_sampler = None

    def _sample_pairs_long_np(self, count: int, radius: int | None) -> "np.ndarray":
        """Пары дальней фазы (§5.3): первая точка — любая, вторая — ячейка в радиусе r; массив count × 4."""
        sampler = self._compiled_sampler
        if sampler is None:
            return self._sample_pairs_long_numpy(count, radius)
        if not self._sampler_check_due("long"):
            return sampler.long_pairs(self._np_rng, self._pos_np, count, radius)
        # Сверка: выборка numpy идёт на рабочем генераторе (её пары и уходят в шаг), скомпилированная — на копии.
        clone = sampler.clone_generator(self._np_rng)
        reference = self._sample_pairs_long_numpy(count, radius)
        compiled = sampler.long_pairs(clone, self._pos_np, count, radius)
        self._sampler_checked(
            "long",
            bool(np.array_equal(reference, compiled))
            and clone.bit_generator.state == self._np_rng.bit_generator.state,
        )
        return reference

    def _sample_pairs_long_numpy(self, count: int, radius: int | None) -> "np.ndarray":
        """Пары дальней фазы (§5.3) выборкой numpy: эталон для скомпилированной выборки."""
        idx_a = self._np_rng.integers(0, self._point_count, count)
        y1 = self._pos_np[idx_a, 0]
        x1 = self._pos_np[idx_a, 1]
        y2, x2 = self._sample_nearby_np(y1, x1, radius, None)
        return np.stack([y1, x1, y2, x2], axis=1)

    def _pair_similarity_np(self, idx_a: "np.ndarray", idx_b: "np.ndarray") -> "np.ndarray":
        """sim(a, b) для массива пар индексов без порога (§2.2.4) — для ранней отсечки пар (§5.9.1)."""
        if self._packed_codes is not None:
            # Число общих бит — popcount(a & b) по 64-битным словам: то же целое, что сумма произведений бит.
            common = np.bitwise_count(self._packed_codes[idx_a] & self._packed_codes[idx_b]).sum(axis=1, dtype=np.int64)
        else:
            common = (self._code_matrix[idx_a].astype(np.int64) * self._code_matrix[idx_b].astype(np.int64)).sum(axis=1)
        ones_a = self._ones_array[idx_a].astype(np.float64)
        ones_b = self._ones_array[idx_b].astype(np.float64)
        if self._similarity == "cosine":
            denominator = np.sqrt(ones_a * ones_b)
        else:
            denominator = ones_a + ones_b - common
        sim = np.divide(
            common.astype(np.float64),
            denominator,
            out=np.zeros(idx_a.size, dtype=np.float64),
            where=(denominator > 0) & (ones_a > 0) & (ones_b > 0),
        )
        sim[idx_a == idx_b] = 1.0
        return sim

    def _sample_pairs_short_tile_np(
        self,
        count: int,
        *,
        pair_radius: int | None,
        similarity_cutoff: float | None,
        weighted_first: bool,
        step_index: int | None,
        candidate_indices: "np.ndarray | None" = None,
        bounds: tuple[int, int, int, int] | None = None,
    ) -> "np.ndarray":
        """Пары ближней фазы (§5.3, §5.9.1, §5.9.3) массивом: первая точка по весам −ln Ê, вторая рядом,
        пары с sim < отсечки отбрасываются; не добранное за 5·count попыток — как в дальней фазе."""
        candidates = (
            np.arange(self._point_count, dtype=np.int64) if candidate_indices is None else np.asarray(candidate_indices)
        )
        if candidates.size == 0:
            return np.zeros((0, 4), dtype=np.int64)
        weights = None
        total = 0.0
        if weighted_first and self._short_energy_weights:
            if self._short_weights_source is not self._short_energy_weights:
                # Веса меняются раз в energy_recalc_every шагов (§5.9.3): массив строится один раз на список.
                self._short_weights_np = np.asarray(self._short_energy_weights, dtype=np.float64)
                self._short_weights_source = self._short_energy_weights
            weights = self._short_weights_np[candidates]
            total = float(weights.sum())
            if not total > 0.0:
                weights = None
        sampler = self._compiled_sampler if self._packed_codes is not None else None

        def probabilities_of() -> "np.ndarray | None":
            """Вероятности первых точек P(p) ∝ −ln Ê (§5.9.3) для выборки numpy."""
            return None if weights is None else weights / total

        if sampler is None:
            pairs, rejected_similarity, attempts = self._short_tile_pairs_numpy(
                count, candidates, probabilities_of(), pair_radius, similarity_cutoff, bounds
            )
        else:
            compiled_arguments = dict(
                pos=self._pos_np,
                grid=self._grid_np,
                packed_codes=self._packed_codes,
                ones=self._ones_float64,
                cosine=self._similarity == "cosine",
                similarity_cutoff=similarity_cutoff,
                count=count,
                radius=pair_radius,
                bounds=bounds,
            )
            if not self._sampler_check_due("short"):
                pairs, rejected_similarity, attempts = sampler.short_tile_pairs(
                    self._np_rng, candidates, weights, total, **compiled_arguments
                )
            else:
                # Сверка: выборка numpy идёт на рабочем генераторе (её пары и уходят в шаг), скомпилированная — на копии.
                clone = sampler.clone_generator(self._np_rng)
                pairs, rejected_similarity, attempts = self._short_tile_pairs_numpy(
                    count, candidates, probabilities_of(), pair_radius, similarity_cutoff, bounds
                )
                compiled = sampler.short_tile_pairs(clone, candidates, weights, total, **compiled_arguments)
                self._sampler_checked(
                    "short",
                    bool(np.array_equal(pairs, compiled[0]))
                    and (rejected_similarity, attempts) == compiled[1:]
                    and clone.bit_generator.state == self._np_rng.bit_generator.state,
                )
        if pairs.shape[0] < count:
            pairs = np.concatenate([pairs, self._sample_pairs_long_np(count - pairs.shape[0], pair_radius)], axis=0)
        event = "layout.short.pair_sampling"
        if LOGGER.should_log(event):
            LOGGER.event(
                event,
                section=OPTIM_PAIR_SELECTION,
                data={
                    "pairs_requested": count,
                    "pairs_generated": int(pairs.shape[0]),
                    "rejected_similarity": rejected_similarity,
                    "attempts": attempts,
                    "pair_radius": "auto" if pair_radius is None else pair_radius,
                    "step": 0 if step_index is None else step_index,
                    "weighted": weighted_first,
                    "similarity_cutoff": similarity_cutoff,
                    "candidate_indices": -1 if candidate_indices is None else int(candidates.size),
                    "bounds": bounds,
                },
                force=True,
            )
        return pairs

    def _short_tile_pairs_numpy(
        self,
        count: int,
        candidates: "np.ndarray",
        probabilities: "np.ndarray | None",
        pair_radius: int | None,
        similarity_cutoff: float | None,
        bounds: tuple[int, int, int, int] | None,
    ) -> tuple["np.ndarray", int, int]:
        """Пары ближней фазы одной плитки выборкой numpy (§5.3, §5.9.1, §5.9.3): (пары, отброшено, попыток).

        Эталон для скомпилированной выборки: первая точка по весам, вторая рядом, пары с sim < отсечки
        отбрасываются, пока не набрано count пар или не исчерпано 5·count попыток.
        """
        rows: list["np.ndarray"] = []
        collected = 0
        attempts = 0
        rejected_similarity = 0
        max_attempts = max(count * 5, count)
        while collected < count and attempts < max_attempts:
            batch = min(count - collected, max_attempts - attempts)
            idx_a = self._np_rng.choice(candidates, size=batch, p=probabilities)
            y1 = self._pos_np[idx_a, 0]
            x1 = self._pos_np[idx_a, 1]
            y2, x2 = self._sample_nearby_np(y1, x1, pair_radius, bounds)
            keep = np.ones(batch, dtype=bool)
            if similarity_cutoff is not None:
                idx_b = self._grid_np[y2, x2]
                has_b = idx_b >= 0
                if has_b.any():
                    sim = self._pair_similarity_np(idx_a[has_b], idx_b[has_b])
                    rejected = sim < similarity_cutoff
                    keep[np.flatnonzero(has_b)[rejected]] = False
                    rejected_similarity += int(rejected.sum())
            attempts += batch
            rows.append(np.stack([y1[keep], x1[keep], y2[keep], x2[keep]], axis=1))
            collected += int(keep.sum())
        pairs = np.concatenate(rows, axis=0)[:count] if rows else np.zeros((0, 4), dtype=np.int64)
        return pairs, rejected_similarity, attempts

    def _sample_pairs_short_np(
        self,
        count: int,
        *,
        pair_radius: int | None,
        similarity_cutoff: float | None,
        weighted_first: bool,
        step_index: int | None,
        partitions: int,
    ) -> "np.ndarray":
        """Пары ближней фазы по случайным плиткам сетки (§5.10): квота каждой плитки — её доля пар."""
        if partitions <= 1:
            return self._sample_pairs_short_tile_np(
                count,
                pair_radius=pair_radius,
                similarity_cutoff=similarity_cutoff,
                weighted_first=weighted_first,
                step_index=step_index,
            )
        row_parts = max(1, int(round(math.sqrt(partitions))))
        col_parts = max(1, math.ceil(partitions / row_parts))
        segments_y = self._random_segments(self.height, row_parts)
        segments_x = self._random_segments(self.width, col_parts)
        y_ends = np.asarray([end for _, end in segments_y], dtype=np.int64)
        x_ends = np.asarray([end for _, end in segments_x], dtype=np.int64)
        tile_count = row_parts * col_parts
        sampler = self._compiled_sampler
        if sampler is not None:
            # Точки каждой плитки по возрастанию номера — те же списки, что даёт отбор по номеру плитки ниже.
            grouped, group_starts = sampler.tile_groups(self._pos_np, y_ends, x_ends, col_parts, tile_count)
            tile_of_point = None
        else:
            tile_row = np.searchsorted(y_ends, self._pos_np[:, 0], side="left")
            tile_col = np.searchsorted(x_ends, self._pos_np[:, 1], side="left")
            tile_of_point = tile_row * col_parts + tile_col
        LOGGER.event(
            "layout.short.tiles",
            section=PARALLEL_PROCESSING,
            data={
                "partitions": tile_count,
                "row_parts": row_parts,
                "col_parts": col_parts,
                "segments_y": segments_y,
                "segments_x": segments_x,
                "points": self._point_count,
            },
        )
        order = list(range(tile_count))
        self._rng.shuffle(order)
        base = count // tile_count
        extra = count % tile_count
        chunks: list["np.ndarray"] = []
        produced = 0
        for position, tile_idx in enumerate(order):
            quota = base + (1 if position < extra else 0)
            if quota <= 0:
                continue
            if tile_of_point is None:
                candidates = grouped[group_starts[tile_idx] : group_starts[tile_idx + 1]]
            else:
                candidates = np.flatnonzero(tile_of_point == tile_idx)
            if candidates.size == 0:
                continue
            y_start, y_end = segments_y[tile_idx // col_parts]
            x_start, x_end = segments_x[tile_idx % col_parts]
            pairs = self._sample_pairs_short_tile_np(
                quota,
                pair_radius=pair_radius,
                similarity_cutoff=similarity_cutoff,
                weighted_first=weighted_first,
                step_index=step_index,
                candidate_indices=candidates,
                bounds=(y_start, y_end, x_start, x_end),
            )
            chunks.append(pairs)
            produced += int(pairs.shape[0])
        remaining = count - produced
        if remaining > 0:
            chunks.append(
                self._sample_pairs_short_tile_np(
                    remaining,
                    pair_radius=pair_radius,
                    similarity_cutoff=similarity_cutoff,
                    weighted_first=weighted_first,
                    step_index=step_index,
                )
            )
        LOGGER.event(
            "layout.short.partition_pairs",
            section=PARALLEL_PROCESSING,
            data={
                "requested": count,
                "produced": produced + max(0, remaining),
                "partitions": tile_count,
                "remaining_fallback": remaining,
            },
        )
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 4), dtype=np.int64)

    def _evaluate_pairs_torch(
        self,
        pairs_np: "np.ndarray",
        *,
        mode: str,
        eval_short: bool,
        kernel: str,
        local_radius: int | None,
        step_index: int | None,
    ) -> "np.ndarray | None":
        """Энергии всех пар шага на torch (§5.10, §5.12): по всему V или по подмножеству B (§5.9.4).

        Возвращает принятые обмены массивом m × 4 (y1, x1, y2, x2) в порядке пар шага (§5.6); None — движка нет.
        """
        engine = self._torch_engine
        if engine is None or pairs_np.shape[0] == 0:
            return np.zeros((0, 4), dtype=np.int64) if engine is not None else None
        sampler = self._compiled_sampler
        occupied: "np.ndarray | None" = None
        if sampler is not None:
            # Те же пары, что даёт отбор numpy ниже: без пар из двух пустых ячеек, непустая точка первая.
            occupied = sampler.occupied_pairs(np.ascontiguousarray(pairs_np, dtype=np.int64), self._grid_np)
            idx_a, idx_b, y1, x1, y2, x2 = (occupied[:, column] for column in range(6))
        else:
            y1, x1, y2, x2 = (pairs_np[:, i].copy() for i in range(4))
            idx_a = self._grid_np[y1, x1]
            idx_b = self._grid_np[y2, x2]
            keep = (idx_a >= 0) | (idx_b >= 0)
            y1, x1, y2, x2, idx_a, idx_b = (arr[keep] for arr in (y1, x1, y2, x2, idx_a, idx_b))
            # Непустая точка всегда первая (вторая ячейка может быть пустой: перенос в пустую клетку).
            flip = idx_a < 0
            idx_a, idx_b = np.where(flip, idx_b, idx_a), np.where(flip, idx_a, idx_b)
            y1, y2 = np.where(flip, y2, y1), np.where(flip, y1, y2)
            x1, x2 = np.where(flip, x2, x1), np.where(flip, x1, x2)
        eval_indices: list[int] | None = None
        if mode == "long" and self._long_subset_size is not None and self._point_count > self._long_subset_size:
            if eval_short:
                # Гауссова окрестность (расширение статьи) — локальная сумма: подмножество B (§5.9.4) даёт
                # несмещённую, но шумную оценку. Оно допустимо, пока в окрестность радиуса r ожидаемо попадает
                # не меньше long_subset_min_neighbours точек B; при малых r энергия считается по всему V.
                density = self._point_count / float(self.height * self.width)
                expected = (self._long_subset_size / self._point_count) * density * math.pi * float(local_radius or 0) ** 2
                if expected >= self._long_subset_min_neighbours:
                    eval_indices = self._ensure_long_subset(step_index)
            else:
                eval_indices = self._ensure_long_subset(step_index)
        if occupied is not None:
            pos_a = occupied[:, 2:4]
            pos_b = occupied[:, 4:6]
        else:
            pos_a = np.stack([y1, x1], axis=1)
            pos_b = np.stack([y2, x2], axis=1)
        euclidean = self._distance_mode("short" if eval_short else "long") == "euclidean"
        host_state: dict[str, "np.ndarray"] = {}
        sample_offsets: tuple["np.ndarray", "np.ndarray"] | None = None
        if mode == "long" and eval_short and eval_indices is None and idx_a.size:
            # Подмножество B уже не годится (в окрестность попадает слишком мало его точек): энергия
            # окрестности оценивается по выборке её ячеек — у каждой пары свои ячейки (§5.9.4).
            sample_offsets = self._neighbourhood_sample(pos_a, pos_b, local_radius, step_index)
        elif mode == "short" and self._neighbourhood_sample_side is not None and idx_a.size:
            # Ближняя фаза большой раскладки: окрестность целиком, но только по ячейкам круга.
            reach = TorchPairEnergyEngine.window_reach(pos_a, pos_b, local_radius)
            if (2 * reach + 1) ** 2 < self._point_count:
                sample_offsets = self._disc_offsets(reach)

        def dense_energies(limit: int | None = None) -> tuple["np.ndarray", "np.ndarray"]:
            """Исходный проход движка; limit — только первые пары шага (сверка на первых порциях)."""
            head = slice(None, limit)
            return engine.pair_energies(
                idx_a[head],
                idx_b[head],
                pos_a[head],
                pos_b[head],
                neighbourhood=eval_short,
                local_radius=local_radius if eval_short else None,
                kernel=kernel,
                euclidean=euclidean,
                distance_eps=self._distance_eps,
                kernel_width=self._long_kernel_width,
                lambda_threshold=self._lambda,
                eta=self._eta,
                eval_indices=eval_indices,
                sample_offsets=sample_offsets,
                **host_state,
            )

        def sparse_energies(signs_only: bool, limit: int | None = None) -> tuple["np.ndarray", "np.ndarray"] | None:
            """φ_c, φ_s по строкам сходства: знак — ядром PairSignKernel, суммы — редукцией исходной формы."""
            head = slice(None, limit)
            return self._sparse_pair_energies(
                sparse,
                idx_a[head],
                idx_b[head],
                pos_a[head],
                pos_b[head],
                occupied=None if occupied is None else occupied[head],
                neighbourhood=eval_short,
                local_radius=local_radius,
                kernel=kernel,
                euclidean=euclidean,
                signs_only=signs_only,
            )

        try:
            # При жёстком пороге суммы по всему V (§5.5.1) и по окрестности (§5.5.2) сводятся к ненулевым sim_λ:
            # те же φ_c, φ_s считаются по строкам сходства.
            sparse = self._sparse_long_engine() if eval_indices is None and idx_a.size else None
            # Расстановка на устройстве нужна расчёту по строкам сходства; слитому ядру — массивы хоста
            # (и по всему V, и по подмножеству B).
            host_state = self._host_layout_state(device_needed=sparse is not None)
            if sparse is None:
                energy_c, energy_s = dense_energies()
            else:
                energy_c, energy_s = self._sparse_checked(
                    "neighbourhood" if eval_short else "long",
                    sparse_energies,
                    dense_energies,
                    step_index=step_index,
                    swap_when_greater=eval_short,
                    pair_count=int(idx_a.size),
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.event(
                "layout.gpu.pair_energy.failed",
                section=GPU_IMPLEMENTATION,
                data={"error": str(exc), "pairs": int(idx_a.size), "mode": mode},
            )
            if self._point_count > _TORCH_FALLBACK_POINT_LIMIT:
                # Запасные пути (OpenGL с матрицей N×N, процессор) на таком числе точек не работают.
                raise
            self._torch_engine = None
            return None
        # Окрестность (§5.5.2, и gauss-дальняя): обмен выгоден при φ_s > φ_c; дальняя §5.5.1: при φ_s < φ_c.
        deltas = (energy_s - energy_c) if eval_short else (energy_c - energy_s)
        if LOGGER.should_log("layout.energy.pairs.gpu"):
            LOGGER.event(
                "layout.energy.pairs.gpu",
                section=GPU_IMPLEMENTATION,
                data={
                    "step": 0 if step_index is None else step_index,
                    "mode": mode,
                    "pairs": int(idx_a.size),
                    "eval_points": self._point_count if eval_indices is None else len(eval_indices),
                    "sampled_cells": 0 if sample_offsets is None else int(sample_offsets[0].size),
                    "kernel": kernel if eval_short else "distance",
                    "local_radius": local_radius if eval_short else None,
                    "favourable": int((deltas > 0.0).sum()),
                    "device": engine.device,
                },
                force=True,
            )
        deltas = deltas.astype(np.float64)
        # Жадное правило §5.6 — пары с выгодной энергией; с температурой — отбор Metropolis: те же пары
        # в том же порядке, что даёт _select_swaps, но без обхода всех пар шага в Python.
        accepted = np.flatnonzero(self._accepted_swaps(deltas))
        return np.stack([y1[accepted], x1[accepted], y2[accepted], x2[accepted]], axis=1)

    def _pull_device(self) -> None:
        """Массивы расстановки хоста догоняют расстановку на устройстве после шагов на нём."""
        if not self._device_ahead:
            return
        self._device_ahead = False
        cells = (self._device_owner or self._device_state).download()
        grid = np.full((self.height, self.width), -1, dtype=np.int64)
        grid[cells[:, 0], cells[:, 1]] = np.arange(self._point_count, dtype=np.int64)
        self._pos_np = cells
        self._grid_np = grid
        self._arrays_stale = False
        self._lists_stale = True

    def _device_long_step(self, pairs_per_step: int, pair_radius: int, step_index: int | None) -> int | None:
        """Шаг дальней фазы с гауссовым ядром целиком на устройстве (§5.3, §5.5, §5.6, §5.10, §5.12).

        None — шаг на устройстве неприменим (нет слитого расчёта с общими битами на лету, нет выборки
        окрестности, энергии считаются по подмножеству B) или отказал: шаг выполняется на хосте.
        """
        engine = self._torch_engine
        if (
            engine is None
            or self._long_kernel != "gauss"
            or self._neighbourhood_sample_side is None
            or self._long_subset_size is not None
        ):
            return None
        local_radius = self._long_kernel_radius(pair_radius)
        energy_arguments = dict(
            kernel="gauss",
            euclidean=self._distance_mode("short") == "euclidean",
            distance_eps=self._distance_eps,
            kernel_width=self._long_kernel_width,
            lambda_threshold=self._lambda,
            eta=self._eta,
        )
        try:
            state = self._device_state
            if state is None:
                fused = engine.fused_on_the_fly()
                if fused is None:
                    self._device_step_enabled = False
                    return None
                state = self._device_state = DeviceLongStep(engine, fused, seed=self._device_seed)
            if state.version != self._positions_version:
                state.upload(self._positions_array(), self._positions_version)
            reference = None
            if self._device_steps % _DEVICE_STEP_VERIFY_EVERY == 0:
                def reference(idx_a, idx_b, pos_a, pos_b, window, reach):  # noqa: ANN001, ANN202
                    cells = state.download()
                    engine.update_positions(cells[:, 0], cells[:, 1])
                    return engine._pair_energies_reference(
                        idx_a, idx_b, pos_a, pos_b, neighbourhood=True, local_radius=local_radius,
                        sample_offsets=window, window_reach=reach, **energy_arguments,
                    )
            applied, length = state.step(
                pairs=pairs_per_step,
                radius=pair_radius,
                local_radius=local_radius,
                offsets=lambda reach: self._neighbourhood_offsets(reach, step_index),
                temperature=self._swap_temperature,
                energy_arguments=energy_arguments,
                step_index=step_index,
                reference=reference,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.event(
                "layout.device_step.failed",
                section=GPU_IMPLEMENTATION,
                data={"error": str(exc), "step": 0 if step_index is None else step_index,
                      "note": "шаг на устройстве отключён: раскладка продолжается шагами на хосте"},
                force=True,
            )
            self._pull_device()
            self._device_step_enabled = False
            self._device_state = None
            return None
        self._device_steps += 1
        if applied:
            self._positions_version += 1
            state.version = self._positions_version
            self._device_ahead = True
            self._device_owner = state
            self._long_swap_radius_sum += length
            self._long_swap_count += applied
            self._gpu_positions_dirty = True
            # Расстановка движка (исходный проход, строки сходства) устарела целиком.
            self._torch_pending_moves.clear()
            self._torch_dirty_points.clear()
            self._torch_dirty_cells.clear()
            self._torch_tracked_changes = 0
        return applied

    def _device_short_state(self) -> DeviceShortStep | None:
        """Расстановка и ядро окна на устройстве для ближней фазы большой раскладки (§5.5.2, §5.7).

        None — недоступно: нет движка torch с CUDA, порог сходства — сигмоида (ядро считает жёсткий порог
        §7.1.7) либо шаг на устройстве отключён.
        """
        engine = self._torch_engine
        if engine is None or not self._device_short_enabled or self._eta is not None:
            return None
        state = self._device_short
        if state is None:
            if not WindowEnergyKernel.supported(engine):
                self._device_short_enabled = False
                return None
            state = self._device_short = DeviceShortStep(engine, WindowEnergyKernel(engine), seed=self._device_seed)
        if state.version != self._positions_version:
            state.upload(self._positions_array(), self._positions_version)
        return state

    def _device_reference_energies(
        self,
        state: DeviceShortStep,
        idx_a: "np.ndarray",
        idx_b: "np.ndarray",
        pos_a: "np.ndarray",
        pos_b: "np.ndarray",
        reach: int,
        *,
        local_radius: float,
        euclidean: bool,
        need_swapped: bool = True,
    ) -> tuple["np.ndarray", "np.ndarray"]:
        """φ_c, φ_s пар исходным проходом движка по расстановке устройства — для сверки ядра окна."""
        engine = self._torch_engine
        cells = state.download()
        engine.update_positions(cells[:, 0], cells[:, 1])
        # Расстановка движка заменена целиком: учёт обменов для неё начинается заново.
        self._torch_pending_moves.clear()
        self._torch_dirty_points.clear()
        self._torch_dirty_cells.clear()
        self._torch_tracked_changes = 0
        self._torch_positions_version = -1
        return engine._pair_energies_reference(
            idx_a, idx_b, pos_a, pos_b,
            neighbourhood=True, local_radius=float(local_radius), kernel="inverse", euclidean=euclidean,
            distance_eps=self._distance_eps, kernel_width=self._long_kernel_width,
            lambda_threshold=self._lambda, eta=self._eta, need_swapped=need_swapped, window_reach=reach,
        )

    def _device_short_step(
        self,
        pairs_per_step: int,
        pair_radius: int,
        local_radius: int,
        step_index: int | None,
        short_config: ShortLayoutOptimConfig | None,
    ) -> int | None:
        """Шаг ближней фазы режима статьи целиком на устройстве (§5.3, §5.5.2, §5.6, §5.9.1, §5.9.3, §5.10, §5.12).

        None — шаг на устройстве неприменим или отказал: шаг выполняется на хосте.
        """
        try:
            state = self._device_short_state()
            if state is None:
                return None
            euclidean = self._distance_mode("short") == "euclidean"
            similarity_cutoff = (
                self._lambda
                if short_config is None or short_config.similarity_cutoff is None
                else short_config.similarity_cutoff
            )
            if short_config is not None and short_config.use_weighted_first_point:
                # §5.9.3: энергии всех точек пересчитываются периодически, P(p) ∝ −ln Ê(p, r).
                key = (float(short_config.energy_radius), float(short_config.energy_eps))
                due = (
                    self._device_weights_step is None
                    or step_index is None
                    or key != self._device_weights_key
                    or step_index - self._device_weights_step >= short_config.energy_recalc_every
                )
                if due:
                    state.refresh_weights(
                        short_config.energy_radius,
                        energy_eps=short_config.energy_eps,
                        distance_eps=self._distance_eps,
                        lambda_threshold=self._lambda,
                        step_index=step_index,
                    )
                    self._device_weights_step = 0 if step_index is None else step_index
                    self._device_weights_key = key
            radius = self._short_neighbourhood_radius(local_radius, pair_radius)
            reference = None
            if self._device_steps % _DEVICE_STEP_VERIFY_EVERY == 0:
                def reference(idx_a, idx_b, pos_a, pos_b, reach):  # noqa: ANN001, ANN202
                    return self._device_reference_energies(
                        state, idx_a, idx_b, pos_a, pos_b, reach, local_radius=radius, euclidean=euclidean
                    )
            applied, _ = state.short_step(
                pairs=pairs_per_step,
                radius=pair_radius,
                local_radius=radius,
                # Порог 0 — без ранней отсечки (§5.9.1): сходство пар не считается, пары не добираются.
                similarity_cutoff=similarity_cutoff if similarity_cutoff > 0.0 else None,
                euclidean=euclidean,
                distance_eps=self._distance_eps,
                lambda_threshold=self._lambda,
                step_index=step_index,
                reference=reference,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.event(
                "layout.device_step.failed",
                section=GPU_IMPLEMENTATION,
                data={"error": str(exc), "step": 0 if step_index is None else step_index, "mode": "short",
                      "note": "шаг на устройстве отключён: раскладка продолжается шагами на хосте"},
                force=True,
            )
            self._pull_device()
            self._device_short_enabled = False
            self._device_short = None
            return None
        self._device_steps += 1
        if applied:
            self._positions_version += 1
            state.version = self._positions_version
            self._device_ahead = True
            self._device_owner = state
            self._gpu_positions_dirty = True
            # Расстановка движка (исходный проход, строки сходства) устарела целиком.
            self._torch_pending_moves.clear()
            self._torch_dirty_points.clear()
            self._torch_dirty_cells.clear()
            self._torch_tracked_changes = 0
        return applied

    def _device_point_energies(self, radius: int) -> "np.ndarray | None":
        """E(c, r) всех точек (§5.7) ядром окна на устройстве; None — ядро недоступно.

        Первый расчёт сверяется с исходным проходом движка на части точек.
        """
        state = self._device_short_state()
        if state is None:
            return None
        energies = state.point_energies(
            float(radius), distance_eps=self._distance_eps, lambda_threshold=self._lambda
        ).cpu().numpy()
        if not self._device_energies_verified:
            self._device_energies_verified = True
            sample = np.linspace(0, self._point_count - 1, num=min(self._point_count, 2048)).astype(np.int64)
            cells = state.download()[sample]
            reach = WindowEnergyKernel.reach(0.0, float(radius))
            expected, _ = self._device_reference_energies(
                state, sample, np.full(sample.size, -1, dtype=np.int64), cells, cells, reach,
                local_radius=float(radius), euclidean=True, need_swapped=False,
            )
            zeros = np.zeros(sample.size, dtype=np.float32)
            if not state.kernel.verify(energies[sample], zeros, expected, zeros, kind="points"):
                raise RuntimeError("device point energies differ from the reference pass")
        return energies

    def _pairs_within_radius(
        self, pairs_np: "np.ndarray", pair_radius: int, *, mode: str, step_index: int | None
    ) -> "np.ndarray":
        """Пары шага, вторая ячейка которых лежит в радиусе выбора r от первой (§5.3).

        Вторая ячейка выбирается попытками в круге радиуса r; если попытки исчерпаны (ячейка у границы сетки
        или плитки), берётся любая ячейка сетки или плитки — возможно, дальше r. Окрестность такой пары
        (её радиус не меньше ‖p1 − p2‖, §5.5.2) растягивает окно ячеек всех пар шага: одна дальняя пара на
        сотни тысяч пар делает шаг на порядки дороже. В больших раскладках такие пары в шаг не входят.
        """
        dy = pairs_np[:, 0] - pairs_np[:, 2]
        dx = pairs_np[:, 1] - pairs_np[:, 3]
        near = dy * dy + dx * dx <= int(pair_radius) * int(pair_radius)
        if bool(near.all()):
            return pairs_np
        if LOGGER.should_log("layout.pairs.beyond_radius"):
            LOGGER.event(
                "layout.pairs.beyond_radius",
                section=PAIR_SELECTION,
                data={
                    "step": 0 if step_index is None else step_index,
                    "mode": mode,
                    "pair_radius": int(pair_radius),
                    "pairs": int(pairs_np.shape[0]),
                    "dropped": int(pairs_np.shape[0] - int(near.sum())),
                    "note": "пары, вторая ячейка которых дальше радиуса выбора r, в шаг не входят (§5.3)",
                },
                force=True,
            )
        return pairs_np[near]

    def _neighbourhood_sample(
        self,
        pos_a: "np.ndarray",
        pos_b: "np.ndarray",
        local_radius: int | None,
        step_index: int | None,
    ) -> tuple["np.ndarray", "np.ndarray"] | None:
        """Выборка ячеек окна окрестности пар шага (§5.9.4): смещения dy, dx от ячейки центра пары.

        Окно (2R+1)² ячеек, покрывающее окрестности всех пар шага (§5.5.2), делится на квадратные слои t × t,
        t = ⌊(2R+1) / side⌋; сетка слоёв сдвинута на случайное начало, из каждого слоя берётся одна случайная
        ячейка. Каждая ячейка окна попадает в выборку с одной и той же вероятностью 1/t², поэтому сумма по
        выборке — несмещённая оценка энергии окрестности с общим для всех пар множителем 1/t² (на знак
        Δ = φ_s − φ_c и на отбор обменов множитель не влияет). Слои дают меньший разброс оценки, чем
        независимые случайные ячейки: веса гауссова ядра меняются плавно. Выборка обновляется каждый шаг.

        Если окно мало (t < 2), окрестность считается целиком, но только по ячейкам круга: окрестность пары —
        круг радиуса не больше R − 1 вокруг её центра, угловые ячейки квадратного окна в сумму не входят
        никогда (около трети окна). Сумма та же, слагаемых меньше.
        None — выборка выключена: окрестность считается по всему квадратному окну.
        """
        if self._neighbourhood_sample_side is None:
            return None
        return self._neighbourhood_offsets(
            TorchPairEnergyEngine.window_reach(pos_a, pos_b, local_radius), step_index
        )

    def _neighbourhood_offsets(self, reach: int, step_index: int | None) -> tuple["np.ndarray", "np.ndarray"]:
        """Смещения ячеек окна окрестности полуширины reach: выборка слоями или все ячейки круга (§5.9.4)."""
        side_cells = int(self._neighbourhood_sample_side)
        side = 2 * reach + 1
        stride = side // side_cells
        if stride < 2:
            return self._disc_offsets(reach)
        strata = -(-(side + stride - 1) // stride)
        rng = self._sample_rng
        origin = rng.integers(0, stride, size=2)
        jitter = rng.integers(0, stride, size=(2, strata, strata))
        base = stride * np.arange(strata, dtype=np.int64) - reach
        dy = (base[:, None] - int(origin[0]) + jitter[0]).reshape(-1)
        dx = (base[None, :] - int(origin[1]) + jitter[1]).reshape(-1)
        if LOGGER.should_log("layout.energy.neighbourhood_sample"):
            LOGGER.event(
                "layout.energy.neighbourhood_sample",
                section=OPTIM_SUBSET,
                data={
                    "step": 0 if step_index is None else step_index,
                    "window_reach": reach,
                    "window_cells": side * side,
                    "stratum_side": stride,
                    "sampled_cells": int(dy.size),
                    "note": "энергия окрестности пар оценивается по выборке ячеек окна: одна случайная ячейка "
                    "из каждого слоя t × t (§5.9.4)",
                },
                force=True,
            )
        return dy, dx

    def _disc_offsets(self, reach: int) -> tuple["np.ndarray", "np.ndarray"]:
        """Смещения ячеек окна полуширины reach, которые могут попасть в окрестность пары (§5.5.2).

        Окрестность — круг радиуса не больше reach − 1 вокруг центра пары; центр отстоит от своей ячейки
        не больше чем на полячейки по каждой оси. Ячейка со смещением (dy, dx) лежит от центра не ближе
        √((|dy| − ½)₊² + (|dx| − ½)₊²): смещения, где это больше reach − 1, в окрестность не попадают.
        """
        cached = self._disc_offsets_cache.get(reach)
        if cached is None:
            span = np.arange(-reach, reach + 1, dtype=np.int64)
            dy, dx = np.meshgrid(span, span, indexing="ij")
            near_y = np.maximum(np.abs(dy) - 0.5, 0.0)
            near_x = np.maximum(np.abs(dx) - 0.5, 0.0)
            inside = near_y * near_y + near_x * near_x <= float(reach - 1) ** 2
            cached = (dy[inside].copy(), dx[inside].copy())
            if len(self._disc_offsets_cache) > 64:
                self._disc_offsets_cache.clear()
            self._disc_offsets_cache[reach] = cached
        return cached

    def _sparse_long_engine(self, *, build: bool = True) -> SparseLongEnergy | None:
        """Расчёт энергий по ненулевым sim_λ (§5.5.1, §5.5.2, §5.9.5) — применим при жёстком пороге η = ∞ (§5.4).

        Строки сходства строятся при первом обращении, при росте λ (§7.1.7) прореживаются, при снижении λ
        строятся заново. None — расчёт неприменим (сигмоида: sim_λ нигде не ноль), строк слишком много
        или их ещё нет, а build = False (разовый расчёт не стоит построения строк).
        """
        engine = self._torch_engine
        if not self._sparse_long_enabled or self._eta is not None or engine is None:
            return None
        sparse = self._sparse_long
        if sparse is not None:
            if sparse.rows.lambda_threshold == self._lambda:
                return sparse
            if sparse.rows.prune(self._lambda):
                self._sparse_verify_pending = dict.fromkeys(self._sparse_verify_pending, True)
                return sparse
            self._sparse_long = None
        if not build:
            return None
        if self._sparse_rejected_lambda is not None and self._lambda <= self._sparse_rejected_lambda:
            return None
        max_edges = self._sparse_max_edges
        free_bytes = engine.free_memory_bytes()
        if free_bytes is not None:
            max_edges = min(max_edges, int(free_bytes * self._sparse_memory_fraction) // SimilarityRows.BYTES_PER_EDGE)
        rows = SimilarityRows.build(engine, lambda_threshold=self._lambda, max_edges=max_edges)
        if rows is None:
            self._sparse_rejected_lambda = self._lambda
            return None
        self._sparse_long = SparseLongEnergy(
            engine,
            rows,
            buffer_bytes=self._sparse_buffer_bytes,
            entry_budget=self._sparse_entry_budget,
            screen_min_chunks=self._sparse_screen_min_chunks,
        )
        self._sparse_verify_pending = dict.fromkeys(self._sparse_verify_pending, True)
        return self._sparse_long

    def _pair_sign_kernel(self, sparse: SparseLongEnergy) -> PairSignKernel | None:
        """Ядро знака φ_c − φ_s пар (§5.6, §5.12) по строкам сходства sparse; None — недоступно или отключено."""
        engine = self._torch_engine
        if not self._pair_kernel_enabled or engine is None:
            return None
        kernel = self._pair_kernel
        if kernel is None or kernel.rows is not sparse.rows:
            if not PairSignKernel.supported(engine):
                self._pair_kernel_enabled = False
                return None
            kernel = PairSignKernel(engine, sparse.rows, segment=self._pair_kernel_segment)
            self._pair_kernel = kernel
        return kernel

    def _sparse_pair_energies(
        self,
        sparse: SparseLongEnergy,
        idx_a: "np.ndarray",
        idx_b: "np.ndarray",
        pos_a: "np.ndarray",
        pos_b: "np.ndarray",
        *,
        occupied: "np.ndarray | None",
        neighbourhood: bool,
        local_radius: int | None,
        kernel: str,
        euclidean: bool,
        signs_only: bool,
    ) -> tuple["np.ndarray", "np.ndarray"] | None:
        """φ_c, φ_s пар по строкам сходства (§5.5.1, §5.5.2): знак разности — ядром, суммы — редукцией SparseLongEnergy.

        signs_only = True (жадное правило §5.6): знак φ_c − φ_s большинства пар выносит PairSignKernel; им
        возвращаются условные значения 1 и 0 с тем же знаком разности, что у исходного прохода. Парам, у которых
        разность лежит в пределах ошибки суммирования float32, суммы считает редукция исходной формы — как
        и всем парам при signs_only = False. None — расчёт по строкам на этих парах неприменим.
        """
        sign_kernel = self._pair_sign_kernel(sparse) if signs_only and (not neighbourhood or kernel == "inverse") else None
        if sign_kernel is None:
            if neighbourhood:
                return sparse.neighbourhood_energies(
                    idx_a, idx_b, pos_a, pos_b,
                    local_radius=local_radius, kernel=kernel, euclidean=euclidean,
                    distance_eps=self._distance_eps, kernel_width=self._long_kernel_width, signs_only=signs_only,
                )
            return sparse.pair_energies(idx_a, idx_b, pos_a, pos_b, euclidean=euclidean, signs_only=signs_only)
        pairs = int(idx_a.size)
        sampler = self._compiled_sampler
        if occupied is not None and sampler is not None:
            table, widest = sampler.row_table(
                np.ascontiguousarray(occupied), sparse.rows.degree_np, self._code_class, sign_kernel.stage(pairs)
            )
        else:
            table, widest = sign_kernel.pair_table(idx_a, idx_b, pos_a, pos_b, self._code_class)
        bounds = None if neighbourhood else self._potential_bounds(sparse, pairs, euclidean)
        if neighbourhood:
            # Ближняя фаза считает окрестность (§5.5.2): суммы границы дальней энергии больше не ведутся.
            self._bounds_on = False
        status = sign_kernel.status_from_table(
            table,
            pairs,
            widest,
            kind="neighbourhood" if neighbourhood else "long",
            euclidean=euclidean,
            local_radius=0.0 if local_radius is None else float(local_radius),
            distance_eps=self._distance_eps,
            bounds=bounds,
        )
        sign_kernel.log_status(status, kind="neighbourhood" if neighbourhood else "long")
        bounded = status == PairSignKernel.BOUNDED
        if bounds is not None:
            self._bounds_used(int(bounded.sum()), pairs)
        energy_c = (status == 1).astype(np.float32)
        energy_s = ((status == -1) | bounded).astype(np.float32)
        undecided = np.flatnonzero(status == 0)
        if undecided.size:
            self._undecided_steps += 1
            self._undecided_pairs += int(undecided.size)

            def portion_sums() -> tuple["np.ndarray", "np.ndarray"]:
                """Суммы пар «не определено» по порциям исходной формы (SparseLongEnergy)."""
                if neighbourhood:
                    exact = sparse.neighbourhood_energies(
                        idx_a, idx_b, pos_a, pos_b,
                        local_radius=local_radius, kernel=kernel, euclidean=euclidean,
                        distance_eps=self._distance_eps, kernel_width=self._long_kernel_width, signs_only=True,
                        pending=undecided,
                    )
                else:
                    exact = sparse.pair_energies(
                        idx_a, idx_b, pos_a, pos_b, euclidean=euclidean, signs_only=True, pending=undecided
                    )
                return exact[0][undecided], exact[1][undecided]

            kind_name = "neighbourhood" if neighbourhood else "long"

            def device_sums() -> tuple["np.ndarray", "np.ndarray"]:
                """Те же суммы float32 на устройстве: по малым участкам редукции (первые расчёты сверяются)."""
                slot_sums = sign_kernel.reference_sums(
                    idx_a, idx_b, pos_a, pos_b, undecided,
                    kind=kind_name, euclidean=euclidean,
                    local_radius=local_radius, distance_eps=self._distance_eps,
                )
                if slot_sums is None:
                    return portion_sums()
                if sign_kernel.reference_check_due():
                    expected = portion_sums()
                    sign_kernel.reference_checked(
                        bool(np.array_equal(slot_sums[0], expected[0]) and np.array_equal(slot_sums[1], expected[1])),
                        kind=kind_name, pairs=int(undecided.size),
                    )
                    return expected
                return slot_sums

            # Сначала — на хосте, в порядке сложения редукции исходного прохода (без обращения к устройству).
            sums = None
            replica = self._reduction_replica(sparse)
            if replica is not None:
                sums = replica.pair_sums(
                    idx_a, idx_b, pos_a, pos_b, undecided, self._pos_np,
                    kind=kind_name, euclidean=euclidean,
                    geometry=sign_kernel.reference_geometry(pos_a, pos_b, kind=kind_name, local_radius=local_radius),
                    local_radius=local_radius, distance_eps=self._distance_eps,
                )
                if sums is not None:
                    self._replica_pairs += int(undecided.size)
                    if replica.check_due():
                        expected = device_sums()
                        replica.checked(
                            bool(np.array_equal(sums[0], expected[0]) and np.array_equal(sums[1], expected[1])),
                            kind=kind_name, pairs=int(undecided.size),
                        )
                        sums = expected
            if sums is None:
                sums = device_sums()
            energy_c[undecided] = sums[0]
            energy_s[undecided] = sums[1]
        return energy_c, energy_s

    def _reduction_replica(self, sparse: SparseLongEnergy) -> ReductionReplica | None:
        """Реплика редукции float32 исходного прохода для строк сходства sparse; None — недоступна или отключена."""
        engine = self._torch_engine
        if not self._replica_enabled or engine is None:
            return None
        replica = self._replica
        if replica is None or replica.rows is not sparse.rows:
            replica = ReductionReplica.create(engine, sparse.rows, max_bytes=self._replica_max_bytes)
            if replica is None:
                self._replica_enabled = False
                return None
            self._replica = replica
        return replica if replica.enabled else None

    # Суммы границы потенциала включаются, когда сдвинутых за шаг точек не больше пары / _BOUNDS_ON_DIVISOR,
    # и выключаются, когда их больше пары / _BOUNDS_OFF_DIVISOR: до этого их учёт дороже расчёта строк.
    _BOUNDS_ON_DIVISOR = 8
    _BOUNDS_OFF_DIVISOR = 4
    # Доля снятых границей пар проверяется раз в столько расчётов; ниже _BOUNDS_MIN_SHARE граница
    # выключается на _BOUNDS_COOLDOWN расчётов (пространство ещё не упорядочено).
    _BOUNDS_WINDOW = 64
    _BOUNDS_MIN_SHARE = 0.25
    _BOUNDS_COOLDOWN = 512
    # Граница окупается, когда строк много: с этого числа слагаемых шага (пары × 2 × средняя степень строки)
    # расчёт строк дороже проверки границы и учёта обменов (три запуска ядер на шаг).
    _BOUNDS_MIN_ENTRIES = 3_000_000

    def _potential_bounds(self, sparse: SparseLongEnergy, pairs: int, euclidean: bool) -> PotentialBounds | None:
        """Действующая граница потенциала точек (§5.5.1) для пар дальней фазы; None — выключена или неприменима.

        Граница требует евклидова расстояния (§5.5) и сумм, согласованных с позициями на устройстве: суммы
        строятся при включении, при смене λ (§7.1.7) и периодически, а между этим обновляются обменами шага.
        """
        engine = self._torch_engine
        if not self._bounds_enabled or not euclidean or engine is None:
            return None
        self._bounds_move_limit = max(8, pairs // self._BOUNDS_OFF_DIVISOR)
        entries = 2.0 * pairs * sparse.rows.edges / max(1, self._point_count)
        if self._bounds_cooldown > 0 or entries < self._BOUNDS_MIN_ENTRIES:
            # Граница не используется: её суммы перестают обновляться обменами шага.
            self._bounds_cooldown = max(0, self._bounds_cooldown - 1)
            self._bounds_on = False
            return None
        bounds = self._bounds
        if bounds is None or bounds.rows is not sparse.rows:
            if not PotentialBounds.supported(engine, (self.height, self.width)):
                self._bounds_enabled = False
                return None
            bounds = PotentialBounds(
                engine,
                sparse.rows,
                grid=self._bounds_grid,
                segment=self._pair_kernel_segment,
                similarity=self._similarity,
                grid_shape=(self.height, self.width),
            )
            self._bounds = bounds
            self._bounds_on = False
        if not self._bounds_on:
            if self._last_moved_points > max(4, pairs // self._BOUNDS_ON_DIVISOR):
                return None
            self._bounds_on = True
            self._bounds_window = [0, 0, 0]
        if bounds.rebuild_due:
            bounds.rebuild()
        return bounds

    def _bounds_used(self, bounded: int, pairs: int) -> None:
        """Учёт работы границы потенциала: при малой доле снятых пар она выключается на время."""
        self._bounded_pairs += bounded
        window = self._bounds_window
        window[0] += 1
        window[1] += bounded
        window[2] += pairs
        if window[0] < self._BOUNDS_WINDOW:
            return
        share = window[1] / max(1, window[2])
        self._bounds_window = [0, 0, 0]
        event = "layout.bounds.share"
        if LOGGER.should_log(event):
            LOGGER.event(
                event,
                section=ENERGY_LONG,
                data={
                    "pairs": window[2],
                    "bounded": window[1],
                    "share": share,
                    "lambda_threshold": self._lambda,
                    "note": "доля пар дальней фазы, у которых φ_c < φ_s следует из нижней границы потенциала точек "
                    "(§5.5.1): их строки сходства не считались",
                },
                force=True,
            )
        if share < self._BOUNDS_MIN_SHARE and self._bounds is not None:
            self._bounds_on = False
            self._bounds.invalidate()
            self._bounds_cooldown = self._BOUNDS_COOLDOWN
            self._bounds.log_disabled(reason=f"share {share:.3f} below {self._BOUNDS_MIN_SHARE}")

    def _sparse_verify_limit(self, kind: str, pair_count: int | None) -> int | None:
        """Число первых пар шага, на которых расчёт по строкам сверяется с исходным проходом; None — на всех.

        Сверка на первых sparse_verify_chunks порциях исходного прохода: у этих пар форма порции та же, что
        в проходе по всем парам шага, а стоимость сверки не растёт с числом пар.
        """
        engine = self._torch_engine
        if pair_count is None or engine is None or self._sparse_verify_chunks <= 0:
            return None
        rows = engine.local_chunk_rows() if kind == "neighbourhood" else engine.dense_chunk_rows()
        limit = rows * self._sparse_verify_chunks
        return None if limit >= pair_count else limit

    def _sparse_checked(
        self,
        kind: str,
        sparse_energies: Any,
        dense_energies: Any,
        *,
        step_index: int | None,
        swap_when_greater: bool,
        values_needed: bool = False,
        pair_count: int | None = None,
    ) -> tuple["np.ndarray", "np.ndarray"]:
        """φ_c, φ_s по ненулевым sim_λ со сверкой с исходным проходом движка (§5.5.1, §5.5.2, §5.7).

        sparse_energies(signs_only[, limit]) — расчёт по строкам сходства (None — на этих парах он неприменим),
        dense_energies([limit]) — исходный проход; limit — только первые пары. Сверка — после построения
        и каждого прореживания строк и затем раз в sparse_verify_every вызовов. Если задано pair_count, оба
        способа сравниваются на первых порциях пар шага (_sparse_verify_limit), а шаг считается по строкам;
        иначе сравниваются все пары и в раскладку идёт результат исходного прохода. При расхождении расчёт
        по строкам отключается до конца работы, шаг пересчитывается исходным проходом.
        """
        # При жадном правиле §5.6 нужен только знак φ_c − φ_s; при отжиге и для энергий точек — сами значения.
        signs_only = not values_needed and self._swap_temperature <= 0.0
        verify = self._sparse_verify_pending[kind] or (
            self._sparse_verify_every > 0 and self._sparse_steps_since_verify[kind] >= self._sparse_verify_every
        )
        sparse = self._sparse_long
        energies: tuple["np.ndarray", "np.ndarray"] | None = None
        try:
            if not verify:
                energies = sparse_energies(signs_only)
                if energies is None:
                    return dense_energies()
                self._sparse_steps_since_verify[kind] += 1
                return energies
            limit = self._sparse_verify_limit(kind, pair_count)
            head = () if limit is None else (limit,)
            exact = sparse_energies(False, *head)
            if exact is None:
                return dense_energies()
            screened = sparse_energies(True, *head) if signs_only else None
            dense = dense_energies(*head)
            matched = sparse.matches(
                exact, screened, dense, kind=kind, step=step_index, swap_when_greater=swap_when_greater
            )
            failure = None if matched else "sparse energies differ from the reference pass"
            if matched:
                # Сверка на всех парах: в раскладку идёт исходный проход; на первых порциях — шаг по строкам.
                energies = dense if limit is None else sparse_energies(signs_only)
        except Exception as exc:  # noqa: BLE001
            matched = False
            failure = str(exc)
        self._sparse_verify_pending[kind] = False
        self._sparse_steps_since_verify[kind] = 0
        if not matched:
            self._sparse_long_enabled = False
            self._sparse_long = None
            self._pair_kernel = None
            self._bounds = None
            self._bounds_on = False
            self._replica = None
            LOGGER.event(
                "layout.sparse.energy.disabled",
                section=GPU_IMPLEMENTATION,
                data={
                    "step": 0 if step_index is None else step_index,
                    "kind": kind,
                    "error": failure,
                    "note": "энергии дальше считаются исходными проходами движка (§5.5.1, §5.5.2, §5.12)",
                },
                force=True,
            )
        return energies if energies is not None else dense_energies()

    def positions(self) -> list[tuple[int, int]]:
        return list(self._positions)

    def labels(self) -> list[str]:
        return list(self._labels)

    def values(self) -> list[str]:
        return list(self._values)

    def hues(self) -> list[float]:
        return self._table.hues.tolist()

    def hue_count(self) -> int:
        """Число различных значений известной компоненты стимула (hue) среди точек."""
        return self._table.hue_count()

    def save_json(self, path: str) -> None:
        if not path:
            raise ValueError("path must be provided")
        # Из базового файла сохраняются только метаданные верхнего уровня (подпись кодера, стимулы);
        # список точек пишется заново из самой раскладки.
        base_payload: dict[str, Any] | None = None
        if os.path.exists(path):
            try:
                base_payload = LayoutJsonWriter.read_header(path)
            except Exception as exc:
                LOGGER.event(
                    "layout.export.json.error",
                    section=OUTSIDE_ARTICLE,
                    data={"path": path, "error": str(exc), "stage": "read"},
                )
                base_payload = None
        # Код, подпись и позиция берутся у одной и той же точки раскладки: порядок точек Layout
        # (сгруппированный по hue) может не совпадать с порядком записей базового файла. Точки пишутся
        # потоком из таблицы кодов и массива ячеек — тот же текст, что давал json.dump словаря точек.
        cells = self._positions_array()
        payload = dict(base_payload) if isinstance(base_payload, dict) else {}
        payload.update(
            {
                "width": self.width,
                "height": self.height,
                "points": self._point_count,
                "similarity": self._similarity,
                "lambda_threshold": self._lambda,
                "eta": self._eta,
                "pair_radius": self._pair_radius,
                LayoutJsonWriter.POINTS_KEY: None,
            }
        )
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        try:
            LayoutJsonWriter.write(path, payload, self._table, y=cells[:, 0], x=cells[:, 1])
        except Exception as exc:
            LOGGER.event(
                "layout.export.json.error",
                section=OUTSIDE_ARTICLE,
                data={"path": path, "error": str(exc)},
            )
            raise
        LOGGER.event(
            "layout.export.json",
            section=OUTSIDE_ARTICLE,
            data={
                "path": path,
                "points": self._point_count,
                "width": self.width,
                "height": self.height,
            },
        )

    @property
    def colorizer(self) -> EncoderCodeColorizer:
        return self._colorizer

    def colors_rgb(self) -> list[tuple[int, int, int]]:
        """Псевдоцвет каждой точки как сумма цветов её единичных битов (§5.14)."""
        if self._colors_rgb_cache is None:
            self._colors_rgb_cache = tuple(
                (int(r), int(g), int(b)) for r, g, b in self._colors_array().tolist()
            )
        return list(self._colors_rgb_cache)

    def _colors_array(self) -> "np.ndarray":
        """Псевдоцвета кодов всех точек массивом uint8 N × 3 (§5.14): от расстановки не зависят."""
        if self._colors_rgb_array is None:
            colors = np.zeros((self._point_count, 3), dtype=np.uint8)
            # Биты кодов распаковываются блоками строк (матрица N × L целиком не создаётся); цвет кода от других
            # кодов не зависит, поэтому от размера блока числа не зависят.
            for start in range(0, self._point_count, _COLOR_BLOCK_ROWS):
                stop = min(self._point_count, start + _COLOR_BLOCK_ROWS)
                colors[start:stop] = self._colorizer.colors(self._table.bits(start, stop))
            self._colors_rgb_array = colors
        return self._colors_rgb_array

    def _component_colors_array(self, period: float | None) -> "np.ndarray":
        """Цвета известной компоненты стимула массивом uint8 N × 3 (рис. 13–16): по цвету на значение hue."""
        cached = self._component_colors_arrays.get(period)
        if cached is None:
            import colorsys

            hues = self._table.hues
            values, inverse = np.unique(hues, return_inverse=True)
            if period is None:
                # Те же действия, что в component_colors_rgb: (hue − min) / span · (1 − 1e-9).
                span = float(values.max() - values.min()) if values.size else 0.0
                normalized = [
                    0.0 if span <= 0.0 else (float(value) - float(values.min())) / span * (1.0 - 1e-9)
                    for value in values.tolist()
                ]
            else:
                if period <= 0.0:
                    raise ValueError("period must be positive")
                normalized = [(float(value) % period) / period for value in values.tolist()]
            palette = np.zeros((len(normalized), 3), dtype=np.uint8)
            for slot, value in enumerate(normalized):
                r, g, b = colorsys.hsv_to_rgb(value, 1.0, 1.0)
                palette[slot] = (int(r * 255), int(g * 255), int(b * 255))
            cached = palette[inverse.reshape(-1)] if hues.size else np.zeros((0, 3), dtype=np.uint8)
            self._component_colors_arrays[period] = cached
        return cached

    def component_colors_rgb(self, *, period: float | None = None) -> list[tuple[int, int, int]]:
        """Диагностическая раскраска по известной компоненте стимула (рис. 13–16).

        Используется только для визуальной проверки раскладки: значение ``hue``
        точки — это исходная величина (например, угол), а не код. В самом
        алгоритме раскладки она не участвует.
        """
        cached = self._component_colors_cache.get(period)
        if cached is not None:
            # Цвет точки зависит только от её hue и периода: считается один раз.
            return list(cached)
        if self._point_count == 0:
            return []
        colors = [(int(r), int(g), int(b)) for r, g, b in self._component_colors_array(period).tolist()]
        self._component_colors_cache[period] = list(colors)
        return colors

    def render_image(self, *, log: bool = True) -> "numpy.ndarray":
        """Карта раскладки в псевдоцветах кодов (§5.14)."""
        colors = self._colors_array()
        image = self._build_image(colors)
        if log:
            LOGGER.event(
                "layout.render_image",
                section=LAID_OUT_STRUCTURE,
                data={"height": self.height, "width": self.width, "coloring": "code_bits"},
                visuals=self._map_visuals("layout/final/image", "layout/final/points", image, colors, static=True),
            )
        return image

    def render_energy_image(
        self, energies: Sequence[float], *, log: bool = True
    ) -> "numpy.ndarray":
        """Композиция V и Ê: цвет точки по коду, яркость по энергии (рис. 18, §5.14)."""
        if len(energies) != self._point_count:
            raise ValueError("energies must match the number of layout points")
        # Канал · яркость с округлением до ближайшего чётного — как int(round(channel * brightness)).
        brightness = np.clip(np.asarray(energies, dtype=np.float64), 0.0, 1.0)
        shaded = np.rint(self._colors_array().astype(np.float64) * brightness[:, None]).astype(np.uint8)
        image = self._build_image(shaded)
        if log:
            LOGGER.event(
                "layout.render_energy_image",
                section=LAID_OUT_STRUCTURE,
                data={
                    "height": self.height,
                    "width": self.width,
                    "coloring": "code_bits_times_energy",
                    "energy_mean": sum(float(e) for e in energies) / max(1, len(energies)),
                },
                visuals=self._map_visuals(
                    "layout/final/energy_image", "layout/final/energy_points", image, shaded, static=True
                ),
            )
        return image

    def render_component_image(
        self, *, period: float | None = None, log: bool = True
    ) -> "numpy.ndarray":
        """Карта раскладки в цветах известной компоненты стимула (ср. рис. 16, §5.13.2)."""
        colors = self._component_colors_array(period)
        image = self._build_image(colors)
        if log:
            LOGGER.event(
                "layout.render_component_image",
                section=MAPPING_CONTINUITY,
                data={
                    "height": self.height,
                    "width": self.width,
                    "coloring": "stimulus_component",
                    "period": period,
                },
                visuals=self._map_visuals(
                    "layout/final/component_image", "layout/final/component_points", image, colors, static=True
                ),
            )
        return image

    def _point_centres(self) -> "np.ndarray":
        """Центры ячеек точек (x + 0.5, y + 0.5) массивом N × 2 — координаты точек на картах раскладки."""
        cells = self._positions_array()
        return np.stack([cells[:, 1] + 0.5, cells[:, 0] + 0.5], axis=1)

    def _map_visuals(
        self, image_path: str, points_path: str, image: "np.ndarray", colors: "np.ndarray", *, static: bool = False
    ) -> list:
        """Карта раскладки для Rerun (§5.14): картинка «ячейка сетки — пиксель» и точки кружками поверх неё.

        У большой раскладки (N > _VISUAL_POINT_LIMIT) кружков нет: все её точки уже на картинке, а миллионы
        кружков окно не тянет. Пустой набор точек убирает с карты кружки меньшей раскладки (затравки большого
        набора), записанные раньше по тому же пути, — иначе они остались бы поверх новой картинки.
        """
        if self._point_count > _VISUAL_POINT_LIMIT:
            positions = np.zeros((0, 2), dtype=np.float32)
            point_colors = np.zeros((0, 3), dtype=np.uint8)
        else:
            positions = self._point_centres()
            point_colors = colors
        return [
            LOGGER.visual_image(image_path, image, static=static),
            LOGGER.visual_points2d(points_path, positions, colors=point_colors, radii=0.5, static=static),
        ]

    def _build_image(
        self,
        colors: Sequence[tuple[int, int, int]] | None = None,
    ) -> "numpy.ndarray":
        try:
            import numpy as np
        except Exception as exc:
            LOGGER.event(
                "layout.image.numpy_unavailable",
                section=LAID_OUT_STRUCTURE,
                data={"error": str(exc)},
            )
            return []

        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        if self._point_count == 0:
            return image
        cells = self._positions_array()
        point_colors = self._colors_array() if colors is None else np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        inside = (cells[:, 0] >= 0) & (cells[:, 0] < self.height) & (cells[:, 1] >= 0) & (cells[:, 1] < self.width)
        if bool(inside.all()):
            image[cells[:, 0], cells[:, 1]] = point_colors
        else:
            image[cells[inside, 0], cells[inside, 1]] = point_colors[inside]
        return image

    def average_local_energy(self, radius: int) -> float:
        """Качество раскладки Ē = (1/|V+|) Σ Ê(v, r_e) по всем точкам, E_max = max по V (§5.7, §5.8)."""
        if radius <= 0:
            raise ValueError("radius must be positive")
        if self._point_count == 0:
            return 0.0
        energies, mode = self._point_energies_with_mode(radius)
        emax = max(energies)
        # Ê = E / E_max поэлементно (то же деление float64) и та же сумма Python, но без обхода точек по одной.
        avg_energy = (
            0.0
            if emax <= 0.0
            else sum((np.asarray(energies, dtype=np.float64) / emax).tolist()) / len(energies)
        )
        LOGGER.event(
            "layout.energy.average",
            section=QUALITY_ASSESS,
            data={
                "radius": radius,
                "energy": avg_energy,
                "energy_max": emax,
                "points": len(energies),
                "mode": mode,
                "note": "Ē по всем точкам V+, нормировка на E_max по всему V (§5.7)",
            },
        )
        return avg_energy

    def _place_points(self) -> None:
        """Начальная расстановка: точки в случайных ячейках сетки, по одной в ячейке (§5.1.2).

        Перемешиваются номера ячеек y·width + x — та же перестановка и тот же расход генератора, что при
        перемешивании списка ячеек (y, x): перестановка зависит только от длины списка. Расстановка пишется
        в массивы; списки (grid, _positions, …) заполняются из них при первом обращении.
        """
        cells = list(range(self.height * self.width))
        self._rng.shuffle(cells)
        chosen = np.asarray(cells[: self._point_count], dtype=np.int64)
        pos_y = chosen // self.width
        pos_x = chosen % self.width
        self._pos_np = np.stack([pos_y, pos_x], axis=1)
        grid = np.full((self.height, self.width), -1, dtype=np.int64)
        grid[pos_y, pos_x] = np.arange(self._point_count, dtype=np.int64)
        self._grid_np = grid
        self._arrays_stale = False
        self._lists_stale = True

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
        # Случайная плитка разбиения может состоять из единственной клетки.
        # В ней нет второй точки, поэтому локальный цикл никогда не завершится.
        if (
            y_min_bound == y_max_bound
            and x_min_bound == x_max_bound
            and y_min_bound == y1
            and x_min_bound == x1
        ):
            y_min_bound = 0
            y_max_bound = self.height - 1
            x_min_bound = 0
            x_max_bound = self.width - 1
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

    def _compute_point_energies(self, radius: int) -> list[float]:
        """Энергии E(c, r) всех точек V (§5.7): шейдер §5.12, если доступен, иначе CPU.

        Результат кешируется до следующего обмена или смены λ, η; возвращаемый список не изменять.
        """
        cached = self._point_energy_cache.get(int(radius))
        if cached is not None and cached[0] == self._positions_version:
            return cached[1]
        energies = self._point_energies_with_mode(radius)[0]
        self._point_energy_cache[int(radius)] = (self._positions_version, energies)
        return energies

    def _point_energies_with_mode(self, radius: int) -> tuple[list[float], str]:
        if radius <= 0:
            raise ValueError("radius must be positive")
        if self._point_count == 0:
            return [], "empty"
        if self._torch_engine is not None and self._device_short_enabled:
            # Большая раскладка режима статьи: энергии точек считает ядро окна на устройстве.
            try:
                device_energies = self._device_point_energies(radius)
            except Exception as exc:  # noqa: BLE001
                LOGGER.event(
                    "layout.point_energy.gpu_failed",
                    section=GPU_IMPLEMENTATION,
                    data={"error": str(exc), "engine": "window_kernel"},
                    force=True,
                )
                self._device_short_enabled = False
                self._device_short = None
                device_energies = None
            if device_energies is not None:
                return device_energies.tolist(), "gpu"
        if self._torch_engine is not None:
            try:
                engine = self._torch_engine
                host_state: dict[str, "np.ndarray"] = {}

                def window_energies() -> tuple["np.ndarray", "np.ndarray"]:
                    energies = engine.point_energies(
                        radius, lambda_threshold=self._lambda, eta=self._eta, distance_eps=self._distance_eps,
                        **host_state,
                    )
                    return energies, np.zeros(len(energies), dtype=np.float32)

                # Строки сходства ради одного расчёта энергий точек не строятся: берутся, если уже есть.
                sparse = self._sparse_long_engine(build=False)
                # Расстановка на устройстве нужна расчёту по строкам сходства; слитому ядру — массивы хоста.
                host_state = self._host_layout_state(device_needed=sparse is not None)
                if sparse is None:
                    torch_energies = window_energies()[0]
                else:
                    self._sync_arrays()
                    indices = np.arange(self._point_count, dtype=np.int64)
                    absent = np.full(self._point_count, -1, dtype=np.int64)
                    cells = self._pos_np
                    sign_kernel = self._pair_sign_kernel(sparse)

                    def row_energies(signs_only: bool) -> tuple["np.ndarray", "np.ndarray"] | None:
                        """E(c, r) по строкам сходства: слагаемые раскладывает ядро, суммирует редукция окна."""
                        if sign_kernel is not None:
                            energies = sign_kernel.point_energies(cells, float(radius), distance_eps=self._distance_eps)
                            if energies is not None:
                                return energies, np.zeros(len(energies), dtype=np.float32)
                        return sparse.neighbourhood_energies(
                            indices, absent, cells, cells,
                            local_radius=float(radius), kernel="inverse", euclidean=True,
                            distance_eps=self._distance_eps, kernel_width=1.0,
                            signs_only=False, need_swapped=False,
                        )

                    torch_energies = self._sparse_checked(
                        "points",
                        row_energies,
                        window_energies,
                        step_index=None,
                        swap_when_greater=True,
                        values_needed=True,
                    )[0]
            except Exception as exc:  # noqa: BLE001
                LOGGER.event(
                    "layout.point_energy.gpu_failed",
                    section=GPU_IMPLEMENTATION,
                    data={"error": str(exc), "engine": "torch"},
                )
                self._torch_engine = None
                torch_energies = None
            if torch_energies is not None and len(torch_energies) == self._point_count:
                # float32 → float числа Python: те же значения, что float(value) по одному.
                return np.asarray(torch_energies, dtype=np.float64).tolist(), "gpu"
        if (
            self._gpu_engine is not None
            and self._ensure_gpu_similarity()
            and self._ensure_gpu_positions()
        ):
            try:
                gpu_energies = self._gpu_engine.point_energies(
                    radius,
                    lambda_threshold=self._lambda,
                    eta=self._eta,
                    distance_eps=self._distance_eps,
                )
            except Exception as exc:
                LOGGER.event(
                    "layout.point_energy.gpu_failed",
                    section=GPU_IMPLEMENTATION,
                    data={"error": str(exc)},
                )
                self._gpu_engine = None
                gpu_energies = None
            if gpu_energies is not None and len(gpu_energies) == self._point_count:
                return [float(value) for value in gpu_energies], "gpu"
        return self._point_energies_cpu(radius), "cpu"

    def _point_energies_cpu(self, radius: int) -> list[float]:
        radius_sq = radius * radius
        energies = [0.0 for _ in range(self._point_count)]
        for idx in range(self._point_count):
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
                    # §5.7: энергия точки использует евклидово расстояние d_c.
                    energy += sim / (math.sqrt(dist_sq) + self._distance_eps)
            energies[idx] = energy
        return energies

    def point_energies(self, radius: int) -> list[float]:
        """Нормированная энергия каждой точки Ê(c, r_e) = E / E_max (§5.7)."""
        if radius <= 0:
            raise ValueError("radius must be positive")
        energies = self._compute_point_energies(radius)
        emax = max(energies, default=0.0)
        normalized = [energy / emax if emax > 0.0 else 0.0 for energy in energies]
        LOGGER.event(
            "layout.point_energy.map",
            section=POINT_ENERGY,
            data={
                "radius": radius,
                "points": self._point_count,
                "energy_max": emax,
                "energy_mean_normalized": (
                    sum(normalized) / len(normalized) if normalized else 0.0
                ),
            },
        )
        return normalized

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
        # §5.9.3: периодически считаются энергии всех точек V, P(p) ∝ −ln Ê(p, r).
        energies = self._compute_point_energies(config.energy_radius)
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
                "weighted": config.use_weighted_first_point,
                "note": "энергии всех точек, пересчёт раз в recalc_every шагов (§5.9.3)",
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
        # §5.5.2: r >= |p1 - p2|; +1 гарантирует точки помимо самой пары.
        base_radius = int(math.ceil(math.sqrt(dist_sq))) + 1
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
        kernel: str = "inverse",
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
                recovered = tensor_engine.release_memory_on_oom(
                    exc,
                    stage="short.similarity_block",
                )
                LOGGER.event(
                    "layout.gpu.tensor.short.sim_failed",
                    section=GPU_IMPLEMENTATION,
                    data={
                        "error": str(exc),
                        "retry": recovered,
                    },
                )
                if recovered:
                    try:
                        missing_matrix = tensor_engine.similarity_block(
                            missing, subset_indices, self._lambda, self._eta
                        )
                    except Exception as retry_exc:
                        LOGGER.event(
                            "layout.gpu.tensor.short.sim_failed",
                            section=GPU_IMPLEMENTATION,
                            data={
                                "error": str(retry_exc),
                                "retry": False,
                                "stage": "retry",
                            },
                        )
                        missing_matrix = None
                else:
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

        recovered_from_oom = False
        try:
            tensor_batch = tensor_engine.short_pair_energies(
                pair_payload,
                self._pos_y,
                self._pos_x,
                self._short_similarity_cache,
                self._distance_eps,
                euclidean=self._short_distance == "euclidean",
                kernel=kernel,
                kernel_width=self._long_kernel_width,
            )
        except Exception as exc:
            recovered_from_oom = tensor_engine.release_memory_on_oom(
                exc,
                stage="short.pair_energies",
            )
            LOGGER.event(
                "layout.gpu.tensor.short.disabled",
                section=GPU_IMPLEMENTATION,
                data={
                    "error": str(exc),
                    "retry": recovered_from_oom,
                },
            )
            if recovered_from_oom:
                try:
                    tensor_batch = tensor_engine.short_pair_energies(
                        pair_payload,
                        self._pos_y,
                        self._pos_x,
                        self._short_similarity_cache,
                        self._distance_eps,
                        euclidean=self._short_distance == "euclidean",
                        kernel=kernel,
                        kernel_width=self._long_kernel_width,
                    )
                except Exception as retry_exc:
                    LOGGER.event(
                        "layout.gpu.tensor.short.disabled",
                        section=GPU_IMPLEMENTATION,
                        data={
                            "error": str(retry_exc),
                            "retry": False,
                            "stage": "retry",
                        },
                    )
                    tensor_batch = None
                    recovered_from_oom = False
            else:
                self._tensor_engine = None
                tensor_batch = None

        if tensor_batch is None:
            if not recovered_from_oom:
                self._tensor_engine = None
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

    def set_swap_temperature(self, temperature: float) -> None:
        """Температура принятия обменов: 0 — жадное правило §5.6, иначе Metropolis по нормированной Δ."""
        if temperature < 0.0:
            raise ValueError("swap temperature must be >= 0")
        self._swap_temperature = float(temperature)
        LOGGER.event(
            "layout.swap.temperature",
            section=POINT_EXCHANGE,
            data={"temperature": self._swap_temperature,
                  "note": "0 — обмен только при выгодной энергии (§5.6); >0 — отжиг (расширение статьи)"},
        )

    def _apply_adaptive_lambda(self, update: _AdaptiveUpdate, *, step: int) -> None:
        """Порог λ повышается ступенью при смене этапа расписания (§7.1.7)."""
        if update.new_lambda is None or update.new_lambda == self._lambda:
            return
        self._lambda = update.new_lambda
        self._invalidate_threshold_caches(reason="adaptive_lambda")
        LOGGER.event(
            "layout.adaptive.lambda",
            section=LAYOUT_PARAMETERS,
            data={
                "step": step,
                "lambda_threshold": self._lambda,
                "swap_ratio": update.swap_ratio,
            },
        )

    def _select_swaps(
        self,
        coords: Sequence[tuple[int, int, int, int]],
        deltas: Sequence[float],
    ) -> list[tuple[tuple[int, int, int, int], float]]:
        """Отбор обменов по Δ (положительная Δ = выгодно): жадно (§5.6) или с температурой."""
        if self._swap_temperature <= 0.0:
            return [(c, float(d)) for c, d in zip(coords, deltas) if d > 0.0]
        accept = self._accepted_swaps(np.asarray(deltas, dtype=np.float64))
        return [(c, float(d)) for c, d, ok in zip(coords, deltas, accept) if ok]

    def _accepted_swaps(self, deltas: "np.ndarray") -> "np.ndarray":
        """Маска принятых обменов по Δ пар шага (float64, положительная Δ = выгодно).

        Температура 0 — жадное правило §5.6. Иначе выгодные обмены принимаются всегда, невыгодные —
        с вероятностью exp(Δ / (T · rms|Δ|)) (отжиг, расширение статьи): по одному числу генератора
        на невыгодную пару в порядке пар шага.
        """
        if self._swap_temperature <= 0.0:
            return deltas > 0.0
        if deltas.size == 0:
            return np.zeros(0, dtype=bool)
        rms = float(np.sqrt(np.mean(deltas * deltas))) or 1.0
        scale = self._swap_temperature * rms
        accept = deltas > 0.0
        losing = ~accept & (deltas < 0.0)
        losing_count = int(np.count_nonzero(losing))
        if losing_count:
            draws = _UniformDraws.take(self._rng, losing_count)
            accept[losing] = draws < np.exp(deltas[losing] / scale)
        if LOGGER.should_log("layout.swap.annealed"):
            LOGGER.event(
                "layout.swap.annealed",
                section=POINT_EXCHANGE,
                data={"pairs": int(deltas.size), "favourable": int((deltas > 0.0).sum()),
                      "accepted_unfavourable": int((accept & (deltas <= 0.0)).sum()),
                      "temperature": self._swap_temperature, "rms_delta": rms},
                force=True,
            )
        return accept

    def _long_kernel_radius(self, pair_radius: int | None) -> int:
        """Радиус окрестности гауссова ядра дальней фазы: следует за радиусом выбора пар, не ниже минимума."""
        if pair_radius is None:
            return self._long_kernel_radius_min
        return max(self._long_kernel_radius_min, int(pair_radius))

    def _distance_mode(self, mode: str) -> str:
        """Режим расстояния энергии фазы: евклидово d (§5.5) или d² (§5.9.2)."""
        return self._long_distance if mode == "long" else self._short_distance

    def _should_swap(
        self,
        y1: int,
        x1: int,
        y2: int,
        x2: int,
        *,
        mode: str,
        local_radius: int | None,
        kernel: str = "inverse",
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
        # §5.5.2: r >= |p1 - p2|; +1 гарантирует точки помимо самой пары.
        base_radius = int(math.ceil(math.sqrt(dist_sq))) + 1
        radius = base_radius if local_radius is None else max(local_radius, base_radius)
        energy_c, energy_s = self._pair_energy_short(
            idx_a, (y1, x1), idx_b, (y2, x2), radius=radius, kernel=kernel
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
            section=SIMILARITY_THRESHOLD,
            data={
                "lambda_threshold": self._lambda,
                "eta": self._eta,
                "cutoff": cutoff,
            },
        )
        LOGGER.event(
            "layout.energy.long.threshold",
            section=SIMILARITY_THRESHOLD,
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
                    euclidean=self._long_distance == "euclidean",
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
            dist1_sq = dy1 * dy1 + dx1 * dx1
            dist2_sq = dy2 * dy2 + dx2 * dx2
            if self._long_distance == "euclidean":
                # §5.5.1: φ = Σ(s1·d1 + s2·d2) с евклидовыми d; иначе d² (§5.9.2).
                dist1 = np.sqrt(dist1_sq)
                dist2 = np.sqrt(dist2_sq)
            else:
                dist1 = dist1_sq
                dist2 = dist2_sq
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
        kernel: str = "inverse",
    ) -> tuple[float, float]:
        if kernel not in _NEIGHBOURHOOD_KERNELS:
            raise ValueError("kernel must be 'inverse' or 'gauss'")
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
                if kernel == "gauss":
                    sig2 = 2.0 * self._long_kernel_width ** 2 * radius_sq
                    w1 = math.exp(-d1 / sig2)
                    w2 = math.exp(-d2 / sig2)
                else:
                    if self._short_distance == "euclidean":
                        # §5.5.2: φ = Σ(s1/d1 + s2/d2) с евклидовыми d; иначе d² (§5.9.2).
                        d1 = math.sqrt(d1)
                        d2 = math.sqrt(d2)
                    w1 = 1.0 / (d1 + self._distance_eps)
                    w2 = 1.0 / (d2 + self._distance_eps)
                energy_c += s1 * w1 + s2 * w2
                energy_s += s2 * w1 + s1 * w2
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

    @staticmethod
    def _first_come_swaps(cell_1: "np.ndarray", cell_2: "np.ndarray") -> "np.ndarray":
        """Номера обменов, которые применяются при обходе по порядку (§5.10): обмен пропускается, если одну
        из его ячеек уже занял ранее применённый обмен.

        Обмен, обе ячейки которого больше ни у кого не встречаются, применяется всегда; по порядку
        обходятся только обмены с общими ячейками — их единицы на шаг.
        """
        count = int(cell_1.size)
        cells = np.concatenate([cell_1, cell_2])
        order = np.argsort(cells, kind="stable")
        ordered = cells[order]
        same = ordered[1:] == ordered[:-1]
        if not same.any():
            return np.arange(count)
        repeated = np.zeros(2 * count, dtype=bool)
        repeated[1:] |= same
        repeated[:-1] |= same
        contested = np.zeros(count, dtype=bool)
        contested[order[repeated] % count] = True
        keep = ~contested
        used: set[int] = set()
        indices = np.flatnonzero(contested)
        for index, first, second in zip(indices.tolist(), cell_1[indices].tolist(), cell_2[indices].tolist()):
            if first in used or second in used:
                continue
            used.add(first)
            used.add(second)
            keep[index] = True
        return np.flatnonzero(keep)

    def _swap_radius_table(self) -> "np.ndarray":
        """math.hypot(dy, dx) для всех смещений сетки: таблица [|dy|, |dx|] (строится при первом обмене)."""
        table = self._swap_radius_cache
        if table is None:
            table = np.array(
                [[math.hypot(dy, dx) for dx in range(self.width)] for dy in range(self.height)], dtype=np.float64
            )
            self._swap_radius_cache = table
        return table

    def _apply_swaps_arrays(self, swaps: "np.ndarray", *, mode: str) -> int:
        """Обмены шага массивом m × 4 (y1, x1, y2, x2): то же, что _apply_tracked_swaps, без обхода в Python.

        Обмены применяются в порядке пар шага; обмен пропускается, если его ячейку уже занял применённый
        обмен (§5.10). Применённые обмены не делят ячеек, поэтому расстановка меняется одним присваиванием.
        Меняются массивы _pos_np, _grid_np; списки расстановки помечаются устаревшими (_sync_lists).
        """
        if swaps.shape[0] == 0:
            return 0
        self._sync_arrays()
        grid_np = self._grid_np
        pos_np = self._pos_np
        width = self.width
        kept = self._first_come_swaps(swaps[:, 0] * width + swaps[:, 1], swaps[:, 2] * width + swaps[:, 3])
        if kept.size != swaps.shape[0]:
            swaps = swaps[kept]
        y1, x1, y2, x2 = swaps[:, 0], swaps[:, 1], swaps[:, 2], swaps[:, 3]
        idx_a = grid_np[y1, x1]
        idx_b = grid_np[y2, x2]
        grid_np[y1, x1] = idx_b
        grid_np[y2, x2] = idx_a
        has_a = idx_a >= 0
        has_b = idx_b >= 0
        moved = np.concatenate([idx_a[has_a], idx_b[has_b]])
        moved_y = np.concatenate([y2[has_a], y1[has_b]])
        moved_x = np.concatenate([x2[has_a], x1[has_b]])
        pos_np[moved, 0] = moved_y
        pos_np[moved, 1] = moved_x
        applied = int(swaps.shape[0])
        self._lists_stale = True
        if mode == "long":
            # Радиусы обменов дальней фазы: те же значения math.hypot и то же сложение по порядку, что
            # в _apply_swaps (накопление слева направо, без попарного суммирования).
            radii = self._swap_radius_table()[np.abs(y1 - y2), np.abs(x1 - x2)]
            running = np.empty(applied + 1, dtype=np.float64)
            running[0] = self._long_swap_radius_sum
            running[1:] = radii
            self._long_swap_radius_sum = float(np.cumsum(running)[-1])
            self._long_swap_count += applied
        # Обмены шага для расстановки на устройстве: загружаются, только если ею пользуются (_ensure_torch_positions).
        self._torch_pending_moves.append((moved, moved_y, moved_x, y1, x1, y2, x2, idx_a, idx_b))
        self._torch_tracked_changes += 1
        # §5.9.3: веса первой точки пересчитываются периодически (energy_recalc_every), а не после обмена.
        self._gpu_positions_dirty = True
        self._positions_version += 1
        return applied

    def _apply_swaps(self, swaps: Iterable[tuple[int, int, int, int]], *, mode: str) -> int:
        self._sync_lists()
        used: set[tuple[int, int]] = set()
        applied = 0
        grid = self._grid_cells
        positions = self._positions_list
        pos_y = self._pos_y_list
        pos_x = self._pos_x_list
        row_occupied = self._row_occupied_sets
        # Массивы меняются вместе со списками, пока соответствуют им; устаревшие пересоберёт _sync_arrays.
        grid_np = self._grid_np
        pos_np = self._pos_np
        arrays_live = not self._arrays_stale and grid_np is not None and pos_np is not None
        for y1, x1, y2, x2 in swaps:
            if (y1, x1) in used or (y2, x2) in used:
                continue
            idx_a = grid[y1][x1]
            idx_b = grid[y2][x2]
            if idx_a is not None:
                row_occupied[y1].discard(x1)
            if idx_b is not None:
                row_occupied[y2].discard(x2)
            grid[y1][x1], grid[y2][x2] = idx_b, idx_a
            if idx_a is not None:
                positions[idx_a] = (y2, x2)
                pos_y[idx_a] = y2
                pos_x[idx_a] = x2
                row_occupied[y2].add(x2)
            if idx_b is not None:
                positions[idx_b] = (y1, x1)
                pos_y[idx_b] = y1
                pos_x[idx_b] = x1
                row_occupied[y1].add(x1)
            if arrays_live:
                grid_np[y1, x1] = -1 if idx_b is None else idx_b
                grid_np[y2, x2] = -1 if idx_a is None else idx_a
                if idx_a is not None:
                    pos_np[idx_a, 0] = y2
                    pos_np[idx_a, 1] = x2
                if idx_b is not None:
                    pos_np[idx_b, 0] = y1
                    pos_np[idx_b, 1] = x1
            used.add((y1, x1))
            used.add((y2, x2))
            applied += 1
            if idx_a is not None:
                self._torch_dirty_points.add(idx_a)
            if idx_b is not None:
                self._torch_dirty_points.add(idx_b)
            self._torch_dirty_cells[(y1, x1)] = -1 if idx_b is None else idx_b
            self._torch_dirty_cells[(y2, x2)] = -1 if idx_a is None else idx_a
            if mode == "long":
                self._register_long_swap_radius(y1, x1, y2, x2)
        if applied:
            # §5.9.3: веса первой точки пересчитываются периодически (energy_recalc_every), а не после обмена.
            self._gpu_positions_dirty = True
            self._positions_version += 1
            self._torch_tracked_changes += 1
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

    @property
    def _code_class(self) -> "np.ndarray":
        """Класс одинаковых кодов каждой точки (int32, длина N): строится при первом обращении."""
        if self._code_class_cache is None:
            self._code_class_cache = self._code_classes()
        return self._code_class_cache

    def _code_classes(self) -> "np.ndarray":
        """Номер класса одинаковых кодов каждой точки (§2.2.4: сходство — функция кодов); int32, длина N."""
        codes = self._packed_codes if self._packed_codes is not None else self._code_matrix
        if codes is None:
            return np.arange(self._point_count, dtype=np.int32)
        _, inverse = np.unique(codes, axis=0, return_inverse=True)
        classes = np.ascontiguousarray(inverse.reshape(-1).astype(np.int32))
        LOGGER.event(
            "layout.codes.classes",
            section=SIMILARITY_MEASURES,
            data={
                "points": self._point_count,
                "distinct_codes": int(classes.max()) + 1 if classes.size else 0,
                "note": "точки с одинаковыми кодами имеют одинаковые строки sim_λ (§2.2.4, §5.4): у пары таких "
                "точек φ_c = φ_s точно, обмена нет (§5.6)",
            },
        )
        return classes

    @staticmethod
    def _pack_codes(code_matrix: "np.ndarray | None") -> "np.ndarray | None":
        """Коды точек 64-битными словами (N × ⌈L/64⌉) для подсчёта общих бит через popcount (§2.2.4)."""
        if code_matrix is None or not hasattr(np, "bitwise_count"):
            return None
        count, length = code_matrix.shape
        words = -(-length // 64)
        padded = np.zeros((count, words * 64), dtype=np.uint8)
        padded[:, :length] = code_matrix != 0
        return np.ascontiguousarray(np.packbits(padded, axis=1)).view(np.uint64)

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
    def _format_label(value: object) -> str:
        if isinstance(value, (int, float)):
            number = float(value)
            if abs(number - round(number)) < 1e-6:
                return str(int(round(number)))
            return f"{number:.3f}"
        return str(value)

    def _coerce_code_entry(
        self,
        entry: LayoutEncodedCode | BitArray | Iterable[int],
        *,
        default_label: float,
    ) -> tuple[BitArray, int, str]:
        if isinstance(entry, LayoutEncodedCode):
            layout_code, ones = self._coerce_code(entry.code)
            label_text = self._format_label(entry.label)
            return layout_code, ones, label_text
        layout_code, ones = self._coerce_code(entry)
        label_text = self._format_label(default_label)
        return layout_code, ones, label_text

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
