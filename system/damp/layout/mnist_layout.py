"""Этап 3 для команды layout: подбор параметров раскладки под размер набора (§7.1.7), кеш раскладки,
раскладка большого набора от затравки (§5.9.4)."""

from __future__ import annotations
import glob
import json
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from damp import settings
from damp.article_refs import LAYOUT_PARAMETERS, OPTIM_SUBSET, OUTSIDE_ARTICLE
from damp.encoding.bitarray import BitArray
from damp.encoding.damp_encoder import EncoderCodeColorizer
from damp.layout.code_table import LayoutCodeTable
from damp.layout.damp_layout import AdaptiveLayoutConfig, Layout, ShortLayoutOptimConfig
from damp.layout.payload import LayoutEncodedCode
from damp.layout.seeded import SeededPlacement
from damp.logging import LOGGER


def _layout_cache_paths_for_count(count: int) -> list[str]:
    paths: list[str] = []
    for pattern_template in settings.LAYOUT_CACHE_PATTERNS:
        pattern = pattern_template.format(count=count)
        paths.extend(
            path
            for path in glob.glob(pattern)
            if not path.endswith(settings.PIPELINE_CHECKPOINT_SUFFIX)
        )
    return sorted(set(paths), key=os.path.getmtime, reverse=True)


@dataclass(frozen=True)
class TunedLayoutParams:
    long_pairs_per_step: int
    long_steps: int
    long_subset_size: int | None
    long_subset_refresh: int
    short_pairs_per_step: int
    short_steps: int
    short_local_radius: int
    short_energy_radius: int
    short_energy_recalc_every: int
    # Число слоёв выборки ячеек окрестности по стороне окна (§5.9.4); None — окрестность считается целиком.
    neighbourhood_sample_side: int | None = None
    # True — пары, вторая ячейка которых дальше радиуса выбора r (§5.3), в шаг не входят.
    enforce_pair_radius: bool = False
    # Наименьший радиус окрестности ближней фазы, следующий за радиусом выбора пар (§5.5.2); None — постоянный.
    neighbourhood_radius_min: int | None = None
    # Число точек затравки большой раскладки (случайное подмножество, раскладываемое целиком); None — без неё.
    seed_points: int | None = None
    # True — шаг дальней фазы большой раскладки выполняется целиком на устройстве.
    device_step: bool = False
    # True — шаг ближней фазы большой раскладки режима статьи выполняется целиком на устройстве.
    device_short_step: bool = False


@dataclass(frozen=True)
class LayoutLambdaSchedule:
    """Пороги сходства раскладки (§5.4, §7.1.7) для кодов со своим распределением сходств.

    start — порог λ в начале раскладки, он же порог ранней отсечки пар ближней фазы (§5.9.1); maximum —
    предел роста λ ступенями (§7.1.7). Константы LAYOUT_LAMBDA_THRESHOLD и LAYOUT_ADAPTIVE_LAMBDA_MAX подобраны
    для первичных кодов; у кодов следующего уровня сходства лежат в другом диапазоне, и пороги для них
    берутся квантилями попарных сходств — той же долей похожих пар, что у первичных кодов.
    """

    start: float
    maximum: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.start <= self.maximum <= 1.0:
            raise ValueError("lambda schedule must satisfy 0 <= start <= maximum <= 1")

    @property
    def short_cutoff(self) -> float:
        """Порог ранней отсечки пар ближней фазы (§5.9.1): совпадает с начальным λ."""
        return self.start

    @classmethod
    def from_quantiles(
        cls,
        start: float,
        maximum: float,
        *,
        start_quantile: float,
        maximum_quantile: float,
    ) -> "LayoutLambdaSchedule":
        """Пороги по квантилям попарных сходств кодов: start и maximum — значения этих квантилей."""
        schedule = cls(start=float(start), maximum=max(float(start), float(maximum)))
        LOGGER.event(
            "layout.lambda_schedule",
            section=LAYOUT_PARAMETERS,
            data={
                "lambda_threshold": schedule.start,
                "lambda_max": schedule.maximum,
                "start_quantile": start_quantile,
                "maximum_quantile": maximum_quantile,
                "primary_lambda_threshold": settings.LAYOUT_LAMBDA_THRESHOLD,
                "primary_lambda_max": settings.LAYOUT_ADAPTIVE_LAMBDA_MAX,
                "note": "§7.1.7: параметры раскладки зависят от задачи; пороги взяты квантилями попарных сходств "
                "кодов — та же доля похожих пар, что дают первичным кодам LAYOUT_LAMBDA_THRESHOLD и "
                "LAYOUT_ADAPTIVE_LAMBDA_MAX",
            },
        )
        return schedule


class LayoutParameterTuner:
    def __init__(
        self,
        *,
        empty_ratio: float | None = None,
        start_radius_factor: float | None = None,
    ) -> None:
        # None — значение из настроек в момент создания.
        self._empty_ratio = settings.LAYOUT_EMPTY_RATIO if empty_ratio is None else empty_ratio
        self._start_radius_factor = settings.LAYOUT_ADAPTIVE_RADIUS_START_FACTOR if start_radius_factor is None else start_radius_factor

    def tune(self, total_points: int) -> TunedLayoutParams:
        if total_points <= 0:
            raise ValueError("total_points must be positive for tuning")
        grid_size = int(math.ceil(math.sqrt(total_points * (1.0 + self._empty_ratio))))
        long_pairs = max(
            settings.LAYOUT_TUNE_LONG_PAIRS_MIN,
            min(settings.LAYOUT_TUNE_LONG_PAIRS_MAX, int(total_points * settings.LAYOUT_TUNE_LONG_PAIR_FACTOR)),
        )
        long_steps = max(
            settings.LAYOUT_TUNE_LONG_STEPS_MIN,
            min(settings.LAYOUT_TUNE_LONG_STEPS_MAX, int(total_points / settings.LAYOUT_TUNE_LONG_STEPS_DIVISOR)),
        )
        if total_points <= settings.LAYOUT_LONG_FULL_SPACE_MAX_POINTS:
            # §5.5.1: дальняя энергия считается по всему пространству V.
            long_subset_size: int | None = None
        else:
            # §5.9.4: для очень больших пространств допускается подмножество.
            long_subset_size = min(
                settings.LAYOUT_TUNE_LONG_SUBSET_MAX,
                max(settings.LAYOUT_TUNE_LONG_SUBSET_MIN, int(math.sqrt(total_points) * settings.LAYOUT_TUNE_LONG_SUBSET_MULTIPLIER)),
            )
            LOGGER.event(
                "layout.tune.long_subset",
                section=OPTIM_SUBSET,
                data={"points": total_points, "long_subset_size": long_subset_size,
                      "full_space_max_points": settings.LAYOUT_LONG_FULL_SPACE_MAX_POINTS},
            )
        long_subset_refresh = max(
            settings.LAYOUT_TUNE_LONG_SUBSET_REFRESH_MIN,
            min(
                settings.LAYOUT_TUNE_LONG_SUBSET_REFRESH_MAX,
                long_steps // settings.LAYOUT_TUNE_LONG_SUBSET_REFRESH_DIVISOR,
            ),
        )
        short_pairs = max(
            settings.LAYOUT_TUNE_SHORT_PAIRS_MIN,
            min(settings.LAYOUT_TUNE_SHORT_PAIRS_MAX, int(total_points * settings.LAYOUT_TUNE_SHORT_PAIR_FACTOR)),
        )
        if settings.LAYOUT_PAIRS_POINT_FRACTION is not None:
            # §5.10: обменов за шаг много меньше числа точек — пар за шаг не больше доли точек.
            pairs_cap = max(1, int(total_points * settings.LAYOUT_PAIRS_POINT_FRACTION))
            long_pairs = min(long_pairs, pairs_cap)
            short_pairs = min(short_pairs, pairs_cap)
        short_steps = max(
            settings.LAYOUT_TUNE_SHORT_STEPS_MIN,
            min(settings.LAYOUT_TUNE_SHORT_STEPS_MAX, int(grid_size * settings.LAYOUT_TUNE_SHORT_STEPS_FACTOR)),
        )
        # §5.5.2: окрестность R должна содержать точки помимо самой пары,
        # поэтому локальный радиус не опускается ниже LAYOUT_SHORT_LOCAL_RADIUS_MIN.
        short_local_radius = max(
            settings.LAYOUT_SHORT_LOCAL_RADIUS_MIN, int(grid_size * settings.LAYOUT_TUNE_SHORT_RADIUS_FACTOR)
        )
        short_energy_radius = max(
            settings.LAYOUT_SHORT_LOCAL_RADIUS_MIN, int(grid_size * settings.LAYOUT_TUNE_SHORT_ENERGY_RADIUS_FACTOR)
        )
        short_energy_recalc_every = max(settings.LAYOUT_TUNE_SHORT_ENERGY_RECALC_FLOOR, short_steps // 6)
        neighbourhood_sample_side: int | None = None
        large_layout = total_points > settings.LAYOUT_LONG_FULL_SPACE_MAX_POINTS
        article_large = large_layout and not settings.LAYOUT_SMOOTH_PINWHEELS
        if article_large:
            # Большая раскладка в режиме статьи: глобальный порядок — из раскладки затравки (дальняя фаза §5.5.1
            # по всему V на миллионах точек неосуществима), ближний порядок — ближней фазой §5.5.2.
            long_subset_size = None
            short_pairs = max(short_pairs, int(total_points * settings.LAYOUT_TUNE_LARGE_SHORT_PAIR_FACTOR))
            if settings.LAYOUT_PAIRS_POINT_FRACTION is not None:
                short_pairs = min(short_pairs, max(1, int(total_points * settings.LAYOUT_PAIRS_POINT_FRACTION)))
            short_local_radius = min(short_local_radius, settings.LAYOUT_TUNE_LARGE_SHORT_RADIUS_MAX)
            short_energy_radius = min(short_energy_radius, settings.LAYOUT_TUNE_LARGE_SHORT_ENERGY_RADIUS_MAX)
            # Ближняя фаза заканчивается по числу обменов (§5.6); бюджет шагов — только верхняя граница
            # (период пересчёта весов первой точки §5.9.3 остаётся прежним).
            short_steps = settings.LAYOUT_TUNE_SHORT_STEPS_MAX
            LOGGER.event(
                "layout.tune.large.article",
                section=OPTIM_SUBSET,
                data={
                    "points": total_points,
                    "seed_points": -1 if settings.LAYOUT_SEED_POINTS is None else settings.LAYOUT_SEED_POINTS,
                    "seed_level_ratio_max": settings.LAYOUT_SEED_LEVEL_RATIO_MAX,
                    "short_radius_factor": settings.LAYOUT_SEED_SHORT_RADIUS_FACTOR,
                    "short_pairs_per_step": short_pairs,
                    "short_steps": short_steps,
                    "short_local_radius": short_local_radius,
                    "short_energy_radius": short_energy_radius,
                    "neighbourhood_radius_min": settings.LAYOUT_TUNE_LARGE_SHORT_NEIGHBOURHOOD_MIN,
                    "note": "большая раскладка режима статьи: дальний порядок — из точной раскладки затравки "
                    "(подмножество, §5.9.4), уровни затравки, на каждом — ближняя фаза §5.5.2 на устройстве; "
                    "пар за шаг — доля точек (§5.10)",
                },
            )
        elif large_layout:
            # Большая раскладка: пар за шаг — доля точек (§5.10), энергии — по подмножеству B и по выборке
            # ячеек окрестности (§5.9.4), радиусы ближней фазы — не выше значений на границе режима.
            long_pairs = max(long_pairs, int(total_points * settings.LAYOUT_TUNE_LARGE_LONG_PAIR_FACTOR))
            short_pairs = max(short_pairs, int(total_points * settings.LAYOUT_TUNE_LARGE_SHORT_PAIR_FACTOR))
            if settings.LAYOUT_SEED_POINTS is not None:
                # Глобальный порядок — из раскладки затравки: больших радиусов и подмножества B нет.
                long_subset_size = None
            else:
                long_subset_size = settings.LAYOUT_TUNE_LARGE_LONG_SUBSET
                long_subset_refresh = settings.LAYOUT_TUNE_LARGE_LONG_SUBSET_REFRESH
            neighbourhood_sample_side = settings.LAYOUT_NEIGHBOURHOOD_SAMPLE_SIDE
            short_local_radius = min(short_local_radius, settings.LAYOUT_TUNE_LARGE_SHORT_RADIUS_MAX)
            short_energy_radius = min(short_energy_radius, settings.LAYOUT_TUNE_LARGE_SHORT_ENERGY_RADIUS_MAX)
            LOGGER.event(
                "layout.tune.large",
                section=OPTIM_SUBSET,
                data={
                    "points": total_points,
                    "long_pairs_per_step": long_pairs,
                    "short_pairs_per_step": short_pairs,
                    "long_subset_size": -1 if long_subset_size is None else long_subset_size,
                    "long_subset_refresh": long_subset_refresh,
                    "seed_points": -1 if settings.LAYOUT_SEED_POINTS is None else settings.LAYOUT_SEED_POINTS,
                    "neighbourhood_sample_side": neighbourhood_sample_side,
                    "short_local_radius": short_local_radius,
                    "short_energy_radius": short_energy_radius,
                    "note": "большая раскладка: пар за шаг — доля точек (§5.10); энергия пары — по подмножеству B, "
                    "затем по выборке ячеек окрестности (§5.9.4)",
                },
            )
        LOGGER.event(
            "layout.tune",
            section=LAYOUT_PARAMETERS,
            data={
                "points": total_points,
                "grid_size_est": grid_size,
                "empty_ratio": self._empty_ratio,
                "start_radius_factor": self._start_radius_factor,
                "long_pairs_per_step": long_pairs,
                "long_steps": long_steps,
                "long_subset_size": -1 if long_subset_size is None else long_subset_size,
                "long_subset_mode": "full_space" if long_subset_size is None else "subset",
                "long_subset_refresh": long_subset_refresh,
                "short_pairs_per_step": short_pairs,
                "short_steps": short_steps,
                "short_local_radius": short_local_radius,
                "short_energy_radius": short_energy_radius,
                "short_energy_recalc_every": short_energy_recalc_every,
                "neighbourhood_sample_side": -1 if neighbourhood_sample_side is None else neighbourhood_sample_side,
                "lambda_threshold": settings.LAYOUT_LAMBDA_THRESHOLD,
                "eta": "hard" if settings.LAYOUT_ETA is None else settings.LAYOUT_ETA,
            },
        )
        return TunedLayoutParams(
            long_pairs_per_step=long_pairs,
            long_steps=long_steps,
            long_subset_size=long_subset_size,
            long_subset_refresh=long_subset_refresh,
            short_pairs_per_step=short_pairs,
            short_steps=short_steps,
            short_local_radius=short_local_radius,
            short_energy_radius=short_energy_radius,
            short_energy_recalc_every=short_energy_recalc_every,
            neighbourhood_sample_side=neighbourhood_sample_side,
            enforce_pair_radius=large_layout,
            neighbourhood_radius_min=settings.LAYOUT_TUNE_LARGE_SHORT_NEIGHBOURHOOD_MIN if article_large else None,
            seed_points=settings.LAYOUT_SEED_POINTS if large_layout else None,
            device_step=bool(settings.LAYOUT_DEVICE_STEP) and large_layout and not article_large,
            device_short_step=bool(settings.LAYOUT_DEVICE_STEP) and article_large,
        )


def _sparse_layout_options() -> dict[str, object]:
    """Параметры Layout для ускоренного расчёта энергий с тем же результатом — по ненулевым sim_λ при жёстком
    пороге и слитым ядром (§5.5.1, §5.5.2, §5.9.5, §5.12): общие для MNIST и pinwheel."""
    return {
        "sparse_long": settings.LAYOUT_SPARSE_LONG,
        "sparse_max_edges": settings.LAYOUT_SPARSE_MAX_EDGES,
        "sparse_memory_fraction": settings.LAYOUT_SPARSE_MEMORY_FRACTION,
        "sparse_buffer_bytes": settings.LAYOUT_SPARSE_BUFFER_BYTES,
        "sparse_entry_budget": settings.LAYOUT_SPARSE_ENTRY_BUDGET,
        "sparse_screen_min_chunks": settings.LAYOUT_SPARSE_SCREEN_MIN_CHUNKS,
        "sparse_verify_every": settings.LAYOUT_SPARSE_VERIFY_EVERY,
        "sparse_verify_chunks": settings.LAYOUT_SPARSE_VERIFY_CHUNKS,
        "pair_kernel": settings.LAYOUT_PAIR_KERNEL,
        "pair_kernel_segment": settings.LAYOUT_PAIR_KERNEL_SEGMENT,
        "compiled_sampler": settings.LAYOUT_COMPILED_SAMPLER,
        "sampler_verify_every": settings.LAYOUT_SAMPLER_VERIFY_EVERY,
        "potential_bounds": settings.LAYOUT_POTENTIAL_BOUNDS,
        "potential_bounds_grid": settings.LAYOUT_POTENTIAL_BOUNDS_GRID,
        "reduction_replica": settings.LAYOUT_REDUCTION_REPLICA,
        "replica_max_bytes": settings.LAYOUT_REPLICA_MAX_BYTES,
        "fused_energy": settings.LAYOUT_FUSED_ENERGY,
        "fused_table_max_bytes": settings.LAYOUT_FUSED_TABLE_MAX_BYTES,
        "fused_table_memory_fraction": settings.LAYOUT_FUSED_TABLE_MEMORY_FRACTION,
        "fused_block_elements": settings.LAYOUT_FUSED_BLOCK_ELEMENTS,
        "fused_verify_every": settings.LAYOUT_FUSED_VERIFY_EVERY,
        "fused_similarity_table": settings.LAYOUT_FUSED_SIMILARITY_TABLE,
    }


def _default_layout_params() -> TunedLayoutParams:
    return TunedLayoutParams(
        long_pairs_per_step=settings.LAYOUT_LONG_PAIRS_PER_STEP,
        long_steps=settings.LAYOUT_LONG_STEPS,
        long_subset_size=settings.LAYOUT_LONG_SUBSET_SIZE,
        long_subset_refresh=settings.LAYOUT_LONG_SUBSET_REFRESH,
        short_pairs_per_step=settings.LAYOUT_SHORT_PAIRS_PER_STEP,
        short_steps=settings.LAYOUT_SHORT_STEPS,
        short_local_radius=settings.LAYOUT_SHORT_LOCAL_RADIUS,
        short_energy_radius=settings.LAYOUT_SHORT_ENERGY_RADIUS,
        short_energy_recalc_every=settings.LAYOUT_SHORT_ENERGY_RECALC_EVERY,
    )


def _layout_json_path(count: int, pair_radius: int | None) -> str:
    suffix = "auto" if pair_radius is None else str(pair_radius)
    return f"data/layout-{count}-{suffix}-{settings.LAYOUT_CACHE_TAG}.json"


def _expected_pair_radius(total_points: int, run_params: TunedLayoutParams) -> int:
    if total_points <= 0:
        raise ValueError("total_points must be positive for pair radius estimation")
    grid_size_est = int(math.ceil(math.sqrt(total_points * (1.0 + settings.LAYOUT_EMPTY_RATIO))))
    expected_short_radius_max = int(grid_size_est * settings.LAYOUT_ADAPTIVE_RADIUS_START_FACTOR)
    expected_short_radius_base = int(
        max(1, round(run_params.short_local_radius * settings.LAYOUT_SHORT_ADAPTIVE_RADIUS_FACTOR))
    )
    return max(
        settings.LAYOUT_ADAPTIVE_RADIUS_MIN,
        min(expected_short_radius_base, expected_short_radius_max),
    )


class LayoutCacheFile:
    """JSON раскладки на диске: проверка совместимости и восстановление Layout (вне статьи)."""

    @staticmethod
    def inspect(path: str) -> tuple[int, int | None, bool] | None:
        try:
            size = os.path.getsize(path)
            if size > settings.LAYOUT_CACHE_MAX_FILE_BYTES:
                LOGGER.event(
                    "layout.cache.too_large",
                    section=OUTSIDE_ARTICLE,
                    data={
                        "path": path,
                        "size_mb": size / 1e6,
                        "max_mb": settings.LAYOUT_CACHE_MAX_FILE_BYTES / 1e6,
                        "note": "файл раскладки слишком велик для чтения кеша (объект на точку): "
                        "раскладка считается заново",
                    },
                )
                return None
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception as exc:
            LOGGER.event(
                "layout.cache.read_error",
                section=OUTSIDE_ARTICLE,
                data={"path": path, "error": str(exc)},
            )
            return None
        points_payload = data.get("layout", [])
        total_points = int(data.get("points") or len(points_payload))
        pair_radius = data.get("pair_radius")
        has_positions = bool(points_payload) and all(
            item.get("x") is not None and item.get("y") is not None for item in points_payload
        )
        ready = has_positions and data.get("width") not in (None, 0) and data.get("height") not in (None, 0)
        LOGGER.event(
            "layout.cache.inspect",
            section=OUTSIDE_ARTICLE,
            data={
                "path": path,
                "points": total_points,
                "pair_radius": pair_radius,
                "ready": ready,
            },
        )
        return total_points, pair_radius, ready

    @staticmethod
    def load(
        path: str,
        run_params: TunedLayoutParams,
        *,
        expected_code_length: int | None = None,
        colorizer: EncoderCodeColorizer | None = None,
        expected_encoder_signature: str | None = None,
        expected_stimulus: Mapping[str, object] | None = None,
    ) -> Layout | None:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        points_payload = data.get("layout", [])
        stored_signature = data.get(settings.ENCODER_SIGNATURE_KEY)
        if (
            expected_encoder_signature is not None
            and stored_signature is not None
            and str(stored_signature) != expected_encoder_signature
        ):
            LOGGER.event(
                "layout.cache.incompatible",
                section=OUTSIDE_ARTICLE,
                data={"path": path, "reason": "encoder_signature"},
            )
            return None
        for key, expected_value in (expected_stimulus or {}).items():
            stored_value = data.get(key, settings.LAYOUT_CACHE_LEGACY_STIMULUS.get(key))
            if stored_value != expected_value:
                LOGGER.event(
                    "layout.cache.incompatible",
                    section=OUTSIDE_ARTICLE,
                    data={"path": path, "reason": key, "stored": str(stored_value), "expected": str(expected_value)},
                )
                return None
        if expected_code_length is not None and points_payload:
            stored_length = len(str(points_payload[0].get("value", "")))
            if stored_length != expected_code_length:
                LOGGER.event(
                    "layout.cache.incompatible",
                    section=OUTSIDE_ARTICLE,
                    data={
                        "path": path,
                        "stored_code_length": stored_length,
                        "expected_code_length": expected_code_length,
                    },
                )
                return None
        width = data.get("width")
        height = data.get("height")
        if width in (None, 0) or height in (None, 0):
            LOGGER.event(
                "layout.cache.incomplete",
                section=OUTSIDE_ARTICLE,
                data={"path": path, "reason": "missing_size"},
            )
            return None
        pair_radius_saved = data.get("pair_radius")
        total_points = int(data.get("points") or len(points_payload))
        if total_points <= 0:
            LOGGER.event(
                "layout.cache.incomplete",
                section=OUTSIDE_ARTICLE,
                data={"path": path, "reason": "empty"},
            )
            return None
        expected_pair_radius = _expected_pair_radius(total_points, run_params)
        planned_pair_radius = data.get(settings.PLANNED_PAIR_RADIUS_KEY)
        if planned_pair_radius is not None:
            # Раскладка считалась с этим плановым радиусом; итоговый радиус может быть меньше (большая
            # раскладка режима статьи заканчивается на своём радиусе), на пригодность кеша он не влияет.
            if int(planned_pair_radius) != expected_pair_radius:
                return None
        elif pair_radius_saved is not None and pair_radius_saved != expected_pair_radius:
            return None
        if not points_payload:
            return None
        if any(item.get("x") is None or item.get("y") is None for item in points_payload):
            LOGGER.event(
                "layout.cache.incomplete",
                section=OUTSIDE_ARTICLE,
                data={"path": path, "reason": "missing_positions"},
            )
            return None
        codes: "OrderedDict[float, list[LayoutEncodedCode]]" = OrderedDict()
        sorted_points = sorted(points_payload, key=lambda item: int(item["index"]))
        for item in sorted_points:
            try:
                hue_value = float(item["hue"])
            except Exception:
                continue
            bit_string = str(item["value"])
            bit_array = BitArray(len(bit_string))
            for idx, ch in enumerate(bit_string):
                if ch == "1":
                    bit_array.set(idx, 1)
            bucket = codes.setdefault(hue_value, [])
            bucket.append(
                LayoutEncodedCode(
                    code=bit_array,
                    label=str(item.get("label", "")),
                )
            )
        layout = Layout(
            codes,
            grid_size=int(width),
            empty_ratio=settings.LAYOUT_EMPTY_RATIO,
            similarity=data.get("similarity", settings.LAYOUT_SIMILARITY),
            lambda_threshold=data.get("lambda_threshold", settings.LAYOUT_LAMBDA_THRESHOLD),
            eta=data.get("eta", settings.LAYOUT_ETA),
            seed=settings.LAYOUT_RANDOM_SEED,
            use_gpu=settings.LAYOUT_USE_GPU,
            long_subset_size=None,
            long_subset_refresh=settings.LAYOUT_LONG_SUBSET_REFRESH,
            colorizer=colorizer,
            long_distance=settings.LAYOUT_LONG_DISTANCE,
            short_distance=settings.LAYOUT_SHORT_DISTANCE,
            long_kernel=settings.LAYOUT_LONG_KERNEL,
            long_kernel_radius_min=settings.LAYOUT_LONG_KERNEL_RADIUS_MIN,
            long_kernel_width=settings.LAYOUT_LONG_KERNEL_WIDTH,
            component_period=settings.LAYOUT_COMPONENT_PERIOD,
            **_sparse_layout_options(),
        )
        if len(layout.positions()) != len(sorted_points):
            raise RuntimeError("layout load mismatch: points count differs")
        positions: list[tuple[int, int]] = [(-1, -1)] * len(sorted_points)
        for item in sorted_points:
            positions[int(item["index"])] = (int(item["y"]), int(item["x"]))
        layout.restore_positions(positions)
        layout.pair_radius = pair_radius_saved
        LOGGER.event(
            "layout.load.cache",
            section=OUTSIDE_ARTICLE,
            data={
                "path": path,
                "points": len(positions),
                "pair_radius": pair_radius_saved,
                "width": width,
                "height": height,
            },
        )
        return layout


def _run_layout(
    codes: "dict[float, list] | LayoutCodeTable",
    params: TunedLayoutParams | None,
    *,
    count: int,
    total_codes: int,
    colorizer: EncoderCodeColorizer | None = None,
    expected_code_length: int | None = None,
    use_cache: bool = True,
) -> Layout:
    run_params = params or _default_layout_params()
    expected_pair_radius = _expected_pair_radius(total_codes, run_params)
    layout_path = _layout_json_path(count, expected_pair_radius)
    if use_cache:
        cached_layout = LayoutCacheFile.load(
            layout_path,
            run_params,
            expected_code_length=expected_code_length,
            colorizer=colorizer,
        )
        if cached_layout is not None:
            return cached_layout
    return _layout_from_codes(codes, run_params, colorizer, layout_path=layout_path)


def _seed_level_points(points: int, seed_points: int) -> int:
    """Число точек затравки очередного уровня большой раскладки из points точек.

    Режим сглаживания — один уровень: seed_points. Режим статьи — уровней столько, чтобы на точку затравки
    приходилось не больше LAYOUT_SEED_LEVEL_RATIO_MAX точек следующего уровня; отношение N / seed_points
    делится между уровнями поровну. Нижний уровень — seed_points точек, он раскладывается целиком.
    """
    if settings.LAYOUT_SMOOTH_PINWHEELS or points <= seed_points:
        return seed_points
    ratio = points / float(seed_points)
    levels = max(1, math.ceil(math.log(ratio) / math.log(settings.LAYOUT_SEED_LEVEL_RATIO_MAX) - 1e-9))
    if levels == 1:
        return seed_points
    return max(seed_points, int(math.ceil(points / ratio ** (1.0 / levels))))


def _layout_from_codes(
    codes: "dict[float, list] | LayoutCodeTable",
    run_params: TunedLayoutParams,
    colorizer: EncoderCodeColorizer | None,
    *,
    layout_path: str | None = None,
    lambda_schedule: LayoutLambdaSchedule | None = None,
) -> Layout:
    """Раскладка кодов: дальняя и ближняя фазы (§5.5) либо большая раскладка от затравки и её доводка.

    lambda_schedule — пороги сходства для кодов с другим распределением сходств (коды следующего уровня);
    None — константы LAYOUT_*. Большая раскладка от затравки со своими порогами идёт только в режиме
    сглаживания: затравка раскладывается с теми же порогами, доводка — дальней фазой с радиуса доводки.
    """
    seeded = (
        run_params.seed_points is not None
        and isinstance(codes, LayoutCodeTable)
        and codes.count > run_params.seed_points
    )
    if lambda_schedule is not None:
        if seeded and not settings.LAYOUT_SMOOTH_PINWHEELS:
            raise ValueError("свои пороги сходства в большой раскладке от затравки поддержаны только в режиме сглаживания")
        if seeded:
            layout, refine_radius, refine_steps = _seeded_layout(
                codes, run_params, colorizer, lambda_schedule=lambda_schedule
            )
        else:
            layout = _new_layout(codes, run_params, colorizer, lambda_threshold=lambda_schedule.start)
            refine_radius = refine_steps = None
        _run_long_phase(
            layout,
            run_params,
            step_offset=1,
            layout_path=layout_path,
            start_radius=refine_radius,
            steps=refine_steps,
            lambda_max=lambda_schedule.maximum,
        )
        _run_short_phase(
            layout,
            run_params,
            step_offset=1 + layout.last_steps,
            similarity_cutoff=lambda_schedule.short_cutoff,
            lambda_max=lambda_schedule.maximum,
        )
        return layout
    if seeded:
        layout, refine_radius, refine_steps = _seeded_layout(codes, run_params, colorizer)
    else:
        layout = _new_layout(codes, run_params, colorizer)
        refine_radius = refine_steps = None
    if refine_steps == 0:
        # Режим статьи: дальний порядок большой раскладки взят у затравки, дальней фазы нет; ближняя фаза
        # начинается с радиуса в несколько блоков затравки и идёт без ранней отсечки пар (§5.9.1).
        _run_short_phase(
            layout,
            run_params,
            step_offset=1,
            start_radius=refine_radius,
            similarity_cutoff=settings.LAYOUT_SEED_SHORT_SIMILARITY_CUTOFF,
        )
        return layout
    _run_long_phase(
        layout, run_params, step_offset=1, layout_path=layout_path, start_radius=refine_radius, steps=refine_steps
    )
    _run_short_phase(layout, run_params, step_offset=1 + layout.last_steps)
    return layout


def _new_layout(
    codes: "dict[float, list] | LayoutCodeTable",
    run_params: TunedLayoutParams,
    colorizer: EncoderCodeColorizer | None,
    *,
    initial_cells: "np.ndarray | None" = None,
    lambda_threshold: float | None = None,
) -> Layout:
    """Раскладка с параметрами конвейера; initial_cells — начальная расстановка вместо случайной (§5.2),
    lambda_threshold — порог сходства λ (§5.4) вместо LAYOUT_LAMBDA_THRESHOLD."""
    return Layout(
        codes,
        empty_ratio=settings.LAYOUT_EMPTY_RATIO,
        similarity=settings.LAYOUT_SIMILARITY,
        lambda_threshold=settings.LAYOUT_LAMBDA_THRESHOLD if lambda_threshold is None else float(lambda_threshold),
        eta=settings.LAYOUT_ETA,
        seed=settings.LAYOUT_RANDOM_SEED,
        use_gpu=settings.LAYOUT_USE_GPU,
        long_subset_size=run_params.long_subset_size,
        long_subset_refresh=run_params.long_subset_refresh,
        colorizer=colorizer,
        long_distance=settings.LAYOUT_LONG_DISTANCE,
        short_distance=settings.LAYOUT_SHORT_DISTANCE,
        long_kernel=settings.LAYOUT_LONG_KERNEL,
        long_kernel_radius_min=settings.LAYOUT_LONG_KERNEL_RADIUS_MIN,
        long_kernel_width=settings.LAYOUT_LONG_KERNEL_WIDTH,
        component_period=settings.LAYOUT_COMPONENT_PERIOD,
        long_subset_min_neighbours=settings.LAYOUT_LONG_SUBSET_MIN_NEIGHBOURS,
        neighbourhood_sample_side=run_params.neighbourhood_sample_side,
        enforce_pair_radius=run_params.enforce_pair_radius,
        neighbourhood_radius_min=run_params.neighbourhood_radius_min,
        initial_cells=initial_cells,
        device_step=run_params.device_step,
        device_short_step=run_params.device_short_step,
        **_sparse_layout_options(),
    )


def _seeded_layout(
    codes: LayoutCodeTable,
    run_params: TunedLayoutParams,
    colorizer: EncoderCodeColorizer | None,
    *,
    lambda_schedule: LayoutLambdaSchedule | None = None,
) -> tuple[Layout, int, int]:
    """Большая раскладка от затравки (вне статьи; ср. подмножество §5.9.4): раскладка, радиус и число шагов доводки.

    Случайное подмножество точек раскладывается целиком обычным алгоритмом (дальняя и ближняя фазы с настройкой
    для своего числа точек); остальные точки ставятся возле самых похожих точек подмножества. Глобальный
    порядок готов — дальняя фаза большой раскладки начинается с радиуса в несколько блоков затравки и идёт
    по тому же геометрическому расписанию, что полная фаза (те же шаги на октаву радиуса, §7.1.7).
    lambda_schedule — свои пороги сходства (коды следующего уровня): с ними раскладываются и затравка, и
    большая раскладка.
    """
    placement = SeededPlacement(
        codes,
        seed_count=_seed_level_points(codes.count, int(run_params.seed_points)),
        seed=settings.LAYOUT_RANDOM_SEED,
        similarity=settings.LAYOUT_SIMILARITY,
    )
    seed_params = LayoutParameterTuner(
        empty_ratio=settings.LAYOUT_EMPTY_RATIO, start_radius_factor=settings.LAYOUT_ADAPTIVE_RADIUS_START_FACTOR
    ).tune(placement.seed_count)
    started = time.perf_counter()
    # Затравка раскладывается тем же путём: целиком либо, если она сама велика, от своей затравки (уровни).
    seed_layout = _layout_from_codes(placement.seed_table(), seed_params, colorizer, lambda_schedule=lambda_schedule)
    seed_cells = seed_layout.cells_array()
    seed_shape = (seed_layout.height, seed_layout.width)
    seed_lambda = seed_layout.lambda_threshold
    seed_seconds = time.perf_counter() - started
    # Таблицы сходства затравки занимают память устройства, нужную большой раскладке.
    seed_layout.release_device()
    del seed_layout
    grid_size = int(math.ceil(math.sqrt(codes.count * (1.0 + settings.LAYOUT_EMPTY_RATIO))))
    started = time.perf_counter()
    cells = placement.cells(
        seed_cells,
        seed_shape,
        (grid_size, grid_size),
        # Режим статьи: место точки в блоке родителя — по похожим на неё точкам затравки (sim ≥ λ затравки).
        neighbour_threshold=None if settings.LAYOUT_SMOOTH_PINWHEELS else seed_lambda,
    )
    layout = _new_layout(
        codes,
        run_params,
        colorizer,
        initial_cells=cells,
        lambda_threshold=None if lambda_schedule is None else lambda_schedule.start,
    )
    if not settings.LAYOUT_SMOOTH_PINWHEELS:
        # Порог λ большой раскладки продолжает расписание затравки (§7.1.7): её дальний порядок уже установлен.
        layout.set_similarity_params(lambda_threshold=seed_lambda)
    full_radius = max(settings.LAYOUT_ADAPTIVE_RADIUS_MIN, int(layout.width * settings.LAYOUT_ADAPTIVE_RADIUS_START_FACTOR))
    # Начальный радиус доводки — в блоках родителя: дальней фазы сглаживания либо ближней фазы режима статьи.
    radius_factor = settings.LAYOUT_SEED_REFINE_RADIUS_FACTOR if settings.LAYOUT_SMOOTH_PINWHEELS else settings.LAYOUT_SEED_SHORT_RADIUS_FACTOR
    refine_radius = min(
        full_radius,
        max(settings.LAYOUT_SEED_REFINE_RADIUS_MIN, int(math.ceil(radius_factor * placement.block_side()))),
    )
    refine_steps = run_params.long_steps
    if full_radius > 1 and refine_radius < full_radius:
        refine_steps = max(1, int(round(run_params.long_steps * math.log(refine_radius) / math.log(full_radius))))
    if not settings.LAYOUT_SMOOTH_PINWHEELS:
        refine_steps = 0
    LOGGER.event(
        "layout.seed.refine",
        section=LAYOUT_PARAMETERS,
        data={
            "points": codes.count,
            "seed_points": placement.seed_count,
            "seed_layout_seconds": seed_seconds,
            "placement_seconds": time.perf_counter() - started,
            "block_side": placement.block_side(),
            "refine_radius": refine_radius,
            "full_radius": full_radius,
            "refine_steps": refine_steps,
            "full_steps": run_params.long_steps,
            "lambda_threshold": layout.lambda_threshold,
            "note": (
                "порядок крупнее блока затравки взят из её раскладки: дальняя фаза идёт с радиуса "
                "refine_radius по тому же геометрическому расписанию (§7.1.7)"
                if settings.LAYOUT_SMOOTH_PINWHEELS
                else "дальний порядок взят из раскладки затравки, порог λ продолжает её расписание (§7.1.7): "
                "идёт ближняя фаза (§5.5.2) с радиуса выбора пар refine_radius"
            ),
        },
    )
    return layout, refine_radius, refine_steps


def _run_long_phase(
    layout: Layout,
    run_params: TunedLayoutParams,
    *,
    step_offset: int,
    layout_path: str | None,
    start_radius: int | None = None,
    steps: int | None = None,
    lambda_max: float | None = None,
) -> None:
    """Дальняя фаза раскладки (§5.5.1, §7.1.7); start_radius и steps — начало и длина расписания доводки,
    lambda_max — предел роста λ вместо LAYOUT_ADAPTIVE_LAMBDA_MAX."""
    long_radius_start = max(
        settings.LAYOUT_ADAPTIVE_RADIUS_MIN,
        int(layout.width * settings.LAYOUT_ADAPTIVE_RADIUS_START_FACTOR),
    )
    if start_radius is not None:
        long_radius_start = int(start_radius)
    adaptive_long = AdaptiveLayoutConfig(
        start_radius=long_radius_start,
        end_radius=settings.LAYOUT_ADAPTIVE_RADIUS_MIN,
        swap_ratio_trigger=settings.LAYOUT_ADAPTIVE_SWAP_TRIGGER,
        lambda_step=settings.LAYOUT_ADAPTIVE_LAMBDA_STEP,
        lambda_max=settings.LAYOUT_ADAPTIVE_LAMBDA_MAX if lambda_max is None else float(lambda_max),
        radius_schedule=settings.LAYOUT_ADAPTIVE_RADIUS_SCHEDULE,
        radius_factor=settings.LAYOUT_ADAPTIVE_RADIUS_TRIGGER_FACTOR,
        swap_window=settings.LAYOUT_ADAPTIVE_SWAP_WINDOW,
        temperature=settings.LAYOUT_ANNEAL_TEMPERATURE,
        temperature_radius_max=int(layout.width * settings.LAYOUT_ANNEAL_RADIUS_MAX_FACTOR),
        temperature_radius_min=settings.LAYOUT_ANNEAL_RADIUS_MIN,
    )
    layout.run(
        steps=run_params.long_steps if steps is None else int(steps),
        pairs_per_step=run_params.long_pairs_per_step,
        pair_radius=adaptive_long.start_radius,
        mode="long",
        min_swap_ratio=settings.LAYOUT_MIN_SWAP_RATIO,
        min_swap_window=settings.LAYOUT_MIN_SWAP_WINDOW,
        log_every=settings.LAYOUT_LOG_EVERY_LONG,
        step_offset=step_offset,
        energy_radius=settings.LAYOUT_ENERGY_RADIUS,
        energy_every=settings.LAYOUT_ENERGY_EVAL_EVERY,
        energy_target=settings.LAYOUT_ENERGY_TARGET,
        log_visuals=settings.LAYOUT_LOG_VISUALS,
        adaptive_params=adaptive_long,
        save_on_stop_path=layout_path,
        long_weighted_first=bool(settings.LAYOUT_LONG_WEIGHTED_FIRST),
    )


def _run_short_phase(
    layout: Layout,
    run_params: TunedLayoutParams,
    *,
    step_offset: int,
    start_radius: int | None = None,
    similarity_cutoff: float | None = None,
    lambda_max: float | None = None,
) -> None:
    """Ближняя фаза раскладки (§5.5.2, §7.1.7); start_radius — начальный радиус выбора пар доводки от затравки,
    similarity_cutoff — порог ранней отсечки пар (§5.9.1) вместо LAYOUT_SHORT_SIMILARITY_CUTOFF,
    lambda_max — предел роста λ вместо LAYOUT_ADAPTIVE_LAMBDA_MAX."""
    short_radius_max = int(layout.width * settings.LAYOUT_ADAPTIVE_RADIUS_START_FACTOR)
    short_radius_base = int(
        max(
            1,
            round(run_params.short_local_radius * settings.LAYOUT_SHORT_ADAPTIVE_RADIUS_FACTOR),
        )
    )
    short_radius_start = max(
        settings.LAYOUT_ADAPTIVE_RADIUS_MIN,
        min(short_radius_base, short_radius_max),
    )
    if start_radius is not None:
        short_radius_start = max(settings.LAYOUT_ADAPTIVE_RADIUS_MIN, min(short_radius_start, int(start_radius)))
    adaptive_short = AdaptiveLayoutConfig(
        start_radius=short_radius_start,
        end_radius=settings.LAYOUT_ADAPTIVE_RADIUS_MIN,
        swap_ratio_trigger=settings.LAYOUT_ADAPTIVE_SWAP_TRIGGER,
        lambda_step=settings.LAYOUT_ADAPTIVE_LAMBDA_STEP,
        lambda_max=settings.LAYOUT_ADAPTIVE_LAMBDA_MAX if lambda_max is None else float(lambda_max),
        radius_schedule=settings.LAYOUT_SHORT_ADAPTIVE_RADIUS_SCHEDULE,
        radius_factor=settings.LAYOUT_ADAPTIVE_RADIUS_TRIGGER_FACTOR,
        swap_window=settings.LAYOUT_ADAPTIVE_SWAP_WINDOW,
    )
    short_optim = ShortLayoutOptimConfig(
        energy_radius=run_params.short_energy_radius,
        energy_recalc_every=run_params.short_energy_recalc_every,
        energy_eps=settings.LAYOUT_SHORT_ENERGY_EPS,
        use_weighted_first_point=settings.LAYOUT_SHORT_WEIGHTED_FIRST,
        similarity_cutoff=settings.LAYOUT_SHORT_SIMILARITY_CUTOFF if similarity_cutoff is None else similarity_cutoff,
        partitions=settings.LAYOUT_SHORT_PARTITIONS,
    )
    layout.run(
        steps=run_params.short_steps,
        pairs_per_step=run_params.short_pairs_per_step,
        pair_radius=short_radius_start,
        mode="short",
        local_radius=run_params.short_local_radius,
        min_swap_ratio=settings.LAYOUT_SHORT_MIN_SWAP_RATIO,
        min_swap_window=settings.LAYOUT_MIN_SWAP_WINDOW,
        log_every=settings.LAYOUT_LOG_EVERY_SHORT,
        step_offset=step_offset,
        adaptive_params=adaptive_short,
        short_optim=short_optim,
        # Картинки и цвета карты — только для журнала и Rerun (анализ 2026-11-21: ближняя фаза рисовала их и без Rerun).
        log_visuals=settings.LAYOUT_LOG_VISUALS,
    )
