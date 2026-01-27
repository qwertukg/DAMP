import glob
import math
from dataclasses import dataclass
import json
import os
from collections import OrderedDict
from typing import ClassVar, Sequence
import argparse

from damp.article_refs import (
    CODE_SPACE_ACTIVATION,
    DETECTOR_CONSTRUCTION,
    DETECTOR_HIERARCHY,
    DETECTOR_RADIUS,
    DETECTOR_INSERTION,
    ENERGY_CALC,
    LAYOUT_PARAMETERS,
    LAYOUT_SYSTEM,
    LAID_OUT_STRUCTURE,
    STIMULUS_DETECTION,
    ENCODING_SYSTEM,
    GPU_IMPLEMENTATION,
)
from damp.activation import ActivationEngine, Detector
from damp.MnistSobelAngleMap import MnistSobelAngleMap
from damp.encoding.damp_encoder import ClosedDimension, Detectors, Encoder, OpenedDimension
from damp.layout.damp_layout import AdaptiveLayoutConfig, Layout, ShortLayoutOptimConfig
from damp.layout.payload import LayoutEncodedCode, LayoutPayloadBuilder
from damp.encoding.bitarray import BitArray
from damp.logging import LOGGER
from damp.decode import DecoderConfig, EmbeddingDatabase, LabeledEmbedding, MnistDecoder, PredictionResult

LOG_INTERVAL_DEFAULT = 1
LOG_INTERVAL_INIT = 1
LOG_INTERVAL_SOBEL_IMAGE = 25
LOG_INTERVAL_SOBEL_PATCH = 2000
LOG_INTERVAL_ENCODER_ENCODE = 100
LOG_INTERVAL_ENCODER_IMAGE = 10
LOG_INTERVAL_LAYOUT_ENERGY = 1000
LOG_INTERVAL_LAYOUT_PAIR = 250000
LOG_INTERVAL_LAYOUT_AVG_ENERGY = 200
LOG_INTERVAL_LAYOUT_VISUAL = 1
LOG_INTERVAL_LAYOUT_ADAPTIVE = 200
LOG_INTERVAL_LAYOUT_SWAP_RATIO = 50
LOG_INTERVAL_LAYOUT_EXPORT = 1
LOG_INTERVAL_LAYOUT_ENERGY_PROGRESS = 1
LOG_INTERVAL_ACTIVATION_INIT = 1
LOG_INTERVAL_ACTIVATION_MAP = 1
LOG_INTERVAL_ACTIVATION_DETECTOR_LEVEL = 10
LOG_INTERVAL_ACTIVATION_DETECTOR_ACTIVE = 1
LOG_INTERVAL_ACTIVATION_EMBEDDING = 1
LOG_INTERVAL_ACTIVATION_COLOR_MERGE = 1
LOG_INTERVAL_ACTIVATION_COVERAGE = 1
LOG_INTERVAL_ACTIVATION_ENERGY_MAP = 1
LOG_INTERVAL_ACTIVATION_DETECTOR_LAYER = 1
LOG_INTERVAL_ACTIVATION_SAMPLE = 1
LOG_INTERVAL_ACTIVATION_VISUAL = 1
LOG_INTERVAL_ACTIVATION_EMBEDDING_SAVE = 1

EMBEDDINGS_JSON_KEY = "embeddings"

LAYOUT_SIMILARITY = "cosine"
LAYOUT_LAMBDA_THRESHOLD = 0.65
LAYOUT_ETA = 0
LAYOUT_EMPTY_RATIO = 0.5
LAYOUT_USE_GPU = True

LAYOUT_LONG_PAIRS_PER_STEP = 24000
LAYOUT_LONG_STEPS = 14000
LAYOUT_LONG_SUBSET_SIZE = 1024
LAYOUT_LONG_SUBSET_REFRESH = 40

LAYOUT_SHORT_PAIRS_PER_STEP = 100
LAYOUT_SHORT_STEPS = 4000
LAYOUT_SHORT_LOCAL_RADIUS = 7
LAYOUT_SHORT_ENERGY_RADIUS = 7
LAYOUT_SHORT_ENERGY_RECALC_EVERY = 75
LAYOUT_SHORT_ENERGY_MAX_POINTS = 256
LAYOUT_SHORT_ENERGY_EPS = 1e-6
LAYOUT_SHORT_WEIGHTED_FIRST = True
LAYOUT_SHORT_SIMILARITY_CUTOFF = LAYOUT_LAMBDA_THRESHOLD
LAYOUT_SHORT_PARTITIONS = 8

MPS_HIGH_WATERMARK_RATIO = 0.0

ENCODER_LOG_EVERY = 50
LAYOUT_LOG_EVERY_LONG = 150
LAYOUT_LOG_EVERY_SHORT = 50
LAYOUT_LOG_VISUALS = True
LAYOUT_ADAPTIVE_RADIUS_START_FACTOR = 0.5
LAYOUT_ADAPTIVE_RADIUS_MIN = 1
LAYOUT_ADAPTIVE_SWAP_TRIGGER = 0.01
LAYOUT_ADAPTIVE_LAMBDA_STEP = 0.05
LAYOUT_SHORT_ADAPTIVE_RADIUS_FACTOR = 1.0
LAYOUT_ENERGY_STABILITY_WINDOW = 20
LAYOUT_ENERGY_STABILITY_DELTA = 0.0005
LAYOUT_ENERGY_STABILITY_EVERY = 200
LAYOUT_ENERGY_STABILITY_MAX_POINTS = 128
LAYOUT_MIN_SWAP_RATIO = 0.003
LAYOUT_MIN_SWAP_WINDOW = 1

ACTIVATION_LAMBDA_THRESHOLD = 0.6
ACTIVATION_ETA = None
ACTIVATION_ENERGY_THRESHOLD = 0.12
ACTIVATION_DETECTOR_THRESHOLD = 0.45
ACTIVATION_SATURATION_LIMIT = 260
ACTIVATION_POINT_ENERGY_RADIUS = 3
ACTIVATION_DETECTOR_RADIUS = 6
ACTIVATION_DETECTOR_RADIUS_MIN = 1
ACTIVATION_DETECTOR_TARGET_POINTS = 28
ACTIVATION_DETECTOR_NMS_MIN_DIST_FACTOR = 0.8
ACTIVATION_DETECTOR_STEP_DENSITY = 6

ACTIVATION_MAX_DETECTORS = 2048
ACTIVATION_OUTPUT_BITS = 8192
ACTIVATION_SAMPLE_IMAGES = 6000
ACTIVATION_LAMBDA_LEVELS = (0.45, 0.55, 0.65, 0.75, 0.85)
ACTIVATION_USE_GPU = True

DECODER_SIMILARITY = "jaccard2"
DECODER_SOFTMAX_TEMPERATURE = 0.35
DECODER_MIN_CONFIDENCE = 0.05
DECODER_EXPECTED_CLASSES = tuple(range(10))
DECODER_DATASET_LIMIT = 64
DECODER_DEFAULT_COUNT = 1000
DECODER_CALIBRATION_TEMPERATURES = (0.25, 0.35, 0.5, 0.75, 1.0)

LAYOUT_TUNE_LONG_PAIR_FACTOR = 0.35
LAYOUT_TUNE_LONG_PAIRS_MIN = 4000
LAYOUT_TUNE_LONG_PAIRS_MAX = 20000
LAYOUT_TUNE_LONG_STEPS_DIVISOR = 25
LAYOUT_TUNE_LONG_STEPS_MIN = 300
LAYOUT_TUNE_LONG_STEPS_MAX = 1500
LAYOUT_TUNE_LONG_SUBSET_MULTIPLIER = 8
LAYOUT_TUNE_LONG_SUBSET_MIN = 512
LAYOUT_TUNE_LONG_SUBSET_MAX = 2048
LAYOUT_TUNE_LONG_SUBSET_REFRESH_DIVISOR = 18
LAYOUT_TUNE_LONG_SUBSET_REFRESH_MIN = 20
LAYOUT_TUNE_LONG_SUBSET_REFRESH_MAX = 80
LAYOUT_TUNE_SHORT_PAIR_FACTOR = 0.2
LAYOUT_TUNE_SHORT_PAIRS_MIN = 1800
LAYOUT_TUNE_SHORT_PAIRS_MAX = 12000
LAYOUT_TUNE_SHORT_STEPS_FACTOR = 1.6
LAYOUT_TUNE_SHORT_STEPS_MIN = 160
LAYOUT_TUNE_SHORT_STEPS_MAX = 900
LAYOUT_TUNE_SHORT_RADIUS_FACTOR = 0.05
LAYOUT_TUNE_SHORT_ENERGY_RADIUS_FACTOR = 0.04
LAYOUT_TUNE_SHORT_ENERGY_RECALC_FLOOR = 30
LOG_INTERVALS = {
    "detectors.init": LOG_INTERVAL_INIT,
    "dimension.closed": LOG_INTERVAL_INIT,
    "dimension.init": LOG_INTERVAL_INIT,
    "encoder.code_length": LOG_INTERVAL_INIT,
    "encoder.detectors": LOG_INTERVAL_INIT,
    "encoder.encode.code": LOG_INTERVAL_ENCODER_ENCODE,
    "encoder.encode.image": LOG_INTERVAL_ENCODER_IMAGE,
    "encoder.encode.input": LOG_INTERVAL_ENCODER_ENCODE,
    "encoder.encode.normalized": LOG_INTERVAL_ENCODER_ENCODE,
    "encoder.init": LOG_INTERVAL_INIT,
    "encoder.layer": LOG_INTERVAL_INIT,
    "encoder.random_bit": LOG_INTERVAL_INIT,
    "layout.codes": LOG_INTERVAL_INIT,
    "layout.distance_eps": LOG_INTERVAL_INIT,
    "layout.energy.average": LOG_INTERVAL_LAYOUT_AVG_ENERGY,
    "layout.energy.long.empty": LOG_INTERVAL_LAYOUT_ENERGY,
    "layout.energy.long.pair": LOG_INTERVAL_LAYOUT_PAIR,
    "layout.energy.long.sim_lambda": LOG_INTERVAL_LAYOUT_ENERGY,
    "layout.energy.long.similarity": LOG_INTERVAL_LAYOUT_ENERGY,
    "layout.energy.long.space": LOG_INTERVAL_LAYOUT_ENERGY,
    "layout.energy.long.threshold": LOG_INTERVAL_LAYOUT_ENERGY,
    "layout.energy.long.tensor": LOG_INTERVAL_LAYOUT_ENERGY,
    "layout.energy.short.tensor": LOG_INTERVAL_LAYOUT_ENERGY,
    "layout.energy.short.tensor.empty_subset": LOG_INTERVAL_LAYOUT_ENERGY,
    "layout.energy.pair.ignore_self": LOG_INTERVAL_LAYOUT_ENERGY,
    "layout.energy.long.progress": LOG_INTERVAL_LAYOUT_ENERGY_PROGRESS,
    "layout.gpu.config": LOG_INTERVAL_INIT,
    "layout.gpu.context_failed": LOG_INTERVAL_INIT,
    "layout.gpu.disabled": LOG_INTERVAL_INIT,
    "layout.gpu.enabled": LOG_INTERVAL_INIT,
    "layout.gpu.import_failed": LOG_INTERVAL_INIT,
    "layout.gpu.init_failed": LOG_INTERVAL_INIT,
    "layout.gpu.tensor.disabled": LOG_INTERVAL_INIT,
    "layout.gpu.tensor.enabled": LOG_INTERVAL_INIT,
    "layout.gpu.tensor.unavailable": LOG_INTERVAL_INIT,
    "layout.gpu.tensor.short.disabled": LOG_INTERVAL_INIT,
    "layout.gpu.tensor.short.sim_failed": LOG_INTERVAL_INIT,
    "layout.gpu.version_too_low": LOG_INTERVAL_INIT,
    "layout.gpu.mps.watermark": LOG_INTERVAL_INIT,
    "layout.gpu.mps.empty_cache": LOG_INTERVAL_INIT,
    "layout.grid": LOG_INTERVAL_INIT,
    "layout.init": LOG_INTERVAL_INIT,
    "layout.parallel": LOG_INTERVAL_INIT,
    "layout.parallel.cpu_count": LOG_INTERVAL_INIT,
    "layout.precompute": LOG_INTERVAL_INIT,
    "layout.precompute_limit": LOG_INTERVAL_INIT,
    "layout.render_image": LOG_INTERVAL_LAYOUT_VISUAL,
    "layout.adaptive.config": LOG_INTERVAL_INIT,
    "layout.adaptive.radius": LOG_INTERVAL_LAYOUT_ADAPTIVE,
    "layout.adaptive.lambda": LOG_INTERVAL_INIT,
    "layout.run.done": LOG_INTERVAL_INIT,
    "layout.run.energy_monitor": LOG_INTERVAL_INIT,
    "layout.run.energy_stability": LOG_INTERVAL_INIT,
    "layout.run.energy_stability.stop": LOG_INTERVAL_INIT,
    "layout.run.mode": LOG_INTERVAL_INIT,
    "layout.run.selection": LOG_INTERVAL_INIT,
    "layout.run.settings": LOG_INTERVAL_INIT,
    "layout.run.swap_monitor": LOG_INTERVAL_INIT,
    "layout.run.swap_ratio": LOG_INTERVAL_LAYOUT_SWAP_RATIO,
    "layout.run.swap_stability.stop": LOG_INTERVAL_INIT,
    "layout.run.save_on_stop": LOG_INTERVAL_LAYOUT_EXPORT,
    "layout.run.save_on_stop.error": LOG_INTERVAL_LAYOUT_EXPORT,
    "layout.sim_cache.done": LOG_INTERVAL_INIT,
    "layout.sim_cache.gpu": LOG_INTERVAL_INIT,
    "layout.sim_cache.parallel": LOG_INTERVAL_INIT,
    "layout.sim_cache.similarity": LOG_INTERVAL_INIT,
    "layout.sim_cache.start": LOG_INTERVAL_INIT,
    "layout.similarity": LOG_INTERVAL_INIT,
    "layout.similarity_params": LOG_INTERVAL_INIT,
    "layout.thresholds": LOG_INTERVAL_INIT,
    "layout.visual": LOG_INTERVAL_LAYOUT_VISUAL,
    "layout.export.json": LOG_INTERVAL_LAYOUT_EXPORT,
    "layout.export.json.error": LOG_INTERVAL_LAYOUT_EXPORT,
    "layout.load.cache": LOG_INTERVAL_INIT,
    "sobel_map.extract": LOG_INTERVAL_SOBEL_IMAGE,
    "sobel_map.init": LOG_INTERVAL_INIT,
    "sobel_map.patch.angle": LOG_INTERVAL_SOBEL_PATCH,
    "sobel_map.patch.metrics": LOG_INTERVAL_SOBEL_PATCH,
    "sobel_map.patch_coords": LOG_INTERVAL_SOBEL_IMAGE,
    "sobel_map.patch_grid": LOG_INTERVAL_SOBEL_IMAGE,
    "activation.backend": LOG_INTERVAL_ACTIVATION_INIT,
    "activation.backend.gpu_unavailable": LOG_INTERVAL_ACTIVATION_INIT,
    "activation.backend.gpu_fallback": LOG_INTERVAL_ACTIVATION_INIT,
    "activation.engine.init": LOG_INTERVAL_ACTIVATION_INIT,
    "activation.map": LOG_INTERVAL_ACTIVATION_MAP,
    "activation.detector.level": LOG_INTERVAL_ACTIVATION_DETECTOR_LEVEL,
    "activation.detector.active": LOG_INTERVAL_ACTIVATION_DETECTOR_ACTIVE,
    "activation.detector.init": LOG_INTERVAL_ACTIVATION_INIT,
    "activation.detector.layer": LOG_INTERVAL_ACTIVATION_DETECTOR_LAYER,
    "activation.embedding": LOG_INTERVAL_ACTIVATION_EMBEDDING,
    "activation.color_merge": LOG_INTERVAL_ACTIVATION_COLOR_MERGE,
    "activation.detector.coverage": LOG_INTERVAL_ACTIVATION_COVERAGE,
    "activation.energy.map": LOG_INTERVAL_ACTIVATION_ENERGY_MAP,
    "activation.sample": LOG_INTERVAL_ACTIVATION_SAMPLE,
    "activation.map.visual": LOG_INTERVAL_ACTIVATION_VISUAL,
    "activation.energy.visual": LOG_INTERVAL_ACTIVATION_VISUAL,
    "activation.embedding.save": LOG_INTERVAL_ACTIVATION_EMBEDDING_SAVE,
}

LAYOUT_CACHE_PATTERN = "data/layout-{count}-*.json"


def _layout_cache_paths_for_count(count: int) -> list[str]:
    pattern = LAYOUT_CACHE_PATTERN.format(count=count)
    return sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)


def _load_layout_metadata_from_path(path: str) -> tuple[int, int | None, bool] | None:
    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except Exception as exc:
        LOGGER.event(
            "layout.cache.read_error",
            section=LAID_OUT_STRUCTURE,
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
        section=LAID_OUT_STRUCTURE,
        data={
            "path": path,
            "points": total_points,
            "pair_radius": pair_radius,
            "ready": ready,
        },
    )
    return total_points, pair_radius, ready


def configure_logging() -> None:
    LOGGER.configure_intervals(LOG_INTERVALS, default_interval=LOG_INTERVAL_DEFAULT)


class GpuEnvironment:
    _applied: ClassVar[bool] = False

    def __init__(self, *, mps_high_watermark: float | None) -> None:
        self._mps_high_watermark = mps_high_watermark

    def apply(self) -> None:
        if GpuEnvironment._applied:
            return
        GpuEnvironment._applied = True
        if self._mps_high_watermark is None:
            return
        target = str(self._mps_high_watermark)
        current = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
        mode = "preserve"
        value = current if current is not None else target
        if current is None:
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = target
            mode = "set"
        LOGGER.event(
            "layout.gpu.mps.watermark",
            section=GPU_IMPLEMENTATION,
            data={
                "mode": mode,
                "value": value,
            },
        )


GPU_ENVIRONMENT = GpuEnvironment(mps_high_watermark=MPS_HIGH_WATERMARK_RATIO)


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


@dataclass(frozen=True)
class EmbeddingRecord:
    class_label: str
    embedding_bits: str

    def as_payload(self) -> dict[str, str]:
        return {
            "class": self.class_label,
            "embedding": self.embedding_bits,
        }


class EmbeddingStore:
    def __init__(self, json_key: str) -> None:
        if not json_key:
            raise ValueError("json_key must be provided for embedding storage")
        self._json_key = json_key

    def save(self, path: str, embeddings: Sequence[EmbeddingRecord]) -> None:
        if not path:
            raise ValueError("path must be provided to save embeddings")
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload: dict[str, object] = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fp:
                    payload = json.load(fp)
            except Exception as exc:
                LOGGER.event(
                    "activation.embedding.save",
                    section=DETECTOR_INSERTION,
                    data={
                        "path": path,
                        "embeddings": len(embeddings),
                        "error": str(exc),
                        "stage": "read",
                    },
                )
                payload = {}
        payload[self._json_key] = [record.as_payload() for record in embeddings]
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=True, indent=2)
        LOGGER.event(
            "activation.embedding.save",
            section=DETECTOR_INSERTION,
            data={
                "path": path,
                "embeddings": len(embeddings),
            "json_key": self._json_key,
        },
    )


def _load_embeddings_from_file(path: str) -> list["EmbeddingRecord"]:
    if not path:
        raise ValueError("path must be provided to load embeddings")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    raw_embeddings = data.get(EMBEDDINGS_JSON_KEY, [])
    embeddings: list[EmbeddingRecord] = []
    for item in raw_embeddings:
        class_label = item.get("class", item.get("label"))
        embedding_bits = item.get("embedding")
        if class_label is None or embedding_bits is None:
            continue
        embeddings.append(
            EmbeddingRecord(
                class_label=str(class_label),
                embedding_bits=str(embedding_bits),
            )
        )
    classes = sorted({int(rec.class_label) for rec in embeddings})
    LOGGER.event(
        "activation.embedding.load",
        section=DETECTOR_HIERARCHY,
        data={
            "path": path,
            "embeddings": len(embeddings),
            "classes": classes,
            "json_key": EMBEDDINGS_JSON_KEY,
        },
    )
    return embeddings


def _embedding_records_to_labeled(records: Sequence["EmbeddingRecord"]) -> list[LabeledEmbedding]:
    labeled: list[LabeledEmbedding] = []
    for record in records:
        bit_string = record.embedding_bits
        bit_array = BitArray(len(bit_string))
        for idx, char in enumerate(bit_string):
            if char == "1":
                bit_array.set(idx, 1)
        labeled.append(
            LabeledEmbedding(
                label=int(record.class_label),
                embedding=bit_array,
            )
        )
    if labeled:
        LOGGER.event(
            "decoder.base.prepare",
            section=DETECTOR_HIERARCHY,
            data={
                "embeddings": len(labeled),
                "classes": len({item.label for item in labeled}),
                "bit_length": len(labeled[0].embedding),
            },
        )
    return labeled


class LayoutParameterTuner:
    def __init__(
        self,
        *,
        empty_ratio: float = LAYOUT_EMPTY_RATIO,
        start_radius_factor: float = LAYOUT_ADAPTIVE_RADIUS_START_FACTOR,
    ) -> None:
        self._empty_ratio = empty_ratio
        self._start_radius_factor = start_radius_factor

    def tune(self, total_points: int) -> TunedLayoutParams:
        if total_points <= 0:
            raise ValueError("total_points must be positive for tuning")
        grid_size = int(math.ceil(math.sqrt(total_points * (1.0 + self._empty_ratio))))
        long_pairs = max(
            LAYOUT_TUNE_LONG_PAIRS_MIN,
            min(LAYOUT_TUNE_LONG_PAIRS_MAX, int(total_points * LAYOUT_TUNE_LONG_PAIR_FACTOR)),
        )
        long_steps = max(
            LAYOUT_TUNE_LONG_STEPS_MIN,
            min(LAYOUT_TUNE_LONG_STEPS_MAX, int(total_points / LAYOUT_TUNE_LONG_STEPS_DIVISOR)),
        )
        long_subset_size = min(
            LAYOUT_TUNE_LONG_SUBSET_MAX,
            max(LAYOUT_TUNE_LONG_SUBSET_MIN, int(math.sqrt(total_points) * LAYOUT_TUNE_LONG_SUBSET_MULTIPLIER)),
        )
        long_subset_refresh = max(
            LAYOUT_TUNE_LONG_SUBSET_REFRESH_MIN,
            min(
                LAYOUT_TUNE_LONG_SUBSET_REFRESH_MAX,
                long_steps // LAYOUT_TUNE_LONG_SUBSET_REFRESH_DIVISOR,
            ),
        )
        short_pairs = max(
            LAYOUT_TUNE_SHORT_PAIRS_MIN,
            min(LAYOUT_TUNE_SHORT_PAIRS_MAX, int(total_points * LAYOUT_TUNE_SHORT_PAIR_FACTOR)),
        )
        short_steps = max(
            LAYOUT_TUNE_SHORT_STEPS_MIN,
            min(LAYOUT_TUNE_SHORT_STEPS_MAX, int(grid_size * LAYOUT_TUNE_SHORT_STEPS_FACTOR)),
        )
        short_local_radius = max(
            LAYOUT_ADAPTIVE_RADIUS_MIN, int(grid_size * LAYOUT_TUNE_SHORT_RADIUS_FACTOR)
        )
        short_energy_radius = max(
            LAYOUT_ADAPTIVE_RADIUS_MIN, int(grid_size * LAYOUT_TUNE_SHORT_ENERGY_RADIUS_FACTOR)
        )
        short_energy_recalc_every = max(LAYOUT_TUNE_SHORT_ENERGY_RECALC_FLOOR, short_steps // 6)
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
                "long_subset_size": long_subset_size,
                "long_subset_refresh": long_subset_refresh,
                "short_pairs_per_step": short_pairs,
                "short_steps": short_steps,
                "short_local_radius": short_local_radius,
                "short_energy_radius": short_energy_radius,
                "short_energy_recalc_every": short_energy_recalc_every,
                "lambda_threshold": LAYOUT_LAMBDA_THRESHOLD,
                "eta": "hard" if LAYOUT_ETA is None else LAYOUT_ETA,
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
        )


def _default_layout_params() -> TunedLayoutParams:
    return TunedLayoutParams(
        long_pairs_per_step=LAYOUT_LONG_PAIRS_PER_STEP,
        long_steps=LAYOUT_LONG_STEPS,
        long_subset_size=LAYOUT_LONG_SUBSET_SIZE,
        long_subset_refresh=LAYOUT_LONG_SUBSET_REFRESH,
        short_pairs_per_step=LAYOUT_SHORT_PAIRS_PER_STEP,
        short_steps=LAYOUT_SHORT_STEPS,
        short_local_radius=LAYOUT_SHORT_LOCAL_RADIUS,
        short_energy_radius=LAYOUT_SHORT_ENERGY_RADIUS,
        short_energy_recalc_every=LAYOUT_SHORT_ENERGY_RECALC_EVERY,
    )


def _layout_json_path(count: int, pair_radius: int | None) -> str:
    suffix = "auto" if pair_radius is None else str(pair_radius)
    return f"data/layout-{count}-{suffix}.json"


def _expected_pair_radius(total_points: int, run_params: TunedLayoutParams) -> int:
    if total_points <= 0:
        raise ValueError("total_points must be positive for pair radius estimation")
    grid_size_est = int(math.ceil(math.sqrt(total_points * (1.0 + LAYOUT_EMPTY_RATIO))))
    expected_short_radius_max = int(grid_size_est * LAYOUT_ADAPTIVE_RADIUS_START_FACTOR)
    expected_short_radius_base = int(
        max(1, round(run_params.short_local_radius * LAYOUT_SHORT_ADAPTIVE_RADIUS_FACTOR))
    )
    return max(
        LAYOUT_ADAPTIVE_RADIUS_MIN,
        min(expected_short_radius_base, expected_short_radius_max),
    )


def _points_in_radius(
    center: tuple[int, int],
    positions: list[tuple[int, int]],
    radius_sq: float,
) -> list[int]:
    cy, cx = center
    covered: list[int] = []
    for idx, (y, x) in enumerate(positions):
        dy = float(cy) - float(y)
        dx = float(cx) - float(x)
        if dy * dy + dx * dx <= radius_sq:
            covered.append(idx)
    return covered


def _compute_point_energies(layout: Layout, radius: int) -> list[float]:
    if radius <= 0:
        raise ValueError("radius must be positive for activation energy map")
    point_count = len(layout._positions)
    if point_count == 0:
        return []
    radius_sq = radius * radius
    energies: list[float] = []
    pos_y = layout._pos_y
    pos_x = layout._pos_x
    row_occupied = layout._row_occupied
    distance_eps = layout._distance_eps
    sim_lambda = layout._sim_lambda_idx
    for idx in range(point_count):
        cy = pos_y[idx]
        cx = pos_x[idx]
        energy = 0.0
        y_min = max(0, cy - radius)
        y_max = min(layout.height - 1, cy + radius)
        x_min = max(0, cx - radius)
        x_max = min(layout.width - 1, cx + radius)
        for y in range(y_min, y_max + 1):
            dy = cy - y
            dy_sq = dy * dy
            occupied = row_occupied[y]
            if not occupied:
                continue
            for x in occupied:
                if x < x_min or x > x_max:
                    continue
                dx = cx - x
                dist_sq = dy_sq + dx * dx
                if dist_sq > radius_sq:
                    continue
                other_idx = layout.grid[y][x]
                if other_idx is None or other_idx == idx:
                    continue
                sim = sim_lambda(idx, other_idx)
                if sim <= 0.0:
                    continue
                energy += sim / (dist_sq + distance_eps)
        energies.append(energy)
    emax = max(energies)
    normalized = [e / emax if emax > 0 else 0.0 for e in energies]
    LOGGER.event(
        "activation.energy.map",
        section=ENERGY_CALC,
        data={
            "radius": radius,
            "points": point_count,
            "energy_max": emax,
        },
    )
    return normalized


def _build_detectors_from_layout(
    layout: Layout,
    energies: list[float],
    *,
    radius: int,
    max_detectors: int,
    output_bits: int,
    lambda_levels: tuple[float, ...],
) -> list[Detector]:
    """
    Построение иерархии детекторов "как в статье":
    - несколько уровней λ (детекторная иерархия);
    - радиус детектора выбирается адаптивно (приближение fill-factor) по целевому числу точек покрытия;
    - центры детекторов выбираются из точек с высокой статической энергией и разреживаются (NMS);
    - выходной код детектора — уникальный бит в своём слое (без коллизий).
    """
    if radius <= 0:
        raise ValueError("detector radius must be positive")
    if max_detectors <= 0:
        raise ValueError("max_detectors must be positive")
    if output_bits <= 0:
        raise ValueError("output_bits must be positive")
    if not lambda_levels:
        raise ValueError("lambda_levels must be non-empty")
    if len(energies) != len(layout._positions):
        raise ValueError("energies size must match layout points")

    layers = len(lambda_levels)
    per_layer_limit = max(1, max_detectors // layers)
    required_bits = per_layer_limit * layers
    if output_bits < required_bits:
        raise ValueError(
            f"output_bits must be >= per_layer_limit*layers ({required_bits}), got {output_bits}"
        )

    # Предпочитаем точки с высокой статической энергией
    sorted_indices = sorted(range(len(energies)), key=lambda i: energies[i], reverse=True)

    # Быстрая выборка индексов по радиусу через grid/row_occupied (как в _compute_point_energies)
    row_occupied = layout._row_occupied
    grid = layout.grid
    height = layout.height
    width = layout.width
    pos_y = layout._pos_y
    pos_x = layout._pos_x

    def covered_indices_for(center_idx: int, r: int) -> list[int]:
        cy = pos_y[center_idx]
        cx = pos_x[center_idx]
        r_sq = r * r
        covered: list[int] = []
        y_min = max(0, cy - r)
        y_max = min(height - 1, cy + r)
        x_min = max(0, cx - r)
        x_max = min(width - 1, cx + r)
        for y in range(y_min, y_max + 1):
            dy = cy - y
            dy_sq = dy * dy
            occupied = row_occupied[y]
            if not occupied:
                continue
            for x in occupied:
                if x < x_min or x > x_max:
                    continue
                dx = cx - x
                if dy_sq + dx * dx > r_sq:
                    continue
                other_idx = grid[y][x]
                if other_idx is None:
                    continue
                covered.append(other_idx)
        return covered

    # Адаптивный радиус (приближение подбора fill-factor)
    r_min = ACTIVATION_DETECTOR_RADIUS_MIN
    r_max = radius
    target_points = max(1, ACTIVATION_DETECTOR_TARGET_POINTS)

    def choose_radius(center_idx: int) -> tuple[int, list[int]]:
        chosen_r = r_max
        chosen_cov: list[int] = []
        for r in range(r_min, r_max + 1):
            cov = covered_indices_for(center_idx, r)
            if len(cov) >= target_points:
                return r, cov
            # запомним лучший (самый большой) на случай, если до цели не дотянули
            if len(cov) > len(chosen_cov):
                chosen_cov = cov
                chosen_r = r
        return chosen_r, chosen_cov

    # NMS: не даём центрам быть слишком близко (избегаем дубликатов детекторов)
    min_dist = max(1.0, float(r_max) * float(ACTIVATION_DETECTOR_NMS_MIN_DIST_FACTOR))
    min_dist_sq = min_dist * min_dist
    chosen_centers: list[tuple[float, float]] = []

    detectors: list[Detector] = []

    for layer_idx, lambda_value in enumerate(lambda_levels):
        layer_detectors: list[Detector] = []
        layer_offset = layer_idx * per_layer_limit

        # Берём кандидатов плотнее, чем "строго per_layer_limit", чтобы NMS не выжигал слой
        step = max(1, len(sorted_indices) // (per_layer_limit * max(1, int(ACTIVATION_DETECTOR_STEP_DENSITY))))

        local_id = 0
        for idx in sorted_indices[::step]:
            if len(layer_detectors) >= per_layer_limit or len(detectors) >= max_detectors:
                break

            cy = pos_y[idx]
            cx = pos_x[idx]
            # NMS check
            too_close = False
            for (py, px) in chosen_centers:
                dy = float(cy) - float(py)
                dx = float(cx) - float(px)
                if dy * dy + dx * dx < min_dist_sq:
                    too_close = True
                    break
            if too_close:
                continue

            chosen_r, covered = choose_radius(idx)
            if not covered:
                continue

            # В статье далее используется энергия точки (static energy) — мы используем сумму в области как e_d
            energy_sum = 0.0
            for i in covered:
                energy_sum += energies[i]
            if energy_sum <= 0.0:
                continue

            # Уникальный бит в своём слое (без коллизий)
            bit_index = layer_offset + local_id
            if bit_index >= output_bits:
                break
            output_code = BitArray(output_bits)
            output_code.set(bit_index, 1)
            local_id += 1

            center = (cy, cx)
            detector = Detector(
                center=center,
                radius=float(chosen_r),
                lambda_threshold=lambda_value,
                energy=energy_sum,
                points_count=len(covered),
                output_code=output_code,
                label=f"λ={lambda_value:.2f}",
            )

            layer_detectors.append(detector)
            detectors.append(detector)
            chosen_centers.append((float(cy), float(cx)))

        LOGGER.event(
            "activation.detector.layer",
            section=DETECTOR_CONSTRUCTION,
            data={
                "lambda": lambda_value,
                "layer_detectors": len(layer_detectors),
                "per_layer_limit": per_layer_limit,
                "total_detectors": len(detectors),
                "radius_min": r_min,
                "radius_max": r_max,
                "target_points": target_points,
            },
        )

        if len(detectors) >= max_detectors:
            break

    if not detectors:
        raise RuntimeError("не удалось построить ни одного детектора для активации")
    return detectors


def _build_activation_engine(layout: Layout, energies: list[float]) -> ActivationEngine:
    detectors = _build_detectors_from_layout(
        layout,
        energies,
        radius=ACTIVATION_DETECTOR_RADIUS,
        max_detectors=ACTIVATION_MAX_DETECTORS,
        output_bits=ACTIVATION_OUTPUT_BITS,
        lambda_levels=ACTIVATION_LAMBDA_LEVELS,
    )
    eta_value = ACTIVATION_ETA
    return ActivationEngine(
        layout=layout,
        detectors=detectors,
        lambda_threshold=ACTIVATION_LAMBDA_THRESHOLD,
        eta=eta_value,
        energy_threshold=ACTIVATION_ENERGY_THRESHOLD,
        detector_threshold=ACTIVATION_DETECTOR_THRESHOLD,
        saturation_limit=ACTIVATION_SATURATION_LIMIT,
        energy_map=energies,
        use_gpu=ACTIVATION_USE_GPU,
    )


def _load_layout_from_file(path: str, run_params: TunedLayoutParams) -> Layout | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    points_payload = data.get("layout", [])
    width = data.get("width")
    height = data.get("height")
    if width in (None, 0) or height in (None, 0):
        LOGGER.event(
            "layout.cache.incomplete",
            section=LAID_OUT_STRUCTURE,
            data={"path": path, "reason": "missing_size"},
        )
        return None
    pair_radius_saved = data.get("pair_radius")
    total_points = int(data.get("points") or len(points_payload))
    if total_points <= 0:
        LOGGER.event(
            "layout.cache.incomplete",
            section=LAID_OUT_STRUCTURE,
            data={"path": path, "reason": "empty"},
        )
        return None
    expected_pair_radius = _expected_pair_radius(total_points, run_params)
    if pair_radius_saved is not None and pair_radius_saved != expected_pair_radius:
        return None
    if not points_payload:
        return None
    if any(item.get("x") is None or item.get("y") is None for item in points_payload):
        LOGGER.event(
            "layout.cache.incomplete",
            section=LAID_OUT_STRUCTURE,
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
        empty_ratio=LAYOUT_EMPTY_RATIO,
        similarity=data.get("similarity", LAYOUT_SIMILARITY),
        lambda_threshold=data.get("lambda_threshold", LAYOUT_LAMBDA_THRESHOLD),
        eta=data.get("eta", LAYOUT_ETA),
        seed=0,
        use_gpu=LAYOUT_USE_GPU,
        long_subset_size=None,
        long_subset_refresh=LAYOUT_LONG_SUBSET_REFRESH,
    )
    if len(layout._positions) != len(sorted_points):
        raise RuntimeError("layout load mismatch: points count differs")
    layout.grid = [[None for _ in range(layout.width)] for _ in range(layout.height)]
    layout._row_occupied = [set() for _ in range(layout.height)]
    positions: list[tuple[int, int]] = [(-1, -1)] * len(sorted_points)
    for item in sorted_points:
        idx = int(item["index"])
        y = int(item["y"])
        x = int(item["x"])
        positions[idx] = (y, x)
        layout.grid[y][x] = idx
        layout._row_occupied[y].add(x)
    layout._positions = positions
    layout._pos_y = [p[0] for p in positions]
    layout._pos_x = [p[1] for p in positions]
    layout.pair_radius = pair_radius_saved
    LOGGER.event(
        "layout.load.cache",
        section=LAID_OUT_STRUCTURE,
        data={
            "path": path,
            "points": len(positions),
            "pair_radius": pair_radius_saved,
            "width": width,
            "height": height,
        },
    )
    return layout


def _build_encoder() -> Encoder:
    return Encoder(
        # Angle
        ClosedDimension(
            "Angle",
            (0.0, 360.0),
            [
                #Detectors(360, 0.4),
                #Detectors(180, 0.4),
                Detectors(90, 0.7),
                Detectors(45, 0.7),
                Detectors(30, 0.7),
                Detectors(10, 0.7),
                #Detectors(5, 0.4),
            ],
        ),
        # X
        OpenedDimension(
            "X",
            (0, 6),
            [
                Detectors(7, 0.7),
                Detectors(6, 0.7),
                Detectors(5, 0.7),
                Detectors(4, 0.7),
                Detectors(3, 0.7),
                Detectors(2, 0.7),
            ],
        ),
        # Y
        OpenedDimension(
            "Y",
            (0, 6),
            [
                Detectors(7, 0.7),
                Detectors(6, 0.7),
                Detectors(5, 0.7),
                Detectors(4, 0.7),
                Detectors(3, 0.7),
                Detectors(2, 0.7),
            ],
        ),
        log_every=ENCODER_LOG_EVERY,
    )


def _collect_codes(
    dataset,
    label: int | None,
    count: int,
    encoder: Encoder,
    extractor: MnistSobelAngleMap,
) -> tuple[LayoutPayloadBuilder, int]:
    payload = LayoutPayloadBuilder(
        similarity=None,
        lambda_threshold=None,
        eta=None,
    )
    total_codes = 0
    found = 0

    for img_tensor, digit in dataset:
        if label is not None and int(digit) != label:
            continue
        found += 1
        if found > count:
            break
        img = img_tensor.squeeze(0).numpy()
        digit_values = extractor.extract(img, digit)
        measurements = digit_values[int(digit)]
        first = True
        for (a, x, y) in measurements:
            log_image = img if first else None
            log_label = int(digit) if first else None
            log_measurements = measurements if first else None
            _, code = encoder.encode(
                float(a),
                float(x),
                float(y),
                log_image=log_image,
                log_label=log_label,
                log_measurements=log_measurements,
            )
            payload.add_code(label=int(digit), hue=float(a), code=code)
            total_codes += 1
            first = False
    return payload, total_codes


def _collect_samples(
    dataset,
    sample_count: int,
    encoder: Encoder,
    extractor: MnistSobelAngleMap,
    *,
    label: int | None,
) -> list[tuple[int, list[BitArray]]]:
    samples: list[tuple[int, list[BitArray]]] = []
    for img_tensor, digit in dataset:
        if label is not None and int(digit) != label:
            continue
        img = img_tensor.squeeze(0).numpy()
        digit_values = extractor.extract(img, digit)
        measurements = digit_values[int(digit)]
        codes: list[BitArray] = []
        first = True
        for (a, x, y) in measurements:
            log_image = img if first else None
            log_label = int(digit) if first else None
            log_measurements = measurements if first else None
            _, code = encoder.encode(
                float(a),
                float(x),
                float(y),
                log_image=log_image,
                log_label=log_label,
                log_measurements=log_measurements,
            )
            codes.append(code)
            first = False
        if codes:
            samples.append((int(digit), codes))
        if len(samples) >= sample_count:
            break
    return samples

SPR = 32
def _run_layout(
    codes: dict[float, list],
    params: TunedLayoutParams | None,
    *,
    count: int,
    total_codes: int,
) -> Layout:
    run_params = params or _default_layout_params()
    expected_pair_radius = _expected_pair_radius(total_codes, run_params)
    layout_path = _layout_json_path(count, expected_pair_radius)
    cached_layout = _load_layout_from_file(layout_path, run_params)
    if cached_layout is not None:
        return cached_layout
    layout = Layout(
        codes,
        empty_ratio=LAYOUT_EMPTY_RATIO,
        similarity=LAYOUT_SIMILARITY,
        lambda_threshold=LAYOUT_LAMBDA_THRESHOLD,
        eta=LAYOUT_ETA,
        seed=0,
        use_gpu=LAYOUT_USE_GPU,
        long_subset_size=run_params.long_subset_size,
        long_subset_refresh=run_params.long_subset_refresh,
    )
    step_offset = 1
    long_radius_start = max(
        LAYOUT_ADAPTIVE_RADIUS_MIN,
        int(layout.width * LAYOUT_ADAPTIVE_RADIUS_START_FACTOR),
    )
    adaptive_long = AdaptiveLayoutConfig(
        start_radius=long_radius_start,
        end_radius=LAYOUT_ADAPTIVE_RADIUS_MIN,
        swap_ratio_trigger=LAYOUT_ADAPTIVE_SWAP_TRIGGER,
        lambda_step=LAYOUT_ADAPTIVE_LAMBDA_STEP,
    )
    layout.run(
        steps=run_params.long_steps,
        pairs_per_step=run_params.long_pairs_per_step,
        pair_radius=adaptive_long.start_radius,
        mode="long",
        min_swap_ratio=LAYOUT_MIN_SWAP_RATIO,
        min_swap_window=LAYOUT_MIN_SWAP_WINDOW,
        log_every=LAYOUT_LOG_EVERY_LONG,
        step_offset=step_offset,
        energy_radius=None,
        energy_stability_window=LAYOUT_ENERGY_STABILITY_WINDOW,
        energy_stability_delta=LAYOUT_ENERGY_STABILITY_DELTA,
        energy_stability_every=LAYOUT_ENERGY_STABILITY_EVERY,
        energy_stability_max_points=LAYOUT_ENERGY_STABILITY_MAX_POINTS,
        log_visuals=LAYOUT_LOG_VISUALS,
        adaptive_params=adaptive_long,
        save_on_stop_path=layout_path,
    )
    step_offset += layout.last_steps
    short_radius_max = int(layout.width * LAYOUT_ADAPTIVE_RADIUS_START_FACTOR)
    short_radius_base = int(
        max(
            1,
            round(run_params.short_local_radius * LAYOUT_SHORT_ADAPTIVE_RADIUS_FACTOR),
        )
    )
    short_radius_start = max(
        LAYOUT_ADAPTIVE_RADIUS_MIN,
        min(short_radius_base, short_radius_max),
    )
    adaptive_short = AdaptiveLayoutConfig(
        start_radius=short_radius_start,
        end_radius=LAYOUT_ADAPTIVE_RADIUS_MIN,
        swap_ratio_trigger=LAYOUT_ADAPTIVE_SWAP_TRIGGER,
        lambda_step=LAYOUT_ADAPTIVE_LAMBDA_STEP,
    )
    short_optim = ShortLayoutOptimConfig(
        energy_radius=run_params.short_energy_radius,
        energy_max_points=LAYOUT_SHORT_ENERGY_MAX_POINTS,
        energy_recalc_every=run_params.short_energy_recalc_every,
        energy_eps=LAYOUT_SHORT_ENERGY_EPS,
        use_weighted_first_point=LAYOUT_SHORT_WEIGHTED_FIRST,
        similarity_cutoff=LAYOUT_SHORT_SIMILARITY_CUTOFF,
        partitions=LAYOUT_SHORT_PARTITIONS,
    )
    layout.run(
        steps=150,
        pairs_per_step=run_params.short_pairs_per_step,
        pair_radius=short_radius_start,
        mode="short",
        local_radius=run_params.short_local_radius,
        min_swap_ratio=LAYOUT_MIN_SWAP_RATIO,
        min_swap_window=LAYOUT_MIN_SWAP_WINDOW,
        log_every=LAYOUT_LOG_EVERY_SHORT,
        step_offset=step_offset,
        adaptive_params=adaptive_short,
        short_optim=short_optim,
    )
    return layout


class MnistLayoutRunner:
    def __init__(
        self,
        *,
        count: int,
        label: int | None = None,
        activation_samples: int = ACTIVATION_SAMPLE_IMAGES,
    ) -> None:
        self.count = count
        self.label = label
        if activation_samples <= 0:
            raise ValueError("activation_samples must be positive")
        self._activation_samples = int(activation_samples)
        configure_logging()
        GPU_ENVIRONMENT.apply()
        self.encoder = _build_encoder()
        self.extractor = MnistSobelAngleMap(angle_in_degrees=True, grad_threshold=0.05)
        self._embedding_store = EmbeddingStore(json_key=EMBEDDINGS_JSON_KEY)

    def _layout_cache_paths(self) -> list[str]:
        return _layout_cache_paths_for_count(self.count)

    def _load_layout_metadata(self, path: str) -> tuple[int, int | None, bool] | None:
        return _load_layout_metadata_from_path(path)

    def _try_load_cached_layout(self) -> tuple[Layout, TunedLayoutParams, int, str] | None:
        for path in self._layout_cache_paths():
            metadata = self._load_layout_metadata(path)
            if metadata is None:
                continue
            total_points, pair_radius, ready = metadata
            if not ready:
                LOGGER.event(
                    "layout.cache.incomplete",
                    section=LAID_OUT_STRUCTURE,
                    data={
                        "path": path,
                        "points": total_points,
                        "pair_radius": pair_radius,
                    },
                )
                continue
            tuned_params = self._tune_layout(total_points)
            layout = _load_layout_from_file(path, tuned_params)
            if layout is None:
                LOGGER.event(
                    "layout.cache.mismatch",
                    section=LAYOUT_PARAMETERS,
                    data={
                        "path": path,
                        "pair_radius": pair_radius,
                        "points": total_points,
                    },
                )
                continue
            LOGGER.event(
                "layout.cache.hit",
                section=LAID_OUT_STRUCTURE,
                data={
                    "path": path,
                    "points": total_points,
                    "pair_radius": pair_radius,
                },
            )
            return layout, tuned_params, total_points, path
        LOGGER.event(
            "layout.cache.miss",
            section=LAYOUT_PARAMETERS,
            data={"count": self.count, "label": self.label},
        )
        return None

    def _build_dataset(self):
        from torchvision import transforms
        from torchvision.datasets import MNIST

        return MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )

    def _tune_layout(self, total_codes: int) -> TunedLayoutParams:
        tuner = LayoutParameterTuner(
            empty_ratio=LAYOUT_EMPTY_RATIO,
            start_radius_factor=LAYOUT_ADAPTIVE_RADIUS_START_FACTOR,
        )
        return tuner.tune(total_codes)

    def _log_dataset(self, angles: int, total_codes: int) -> None:
        LOGGER.event(
            "mnist.codes",
            section=LAYOUT_SYSTEM,
            data={
                "count": self.count,
                "label": "all" if self.label is None else int(self.label),
                "angles": angles,
                "points": total_codes,
            },
        )

    def _export_layout(
        self,
        layout: Layout,
        run_params: TunedLayoutParams,
        *,
        layout_path: str | None = None,
        save_json: bool = True,
    ) -> None:
        from PIL import Image

        resolved_layout_path = layout_path or _layout_json_path(self.count, layout.pair_radius)
        if save_json:
            layout.save_json(resolved_layout_path)
        visual_layout = _load_layout_from_file(resolved_layout_path, run_params)
        if visual_layout is None:
            LOGGER.event(
                "layout.export.json.error",
                section=LAID_OUT_STRUCTURE,
                data={"path": resolved_layout_path, "reason": "load_failed"},
            )
            raise RuntimeError("не удалось загрузить раскладку из json для визуализации")
        image = visual_layout.render_image()
        suffix = "auto" if visual_layout.pair_radius is None else str(visual_layout.pair_radius)
        filename = f"data/{self.count}-{suffix}.png"
        Image.fromarray(image).save(filename)
        LOGGER.event(
            "layout.visual.from_json",
            section=LAID_OUT_STRUCTURE,
            data={
                "path": resolved_layout_path,
                "image": filename,
                "points": len(visual_layout.positions()),
            },
        )

    def _run_activation(self, layout: Layout, dataset) -> list[EmbeddingRecord]:
        energies = _compute_point_energies(layout, ACTIVATION_POINT_ENERGY_RADIUS)
        engine = _build_activation_engine(layout, energies)
        samples = _collect_samples(
            dataset,
            self._activation_samples,
            self.encoder,
            self.extractor,
            label=self.label,
        )
        embeddings: list[EmbeddingRecord] = []
        for idx, (label, codes) in enumerate(samples):
            result = engine.activate(codes)
            LOGGER.event(
                "activation.sample",
                section=STIMULUS_DETECTION,
                data={
                    "index": idx,
                    "sample_label": str(int(label)),
                    "stimuli": len(codes),
                    "active_detectors": len(result.active_detectors),
                    "embedding_bits": result.embedding.count(),
                },
            )
            embeddings.append(
                EmbeddingRecord(
                    class_label=str(int(label)),
                    embedding_bits=result.embedding.to01(),
                )
            )
        return embeddings

    def _save_embeddings(self, layout_path: str, embeddings: Sequence[EmbeddingRecord]) -> None:
        if not layout_path:
            raise ValueError("layout_path must be provided to save embeddings")
        self._embedding_store.save(layout_path, embeddings)

    def run(self) -> None:
        cached_layout = self._try_load_cached_layout()
        dataset = self._build_dataset()
        if cached_layout is not None:
            layout, tuned_params, total_codes, layout_path = cached_layout
            angles = len({float(hue) for hue in layout.hues()})
            self._log_dataset(angles, total_codes)
            LOGGER.event(
                "mnist.encoding.skip",
                section=ENCODING_SYSTEM,
                data={
                    "count": self.count,
                    "label": "all" if self.label is None else int(self.label),
                    "layout_path": layout_path,
                    "points": total_codes,
                    "angles": angles,
                },
            )
            save_json = False
        else:
            payload, total_codes = _collect_codes(
                dataset,
                self.label,
                self.count,
                self.encoder,
                self.extractor,
            )
            codes = payload.codes_by_hue()
            angles = len(codes)
            self._log_dataset(angles, total_codes)
            tuned_params = self._tune_layout(total_codes)
            expected_pair_radius = _expected_pair_radius(total_codes, tuned_params)
            layout_path = _layout_json_path(self.count, expected_pair_radius)
            payload.save_base(layout_path)
            layout = _run_layout(
                codes,
                tuned_params,
                count=self.count,
                total_codes=total_codes,
            )
            save_json = True
        try:
            cached_embeddings = _load_embeddings_from_file(layout_path)
        except FileNotFoundError:
            cached_embeddings = []
        if cached_embeddings:
            embeddings = cached_embeddings
            LOGGER.event(
                "activation.embedding.skip",
                section=DETECTOR_INSERTION,
                data={
                    "path": layout_path,
                    "embeddings": len(cached_embeddings),
                    "reason": "cache_present",
                },
            )
        else:
            embeddings = self._run_activation(layout, dataset)
            self._save_embeddings(layout_path, embeddings)
        self._export_layout(layout, tuned_params, layout_path=layout_path, save_json=save_json)


def _load_image_from_path(path: str) -> "np.ndarray":
    from PIL import Image
    import numpy as np

    target_h, target_w = MnistSobelAngleMap.EXPECTED_SHAPE
    image = Image.open(path).convert("L")
    image = image.resize((target_w, target_h))
    array = np.asarray(image, dtype="float32")
    if array.max() > 1.5:
        array = array / 255.0
    LOGGER.event(
        "decoder.image.load",
        section=STIMULUS_DETECTION,
        data={
            "path": path,
            "shape": array.shape,
            "max": float(array.max()) if array.size else 0.0,
            "min": float(array.min()) if array.size else 0.0,
        },
    )
    return array


class MnistDecodeRunner:
    def __init__(self, *, count: int, layout_path: str | None = None) -> None:
        self.count = count
        self.layout_path = layout_path
        self._decoder: MnistDecoder | None = None
        self._temperature_override: float | None = None
        configure_logging()
        GPU_ENVIRONMENT.apply()
        self.encoder = _build_encoder()
        self.extractor = MnistSobelAngleMap(angle_in_degrees=True, grad_threshold=0.05)
        self._embedding_store = EmbeddingStore(json_key=EMBEDDINGS_JSON_KEY)

    def _load_layout(self) -> tuple[Layout, str, int]:
        candidates: list[str] = []
        if self.layout_path:
            candidates.append(self.layout_path)
        candidates.extend(_layout_cache_paths_for_count(self.count))
        seen: set[str] = set()
        for path in candidates:
            if not path or path in seen or not os.path.exists(path):
                continue
            seen.add(path)
            metadata = _load_layout_metadata_from_path(path)
            if metadata is None:
                continue
            total_points, pair_radius, ready = metadata
            if not ready:
                continue
            tuner = LayoutParameterTuner(
                empty_ratio=LAYOUT_EMPTY_RATIO,
                start_radius_factor=LAYOUT_ADAPTIVE_RADIUS_START_FACTOR,
            )
            tuned_params = tuner.tune(total_points)
            layout = _load_layout_from_file(path, tuned_params)
            if layout is None:
                continue
            LOGGER.event(
                "decoder.layout.use",
                section=LAID_OUT_STRUCTURE,
                data={
                    "path": path,
                    "points": total_points,
                    "pair_radius": pair_radius,
                },
            )
            self.layout_path = path
            return layout, path, total_points
        raise RuntimeError("готовая раскладка не найдена; запустите layout")

    def _ensure_decoder(self) -> MnistDecoder:
        if self._decoder is not None:
            return self._decoder
        layout, layout_path, total_points = self._load_layout()
        energies = _compute_point_energies(layout, ACTIVATION_POINT_ENERGY_RADIUS)
        embeddings = _load_embeddings_from_file(layout_path)
        if not embeddings:
            embeddings = self._build_embeddings_for_layout(
                layout,
                layout_path,
                energies,
            )
        labeled_embeddings = _embedding_records_to_labeled(embeddings)
        if not labeled_embeddings:
            raise RuntimeError("в раскладке нет базы эмбеддингов")
        activation_engine = _build_activation_engine(layout, energies)
        database = EmbeddingDatabase(
            labeled_embeddings,
            DecoderConfig(
                similarity=DECODER_SIMILARITY,
                temperature=DECODER_SOFTMAX_TEMPERATURE,
                min_confidence=DECODER_MIN_CONFIDENCE,
                expected_classes=DECODER_EXPECTED_CLASSES,
                knn_top_k=200,
                top_k_per_class=20,
                min_similarity=0.0,
            ),
        )
        self._decoder = MnistDecoder(
            encoder=self.encoder,
            extractor=self.extractor,
            activation_engine=activation_engine,
            embedding_db=database,
        )
        LOGGER.event(
            "decoder.build",
            section=STIMULUS_DETECTION,
            data={
                "layout_path": layout_path,
                "embeddings": len(labeled_embeddings),
                "points": total_points,
                "temperature": self._decoder.base_temperature,
            },
        )
        return self._decoder

    def _build_embeddings_for_layout(
        self,
        layout: Layout,
        layout_path: str,
        energies: list[float],
    ) -> list[EmbeddingRecord]:
        engine = _build_activation_engine(layout, energies)
        dataset = self._build_dataset(train=True)
        samples = _collect_samples(
            dataset,
            DECODER_DATASET_LIMIT,
            self.encoder,
            self.extractor,
            label=None,
        )
        embeddings: list[EmbeddingRecord] = []
        for idx, (label, codes) in enumerate(samples):
            result = engine.activate(codes)
            embeddings.append(
                EmbeddingRecord(
                    class_label=str(int(label)),
                    embedding_bits=result.embedding.to01(),
                )
            )
            LOGGER.event(
                "decoder.base.generate",
                section=DETECTOR_INSERTION,
                data={
                    "index": idx,
                    "label": int(label),
                    "stimuli": len(codes),
                    "bits": result.embedding.count(),
                },
            )
        self._embedding_store.save(layout_path, embeddings)
        LOGGER.event(
            "decoder.base.generated",
            section=DETECTOR_INSERTION,
            data={
                "embeddings": len(embeddings),
                "classes": len({int(item.class_label) for item in embeddings}),
                "path": layout_path,
            },
        )
        return embeddings

    def predict_image(
        self,
        image: "np.ndarray",
        *,
        temperature: float | None = None,
        log_image: bool = False,
    ) -> PredictionResult:
        decoder = self._ensure_decoder()
        temp_value = temperature
        if temp_value is None:
            temp_value = self._temperature_override
        return decoder.predict_image(
            image,
            label_hint=None,
            log_image=log_image,
            temperature=temp_value,
        )

    def predict_image_from_path(
        self,
        path: str,
        *,
        temperature: float | None = None,
        log_image: bool = False,
    ) -> PredictionResult:
        image = _load_image_from_path(path)
        return self.predict_image(image, temperature=temperature, log_image=log_image)

    def _build_dataset(self, *, train: bool):
        from torchvision import transforms
        from torchvision.datasets import MNIST

        return MNIST(
            root="./data",
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )

    def predict_dataset(
        self,
        *,
        split: str,
        limit: int,
        temperature: float | None = None,
        log_first: bool = False,
    ) -> tuple[list[PredictionResult], float]:
        decoder = self._ensure_decoder()
        dataset = self._build_dataset(train=split == "train")
        images: list["np.ndarray"] = []
        labels: list[int] = []
        for idx, (img_tensor, label) in enumerate(dataset):
            if idx >= limit:
                break
            images.append(img_tensor.squeeze(0).numpy())
            labels.append(int(label))
        results, accuracy = decoder.predict_batch(
            images,
            labels=labels,
            temperature=temperature if temperature is not None else self._temperature_override,
            log_first=log_first,
        )
        LOGGER.event(
            "decoder.dataset.eval",
            section=STIMULUS_DETECTION,
            data={
                "split": split,
                "samples": len(results),
                "accuracy": accuracy,
                "temperature": temperature if temperature is not None else decoder.base_temperature,
            },
        )
        return results, 0.0 if accuracy is None else accuracy

    def calibrate_temperature(self, *, split: str, limit: int) -> tuple[float, float]:
        decoder = self._ensure_decoder()
        dataset = self._build_dataset(train=split == "train")
        images: list["np.ndarray"] = []
        labels: list[int] = []
        for idx, (img_tensor, label) in enumerate(dataset):
            if idx >= limit:
                break
            images.append(img_tensor.squeeze(0).numpy())
            labels.append(int(label))
        best_temp = decoder.base_temperature
        best_acc = -1.0
        if not images:
            self._temperature_override = best_temp
            LOGGER.event(
                "decoder.calibrate",
                section=SIMILARITY_MEASURES,
                data={
                    "split": split,
                    "samples": 0,
                    "best_temperature": best_temp,
                    "accuracy": 0.0,
                    "candidates": len(DECODER_CALIBRATION_TEMPERATURES),
                    "reason": "empty_dataset",
                },
            )
            return best_temp, 0.0
        for candidate in DECODER_CALIBRATION_TEMPERATURES:
            _, acc = decoder.predict_batch(
                images,
                labels=labels,
                temperature=candidate,
                log_first=False,
            )
            if acc is None:
                continue
            if acc > best_acc:
                best_acc = acc
                best_temp = candidate
        self._temperature_override = best_temp
        LOGGER.event(
            "decoder.calibrate",
            section=SIMILARITY_MEASURES,
            data={
                "split": split,
                "samples": len(images),
                "best_temperature": best_temp,
                "accuracy": best_acc,
                "candidates": len(DECODER_CALIBRATION_TEMPERATURES),
            },
        )
        return best_temp, best_acc


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DAMP MNIST demo")
    subparsers = parser.add_subparsers(dest="command")
    layout_parser = subparsers.add_parser("layout", help="построение раскладки и базы эмбеддингов")
    layout_parser.add_argument("--count", type=int, default=DECODER_DEFAULT_COUNT)
    layout_parser.add_argument("--label", type=int, default=None)
    layout_parser.add_argument(
        "--activation-samples",
        type=int,
        default=ACTIVATION_SAMPLE_IMAGES,
        help="количество изображений для базы эмбеддингов активации",
    )
    predict_parser = subparsers.add_parser("predict", help="декодирование картинки или батча")
    predict_parser.add_argument("--layout-path", type=str, default=None)
    predict_parser.add_argument("--count", type=int, default=DECODER_DEFAULT_COUNT)
    predict_parser.add_argument("--image", type=str, default=None)
    predict_parser.add_argument("--dataset", choices=["train", "test"], default=None)
    predict_parser.add_argument("--limit", type=int, default=DECODER_DATASET_LIMIT)
    predict_parser.add_argument("--temperature", type=float, default=None)
    predict_parser.add_argument("--calibrate", action="store_true")
    evaluate_parser = subparsers.add_parser("evaluate", help="оценка на MNIST")
    evaluate_parser.add_argument("--layout-path", type=str, default=None)
    evaluate_parser.add_argument("--count", type=int, default=DECODER_DEFAULT_COUNT)
    evaluate_parser.add_argument("--limit", type=int, default=DECODER_DATASET_LIMIT)
    evaluate_parser.add_argument("--split", choices=["train", "test"], default="test")
    evaluate_parser.add_argument("--temperature", type=float, default=None)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    command = args.command or "layout"
    if command == "layout":
        runner = MnistLayoutRunner(
            count=args.count,
            label=args.label,
            activation_samples=args.activation_samples,
        )
        runner.run()
        return
    if command == "predict":
        decode_runner = MnistDecodeRunner(count=args.count, layout_path=args.layout_path)
        temperature = args.temperature
        if args.calibrate:
            best_temp, best_acc = decode_runner.calibrate_temperature(
                split=args.dataset or "test",
                limit=args.limit,
            )
            temperature = best_temp
            print(f"calibration: temp={best_temp:.3f} acc={best_acc:.4f}")
        if args.image:
            result = decode_runner.predict_image_from_path(
                args.image,
                temperature=temperature,
                log_image=True,
            )
            print(
                f"image={args.image} pred={result.predicted_class} "
                f"conf={result.confidence:.4f} probs={result.probabilities}"
            )
            return
        split = args.dataset or "test"
        _, accuracy = decode_runner.predict_dataset(
            split=split,
            limit=args.limit,
            temperature=temperature,
            log_first=True,
        )
        print(f"dataset={split} samples={args.limit} accuracy={accuracy:.4f}")
        return
    if command == "evaluate":
        decode_runner = MnistDecodeRunner(count=args.count, layout_path=args.layout_path)
        _, accuracy = decode_runner.predict_dataset(
            split=args.split,
            limit=args.limit,
            temperature=args.temperature,
            log_first=False,
        )
        print(f"dataset={args.split} samples={args.limit} accuracy={accuracy:.4f}")
        return
    parser.error(f"unknown command {command}")


if __name__ == "__main__":
    main()