import glob
import math
import time
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
    PARALLEL_PROCESSING,
)
from damp.activation import (
    ActivationEngine,
    Detector,
    DetectorHierarchyBuilder as ArticleDetectorHierarchyBuilder,
    DetectorHierarchyScale,
    DetectorHierarchyScaleResolver,
    DetectorLayerConfigFactory,
    EmbeddingSaturationResolver,
)
from damp.MnistSobelAngleMap import MnistSobelAngleMap
from damp.encoding.damp_encoder import (
    EncoderCodeColorizer,
)
from damp.encoding.orientation_encoder import (
    RetinotopicOrientationEncoder,
    RetinotopicOrientationEncoderFactory,
)
from damp.encoding.stimulus_embedding import StimulusEmbeddingBuilder
from damp.layout.damp_layout import (
    AdaptiveLayoutConfig,
    ContinuityLayoutConfig,
    Layout,
    ShortLayoutOptimConfig,
)
from damp.layout.payload import LayoutEncodedCode, LayoutPayloadBuilder
from damp.encoding.bitarray import BitArray
from damp.logging import LOGGER
from damp.decode import (
    DecoderConfig,
    DecoderParameterScaleResolver,
    EmbeddingDatabase,
    LabeledEmbedding,
    MnistDecoder,
    PredictionResult,
)

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
LOG_INTERVAL_LAYOUT_ADAPTIVE = 5
LOG_INTERVAL_LAYOUT_SWAP_RATIO = 5
LOG_INTERVAL_LAYOUT_EXPORT = 1
LOG_INTERVAL_LAYOUT_ENERGY_PROGRESS = 1
LOG_INTERVAL_ACTIVATION_INIT = 1
LOG_INTERVAL_ACTIVATION_MAP = 100
LOG_INTERVAL_ACTIVATION_DETECTOR_LEVEL = 10000
LOG_INTERVAL_ACTIVATION_DETECTOR_ACTIVE = 10000
LOG_INTERVAL_ACTIVATION_EMBEDDING = 100
LOG_INTERVAL_ACTIVATION_COLOR_MERGE = 100
LOG_INTERVAL_ACTIVATION_COVERAGE = 100
LOG_INTERVAL_ACTIVATION_ENERGY_MAP = 1
LOG_INTERVAL_ACTIVATION_DETECTOR_LAYER = 1
LOG_INTERVAL_ACTIVATION_DETECTOR_CLUSTER = 100
LOG_INTERVAL_ACTIVATION_DETECTOR_CENTER = 100
LOG_INTERVAL_ACTIVATION_SAMPLE = 100
LOG_INTERVAL_ACTIVATION_VISUAL = 100
LOG_INTERVAL_ACTIVATION_EMBEDDING_SAVE = 1
LOG_INTERVAL_EVALUATION_PROGRESS = 10

LAYOUT_COMPLETION_EVENT = "pipeline.layout.completed"
LAYOUT_COMPLETION_TITLE = "================ РАСКЛАДКА ЗАВЕРШЕНА ================"
EVALUATION_COMPLETION_EVENT = "pipeline.evaluation.completed"
EVALUATION_COMPLETION_TITLE = "================ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО ================"
LAYOUT_LOG_PATH = "layout.log"
EVALUATION_LOG_PATH = "evaluate.log"

EMBEDDINGS_JSON_KEY = "embeddings"
EMBEDDINGS_VERSION_KEY = "embeddings_version"
EMBEDDINGS_VERSION = 14
PIPELINE_CHECKPOINT_SUFFIX = ".checkpoint.json"
PIPELINE_CHECKPOINT_VERSION = 13
PIPELINE_CHECKPOINT_REPLACE_ATTEMPTS = 8
PIPELINE_CHECKPOINT_REPLACE_DELAY = 0.1
LAYOUT_CACHE_TAG = "orientation-topology-v3"
LAYOUT_LONG_PHASE_SUFFIX = ".long-phase.json"
LAYOUT_SHORT_PHASE_SUFFIX = ".short-phase.json"
LAYOUT_CONTINUITY_PHASE_SUFFIX = ".continuity-phase.json"

LAYOUT_SIMILARITY = "cosine"
LAYOUT_LAMBDA_THRESHOLD = 0.65
LAYOUT_ETA = 10.0
LAYOUT_EMPTY_RATIO = 0.5
LAYOUT_USE_GPU = True
LAYOUT_PSEUDO_COLOR_MIN_HUE = 0.0
LAYOUT_PSEUDO_COLOR_MAX_HUE = 300.0
LAYOUT_COLORIZER = EncoderCodeColorizer(
    min_hue=LAYOUT_PSEUDO_COLOR_MIN_HUE,
    max_hue=LAYOUT_PSEUDO_COLOR_MAX_HUE,
)

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
ENCODER_RANDOM_SEED = 0
MNIST_ANGLE_IN_DEGREES = True
MNIST_GRAD_THRESHOLD = 0.05
MNIST_LOCAL_ORIENTATION_LIMIT = 4
MNIST_STIMULUS_EMBEDDING_MAX_DENSITY = 0.25
MNIST_REFERENCE_LAYOUT_COUNT = 100
MNIST_ORIENTATION_MIN = 0.0
MNIST_ORIENTATION_MAX = 180.0
MNIST_ORIENTATION_COLOR_MAX_HUE = 360.0
MNIST_POSITION_MIN = 0.0
MNIST_POSITION_MAX = 6.0
MNIST_POSITION_GRID_SIZE = 7
MNIST_ORIENTATION_RADIUS_FACTOR = 0.5
MNIST_ORIENTATION_ISOLATED_RADIUS = 0.5
MNIST_ORIENTATION_SPATIAL_BOUNDS = ((-0.75, 6.75), (-0.75, 6.75))
MNIST_ORIENTATION_SPATIAL_JOINT_LAYERS = (
    ((7, 7), (0.75, 0.75)),
    ((14, 14), (0.70, 0.70)),
    ((28, 28), (0.60, 0.60)),
)
MNIST_ORIENTATION_ANGLE_LAYERS = ((180, 0.50),)
MNIST_ORIENTATION_SMOOTH_DELTA = 15.0
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
LAYOUT_ENERGY_STABILITY_EVERY = 25
LAYOUT_ENERGY_STABILITY_MAX_POINTS = 128
LAYOUT_MIN_SWAP_RATIO = 0.003
LAYOUT_MIN_SWAP_WINDOW = 1
LAYOUT_CONTINUITY_SWEEPS = 20
LAYOUT_CONTINUITY_DEFECT_POINTS = 256
LAYOUT_CONTINUITY_CANDIDATES = 256
LAYOUT_CONTINUITY_MIN_GAIN = 1e-5
LAYOUT_CONTINUITY_STALL_SWEEPS = 3

PINWHEEL_ANGLE_MIN = 0.0
PINWHEEL_ANGLE_MAX = 180.0
PINWHEEL_ANGLE_COUNT = 360
PINWHEEL_POSITION_MIN = 0.0
PINWHEEL_POSITION_MAX = 1.0
PINWHEEL_POSITION_COORDINATES = (0.25, 0.50, 0.75)
PINWHEEL_COUNT = len(PINWHEEL_POSITION_COORDINATES) ** 2
PINWHEEL_RADIUS_FACTOR = 0.50
PINWHEEL_ISOLATED_RADIUS = 0.125
PINWHEEL_SPATIAL_BOUND_MARGIN_FACTOR = 0.05
PINWHEEL_POSITION_GENERATOR_MARGIN = 0.12
PINWHEEL_POSITION_GENERATOR_CANDIDATES = 64
PINWHEEL_LONG_SPATIAL_JOINT_LAYERS = (
    ((3, 3), (0.80, 0.80)),
    ((6, 6), (0.80, 0.80)),
    ((12, 12), (0.75, 0.75)),
)
PINWHEEL_SHORT_SPATIAL_JOINT_LAYERS = (
    ((12, 12), (0.75, 0.75)),
    ((24, 24), (0.70, 0.70)),
    ((48, 48), (0.60, 0.60)),
)
PINWHEEL_LONG_ANGLE_LAYERS = ((12, 0.50),)
PINWHEEL_SHORT_ANGLE_LAYERS = ((360, 0.50),)
PINWHEEL_COLOR_MIN_HUE = 0.0
PINWHEEL_COLOR_MAX_HUE = 360.0
PINWHEEL_SMOOTH_DELTA = 15.0
PINWHEEL_MIN_UNIQUE_COLORS = 300
PINWHEEL_EMPTY_RATIO = 0.20
PINWHEEL_SIMILARITY = "cosine"
PINWHEEL_LAMBDA_THRESHOLD = 0.0
PINWHEEL_ADAPTIVE_LAMBDA_STEP = None
PINWHEEL_ETA = 0.0
PINWHEEL_SEED = 2
PINWHEEL_LONG_STEPS = 1000
PINWHEEL_LONG_PAIRS_PER_STEP = 2400
PINWHEEL_SHORT_STEPS = 600
PINWHEEL_SHORT_PAIRS_PER_STEP = 1200
PINWHEEL_SHORT_LOCAL_RADIUS_FACTOR = 0.06
PINWHEEL_SHORT_PARTITIONS = 1
PINWHEEL_MIN_SWAP_RATIO = 0.003
PINWHEEL_MIN_SWAP_WINDOW = 150
PINWHEEL_CONTINUITY_SWEEPS = 60
PINWHEEL_CONTINUITY_DEFECT_POINTS = 384
PINWHEEL_CONTINUITY_CANDIDATES = 512
PINWHEEL_OUTPUT_PATH = "pinwheel_test.png"
PINWHEEL_LAYOUT_PATH = "pinwheel_test.json"
PINWHEEL_LONG_LAYOUT_PATH = "pinwheel_test.long.checkpoint.json"
PINWHEEL_RESUME_LONG_CHECKPOINT = True

ACTIVATION_LAMBDA_THRESHOLD = 0.55
ACTIVATION_ETA = 10.0
ACTIVATION_ENERGY_THRESHOLD = 0.25
ACTIVATION_DETECTOR_THRESHOLD = 0.5
ACTIVATION_OUTPUT_BITS = 2048
ACTIVATION_MAX_EMBEDDING_DENSITY = 0.25
ACTIVATION_TARGET_ACTIVE_BITS = 125
ACTIVATION_SATURATION_LIMIT = min(
    ACTIVATION_TARGET_ACTIVE_BITS,
    int(ACTIVATION_OUTPUT_BITS * ACTIVATION_MAX_EMBEDDING_DENSITY),
)
ACTIVATION_POINT_ENERGY_RADIUS = 3
ACTIVATION_DETECTOR_RADIUS_MIN = 1
ACTIVATION_DETECTOR_RADIUS_MAX = 24
ACTIVATION_DETECTOR_REFERENCE_EXTENT = 51
ACTIVATION_DETECTOR_MIN_POINTS = 3
ACTIVATION_DETECTOR_ATTEMPTS_WITHOUT_CHANGE = 1000
ACTIVATION_DETECTOR_SEED = 0
ACTIVATION_DETECTOR_SHAPE = "circle"
ACTIVATION_DETECTOR_MAX_AXIS_RATIO = 3.0

ACTIVATION_MAX_DETECTORS = 256
ACTIVATION_DETECTOR_LAYER_SPECS = (
    (0.50, 0.35, 3.50),
    (0.60, 0.29, 3.20),
    (0.70, 0.24, 2.65),
    (0.75, 0.20, 2.40),
    (0.80, 0.16, 2.00),
    (0.85, 0.12, 1.30),
)
ACTIVATION_USE_GPU = True
ACTIVATION_PARALLEL_WORKERS = 4
ACTIVATION_BATCH_SIZE = 10
ACTIVATION_CHECKPOINT_EVERY = 10
ACTIVATION_STRUCTURAL_DIAGNOSTIC_EVERY = 100
ACTIVATION_STRUCTURAL_DIAGNOSTICS_REQUIRE_RERUN = True

DECODER_SIMILARITY = "jaccard"
DECODER_SOFTMAX_TEMPERATURE = 0.05
DECODER_MIN_CONFIDENCE = 0.05
DECODER_EXPECTED_CLASSES = tuple(range(10))
DECODER_KNN_TOP_K = 5
DECODER_REFERENCE_EMBEDDINGS = 100
DECODER_SOFTMAX_TEMPERATURE_MAX = 0.10
DECODER_TOP_K_PER_CLASS = 0
DECODER_MIN_SIMILARITY = 0.0
DECODER_DATASET_LIMIT = 64
DECODER_DEFAULT_COUNT = 1000
DECODER_CALIBRATION_TEMPERATURES = (0.01, 0.02, 0.03, 0.05, 0.1)
EVALUATION_CONFIDENCE_BINS = 10
EVALUATION_ERROR_EXAMPLES = 16

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
    "encoder.joint.code": LOG_INTERVAL_ENCODER_ENCODE,
    "encoder.joint.detectors": LOG_INTERVAL_ENCODER_ENCODE,
    "encoder.joint.input": LOG_INTERVAL_ENCODER_ENCODE,
    "encoder.joint.normalized": LOG_INTERVAL_ENCODER_ENCODE,
    "pinwheel.surface.orientation": LOG_INTERVAL_ENCODER_ENCODE,
    "pinwheel.encoder.code_union": LOG_INTERVAL_ENCODER_ENCODE,
    "pinwheel.encoder.code": LOG_INTERVAL_ENCODER_ENCODE,
    "pinwheel.topology.angle": LOG_INTERVAL_ENCODER_ENCODE,
    "pinwheel.topology.solve": LOG_INTERVAL_INIT,
    "orientation.encoder.init": LOG_INTERVAL_INIT,
    "orientation.encoder.code": LOG_INTERVAL_ENCODER_ENCODE,
    "orientation.encoder.input": LOG_INTERVAL_ENCODER_ENCODE,
    "encoder.code.color": LOG_INTERVAL_ENCODER_ENCODE,
    "encoder.code.color_component": LOG_INTERVAL_ENCODER_ENCODE,
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
    "layout.run.progress": LOG_INTERVAL_LAYOUT_SWAP_RATIO,
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
    "layout.payload.load": LOG_INTERVAL_INIT,
    "layout.payload.resume": LOG_INTERVAL_INIT,
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
    "activation.detector.layers.resolve": LOG_INTERVAL_INIT,
    "activation.detector.builder.init": LOG_INTERVAL_INIT,
    "activation.detector.cluster": LOG_INTERVAL_ACTIVATION_DETECTOR_CLUSTER,
    "activation.detector.center": LOG_INTERVAL_ACTIVATION_DETECTOR_CENTER,
    "activation.detector.ellipse": LOG_INTERVAL_ACTIVATION_DETECTOR_CENTER,
    "activation.embedding": LOG_INTERVAL_ACTIVATION_EMBEDDING,
    "activation.color_merge": LOG_INTERVAL_ACTIVATION_COLOR_MERGE,
    "activation.detector.coverage": LOG_INTERVAL_ACTIVATION_COVERAGE,
    "activation.energy.map": LOG_INTERVAL_ACTIVATION_ENERGY_MAP,
    "activation.sample": LOG_INTERVAL_ACTIVATION_SAMPLE,
    "activation.progress": LOG_INTERVAL_ACTIVATION_SAMPLE,
    "activation.map.visual": LOG_INTERVAL_ACTIVATION_VISUAL,
    "activation.energy.visual": LOG_INTERVAL_ACTIVATION_VISUAL,
    "activation.embedding.save": LOG_INTERVAL_ACTIVATION_EMBEDDING_SAVE,
    "decoder.evaluation.sample": LOG_INTERVAL_EVALUATION_PROGRESS,
    "decoder.evaluation.summary": LOG_INTERVAL_INIT,
    LAYOUT_COMPLETION_EVENT: LOG_INTERVAL_INIT,
    EVALUATION_COMPLETION_EVENT: LOG_INTERVAL_INIT,
}

RERUN_LAYOUT_EVENTS = {
    "mnist.codes",
    "mnist.encoding.skip",
    "layout.cache.hit",
    "layout.reference.clone",
    "layout.cache.miss",
    "layout.payload.resume",
    "layout.tune",
    "layout.init",
    "layout.grid",
    "layout.run.mode",
    "layout.run.settings",
    "layout.run.progress",
    "layout.energy.long.progress",
    "layout.adaptive.radius",
    "layout.run.swap_stability.stop",
    "layout.run.energy_stability.stop",
    "layout.run.done",
    "layout.continuity.start",
    "layout.continuity.sweep",
    "layout.continuity.done",
    "layout.visual",
    "layout.render_image",
    "layout.export.json",
    "activation.structure.visual",
    "activation.map.visual",
    "activation.detector.layer",
    "activation.detector.layers.resolve",
    "activation.engine.init",
    "activation.sample",
    "activation.progress",
    "activation.embedding.save",
    "encoder.stimulus.color_merge",
    "pipeline.checkpoint.save",
    "pipeline.checkpoint.replace.retry",
    LAYOUT_COMPLETION_EVENT,
}

RERUN_EVALUATE_EVENTS = {
    "decoder.layout.use",
    "decoder.build",
    "decoder.parameters.scale.resolve",
    "decoder.evaluation.sample",
    "decoder.evaluation.summary",
    "decoder.dataset.eval",
    "activation.structure.visual",
    "activation.map.visual",
    EVALUATION_COMPLETION_EVENT,
}

FILE_LAYOUT_EVENTS = {
    "mnist.codes",
    "layout.cache.hit",
    "layout.cache.miss",
    "layout.payload.resume",
    "layout.tune",
    "layout.run.settings",
    "layout.run.progress",
    "layout.energy.long.progress",
    "layout.run.swap_stability.stop",
    "layout.run.energy_stability.stop",
    "layout.run.done",
    "layout.continuity.start",
    "layout.continuity.sweep",
    "layout.continuity.done",
    "activation.detector.scale.resolve",
    "activation.detector.layers.resolve",
    "activation.embedding.saturation.resolve",
    "activation.detector.layer",
    "activation.engine.init",
    "encoder.stimulus.color_merge",
    "activation.embedding.save",
    "pipeline.checkpoint.save",
    "pipeline.checkpoint.replace.retry",
    LAYOUT_COMPLETION_EVENT,
}

FILE_EVALUATE_EVENTS = {
    "decoder.layout.use",
    "decoder.build",
    "decoder.parameters.scale.resolve",
    "activation.detector.scale.resolve",
    "activation.detector.layers.resolve",
    "activation.embedding.saturation.resolve",
    "activation.detector.layer",
    "activation.engine.init",
    "decoder.evaluation.sample",
    "decoder.evaluation.summary",
    "decoder.predict.batch",
    "decoder.dataset.eval",
    EVALUATION_COMPLETION_EVENT,
}

LAYOUT_CACHE_PATTERNS = (
    f"data/layout-{{count}}-*-{LAYOUT_CACHE_TAG}.json",
    f"layout-{{count}}-*-{LAYOUT_CACHE_TAG}.json",
)


def _layout_cache_paths_for_count(count: int) -> list[str]:
    paths: list[str] = []
    for pattern_template in LAYOUT_CACHE_PATTERNS:
        pattern = pattern_template.format(count=count)
        paths.extend(
            path
            for path in glob.glob(pattern)
            if not path.endswith(PIPELINE_CHECKPOINT_SUFFIX)
        )
    return sorted(set(paths), key=os.path.getmtime, reverse=True)


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


def configure_logging(*, profile: str | None = None) -> None:
    LOGGER.configure_intervals(LOG_INTERVALS, default_interval=LOG_INTERVAL_DEFAULT)
    rerun_events = None
    if profile == "layout":
        rerun_events = RERUN_LAYOUT_EVENTS
    elif profile == "evaluate":
        rerun_events = RERUN_EVALUATE_EVENTS
    LOGGER.configure_rerun(events=rerun_events, dashboard_profile=profile)
    active_log_path = None
    file_events = None
    if profile == "layout":
        active_log_path = LAYOUT_LOG_PATH
        file_events = FILE_LAYOUT_EVENTS
    elif profile == "evaluate":
        active_log_path = EVALUATION_LOG_PATH
        file_events = FILE_EVALUATE_EVENTS
    LOGGER.configure_file(
        all_paths=(LAYOUT_LOG_PATH, EVALUATION_LOG_PATH),
        active_path=active_log_path,
        events=file_events,
    )


def _structural_diagnostics_enabled() -> bool:
    if not ACTIVATION_STRUCTURAL_DIAGNOSTICS_REQUIRE_RERUN:
        return True
    return os.environ.get("RERUN", "off").strip().lower() == "on"


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
    def __init__(self, json_key: str, *, version_key: str, version: int) -> None:
        if not json_key:
            raise ValueError("json_key must be provided for embedding storage")
        self._json_key = json_key
        self._version_key = str(version_key)
        self._version = int(version)

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
        payload[self._version_key] = self._version
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
    stored_version = data.get(EMBEDDINGS_VERSION_KEY)
    if stored_version != EMBEDDINGS_VERSION:
        LOGGER.event(
            "activation.embedding.incompatible",
            section=DETECTOR_HIERARCHY,
            data={
                "path": path,
                "stored_version": stored_version,
                "expected_version": EMBEDDINGS_VERSION,
            },
        )
        return []
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


class PipelineCheckpointStore:
    """Атомарно сохраняет прогресс этапов и частичные эмбеддинги."""

    def __init__(
        self,
        *,
        suffix: str,
        version: int,
        replace_attempts: int,
        replace_delay: float,
    ) -> None:
        if replace_attempts <= 0 or replace_delay < 0.0:
            raise ValueError("invalid checkpoint replacement retry settings")
        self._suffix = str(suffix)
        self._version = int(version)
        self._replace_attempts = int(replace_attempts)
        self._replace_delay = float(replace_delay)

    def path_for(self, layout_path: str) -> str:
        return f"{layout_path}{self._suffix}"

    def load(self, layout_path: str) -> dict[str, object]:
        path = self.path_for(layout_path)
        if not os.path.exists(path):
            return self._empty_payload()
        try:
            with open(path, "r", encoding="utf-8") as fp:
                payload = json.load(fp)
        except Exception as exc:
            LOGGER.event(
                "pipeline.checkpoint.load.error",
                section=PARALLEL_PROCESSING,
                data={"path": path, "error": str(exc)},
            )
            return self._empty_payload()
        if payload.get("version") != self._version:
            LOGGER.event(
                "pipeline.checkpoint.incompatible",
                section=PARALLEL_PROCESSING,
                data={
                    "path": path,
                    "stored_version": payload.get("version"),
                    "expected_version": self._version,
                },
            )
            return self._empty_payload()
        LOGGER.event(
            "pipeline.checkpoint.load",
            section=PARALLEL_PROCESSING,
            data={
                "path": path,
                "completed": len(payload.get("embeddings", [])),
                "stages": payload.get("stages", {}),
            },
        )
        return payload

    def save(
        self,
        layout_path: str,
        *,
        stages: dict[str, bool],
        target_samples: int,
        embeddings: Sequence[EmbeddingRecord],
    ) -> None:
        path = self.path_for(layout_path)
        payload = {
            "version": self._version,
            "target_samples": int(target_samples),
            "stages": dict(stages),
            "embeddings": [record.as_payload() for record in embeddings],
        }
        temporary_path = f"{path}.tmp"
        with open(temporary_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=True, indent=2)
            fp.flush()
            os.fsync(fp.fileno())
        self._replace(temporary_path, path)
        LOGGER.event(
            "pipeline.checkpoint.save",
            section=PARALLEL_PROCESSING,
            data={
                "path": path,
                "completed": len(embeddings),
                "target": int(target_samples),
                "stages": stages,
            },
        )

    def _replace(self, temporary_path: str, path: str) -> None:
        for attempt in range(1, self._replace_attempts + 1):
            try:
                os.replace(temporary_path, path)
                return
            except PermissionError as exc:
                if attempt >= self._replace_attempts:
                    raise
                LOGGER.event(
                    "pipeline.checkpoint.replace.retry",
                    section=PARALLEL_PROCESSING,
                    data={
                        "path": path,
                        "attempt": attempt,
                        "max_attempts": self._replace_attempts,
                        "delay_seconds": self._replace_delay,
                        "error": str(exc),
                    },
                )
                time.sleep(self._replace_delay)

    def records(self, payload: dict[str, object]) -> list[EmbeddingRecord]:
        records: list[EmbeddingRecord] = []
        for item in payload.get("embeddings", []):
            if not isinstance(item, dict):
                continue
            class_label = item.get("class")
            embedding = item.get("embedding")
            if class_label is None or embedding is None:
                continue
            records.append(EmbeddingRecord(str(class_label), str(embedding)))
        return records

    def _empty_payload(self) -> dict[str, object]:
        return {"version": self._version, "target_samples": 0, "stages": {}, "embeddings": []}


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
    return f"data/layout-{count}-{suffix}-{LAYOUT_CACHE_TAG}.json"


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
                distance = math.sqrt(dist_sq)
                energy += sim / (distance + distance_eps)
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
    max_detectors: int,
    output_bits: int,
    hierarchy_scale: DetectorHierarchyScale | None = None,
) -> list[Detector]:
    if hierarchy_scale is None:
        hierarchy_scale = DetectorHierarchyScaleResolver(
            reference_extent=ACTIVATION_DETECTOR_REFERENCE_EXTENT,
            base_radius_max=ACTIVATION_DETECTOR_RADIUS_MAX,
            base_max_detectors=max_detectors,
        ).resolve(layout)
    layer_configs = DetectorLayerConfigFactory(
        radius_min=ACTIVATION_DETECTOR_RADIUS_MIN,
        radius_max=hierarchy_scale.radius_max,
        min_points=ACTIVATION_DETECTOR_MIN_POINTS,
    ).build(layout, ACTIVATION_DETECTOR_LAYER_SPECS)
    return ArticleDetectorHierarchyBuilder(
        layout=layout,
        energies=energies,
        radius_min=ACTIVATION_DETECTOR_RADIUS_MIN,
        max_detectors=hierarchy_scale.max_detectors,
        output_bits=output_bits,
        layer_configs=layer_configs,
        energy_threshold=ACTIVATION_ENERGY_THRESHOLD,
        eta=ACTIVATION_ETA,
        attempts_without_change=ACTIVATION_DETECTOR_ATTEMPTS_WITHOUT_CHANGE,
        seed=ACTIVATION_DETECTOR_SEED,
        receptive_field_shape=ACTIVATION_DETECTOR_SHAPE,
        max_axis_ratio=ACTIVATION_DETECTOR_MAX_AXIS_RATIO,
    ).build()


def _build_activation_engine(layout: Layout, energies: list[float]) -> ActivationEngine:
    hierarchy_scale = DetectorHierarchyScaleResolver(
        reference_extent=ACTIVATION_DETECTOR_REFERENCE_EXTENT,
        base_radius_max=ACTIVATION_DETECTOR_RADIUS_MAX,
        base_max_detectors=ACTIVATION_MAX_DETECTORS,
    ).resolve(layout)
    detectors = _build_detectors_from_layout(
        layout,
        energies,
        max_detectors=ACTIVATION_MAX_DETECTORS,
        output_bits=ACTIVATION_OUTPUT_BITS,
        hierarchy_scale=hierarchy_scale,
    )
    saturation_limit = EmbeddingSaturationResolver(
        output_bits=ACTIVATION_OUTPUT_BITS,
        max_density=ACTIVATION_MAX_EMBEDDING_DENSITY,
        base_active_bits=ACTIVATION_TARGET_ACTIVE_BITS,
    ).resolve(hierarchy_scale.factor)
    eta_value = ACTIVATION_ETA
    return ActivationEngine(
        layout=layout,
        detectors=detectors,
        lambda_threshold=ACTIVATION_LAMBDA_THRESHOLD,
        eta=eta_value,
        energy_threshold=ACTIVATION_ENERGY_THRESHOLD,
        detector_threshold=ACTIVATION_DETECTOR_THRESHOLD,
        saturation_limit=saturation_limit,
        energy_map=energies,
        use_gpu=ACTIVATION_USE_GPU,
    )


def _load_layout_from_file(
    path: str,
    run_params: TunedLayoutParams,
    *,
    colorizer: EncoderCodeColorizer = LAYOUT_COLORIZER,
    prepare_compute: bool = True,
) -> Layout | None:
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
        precompute_similarity=prepare_compute,
        use_gpu=LAYOUT_USE_GPU if prepare_compute else False,
        long_subset_size=None,
        long_subset_refresh=LAYOUT_LONG_SUBSET_REFRESH,
        colorizer=colorizer,
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


def _build_encoder() -> RetinotopicOrientationEncoder:
    centers = tuple(
        (float(x), float(y))
        for y in range(MNIST_POSITION_GRID_SIZE)
        for x in range(MNIST_POSITION_GRID_SIZE)
    )
    return RetinotopicOrientationEncoderFactory.create(
        centers,
        spatial_bounds=MNIST_ORIENTATION_SPATIAL_BOUNDS,
        spatial_layers=MNIST_ORIENTATION_SPATIAL_JOINT_LAYERS,
        angle_layers=MNIST_ORIENTATION_ANGLE_LAYERS,
        angle_minimum=MNIST_ORIENTATION_MIN,
        angle_maximum=MNIST_ORIENTATION_MAX,
        radius_factor=MNIST_ORIENTATION_RADIUS_FACTOR,
        isolated_radius=MNIST_ORIENTATION_ISOLATED_RADIUS,
        color_minimum_hue=LAYOUT_PSEUDO_COLOR_MIN_HUE,
        color_maximum_hue=MNIST_ORIENTATION_COLOR_MAX_HUE,
        random_seed=ENCODER_RANDOM_SEED,
        log_every=ENCODER_LOG_EVERY,
    )


def _build_stimulus_embedding_builder(
    encoder: RetinotopicOrientationEncoder,
) -> StimulusEmbeddingBuilder:
    return StimulusEmbeddingBuilder(
        code_length=encoder.code_length,
        spatial_priority_bits=encoder.spatial_code_length,
        max_density=MNIST_STIMULUS_EMBEDDING_MAX_DENSITY,
    )


def _collect_codes(
    dataset,
    label: int | None,
    count: int,
    encoder: RetinotopicOrientationEncoder,
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
            encoded_values, code = encoder.encode(
                float(a),
                float(x),
                float(y),
                log_image=log_image,
                log_label=log_label,
                log_measurements=log_measurements,
            )
            payload.add_code(
                label=int(digit),
                hue=float(encoded_values[0]),
                code=code,
            )
            total_codes += 1
            first = False
    return payload, total_codes


def _collect_samples(
    dataset,
    sample_count: int,
    encoder: RetinotopicOrientationEncoder,
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

def _run_layout(
    codes: dict[float, list],
    params: TunedLayoutParams | None,
    *,
    count: int,
    total_codes: int,
    colorizer: EncoderCodeColorizer,
) -> Layout:
    run_params = params or _default_layout_params()
    expected_pair_radius = _expected_pair_radius(total_codes, run_params)
    layout_path = _layout_json_path(count, expected_pair_radius)
    cached_layout = _load_layout_from_file(
        layout_path,
        run_params,
        colorizer=colorizer,
    )
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
        colorizer=colorizer,
    )
    long_checkpoint_path = f"{layout_path}{LAYOUT_LONG_PHASE_SUFFIX}"
    short_checkpoint_path = f"{layout_path}{LAYOUT_SHORT_PHASE_SUFFIX}"
    continuity_checkpoint_path = f"{layout_path}{LAYOUT_CONTINUITY_PHASE_SUFFIX}"
    continuity_ready = layout.restore_positions_json(continuity_checkpoint_path)
    short_ready = False
    long_ready = False
    if not continuity_ready:
        short_ready = layout.restore_positions_json(short_checkpoint_path)
    if not continuity_ready and not short_ready:
        long_ready = layout.restore_positions_json(long_checkpoint_path)
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
    if not (long_ready or short_ready or continuity_ready):
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
        )
        layout.save_json(long_checkpoint_path)
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
    if not (short_ready or continuity_ready):
        layout.run(
            steps=run_params.short_steps,
            pairs_per_step=run_params.short_pairs_per_step,
            pair_radius=short_radius_start,
            mode="short",
            local_radius=run_params.short_local_radius,
            min_swap_ratio=LAYOUT_MIN_SWAP_RATIO,
            min_swap_window=LAYOUT_MIN_SWAP_WINDOW,
            log_every=LAYOUT_LOG_EVERY_SHORT,
            step_offset=step_offset,
            energy_radius=run_params.short_energy_radius,
            energy_stability_window=LAYOUT_ENERGY_STABILITY_WINDOW,
            energy_stability_delta=LAYOUT_ENERGY_STABILITY_DELTA,
            energy_stability_every=LAYOUT_ENERGY_STABILITY_EVERY,
            energy_stability_max_points=LAYOUT_ENERGY_STABILITY_MAX_POINTS,
            adaptive_params=adaptive_short,
            short_optim=short_optim,
        )
        layout.save_json(short_checkpoint_path)
    if not continuity_ready:
        layout.run_continuity_relaxation(
            ContinuityLayoutConfig(
                sweeps=LAYOUT_CONTINUITY_SWEEPS,
                defect_points=LAYOUT_CONTINUITY_DEFECT_POINTS,
                candidates_per_point=LAYOUT_CONTINUITY_CANDIDATES,
                min_gain=LAYOUT_CONTINUITY_MIN_GAIN,
                stall_sweeps=LAYOUT_CONTINUITY_STALL_SWEEPS,
                cyclic_period=MNIST_ORIENTATION_MAX - MNIST_ORIENTATION_MIN,
                cyclic_sharp_threshold=MNIST_ORIENTATION_SMOOTH_DELTA,
            )
        )
        layout.save_json(continuity_checkpoint_path)
    return layout


class MnistLayoutRunner:
    def __init__(
        self,
        *,
        count: int,
        label: int | None = None,
        activation_samples: int | None = None,
    ) -> None:
        self.count = count
        self.label = label
        resolved_activation_samples = count if activation_samples is None else int(activation_samples)
        if resolved_activation_samples <= 0:
            raise ValueError("activation_samples must be positive")
        self._activation_samples = resolved_activation_samples
        configure_logging(profile="layout")
        GPU_ENVIRONMENT.apply()
        self.encoder = _build_encoder()
        self.extractor = MnistSobelAngleMap(
            angle_in_degrees=MNIST_ANGLE_IN_DEGREES,
            grad_threshold=MNIST_GRAD_THRESHOLD,
            local_orientation_limit=MNIST_LOCAL_ORIENTATION_LIMIT,
        )
        self._stimulus_embedding_builder = _build_stimulus_embedding_builder(self.encoder)
        self._embedding_store = EmbeddingStore(
            json_key=EMBEDDINGS_JSON_KEY,
            version_key=EMBEDDINGS_VERSION_KEY,
            version=EMBEDDINGS_VERSION,
        )
        self._checkpoint_store = PipelineCheckpointStore(
            suffix=PIPELINE_CHECKPOINT_SUFFIX,
            version=PIPELINE_CHECKPOINT_VERSION,
            replace_attempts=PIPELINE_CHECKPOINT_REPLACE_ATTEMPTS,
            replace_delay=PIPELINE_CHECKPOINT_REPLACE_DELAY,
        )

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
            pipeline_checkpoint = self._checkpoint_store.load(path)
            completed_embeddings = len(
                self._checkpoint_store.records(pipeline_checkpoint)
            )
            stages = pipeline_checkpoint.get("stages", {})
            prepare_compute = not (
                bool(stages.get("activation"))
                and completed_embeddings >= self._activation_samples
            )
            layout = _load_layout_from_file(
                path,
                tuned_params,
                colorizer=self.encoder.colorizer,
                prepare_compute=prepare_compute,
            )
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
        cloned = self._clone_reference_layout()
        if cloned is not None:
            return cloned
        LOGGER.event(
            "layout.cache.miss",
            section=LAYOUT_PARAMETERS,
            data={"count": self.count, "label": self.label},
        )
        return None

    def _clone_reference_layout(self) -> tuple[Layout, TunedLayoutParams, int, str] | None:
        if self.count <= MNIST_REFERENCE_LAYOUT_COUNT:
            return None
        for source_path in _layout_cache_paths_for_count(MNIST_REFERENCE_LAYOUT_COUNT):
            metadata = self._load_layout_metadata(source_path)
            if metadata is None:
                continue
            total_points, pair_radius, ready = metadata
            if not ready or pair_radius is None:
                continue
            tuned_params = self._tune_layout(total_points)
            layout = _load_layout_from_file(
                source_path,
                tuned_params,
                colorizer=self.encoder.colorizer,
                prepare_compute=False,
            )
            if layout is None:
                continue
            target_path = _layout_json_path(self.count, pair_radius)
            layout.save_json(target_path)
            LOGGER.event(
                "layout.reference.clone",
                section=DETECTOR_HIERARCHY,
                data={
                    "source": source_path,
                    "target": target_path,
                    "reference_images": MNIST_REFERENCE_LAYOUT_COUNT,
                    "embedding_target": self._activation_samples,
                    "points": total_points,
                },
            )
            return layout, tuned_params, total_points, target_path
        return None

    def _try_load_encoded_payload(
        self,
    ) -> tuple[LayoutPayloadBuilder, int, TunedLayoutParams, str] | None:
        # Старые файлы без метаданных label безопасны только для общего MNIST-набора.
        if self.label is not None:
            return None
        for path in self._layout_cache_paths():
            metadata = self._load_layout_metadata(path)
            if metadata is None:
                continue
            total_points, _, ready = metadata
            if ready or total_points <= 0:
                continue
            checkpoint = self._checkpoint_store.load(path)
            if not bool(checkpoint.get("stages", {}).get("encoding")):
                continue
            try:
                payload = LayoutPayloadBuilder.load_base(path)
            except Exception as exc:
                LOGGER.event(
                    "layout.payload.resume.error",
                    section=LAID_OUT_STRUCTURE,
                    data={"path": path, "error": str(exc)},
                )
                continue
            if len(payload.records) != total_points:
                continue
            code_lengths = {len(record.code) for record in payload.records}
            if code_lengths != {self.encoder.code_length}:
                continue
            tuned_params = self._tune_layout(total_points)
            expected_path = _layout_json_path(
                self.count,
                _expected_pair_radius(total_points, tuned_params),
            )
            if os.path.normcase(os.path.normpath(path)) != os.path.normcase(
                os.path.normpath(expected_path)
            ):
                continue
            LOGGER.event(
                "layout.payload.resume",
                section=LAID_OUT_STRUCTURE,
                data={
                    "path": path,
                    "points": total_points,
                    "code_length": self.encoder.code_length,
                    "next_stage": "long",
                },
            )
            return payload, total_points, tuned_params, path
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
        *,
        layout_path: str | None = None,
        save_json: bool = True,
    ) -> None:
        from PIL import Image

        resolved_layout_path = layout_path or _layout_json_path(self.count, layout.pair_radius)
        if save_json:
            layout.save_json(resolved_layout_path)
        image = layout.render_image()
        suffix = "auto" if layout.pair_radius is None else str(layout.pair_radius)
        filename = f"data/{self.count}-{suffix}.png"
        Image.fromarray(image).save(filename)
        LOGGER.event(
            "layout.visual.from_json",
            section=LAID_OUT_STRUCTURE,
            data={
                "path": resolved_layout_path,
                "image": filename,
                "points": len(layout.positions()),
            },
        )

    def _run_activation(
        self,
        layout: Layout,
        dataset,
        *,
        layout_path: str,
        stages: dict[str, bool],
        initial_embeddings: Sequence[EmbeddingRecord],
    ) -> list[EmbeddingRecord]:
        engine = None
        if _structural_diagnostics_enabled():
            energies = _compute_point_energies(layout, ACTIVATION_POINT_ENERGY_RADIUS)
            engine = _build_activation_engine(layout, energies)
        samples = _collect_samples(
            dataset,
            self._activation_samples,
            self.encoder,
            self.extractor,
            label=self.label,
        )
        embeddings = list(initial_embeddings[: len(samples)])
        start_index = len(embeddings)
        LOGGER.event(
            "activation.resume",
            section=PARALLEL_PROCESSING,
            data={
                "completed": start_index,
                "target": len(samples),
                "workers": ACTIVATION_PARALLEL_WORKERS,
                "batch_size": ACTIVATION_BATCH_SIZE,
            },
        )
        for batch_start in range(start_index, len(samples), ACTIVATION_BATCH_SIZE):
            batch = samples[batch_start : batch_start + ACTIVATION_BATCH_SIZE]
            for offset, (label, codes) in enumerate(batch):
                index = batch_start + offset
                structural_result = None
                if (
                    engine is not None
                    and (
                    index == start_index
                    or (index + 1) % ACTIVATION_STRUCTURAL_DIAGNOSTIC_EVERY == 0
                    or index + 1 == len(samples)
                    )
                ):
                    structural_result = engine.activate(codes)
                classification_embedding = self._stimulus_embedding_builder.build(codes)
                LOGGER.event(
                    "activation.sample",
                    section=STIMULUS_DETECTION,
                    data={
                        "index": index,
                        "sample_label": str(int(label)),
                        "stimuli": len(codes),
                        "active_detectors": len(structural_result.active_detectors)
                        if structural_result is not None
                        else 0,
                        "structural_embedding_bits": structural_result.embedding.count()
                        if structural_result is not None
                        else 0,
                        "embedding_bits": classification_embedding.count(),
                        "structural_evaluated": structural_result is not None,
                    },
                )
                embeddings.append(
                    EmbeddingRecord(
                        class_label=str(int(label)),
                        embedding_bits=classification_embedding.to01(),
                    )
                )
            if (
                len(embeddings) % ACTIVATION_CHECKPOINT_EVERY == 0
                or len(embeddings) == len(samples)
            ):
                self._checkpoint_store.save(
                    layout_path,
                    stages=stages,
                    target_samples=self._activation_samples,
                    embeddings=embeddings,
                )
        return embeddings

    def _save_embeddings(self, layout_path: str, embeddings: Sequence[EmbeddingRecord]) -> None:
        if not layout_path:
            raise ValueError("layout_path must be provided to save embeddings")
        self._embedding_store.save(layout_path, embeddings)

    def run(self) -> None:
        started_at = time.perf_counter()
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
            encoded_payload = self._try_load_encoded_payload()
            if encoded_payload is None:
                payload, total_codes = _collect_codes(
                    dataset,
                    self.label,
                    self.count,
                    self.encoder,
                    self.extractor,
                )
                tuned_params = self._tune_layout(total_codes)
                expected_pair_radius = _expected_pair_radius(total_codes, tuned_params)
                layout_path = _layout_json_path(self.count, expected_pair_radius)
                payload.save_base(layout_path)
                encoding_checkpoint = self._checkpoint_store.load(layout_path)
                encoding_stages = dict(encoding_checkpoint.get("stages", {}))
                encoding_stages["encoding"] = True
                self._checkpoint_store.save(
                    layout_path,
                    stages=encoding_stages,
                    target_samples=self._activation_samples,
                    embeddings=self._checkpoint_store.records(encoding_checkpoint),
                )
            else:
                payload, total_codes, tuned_params, layout_path = encoded_payload
            codes = payload.codes_by_hue()
            angles = len(codes)
            self._log_dataset(angles, total_codes)
            layout = _run_layout(
                codes,
                tuned_params,
                count=self.count,
                total_codes=total_codes,
                colorizer=self.encoder.colorizer,
            )
            save_json = True
        checkpoint = self._checkpoint_store.load(layout_path)
        stages = dict(checkpoint.get("stages", {}))
        stages["encoding"] = True
        stages["layout"] = True
        checkpoint_embeddings = self._checkpoint_store.records(checkpoint)
        self._checkpoint_store.save(
            layout_path,
            stages=stages,
            target_samples=self._activation_samples,
            embeddings=checkpoint_embeddings,
        )
        try:
            cached_embeddings = _load_embeddings_from_file(layout_path)
        except FileNotFoundError:
            cached_embeddings = []
        resume_embeddings = (
            checkpoint_embeddings
            if len(checkpoint_embeddings) >= len(cached_embeddings)
            else cached_embeddings
        )
        if len(resume_embeddings) >= self._activation_samples:
            embeddings = resume_embeddings[: self._activation_samples]
            LOGGER.event(
                "activation.embedding.skip",
                section=DETECTOR_INSERTION,
                data={
                    "path": layout_path,
                    "embeddings": len(embeddings),
                    "reason": "target_present",
                },
            )
        else:
            embeddings = self._run_activation(
                layout,
                dataset,
                layout_path=layout_path,
                stages=stages,
                initial_embeddings=resume_embeddings,
            )
            self._save_embeddings(layout_path, embeddings)
        stages["activation"] = True
        self._checkpoint_store.save(
            layout_path,
            stages=stages,
            target_samples=self._activation_samples,
            embeddings=embeddings,
        )
        self._export_layout(layout, layout_path=layout_path, save_json=save_json)
        stages["export"] = True
        self._checkpoint_store.save(
            layout_path,
            stages=stages,
            target_samples=self._activation_samples,
            embeddings=embeddings,
        )
        elapsed_seconds = time.perf_counter() - started_at
        LOGGER.event(
            LAYOUT_COMPLETION_EVENT,
            title=LAYOUT_COMPLETION_TITLE,
            section=LAYOUT_SYSTEM,
            data={
                "elapsed_seconds": elapsed_seconds,
                "elapsed_minutes": elapsed_seconds / 60.0,
                "images": self.count,
                "embeddings": len(embeddings),
                "layout_path": layout_path,
            },
            force=True,
        )


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
    def __init__(
        self,
        *,
        count: int,
        layout_path: str | None = None,
        logging_profile: str | None = None,
    ) -> None:
        self.count = count
        self.layout_path = layout_path
        self._logging_profile = logging_profile
        self._decoder: MnistDecoder | None = None
        self._temperature_override: float | None = None
        configure_logging(profile=logging_profile)
        GPU_ENVIRONMENT.apply()
        self.encoder = _build_encoder()
        self.extractor = MnistSobelAngleMap(
            angle_in_degrees=MNIST_ANGLE_IN_DEGREES,
            grad_threshold=MNIST_GRAD_THRESHOLD,
            local_orientation_limit=MNIST_LOCAL_ORIENTATION_LIMIT,
        )
        self._stimulus_embedding_builder = _build_stimulus_embedding_builder(self.encoder)
        self._embedding_store = EmbeddingStore(
            json_key=EMBEDDINGS_JSON_KEY,
            version_key=EMBEDDINGS_VERSION_KEY,
            version=EMBEDDINGS_VERSION,
        )

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
            layout = _load_layout_from_file(
                path,
                tuned_params,
                colorizer=self.encoder.colorizer,
            )
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
        activation_engine = (
            _build_activation_engine(layout, energies)
            if _structural_diagnostics_enabled()
            else None
        )
        decoder_parameters = DecoderParameterScaleResolver(
            reference_embeddings=DECODER_REFERENCE_EMBEDDINGS,
            base_top_k=DECODER_KNN_TOP_K,
            base_temperature=DECODER_SOFTMAX_TEMPERATURE,
            max_temperature=DECODER_SOFTMAX_TEMPERATURE_MAX,
        ).resolve(len(labeled_embeddings))
        database = EmbeddingDatabase(
            labeled_embeddings,
            DecoderConfig(
                similarity=DECODER_SIMILARITY,
                temperature=decoder_parameters.temperature,
                min_confidence=DECODER_MIN_CONFIDENCE,
                expected_classes=DECODER_EXPECTED_CLASSES,
                knn_top_k=decoder_parameters.knn_top_k,
                top_k_per_class=DECODER_TOP_K_PER_CLASS,
                min_similarity=DECODER_MIN_SIMILARITY,
            ),
        )
        self._decoder = MnistDecoder(
            encoder=self.encoder,
            extractor=self.extractor,
            activation_engine=activation_engine,
            embedding_db=database,
            evaluation_confidence_bins=EVALUATION_CONFIDENCE_BINS,
            evaluation_error_examples=EVALUATION_ERROR_EXAMPLES,
            stimulus_embedding_builder=self._stimulus_embedding_builder,
            structural_activation_interval=ACTIVATION_STRUCTURAL_DIAGNOSTIC_EVERY,
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
                    embedding_bits=self._stimulus_embedding_builder.build(codes).to01(),
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
        started_at = time.perf_counter()
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
        resolved_accuracy = 0.0 if accuracy is None else accuracy
        if self._logging_profile == "evaluate":
            elapsed_seconds = time.perf_counter() - started_at
            LOGGER.event(
                EVALUATION_COMPLETION_EVENT,
                title=EVALUATION_COMPLETION_TITLE,
                section=STIMULUS_DETECTION,
                data={
                    "elapsed_seconds": elapsed_seconds,
                    "elapsed_minutes": elapsed_seconds / 60.0,
                    "samples": len(results),
                    "average_accuracy": resolved_accuracy,
                    "average_accuracy_percent": resolved_accuracy * 100.0,
                    "split": split,
                },
                force=True,
            )
        return results, resolved_accuracy

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
        default=None,
        help="количество эмбеддингов; по умолчанию равно --count",
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
        decode_runner = MnistDecodeRunner(
            count=args.count,
            layout_path=args.layout_path,
            logging_profile="predict",
        )
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
        decode_runner = MnistDecodeRunner(
            count=args.count,
            layout_path=args.layout_path,
            logging_profile="evaluate",
        )
        decode_runner.predict_dataset(
            split=args.split,
            limit=args.limit,
            temperature=args.temperature,
            log_first=False,
        )
        return
    parser.error(f"unknown command {command}")


if __name__ == "__main__":
    main()
