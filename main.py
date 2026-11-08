import glob
import math
from dataclasses import dataclass
import json
import os
from collections import OrderedDict
from typing import ClassVar, Mapping, Sequence
import argparse

from damp.article_refs import (
    LAYOUT_PARAMETERS,
    OPTIM_SUBSET,
    OUTSIDE_ARTICLE,
    STIMULUS_DETECTION,
)
from damp.activation import (
    ActivationEngine,
    DbscanClusterer,
    Detector,
    DetectorHierarchyBuilder,
    DetectorLayerConfig,
    DetectorOutputSizer,
    OutputCodeSize,
    reassign_output_bits,
)
from damp.MnistSobelAngleMap import MnistSobelAngleMap
from damp.encoding.damp_encoder import (
    ClosedDimension,
    Encoder,
    EncoderCodeColorizer,
    OpenedDimension,
    geometric_detector_layers,
)
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
LOG_INTERVAL_ACTIVATION_MAP = 100
LOG_INTERVAL_ACTIVATION_DETECTOR_LEVEL = 10000
LOG_INTERVAL_ACTIVATION_DETECTOR_ACTIVE = 10000
LOG_INTERVAL_ACTIVATION_EMBEDDING = 100
LOG_INTERVAL_ACTIVATION_COLOR_MERGE = 100
LOG_INTERVAL_ACTIVATION_COVERAGE = 100
LOG_INTERVAL_ACTIVATION_DETECTOR_LAYER = 1
LOG_INTERVAL_ACTIVATION_SAMPLE = 100
LOG_INTERVAL_ACTIVATION_VISUAL = 100
LOG_INTERVAL_ACTIVATION_EMBEDDING_SAVE = 1

EMBEDDINGS_JSON_KEY = "embeddings"
EMBEDDINGS_VERSION_KEY = "embeddings_version"
EMBEDDINGS_VERSION = 5
ACTIVATION_METADATA_KEY = "activation"
ENCODER_SIGNATURE_KEY = "encoder_signature"
PIPELINE_CHECKPOINT_SUFFIX = ".checkpoint.json"
PIPELINE_CHECKPOINT_VERSION = 3

# --- Первичное кодирование стимулов MNIST (угол, X, Y) по схеме Angle-Encoder:
# случайный бит цветовой полосы у каждого детектора, слои от самого широкого к самому узкому
# (§2.2.3, §3.2, §4.2.1, §4.4, §4.5).
# §2.2.3: коды координат можно слить в общий код той же длины (OR, §2.2.1) или склеить (конкатенация).
# True — общий код длины ENCODER_SHARED_CODE_LENGTH для угла и позиции; False — сегменты
# MNIST_ANGLE_CODE_LENGTH + 2·MNIST_POSITION_CODE_LENGTH.
ENCODER_SHARED_CODE = True
ENCODER_SHARED_CODE_LENGTH = 512
ENCODER_RANDOM_SEED = 0
ENCODER_LOG_EVERY = 50
ENCODER_LAYER_PHASE = True
MNIST_ANGLE_IN_DEGREES = True
MNIST_GRAD_THRESHOLD = 0.05
MNIST_ANGLE_MIN = 0.0
MNIST_ANGLE_MAX = 360.0
MNIST_ANGLE_CODE_LENGTH = 256
MNIST_ANGLE_LAYERS = 22
MNIST_ANGLE_WIDEST_DETECTORS = 2
MNIST_ANGLE_NARROWEST_DETECTORS = 180
MNIST_ANGLE_OVERLAP = 0.4
MNIST_POSITION_MIN = 0.0
MNIST_POSITION_MAX = 28.0
MNIST_POSITION_CODE_LENGTH = 128
# 15 слоёв — минимальное уменьшение с 22, при котором в pinwheel-тесте (позиции 2×2, режим сглаживания)
# пропадает дыра на стыке четырёх позиций: соседние позиции достаточно похожи, чтобы точки на стыке
# получали энергию от чужих кластеров (22 слоя: 3–10 пустых ячеек в центре, 16: 0–5, 15: 1–2).
MNIST_POSITION_LAYERS = 22
MNIST_POSITION_WIDEST_DETECTORS = 2
# Самый узкий слой позиции — 28 детекторов (самый широкий — MNIST_POSITION_WIDEST_DETECTORS = 2);
# между ними слои растут геометрически, у каждого слоя своё число детекторов.
MNIST_POSITION_NARROWEST_DETECTORS = 28
# Перекрытие соседних позиционных детекторов слоя (§4.2.1: «detectors must overlap at each level»).
MNIST_POSITION_OVERLAP = 0.4
# False — в слое ровно MNIST_POSITION_WIDEST…NARROWEST детекторов внутри отрезка (2 … 14), без детекторов
# за краями; True — детекторы заходят за концы отрезка, как в Angle-Encoder.
MNIST_POSITION_EDGE_DETECTORS = False
# Шаг сетки позиций (индексы участков): на нём проверяется дальний порядок кодов позиций (§4.5).
MNIST_POSITION_STEP = 1.0

# --- Псевдоцвета карт: младший бит красный, старший фиолетовый (§5.14);
# диагностическая раскраска по углу стимула как на рис. 16 (§5.13.2).
LAYOUT_PSEUDO_COLOR_MIN_HUE = 0.0
LAYOUT_PSEUDO_COLOR_MAX_HUE = 300.0
LAYOUT_COMPONENT_IMAGE_SUFFIX = ".angle.png"
# Период угла для раскраски хода раскладки по углу стимула в Rerun (как pinwheel-тест, рис. 16).
LAYOUT_COMPONENT_PERIOD = MNIST_ANGLE_MAX - MNIST_ANGLE_MIN
LAYOUT_ENERGY_IMAGE_SUFFIX = ".energy.png"

# --- Раскладка (§5.4, §7.1.6, §7.1.7): cosine, порог λ с сигмоидой η,
# запас пустых ячеек ≈15%, дальняя энергия по всему пространству V (§5.5.1).
# Сглаживание пространства между пинвилами (расширение статьи): гауссово ядро дальней фазы, старт радиуса
# со всей карты, геометрическое убывание радиуса, фиксированный λ и слабый отжиг разворачивают соседние
# пинвилы друг к другу и убирают складки на их границах (docs/ARTICLE_COMPLIANCE.md).
# False — дальняя фаза строго по статье: φ = Σ s·d по всему V (§5.5.1), радиус пар линейно от d/2 к 1,
# триггер 1 % обменов и рост λ ступенями (§7.1.7), жадное правило обмена без отжига (§5.6).
LAYOUT_SMOOTH_PINWHEELS = True
# Метка кеша раскладки: раскладки старых версий Encoder и другого режима дальней фазы не переиспользуются.
LAYOUT_CACHE_TAG = "gauss-kernel-v1" if LAYOUT_SMOOTH_PINWHEELS else "article-kernel-v1"
LAYOUT_SIMILARITY = "cosine"
# Расстояние в парных энергиях: "euclidean" — d из §5.5, "squared" — d² без корня (§5.9.2).
LAYOUT_LONG_DISTANCE = "euclidean"
LAYOUT_SHORT_DISTANCE = "squared"
# Ядро дальней энергии: "distance" — φ = Σ s·d по §5.5.1; "gauss" — расширение статьи: ограниченное
# гауссово ядро окрестности радиуса r(t), r(t) следует за радиусом выбора пар (§7.1.7), не ниже минимума.
# Ограниченное ядро даёт гладкую карту из нескольких пинвилов с чередованием закрутки (как в V1).
LAYOUT_LONG_KERNEL = "gauss" if LAYOUT_SMOOTH_PINWHEELS else "distance"
LAYOUT_LONG_KERNEL_RADIUS_MIN = 3
LAYOUT_LONG_KERNEL_WIDTH = 1.0   # σ гауссова ядра = width · r(t); при σ = r связь через границы кластеров сильнее
LAYOUT_LAMBDA_THRESHOLD = 0.65
# Крутизна сигмоиды порога (§5.4); None — жёсткий порог η = ∞, как в параметрах раскладки §7.1.7.
LAYOUT_ETA = 10.0 if LAYOUT_SMOOTH_PINWHEELS else None
LAYOUT_EMPTY_RATIO = 0.15
LAYOUT_USE_GPU = True
LAYOUT_RANDOM_SEED = 0
LAYOUT_LONG_FULL_SPACE_MAX_POINTS = 8000

LAYOUT_LONG_PAIRS_PER_STEP = 24000
LAYOUT_LONG_STEPS = 14000
LAYOUT_LONG_SUBSET_SIZE = None
LAYOUT_LONG_SUBSET_REFRESH = 40

LAYOUT_SHORT_PAIRS_PER_STEP = 100
LAYOUT_SHORT_STEPS = 4000
LAYOUT_SHORT_LOCAL_RADIUS = 7
LAYOUT_SHORT_ENERGY_RADIUS = 7
LAYOUT_SHORT_ENERGY_RECALC_EVERY = 75
LAYOUT_SHORT_ENERGY_EPS = 1e-6
LAYOUT_SHORT_WEIGHTED_FIRST = True
LAYOUT_SHORT_SIMILARITY_CUTOFF = LAYOUT_LAMBDA_THRESHOLD
LAYOUT_SHORT_PARTITIONS = 8

MPS_HIGH_WATERMARK_RATIO = 0.0

LAYOUT_LOG_EVERY_LONG = 150
LAYOUT_LOG_EVERY_SHORT = 50
LAYOUT_LOG_VISUALS = True
# Стартовый радиус выбора пар в долях стороны карты: 0.5 — d/2 по §7.1.7; 1.0 — вся карта,
# как начальная ширина окрестности в SOM (нужно гауссову ядру, иначе позиции распадаются на куски).
LAYOUT_ADAPTIVE_RADIUS_START_FACTOR = 1.0 if LAYOUT_SMOOTH_PINWHEELS else 0.5
# Расписание радиуса выбора пар: "swaps" — по статье, λ и r меняются, когда число обменов за шаг
# падает до LAYOUT_ADAPTIVE_SWAP_TRIGGER от начального значения этапа (§7.1.7); "geometric" и "linear" —
# расширение статьи: r убывает по номеру шага (геометрически в дальней фазе, линейно в ближней).
LAYOUT_ADAPTIVE_RADIUS_SCHEDULE = "geometric" if LAYOUT_SMOOTH_PINWHEELS else "swaps"
LAYOUT_SHORT_ADAPTIVE_RADIUS_SCHEDULE = "linear" if LAYOUT_SMOOTH_PINWHEELS else "swaps"
# Во сколько раз уменьшается r при срабатывании триггера (статья: «постепенно уменьшали до 1», §7.1.7).
LAYOUT_ADAPTIVE_RADIUS_TRIGGER_FACTOR = 0.5
# Окно шагов, по которому усредняется число обменов («обмены в единицу времени», §7.1.7).
LAYOUT_ADAPTIVE_SWAP_WINDOW = 10
# Отжиг в дальней фазе (расширение статьи): температура принятия невыгодных обменов в долях rms|Δ|
# на радиусах от LAYOUT_ANNEAL_RADIUS_MIN до LAYOUT_ANNEAL_RADIUS_MAX_FACTOR·d — масштаб, на котором
# соседние кластеры согласуют закрутку и фазу; снимает замёрзшие складки между кластерами.
# 0 — отжига нет, обмен только при выигрыше энергии (§5.6).
LAYOUT_ANNEAL_TEMPERATURE = 0.1 if LAYOUT_SMOOTH_PINWHEELS else 0.0
LAYOUT_ANNEAL_RADIUS_MAX_FACTOR = 0.4
LAYOUT_ANNEAL_RADIUS_MIN = 5
LAYOUT_ADAPTIVE_RADIUS_MIN = 1
# Триггер по доле обменов (1% от начальной, §7.1.7): 0 — расписание не ускоряется по обменам.
LAYOUT_ADAPTIVE_SWAP_TRIGGER = 0.0 if LAYOUT_SMOOTH_PINWHEELS else 0.01
# Рост λ ступенями до LAYOUT_ADAPTIVE_LAMBDA_MAX (§7.1.7); None — λ фиксирован: с гауссовым ядром
# рост λ ослабляет связь через границы кластеров и оставляет складки.
LAYOUT_ADAPTIVE_LAMBDA_STEP = None if LAYOUT_SMOOTH_PINWHEELS else 0.05
# λ поднимают с 0.65 до 0.8 (§7.1.7).
LAYOUT_ADAPTIVE_LAMBDA_MAX = 0.8
LAYOUT_SHORT_ADAPTIVE_RADIUS_FACTOR = 1.0
# Остановка (§5.6): число обменов за шаг ниже порога или качество Ē (§5.8) достигло заданного значения.
# Ē считается каждые LAYOUT_ENERGY_EVAL_EVERY шагов дальней фазы по всем точкам V с нормировкой на E_max по V
# (§5.7, §5.8); LAYOUT_ENERGY_TARGET = None — только по обменам.
LAYOUT_ENERGY_EVAL_EVERY = 200
# Радиус r_e энергии точки: одна матрица Ê для качества Ē (§5.8) и для детекторов (§6.3); «r_e ≤ 5 достаточно» (§5.7).
LAYOUT_ENERGY_RADIUS = 3
LAYOUT_ENERGY_TARGET = None
LAYOUT_MIN_SWAP_RATIO = 0.0001
LAYOUT_MIN_SWAP_WINDOW = 20

# --- Детекторная иерархия (§6.3–§6.5, §7.1.10, §7.1.12). Порог активации λ_a
# равен порогу нижнего слоя детекторов (§7.1.11); размер и насыщенность
# выходного кода вычисляются по среднему числу активных детекторов (§6.6).
ACTIVATION_LAMBDA_THRESHOLD = 0.7
ACTIVATION_ETA = None
ACTIVATION_ENERGY_THRESHOLD = 0.12
ACTIVATION_DETECTOR_THRESHOLD = 0.5
ACTIVATION_OUTPUT_BITS = 256
ACTIVATION_SATURATION_LIMIT = 50
ACTIVATION_MAX_EMBEDDING_DENSITY = 0.25
ACTIVATION_SATURATION_MIN = 16
ACTIVATION_SATURATION_MAX = 128
ACTIVATION_OUTPUT_CALIBRATION_SAMPLES = 100
ACTIVATION_POINT_ENERGY_RADIUS = LAYOUT_ENERGY_RADIUS
ACTIVATION_DETECTOR_RADIUS_MIN = 1
# Слои иерархии (§7.1.10): порог λ_d, радиус активации r_a как доля стороны
# карты (нижние слои — области и гиперкластеры, верхние — кластеры и секторы)
# и минимальное число точек кластера.
ACTIVATION_DETECTOR_LAYER_SPECS = (
    (0.70, 0.35, 12),
    (0.75, 0.28, 10),
    (0.80, 0.22, 8),
    (0.85, 0.16, 6),
    (0.90, 0.12, 4),
)
ACTIVATION_DETECTOR_ATTEMPTS_WITHOUT_CHANGE = 400
ACTIVATION_DETECTOR_SEED = 0
ACTIVATION_DBSCAN_EPS = 1.5
ACTIVATION_DBSCAN_MIN_SAMPLES = 3
ACTIVATION_USE_GPU = True
ACTIVATION_PARALLEL_WORKERS = 4
ACTIVATION_BATCH_SIZE = 10
ACTIVATION_CHECKPOINT_EVERY = 10

# --- Декодер: нечёткий поиск по базе эмбеддингов (§2.2.5).
# J2 (§2.2.4.2); для битовых векторов Σa·b / Σmax(a², b²) = |a∧b| / |a∨b|, то есть совпадает с J.
DECODER_SIMILARITY = "jaccard2"
DECODER_SOFTMAX_TEMPERATURE = 0.35
DECODER_MIN_CONFIDENCE = 0.05
DECODER_EXPECTED_CLASSES = tuple(range(10))
DECODER_KNN_TOP_K = 200
DECODER_TOP_K_PER_CLASS = 20
DECODER_MIN_SIMILARITY = 0.0
DECODER_DATASET_LIMIT = 64
DECODER_DEFAULT_COUNT = 1000
DECODER_CALIBRATION_TEMPERATURES = (0.25, 0.35, 0.5, 0.75, 1.0)

# §5.10: пары шага считаются спекулятивно по старым позициям; это не влияет на результат, пока обменов
# за шаг много меньше числа точек. В режиме статьи число пар за шаг (а значит, и обменов) ограничено
# долей LAYOUT_PAIRS_POINT_FRACTION от числа точек в обеих фазах; None — без ограничения (режим сглаживания,
# у которого расписание радиуса привязано к номеру шага и подобрано под прежний бюджет пар).
LAYOUT_PAIRS_POINT_FRACTION = None if LAYOUT_SMOOTH_PINWHEELS else 0.1
LAYOUT_TUNE_LONG_PAIR_FACTOR = 0.35
LAYOUT_TUNE_LONG_PAIRS_MIN = 4000
LAYOUT_TUNE_LONG_PAIRS_MAX = 20000
LAYOUT_TUNE_LONG_STEPS_DIVISOR = 25
# При сглаживании бюджет задаёт пошаговое расписание радиуса; по статье дальняя фаза заканчивается по числу
# обменов (§5.6), а при парах ≤ 0.1·N (§5.10) этапу §7.1.7 нужно ~10³ шагов независимо от N (каждая точка
# попадает в пару ~0.2 раза за шаг) — бюджет только верхняя граница.
LAYOUT_TUNE_LONG_STEPS_MIN = 2500 if LAYOUT_SMOOTH_PINWHEELS else 50000
LAYOUT_TUNE_LONG_STEPS_MAX = 5000 if LAYOUT_SMOOTH_PINWHEELS else 50000
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
# При сглаживании радиус ближней фазы убывает по номеру шага, и бюджет задаёт расписание; по статье фаза
# заканчивается по числу обменов (§5.6), и бюджет — только верхняя граница, которой должно хватить на r → 1.
LAYOUT_TUNE_SHORT_STEPS_MIN = 160 if LAYOUT_SMOOTH_PINWHEELS else 2500
LAYOUT_TUNE_SHORT_STEPS_MAX = 900 if LAYOUT_SMOOTH_PINWHEELS else 5000
LAYOUT_TUNE_SHORT_RADIUS_FACTOR = 0.08
LAYOUT_SHORT_LOCAL_RADIUS_MIN = 3
LAYOUT_TUNE_SHORT_ENERGY_RADIUS_FACTOR = 0.06
LAYOUT_TUNE_SHORT_ENERGY_RECALC_FLOOR = 30

# --- Тест MNIST-кодера: по 360 углов при двух позициях в сетке участков.
# Код Angle || X || Y и параметры раскладки общие с MNIST (§2.2.3, §4.6, §5).
PINWHEEL_ANGLE_COUNT = 360
PINWHEEL_POSITION_X = (MNIST_POSITION_MIN + MNIST_POSITION_MAX) / 2.0
PINWHEEL_POSITION_Y = (MNIST_POSITION_MIN + MNIST_POSITION_MAX) / 2.0
PINWHEEL_POSITIONS = (
    (10.0, 10.0),
    (20.0, 10.0),
    (10.0, 20.0),
    (20.0, 20.0),

)


PINWHEEL_SMOOTH_DELTA = 15.0
PINWHEEL_OUTPUT_PATH = "pinwheel_test.png"
PINWHEEL_COMPONENT_OUTPUT_PATH = "pinwheel_test.angle.png"
PINWHEEL_ENERGY_OUTPUT_PATH = "pinwheel_test.energy.png"
PINWHEEL_LAYOUT_PATH = "pinwheel_test.json"

# --- Визуализация первичных детекторов угла и позиции (detectors_visual.py): слои, рецептивные поля,
# их перекрытие и цветовые полосы бит (§4.1, §4.2.1, §3.2). Детекторы, активные для примера стимула
# (угол, X, Y), выделяются; остальные рисуются с прозрачностью DETECTORS_VIEW_INACTIVE_ALPHA.
DETECTORS_VIEW_OUTPUT_PATH = "detectors_view.png"
DETECTORS_VIEW_STIMULUS = (45.0, 14.0, 14.0)
DETECTORS_VIEW_INACTIVE_ALPHA = 0.3
DETECTORS_VIEW_DPI = 110
DETECTORS_VIEW_WIDTH_INCHES = 18.0
LOG_INTERVALS = {
    "detectors.init": LOG_INTERVAL_INIT,
    "detectors.geometric_layers": LOG_INTERVAL_INIT,
    "encoder.random_bits": LOG_INTERVAL_INIT,
    "encoder.colorizer.dimension": LOG_INTERVAL_INIT,
    "encoder.code.color": LOG_INTERVAL_LAYOUT_PAIR,
    "layout.similarity_threshold": LOG_INTERVAL_INIT,
    "layout.point_energy.map": LOG_INTERVAL_INIT,
    "layout.render_component_image": LOG_INTERVAL_LAYOUT_VISUAL,
    "layout.render_energy_image": LOG_INTERVAL_LAYOUT_VISUAL,
    "layout.visual.save": LOG_INTERVAL_LAYOUT_EXPORT,
    "activation.detector.builder.init": LOG_INTERVAL_INIT,
    "activation.detector.levels": LOG_INTERVAL_INIT,
    "activation.detector.hierarchy": LOG_INTERVAL_INIT,
    "activation.detector.layer.filled": LOG_INTERVAL_INIT,
    "activation.detector.cluster": LOG_INTERVAL_ACTIVATION_DETECTOR_LEVEL,
    "activation.detector.cluster.noise": LOG_INTERVAL_ACTIVATION_DETECTOR_LEVEL,
    "activation.detector.center": LOG_INTERVAL_ACTIVATION_DETECTOR_LEVEL,
    "activation.detector.radius": LOG_INTERVAL_ACTIVATION_DETECTOR_LEVEL,
    "activation.detector.params": LOG_INTERVAL_ACTIVATION_DETECTOR_LEVEL,
    "activation.embedding.saturation": LOG_INTERVAL_ACTIVATION_EMBEDDING,
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
    "encoder.colour_band": LOG_INTERVAL_INIT,
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
    "layout.run.energy_target": LOG_INTERVAL_INIT,
    "layout.run.energy_target.stop": LOG_INTERVAL_INIT,
    "layout.adaptive.trigger": LOG_INTERVAL_INIT,
    "layout.adaptive.stage.forced": LOG_INTERVAL_INIT,
    "layout.adaptive.incomplete": LOG_INTERVAL_INIT,
    "layout.run.mode": LOG_INTERVAL_INIT,
    "layout.run.selection": LOG_INTERVAL_INIT,
    "layout.run.speculative": LOG_INTERVAL_INIT,
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
    "activation.map.init": LOG_INTERVAL_ACTIVATION_INIT,
    "activation.map": LOG_INTERVAL_ACTIVATION_MAP,
    "activation.detector.level": LOG_INTERVAL_ACTIVATION_DETECTOR_LEVEL,
    "activation.detector.active": LOG_INTERVAL_ACTIVATION_DETECTOR_ACTIVE,
    "activation.detector.init": LOG_INTERVAL_ACTIVATION_INIT,
    "activation.detector.layer": LOG_INTERVAL_ACTIVATION_DETECTOR_LAYER,
    "activation.embedding": LOG_INTERVAL_ACTIVATION_EMBEDDING,
    "activation.color_merge": LOG_INTERVAL_ACTIVATION_COLOR_MERGE,
    "activation.detector.coverage": LOG_INTERVAL_ACTIVATION_COVERAGE,
    "activation.sample": LOG_INTERVAL_ACTIVATION_SAMPLE,
    "activation.map.visual": LOG_INTERVAL_ACTIVATION_VISUAL,
    "activation.energy.visual": LOG_INTERVAL_ACTIVATION_VISUAL,
    "activation.embedding.save": LOG_INTERVAL_ACTIVATION_EMBEDDING_SAVE,
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
            section=OUTSIDE_ARTICLE,
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

    def save(
        self,
        path: str,
        embeddings: Sequence[EmbeddingRecord],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
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
                    section=OUTSIDE_ARTICLE,
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
        if metadata is not None:
            payload[ACTIVATION_METADATA_KEY] = dict(metadata)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=True, indent=2)
        LOGGER.event(
            "activation.embedding.save",
            section=OUTSIDE_ARTICLE,
            data={
                "path": path,
                "embeddings": len(embeddings),
            "json_key": self._json_key,
        },
    )

    def load(self, path: str) -> list["EmbeddingRecord"]:
        """Эмбеддинги базы, сохранённые в JSON раскладки (вне статьи)."""
        if not path:
            raise ValueError("path must be provided to load embeddings")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        stored_version = data.get(self._version_key)
        if stored_version != self._version:
            LOGGER.event(
                "activation.embedding.incompatible",
                section=OUTSIDE_ARTICLE,
                data={
                    "path": path,
                    "stored_version": stored_version,
                    "expected_version": self._version,
                },
            )
            return []
        raw_embeddings = data.get(self._json_key, [])
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
            section=OUTSIDE_ARTICLE,
            data={
                "path": path,
                "embeddings": len(embeddings),
                "classes": classes,
                "json_key": self._json_key,
            },
        )
        return embeddings

    @staticmethod
    def to_labeled(records: Sequence["EmbeddingRecord"]) -> list[LabeledEmbedding]:
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
                section=OUTSIDE_ARTICLE,
                data={
                    "embeddings": len(labeled),
                    "classes": len({item.label for item in labeled}),
                    "bit_length": len(labeled[0].embedding),
                },
            )
        return labeled

    def load_metadata(self, path: str) -> dict[str, object]:
        """Размер выходного кода и насыщенность, сохранённые при обучении (§6.6)."""
        if not path or not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if data.get(self._version_key) != self._version:
            return {}
        metadata = data.get(ACTIVATION_METADATA_KEY)
        return dict(metadata) if isinstance(metadata, dict) else {}



class PipelineCheckpointStore:
    """Атомарно сохраняет прогресс этапов и частичные эмбеддинги."""

    def __init__(self, *, suffix: str, version: int) -> None:
        self._suffix = str(suffix)
        self._version = int(version)

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
                section=OUTSIDE_ARTICLE,
                data={"path": path, "error": str(exc)},
            )
            return self._empty_payload()
        if payload.get("version") != self._version:
            LOGGER.event(
                "pipeline.checkpoint.incompatible",
                section=OUTSIDE_ARTICLE,
                data={
                    "path": path,
                    "stored_version": payload.get("version"),
                    "expected_version": self._version,
                },
            )
            return self._empty_payload()
        LOGGER.event(
            "pipeline.checkpoint.load",
            section=OUTSIDE_ARTICLE,
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
        os.replace(temporary_path, path)
        LOGGER.event(
            "pipeline.checkpoint.save",
            section=OUTSIDE_ARTICLE,
            data={
                "path": path,
                "completed": len(embeddings),
                "target": int(target_samples),
                "stages": stages,
            },
        )

    def reset(self, layout_path: str, *, target_samples: int) -> None:
        """Пустой чекпоинт: этапы и эмбеддинги прошлого запуска не используются (запуск без кеша)."""
        self.save(layout_path, stages={}, target_samples=target_samples, embeddings=[])

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
        if total_points <= LAYOUT_LONG_FULL_SPACE_MAX_POINTS:
            # §5.5.1: дальняя энергия считается по всему пространству V.
            long_subset_size: int | None = None
        else:
            # §5.9.4: для очень больших пространств допускается подмножество.
            long_subset_size = min(
                LAYOUT_TUNE_LONG_SUBSET_MAX,
                max(LAYOUT_TUNE_LONG_SUBSET_MIN, int(math.sqrt(total_points) * LAYOUT_TUNE_LONG_SUBSET_MULTIPLIER)),
            )
            LOGGER.event(
                "layout.tune.long_subset",
                section=OPTIM_SUBSET,
                data={"points": total_points, "long_subset_size": long_subset_size,
                      "full_space_max_points": LAYOUT_LONG_FULL_SPACE_MAX_POINTS},
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
        if LAYOUT_PAIRS_POINT_FRACTION is not None:
            # §5.10: обменов за шаг много меньше числа точек — пар за шаг не больше доли точек.
            pairs_cap = max(1, int(total_points * LAYOUT_PAIRS_POINT_FRACTION))
            long_pairs = min(long_pairs, pairs_cap)
            short_pairs = min(short_pairs, pairs_cap)
        short_steps = max(
            LAYOUT_TUNE_SHORT_STEPS_MIN,
            min(LAYOUT_TUNE_SHORT_STEPS_MAX, int(grid_size * LAYOUT_TUNE_SHORT_STEPS_FACTOR)),
        )
        # §5.5.2: окрестность R должна содержать точки помимо самой пары,
        # поэтому локальный радиус не опускается ниже LAYOUT_SHORT_LOCAL_RADIUS_MIN.
        short_local_radius = max(
            LAYOUT_SHORT_LOCAL_RADIUS_MIN, int(grid_size * LAYOUT_TUNE_SHORT_RADIUS_FACTOR)
        )
        short_energy_radius = max(
            LAYOUT_SHORT_LOCAL_RADIUS_MIN, int(grid_size * LAYOUT_TUNE_SHORT_ENERGY_RADIUS_FACTOR)
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
                "long_subset_size": -1 if long_subset_size is None else long_subset_size,
                "long_subset_mode": "full_space" if long_subset_size is None else "subset",
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


def _detector_layer_configs(layout: Layout) -> tuple[DetectorLayerConfig, ...]:
    """Слои иерархии: один порог λ_d на слой, r_a и минимум точек (§6.4, §7.1.10)."""
    extent = max(1, min(layout.width, layout.height))
    return tuple(
        DetectorLayerConfig(
            lambda_threshold=float(level),
            activation_radius=max(ACTIVATION_DETECTOR_RADIUS_MIN, int(round(extent * float(radius_factor)))),
            min_points=int(min_points),
        )
        for level, radius_factor, min_points in ACTIVATION_DETECTOR_LAYER_SPECS
    )


def _build_detectors_from_layout(
    layout: Layout,
    energies: list[float],
    *,
    output_bits: int,
) -> list[Detector]:
    return DetectorHierarchyBuilder(
        layout=layout,
        energies=energies,
        layer_configs=_detector_layer_configs(layout),
        radius_min=ACTIVATION_DETECTOR_RADIUS_MIN,
        output_bits=output_bits,
        energy_threshold=ACTIVATION_ENERGY_THRESHOLD,
        eta=ACTIVATION_ETA,
        attempts_without_change=ACTIVATION_DETECTOR_ATTEMPTS_WITHOUT_CHANGE,
        seed=ACTIVATION_DETECTOR_SEED,
        clusterer=DbscanClusterer(
            eps=ACTIVATION_DBSCAN_EPS,
            min_samples=ACTIVATION_DBSCAN_MIN_SAMPLES,
        ),
    ).build()


def _build_activation_engine(
    layout: Layout,
    energies: list[float],
    *,
    detectors: Sequence[Detector] | None = None,
    saturation_limit: int = ACTIVATION_SATURATION_LIMIT,
) -> ActivationEngine:
    if detectors is None:
        detectors = _build_detectors_from_layout(
            layout,
            energies,
            output_bits=ACTIVATION_OUTPUT_BITS,
        )
    return ActivationEngine(
        layout=layout,
        detectors=detectors,
        lambda_threshold=ACTIVATION_LAMBDA_THRESHOLD,
        eta=ACTIVATION_ETA,
        energy_threshold=ACTIVATION_ENERGY_THRESHOLD,
        detector_threshold=ACTIVATION_DETECTOR_THRESHOLD,
        saturation_limit=saturation_limit,
        energy_map=energies,
        use_gpu=ACTIVATION_USE_GPU,
    )


def _output_sizer() -> DetectorOutputSizer:
    return DetectorOutputSizer(
        max_density=ACTIVATION_MAX_EMBEDDING_DENSITY,
        min_output_bits=ACTIVATION_OUTPUT_BITS,
        min_saturation=ACTIVATION_SATURATION_MIN,
        max_saturation=ACTIVATION_SATURATION_MAX,
    )


def _calibrated_activation_engine(
    layout: Layout,
    energies: list[float],
    calibration_batches: Sequence[Sequence[BitArray]],
) -> tuple[ActivationEngine, OutputCodeSize]:
    """§6.6: строит детекторы, измеряет среднее число активных и задаёт размер выхода."""
    detectors = _build_detectors_from_layout(layout, energies, output_bits=ACTIVATION_OUTPUT_BITS)
    provisional = _build_activation_engine(layout, energies, detectors=detectors)
    size = _output_sizer().measure(provisional, calibration_batches)
    resized = reassign_output_bits(detectors, output_bits=size.output_bits, seed=ACTIVATION_DETECTOR_SEED)
    engine = _build_activation_engine(
        layout, energies, detectors=resized, saturation_limit=size.saturation_limit
    )
    return engine, size


def _activation_engine_from_size(
    layout: Layout,
    energies: list[float],
    *,
    output_bits: int,
    saturation_limit: int,
) -> ActivationEngine:
    """Восстанавливает детекторы и выходной код тех же размеров, что при обучении."""
    detectors = _build_detectors_from_layout(layout, energies, output_bits=ACTIVATION_OUTPUT_BITS)
    resized = reassign_output_bits(detectors, output_bits=output_bits, seed=ACTIVATION_DETECTOR_SEED)
    return _build_activation_engine(layout, energies, detectors=resized, saturation_limit=saturation_limit)


def _activation_metadata(size: OutputCodeSize) -> dict[str, object]:
    return {
        "output_bits": int(size.output_bits),
        "saturation_limit": int(size.saturation_limit),
        "mean_active_detectors": float(size.mean_active_detectors),
        "calibration_samples": int(size.samples),
    }


class LayoutCacheFile:
    """JSON раскладки на диске: проверка совместимости и восстановление Layout (вне статьи)."""

    @staticmethod
    def inspect(path: str) -> tuple[int, int | None, bool] | None:
        try:
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
    ) -> Layout | None:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        points_payload = data.get("layout", [])
        stored_signature = data.get(ENCODER_SIGNATURE_KEY)
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
        if pair_radius_saved is not None and pair_radius_saved != expected_pair_radius:
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
            empty_ratio=LAYOUT_EMPTY_RATIO,
            similarity=data.get("similarity", LAYOUT_SIMILARITY),
            lambda_threshold=data.get("lambda_threshold", LAYOUT_LAMBDA_THRESHOLD),
            eta=data.get("eta", LAYOUT_ETA),
            seed=LAYOUT_RANDOM_SEED,
            use_gpu=LAYOUT_USE_GPU,
            long_subset_size=None,
            long_subset_refresh=LAYOUT_LONG_SUBSET_REFRESH,
            colorizer=colorizer,
            long_distance=LAYOUT_LONG_DISTANCE,
            short_distance=LAYOUT_SHORT_DISTANCE,
            long_kernel=LAYOUT_LONG_KERNEL,
            long_kernel_radius_min=LAYOUT_LONG_KERNEL_RADIUS_MIN,
            long_kernel_width=LAYOUT_LONG_KERNEL_WIDTH,
            component_period=LAYOUT_COMPONENT_PERIOD,
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


def _build_encoder() -> Encoder:
    """Кодер MNIST-стимула (угол, X, Y): общий код или конкатенация сегментов (§2.2.3, §4.2.1, §4.4)."""
    angle_layers = geometric_detector_layers(
        layers_count=MNIST_ANGLE_LAYERS,
        widest_detectors=MNIST_ANGLE_WIDEST_DETECTORS,
        narrowest_detectors=MNIST_ANGLE_NARROWEST_DETECTORS,
        overlap=MNIST_ANGLE_OVERLAP,
    )
    position_layers = geometric_detector_layers(
        layers_count=MNIST_POSITION_LAYERS,
        widest_detectors=MNIST_POSITION_WIDEST_DETECTORS,
        narrowest_detectors=MNIST_POSITION_NARROWEST_DETECTORS,
        overlap=MNIST_POSITION_OVERLAP,
    )
    angle_length = ENCODER_SHARED_CODE_LENGTH if ENCODER_SHARED_CODE else MNIST_ANGLE_CODE_LENGTH
    position_length = ENCODER_SHARED_CODE_LENGTH if ENCODER_SHARED_CODE else MNIST_POSITION_CODE_LENGTH
    encoder = Encoder(
        ClosedDimension(
            "Angle",
            (MNIST_ANGLE_MIN, MNIST_ANGLE_MAX),
            angle_layers,
            code_length=angle_length,
        ),
        OpenedDimension(
            "X",
            (MNIST_POSITION_MIN, MNIST_POSITION_MAX),
            position_layers,
            code_length=position_length,
            edge_detectors=MNIST_POSITION_EDGE_DETECTORS,
        ),
        OpenedDimension(
            "Y",
            (MNIST_POSITION_MIN, MNIST_POSITION_MAX),
            position_layers,
            code_length=position_length,
            edge_detectors=MNIST_POSITION_EDGE_DETECTORS,
        ),
        random_seed=ENCODER_RANDOM_SEED,
        log_every=ENCODER_LOG_EVERY,
        layer_phase=ENCODER_LAYER_PHASE,
        shared_code=ENCODER_SHARED_CODE,
    )
    # §4.5, §5.13.1: у каждой координаты сходство кодов должно убывать с расстоянием
    # на всём диапазоне, иначе раскладка не воспроизведёт её топологию.
    for dimension in encoder.dimensions:
        step = None if dimension.closed else MNIST_POSITION_STEP
        # Encoder.long_range_order сам логирует профиль и предупреждение (§4.5, §5.13.1).
        encoder.long_range_order(dimension.title, step=step)
    return encoder


def _build_colorizer() -> EncoderCodeColorizer:
    """Псевдоцвет всего кода: младший бит красный, старший фиолетовый (§5.14)."""
    return EncoderCodeColorizer(
        min_hue=LAYOUT_PSEUDO_COLOR_MIN_HUE,
        max_hue=LAYOUT_PSEUDO_COLOR_MAX_HUE,
    )


def _build_extractor() -> MnistSobelAngleMap:
    return MnistSobelAngleMap(
        angle_in_degrees=MNIST_ANGLE_IN_DEGREES,
        grad_threshold=MNIST_GRAD_THRESHOLD,
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
        metadata={ENCODER_SIGNATURE_KEY: encoder.signature},
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

def _run_layout(
    codes: dict[float, list],
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
    layout = Layout(
        codes,
        empty_ratio=LAYOUT_EMPTY_RATIO,
        similarity=LAYOUT_SIMILARITY,
        lambda_threshold=LAYOUT_LAMBDA_THRESHOLD,
        eta=LAYOUT_ETA,
        seed=LAYOUT_RANDOM_SEED,
        use_gpu=LAYOUT_USE_GPU,
        long_subset_size=run_params.long_subset_size,
        long_subset_refresh=run_params.long_subset_refresh,
        colorizer=colorizer,
        long_distance=LAYOUT_LONG_DISTANCE,
        short_distance=LAYOUT_SHORT_DISTANCE,
        long_kernel=LAYOUT_LONG_KERNEL,
        long_kernel_radius_min=LAYOUT_LONG_KERNEL_RADIUS_MIN,
        long_kernel_width=LAYOUT_LONG_KERNEL_WIDTH,
        component_period=LAYOUT_COMPONENT_PERIOD,
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
        lambda_max=LAYOUT_ADAPTIVE_LAMBDA_MAX,
        radius_schedule=LAYOUT_ADAPTIVE_RADIUS_SCHEDULE,
        radius_factor=LAYOUT_ADAPTIVE_RADIUS_TRIGGER_FACTOR,
        swap_window=LAYOUT_ADAPTIVE_SWAP_WINDOW,
        temperature=LAYOUT_ANNEAL_TEMPERATURE,
        temperature_radius_max=int(layout.width * LAYOUT_ANNEAL_RADIUS_MAX_FACTOR),
        temperature_radius_min=LAYOUT_ANNEAL_RADIUS_MIN,
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
        energy_radius=LAYOUT_ENERGY_RADIUS,
        energy_every=LAYOUT_ENERGY_EVAL_EVERY,
        energy_target=LAYOUT_ENERGY_TARGET,
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
        lambda_max=LAYOUT_ADAPTIVE_LAMBDA_MAX,
        radius_schedule=LAYOUT_SHORT_ADAPTIVE_RADIUS_SCHEDULE,
        radius_factor=LAYOUT_ADAPTIVE_RADIUS_TRIGGER_FACTOR,
        swap_window=LAYOUT_ADAPTIVE_SWAP_WINDOW,
    )
    short_optim = ShortLayoutOptimConfig(
        energy_radius=run_params.short_energy_radius,
        energy_recalc_every=run_params.short_energy_recalc_every,
        energy_eps=LAYOUT_SHORT_ENERGY_EPS,
        use_weighted_first_point=LAYOUT_SHORT_WEIGHTED_FIRST,
        similarity_cutoff=LAYOUT_SHORT_SIMILARITY_CUTOFF,
        partitions=LAYOUT_SHORT_PARTITIONS,
    )
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
        adaptive_params=adaptive_short,
        short_optim=short_optim,
    )
    return layout


class MnistLayoutRunner:
    @staticmethod
    def _save_layout_images(
        layout: Layout,
        *,
        image_path: str,
        energies: Sequence[float] | None = None,
    ) -> None:
        """Карта в псевдоцветах кодов (§5.14), по компоненте угла (рис. 16) и с яркостью Ê (рис. 18)."""
        from PIL import Image

        directory = os.path.dirname(image_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        Image.fromarray(layout.render_image()).save(image_path)
        stem = os.path.splitext(image_path)[0]
        component_path = stem + LAYOUT_COMPONENT_IMAGE_SUFFIX
        Image.fromarray(
            layout.render_component_image(period=MNIST_ANGLE_MAX - MNIST_ANGLE_MIN)
        ).save(component_path)
        energy_path = None
        if energies is not None:
            energy_path = stem + LAYOUT_ENERGY_IMAGE_SUFFIX
            Image.fromarray(layout.render_energy_image(energies)).save(energy_path)
        LOGGER.event(
            "layout.visual.save",
            section=OUTSIDE_ARTICLE,
            data={
                "image": image_path,
                "component_image": component_path,
                "energy_image": energy_path,
                "points": len(layout.positions()),
            },
        )

    def __init__(
        self,
        *,
        count: int,
        label: int | None = None,
        activation_samples: int | None = None,
        reuse_cache: bool = True,
    ) -> None:
        self.count = count
        self.label = label
        # False (--rebuild) — готовые раскладка, чекпоинт и эмбеддинги игнорируются, всё считается заново
        # и записывается в кеш поверх старого.
        self._reuse_cache = bool(reuse_cache)
        resolved_activation_samples = count if activation_samples is None else int(activation_samples)
        if resolved_activation_samples <= 0:
            raise ValueError("activation_samples must be positive")
        self._activation_samples = resolved_activation_samples
        configure_logging()
        GPU_ENVIRONMENT.apply()
        self.encoder = _build_encoder()
        self.colorizer = _build_colorizer()
        self.extractor = _build_extractor()
        self._embedding_store = EmbeddingStore(
            json_key=EMBEDDINGS_JSON_KEY,
            version_key=EMBEDDINGS_VERSION_KEY,
            version=EMBEDDINGS_VERSION,
        )
        self._checkpoint_store = PipelineCheckpointStore(
            suffix=PIPELINE_CHECKPOINT_SUFFIX,
            version=PIPELINE_CHECKPOINT_VERSION,
        )
        self._activation_metadata: dict[str, object] | None = None

    def _layout_cache_paths(self) -> list[str]:
        return _layout_cache_paths_for_count(self.count)

    def _load_layout_metadata(self, path: str) -> tuple[int, int | None, bool] | None:
        return LayoutCacheFile.inspect(path)

    def _try_load_cached_layout(self) -> tuple[Layout, TunedLayoutParams, int, str] | None:
        for path in self._layout_cache_paths():
            metadata = self._load_layout_metadata(path)
            if metadata is None:
                continue
            total_points, pair_radius, ready = metadata
            if not ready:
                LOGGER.event(
                    "layout.cache.incomplete",
                    section=OUTSIDE_ARTICLE,
                    data={
                        "path": path,
                        "points": total_points,
                        "pair_radius": pair_radius,
                    },
                )
                continue
            tuned_params = self._tune_layout(total_points)
            layout = LayoutCacheFile.load(
                path,
                tuned_params,
                expected_code_length=self.encoder.code_length,
                colorizer=self.colorizer,
                expected_encoder_signature=self.encoder.signature,
            )
            if layout is None:
                LOGGER.event(
                    "layout.cache.mismatch",
                    section=OUTSIDE_ARTICLE,
                    data={
                        "path": path,
                        "pair_radius": pair_radius,
                        "points": total_points,
                    },
                )
                continue
            LOGGER.event(
                "layout.cache.hit",
                section=OUTSIDE_ARTICLE,
                data={
                    "path": path,
                    "points": total_points,
                    "pair_radius": pair_radius,
                },
            )
            return layout, tuned_params, total_points, path
        LOGGER.event(
            "layout.cache.miss",
            section=OUTSIDE_ARTICLE,
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
            section=OUTSIDE_ARTICLE,
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
        resolved_layout_path = layout_path or _layout_json_path(self.count, layout.pair_radius)
        if save_json:
            layout.save_json(resolved_layout_path)
        suffix = "auto" if layout.pair_radius is None else str(layout.pair_radius)
        self._save_layout_images(
            layout,
            image_path=f"data/{self.count}-{suffix}.png",
            energies=layout.point_energies(ACTIVATION_POINT_ENERGY_RADIUS),
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
        energies = layout.point_energies(ACTIVATION_POINT_ENERGY_RADIUS)
        samples = _collect_samples(
            dataset,
            self._activation_samples,
            self.encoder,
            self.extractor,
            label=self.label,
        )
        engine, size = _calibrated_activation_engine(
            layout,
            energies,
            [codes for _, codes in samples[:ACTIVATION_OUTPUT_CALIBRATION_SAMPLES]],
        )
        self._activation_metadata = _activation_metadata(size)
        embeddings = list(initial_embeddings[: len(samples)])
        start_index = len(embeddings)
        LOGGER.event(
            "activation.resume",
            section=OUTSIDE_ARTICLE,
            data={
                "completed": start_index,
                "target": len(samples),
                "workers": ACTIVATION_PARALLEL_WORKERS,
                "batch_size": ACTIVATION_BATCH_SIZE,
            },
        )
        for batch_start in range(start_index, len(samples), ACTIVATION_BATCH_SIZE):
            batch = samples[batch_start : batch_start + ACTIVATION_BATCH_SIZE]
            results = engine.activate_many(
                [codes for _, codes in batch],
                max_workers=ACTIVATION_PARALLEL_WORKERS,
            )
            for offset, ((label, codes), result) in enumerate(zip(batch, results)):
                index = batch_start + offset
                LOGGER.event(
                    "activation.sample",
                    section=STIMULUS_DETECTION,
                    data={
                        "index": index,
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
        self._embedding_store.save(layout_path, embeddings, metadata=self._activation_metadata)

    def run(self) -> None:
        if self._reuse_cache:
            cached_layout = self._try_load_cached_layout()
        else:
            cached_layout = None
            LOGGER.event(
                "layout.cache.rebuild",
                section=OUTSIDE_ARTICLE,
                data={"count": self.count, "note": "--rebuild: готовый кеш игнорируется, новый записывается поверх"},
            )
        dataset = self._build_dataset()
        if cached_layout is not None:
            layout, tuned_params, total_codes, layout_path = cached_layout
            angles = len({float(hue) for hue in layout.hues()})
            self._log_dataset(angles, total_codes)
            LOGGER.event(
                "mnist.encoding.skip",
                section=OUTSIDE_ARTICLE,
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
            if not self._reuse_cache:
                self._checkpoint_store.reset(layout_path, target_samples=self._activation_samples)
            encoding_checkpoint = self._checkpoint_store.load(layout_path)
            encoding_stages = dict(encoding_checkpoint.get("stages", {}))
            encoding_stages["encoding"] = True
            self._checkpoint_store.save(
                layout_path,
                stages=encoding_stages,
                target_samples=self._activation_samples,
                embeddings=self._checkpoint_store.records(encoding_checkpoint),
            )
            layout = _run_layout(
                codes,
                tuned_params,
                count=self.count,
                total_codes=total_codes,
                colorizer=self.colorizer,
                expected_code_length=self.encoder.code_length,
                use_cache=self._reuse_cache,
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
            cached_embeddings = self._embedding_store.load(layout_path)
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
                section=OUTSIDE_ARTICLE,
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
        self._export_layout(layout, tuned_params, layout_path=layout_path, save_json=save_json)
        stages["export"] = True
        self._checkpoint_store.save(
            layout_path,
            stages=stages,
            target_samples=self._activation_samples,
            embeddings=embeddings,
        )



class MnistDecodeRunner:
    def __init__(self, *, count: int, layout_path: str | None = None) -> None:
        self.count = count
        self.layout_path = layout_path
        self._decoder: MnistDecoder | None = None
        self._temperature_override: float | None = None
        configure_logging()
        GPU_ENVIRONMENT.apply()
        self.encoder = _build_encoder()
        self.colorizer = _build_colorizer()
        self.extractor = _build_extractor()
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
            metadata = LayoutCacheFile.inspect(path)
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
            layout = LayoutCacheFile.load(
                path,
                tuned_params,
                expected_code_length=self.encoder.code_length,
                colorizer=self.colorizer,
                expected_encoder_signature=self.encoder.signature,
            )
            if layout is None:
                continue
            LOGGER.event(
                "decoder.layout.use",
                section=OUTSIDE_ARTICLE,
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
        energies = layout.point_energies(ACTIVATION_POINT_ENERGY_RADIUS)
        embeddings = self._embedding_store.load(layout_path)
        metadata = self._embedding_store.load_metadata(layout_path)
        activation_engine: ActivationEngine | None = None
        if not embeddings or "output_bits" not in metadata:
            embeddings, activation_engine = self._build_embeddings_for_layout(
                layout,
                layout_path,
                energies,
            )
        labeled_embeddings = EmbeddingStore.to_labeled(embeddings)
        if not labeled_embeddings:
            raise RuntimeError("в раскладке нет базы эмбеддингов")
        if activation_engine is None:
            activation_engine = _activation_engine_from_size(
                layout,
                energies,
                output_bits=int(metadata["output_bits"]),
                saturation_limit=int(metadata["saturation_limit"]),
            )
        database = EmbeddingDatabase(
            labeled_embeddings,
            DecoderConfig(
                similarity=DECODER_SIMILARITY,
                temperature=DECODER_SOFTMAX_TEMPERATURE,
                min_confidence=DECODER_MIN_CONFIDENCE,
                expected_classes=DECODER_EXPECTED_CLASSES,
                knn_top_k=DECODER_KNN_TOP_K,
                top_k_per_class=DECODER_TOP_K_PER_CLASS,
                min_similarity=DECODER_MIN_SIMILARITY,
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
            section=OUTSIDE_ARTICLE,
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
    ) -> tuple[list[EmbeddingRecord], ActivationEngine]:
        dataset = self._build_dataset(train=True)
        samples = _collect_samples(
            dataset,
            DECODER_DATASET_LIMIT,
            self.encoder,
            self.extractor,
            label=None,
        )
        engine, size = _calibrated_activation_engine(
            layout,
            energies,
            [codes for _, codes in samples[:ACTIVATION_OUTPUT_CALIBRATION_SAMPLES]],
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
                section=STIMULUS_DETECTION,
                data={
                    "index": idx,
                    "label": int(label),
                    "stimuli": len(codes),
                    "bits": result.embedding.count(),
                },
            )
        self._embedding_store.save(layout_path, embeddings, metadata=_activation_metadata(size))
        LOGGER.event(
            "decoder.base.generated",
            section=STIMULUS_DETECTION,
            data={
                "embeddings": len(embeddings),
                "classes": len({int(item.class_label) for item in embeddings}),
                "path": layout_path,
            },
        )
        return embeddings, engine

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
        image = MnistSobelAngleMap.load_image(path)
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
            section=OUTSIDE_ARTICLE,
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
                section=OUTSIDE_ARTICLE,
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
            section=OUTSIDE_ARTICLE,
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
    layout_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="игнорировать готовую раскладку и эмбеддинги в data/: посчитать заново и записать новый кеш "
        "(ход раскладки виден в Rerun)",
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
            reuse_cache=not args.rebuild,
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
