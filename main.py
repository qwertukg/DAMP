import concurrent.futures
import glob
import hashlib
import math
import time
from dataclasses import dataclass
import json
import os
from collections import OrderedDict
from typing import ClassVar, Mapping, Sequence
import argparse

from damp.article_refs import (
    DETECTOR_CONSTRUCTION,
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
from damp.activation.plane_detection import (
    BestMatchActivation,
    PlaneDetection,
    PlaneDetectorSpace,
    StimulusSetCodes,
)
from damp.MnistSobelAngleMap import MnistSobelAngleMap, UpscaledSobel
from damp.encoding.colour_merge import ColourMerge
from damp.encoding.damp_encoder import (
    ClosedDimension,
    Encoder,
    EncoderCodeColorizer,
    OpenedDimension,
    PlaneDimension,
    geometric_detector_layers,
    plane_detector_layers,
)
import numpy as np

from damp.layout.code_table import LayoutCodeTable
from damp.layout.damp_layout import AdaptiveLayoutConfig, Layout, ShortLayoutOptimConfig
from damp.layout.payload import LayoutEncodedCode, LayoutPayloadBuilder
from damp.layout.seeded import SeededPlacement
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
# Выборка пар ближней фазы по плиткам (§5.10): событие на каждую плитку каждого шага — раз в столько вызовов.
LOG_INTERVAL_LAYOUT_SHORT_SAMPLING = 500
LOG_INTERVAL_LAYOUT_SHORT_TILES = 50
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
LOG_INTERVAL_SECOND_LEVEL_IMAGE = 100
LOG_INTERVAL_SECOND_LEVEL_VISUAL = 25

EMBEDDINGS_JSON_KEY = "embeddings"
EMBEDDINGS_VERSION_KEY = "embeddings_version"
EMBEDDINGS_VERSION = 5
ACTIVATION_METADATA_KEY = "activation"
# Детекторы иерархии (§6.3) сохраняются вместе с эмбеддингами: evaluate/predict не строят их заново.
ACTIVATION_DETECTORS_KEY = "detectors"
ENCODER_SIGNATURE_KEY = "encoder_signature"
# Как получены стимулы раскладки (окно Собеля и режим углопозиций). В старых файлах этих ключей нет —
# они собраны окнами 4×4 по среднему градиенту окна; кеш с другими значениями не подхватывается.
PATCH_SIZE_KEY = "patch_size"
STIMULUS_MODE_KEY = "stimulus_mode"
LAYOUT_CACHE_LEGACY_STIMULUS = {PATCH_SIZE_KEY: 4, STIMULUS_MODE_KEY: "window"}
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
# Сторона окна Собеля, по которому берётся углопозиция (вне статьи), — в пикселях картинки, на которой
# считается Собель: стимула 28×28 или, при MNIST_SOBEL_UPSCALE_ENABLED, увеличенной (28·MNIST_SOBEL_UPSCALE).
# Сетка окон — сторона // окно (без увеличения: 4 → 7×7, 3 → 9×9, 2 → 14×14; при ×8: 16 → 14×14, 3 → 74×74).
# Остаток у правого и нижнего края в окна не попадает. При 3 ядро 3×3 совпадает с окном, меньше 3 — захватывает
# соседей. X, Y углопозиции — индексы окон 0…сетка−1: диапазон MNIST_POSITION_MIN…MAX должен их покрывать.
# После смены пересоберите раскладку: python main.py layout --rebuild (кеш на другом окне не подхватится).
MNIST_PATCH_SIZE = 9   # при ×8: окно 9×9 увеличенных пикселей, сетка 224 // 9 = 24×24, X, Y ∈ 0…23
# True — углопозиции только по контуру цифры: окно берётся, если через него проходит контур (пиксель штриха
# ≥ MNIST_CONTOUR_LEVEL рядом с фоном), угол — по градиенту в пикселях контура. Залитая середина штриха и фон
# пропускаются, контур не рвётся на границах окон. False — прежний режим: средний градиент по всему окну.
MNIST_CONTOUR_ONLY = True
MNIST_CONTOUR_LEVEL = 0.5
# Окно даёт ровно одну углопозицию. Если края в окне смотрят в противоположные стороны (|Σg| / Σ|g| ниже
# порога — тонкий штрих целиком в окне), их градиенты гасили бы друг друга — берётся сильнейший край.
# 0.5 — наименьшая ошибка ориентации линий относительно контура на 1000 цифрах MNIST (0.3: 9.79°, 0.5: 9.58°,
# 0.7: 9.61°, 0.9: 9.63°). Линия толщиной в пиксель (градиент в её середине гасится) берётся по окну с кольцом
# соседних пикселей — на 3000 цифрах ни одного окна контура без углопозиции.
MNIST_CONTOUR_TWO_EDGE_COHERENCE = 0.5
# Предобработка перед Собелем как в process_image (Rust): увеличение ×MNIST_SOBEL_UPSCALE фильтром Lanczos3,
# контраст MNIST_SOBEL_CONTRAST (image::imageops::contrast: отклонение от середины × ((100 + c) / 100)²),
# Собель в целых числах. Контур и градиент ищутся на увеличенной картинке, окно MNIST_PATCH_SIZE — в её пикселях.
# Отсечка ±255 из оригинала (u8-каналы картинки) — только у картинок gx, gy в Rerun: углы считаются по полному
# отклику, иначе края с наклоном ~15…75° слипаются в 45° (на 1000 цифрах ошибка ориентации 13.6° против 8.9°).
# Увеличенная картинка и отклики gx, gy (красный — плюс, синий — минус) — в Rerun, MNIST_SOBEL_ENTITY_PATH.
# False — без предобработки; в digit_sketch.py и digits_contour_test.py переопределяется ключом --upscale.
MNIST_SOBEL_UPSCALE_ENABLED = True
MNIST_SOBEL_UPSCALE = 8
MNIST_SOBEL_CONTRAST = 100.0
MNIST_SOBEL_ENTITY_PATH = "/sobel"
# Кодирование цифр рабочими процессами (вне статьи: подготовка кодов большого набора). Набор, в котором
# не хватает от стольких цифр, кодируется процессами; меньший — в основном процессе, с записями журнала кодера.
ENCODING_PARALLEL_MIN_DIGITS = 2000
# Число рабочих процессов; None — по числу ядер без одного, не больше ENCODING_WORKERS_MAX.
ENCODING_WORKERS = None
ENCODING_WORKERS_MAX = 12
# Цифр в одном задании рабочего процесса.
ENCODING_CHUNK_DIGITS = 200
# Коды цифр большого набора сохраняются на диск (вне статьи): повторный запуск с тем же кодером берёт их
# из файла data/codes-<набор>-<отпечаток>.npz и не кодирует набор заново (60 000 цифр — 2 минуты). Отпечаток —
# всё, от чего зависят углопозиции и коды: подпись кодера, константы MNIST_* и ENCODER_* этого файла, тексты
# модулей кодирования и извлечения стимулов; при любом их изменении файл не подходит и набор кодируется
# заново. Файл пишется и читается для наборов от ENCODING_CODES_CACHE_MIN_DIGITS цифр; ключ --rebuild
# относится к раскладке и на него не влияет. False — набор кодируется при каждом запуске.
ENCODING_CODES_CACHE = True
ENCODING_CODES_CACHE_MIN_DIGITS = 2000
ENCODING_CODES_CACHE_DIR = "data"
MNIST_ANGLE_MIN = 0.0
MNIST_ANGLE_MAX = 360.0
MNIST_ANGLE_CODE_LENGTH = 256
# Слоёв и перекрытия у угла и позиции поровну (10 слоёв, перекрытие 0.7): в режиме статьи
# (LAYOUT_SMOOTH_PINWHEELS = False) соседних ячеек, разнящихся по углу больше чем на 30° или по позиции больше
# чем на 3 окна, втрое меньше, чем при прежних 22 слоях и перекрытии 0.4 (300 цифр: 19.1 % → 5.5 %,
# 60 000 цифр: 1.21 % → 0.37 %). Перекрытие убирает стенки между областями, отношение слоёв угла и позиции
# решает, какая координата выходит непрерывной (§4.2.1, §4.5, §5.13.2.2). Цена: различных кодов на
# 60 000 цифр 256 тыс. вместо 456 тыс. Прежние значения оставлены в закомментированных строках.
# MNIST_ANGLE_LAYERS = 22
MNIST_ANGLE_LAYERS = 10
MNIST_ANGLE_WIDEST_DETECTORS = 2
MNIST_ANGLE_NARROWEST_DETECTORS = 180
# MNIST_ANGLE_OVERLAP = 0.4
MNIST_ANGLE_OVERLAP = 0.7
# Позиции MNIST — индексы участков 0…6. Нижняя граница вынесена на шаг за край данных: детекторов за краями
# отрезка нет (MNIST_POSITION_EDGE_DETECTORS = False), и при MIN = 0 участок 0 лежал на самой границе —
# 22 бита вместо ~34 и J(0, 1) = 0,81 при ~0,57 у остальных соседей. С MIN = -1 перекрытие соседних
# участков ровное (J = 0,55…0,61), их коды различимы одинаково (§4.2.1, §4.5).
MNIST_POSITION_MIN = -1.0
MNIST_POSITION_MAX = 28.0
MNIST_POSITION_CODE_LENGTH = 128
# 15 слоёв — минимальное уменьшение с 22, при котором в pinwheel-тесте (позиции 2×2, режим сглаживания)
# пропадает дыра на стыке четырёх позиций: соседние позиции достаточно похожи, чтобы точки на стыке
# получали энергию от чужих кластеров (22 слоя: 3–10 пустых ячеек в центре, 16: 0–5, 15: 1–2).
# MNIST_POSITION_LAYERS = 22
MNIST_POSITION_LAYERS = 10
MNIST_POSITION_WIDEST_DETECTORS = 2
# Самый узкий слой позиции — 28 детекторов (самый широкий — MNIST_POSITION_WIDEST_DETECTORS = 2);
# между ними слои растут геометрически, у каждого слоя своё число детекторов.
MNIST_POSITION_NARROWEST_DETECTORS = 28
# Перекрытие соседних позиционных детекторов слоя (§4.2.1: «detectors must overlap at each level»).
# MNIST_POSITION_OVERLAP = 0.4
MNIST_POSITION_OVERLAP = 0.7
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
LAYOUT_EMPTY_RATIO = 0.50
LAYOUT_USE_GPU = True
LAYOUT_RANDOM_SEED = 0
# Дальняя энергия по всему V (§5.5.1), пока точек не больше этого числа; выше — подмножество B (§5.9.4).
# В режиме статьи (жёсткий порог, триггер 1 % §7.1.7) подмножество |B| = 745 на 8 693 точках давало шумную Δ:
# ~1 % случайных пар всегда «выгодны», обмены не падают до 1 % от стартовых, триггер не срабатывает,
# λ = 0.65 и r = d/2 все 50 000 шагов — расписание доделывала ближняя фаза. С полным V на 100 цифрах
# расписание завершается к шагу ~18 000, ближняя фаза только полирует (подбор 2026-11-10).
# Цена плотного прохода: шаг стоит p·N·512 произведений (на 88k точек ≈ 1.1 с/шаг); при жёстком пороге шаг
# считается по ненулевым sim_λ (LAYOUT_SPARSE_LONG) с тем же результатом.
LAYOUT_LONG_FULL_SPACE_MAX_POINTS = 100000
# Подмножество B в гауссовой дальней фазе допустимо, пока в окрестность радиуса r(t) ожидаемо попадает
# не меньше стольких точек B (иначе оценка локальной энергии шумная и обмены случайны); ниже — всё V.
# На 100 цифрах (8 693 точки, |B| = 745) подмножество на всём протяжении фазы давало Ē 0.26 вместо 0.70.
LAYOUT_LONG_SUBSET_MIN_NEIGHBOURS = 50
# Энергии по ненулевым sim_λ (§5.5.1, §5.5.2, §5.7, §5.9.5). При жёстком пороге (η = ∞, §7.1.7) sim_λ(a, v) = 0
# для всех v с sim < λ, поэтому суммы по всему V, по окрестности пары и энергии точек сводятся к точкам,
# похожим на a или b: строки сходства строятся один раз, прореживаются при росте λ, шаг стоит O(p·k)
# вместо O(p·N) (k — число похожих точек). Суммы φ_c, φ_s побитово те же, что у исходных проходов (та же
# редукция float32 той же формы), поэтому раскладка совпадает шаг в шаг (docs/ARTICLE_COMPLIANCE.md,
# «Ускорение без изменения результата»). False — всегда исходные проходы по всему V и по окну ячеек.
LAYOUT_SPARSE_LONG = True
# Наибольшее число ненулевых элементов строк сходства (8 байт на элемент); больше — плотный проход.
LAYOUT_SPARSE_MAX_EDGES = 400_000_000
# Доля свободной памяти устройства, которую могут занять строки сходства.
LAYOUT_SPARSE_MEMORY_FRACTION = 0.5
# Память под буферы плотной редукции (участки [порция пар, N] для φ_c и φ_s).
LAYOUT_SPARSE_BUFFER_BYTES = 256 * 1024 * 1024
# Предел элементов (строки точек × наибольшая степень) на один блок расчёта слагаемых.
LAYOUT_SPARSE_ENTRY_BUDGET = 4_000_000
# С этого числа порций плотного прохода на шаг пары сначала отсеиваются по оценке ошибки суммирования
# float32: знак φ_c − φ_s известен без плотной редукции, она остаётся только парам с разностью у нуля.
LAYOUT_SPARSE_SCREEN_MIN_CHUNKS = 1
# Сверка с плотным проходом: после построения и каждого прореживания строк и затем раз в столько шагов
# шаг считается обоими способами (в раскладку идёт результат исходного прохода); при расхождении расчёт
# по строкам отключается. 0 — сверка только после построения и прореживания.
LAYOUT_SPARSE_VERIFY_EVERY = 2000
# Сверка пар шага идёт на стольких первых порциях исходного прохода (порция плотного прохода — 4 млн элементов
# «пары × точки»): форма порции та же, что в проходе по всем парам, а цена сверки не растёт с числом пар
# (на 59 тыс. точек проход по всем парам шага — 0.5 с). 0 — сверка на всех парах шага.
LAYOUT_SPARSE_VERIFY_CHUNKS = 1
# Знак φ_c − φ_s пар шага (§5.6) одним проходом ядра устройства по строкам сходства (§5.12): поток на отрезок
# строки из LAYOUT_PAIR_KERNEL_SEGMENT слагаемых, частичные суммы float32, решение — по оценке ошибки
# суммирования. Пары, у которых разность сумм лежит в пределах этой ошибки (единицы на шаг), считаются
# редукцией исходного прохода, поэтому обмены те же шаг в шаг. False — знак по суммам блоков строк torch.
LAYOUT_PAIR_KERNEL = True
LAYOUT_PAIR_KERNEL_SEGMENT = 16
# Выборка пар шага (§5.3, §5.9.1, §5.9.3) скомпилированными циклами numba на том же генераторе numpy: те же
# запросы в том же порядке — те же пары и то же состояние генератора. Сверка с выборкой numpy — на первых
# вызовах и затем раз в LAYOUT_SAMPLER_VERIFY_EVERY вызовов (0 — только на первых); при расхождении и без
# numba пары выбирает numpy.
LAYOUT_COMPILED_SAMPLER = True
LAYOUT_SAMPLER_VERIFY_EVERY = 1000
# Отсев пар дальней фазы по нижней границе потенциала точки (§5.5.1, §5.9.4 — по духу: агрегаты вместо всех точек,
# но без приближения): похожие точки каждой точки сведены в группы по ячейкам грубой сетки
# LAYOUT_POTENTIAL_BOUNDS_GRID × LAYOUT_POTENTIAL_BOUNDS_GRID (масса и центр масс группы). По неравенству Йенсена
# U_a(p) ≥ Σ_групп M·‖p − центр‖; если из этого следует φ_c < φ_s с запасом на ошибку суммирования float32,
# обмена нет (§5.6) и строки сходства пары не считаются. В разложенном пространстве так снимается 94–98 % пар
# шага; остальные считаются как раньше, поэтому обмены те же шаг в шаг. False — все пары считаются по строкам.
LAYOUT_POTENTIAL_BOUNDS = True
LAYOUT_POTENTIAL_BOUNDS_GRID = 8
# Пары, у которых φ_c − φ_s лежит в пределах ошибки суммирования float32 (единицы на шаг), решает само число
# редукции исходного прохода. True — оно считается на хосте в том же порядке сложения, что у редукции устройства
# (порядок сверяется с устройством для каждой формы порции и затем периодически): без загрузки, запуска ядер
# и выгрузки на каждом таком шаге. Для этого строки сходства копируются на хост, пока занимают не больше
# LAYOUT_REPLICA_MAX_BYTES (8 байт на элемент; 59 тыс. точек при λ = 0.65 — 0.73 ГБ). False или больший объём —
# такие пары считает устройство.
LAYOUT_REDUCTION_REPLICA = True
LAYOUT_REPLICA_MAX_BYTES = 1 << 30
# Слитый расчёт энергий (§5.5, §5.7, §5.9.5, §5.12; нужен torch с CUDA). Исходный проход строит слагаемые
# φ_c, φ_s полусотней операций torch на порцию пар — время шага определяет число проходов по памяти
# устройства. Слитый расчёт берёт сходство кодов из таблицы, предвычисленной один раз (§5.9.5: сходство
# от расстановки не зависит), а всё слагаемое считает одним ядром на элемент (пара, точка) теми же
# операциями float32 в том же порядке; суммирует та же редукция той же формы. φ_c, φ_s побитово те же,
# раскладка совпадает шаг в шаг в обоих режимах (сигмоида и жёсткий порог) — docs/ARTICLE_COMPLIANCE.md,
# «Слитое ядро энергий». Заодно шаг раскладки идёт по массивам (отбор и применение обменов без обхода
# пар в Python). False — всегда исходный проход.
LAYOUT_FUSED_ENERGY = True
# Таблицы сходства слитого расчёта (§5.9.5), по убыванию скорости: значения sim_λ всех пар точек (float32,
# 4 байта на пару; строится заново при смене λ, §7.1.7) → число общих бит кодов (uint8, байт на пару точек или
# на пару различных кодов; от λ и η не зависит) → общие биты на лету. Берётся первая, которая помещается
# в LAYOUT_FUSED_TABLE_MAX_BYTES и в долю LAYOUT_FUSED_TABLE_MEMORY_FRACTION свободной памяти устройства.
# 100 цифр (8 693 точки): sim_λ — 302 МБ; 1000 цифр (88 317 точек, 67 378 различных кодов): общие биты — 4.5 ГБ.
LAYOUT_FUSED_SIMILARITY_TABLE = True
LAYOUT_FUSED_TABLE_MAX_BYTES = 6 * 1024 * 1024 * 1024
LAYOUT_FUSED_TABLE_MEMORY_FRACTION = 0.7
# Предел элементов (пары × точки) на один блок слитого ядра: ограничивает память под слагаемые.
LAYOUT_FUSED_BLOCK_ELEMENTS = 40_000_000
# Сверка с исходным проходом: при первом расчёте каждого вида (по всему V, по окну ячеек; ядро гаусса,
# 1/d, d) и затем раз в столько расчётов шаг считается обоими способами (в раскладку идёт результат
# исходного прохода); при расхождении слитый расчёт отключается. 0 — сверка только при первом расчёте.
LAYOUT_FUSED_VERIFY_EVERY = 2000

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
# Отдельный порог остановки ближней фазы (вне статьи): средняя доля обменов за LAYOUT_MIN_SWAP_WINDOW шагов,
# ниже которой фаза переходит к следующему этапу расписания радиуса, а на последнем — останавливается.
# С общим 0.0001 (меньше 0.2 обмена на шаг) ближняя фаза почти всегда идёт все шаги, хотя уже почти ничего не меняет:
# 100 цифр MNIST (окно 9, ×8, 8 693 точки) — доля обменов всю фазу 0.002…0.006, Ē после дальней фазы 0.69778;
# порог 0.005 → 20 шагов, 14 с, Ē 0.69787; 0.003 → 71 шаг, 45 с, Ē 0.69794; 0.0001 → 160 шагов, 93 с, Ē 0.69393.
# При LAYOUT_SMOOTH_PINWHEELS = False (расписание "swaps") порог сначала переводит фазу к следующему этапу r и λ.
LAYOUT_SHORT_MIN_SWAP_RATIO = 0.005

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

# --- Второй уровень: детекторы-круги над первой раскладкой (§4.3, §6.2, §6.5, §6.6; расширение статьи).
# Раскладка заполняется кругами всех слоёв равномерно, как детекторы двумерной позиции (§4.3), а не по
# кластерам раскладки (§6.4). Слои, перекрытие, цветовые полосы и случайные биты — по тем же правилам,
# что у X и Y стимула (MNIST_POSITION_*); радиус круга задаётся долей стороны карты: от самого широкого
# слоя к самому узкому, между ними число кругов по стороне растёт геометрически (§4.2, §4.5).
# Стимулы набора (все углопозиции одной картинки) зажигают самые похожие точки первой раскладки, зажжённые
# точки активируют круги, биты активных кругов объединяются в код набора (§2.2.1); коды наборов
# раскладываются тем же Layout (§6.2, §6.6).
SECOND_LEVEL_CODE_LENGTH = 16384
SECOND_LEVEL_LAYERS = MNIST_POSITION_LAYERS
SECOND_LEVEL_OVERLAP = MNIST_POSITION_OVERLAP
# 0.85 стороны — два круга по стороне, как MNIST_POSITION_WIDEST_DETECTORS = 2: самый широкий слой
# накрывает всю карту и даёт сходство кодам любых двух позиций (дальний порядок, §4.5).
SECOND_LEVEL_RADIUS_MAX = 0.85
SECOND_LEVEL_RADIUS_MIN = 0.012
SECOND_LEVEL_RANDOM_SEED = ENCODER_RANDOM_SEED
# Стимул зажигает точки с наибольшим сходством, если оно не ниже λ_a (§6.1); порог тот же, что у активации §6.5.
SECOND_LEVEL_ACTIVATION_LAMBDA = ACTIVATION_LAMBDA_THRESHOLD
# §2: плотность кода не выше 25 %. Если единиц в объединении больше, биты отбираются colour merge (§3.3):
# "short" сохраняет коротковолновые (старшие) биты узких слоёв, "long" — длинноволновые (§3.3.2).
SECOND_LEVEL_MAX_DENSITY = 0.25
SECOND_LEVEL_MERGE_KEEP = "short"
# Подпись и цвет точки второй раскладки — класс цифры: hue = класс · шаг (10 классов на круге 360°).
SECOND_LEVEL_CLASS_HUE_STEP = 36.0
# Число тестовых картинок проверки нечётким поиском по кодам второго уровня (§2.2.5).
SECOND_LEVEL_EVALUATE_LIMIT = 200
SECOND_LEVEL_FILE_PREFIX = "data/second"

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
# Большие раскладки (N > LAYOUT_LONG_FULL_SPACE_MAX_POINTS, режим сглаживания). Пределы числа пар шага,
# подобранные до 100 тыс. точек, на миллионах точек дают единицы оценок обмена на точку вместо сотен;
# точные суммы по окрестности стоят N·r². Поэтому (§5.9.4, §5.10):
#  - пар за шаг — доля точек (та же, что у настройки на границе 100 тыс.: 20000/100000 и 12000/100000);
#  - на больших радиусах энергия пары считается по подмножеству B из LAYOUT_TUNE_LARGE_LONG_SUBSET точек,
#    обновляемому каждый шаг, пока в окрестность ожидаемо попадает ≥ LAYOUT_LONG_SUBSET_MIN_NEIGHBOURS точек B;
#  - дальше — по выборке ячеек окрестности: окно делится на LAYOUT_NEIGHBOURHOOD_SAMPLE_SIDE² слоёв,
#    из слоя берётся одна случайная ячейка (≈ 0.52·side² точек окрестности: при 12 — около 75);
#  - радиусы ближней фазы не растут с сеткой выше значений на 100 тыс. точек (0.08·G и 0.06·G при G = 388).
#  - глобальный порядок берётся из раскладки затравки — случайного подмножества из LAYOUT_SEED_POINTS точек,
#    разложенного целиком обычным алгоритмом; остальные точки ставятся возле самых похожих точек затравки,
#    и дальняя фаза идёт с радиуса LAYOUT_SEED_REFINE_RADIUS_FACTOR блоков затравки (блок — √(N / затравка)
#    ячеек), но не меньше LAYOUT_SEED_REFINE_RADIUS_MIN. None — без затравки: со случайной расстановки,
#    с подмножеством B и выборкой окрестности на всех радиусах (шум оценок на больших радиусах даёт разрывы).
LAYOUT_SEED_POINTS = 30000
# Режим статьи: затравка многоуровневая. На точку затравки очередного уровня приходится не больше
# LAYOUT_SEED_LEVEL_RATIO_MAX точек следующего: 30 тыс. → 400 тыс. → 5.3 млн вместо 30 тыс. → 5.3 млн.
# Ближняя фаза уровня начинается с радиуса в LAYOUT_SEED_SHORT_RADIUS_FACTOR блоков родителя (блок —
# √отношения ячеек, не меньше LAYOUT_SEED_REFINE_RADIUS_MIN): стоимость её первых этапов растёт как квадрат
# радиуса, поэтому два уровня по 13 точек на родителя на порядок дешевле одного уровня в 177 точек,
# а промежуточный уровень в 13 раз меньше полного. Измерено на 10 000 цифр с теми же отношениями уровней,
# что у 60 000 (Ē(3) при λ = 0.8 / сходство соседей / угол соседей): один уровень, радиус 27 — 0.857 / 0.928 /
# 4.23°; два уровня, радиус 2 блока — 0.868 / 0.940 / 3.73°; 3 блока — 0.871 / 0.941 / 3.61°; 4 блока —
# 0.875 / 0.943 / 3.48°; 6 блоков — 0.875 / 0.942 / 3.45° (дальше качество не растёт, время растёт).
LAYOUT_SEED_LEVEL_RATIO_MAX = 16
LAYOUT_SEED_SHORT_RADIUS_FACTOR = 4.0
# Порог ранней отсечки пар (§5.9.1) в ближней фазе уровня большой раскладки режима статьи; 0 — без отсечки.
# Отсечка sim(a, b) ≥ t оставляет в шаге только пары похожих точек: статья советует её «на поздних стадиях,
# когда двигать надо похожие точки». У точной раскладки непохожие точки разводит дальняя фаза; на уровнях
# выше затравки её нет, а точки соседних родителей после расстановки перемешаны на ширине блока — развести
# их и провести границы областей должна ближняя фаза, и с отсечкой она этого не может (пара точек из разных
# областей в шаг не попадает). Измерено на 10 000 цифр (уровни 5000 → 66 тыс. → 883 тыс.), Ē(3) при λ = 0.8:
# с отсечкой 0.65 — 0.853, без отсечки — 0.887 (при прежней расстановке ровно в блок родителя: 0.875 и 0.888).
LAYOUT_SEED_SHORT_SIMILARITY_CUTOFF = 0.0
# Файл раскладки больше этого размера как кеш не читается (вне статьи): чтение строит по объекту на точку,
# и на миллионах точек они не помещаются в память — раскладка считается заново. 256 МБ — около 380 тыс. точек.
LAYOUT_CACHE_MAX_FILE_BYTES = 256 * 1024 * 1024
#  - шаг дальней фазы идёт целиком на устройстве (LAYOUT_DEVICE_STEP): выбор пар, энергии, отбор и применение
#    обменов — те же действия, что на хосте, но случайные числа берёт генератор устройства. На миллионе пар
#    за шаг массивы numpy на хосте занимали половину времени шага.
LAYOUT_DEVICE_STEP = True
LAYOUT_SEED_REFINE_RADIUS_FACTOR = 2.0
LAYOUT_SEED_REFINE_RADIUS_MIN = 8
LAYOUT_TUNE_LARGE_LONG_PAIR_FACTOR = 0.2
LAYOUT_TUNE_LARGE_SHORT_PAIR_FACTOR = 0.12
LAYOUT_TUNE_LARGE_LONG_SUBSET = 256
LAYOUT_TUNE_LARGE_LONG_SUBSET_REFRESH = 1
LAYOUT_NEIGHBOURHOOD_SAMPLE_SIDE = 12
LAYOUT_TUNE_LARGE_SHORT_RADIUS_MAX = 31
LAYOUT_TUNE_LARGE_SHORT_ENERGY_RADIUS_MAX = 23
# Режим статьи, большие раскладки: радиус окрестности пары в ближней фазе — наименьший допустимый статьёй
# (§5.5.2: r ≥ ‖p1 − p2‖): r = max(⌈‖p1 − p2‖⌉ + 1, это значение); None — постоянный радиус фазы.
LAYOUT_TUNE_LARGE_SHORT_NEIGHBOURHOOD_MIN = 5
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

# --- Проверка, как цифра строится из углопозиций (digit_sketch.py): случайный стимул MNIST заданной цифры,
# его углопозиции (угол, X, Y) от того же MnistSobelAngleMap, что и в конвейере (§4.4, §4.6), и по каждой —
# отрезок длиной в окно Собеля, перпендикулярный градиенту (край штриха). Цвет отрезка — угол, как на рис. 16.
# Размер окна — MNIST_PATCH_SIZE, у скрипта его можно переопределить ключом --patch-size.
DIGIT_SKETCH_MNIST_ROOT = "./data"
DIGIT_SKETCH_SPLIT = "train"
DIGIT_SKETCH_LINE_SCALE = 1.0   # длина отрезка в сторонах окна: 1.0 — ровно окно
DIGIT_SKETCH_LINE_RADIUS_SCALE = 0.1   # толщина (радиус) отрезка в сторонах окна
# Окна, через которые проходит контур (MNIST_CONTOUR_LEVEL), но углопозиции нет, подсвечиваются
# DIGIT_SKETCH_LOST_COLOR — там контур цифры рвётся.
DIGIT_SKETCH_GRID_COLOR = (255, 255, 255, 50)
DIGIT_SKETCH_LOST_COLOR = (255, 60, 60, 220)
DIGIT_SKETCH_ENTITY_PATH = "/digit"

# --- Тест «цифры из контуров» (digits_contour_test.py): по случайному стимулу каждой цифры — строка таблицы
# «оригинал | цифра из линий углопозиций в режиме контура» для каждого окна из DIGITS_CONTOUR_PATCH_SIZES.
# Режим контура включается в тесте всегда, независимо от MNIST_CONTOUR_ONLY; seed фиксирован — таблица
# воспроизводится, ключ --seed выбирает другие стимулы. Цвета сетки и пропусков — DIGIT_SKETCH_*.
DIGITS_CONTOUR_DIGITS = tuple(range(10))
DIGITS_CONTOUR_PATCH_SIZES = (MNIST_PATCH_SIZE,)   # то же окно, что у конвейера
DIGITS_CONTOUR_SEED = 0
DIGITS_CONTOUR_SCALE = 16        # пикселей таблицы на пиксель стимула
DIGITS_CONTOUR_LINE_WIDTH = 4
DIGITS_CONTOUR_GAP = 10
DIGITS_CONTOUR_OUTPUT_PATH = "digits_contour_test.png"
DIGITS_CONTOUR_ENTITY_PATH = "/digits_contour"
LOG_INTERVAL_ANGLE_LINES = 100
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
    "layout.energy.pairs.gpu": LOG_INTERVAL_LAYOUT_ENERGY,
    "layout.sparse.energy.pairs": LOG_INTERVAL_LAYOUT_ENERGY,
    "layout.sparse.energy.neighbourhood": LOG_INTERVAL_LAYOUT_ENERGY,
    "layout.short.pair_sampling": LOG_INTERVAL_LAYOUT_SHORT_SAMPLING,
    "layout.short.tiles": LOG_INTERVAL_LAYOUT_SHORT_TILES,
    "layout.short.partition_pairs": LOG_INTERVAL_LAYOUT_SHORT_TILES,
    "layout.pair_sampling.merged_draws": LOG_INTERVAL_INIT,
    "layout.pair_sampling.compiled": LOG_INTERVAL_INIT,
    "layout.pair_sampling.compiled.unavailable": LOG_INTERVAL_INIT,
    "layout.pair_sampling.compiled.verify": LOG_INTERVAL_INIT,
    "layout.pair_sampling.compiled.mismatch": LOG_INTERVAL_INIT,
    "layout.pair_kernel.init": LOG_INTERVAL_INIT,
    "layout.pair_kernel.status": LOG_INTERVAL_LAYOUT_ENERGY,
    "layout.pair_kernel.calibration": LOG_INTERVAL_INIT,
    "layout.pair_kernel.reference.verify": LOG_INTERVAL_INIT,
    "layout.replica.init": LOG_INTERVAL_INIT,
    "layout.replica.calibration": LOG_INTERVAL_INIT,
    "layout.replica.rows": LOG_INTERVAL_INIT,
    "layout.replica.rows.skipped": LOG_INTERVAL_INIT,
    "layout.replica.verify": LOG_INTERVAL_INIT,
    "layout.bounds.init": LOG_INTERVAL_INIT,
    "layout.bounds.rebuild": LOG_INTERVAL_INIT,
    "layout.bounds.disabled": LOG_INTERVAL_INIT,
    "layout.bounds.share": LOG_INTERVAL_LAYOUT_ADAPTIVE,
    "layout.codes.classes": LOG_INTERVAL_INIT,
    "layout.kernel.cache": LOG_INTERVAL_INIT,
    "layout.sparse.energy.init": LOG_INTERVAL_INIT,
    "layout.sparse.energy.verify": LOG_INTERVAL_INIT,
    "layout.sparse.energy.disabled": LOG_INTERVAL_INIT,
    "layout.sparse.rows.build": LOG_INTERVAL_INIT,
    "layout.sparse.rows.prune": LOG_INTERVAL_INIT,
    "layout.sparse.rows.skipped": LOG_INTERVAL_INIT,
    "layout.fused.table.build": LOG_INTERVAL_INIT,
    "layout.fused.table.skipped": LOG_INTERVAL_INIT,
    "layout.fused.similarity_table.build": LOG_INTERVAL_INIT,
    "layout.fused.packed_codes": LOG_INTERVAL_INIT,
    "layout.fused.energy.init": LOG_INTERVAL_INIT,
    "layout.fused.energy.verify": LOG_INTERVAL_INIT,
    "layout.fused.energy.disabled": LOG_INTERVAL_INIT,
    "layout.fused.energy.merge_disabled": LOG_INTERVAL_INIT,
    "layout.fused.kernel_cache": LOG_INTERVAL_INIT,
    "layout.swap.bulk_draws": LOG_INTERVAL_INIT,
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
    # Отбор обменов при отжиге идёт на каждом шаге дальней фазы: событие раз в столько шагов.
    "layout.swap.annealed": LOG_INTERVAL_LAYOUT_SWAP_RATIO,
    "layout.device_step": LOG_INTERVAL_LAYOUT_SWAP_RATIO,
    "layout.energy.neighbourhood_sample": LOG_INTERVAL_LAYOUT_SWAP_RATIO,
    "layout.pairs.beyond_radius": LOG_INTERVAL_LAYOUT_SWAP_RATIO,
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
    "sobel_map.contour": LOG_INTERVAL_SOBEL_IMAGE,
    "sobel_map.contour_patches": LOG_INTERVAL_SOBEL_IMAGE,
    "upscaled_sobel.init": LOG_INTERVAL_INIT,
    "upscaled_sobel.apply": LOG_INTERVAL_SOBEL_IMAGE,
    "angle_lines.angle_position": LOG_INTERVAL_ANGLE_LINES,
    "angle_lines.segment": LOG_INTERVAL_ANGLE_LINES,
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
    # Второй уровень: события на каждую картинку — раз в столько картинок.
    "encoder.encode.set": LOG_INTERVAL_SECOND_LEVEL_IMAGE,
    "colour_merge.merge": LOG_INTERVAL_SECOND_LEVEL_IMAGE,
    "plane_detection.activation": LOG_INTERVAL_SECOND_LEVEL_IMAGE,
    "plane_detection.detect": LOG_INTERVAL_SECOND_LEVEL_IMAGE,
    "plane_detection.code": LOG_INTERVAL_SECOND_LEVEL_IMAGE,
    "plane_detection.visual": LOG_INTERVAL_SECOND_LEVEL_VISUAL,
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
        neighbourhood_sample_side: int | None = None
        large_layout = total_points > LAYOUT_LONG_FULL_SPACE_MAX_POINTS
        article_large = large_layout and not LAYOUT_SMOOTH_PINWHEELS
        if article_large:
            # Большая раскладка в режиме статьи: глобальный порядок — из раскладки затравки (дальняя фаза §5.5.1
            # по всему V на миллионах точек неосуществима), ближний порядок — ближней фазой §5.5.2.
            long_subset_size = None
            short_pairs = max(short_pairs, int(total_points * LAYOUT_TUNE_LARGE_SHORT_PAIR_FACTOR))
            if LAYOUT_PAIRS_POINT_FRACTION is not None:
                short_pairs = min(short_pairs, max(1, int(total_points * LAYOUT_PAIRS_POINT_FRACTION)))
            short_local_radius = min(short_local_radius, LAYOUT_TUNE_LARGE_SHORT_RADIUS_MAX)
            short_energy_radius = min(short_energy_radius, LAYOUT_TUNE_LARGE_SHORT_ENERGY_RADIUS_MAX)
            # Ближняя фаза заканчивается по числу обменов (§5.6); бюджет шагов — только верхняя граница
            # (период пересчёта весов первой точки §5.9.3 остаётся прежним).
            short_steps = LAYOUT_TUNE_SHORT_STEPS_MAX
            LOGGER.event(
                "layout.tune.large.article",
                section=OPTIM_SUBSET,
                data={
                    "points": total_points,
                    "seed_points": -1 if LAYOUT_SEED_POINTS is None else LAYOUT_SEED_POINTS,
                    "seed_level_ratio_max": LAYOUT_SEED_LEVEL_RATIO_MAX,
                    "short_radius_factor": LAYOUT_SEED_SHORT_RADIUS_FACTOR,
                    "short_pairs_per_step": short_pairs,
                    "short_steps": short_steps,
                    "short_local_radius": short_local_radius,
                    "short_energy_radius": short_energy_radius,
                    "neighbourhood_radius_min": LAYOUT_TUNE_LARGE_SHORT_NEIGHBOURHOOD_MIN,
                    "note": "большая раскладка режима статьи: дальний порядок — из точной раскладки затравки "
                    "(подмножество, §5.9.4), уровни затравки, на каждом — ближняя фаза §5.5.2 на устройстве; "
                    "пар за шаг — доля точек (§5.10)",
                },
            )
        elif large_layout:
            # Большая раскладка: пар за шаг — доля точек (§5.10), энергии — по подмножеству B и по выборке
            # ячеек окрестности (§5.9.4), радиусы ближней фазы — не выше значений на границе режима.
            long_pairs = max(long_pairs, int(total_points * LAYOUT_TUNE_LARGE_LONG_PAIR_FACTOR))
            short_pairs = max(short_pairs, int(total_points * LAYOUT_TUNE_LARGE_SHORT_PAIR_FACTOR))
            if LAYOUT_SEED_POINTS is not None:
                # Глобальный порядок — из раскладки затравки: больших радиусов и подмножества B нет.
                long_subset_size = None
            else:
                long_subset_size = LAYOUT_TUNE_LARGE_LONG_SUBSET
                long_subset_refresh = LAYOUT_TUNE_LARGE_LONG_SUBSET_REFRESH
            neighbourhood_sample_side = LAYOUT_NEIGHBOURHOOD_SAMPLE_SIDE
            short_local_radius = min(short_local_radius, LAYOUT_TUNE_LARGE_SHORT_RADIUS_MAX)
            short_energy_radius = min(short_energy_radius, LAYOUT_TUNE_LARGE_SHORT_ENERGY_RADIUS_MAX)
            LOGGER.event(
                "layout.tune.large",
                section=OPTIM_SUBSET,
                data={
                    "points": total_points,
                    "long_pairs_per_step": long_pairs,
                    "short_pairs_per_step": short_pairs,
                    "long_subset_size": -1 if long_subset_size is None else long_subset_size,
                    "long_subset_refresh": long_subset_refresh,
                    "seed_points": -1 if LAYOUT_SEED_POINTS is None else LAYOUT_SEED_POINTS,
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
            neighbourhood_sample_side=neighbourhood_sample_side,
            enforce_pair_radius=large_layout,
            neighbourhood_radius_min=LAYOUT_TUNE_LARGE_SHORT_NEIGHBOURHOOD_MIN if article_large else None,
            seed_points=LAYOUT_SEED_POINTS if large_layout else None,
            device_step=bool(LAYOUT_DEVICE_STEP) and large_layout and not article_large,
            device_short_step=bool(LAYOUT_DEVICE_STEP) and article_large,
        )


def _sparse_layout_options() -> dict[str, object]:
    """Параметры Layout для ускоренного расчёта энергий с тем же результатом — по ненулевым sim_λ при жёстком
    пороге и слитым ядром (§5.5.1, §5.5.2, §5.9.5, §5.12): общие для MNIST и pinwheel."""
    return {
        "sparse_long": LAYOUT_SPARSE_LONG,
        "sparse_max_edges": LAYOUT_SPARSE_MAX_EDGES,
        "sparse_memory_fraction": LAYOUT_SPARSE_MEMORY_FRACTION,
        "sparse_buffer_bytes": LAYOUT_SPARSE_BUFFER_BYTES,
        "sparse_entry_budget": LAYOUT_SPARSE_ENTRY_BUDGET,
        "sparse_screen_min_chunks": LAYOUT_SPARSE_SCREEN_MIN_CHUNKS,
        "sparse_verify_every": LAYOUT_SPARSE_VERIFY_EVERY,
        "sparse_verify_chunks": LAYOUT_SPARSE_VERIFY_CHUNKS,
        "pair_kernel": LAYOUT_PAIR_KERNEL,
        "pair_kernel_segment": LAYOUT_PAIR_KERNEL_SEGMENT,
        "compiled_sampler": LAYOUT_COMPILED_SAMPLER,
        "sampler_verify_every": LAYOUT_SAMPLER_VERIFY_EVERY,
        "potential_bounds": LAYOUT_POTENTIAL_BOUNDS,
        "potential_bounds_grid": LAYOUT_POTENTIAL_BOUNDS_GRID,
        "reduction_replica": LAYOUT_REDUCTION_REPLICA,
        "replica_max_bytes": LAYOUT_REPLICA_MAX_BYTES,
        "fused_energy": LAYOUT_FUSED_ENERGY,
        "fused_table_max_bytes": LAYOUT_FUSED_TABLE_MAX_BYTES,
        "fused_table_memory_fraction": LAYOUT_FUSED_TABLE_MEMORY_FRACTION,
        "fused_block_elements": LAYOUT_FUSED_BLOCK_ELEMENTS,
        "fused_verify_every": LAYOUT_FUSED_VERIFY_EVERY,
        "fused_similarity_table": LAYOUT_FUSED_SIMILARITY_TABLE,
    }


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


@dataclass(frozen=True)
class CalibratedActivation:
    """Движок после калибровки размера выхода и эмбеддинги калибровочных стимулов (§6.6)."""

    engine: ActivationEngine
    size: OutputCodeSize
    embeddings: tuple[BitArray, ...]
    active_counts: tuple[int, ...]


def _calibrated_activation_engine(
    layout: Layout,
    energies: list[float],
    calibration_batches: Sequence[Sequence[BitArray]],
) -> CalibratedActivation:
    """§6.6: строит детекторы, измеряет среднее число активных и задаёт размер выхода.

    Калибровочные стимулы активируются один раз: уровни детекторов не зависят от выходных кодов и σ
    (§6.5), поэтому после перевыдачи битов (§6.3) их эмбеддинги собираются из тех же уровней.
    """
    detectors = _build_detectors_from_layout(layout, energies, output_bits=ACTIVATION_OUTPUT_BITS)
    provisional = _build_activation_engine(layout, energies, detectors=detectors)
    size, active_lists = _output_sizer().measure_with_results(provisional, calibration_batches)
    resized = reassign_output_bits(detectors, output_bits=size.output_bits, seed=ACTIVATION_DETECTOR_SEED)
    engine = provisional.rebind(resized, saturation_limit=size.saturation_limit)
    embeddings = tuple(engine.embedding_from_active(active) for active in active_lists)
    return CalibratedActivation(
        engine=engine,
        size=size,
        embeddings=embeddings,
        active_counts=tuple(len(active) for active in active_lists),
    )


def _activation_engine_from_size(
    layout: Layout,
    energies: list[float],
    *,
    output_bits: int,
    saturation_limit: int,
    stored_detectors: Sequence[Mapping[str, object]] | None = None,
) -> ActivationEngine:
    """Восстанавливает детекторы и выходной код тех же размеров, что при обучении.

    Если детекторы сохранены вместе с эмбеддингами, они читаются из файла; иначе строятся заново
    с тем же seed (§6.4 — построение стохастическое, но воспроизводимое).
    """
    if stored_detectors:
        resized = _detectors_from_payload(stored_detectors, output_bits=output_bits)
        LOGGER.event(
            "activation.detectors.restored",
            section=DETECTOR_CONSTRUCTION,
            data={"detectors": len(resized), "output_bits": output_bits},
        )
    else:
        detectors = _build_detectors_from_layout(layout, energies, output_bits=ACTIVATION_OUTPUT_BITS)
        resized = reassign_output_bits(detectors, output_bits=output_bits, seed=ACTIVATION_DETECTOR_SEED)
    return _build_activation_engine(layout, energies, detectors=resized, saturation_limit=saturation_limit)


def _detectors_payload(detectors: Sequence[Detector]) -> list[dict[str, object]]:
    """Детекторы (§6.3) в JSON: центр, радиус, λ_d, e_d, n_d, единичные биты b_d."""
    return [
        {
            "center": [float(detector.center[0]), float(detector.center[1])],
            "radius": float(detector.radius),
            "lambda": float(detector.lambda_threshold),
            "energy": float(detector.energy),
            "points": int(detector.points_count),
            "bits": [int(bit) for bit in detector.output_code.active_indices()],
            "label": detector.label,
        }
        for detector in detectors
    ]


def _detectors_from_payload(items: Sequence[Mapping[str, object]], *, output_bits: int) -> list[Detector]:
    detectors: list[Detector] = []
    for item in items:
        center = item["center"]
        detectors.append(
            Detector(
                center=(float(center[0]), float(center[1])),
                radius=float(item["radius"]),
                lambda_threshold=float(item["lambda"]),
                energy=float(item["energy"]),
                points_count=int(item["points"]),
                output_code=BitArray.from_indices(output_bits, [int(bit) for bit in item["bits"]]),
                label=None if item.get("label") is None else str(item["label"]),
            )
        )
    return detectors


def _activation_metadata(size: OutputCodeSize, detectors: Sequence[Detector] | None = None) -> dict[str, object]:
    metadata: dict[str, object] = {
        "output_bits": int(size.output_bits),
        "saturation_limit": int(size.saturation_limit),
        "mean_active_detectors": float(size.mean_active_detectors),
        "calibration_samples": int(size.samples),
    }
    if detectors is not None:
        metadata[ACTIVATION_DETECTORS_KEY] = _detectors_payload(detectors)
    return metadata


class LayoutCacheFile:
    """JSON раскладки на диске: проверка совместимости и восстановление Layout (вне статьи)."""

    @staticmethod
    def inspect(path: str) -> tuple[int, int | None, bool] | None:
        try:
            size = os.path.getsize(path)
            if size > LAYOUT_CACHE_MAX_FILE_BYTES:
                LOGGER.event(
                    "layout.cache.too_large",
                    section=OUTSIDE_ARTICLE,
                    data={
                        "path": path,
                        "size_mb": size / 1e6,
                        "max_mb": LAYOUT_CACHE_MAX_FILE_BYTES / 1e6,
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
        for key, expected_value in (expected_stimulus or {}).items():
            stored_value = data.get(key, LAYOUT_CACHE_LEGACY_STIMULUS.get(key))
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


def _build_plane_encoder(layout: Layout) -> Encoder:
    """Кодер двумерной позиции на карте раскладки (§4.3): круги всех слоёв заполняют её сетку ячеек."""
    layers = plane_detector_layers(
        layers_count=SECOND_LEVEL_LAYERS,
        widest_radius=SECOND_LEVEL_RADIUS_MAX,
        narrowest_radius=SECOND_LEVEL_RADIUS_MIN,
        overlap=SECOND_LEVEL_OVERLAP,
    )
    encoder = Encoder(
        PlaneDimension(
            "Map",
            PlaneDetectorSpace.plane_size(layout),
            layers,
            code_length=SECOND_LEVEL_CODE_LENGTH,
        ),
        random_seed=SECOND_LEVEL_RANDOM_SEED,
        layer_phase=ENCODER_LAYER_PHASE,
    )
    # §4.5, §5.13.1: сходство кодов позиций должно убывать с расстоянием на карте на всём её протяжении;
    # Encoder.long_range_order сам логирует профиль и предупреждение.
    encoder.long_range_order("Map")
    return encoder


def _build_plane_detection(layout: Layout) -> PlaneDetection:
    """Детекторы-круги над раскладкой и зажигание её точек стимулами: код набора стимулов (§4.3, §6.2, §6.5)."""
    space = PlaneDetectorSpace(
        layout,
        _build_plane_encoder(layout),
        merge=ColourMerge(
            max(1, int(SECOND_LEVEL_CODE_LENGTH * SECOND_LEVEL_MAX_DENSITY)),
            keep=SECOND_LEVEL_MERGE_KEEP,
        ),
    )
    activation = BestMatchActivation(
        layout.code_table,
        similarity=LAYOUT_SIMILARITY,
        lambda_threshold=SECOND_LEVEL_ACTIVATION_LAMBDA,
        use_gpu=ACTIVATION_USE_GPU,
    )
    return PlaneDetection(activation, space)


def _build_extractor(
    patch_size: int = MNIST_PATCH_SIZE,
    *,
    contour_only: bool = MNIST_CONTOUR_ONLY,
    upscale: bool = MNIST_SOBEL_UPSCALE_ENABLED,
) -> MnistSobelAngleMap:
    upscaled = None
    if upscale:
        upscaled = UpscaledSobel(
            scale=MNIST_SOBEL_UPSCALE,
            contrast=MNIST_SOBEL_CONTRAST,
            entity_path=MNIST_SOBEL_ENTITY_PATH,
        )
    return MnistSobelAngleMap(
        angle_in_degrees=MNIST_ANGLE_IN_DEGREES,
        grad_threshold=MNIST_GRAD_THRESHOLD,
        patch_size=patch_size,
        contour_only=contour_only,
        contour_level=MNIST_CONTOUR_LEVEL,
        two_edge_coherence=MNIST_CONTOUR_TWO_EDGE_COHERENCE,
        upscaled=upscaled,
    )


def _stimulus_metadata(extractor: MnistSobelAngleMap) -> dict[str, object]:
    """Параметры получения стимулов, без совпадения которых кеш раскладки не подходит."""
    return {PATCH_SIZE_KEY: extractor.patch_size, STIMULUS_MODE_KEY: extractor.stimulus_mode}



@dataclass(frozen=True)
class DigitCodes:
    """Одна цифра набора: метка, углы её углопозиций и их коды (§4.4, §4.6).

    Коды хранятся матрицей бит, упакованной по 8 в байт (k × ⌈L/8⌉): объект BitArray на каждый код для
    больших наборов не помещается в память. codes строит объекты по требованию (активация, декодер).
    """

    label: int
    angles: tuple[float, ...]
    packed: "np.ndarray"
    code_length: int

    @property
    def count(self) -> int:
        """Число углопозиций цифры."""
        return len(self.angles)

    @property
    def codes(self) -> tuple[BitArray, ...]:
        bits = np.unpackbits(self.packed, axis=1, count=self.code_length)
        return tuple(
            BitArray.from_bits(bits[row].tobytes(), np.flatnonzero(bits[row]).tolist()) for row in range(bits.shape[0])
        )


class DigitEncodingWorker:
    """Кодирование цифр в рабочем процессе: свой кодер и своё извлечение углопозиций (вне статьи).

    Результат — те же углопозиции и коды, что в основном процессе (тот же код, те же константы); журнал
    рабочего процесса выключен, сводку ведёт основной процесс.
    """

    _encoder: ClassVar[Encoder | None] = None
    _extractor: ClassVar[MnistSobelAngleMap | None] = None

    @classmethod
    def initialize(cls) -> None:
        LOGGER.mute()
        cls._encoder = _build_encoder()
        cls._extractor = _build_extractor()

    @classmethod
    def encode(cls, task: list[tuple["np.ndarray", int]]) -> list[tuple[int, tuple[float, ...], "np.ndarray"]]:
        """Цифры задания (изображение, метка) → (метка, углы углопозиций, упакованные коды)."""
        if cls._encoder is None or cls._extractor is None:
            cls.initialize()
        result = []
        for img, digit in task:
            item = StimulusCodeCache.encode_digit(cls._encoder, cls._extractor, img, digit, log=False)
            result.append((item.label, item.angles, item.packed))
        return result


class EncodedCodesFile:
    """Коды цифр набора на диске (вне статьи): повторный запуск с тем же кодером не кодирует набор заново.

    Файл хранит первые D цифр набора в его порядке: метки, число углопозиций каждой цифры, углы и упакованные
    коды — те же байты, что дало кодирование. Имя файла содержит отпечаток всего, от чего зависят углопозиции
    (§4.4) и коды (§4.6): подпись кодера, значения констант MNIST_* и ENCODER_* main.py, тексты модулей
    кодирования и извлечения стимулов, часть набора и отбор по метке. Изменилось что-то из этого — имя другое,
    файл не находится, набор кодируется заново.
    """

    # Модули, от текста которых зависят углопозиции и коды.
    SOURCES = (
        os.path.join("damp", "MnistSobelAngleMap.py"),
        os.path.join("damp", "encoding", "bitarray.py"),
        os.path.join("damp", "encoding", "damp_encoder.py"),
        os.path.join("damp", "layout", "code_table.py"),
    )
    CONSTANT_PREFIXES = ("MNIST_", "ENCODER_")

    def __init__(self, directory: str, *, encoder: Encoder, split: str, label: int | None) -> None:
        self._code_length = int(encoder.code_length)
        self._fingerprint = self._compute_fingerprint(encoder, split, label)
        suffix = "" if label is None else f"-label{int(label)}"
        self._path = os.path.join(directory, f"codes-{split}{suffix}-{self._fingerprint[:16]}.npz")

    @classmethod
    def _compute_fingerprint(cls, encoder: Encoder, split: str, label: int | None) -> str:
        digest = hashlib.sha256()
        digest.update(f"{encoder.signature}|{encoder.code_length}|{split}|{label}".encode("utf-8"))
        constants = globals()
        for name in sorted(constants):
            if name.startswith(cls.CONSTANT_PREFIXES):
                digest.update(f"{name}={constants[name]!r};".encode("utf-8"))
        root = os.path.dirname(os.path.abspath(__file__))
        for source in cls.SOURCES:
            with open(os.path.join(root, source), "rb") as fp:
                digest.update(fp.read())
        return digest.hexdigest()

    @property
    def path(self) -> str:
        return self._path

    def load(self, digits: int) -> "list[DigitCodes] | None":
        """Первые digits цифр набора из файла; None — файла нет, он от другого кодера или цифр в нём меньше."""
        if not os.path.exists(self._path):
            return None
        started = time.perf_counter()
        try:
            with np.load(self._path, allow_pickle=False) as data:
                if str(data["fingerprint"]) != self._fingerprint or int(data["code_length"]) != self._code_length:
                    return None
                counts = data["counts"]
                if counts.size < digits:
                    return None
                counts = counts[:digits].astype(np.int64)
                total = int(counts.sum())
                labels = data["labels"][:digits].tolist()
                angles = data["angles"][:total]
                packed = data["packed"][:total]
        except Exception as exc:  # noqa: BLE001
            LOGGER.event(
                "mnist.encoding.cache.error",
                section=OUTSIDE_ARTICLE,
                data={"path": self._path, "error": str(exc), "stage": "read"},
            )
            return None
        stops = np.cumsum(counts).tolist()
        items: list[DigitCodes] = []
        start = 0
        for label, stop in zip(labels, stops):
            items.append(
                DigitCodes(
                    label=int(label),
                    angles=tuple(angles[start:stop].tolist()),
                    packed=packed[start:stop],
                    code_length=self._code_length,
                )
            )
            start = stop
        LOGGER.event(
            "mnist.encoding.cache.load",
            section=OUTSIDE_ARTICLE,
            data={
                "path": self._path,
                "digits": digits,
                "codes": total,
                "seconds": time.perf_counter() - started,
                "note": "углопозиции (§4.4) и коды (§4.6) цифр взяты из файла: кодер, константы MNIST_* и ENCODER_* "
                "и тексты модулей кодирования — те же, что при записи файла",
            },
        )
        return items

    def save(self, items: "Sequence[DigitCodes]") -> None:
        """Записать цифры набора (в его порядке), если в файле их меньше."""
        digits = len(items)
        try:
            if os.path.exists(self._path):
                with np.load(self._path, allow_pickle=False) as data:
                    if str(data["fingerprint"]) == self._fingerprint and int(data["counts"].size) >= digits:
                        return
            started = time.perf_counter()
            counts = np.asarray([item.count for item in items], dtype=np.int32)
            total = int(counts.sum())
            packed = np.zeros((total, -(-self._code_length // 8)), dtype=np.uint8)
            angles = np.zeros(total, dtype=np.float64)
            start = 0
            for item in items:
                stop = start + item.count
                if item.count:
                    packed[start:stop] = item.packed
                    angles[start:stop] = item.angles
                start = stop
            directory = os.path.dirname(self._path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            temporary = self._path + ".tmp.npz"
            np.savez(
                temporary,
                fingerprint=np.asarray(self._fingerprint),
                code_length=np.asarray(self._code_length),
                labels=np.asarray([item.label for item in items], dtype=np.int16),
                counts=counts,
                angles=angles,
                packed=packed,
            )
            os.replace(temporary, self._path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.event(
                "mnist.encoding.cache.error",
                section=OUTSIDE_ARTICLE,
                data={"path": self._path, "error": str(exc), "stage": "write"},
            )
            return
        LOGGER.event(
            "mnist.encoding.cache.save",
            section=OUTSIDE_ARTICLE,
            data={"path": self._path, "digits": digits, "codes": total, "seconds": time.perf_counter() - started},
        )


class StimulusCodeCache:
    """Углопозиции и коды цифр считаются один раз: раскладка, калибровка и база эмбеддингов берут их отсюда.

    Цифры идут в порядке набора; все углопозиции цифры кодируются одним батчем (Encoder.encode_matrix).
    Цифры без углопозиций остаются в списке пустыми: раскладка учитывает их в счётчике цифр, а база
    эмбеддингов пропускает — как и раньше, когда каждый потребитель читал набор заново (вне статьи).
    Большой набор (от ENCODING_PARALLEL_MIN_DIGITS недостающих цифр) кодируется рабочими процессами.
    """

    def __init__(
        self,
        dataset,
        encoder: Encoder,
        extractor: MnistSobelAngleMap,
        *,
        label: int | None,
        store: "EncodedCodesFile | None" = None,
    ) -> None:
        self._iterator = iter(dataset)
        self._encoder = encoder
        self._extractor = extractor
        self._label = label
        self._items: list[DigitCodes] = []
        self._exhausted = False
        # Файл кодов набора на диске (большие наборы); _skip — сколько цифр набора взято из файла и ещё
        # не пропущено в самом наборе (набор читается дальше, только если понадобятся следующие цифры).
        self._store = store
        self._skip = 0

    @staticmethod
    def encode_digit(
        encoder: Encoder, extractor: MnistSobelAngleMap, img: "np.ndarray", digit: int, *, log: bool = True
    ) -> DigitCodes:
        """Углопозиции цифры (§4.4) и их коды (§4.6) одним батчем; log — записи журнала кодера с картинкой цифры."""
        measurements = extractor.extract(img, digit)[int(digit)]
        packed = np.zeros((0, -(-encoder.code_length // 8)), dtype=np.uint8)
        if measurements:
            bits = encoder.encode_matrix(
                [(float(a), float(x), float(y)) for a, x, y in measurements],
                log_image=img if log else None,
                log_label=int(digit) if log else None,
                log_measurements=measurements if log else None,
            )
            packed = LayoutCodeTable.pack_bits(bits)
        return DigitCodes(
            label=int(digit),
            angles=tuple(float(a) for a, _, _ in measurements),
            packed=packed,
            code_length=encoder.code_length,
        )

    def _next_image(self) -> tuple["np.ndarray", int] | None:
        """Следующая цифра набора с нужной меткой: изображение и метка; None — набор кончился."""
        for img_tensor, digit in self._iterator:
            if self._label is not None and int(digit) != self._label:
                continue
            if self._skip:
                # Эта цифра уже взята из файла кодов.
                self._skip -= 1
                continue
            return img_tensor.squeeze(0).numpy(), int(digit)
        self._exhausted = True
        return None

    def _load_next(self) -> bool:
        image = self._next_image()
        if image is None:
            return False
        self._items.append(self.encode_digit(self._encoder, self._extractor, image[0], image[1]))
        return True

    @staticmethod
    def _worker_count() -> int:
        if ENCODING_WORKERS is not None:
            return max(1, int(ENCODING_WORKERS))
        return max(1, min(ENCODING_WORKERS_MAX, (os.cpu_count() or 1) - 1))

    def _load_parallel(self, count: int) -> None:
        """Следующие count цифр набора — рабочими процессами, в порядке набора."""
        workers = self._worker_count()
        if workers <= 1:
            return
        started = time.perf_counter()
        tasks: list[list[tuple["np.ndarray", int]]] = []
        task: list[tuple["np.ndarray", int]] = []
        digits = 0
        while digits < count:
            image = self._next_image()
            if image is None:
                break
            task.append(image)
            digits += 1
            if len(task) >= ENCODING_CHUNK_DIGITS:
                tasks.append(task)
                task = []
        if task:
            tasks.append(task)
        if not tasks:
            return
        LOGGER.event(
            "mnist.encoding.parallel",
            section=OUTSIDE_ARTICLE,
            data={
                "digits": digits,
                "workers": workers,
                "chunk_digits": ENCODING_CHUNK_DIGITS,
                "note": "углопозиции (§4.4) и коды (§4.6) этих цифр считают рабочие процессы тем же кодером; "
                "записи журнала кодера по ним не выводятся",
            },
        )
        code_length = self._encoder.code_length
        codes = 0
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)), initializer=DigitEncodingWorker.initialize
        ) as executor:
            for encoded in executor.map(DigitEncodingWorker.encode, tasks):
                for label, angles, packed in encoded:
                    self._items.append(DigitCodes(label=label, angles=angles, packed=packed, code_length=code_length))
                    codes += len(angles)
        LOGGER.event(
            "mnist.encoding.parallel.done",
            section=OUTSIDE_ARTICLE,
            data={"digits": digits, "codes": codes, "seconds": time.perf_counter() - started},
        )

    def take(self, count: int) -> list[DigitCodes]:
        """Первые count цифр набора (включая цифры без углопозиций)."""
        store = self._store if count >= ENCODING_CODES_CACHE_MIN_DIGITS else None
        if store is not None and not self._items and not self._exhausted:
            cached = store.load(count)
            if cached is not None:
                self._items = cached
                self._skip = len(cached)
        missing = count - len(self._items)
        encoded = False
        if missing >= ENCODING_PARALLEL_MIN_DIGITS and not self._exhausted:
            self._load_parallel(missing)
            encoded = True
        while len(self._items) < count and not self._exhausted:
            self._load_next()
            encoded = True
        if store is not None and encoded:
            store.save(self._items[:count])
        return self._items[:count]

    def take_nonempty(self, count: int) -> list[DigitCodes]:
        """Первые count цифр, у которых есть хотя бы одна углопозиция."""
        while sum(1 for item in self._items if item.count) < count and not self._exhausted:
            self._load_next()
        result = [item for item in self._items if item.count]
        return result[:count]


def _collect_codes(
    cache: StimulusCodeCache,
    count: int,
    encoder: Encoder,
    extractor: MnistSobelAngleMap,
) -> tuple[LayoutPayloadBuilder, int]:
    payload = LayoutPayloadBuilder(
        similarity=None,
        lambda_threshold=None,
        eta=None,
        metadata={ENCODER_SIGNATURE_KEY: encoder.signature, **_stimulus_metadata(extractor)},
    )
    total_codes = 0
    for item in cache.take(count):
        # Коды цифры уходят в таблицу массивами, без объекта на каждый код.
        payload.add_packed(label=item.label, hues=item.angles, packed=item.packed, code_length=item.code_length)
        total_codes += item.count
    return payload, total_codes


def _collect_samples(cache: StimulusCodeCache, sample_count: int) -> list[tuple[int, list[BitArray]]]:
    return [(item.label, list(item.codes)) for item in cache.take_nonempty(sample_count)]


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
    if LAYOUT_SMOOTH_PINWHEELS or points <= seed_points:
        return seed_points
    ratio = points / float(seed_points)
    levels = max(1, math.ceil(math.log(ratio) / math.log(LAYOUT_SEED_LEVEL_RATIO_MAX) - 1e-9))
    if levels == 1:
        return seed_points
    return max(seed_points, int(math.ceil(points / ratio ** (1.0 / levels))))


def _layout_from_codes(
    codes: "dict[float, list] | LayoutCodeTable",
    run_params: TunedLayoutParams,
    colorizer: EncoderCodeColorizer | None,
    *,
    layout_path: str | None = None,
) -> Layout:
    """Раскладка кодов: дальняя и ближняя фазы (§5.5) либо большая раскладка от затравки и её доводка."""
    seeded = (
        run_params.seed_points is not None
        and isinstance(codes, LayoutCodeTable)
        and codes.count > run_params.seed_points
    )
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
            similarity_cutoff=LAYOUT_SEED_SHORT_SIMILARITY_CUTOFF,
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
) -> Layout:
    """Раскладка с параметрами конвейера; initial_cells — начальная расстановка вместо случайной (§5.2)."""
    return Layout(
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
        long_subset_min_neighbours=LAYOUT_LONG_SUBSET_MIN_NEIGHBOURS,
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
) -> tuple[Layout, int, int]:
    """Большая раскладка от затравки (вне статьи; ср. подмножество §5.9.4): раскладка, радиус и число шагов доводки.

    Случайное подмножество точек раскладывается целиком обычным алгоритмом (дальняя и ближняя фазы с настройкой
    для своего числа точек); остальные точки ставятся возле самых похожих точек подмножества. Глобальный
    порядок готов — дальняя фаза большой раскладки начинается с радиуса в несколько блоков затравки и идёт
    по тому же геометрическому расписанию, что полная фаза (те же шаги на октаву радиуса, §7.1.7).
    """
    placement = SeededPlacement(
        codes,
        seed_count=_seed_level_points(codes.count, int(run_params.seed_points)),
        seed=LAYOUT_RANDOM_SEED,
        similarity=LAYOUT_SIMILARITY,
    )
    seed_params = LayoutParameterTuner(
        empty_ratio=LAYOUT_EMPTY_RATIO, start_radius_factor=LAYOUT_ADAPTIVE_RADIUS_START_FACTOR
    ).tune(placement.seed_count)
    started = time.perf_counter()
    # Затравка раскладывается тем же путём: целиком либо, если она сама велика, от своей затравки (уровни).
    seed_layout = _layout_from_codes(placement.seed_table(), seed_params, colorizer)
    seed_cells = seed_layout.cells_array()
    seed_shape = (seed_layout.height, seed_layout.width)
    seed_lambda = seed_layout.lambda_threshold
    seed_seconds = time.perf_counter() - started
    # Таблицы сходства затравки занимают память устройства, нужную большой раскладке.
    seed_layout.release_device()
    del seed_layout
    grid_size = int(math.ceil(math.sqrt(codes.count * (1.0 + LAYOUT_EMPTY_RATIO))))
    started = time.perf_counter()
    cells = placement.cells(
        seed_cells,
        seed_shape,
        (grid_size, grid_size),
        # Режим статьи: место точки в блоке родителя — по похожим на неё точкам затравки (sim ≥ λ затравки).
        neighbour_threshold=None if LAYOUT_SMOOTH_PINWHEELS else seed_lambda,
    )
    layout = _new_layout(codes, run_params, colorizer, initial_cells=cells)
    if not LAYOUT_SMOOTH_PINWHEELS:
        # Порог λ большой раскладки продолжает расписание затравки (§7.1.7): её дальний порядок уже установлен.
        layout.set_similarity_params(lambda_threshold=seed_lambda)
    full_radius = max(LAYOUT_ADAPTIVE_RADIUS_MIN, int(layout.width * LAYOUT_ADAPTIVE_RADIUS_START_FACTOR))
    # Начальный радиус доводки — в блоках родителя: дальней фазы сглаживания либо ближней фазы режима статьи.
    radius_factor = LAYOUT_SEED_REFINE_RADIUS_FACTOR if LAYOUT_SMOOTH_PINWHEELS else LAYOUT_SEED_SHORT_RADIUS_FACTOR
    refine_radius = min(
        full_radius,
        max(LAYOUT_SEED_REFINE_RADIUS_MIN, int(math.ceil(radius_factor * placement.block_side()))),
    )
    refine_steps = run_params.long_steps
    if full_radius > 1 and refine_radius < full_radius:
        refine_steps = max(1, int(round(run_params.long_steps * math.log(refine_radius) / math.log(full_radius))))
    if not LAYOUT_SMOOTH_PINWHEELS:
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
                if LAYOUT_SMOOTH_PINWHEELS
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
) -> None:
    """Дальняя фаза раскладки (§5.5.1, §7.1.7); start_radius и steps — начало и длина расписания доводки."""
    long_radius_start = max(
        LAYOUT_ADAPTIVE_RADIUS_MIN,
        int(layout.width * LAYOUT_ADAPTIVE_RADIUS_START_FACTOR),
    )
    if start_radius is not None:
        long_radius_start = int(start_radius)
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
        steps=run_params.long_steps if steps is None else int(steps),
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


def _run_short_phase(
    layout: Layout,
    run_params: TunedLayoutParams,
    *,
    step_offset: int,
    start_radius: int | None = None,
    similarity_cutoff: float | None = None,
) -> None:
    """Ближняя фаза раскладки (§5.5.2, §7.1.7); start_radius — начальный радиус выбора пар доводки от затравки,
    similarity_cutoff — порог ранней отсечки пар (§5.9.1) вместо LAYOUT_SHORT_SIMILARITY_CUTOFF."""
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
    if start_radius is not None:
        short_radius_start = max(LAYOUT_ADAPTIVE_RADIUS_MIN, min(short_radius_start, int(start_radius)))
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
        similarity_cutoff=LAYOUT_SHORT_SIMILARITY_CUTOFF if similarity_cutoff is None else similarity_cutoff,
        partitions=LAYOUT_SHORT_PARTITIONS,
    )
    layout.run(
        steps=run_params.short_steps,
        pairs_per_step=run_params.short_pairs_per_step,
        pair_radius=short_radius_start,
        mode="short",
        local_radius=run_params.short_local_radius,
        min_swap_ratio=LAYOUT_SHORT_MIN_SWAP_RATIO,
        min_swap_window=LAYOUT_MIN_SWAP_WINDOW,
        log_every=LAYOUT_LOG_EVERY_SHORT,
        step_offset=step_offset,
        adaptive_params=adaptive_short,
        short_optim=short_optim,
    )


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
                "points": layout.point_count,
            },
        )

    def __init__(
        self,
        *,
        count: int,
        label: int | None = None,
        activation_samples: int | None = None,
        reuse_cache: bool = True,
        layout_only: bool = False,
    ) -> None:
        self.count = count
        self.label = label
        # False (--rebuild) — готовые раскладка, чекпоинт и эмбеддинги игнорируются, всё считается заново
        # и записывается в кеш поверх старого.
        self._reuse_cache = bool(reuse_cache)
        # True (--layout-only) — конвейер заканчивается раскладкой: JSON и картинки сохраняются,
        # активация и эмбеддинги не считаются.
        self._layout_only = bool(layout_only)
        resolved_activation_samples = count if activation_samples is None else int(activation_samples)
        if resolved_activation_samples <= 0:
            raise ValueError("activation_samples must be positive")
        self._activation_samples = resolved_activation_samples
        configure_logging()
        GPU_ENVIRONMENT.apply()
        self.encoder = _build_encoder()
        self.colorizer = _build_colorizer()
        self.extractor = _build_extractor()
        self._codes: StimulusCodeCache | None = None
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
                expected_stimulus=_stimulus_metadata(self.extractor),
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

    def _code_cache(self, dataset) -> StimulusCodeCache:
        """Коды цифр набора — один общий кеш для раскладки и активации."""
        if self._codes is None:
            store = None
            if ENCODING_CODES_CACHE:
                store = EncodedCodesFile(ENCODING_CODES_CACHE_DIR, encoder=self.encoder, split="train", label=self.label)
            self._codes = StimulusCodeCache(dataset, self.encoder, self.extractor, label=self.label, store=store)
        return self._codes

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
        samples = _collect_samples(self._code_cache(dataset), self._activation_samples)
        calibrated = _calibrated_activation_engine(
            layout,
            energies,
            [codes for _, codes in samples[:ACTIVATION_OUTPUT_CALIBRATION_SAMPLES]],
        )
        engine = calibrated.engine
        self._activation_metadata = _activation_metadata(calibrated.size, detectors=engine.detectors)
        embeddings = list(initial_embeddings[: len(samples)])
        start_index = len(embeddings)
        # Калибровочные стимулы уже активированы (§6.6): их эмбеддинги готовы, повторно не считаются.
        precomputed: dict[int, tuple[BitArray, int]] = {
            index: (embedding, count)
            for index, (embedding, count) in enumerate(zip(calibrated.embeddings, calibrated.active_counts))
            if index >= start_index
        }
        LOGGER.event(
            "activation.resume",
            section=OUTSIDE_ARTICLE,
            data={
                "completed": start_index,
                "target": len(samples),
                "precomputed": len(precomputed),
                "workers": ACTIVATION_PARALLEL_WORKERS,
                "batch_size": ACTIVATION_BATCH_SIZE,
            },
        )
        for batch_start in range(start_index, len(samples), ACTIVATION_BATCH_SIZE):
            batch = samples[batch_start : batch_start + ACTIVATION_BATCH_SIZE]
            pending = [
                (batch_start + offset, codes)
                for offset, (_, codes) in enumerate(batch)
                if batch_start + offset not in precomputed
            ]
            computed: dict[int, tuple[BitArray, int]] = {}
            if pending:
                results = engine.activate_many(
                    [codes for _, codes in pending],
                    max_workers=ACTIVATION_PARALLEL_WORKERS,
                )
                computed = {
                    index: (result.embedding, len(result.active_detectors))
                    for (index, _), result in zip(pending, results)
                }
            for offset, (label, codes) in enumerate(batch):
                index = batch_start + offset
                embedding, active_count = precomputed.get(index) or computed[index]
                LOGGER.event(
                    "activation.sample",
                    section=STIMULUS_DETECTION,
                    data={
                        "index": index,
                        "sample_label": str(int(label)),
                        "stimuli": len(codes),
                        "active_detectors": active_count,
                        "embedding_bits": embedding.count(),
                        "from_calibration": index in precomputed,
                    },
                )
                embeddings.append(
                    EmbeddingRecord(
                        class_label=str(int(label)),
                        embedding_bits=embedding.to01(),
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

    def _export_stage(
        self,
        layout: Layout,
        run_params: TunedLayoutParams,
        *,
        layout_path: str,
        save_json: bool,
        stages: dict[str, bool],
        embeddings: Sequence[EmbeddingRecord],
    ) -> None:
        """Последний этап конвейера: JSON и картинки раскладки, отметка этапа в чекпоинте (вне статьи)."""
        self._export_layout(layout, run_params, layout_path=layout_path, save_json=save_json)
        stages["export"] = True
        self._checkpoint_store.save(
            layout_path,
            stages=stages,
            target_samples=self._activation_samples,
            embeddings=embeddings,
        )

    def digit_codes(self, count: int) -> list[DigitCodes]:
        """Первые count цифр обучающего набора с их углопозициями и кодами — из общего кеша кодов."""
        return self._code_cache(self._build_dataset()).take_nonempty(count)

    def run(self) -> Layout:
        """Конвейер MNIST: раскладка, затем (без --layout-only) активация и эмбеддинги; возвращает раскладку."""
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
            angles = layout.hue_count()
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
                self._code_cache(dataset),
                self.count,
                self.encoder,
                self.extractor,
            )
            # Коды в порядке точек раскладки (группы по углу в порядке первого появления) — таблицей массивов.
            codes = payload.code_table()
            angles = codes.hue_count()
            self._log_dataset(angles, total_codes)
            tuned_params = self._tune_layout(total_codes)
            expected_pair_radius = _expected_pair_radius(total_codes, tuned_params)
            layout_path = _layout_json_path(self.count, expected_pair_radius)
            # Файл больше LAYOUT_CACHE_MAX_FILE_BYTES как кеш не читается: до раскладки пишется только заголовок.
            payload.save_base(layout_path, max_points_bytes=LAYOUT_CACHE_MAX_FILE_BYTES)
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
        if self._layout_only:
            # Эмбеддинги чекпоинта остаются как были: обычный запуск возьмёт эту раскладку из кеша
            # и продолжит с активации.
            LOGGER.event(
                "activation.skip",
                section=OUTSIDE_ARTICLE,
                data={
                    "path": layout_path,
                    "reason": "layout_only",
                    "embeddings": len(checkpoint_embeddings),
                    "note": "--layout-only: активация и эмбеддинги не считаются",
                },
            )
            self._export_stage(
                layout,
                tuned_params,
                layout_path=layout_path,
                save_json=save_json,
                stages=stages,
                embeddings=checkpoint_embeddings,
            )
            return layout
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
        self._export_stage(
            layout,
            tuned_params,
            layout_path=layout_path,
            save_json=save_json,
            stages=stages,
            embeddings=embeddings,
        )
        return layout



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
                expected_stimulus=_stimulus_metadata(self.extractor),
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
                stored_detectors=metadata.get(ACTIVATION_DETECTORS_KEY),
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
        cache = StimulusCodeCache(dataset, self.encoder, self.extractor, label=None)
        samples = _collect_samples(cache, DECODER_DATASET_LIMIT)
        calibrated = _calibrated_activation_engine(
            layout,
            energies,
            [codes for _, codes in samples[:ACTIVATION_OUTPUT_CALIBRATION_SAMPLES]],
        )
        engine = calibrated.engine
        embeddings: list[EmbeddingRecord] = []
        for idx, (label, codes) in enumerate(samples):
            if idx < len(calibrated.embeddings):
                embedding = calibrated.embeddings[idx]
            else:
                embedding = engine.activate(codes).embedding
            embeddings.append(
                EmbeddingRecord(
                    class_label=str(int(label)),
                    embedding_bits=embedding.to01(),
                )
            )
            LOGGER.event(
                "decoder.base.generate",
                section=STIMULUS_DETECTION,
                data={
                    "index": idx,
                    "label": int(label),
                    "stimuli": len(codes),
                    "bits": embedding.count(),
                },
            )
        self._embedding_store.save(
            layout_path, embeddings, metadata=_activation_metadata(calibrated.size, detectors=engine.detectors)
        )
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


class MnistSecondLevelRunner:
    """Второй уровень на MNIST (§6.2, §6.6; расширение статьи): код картинки и раскладка кодов картинок.

    1. Первая раскладка углопозиций — обычным конвейером (MnistLayoutRunner, только раскладка).
    2. Над ней строятся детекторы-круги (§4.3); все углопозиции картинки зажигают самые похожие точки
       раскладки, зажжённые точки активируют круги, биты кругов дают код картинки (PlaneDetection).
    3. Коды картинок раскладываются тем же Layout: точка второй раскладки — картинка, цвет — класс цифры.
    4. Проверка: нечёткий поиск тестовых картинок по базе кодов обучающих (§2.2.5), тем же декодером.
    """

    def __init__(
        self,
        *,
        count: int,
        reuse_cache: bool = True,
        evaluate_limit: int = SECOND_LEVEL_EVALUATE_LIMIT,
        split: str = "test",
    ) -> None:
        if evaluate_limit < 0:
            raise ValueError("evaluate_limit must be non-negative")
        self.count = int(count)
        self._evaluate_limit = int(evaluate_limit)
        self._split = split
        self._primary = MnistLayoutRunner(count=count, reuse_cache=reuse_cache, layout_only=True)

    def run(self) -> float | None:
        """Весь путь второго уровня; возвращает точность нечёткого поиска (None — проверка отключена)."""
        primary_layout = self._primary.run()
        detection = _build_plane_detection(primary_layout)
        codes = self._encode_digits(detection)
        codes.log_summary(detection.space, similarity=LAYOUT_SIMILARITY, seed=SECOND_LEVEL_RANDOM_SEED)
        second_layout = self._layout_codes(codes)
        self._export(second_layout)
        if self._evaluate_limit == 0:
            return None
        return self._evaluate(detection, codes)

    def _encode_digits(self, detection: PlaneDetection) -> StimulusSetCodes:
        """Коды второго уровня обучающих картинок: набор стимулов — все углопозиции одной цифры."""
        codes = StimulusSetCodes(detection.code_length)
        for item in self._primary.digit_codes(self.count):
            codes.add(item.label, detection.encode_packed(item.packed, label=item.label))
        return codes

    def _layout_codes(self, codes: StimulusSetCodes) -> Layout:
        """Раскладка кодов картинок тем же алгоритмом и с той же настройкой, что первая (§6.2, §6.6)."""
        table = codes.table(hue_step=SECOND_LEVEL_CLASS_HUE_STEP).grouped_by_hue()
        tuner = LayoutParameterTuner(
            empty_ratio=LAYOUT_EMPTY_RATIO,
            start_radius_factor=LAYOUT_ADAPTIVE_RADIUS_START_FACTOR,
        )
        return _layout_from_codes(table, tuner.tune(table.count), self._primary.colorizer)

    def _export(self, layout: Layout) -> None:
        """JSON и карты второй раскладки: псевдоцвета кода, раскраска по классу цифры, яркость по энергии."""
        suffix = "auto" if layout.pair_radius is None else str(layout.pair_radius)
        stem = f"{SECOND_LEVEL_FILE_PREFIX}-{self.count}-{suffix}"
        layout.save_json(f"{stem}-{LAYOUT_CACHE_TAG}.json")
        MnistLayoutRunner._save_layout_images(
            layout,
            image_path=f"{stem}.png",
            energies=layout.point_energies(ACTIVATION_POINT_ENERGY_RADIUS),
        )

    def _evaluate(self, detection: PlaneDetection, codes: StimulusSetCodes) -> float:
        """Нечёткий поиск (§2.2.5): класс тестовой картинки — по самым похожим кодам обучающих картинок."""
        from torchvision import transforms
        from torchvision.datasets import MNIST

        database = EmbeddingDatabase(
            [
                LabeledEmbedding(label=label, embedding=embedding)
                for label, embedding in zip(codes.labels, codes.embeddings())
            ],
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
        decoder = MnistDecoder(
            encoder=self._primary.encoder,
            extractor=self._primary.extractor,
            activation_engine=detection,
            embedding_db=database,
        )
        dataset = MNIST(root="./data", train=self._split == "train", download=True, transform=transforms.ToTensor())
        images: list["np.ndarray"] = []
        labels: list[int] = []
        for index, (image, label) in enumerate(dataset):
            if index >= self._evaluate_limit:
                break
            images.append(image.squeeze(0).numpy())
            labels.append(int(label))
        _, accuracy = decoder.predict_batch(images, labels=labels)
        return 0.0 if accuracy is None else float(accuracy)


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
    layout_parser.add_argument(
        "--layout-only",
        action="store_true",
        help="только раскладка: сохранить JSON и картинки раскладки в data/ и выйти, активация и эмбеддинги "
        "не считаются (обычный запуск потом возьмёт эту раскладку из кеша и продолжит с активации)",
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
    second_parser = subparsers.add_parser(
        "second",
        help="второй уровень: коды картинок из детекторов-кругов над первой раскладкой, их раскладка и проверка",
    )
    second_parser.add_argument("--count", type=int, default=DECODER_DEFAULT_COUNT)
    second_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="игнорировать готовую первую раскладку в data/ и посчитать её заново",
    )
    second_parser.add_argument(
        "--limit",
        type=int,
        default=SECOND_LEVEL_EVALUATE_LIMIT,
        help="число картинок проверки нечётким поиском; 0 — без проверки",
    )
    second_parser.add_argument("--split", choices=["train", "test"], default="test")
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
            layout_only=args.layout_only,
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
    if command == "second":
        second_runner = MnistSecondLevelRunner(
            count=args.count,
            reuse_cache=not args.rebuild,
            evaluate_limit=args.limit,
            split=args.split,
        )
        second_accuracy = second_runner.run()
        if second_accuracy is not None:
            print(f"second level: dataset={args.split} samples={args.limit} accuracy={second_accuracy:.4f}")
        return
    parser.error(f"unknown command {command}")


if __name__ == "__main__":
    main()
