"""Настройки DAMP: значения из client/config.toml — имена этого модуля (вне статьи).

Клиентский слой задаёт значения в client/config.toml (порядок там — по значимости). Здесь — только системная
часть: чтение файла, имена ключей в файлах результата (формат, не настройки) и правила, по которым одни значения
выводятся из других (_derive). Код читает настройки как settings.ИМЯ в момент вызова, поэтому подмена действует
сразу: overridden({"ИМЯ": значение}) — значения поверх файла на время вызова (так их задаёт клиент через damp_api),
выводимые значения пересчитываются; в тестах — и mock.patch.object(settings, "ИМЯ", ...).

Соглашения файла: "none" — None; массивы — кортежи; { smooth = …, article = … } — значение по режиму раскладки
LAYOUT_SMOOTH_PINWHEELS. Другой файл настроек — переменная окружения DAMP_CONFIG.
"""
# ruff: noqa: F821 — имена приходят из файла настроек через globals().update
from __future__ import annotations

import os
import tomllib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

MODE_KEYS = frozenset({"smooth", "article"})


def _config_path() -> Path:
    """Файл настроек: DAMP_CONFIG либо client/config.toml в ближайшей сверху папке, где он есть."""
    explicit = os.environ.get("DAMP_CONFIG")
    if explicit:
        return Path(explicit)
    for folder in Path(__file__).resolve().parents:
        candidate = folder / "client" / "config.toml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("client/config.toml не найден; задайте путь в DAMP_CONFIG")


def _plain(value: object) -> object:
    """Значение файла → значение Python: "none" — None, массив — кортеж (вглубь)."""
    if value == "none":
        return None
    if isinstance(value, list):
        return tuple(_plain(item) for item in value)
    if isinstance(value, dict) and set(value) != MODE_KEYS:
        return {key: _plain(item) for key, item in value.items()}
    return value


def _tables(path: Path) -> dict[str, object]:
    """Все ключи файла одним словарём, как в файле: разделы только группируют, имена уникальны во всём файле."""
    with open(path, "rb") as stream:
        tables = tomllib.load(stream)
    values: dict[str, object] = {}
    for table, keys in tables.items():
        if not isinstance(keys, dict):
            raise ValueError(f"{path}: ключ {table} вне раздела")
        for name, value in keys.items():
            if name in values:
                raise ValueError(f"{path}: ключ {name} повторяется")
            values[name] = value
    return values


def _resolve(raw: Mapping[str, object]) -> dict[str, object]:
    """Значения файла → значения Python; { smooth = …, article = … } — по режиму LAYOUT_SMOOTH_PINWHEELS."""
    values = {name: _plain(value) for name, value in raw.items()}
    smooth = bool(values["LAYOUT_SMOOTH_PINWHEELS"])
    for name, value in list(values.items()):
        if isinstance(value, dict) and set(value) == MODE_KEYS:
            values[name] = _plain(value["smooth" if smooth else "article"])
    return values


def _load(path: Path) -> dict[str, object]:
    """Все ключи файла одним словарём со значениями Python."""
    return _resolve(_tables(path))


CONFIG_PATH = _config_path()
_RAW = _tables(CONFIG_PATH)
KEYS = frozenset(_RAW)  # имена ключей файла настроек
globals().update(_resolve(_RAW))

# --- Имена ключей и версии форматов файлов результата (не настройки: менять — значит менять формат файлов).
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
# Радиус выбора пар ближней фазы, с которым раскладка была запланирована (по нему же назван файл). Кеш сверяется
# по нему: итоговый радиус большой раскладки режима статьи меньше планового, и сверка по итоговому отвергала
# готовую раскладку. В файлах без этого ключа сверяется итоговый радиус, как раньше.
PLANNED_PAIR_RADIUS_KEY = "planned_pair_radius"
PIPELINE_CHECKPOINT_SUFFIX = ".checkpoint.json"
PIPELINE_CHECKPOINT_VERSION = 3

# --- Выводимые значения: правила, а не настройки (значение следует за своим источником).
def _derive() -> None:
    """Пересчитать выводимые значения по нынешним значениям ключей файла."""
    global LAYOUT_CACHE_LEGACY_STIMULUS, LAYOUT_COMPONENT_PERIOD, LAYOUT_SHORT_SIMILARITY_CUTOFF, ACTIVATION_POINT_ENERGY_RADIUS
    global LADDER_CLUSTER_EPS, LADDER_CLUSTER_MIN_SAMPLES, LADDER_CLUSTER_RADIUS_MIN, LADDER_CLUSTER_ENERGY_RADIUS
    global LADDER_ANGLE_CODE_LENGTH, LADDER_ANGLE_LAYERS, LADDER_ANGLE_WIDEST_DETECTORS, LADDER_ANGLE_NARROWEST_DETECTORS
    global LADDER_ANGLE_OVERLAP, LADDER_DETECTOR_THRESHOLD, LADDER_SATURATION, LADDER_RANDOM_SEED
    global PINWHEEL_POSITION_X, PINWHEEL_POSITION_Y, DIGITS_CONTOUR_PATCH_SIZES, LOG_INTERVALS
    global LAYOUT_CACHE_PATTERNS
    LAYOUT_CACHE_LEGACY_STIMULUS = {PATCH_SIZE_KEY: 4, STIMULUS_MODE_KEY: "window"}
    # Период угла для раскраски хода раскладки по углу стимула в Rerun (как pinwheel-тест, рис. 16).
    LAYOUT_COMPONENT_PERIOD = MNIST_ANGLE_MAX - MNIST_ANGLE_MIN
    LAYOUT_SHORT_SIMILARITY_CUTOFF = LAYOUT_LAMBDA_THRESHOLD
    ACTIVATION_POINT_ENERGY_RADIUS = LAYOUT_ENERGY_RADIUS
    # Радиусы построения полей — в шагах между соседними элементами уровня (медиана расстояния до ближайшего
    # соседа): окрестность DBSCAN, радиус активации r_a, наименьший радиус поля и радиус энергии точки r_e. У
    # углопозиций шаг — одна клетка, и значения совпадают с настройками детекторов над раскладкой.
    LADDER_CLUSTER_EPS = ACTIVATION_DBSCAN_EPS
    LADDER_CLUSTER_MIN_SAMPLES = ACTIVATION_DBSCAN_MIN_SAMPLES
    LADDER_CLUSTER_RADIUS_MIN = float(ACTIVATION_DETECTOR_RADIUS_MIN)
    LADDER_CLUSTER_ENERGY_RADIUS = float(LAYOUT_ENERGY_RADIUS)
    # Код направления углопозиции — «что» уровня 0: детекторы угла (§4.4), слои от самого широкого к самому узкому.
    LADDER_ANGLE_CODE_LENGTH = MNIST_ANGLE_CODE_LENGTH
    LADDER_ANGLE_LAYERS = MNIST_ANGLE_LAYERS
    LADDER_ANGLE_WIDEST_DETECTORS = MNIST_ANGLE_WIDEST_DETECTORS
    LADDER_ANGLE_NARROWEST_DETECTORS = MNIST_ANGLE_NARROWEST_DETECTORS
    LADDER_ANGLE_OVERLAP = MNIST_ANGLE_OVERLAP
    # Порог срабатывания детектора μ_d на набор элементов поля (§6.5).
    LADDER_DETECTOR_THRESHOLD = ACTIVATION_DETECTOR_THRESHOLD
    # Эмбеддинг детали (§6.5, §7.1.12): не больше LADDER_SATURATION самых сильных детекторов, длина кода под
    # среднюю плотность LADDER_TARGET_DENSITY (§6.6), подбирается по первым LADDER_CALIBRATION_SETS полям.
    LADDER_SATURATION = ACTIVATION_SATURATION_LIMIT
    LADDER_RANDOM_SEED = ENCODER_RANDOM_SEED
    PINWHEEL_POSITION_X = (MNIST_POSITION_MIN + MNIST_POSITION_MAX) / 2.0
    PINWHEEL_POSITION_Y = (MNIST_POSITION_MIN + MNIST_POSITION_MAX) / 2.0
    DIGITS_CONTOUR_PATCH_SIZES = (MNIST_PATCH_SIZE,)  # то же окно, что у конвейера
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
        # Кодирование набора стимулов (Encoder.encode_set): событие — раз в столько наборов.
        "encoder.encode.set": LOG_INTERVAL_ENCODER_SET,
    }
    LAYOUT_CACHE_PATTERNS = (
        f"data/layout-{{count}}-*-{LAYOUT_CACHE_TAG}.json",
        f"layout-{{count}}-*-{LAYOUT_CACHE_TAG}.json",
    )


# Выводимые имена: не ключи файла, следуют за своим источником.
DERIVED = frozenset(
    (
        'LAYOUT_CACHE_LEGACY_STIMULUS', 'LAYOUT_COMPONENT_PERIOD', 'LAYOUT_SHORT_SIMILARITY_CUTOFF', 'ACTIVATION_POINT_ENERGY_RADIUS',
        'LADDER_CLUSTER_EPS', 'LADDER_CLUSTER_MIN_SAMPLES', 'LADDER_CLUSTER_RADIUS_MIN', 'LADDER_CLUSTER_ENERGY_RADIUS',
        'LADDER_ANGLE_CODE_LENGTH', 'LADDER_ANGLE_LAYERS', 'LADDER_ANGLE_WIDEST_DETECTORS', 'LADDER_ANGLE_NARROWEST_DETECTORS',
        'LADDER_ANGLE_OVERLAP', 'LADDER_DETECTOR_THRESHOLD', 'LADDER_SATURATION', 'LADDER_RANDOM_SEED',
        'PINWHEEL_POSITION_X', 'PINWHEEL_POSITION_Y', 'DIGITS_CONTOUR_PATCH_SIZES', 'LOG_INTERVALS',
        'LAYOUT_CACHE_PATTERNS',
    )
)
_derive()


def check_changes(changes: Mapping[str, object]) -> None:
    """Изменять можно только ключи файла настроек: выводимые значения следуют за своим источником."""
    unknown = sorted(set(changes) - KEYS)
    if unknown:
        derived = [name for name in unknown if name in DERIVED]
        hint = f"; выводятся из других ключей: {derived}" if derived else ""
        raise KeyError(f"нет таких ключей в {CONFIG_PATH.name}: {unknown}{hint}")


@contextmanager
def overridden(changes: Mapping[str, object]) -> Iterator[None]:
    """Внутри — значения changes поверх нынешних (имя ключа → значение, "none" — None, список — кортеж);
    выводимые значения пересчитываются; на выходе всё как было."""
    if not changes:
        yield
        return
    check_changes(changes)
    space = globals()
    saved = {name: space[name] for name in KEYS | DERIVED}
    values = {name: _plain(value) for name, value in changes.items()}
    if "LAYOUT_SMOOTH_PINWHEELS" in values:
        # Значения по режиму раскладки берутся из файла заново — кроме изменённых явно.
        mode = "smooth" if bool(values["LAYOUT_SMOOTH_PINWHEELS"]) else "article"
        for name, value in _RAW.items():
            if isinstance(value, dict) and set(value) == MODE_KEYS and name not in values:
                values[name] = _plain(value[mode])
    try:
        space.update(values)
        _derive()
        yield
    finally:
        space.update(saved)
