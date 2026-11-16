"""Этапы пайплайна как поля `Damp` — в порядке и с названиями, как их видит автор проекта.

1 стимулы → 2 первичное кодирование (↻ начало цикла) → 3 раскладка → 4 детекторы и активации → 5 эмбеддинги
первого уровня → 6 цикл (↺ конец цикла: детали снова идут на этап 2) → 7 распознавание.

У каждого этапа два вида вызовов:

- **кубики цикла** — один шаг этапа; результат (часть из `damp_api.parts`) идёт в следующий вызов, и клиент
  сам собирает свой проход: `digits` → `sticks` → `keyholes` → коды окон → память → детекторы → срабатывания →
  детали → …;
- **готовые прогоны** — то же, собранное системой целиком (цикл, проход без цикла, рисунки).

| Этап | Класс | Поле `Damp` | Кубики | Готовые прогоны |
|---|---|---|---|---|
| 1 | `Stimuli` | `stimuli` | `digits` | `of_image`, `sketch`, `contour_table` |
| 2 | `Coding` | `coding` | `sticks`, `keyholes` | `encode`, `detectors_picture` |
| 3 | `Layout` | `layout` | `memory` | `build`, `pinwheel` |
| 4 | `Detectors` | `detectors` | `build`, `fire` | — |
| 5 | `Embeddings` | `embeddings` | `details` | `build` |
| 6 | `Cycle` | `cycle` | `keyholes`, `step`, `covers`, `trained` | `run` |
| 7 | `Recognition` | `recognition` | `by_response`, `vote` | `evaluate`, `predict`, `predict_dataset`, `calibrate`, `bench` |
| — | `Figures` | `figures` | — | `readme`, `structure` |
"""
from __future__ import annotations

from pathlib import Path

from ._system import system
from .parts import Digits, Elements, Firing, Level, MapDetectors, Memory, Session, KeyholeCodes, Keyholes, TrainedLadder
from .reports import (
    Code,
    CycleLevel,
    BenchLevel,
    BenchMethod,
    BenchReport,
    BenchRobustness,
    BenchScore,
    CycleReport,
    Evaluation,
    LayoutSummary,
    LevelFailed,
    PinwheelReport,
    Prediction,
    Stage,
    Stimulus,
)

STAGES = (
    Stage(1, "stimuli", "Стимулы: углопозиции MNIST", "картинка → углопозиции: угол края и клетка сетки 24×24",
          "system/damp/stimuli/", "вне статьи"),
    Stage(2, "coding", "Первичное кодирование", "угол и место → разреженный код; окно → код «что и где»",
          "system/damp/encoding/", "§2, §3.2, §4; П.4–П.7", cycle="начало"),
    Stage(3, "layout", "Раскладка", "коды → карта, похожие рядом", "system/damp/layout/", "§5, §7.1.7"),
    Stage(4, "detectors", "Детекторы и активации",
          "карта → иерархия детекторов; стимул → зажжённые точки → сработавшие детекторы",
          "system/damp/activation/", "§6.1–§6.5, §7.1.10–§7.1.11"),
    Stage(5, "embeddings", "Эмбеддинги первого уровня", "сработавшие детекторы → код детали или картинки",
          "system/damp/activation/", "§6.5, §6.6, §7.1.12; П.1"),
    Stage(6, "cycle", "Цикл: уровни выше первого", "детали уровня → снова этап 2, пока окно не увидит цифру целиком",
          "system/damp/cycle/", "§6.2, §6.6; П.2, П.3, П.9", cycle="конец"),
    Stage(7, "recognition", "Распознавание", "эмбеддинги и детали верхнего уровня → класс цифры",
          "system/damp/decode/", "§2.2.5; мерка вне статьи"),
)


class _Stage:
    """Основа этапов: описание этапа (`about`) и настройки своего `Damp`."""

    number = 0

    def __init__(self, session: Session) -> None:
        self._session = session

    @property
    def about(self) -> Stage:
        """Описание этапа: номер, название, что делает, где в коде, по каким источникам (`Stage`)."""
        return STAGES[self.number - 1]

    def _step(self, digits: Digits, map_points: int | None = None):  # noqa: ANN202
        """Системный проход уровня над цифрами digits: общие правила цикла для всех кубиков."""
        limit = system.settings.LADDER_MAP_MAX_POINTS if map_points is None else int(map_points)
        return system.cycle_step(digits._groups, map_points=limit)


class Stimuli(_Stage):
    """Этап 1. Стимулы (вне статьи): картинка → углопозиции, то есть угол края контура и клетка сетки 24×24.

    Вход всего пайплайна. Для цикла нужен кубик `digits`; остальное — разовые прогоны.

    Example:
        >>> digits = damp.stimuli.digits(train=100, test=500)
        >>> damp.stimuli.of_image(digits.picture(0).image)[:2]   # первые две углопозиции пятёрки-примера
    """

    number = 1

    def digits(self, train: int, test: int = 0, *, split: str = "test", shifts: tuple[tuple[int, int], ...] = (),
               shift_pictures: int | None = None, train_views: tuple[tuple[int, int], ...] | None = None) -> Digits:
        """Кубик. Цифры MNIST с углопозициями: вход цикла.

        Args:
            train: сколько обучающих цифр; по ним строится память.
            test: сколько проверочных цифр.
            split: из какого набора MNIST брать проверочные: `"test"` или `"train"`.
            shifts: сдвиги `(dy, dx)` для проверки устойчивости; пусто — без сдвигов.
            shift_pictures: у скольких первых проверочных картинок делать сдвиги; `None` — `LADDER_SHIFT_PICTURES`.
            train_views: виды каждой обучающей цифры — сдвиги `(dy, dx)` на пиксели (§7.1.4: несколько разбиений
                одной картинки на клетки сетки); виды тоже обучающие. `None` — `LADDER_TRAIN_VIEWS`, как у готового
                прогона цикла; `()` — без видов.

        Returns:
            `Digits`: сначала обучающие картинки, следом их виды, затем проверочные и их сдвиги.

        Example:
            >>> digits = Damp().stimuli.digits(100, 500, train_views=((0, 1), (0, -1)))
        """
        with self._session.active():
            pictures = system.settings.LADDER_SHIFT_PICTURES if shift_pictures is None else int(shift_pictures)
            views = system.settings.LADDER_TRAIN_VIEWS if train_views is None else train_views
            groups = system.digits(train=int(train), test=int(test), split=split, shifts=shifts, shift_pictures=pictures,
                                   train_views=views)
        return Digits(self._session, groups)

    def of_image(self, image, label: int = 0) -> list[Stimulus]:  # noqa: ANN001
        """Прогон. Углопозиции одной картинки.

        Args:
            image: картинка 28×28, значения 0…1 или 0…255.
            label: цифра на картинке, 0…9; ключ словаря извлекателя, на результат не влияет.

        Returns:
            Список `Stimulus`; пустой, если контура нет.
        """
        import numpy as np

        pixels = np.asarray(image, dtype=np.float32)
        if pixels.max() > 1.0:
            pixels = pixels / 255.0
        with self._session.active():
            found = system.extractor().extract(pixels, label=int(label)).get(int(label), [])
        return [Stimulus(float(angle), float(x), float(y)) for angle, x, y in found]

    def sketch(self, digit: int, *, seed: int | None = None, split: str | None = None, patch_size: int | None = None,
               upscale: bool | None = None) -> None:
        """Прогон. Одна цифра MNIST из линий её углопозиций — в окне Rerun (нужен `RERUN=on`).

        Args:
            digit: какую цифру взять, 0…9.
            seed: зерно выбора картинки; `None` — из настроек.
            split: набор MNIST; `None` — `DIGIT_SKETCH_SPLIT`.
            patch_size: размер клетки, пикселей; `None` — `MNIST_PATCH_SIZE`.
            upscale: увеличивать ли картинку перед поиском краёв; `None` — `MNIST_SOBEL_UPSCALE_ENABLED`.
        """
        with self._session.active():
            s, tools = system.settings, system.sketch_tools()
            patch_size = s.MNIST_PATCH_SIZE if patch_size is None else int(patch_size)
            upscale = s.MNIST_SOBEL_UPSCALE_ENABLED if upscale is None else bool(upscale)
            sketch = tools.AnglePositionSketch(
                image_side=system.picture_side,
                patch_size=patch_size,
                line_scale=s.DIGIT_SKETCH_LINE_SCALE,
                line_radius_scale=s.DIGIT_SKETCH_LINE_RADIUS_SCALE,
                angle_in_degrees=s.MNIST_ANGLE_IN_DEGREES,
                angle_range=(s.MNIST_ANGLE_MIN, s.MNIST_ANGLE_MAX),
                grid_color=s.DIGIT_SKETCH_GRID_COLOR,
                lost_color=s.DIGIT_SKETCH_LOST_COLOR,
                entity_path=s.DIGIT_SKETCH_ENTITY_PATH,
                sobel_path=s.MNIST_SOBEL_ENTITY_PATH if upscale else None,
                pixel_scale=s.MNIST_SOBEL_UPSCALE if upscale else 1,
            )
            sampler = tools.MnistDigitSampler(root=s.DIGIT_SKETCH_MNIST_ROOT, split=split or s.DIGIT_SKETCH_SPLIT, seed=seed)
            extractor = system.extractor(patch_size, upscale=upscale)
            stimulus = sampler.sample(int(digit))
            measurements = extractor.extract(stimulus.image, stimulus.label)[stimulus.label]
            contour = extractor.contour_patches(stimulus.image)
            sketch.show(extractor.work_image(stimulus.image), stimulus.label, measurements, contour)

    def contour_table(self, *, seed: int | None = None, split: str | None = None, upscale: bool | None = None,
                      path: str | None = None) -> str:
        """Прогон. Таблица: цифры 0…9 и они же из углопозиций контура.

        Args:
            seed: зерно выбора картинок; `None` — `DIGITS_CONTOUR_SEED`.
            split: набор MNIST; `None` — `DIGIT_SKETCH_SPLIT`.
            upscale: увеличивать ли картинку; `None` — `MNIST_SOBEL_UPSCALE_ENABLED`.
            path: куда сохранить; `None` — `DIGITS_CONTOUR_OUTPUT_PATH`.

        Returns:
            Путь к сохранённой картинке.
        """
        with self._session.active():
            s, tools = system.settings, system.sketch_tools()
            upscale = s.MNIST_SOBEL_UPSCALE_ENABLED if upscale is None else bool(upscale)
            sheet = tools.AnglePositionSheet(
                image_side=system.picture_side,
                patch_sizes=s.DIGITS_CONTOUR_PATCH_SIZES,
                line_scale=s.DIGIT_SKETCH_LINE_SCALE,
                angle_in_degrees=s.MNIST_ANGLE_IN_DEGREES,
                angle_range=(s.MNIST_ANGLE_MIN, s.MNIST_ANGLE_MAX),
                scale=s.DIGITS_CONTOUR_SCALE,
                line_width=s.DIGITS_CONTOUR_LINE_WIDTH,
                gap=s.DIGITS_CONTOUR_GAP,
                grid_color=s.DIGIT_SKETCH_GRID_COLOR,
                lost_color=s.DIGIT_SKETCH_LOST_COLOR,
                entity_path=s.DIGITS_CONTOUR_ENTITY_PATH,
                sobel_path=s.MNIST_SOBEL_ENTITY_PATH if upscale else None,
                pixel_scale=s.MNIST_SOBEL_UPSCALE if upscale else 1,
            )
            sampler = tools.MnistDigitSampler(
                root=s.DIGIT_SKETCH_MNIST_ROOT,
                split=split or s.DIGIT_SKETCH_SPLIT,
                seed=s.DIGITS_CONTOUR_SEED if seed is None else seed,
            )
            extractors = {size: system.extractor(size, contour_only=True, upscale=upscale) for size in s.DIGITS_CONTOUR_PATCH_SIZES}
            for digit in s.DIGITS_CONTOUR_DIGITS:
                stimulus = sampler.sample(digit)
                cells = [
                    tools.SheetCell(
                        patch_size=size,
                        measurements=extractor.extract(stimulus.image, stimulus.label)[stimulus.label],
                        contour_patches=extractor.contour_patches(stimulus.image),
                    )
                    for size, extractor in extractors.items()
                ]
                first = next(iter(extractors.values()))
                sheet.add(stimulus, cells, display_image=first.work_image(stimulus.image) if upscale else None)
            target = path or s.DIGITS_CONTOUR_OUTPUT_PATH
            sheet.save(target)
        return target


class Coding(_Stage):
    """Этап 2. Первичное кодирование (§2, §3.2, §4; П.4–П.7). С него цикл начинается снова на каждом уровне (↻).

    Угол и место становятся разреженным кодом; окно — кодом «что и где».
    """

    number = 2

    def sticks(self, digits: Digits) -> Elements:
        """Кубик. Элементы уровня 0: углопозиции цифр и коды их направлений (§4.4).

        Место здесь не кодируется — оно входит в код окна (`keyholes`).

        Args:
            digits: цифры прогона (`Stimuli.digits`).

        Returns:
            `Elements` уровня 0.
        """
        with self._session.active():
            elements = system.stick_coder().encode(digits._groups.atoms)
        return Elements(self._session, elements, digits, level=0)

    def keyholes(self, keyholes: Keyholes) -> KeyholeCodes:
        """Кубик. Код каждого окна внимания: биты места (OR) и коды его элементов под маской своего места (П.4–П.7).

        Args:
            keyholes: окна уровня (`Cycle.keyholes`).

        Returns:
            `KeyholeCodes`: код на каждое окно.

        Raises:
            LevelFailed: коды окон слились в одну точку.
        """
        elements = keyholes.elements
        with self._session.active():
            try:
                codes = self._step(elements.digits).encode(elements._elements, keyholes._keyholes)
            except system.level_failed as error:
                raise LevelFailed(str(error)) from error
        return KeyholeCodes(self._session, codes, keyholes)

    def encode(self, angle: float, x: float, y: float) -> Code:
        """Прогон. Код стимула (угол, X, Y) первичным кодером команды `layout`: общий код всех координат (§2.2.1, §4).

        Args:
            angle: угол, градусы.
            x: координата X.
            y: координата Y.

        Returns:
            `Code`: длина кода и номера единичных бит.
        """
        with self._session.active():
            encoder = system.encoder()
            _, bits = encoder.encode(float(angle), float(x), float(y))
            return Code(int(encoder.code_length), tuple(int(i) for i in bits.active_indices()))

    def detectors_picture(self, angle: float | None = None, x: float | None = None, y: float | None = None, *,
                          path: str | None = None) -> str:
        """Прогон. Картинка детекторов первичного кодера: угол — кольца, X и Y — полосы, код стимула (§4.1–§4.4).

        Без значений — пример из настроек `DETECTORS_VIEW_STIMULUS`; `angle=None` при заданных `x`, `y` — все
        детекторы без стимула.

        Args:
            angle: угол, градусы.
            x: координата X.
            y: координата Y.
            path: куда сохранить; `None` — `DETECTORS_VIEW_OUTPUT_PATH`.

        Returns:
            Путь к картинке.
        """
        with self._session.active():
            s = system.settings
            view = system.detector_view(
                system.encoder(),
                min_hue=s.LAYOUT_PSEUDO_COLOR_MIN_HUE,
                max_hue=s.LAYOUT_PSEUDO_COLOR_MAX_HUE,
                inactive_alpha=s.DETECTORS_VIEW_INACTIVE_ALPHA,
                panel=s.DETECTORS_VIEW_PANEL,
                code_columns=s.DETECTORS_VIEW_CODE_COLUMNS,
                code_cell=s.DETECTORS_VIEW_CODE_CELL,
            )
            values = tuple(s.DETECTORS_VIEW_STIMULUS) if angle is None and x is None and y is None else (
                None if angle is None else (float(angle), float(x), float(y))
            )
            target = path or s.DETECTORS_VIEW_OUTPUT_PATH
            view.save(target, values)
        return target


class Layout(_Stage):
    """Этап 3. Раскладка (§5, §7.1.7): коды ложатся на двумерную карту, похожие рядом.

    Для цикла нужен кубик `memory`: одна карта памяти на все окна уровня.
    """

    number = 3

    def memory(self, codes: KeyholeCodes, *, map_points: int | None = None) -> Memory:
        """Кубик. Карта памяти уровня из кодов окон обучающих картинок (§5); похожие коды сливаются в образец (§7.1.7).

        Args:
            codes: коды окон (`Coding.keyholes`).
            map_points: предел точек карты; `None` — `LADDER_MAP_MAX_POINTS`. С пределом окна берутся вразброс.

        Returns:
            `Memory`: карта на видеокарте; освободить — `Memory.release`.

        Raises:
            LevelFailed: окон слишком мало для карты.
        """
        keyholes = codes.keyholes
        with self._session.active():
            try:
                memory = self._step(keyholes.elements.digits, map_points).remember(codes._codes, keyholes._keyholes)
            except system.level_failed as error:
                raise LevelFailed(str(error)) from error
        return Memory(self._session, memory, codes)

    def build(self, count: int, *, label: int | None = None, rebuild: bool = False) -> LayoutSummary:
        """Прогон без цикла. Раскладка углопозиций обучающих цифр, без детекторов и эмбеддингов; результат кешируется в `data/`.

        Args:
            count: сколько обучающих цифр.
            label: взять только эту цифру, 0…9; `None` — все.
            rebuild: `True` — не брать кеш, считать заново.

        Returns:
            `LayoutSummary`: число точек и сетка карты.
        """
        with self._session.active():
            layout = system.first_level(count=count, label=label, reuse_cache=not rebuild, layout_only=True).run()
            return LayoutSummary(points=len(layout.cells_array()), width=int(layout.width), height=int(layout.height))

    def pinwheel(self) -> PinwheelReport:
        """Прогон. Тест пинвила (§5.13.2): 360 кодов угла на каждую позицию в одной раскладке.

        Returns:
            `PinwheelReport`: качество карты по углу и пути к картинкам.
        """
        with self._session.active():
            _, quality = system.pinwheel().run()
            s = system.settings
            return PinwheelReport(
                mean_neighbour_delta=float(quality.mean_neighbour_delta),
                sharp_transition_ratio=float(quality.sharp_transition_ratio),
                singularities=int(quality.singularities),
                images=(s.PINWHEEL_OUTPUT_PATH, s.PINWHEEL_COMPONENT_OUTPUT_PATH, s.PINWHEEL_ENERGY_OUTPUT_PATH),
            )


class Detectors(_Stage):
    """Этап 4. Детекторы и активации (§6.1–§6.5, §7.1.10–§7.1.11): иерархия детекторов над картой и их срабатывание.

    Кубики: `build` строит детекторы над картой памяти, `fire` (или `MapDetectors.fire`) проверяет, какие из них
    сработали на кодах окон.
    """

    number = 4

    def build(self, memory: Memory) -> MapDetectors:
        """Кубик. Детекторы над картой памяти по слоям порогов λ_d (§6.4, §7.1.10) и порог активации λ_a (§7.1.11).

        Args:
            memory: карта памяти уровня (`Layout.memory`).

        Returns:
            `MapDetectors`: детекторы по слоям.

        Raises:
            LevelFailed: над картой не нашлось детекторов.
        """
        with self._session.active():
            try:
                detectors = self._step(memory.codes.keyholes.elements.digits).detect(memory._memory)
            except system.level_failed as error:
                raise LevelFailed(str(error)) from error
        return MapDetectors(self._session, detectors, memory)

    def fire(self, detectors: MapDetectors, codes: KeyholeCodes) -> Firing:
        """Кубик. Какие детекторы сработали на каждом окне (§6.1, §6.5); то же, что `detectors.fire(codes)`.

        Уровни активации нормируются в [0, 1] по окнам обучающих картинок (`LADDER_LEVEL_NORMALISATION`; способ нормировки —
        доработка автора проекта, вне статьи: §7.1.12 и §9.3 требуют лишь значений в [0, 1]).

        Args:
            detectors: детекторы над картой (`build`).
            codes: коды окон.

        Returns:
            `Firing`: срабатывания.
        """
        return detectors.fire(codes)


class Embeddings(_Stage):
    """Этап 5. Эмбеддинги (§6.5, §6.6, §7.1.12; П.1): код детали по сработавшим детекторам.

    Кубик `details` даёт элементы следующего уровня; `build` — проход без цикла.
    """

    number = 5

    def details(self, firing: Firing) -> Elements:
        """Кубик. Детали окон — эмбеддинги по сработавшим детекторам (§6.5, §7.1.12).

        Деталь стоит в центре окна, её уровень активации — самый сильный детектор (П.1). Это элементы
        следующего уровня: с них цикл снова идёт на этап 2. Длина кода детали подбирается только по окнам
        обучающих картинок.

        Args:
            firing: срабатывания (`MapDetectors.fire`).

        Returns:
            `Elements` уровня окон.
        """
        keyholes = firing.codes.keyholes
        digits = keyholes.elements.digits
        with self._session.active():
            _, _, _, elements = self._step(digits).embed(firing._firing, firing.detectors._detectors, keyholes._keyholes)
        return Elements(self._session, elements, digits, level=keyholes.level)

    def build(self, count: int, *, samples: int | None = None, label: int | None = None,
              rebuild: bool = False) -> LayoutSummary:
        """Прогон без цикла. Раскладка, детекторы и база эмбеддингов обучающих цифр; кеш — в `data/`.

        Args:
            count: сколько обучающих цифр.
            samples: сколько эмбеддингов записать в базу; `None` — по настройкам.
            label: взять только эту цифру, 0…9; `None` — все.
            rebuild: `True` — не брать кеш, считать заново.

        Returns:
            `LayoutSummary`: число точек и сетка карты.
        """
        with self._session.active():
            layout = system.first_level(
                count=count, label=label, activation_samples=samples, reuse_cache=not rebuild, layout_only=False
            ).run()
            return LayoutSummary(points=len(layout.cells_array()), width=int(layout.width), height=int(layout.height))


class Cycle(_Stage):
    """Этап 6. Цикл (§6.2, §6.6; П.2, П.3, П.9): детали уровня снова идут на этап 2 (↺), пока окно не увидит цифру целиком.

    Кубики: `keyholes`, `step`, `covers`, `trained` (обученный цикл для новых картинок — дашборд). Готовый прогон —
    `run` (команда `ladder`).

    Example:
        >>> elements = damp.coding.sticks(damp.stimuli.digits(train=100, test=500))
        >>> while True:
        ...     level = damp.cycle.step(elements)
        ...     elements = level.details
        ...     if damp.cycle.covers(level):
        ...         break
    """

    number = 6

    def __init__(self, session: Session, stages: dict[str, _Stage]) -> None:
        super().__init__(session)
        self._stages = stages

    def keyholes(self, elements: Elements) -> Keyholes:
        """Кубик. Окна внимания следующего уровня: окно на каждом элементе, радиус — по кластеру похожих соседей.

        Радиус — по §6.4.1–§6.4.3 и П.2; окно растёт от уровня к уровню (П.3).

        Args:
            elements: элементы уровня `n` (штрихи или детали).

        Returns:
            `Keyholes` уровня `n + 1`.

        Raises:
            LevelFailed: окон на обучающих картинках меньше `LADDER_MAP_MIN_POINTS`.
        """
        number = elements.level + 1
        with self._session.active():
            keyholes = self._step(elements.digits).keyholes(elements._elements, number)
            if keyholes.taught < system.settings.LADDER_MAP_MIN_POINTS:
                raise LevelFailed(f"уровень {number}: окон на обучающих картинках — {keyholes.taught}")
        return Keyholes(self._session, keyholes, elements)

    def step(self, elements: Elements, *, map_points: int | None = None) -> Level:
        """Кубик. Один проход цикла кубиками этапов 6 → 2 → 3 → 4 → 5: окна, коды, память, детекторы, срабатывания, детали.

        Детали (`level.details`) — вход следующего шага. Карта на видеокарте после прохода освобождается.

        Args:
            elements: элементы текущего уровня.
            map_points: предел точек карты; `None` — `LADDER_MAP_MAX_POINTS`.

        Returns:
            `Level`: все части прохода.

        Raises:
            LevelFailed: уровень не сложился.
        """
        stages = self._stages
        keyholes = self.keyholes(elements)
        codes = stages["coding"].keyholes(keyholes)
        memory = stages["layout"].memory(codes, map_points=map_points)
        detectors = stages["detectors"].build(memory)
        firing = detectors.fire(codes)
        details = stages["embeddings"].details(firing)
        memory.release()  # карта на видеокарте больше не нужна: детекторы и детали уже посчитаны
        return Level(keyholes=keyholes, codes=codes, memory=memory, detectors=detectors, firing=firing, details=details)

    def covers(self, level: Level) -> bool:
        """Кубик. Окно уровня уже видит цифру целиком: доля углопозиций не меньше `LADDER_TOP_COVERAGE`.

        Args:
            level: проход уровня (`step`).

        Returns:
            `True` — цикл окончен, верхний уровень достигнут.
        """
        with self._session.active():
            return level.keyholes.picture_seen >= float(system.settings.LADDER_TOP_COVERAGE)

    def trained(self, train: int | None = None, *, map_points: int | None = None, decay: float | None = None) -> TrainedLadder:
        """Кубик. Обученный цикл и его линейный читатель (§6.6, §9.3): по ним идёт любая новая картинка 28×28.

        Цикл учится на `train` обучающих цифрах с их видами (`LADDER_TRAIN_VIEWS`, §7.1.4), без проверочных; шаги —
        из кеша, если такой цикл уже был. Читатель учится на пиках срабатываний уровней обучающих картинок.

        Args:
            train: сколько обучающих цифр; `None` — `DASHBOARD_TRAIN`.
            map_points: предел точек карты уровня; `None` — `DASHBOARD_MAP_POINTS`.
            decay: штраф L2 линейного читателя; `None` — `DASHBOARD_LINEAR_DECAY`.

        Returns:
            `TrainedLadder`: `see(картинка)` — путь картинки по уровням и вероятности цифр; `maps()` — карты уровней.

        Raises:
            RuntimeError: цикл не построил ни одного уровня.

        Example:
            >>> ladder = damp.cycle.trained(100)
            >>> ladder.see(picture).digit
        """
        with self._session.active():
            s = system.settings
            ladder = system.trained_ladder(
                int(s.DASHBOARD_TRAIN if train is None else train),
                map_points=s.DASHBOARD_MAP_POINTS if map_points is None else int(map_points),
                decay=float(s.DASHBOARD_LINEAR_DECAY if decay is None else decay),
            )
        return TrainedLadder(self._session, ladder)

    def run(self, train: int, test: int, *, split: str = "test", map_points: int | None = None,
            shifts: tuple[tuple[int, int], ...] | None = None) -> CycleReport:
        """Прогон. Цикл уровней на обучающих цифрах и проверка на проверочных (команда `ladder`).

        Args:
            train: сколько обучающих цифр.
            test: сколько проверочных цифр.
            split: набор MNIST для проверочных: `"test"` или `"train"`.
            map_points: предел точек карты уровня; `None` — `LADDER_MAP_MAX_POINTS`.
            shifts: сдвиги `(dy, dx)` проверочных; `None` — `LADDER_SHIFTS`.

        Returns:
            `CycleReport`: мерки уровней и голосование верхних деталей.
        """
        with self._session.active():
            s = system.settings
            runner = system.cycle(
                count=int(train),
                evaluate_limit=int(test),
                split=split,
                map_max_points=s.LADDER_MAP_MAX_POINTS if map_points is None else int(map_points),
                shifts=s.LADDER_SHIFTS if shifts is None else tuple(shifts),
            )
            runner.run()
        levels = tuple(
            CycleLevel(
                number=r.number,
                field_radius=float(r.field_radius),
                picture_seen=float(r.picture_seen),
                keyholes_per_picture=float(r.fields_per_picture),
                elements_per_keyhole=float(r.elements_per_field),
                map_points=int(r.map_points),
                detectors=int(r.detectors),
                same_label=float(r.same_label),
                response=r.alone,
                peak=r.peak,
                glued=r.glued,
                energy_mean=float(r.energy_mean),
                code_resolution=float(r.code_resolution),
                code_largest=int(r.code_largest),
                detail_bits=float(r.detail_bits),
                detail_empty=float(r.detail_empty),
                detail_full=float(r.detail_full),
            )
            for r in runner.reports
        )
        return CycleReport(levels=levels, vote=runner.top_accuracy)


class Recognition(_Stage):
    """Этап 7. Распознавание (§2.2.5; мерка вне статьи): детали верхнего уровня голосуют за класс цифры.

    Кубики цикла: `by_response`, `vote`, `linear`. Проход без цикла: `evaluate`, `predict`, `predict_dataset`,
    `calibrate`. Честное сравнение с kNN по пикселям и обычной CNN: `bench`.
    """

    number = 7

    def linear(self, firings: Firing | list[Firing] | tuple[Firing, ...], decays: tuple[float, ...] | None = None) -> dict[float, float]:
        """Кубик. Доля верных ответов линейного читателя над нормированными уровнями активации иерархии детекторов (§9.3).

        §9.3: уровни активации — эмбеддинг в ℝⁿ, вход нейросети. Эмбеддинг картинки — самый сильный уровень
        активации каждого детектора по её окнам, у каждого уровня свой сегмент единичной длины; класс читает один
        линейный слой с softmax, обученный на обучающих картинках (вместе с видами) со штрафом L2 `λ` на веса.
        Читатель — с учителем и вне памяти DAMP; проверочные на него не влияют.

        Args:
            firings: срабатывания уровней (`Level.firing` или `MapDetectors.fire`): один уровень или уровни 1…k
                одного прогона.
            decays: штрафы `λ`; `None` — `BENCH_LINEAR_DECAYS`.

        Returns:
            `{λ: доля верных ответов}` на проверочных.

        Raises:
            ValueError: проверочных картинок нет (`digits(train, test)` с `test > 0`), срабатываний нет, или `λ`
                не заданы.

        Example:
            >>> levels = [damp.cycle.step(elements)]  # и дальше по деталям уровней
            >>> damp.recognition.linear([level.firing for level in levels])
        """
        import numpy as np

        items = [firings] if isinstance(firings, Firing) else list(firings)
        if not items:
            raise ValueError("нет срабатываний: передайте Level.firing хотя бы одного уровня")
        groups = items[0].codes.keyholes.elements.digits._groups
        if "test" not in groups.ranges:
            raise ValueError("нет проверочных картинок: digits(train, test) с test > 0")
        with self._session.active():
            values = tuple(float(value) for value in (system.settings.BENCH_LINEAR_DECAYS if decays is None else decays))
            if not values:
                raise ValueError("нет λ: задайте decays или BENCH_LINEAR_DECAYS")
            recognition = system.level_recognition(groups.labels, groups.ranges["train"])
            peaks = []
            for firing in items:
                found = firing._firing
                _, peak = recognition.responses(
                    firing.codes.keyholes._keyholes.pieces.picture, found.rows, found.columns, found.levels,
                    groups.pictures, firing.detectors.count,
                )
                peaks.append(peak)
            first, last = groups.ranges["test"]
            answers = system.linear_reader().answers_for(
                np.concatenate(peaks, axis=1), groups.labels, groups.ranges["train"], {"test": (first, last)}, values
            )
        truth = groups.labels[first:last]
        return {decay: float(np.mean((found["test"] == truth) & (found["test"] >= 0))) for decay, found in answers.items()}

    def by_response(self, firing: Firing) -> dict[int, float]:
        """Кубик. Доля верных ответов по отклику картинки на уровне: сколько раз сработал каждый детектор на её окнах.

        Ответ — класс k самых похожих откликов обучающих (§2.2.5).

        Args:
            firing: срабатывания уровня (`MapDetectors.fire`).

        Returns:
            `{k: доля верных ответов}` на проверочных.

        Raises:
            ValueError: проверочных картинок нет (`digits(train, test)` с `test > 0`).
        """
        keyholes = firing.codes.keyholes
        digits = keyholes.elements.digits
        groups = digits._groups
        if "test" not in groups.ranges:
            raise ValueError("нет проверочных картинок: digits(train, test) с test > 0")
        with self._session.active():
            recognition = system.level_recognition(groups.labels, groups.ranges["train"])
            found = firing._firing
            response, _ = recognition.responses(
                keyholes._keyholes.pieces.picture, found.rows, found.columns, found.levels, groups.pictures, firing.detectors.count
            )
            return recognition.by_response(response, groups.ranges["test"])

    def vote(self, details: Elements) -> dict[int, float]:
        """Кубик. Доля верных ответов голосованием деталей: каждая деталь проверочной картинки голосует классами k самых похожих деталей обучающих (§2.2.5).

        Для верхнего уровня — ответ цикла.

        Args:
            details: детали уровня (`Embeddings.details` или `Level.details`).

        Returns:
            `{k: доля верных ответов}` на проверочных.

        Raises:
            ValueError: проверочных картинок нет (`digits(train, test)` с `test > 0`).
        """
        groups = details.digits._groups
        if "test" not in groups.ranges:
            raise ValueError("нет проверочных картинок: digits(train, test) с test > 0")
        found = details._elements
        with self._session.active():
            recognition = system.level_recognition(groups.labels, groups.ranges["train"])
            return recognition.by_vote(found.codes(), found.what_length, found.picture, groups.ranges["test"])

    def bench(self, train: int | None = None, sets: int | None = None, validation: int | None = None, test: int | None = None,
              map_points: int | None = None, methods: tuple[str, ...] | None = None,
              augment: tuple[str, ...] | None = None) -> BenchReport:
        """Прогон. Стенд честного сравнения DAMP, kNN по пикселям и обычной CNN на одних и тех же картинках (команда `bench`; вне статьи).

        Картинки — случайная выборка номеров MNIST с закреплённым зерном (`BENCH_SEED`), а не сплошные блоки: MNIST train
        не перемешан по авторам, и соседние блоки заметно отличаются по трудности. Обучающие наборы (по `train` картинок,
        не пересекаются) берутся из пула `BENCH_TRAIN_POOL`, отложенная выборка `validation` — из `BENCH_VALIDATION_POOL`;
        те же картинки у всех методов. Отложенная выборка делится на две части. **select** (доля `BENCH_SELECT_SHARE`):
        по ней выбираются k у DAMP и kNN, конфигурация CNN и момент ранней остановки CNN. **report**: в выборе не участвует,
        по ней считаются итоговая точность `validation` каждого метода и разрыв до CNN. `test` (MNIST t10k) считается
        лишь для выбранных настроек, один раз в конце этапа, и на выбор не влияет. Итог по методу — среднее, стандартное
        отклонение и 95 % t-интервал по наборам; у разрыва DAMP − CNN парный интервал по наборам; ошибка самой конечной
        выборки report — биномиальная. Мерки статьи по уровням DAMP (§5.8, §5.13.2.5, §7.1.8.1) и устойчивость к
        сдвигу на пиксели (доля неизменившихся ответов) — в отчёте; все цифры, настройки и номера картинок — в JSON из
        `BenchReport.path`. Каждый DAMP-прогон идёт в своей записи Rerun со всеми вкладками, сама запись «bench» — таблица
        результатов и ход обучения CNN.

        Args:
            train: обучающих картинок в наборе; `None` — `BENCH_TRAIN`.
            sets: сколько наборов; `None` — `BENCH_SETS`.
            validation: картинок отложенной выборки (select + report, не меньше 2); `None` — `BENCH_VALIDATION`.
            test: проверочных картинок t10k; `None` — `BENCH_TEST` (0 — не считать).
            map_points: предел точек карты уровня у DAMP; `None` — `BENCH_MAP_POINTS`.
            methods: из `"damp"`, `"pixels"`, `"cnn"`; `None` — `BENCH_METHODS`.
            augment: режимы аугментации, `"none"` и `"shifts"`; `None` — `BENCH_AUGMENT`. У DAMP сдвиги задают виды
                обучающих `LADDER_TRAIN_VIEWS` (§7.1.4), у kNN `"shifts"` не поддержан (в отчёте строка без точностей).

        Returns:
            `BenchReport`: методы, уровни DAMP, разрыв до CNN и путь к JSON.

        Raises:
            ValueError: неизвестный метод или режим; `train`, `sets` не положительны, `test` отрицателен; в отложенной
                выборке меньше двух картинок (её нельзя разделить на select и report); обучающие наборы, отложенная
                выборка или test не помещаются в свои пулы (`train * sets` больше `BENCH_TRAIN_POOL` и т. п.).

        Example:
            >>> report = damp.recognition.bench(train=100, sets=5, validation=500, test=500)
            >>> [(m.method, m.augment, m.setting, round(m.validation.mean, 3)) for m in report.methods if m.supported]
            >>> report.gap["damp_minus_best_cnn"]["validation"]  # доля, по report
        """
        with self._session.active():
            result = system.bench(
                train=train, sets=sets, validation=validation, test=test, map_points=map_points,
                methods=methods, augment=augment,
            ).run()
        protocol = result["protocol"]

        def score(found: dict | None) -> BenchScore | None:
            if found is None:
                return None
            return BenchScore(
                mean=float(found["mean"]), std=float(found["std"]),
                ci_low=None if found["ci_low"] is None else float(found["ci_low"]),
                ci_high=None if found["ci_high"] is None else float(found["ci_high"]),
                per_set=tuple(float(v) for v in found["per_set"]),
            )

        def robustness(found: dict | None) -> BenchRobustness | None:
            if found is None:
                return None
            return BenchRobustness(
                pictures=int(found["pictures"]), shift=(int(found["shift"][0]), int(found["shift"][1])),
                same_answers=float(found["same_answers"]), accuracy_plain=float(found["accuracy_plain"]),
                accuracy_shifted=float(found["accuracy_shifted"]),
            )

        methods_found = tuple(
            BenchMethod(
                method=row["method"], augment=row["augment"], supported=bool(row["supported"]), setting=row["setting"],
                candidates={str(k): float(v) for k, v in row["candidates"].items()},
                validation=score(row["validation"]), test=score(row["test"]), robustness=robustness(row["robustness"]),
                note=str(row["note"]),
            )
            for row in result["methods"]
        )
        levels = tuple(
            BenchLevel(
                number=int(level["number"]), sets=int(level["sets"]), picture_seen=float(level["picture_seen"]),
                map_points=int(round(level["map_points"])), detectors=int(round(level["detectors"])),
                same_label=float(level["same_label"]), energy_mean=float(level["energy_mean"]),
                code_resolution=float(level["code_resolution"]), code_largest=int(round(level["code_largest"])),
                detail_bits=float(level["detail_bits"]), detail_empty=float(level["detail_empty"]),
                detail_full=float(level["detail_full"]),
                alone={int(k): float(v) for k, v in level["alone"].items()},
                glued={int(k): float(v) for k, v in level["glued"].items()},
                response_cosine=level["response_cosine"],
            )
            for level in result["levels"]
        )
        return BenchReport(
            methods=methods_found, levels=levels, train=int(protocol["train"]), sets=int(protocol["sets"]),
            validation=int(protocol["validation"]), test=int(protocol["test"]), map_points=int(protocol["map_points"]),
            gap=result["gap"], path=result["path"], seconds=dict(result["seconds"]),
        )

    def evaluate(self, count: int, *, split: str = "test", limit: int | None = None, temperature: float | None = None,
                 layout_path: str | None = None) -> Evaluation:
        """Прогон без цикла. Точность на картинках выборки по базе эмбеддингов обучающих цифр.

        Args:
            count: сколько обучающих цифр в базе.
            split: выборка MNIST: `"test"` или `"train"`.
            limit: сколько картинок проверить; `None` — `DECODER_DATASET_LIMIT`.
            temperature: температура softmax; `None` — из настроек.
            layout_path: файл раскладки; `None` — по умолчанию.

        Returns:
            `Evaluation`: число картинок и доля верных ответов.
        """
        with self._session.active():
            limit = system.settings.DECODER_DATASET_LIMIT if limit is None else int(limit)
            _, accuracy = system.decoder(count=count, layout_path=layout_path).predict_dataset(
                split=split, limit=limit, temperature=temperature, log_first=False
            )
        return Evaluation(split=split, samples=limit, accuracy=float(accuracy))

    def predict(self, image: str | Path, count: int, *, temperature: float | None = None,
                layout_path: str | None = None) -> Prediction:
        """Прогон без цикла. Цифра на картинке по базе эмбеддингов обучающих цифр.

        Args:
            image: файл картинки 28×28 (см. `client/examples/`).
            count: сколько обучающих цифр в базе.
            temperature: температура softmax; `None` — из настроек.
            layout_path: файл раскладки; `None` — по умолчанию.

        Returns:
            `Prediction`: цифра, уверенность и вероятности всех цифр.
        """
        with self._session.active():
            result = system.decoder(count=count, layout_path=layout_path).predict_image_from_path(
                str(image), temperature=temperature, log_image=True
            )
        return Prediction(int(result.predicted_class), float(result.confidence), tuple(float(p) for p in result.probabilities))

    def predict_dataset(self, count: int, *, split: str = "test", limit: int | None = None,
                        temperature: float | None = None, layout_path: str | None = None) -> Evaluation:
        """Прогон без цикла. То же, что `evaluate`, плюс журнал первых ответов.

        Args:
            count: сколько обучающих цифр в базе.
            split: выборка MNIST: `"test"` или `"train"`.
            limit: сколько картинок проверить; `None` — `DECODER_DATASET_LIMIT`.
            temperature: температура softmax; `None` — из настроек.
            layout_path: файл раскладки; `None` — по умолчанию.

        Returns:
            `Evaluation`.
        """
        with self._session.active():
            limit = system.settings.DECODER_DATASET_LIMIT if limit is None else int(limit)
            _, accuracy = system.decoder(count=count, layout_path=layout_path).predict_dataset(
                split=split, limit=limit, temperature=temperature, log_first=True
            )
        return Evaluation(split=split, samples=limit, accuracy=float(accuracy))

    def calibrate(self, count: int, *, split: str = "test", limit: int | None = None,
                  layout_path: str | None = None) -> tuple[float, float]:
        """Прогон без цикла. Подбор температуры softmax с лучшей точностью.

        Args:
            count: сколько обучающих цифр в базе.
            split: выборка MNIST: `"test"` или `"train"`.
            limit: сколько картинок проверить; `None` — `DECODER_DATASET_LIMIT`.
            layout_path: файл раскладки; `None` — по умолчанию.

        Returns:
            Пара `(температура, точность)`; точность — доля верных ответов, 0…1.
        """
        with self._session.active():
            limit = system.settings.DECODER_DATASET_LIMIT if limit is None else int(limit)
            temperature, accuracy = system.decoder(count=count, layout_path=layout_path).calibrate_temperature(split=split, limit=limit)
        return float(temperature), float(accuracy)


class Figures:
    """Рисунки README: прогон цикла с наблюдателем, тест пинвила, схемы — всё в `docs/`.

    Поле `Damp.figures`; не этап пайплайна.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def readme(self) -> str:
        """Все рисунки README: прогон цикла, тест пинвила и схемы (около 5 минут).

        Returns:
            Папка с рисунками (`README_FIGURES_DIR`).
        """
        with self._session.active():
            system.readme_figures().run()
            return str(system.settings.README_FIGURES_DIR)

    def structure(self) -> str:
        """Только рисунок устройства проекта (`docs/structure.png`), около секунды.

        Returns:
            Путь к рисунку.
        """
        with self._session.active():
            return system.readme_figures().structure()

    def rerun(self) -> list[str]:
        """Скриншоты вкладок цикла в Rerun для README: Обзор и этапы 1–7 на прогоне README.

        Прогон пишется в `.rrd` (из кеша шагов — около полуминуты), потом каждая вкладка ненадолго открывается в
        окне Rerun и снимается (`RERUN_SHOT_DELAY` секунд на вкладку).

        Returns:
            Пути картинок `docs/rerun-<номер>-<вкладка>.png`.
        """
        with self._session.active():
            return system.rerun_screenshots().run()
