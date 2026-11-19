"""Простые результаты фреймворка: неизменяемые записи без системных объектов.

Их возвращают готовые прогоны (`Damp.cycle.run`, `Damp.recognition.evaluate` и другие) и описания этапов.
Все записи — `dataclass(frozen=True)`: поля читаются, менять их нельзя.

| Тип | Что описывает | Кто возвращает |
|---|---|---|
| `Stage` | этап пайплайна | `Damp.stages`, поле `about` каждого этапа |
| `Stimulus`, `Picture` | углопозиция и картинка прогона | `Stimuli.of_image`, `Digits.picture` |
| `Code` | разреженный код | `Coding.encode` |
| `LayoutSummary`, `PinwheelReport` | итог раскладки, тест пинвила | `Layout.build`, `Layout.pinwheel` |
| `CycleLevel`, `CycleReport` | уровень цикла и итог цикла | `Cycle.run` |
| `Sight`, `SightLevel` | путь новой картинки через обученный цикл | `TrainedLadder.see` |
| `LevelMap`, `TrainingSticks` | карты уровней обученного цикла и углопозиции обучающих картинок | `TrainedLadder.maps`, `TrainedLadder.sticks` |
| `Prediction`, `Evaluation` | ответ и точность распознавания | `Recognition.predict`, `Recognition.evaluate` |
| `BenchScore`, `BenchRobustness`, `BenchMethod`, `BenchLevel`, `BenchReport` | честное сравнение DAMP, kNN и CNN | `Recognition.bench` |
| `LevelFailed` | ошибка: уровень цикла не сложился | кубики цикла |
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Stage:
    """Этап пайплайна: номер, название, что делает, где в коде, по каким источникам.

    Пайплайн из семи этапов: 1 стимулы, 2 первичное кодирование (начало цикла), 3 раскладка, 4 детекторы и
    активации, 5 эмбеддинги, 6 цикл (конец цикла), 7 распознавание.

    Attributes:
        number: номер этапа, 1…7.
        name: имя поля этапа у `Damp` латиницей: `"stimuli"`, `"coding"`, `"layout"`, `"detectors"`,
            `"embeddings"`, `"cycle"`, `"recognition"`.
        title: название этапа по-русски.
        does: что этап делает, одной фразой.
        code: папка этапа в системном слое, например `"system/damp/encoding/"`.
        sources: источники: параграфы статьи (`§6.5`), разделы продолжения (`П.5`) или «вне статьи».
        cycle: `"начало"` у этапа, с которого цикл повторяется (этап 2); `"конец"` у этапа, откуда детали
            возвращаются на этап 2 (этап 6); у остальных пусто.
    """

    number: int
    name: str
    title: str
    does: str
    code: str
    sources: str
    cycle: str = ""


@dataclass(frozen=True)
class Stimulus:
    """Углопозиция, или штрих (вне статьи): угол края контура цифры и клетка сетки 24×24, где он найден.

    Attributes:
        angle: направление перепада яркости, градусы; период 360.
        x: место по горизонтали в клетках сетки: целое — середина клетки; дробное — где край проходит внутри клетки
            (`MNIST_SUBCELL_POSITION`).
        y: то же по вертикали.
        span: полудлина поля штриха вдоль края, в клетках: расстояние от центра детектора до самой дальней точки его поля (§6.4.3);
            у штриха по сетке — 0.5, половина клетки.
    """

    angle: float
    x: float
    y: float
    span: float = 0.5


@dataclass(frozen=True)
class Picture:
    """Картинка прогона вместе с цифрой и найденными на ней углопозициями.

    Attributes:
        index: номер картинки в прогоне (обучающие идут первыми).
        digit: цифра на картинке, 0…9.
        group: группа картинки: `"train"`, `"validation"`, `"test"`, сдвиг на клетки `"dy,dx"` или сдвиг на
            пиксели `"pixel:dy,dx"`.
        image: сама картинка, массив numpy 28×28 со значениями 0…255.
        stimuli: углопозиции картинки.
        mnist: номер картинки в своём наборе MNIST (train либо t10k); -1 — неизвестен.
    """

    index: int
    digit: int
    group: str
    image: object  # numpy-массив 28×28
    stimuli: tuple[Stimulus, ...]
    mnist: int = -1


class LevelFailed(RuntimeError):
    """Уровень цикла не сложился: окон на обучающих картинках меньше `LADDER_MAP_MIN_POINTS`, коды окон слились
    в одну точку либо над картой памяти не нашлось детекторов.

    Наследует `RuntimeError`. Её поднимают кубики цикла `Cycle.keyholes`, `Coding.keyholes`, `Layout.memory`
    и `Detectors.build`. Готовый прогон `Cycle.run` уровни, что не сложились, обрабатывает сам.

    Example:
        >>> try:
        ...     level = damp.cycle.step(elements)
        ... except LevelFailed as error:
        ...     print("цикл остановлен:", error)
    """


@dataclass(frozen=True)
class Code:
    """Разреженный код (§2): длина и номера единичных бит.

    Attributes:
        length: длина кода в битах.
        ones: номера единичных бит по возрастанию, от 0 до `length - 1`.
    """

    length: int
    ones: tuple[int, ...]

    @property
    def density(self) -> float:
        """Плотность кода: доля единичных бит, от 0 до 1 (0 при нулевой длине)."""
        return len(self.ones) / self.length if self.length else 0.0


@dataclass(frozen=True)
class LayoutSummary:
    """Итог раскладки (§5): сколько точек и какая сетка.

    Attributes:
        points: число точек на карте.
        width: ширина сетки карты, клеток.
        height: высота сетки карты, клеток.
    """

    points: int
    width: int
    height: int


@dataclass(frozen=True)
class PinwheelReport:
    """Тест пинвила (§5.13.2): качество карты по углу и файлы картинок.

    Attributes:
        mean_neighbour_delta: средняя разность угла между соседними клетками, градусы; чем меньше, тем плавнее.
        sharp_transition_ratio: доля резких переходов угла между соседними клетками.
        singularities: число особых точек (центров вертушек).
        images: пути к картинкам теста: угол, его компоненты, энергия.
    """

    mean_neighbour_delta: float
    sharp_transition_ratio: float
    singularities: int
    images: tuple[str, ...]


@dataclass(frozen=True)
class CycleLevel:
    """Один уровень цикла: что видит окно, память уровня и узнавание по его отклику (§6.6; П.2, П.3).

    Attributes:
        number: номер уровня (первый уровень цикла — 1).
        field_radius: сколько клеток картинки видит окно, в среднем по обучающим.
        picture_seen: доля углопозиций своей картинки, которую видит окно, в среднем (0…1).
        keyholes_per_picture: окон на обучающей картинке, в среднем.
        elements_per_keyhole: элементов в окне, в среднем.
        map_points: число точек карты памяти уровня.
        detectors: число детекторов над картой.
        same_label: доля соседних точек карты с одной цифрой (4 соседа); у случайной карты около 0.1.
        response: доля верных ответов по отклику уровня, `{k: доля}` для k самых похожих откликов обучающих;
            `None`, если проверочных картинок нет.
        peak: то же по сильнейшему отклику каждого детектора; `None`, если проверочных нет.
        glued: то же по склейке откликов уровней 1…number; `None`, если проверочных нет.
        energy_mean: качество раскладки Ē карты уровня: средняя нормированная энергия точек (§5.8), 0…1.
        code_resolution: окон обучающих картинок на один различный код детали (§5.13.2.5, «resolution per code»).
        code_largest: самый большой набор окон обучающих картинок с одним кодом детали (§5.13.2.5, «largest cluster size»).
        detail_bits: средняя насыщенность кода детали: единиц на деталь (§7.1.8.1).
        detail_empty: доля обучающих окон с пустой деталью: ни один детектор не сработал, 0…1.
        detail_full: доля обучающих окон, у которых сработавших детекторов не меньше σ (код упёрся в предел), 0…1.
    """

    number: int
    field_radius: float
    picture_seen: float
    keyholes_per_picture: float
    elements_per_keyhole: float
    map_points: int
    detectors: int
    same_label: float
    response: dict[int, float] | None
    peak: dict[int, float] | None
    glued: dict[int, float] | None
    energy_mean: float
    code_resolution: float
    code_largest: int
    detail_bits: float
    detail_empty: float
    detail_full: float


@dataclass(frozen=True)
class CycleReport:
    """Итог цикла: уровни и голосование верхних деталей.

    Attributes:
        levels: уровни по порядку, от первого к верхнему.
        vote: доля верных ответов голосованием деталей верхнего уровня (§2.2.5), `{k: доля}` для k = 1 и 5
            самых похожих деталей; `None`, если проверочных картинок нет.
    """

    levels: tuple[CycleLevel, ...]
    vote: dict[int, float] | None = field(default=None)


@dataclass(frozen=True)
class SightLevel:
    """Что один уровень обученного цикла сделал с новой картинкой: кусочки, зажжённые точки, детекторы, детали.

    Кусочек — окно внимания (П.2) на элементе уровня. Его код зажигает точки карты (§6.1), по ним срабатывают
    детекторы (§6.5), сильнейшие из них дают деталь окна (§6.5, §7.1.12) — элемент следующего уровня (П.1).
    Все поля — массивы numpy; номера окон — внутри картинки (0…K−1), точки — в порядке карты (`LevelMap`),
    детекторы — по слоям, как в `LevelMap.detectors`.

    Attributes:
        number: номер уровня, 1…
        keyhole_centre: центр каждого окна `(y, x)`, узлы сетки картинки, форма (K, 2).
        keyhole_extent: сколько клеток картинки видит окно, (K,).
        entry_keyhole: окно каждого вхождения элемента, (E,).
        entry_element: элемент вхождения: номер углопозиции на уровне 1, номер окна прошлого уровня выше, (E,).
        lit_keyhole: окно, зажёгшее точку карты, (L,).
        lit_point: зажжённая точка карты, (L,).
        lit_value: её активация a_j, (L,).
        fired_keyhole: окно, на котором сработал детектор, (R,).
        fired_detector: сработавший детектор, (R,).
        fired_level: его уровень активации E(d), (R,).
        detail_length: длина кода детали, бит.
        strength: уровень активации детали каждого окна — сильнейший сработавший детектор, (K,).
        peak: сильнейший уровень каждого детектора по окнам картинки, единичной длины (§9.3), (D,).
    """

    number: int
    keyhole_centre: object
    keyhole_extent: object
    entry_keyhole: object
    entry_element: object
    lit_keyhole: object
    lit_point: object
    lit_value: object
    fired_keyhole: object
    fired_detector: object
    fired_level: object
    detail_length: int
    strength: object
    peak: object


@dataclass(frozen=True)
class PrimaryMap:
    """Первичная карта обученного цикла — «уровень 0»: раскладка первичных кодов углопозиций (§4.6, §5).

    Точка карты хранит код углопозиции обучающей цифры — угол и место (этапы 2–3). Рядом стоят штрихи с близкими углом
    и местом: карта, раскрашенная по углу, показывает пинвилы (§5.13.2). С уровнем 0 в цикле (`LADDER_LEVEL0`) штрихи
    идут в окна уровня 1 с кодами деталей этой карты (§6.5, §7.1.12); без него карта только для показа.

    Attributes:
        height: высота карты, клеток.
        width: ширина карты, клеток.
        cells: клетка каждой точки `(y, x)`, (P, 2).
        angle: угол углопозиции каждой точки, градусы; у штрихов по середине линии — 0…180°, (P,).
        y: место углопозиции на сетке картинки по вертикали, в клетках, (P,).
        x: то же по горизонтали, (P,).
        labels: цифра обучающей картинки, с которой взята углопозиция, (P,).
        angle_period: период угла, градусы: 180 у штрихов без стрелки, иначе 360.
        detectors: детекторы над картой, строки `(центр y, центр x, радиус, порог λ_d, энергия e_d)`, (D, 5).
        layer: слой каждого детектора, (D,).
        layer_thresholds: пороги λ_d слоёв по возрастанию.
        activation_threshold: порог зажигания точки λ_a (§6.1).
        in_cycle: карта — уровень 0 цикла (`LADDER_LEVEL0`): элементы уровня 1 — её детали, а не сами углопозиции.
    """

    height: int
    width: int
    cells: object
    angle: object
    y: object
    x: object
    labels: object
    angle_period: float
    detectors: object
    layer: object
    layer_thresholds: tuple[float, ...]
    activation_threshold: float
    in_cycle: bool = False


@dataclass(frozen=True)
class PrimarySight:
    """Что первичная карта («уровень 0») сделала с углопозициями новой картинки (§6.1, §6.5).

    Attributes:
        points: сколько точек на первичной карте.
        detectors: сколько детекторов над ней.
        lit_stick: углопозиция картинки, зажёгшая точку карты, (L,).
        lit_point: зажжённая точка карты, (L,).
        lit_value: её активация a_j, (L,).
        fired_stick: углопозиция, на которой сработал детектор, (R,).
        fired_detector: сработавший детектор, (R,).
        fired_level: его уровень активации E(d), (R,).
    """

    points: int
    detectors: int
    lit_stick: object
    lit_point: object
    lit_value: object
    fired_stick: object
    fired_detector: object
    fired_level: object


@dataclass(frozen=True)
class LevelVotes:
    """Голоса памяти одного уровня для новой картинки (§2.2.5, §7.1.13.3).

    Кусочек голосует цифрой своей самой активной точки памяти — ближайшего воспоминания. Знание уровня — насколько
    память знает то, что видят окна этого уровня: средняя по кусочкам наибольшая активация точки памяти.

    Attributes:
        number: номер уровня, 1…
        votes: доля кусочков уровня за каждую цифру 0…9; все нули — никто не голосовал.
        digit: цифра-победитель; -1 — голосов нет.
        share: доля голосов победителя.
        knowledge: знание уровня, 0…1.
        threshold: порог знания уровня — по обучающим картинкам без своих точек памяти (`RECOGNITION_KNOWLEDGE_QUANTILE`).
        knows: знание не ниже порога.
        keyholes: сколько кусочков (окон) у картинки на уровне.
        voted: сколько из них зажгли хотя бы одну точку памяти.
    """

    number: int
    votes: tuple[float, ...]
    digit: int
    share: float
    knowledge: float
    threshold: float
    knows: bool
    keyholes: int
    voted: int


@dataclass(frozen=True)
class Sight:
    """Путь новой картинки через обученный цикл (`TrainedLadder.see`): углопозиции, уровни и класс.

    Класс читается двумя способами. Память (§2.2.5, §7.1.13.3): кусочек голосует цифрой своей самой похожей точки
    памяти; ответ — цифра с наибольшей средней долей голосов по уровням; уверенность — согласие уровней.
    Линейный читатель (§9.3) — мерка для целых цифр.

    Attributes:
        stick_angle: угол каждой углопозиции, градусы (§4.4).
        stick_y: место на сетке картинки по вертикали, в клетках; дробное при `MNIST_SUBCELL_POSITION`.
        stick_x: то же по горизонтали.
        stick_strength: сила штриха, доля сильнейшего штриха картинки.
        stick_span: полудлина поля штриха вдоль края, в клетках (до самой дальней точки поля детектора, §6.4.3): штрих
            рисуется по длине своего поля.
        levels: уровни, до которых картинка дошла, по порядку.
        probabilities: вероятности цифр 0…9 по линейному читателю (§9.3); все нули — ответа нет.
        digit: самая вероятная цифра по линейному читателю; -1 — ни один детектор не сработал.
        seconds: сколько секунд шёл путь картинки.
        votes: голоса памяти по уровням, до которых картинка дошла (`LevelVotes`).
        verdict: уверенность ответа памяти — согласие уровней: `"agree"` — все уровни назвали эту цифру, `"mostly"` —
            её назвали два верхних уровня, `"split"` — уровни расходятся, `"unknown"` — ни одна точка памяти не зажглась.
        memory_digit: цифра по памяти — с наибольшей средней долей голосов по уровням; -1 — не знаю.
        memory_share: средняя по уровням доля голосов этой цифры.
        memory_agreed: сколько уровней назвали эту цифру победителем своих голосов.
        memory_votes: средние по уровням доли голосов за цифры 0…9; все нули — голосов нет.
        primary: уровень 0 — зажжённые точки и детекторы первичной карты (`PrimarySight`); `None` — карты нет.
    """

    stick_angle: object
    stick_y: object
    stick_x: object
    stick_strength: object
    stick_span: object
    levels: tuple[SightLevel, ...]
    probabilities: tuple[float, ...]
    digit: int
    seconds: float
    votes: tuple[LevelVotes, ...] = ()
    verdict: str = "unknown"
    memory_digit: int = -1
    memory_share: float = 0.0
    memory_agreed: int = 0
    memory_votes: tuple[float, ...] = ()
    primary: "PrimarySight | None" = None


@dataclass(frozen=True)
class LevelMap:
    """Карта уровня обученного цикла для показа: точки, откуда взята каждая, и детекторы над картой (§5, §6.4).

    Точка карты хранит код окна обучающей картинки (§5): по картинке, центру и полю окна видно её содержимое.

    Attributes:
        number: номер уровня.
        height: высота карты, клеток.
        width: ширина карты, клеток.
        cells: клетка каждой точки `(y, x)`, (P, 2).
        labels: цифра обучающей картинки каждой точки, (P,).
        energies: нормированная энергия точки Ê (§5.7), (P,).
        point_picture: обучающая картинка окна точки — номер в `TrainingSticks`, (P,).
        point_centre: центр этого окна `(y, x)` в клетках сетки картинки; дробный при `MNIST_SUBCELL_POSITION`, (P, 2).
        point_extent: сколько клеток картинки видит окно, (P,).
        detectors: детекторы по слоям, строки `(центр y, центр x, радиус, порог λ_d, энергия e_d)` в клетках карты, (D, 5).
        layer: слой каждого детектора; 0 — самый низкий порог λ_d, самые широкие поля (§7.1.10), (D,).
        layer_thresholds: пороги λ_d слоёв по возрастанию.
        activation_threshold: порог зажигания точки λ_a (§6.1, §7.1.11).
        merge: чистота слияния (мерка, вне статьи; словарь с ключами `merge_*`): одной ли цифры картинки, чьи окна
            слились в одну точку карты; `None` — не мерена.
        field_half: половина поля зрения уровня в клетках картинки — ячейка рисунка содержимого.
    """

    number: int
    height: int
    width: int
    cells: object
    labels: object
    energies: object
    point_picture: object
    point_centre: object
    point_extent: object
    detectors: object
    layer: object
    layer_thresholds: tuple[float, ...]
    activation_threshold: float
    field_half: int
    merge: object = None


@dataclass(frozen=True)
class TrainingSticks:
    """Углопозиции обучающих картинок обученного цикла (с видами §7.1.4): из них рисуется содержимое точек карт.

    Attributes:
        picture: картинка каждой углопозиции, по возрастанию, (S,).
        y: место на сетке картинки по вертикали, в клетках; дробное при `MNIST_SUBCELL_POSITION`, (S,).
        x: то же по горизонтали, (S,).
        angle: угол, градусы, (S,).
        span: полудлина поля штриха вдоль края, в клетках (§6.4.3), (S,).
        labels: цифра каждой картинки, (N,).
        index: номер картинки в MNIST train, (N,).
        originals: сколько первых картинок — исходные цифры; дальше — их сдвинутые виды.
    """

    picture: object
    y: object
    x: object
    angle: object
    span: object
    labels: object
    index: object
    originals: int


@dataclass(frozen=True)
class Prediction:
    """Ответ распознавания одной картинки (§2.2.5).

    Attributes:
        digit: найденная цифра, 0…9.
        confidence: уверенность в ответе, 0…1 (вероятность цифры по softmax).
        probabilities: вероятности всех цифр 0…9, сумма равна 1.
    """

    digit: int
    confidence: float
    probabilities: tuple[float, ...]


@dataclass(frozen=True)
class Evaluation:
    """Точность распознавания на выборке.

    Attributes:
        split: выборка MNIST: `"train"` или `"test"`.
        samples: сколько картинок проверено.
        accuracy: доля верных ответов, 0…1.
    """

    split: str
    samples: int
    accuracy: float


@dataclass(frozen=True)
class CacheStep:
    """Записи кеша одного шага (вне статьи): сколько, сколько места, когда последний раз пригодились.

    Attributes:
        step: имя шага, например `"layout.memory"` (карта памяти уровня) или `"stimuli.sticks"` (штрихи картинок).
        entries: число записей: у шагов цикла — по одной на каждый разный вход, у штрихов — по файлу на набор MNIST.
        bytes: место на диске, байт (массивы, общие для нескольких записей, учтены в каждой).
        last_used_seconds: сколько секунд назад запись шага последний раз бралась или писалась.
    """

    step: str
    entries: int
    bytes: int
    last_used_seconds: float


@dataclass(frozen=True)
class BenchScore:
    """Точность метода по обучающим наборам стенда: среднее, разброс и 95 % доверительный интервал (вне статьи).

    Наборы — случайные выборки номеров MNIST train с закреплённым зерном; точность каждого набора считается на одной и
    той же конечной выборке (report или test), поэтому интервал по наборам не включает её собственную ошибку: её
    биномиальная ошибка записана в JSON стенда.

    Attributes:
        mean: средняя доля верных ответов по наборам, 0…1.
        std: стандартное отклонение по наборам (n − 1); у одного набора 0.
        ci_low: нижняя граница 95 % интервала (t-распределение); `None`, если набор один.
        ci_high: верхняя граница интервала; `None`, если набор один.
        per_set: доля верных ответов по наборам, по порядку.
    """

    mean: float
    std: float
    ci_low: float | None
    ci_high: float | None
    per_set: tuple[float, ...]


@dataclass(frozen=True)
class BenchRobustness:
    """Устойчивость метода к деформации: картинки отложенной выборки и они же, сдвинутые на пиксели (вне статьи).

    Берутся первые `pictures` картинок отложенной выборки (`BENCH_PIXEL_PICTURES`), сдвиг — `BENCH_PIXEL_SHIFT`.
    Мерка не связана с выбором настроек.

    Attributes:
        pictures: сколько картинок отложенной выборки сдвигалось.
        shift: сдвиг `(dy, dx)` в пикселях.
        same_answers: доля картинок, чей ответ при сдвиге не изменился, среднее по наборам, 0…1.
        accuracy_plain: доля верных ответов на этих картинках без сдвига, среднее по наборам.
        accuracy_shifted: то же на сдвинутых картинках.
    """

    pictures: int
    shift: tuple[int, int]
    same_answers: float
    accuracy_plain: float
    accuracy_shifted: float


@dataclass(frozen=True)
class BenchMethod:
    """Итог метода при одном режиме аугментации: настройка, выбранная по select, и её точность на report и test (вне статьи).

    Настройка выбрана по средней точности на части select отложенной выборки (при ничьей — первая по порядку), ранняя
    остановка CNN тоже по select. `validation` — точность на части report, которая ни в каком выборе не участвовала,
    `test` — на MNIST t10k, только у выбранной настройки.

    Attributes:
        method: `"damp"`, `"pixels"` (kNN по пикселям) или `"cnn"`.
        augment: `"none"` или `"shifts"` (случайный сдвиг при обучении).
        supported: `False` — метод не умеет этот режим (у DAMP `"shifts"` — не отдельный режим: сдвиги задают виды обучающих `LADDER_TRAIN_VIEWS`, §7.1.4; у kNN — не поддержано); точностей нет.
        setting: выбранная настройка: `"k=5"` у DAMP и kNN, имя конфигурации у CNN; `None`, если не посчитано.
        candidates: средняя точность на select каждой настройки перебора, `{настройка: доля}`.
        validation: точность выбранной настройки на части report отложенной выборки, по наборам.
        test: точность на проверочных (MNIST t10k) по наборам; `None`, если `test=0`. Считается только у выбранной настройки.
        robustness: устойчивость к сдвигу; `None`, если картинок для неё нет.
        note: пояснение, почему строка пустая.
    """

    method: str
    augment: str
    supported: bool
    setting: str | None
    candidates: dict[str, float]
    validation: BenchScore | None
    test: BenchScore | None
    robustness: BenchRobustness | None
    note: str = ""


@dataclass(frozen=True)
class BenchLevel:
    """Уровень цикла DAMP в стенде: мерки статьи и узнавание на report, среднее по наборам (§5.8, §5.13.2; §7.1.8.1; вне статьи).

    Мерки `energy_mean`, `code_resolution`, `code_largest`, `detail_empty`, `detail_full` считаются по обучающим окнам; `detail_bits` — по всем окнам уровня.
    `code_resolution` и `code_largest` — мерки §5.13.2.5, перенесённые с кодов точек карты на коды деталей (вне статьи).

    Attributes:
        number: номер уровня.
        sets: в скольких наборах такой уровень построен.
        picture_seen: доля углопозиций картинки, которую видит окно.
        map_points: точек карты уровня.
        detectors: детекторов над картой.
        same_label: доля соседних точек карты с одной цифрой.
        energy_mean: качество раскладки Ē (§5.8).
        code_resolution: окон обучающих картинок на один различный код детали (§5.13.2.5).
        code_largest: самый большой набор окон с одним кодом детали (§5.13.2.5).
        detail_bits: единиц в коде детали, среднее по всем окнам уровня (насыщенность, §7.1.8.1).
        detail_empty: доля обучающих окон с пустой деталью: ни один детектор не сработал (вне статьи).
        detail_full: доля обучающих окон, чей код упёрся в предел σ (§7.1.8.1).
        alone: узнавание по отклику уровня на report, `{k: доля}`; картинка без отклика считается неверной.
        glued: узнавание по склейке откликов уровней 1…number на report, `{k: доля}`.
        response_cosine: косинус откликов картинки и её сдвига; `None`, если сдвинутых картинок нет.
    """

    number: int
    sets: int
    picture_seen: float
    map_points: int
    detectors: int
    same_label: float
    energy_mean: float
    code_resolution: float
    code_largest: int
    detail_bits: float
    detail_empty: float
    detail_full: float
    alone: dict[int, float]
    glued: dict[int, float]
    response_cosine: float | None


@dataclass(frozen=True)
class BenchReport:
    """Итог стенда честного сравнения: методы, уровни DAMP и разрыв до CNN (вне статьи).

    Протокол: случайные выборки номеров MNIST с закреплённым зерном; отложенная выборка делится на select (выбор настроек,
    ранняя остановка CNN) и report (итог, ни в чём не участвовала); test считается один раз, у выбранных настроек.

    Attributes:
        methods: строки по методу и режиму аугментации, в порядке таблицы.
        levels: уровни DAMP (среднее по наборам); пусто, если DAMP не считался.
        train: обучающих картинок в наборе.
        sets: сколько наборов.
        validation: картинок отложенной выборки (select + report).
        test: проверочных картинок (0 — не считались).
        map_points: предел точек карты уровня у DAMP.
        gap: сколько DAMP до CNN, доля: `damp_minus_cnn_none` и `damp_minus_best_cnn` (по report, ключ `validation`, и по
            `test`), `intervals` — парные 95 % интервалы по наборам, `best_cnn` — лучшая CNN (выбрана по select); `None`,
            если не считались DAMP или CNN.
        path: файл `data/bench/bench-<время>.json` со всеми цифрами; `None`, если не сохранялся.
        seconds: время по методам и `total`, секунд.
    """

    methods: tuple[BenchMethod, ...]
    levels: tuple[BenchLevel, ...]
    train: int
    sets: int
    validation: int
    test: int
    map_points: int
    gap: dict[str, object] | None
    path: str | None
    seconds: dict[str, float]
