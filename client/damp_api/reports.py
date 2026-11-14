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
| `Prediction`, `Evaluation` | ответ и точность распознавания | `Recognition.predict`, `Recognition.evaluate` |
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
        x: номер клетки сетки по горизонтали.
        y: номер клетки сетки по вертикали.
    """

    angle: float
    x: int
    y: int


@dataclass(frozen=True)
class Picture:
    """Картинка прогона вместе с цифрой и найденными на ней углопозициями.

    Attributes:
        index: номер картинки в прогоне (обучающие идут первыми).
        digit: цифра на картинке, 0…9.
        group: группа картинки: `"train"`, `"test"` или сдвиг в виде `"dy,dx"`.
        image: сама картинка, массив numpy 28×28 со значениями 0…255.
        stimuli: углопозиции картинки.
    """

    index: int
    digit: int
    group: str
    image: object  # numpy-массив 28×28
    stimuli: tuple[Stimulus, ...]


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
