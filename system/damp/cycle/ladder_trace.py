"""Наблюдатель цикла уровней: что получилось на каждом проходе, без вмешательства в счёт.

Прогон цикла (MnistLadderRunner, cycle/ladder.py) сам передаёт наблюдателю итоги прохода: окна внимания и их коды
(продолжение статьи, П.2, П.4–П.5), карту памяти уровня (§5), детекторы над ней (§6.4), срабатывания (§6.1, §6.5) и
детали — эмбеддинги окон (§6.5, §6.6, П.1). Наблюдатель ничего не возвращает и на результат прогона не влияет. По
этим данным строятся рисунки README (system/damp/figures/).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from damp.encoding.damp_encoder import Encoder


@dataclass(frozen=True)
class LadderRunTrace:
    """Вход прогона: картинки, их углопозиции и группы картинок (обучающие, проверочные)."""

    images: np.ndarray  # (N, 28, 28) картинки, 0…255
    labels: np.ndarray  # (N,) цифра каждой картинки
    stick_picture: np.ndarray  # (S,) картинка каждой углопозиции, по возрастанию
    stick_y: np.ndarray  # (S,) место на сетке углопозиций, в клетках (дробное при MNIST_SUBCELL_POSITION)
    stick_x: np.ndarray
    stick_angle: np.ndarray  # (S,) направление перепада яркости, градусы
    ranges: dict[str, tuple[int, int]]  # группа картинок → (первая, после последней)
    angle_encoder: "Encoder"  # детекторы угла: код «что» углопозиции (§4.4)
    example: int | None = None  # номер картинки-примера в наборе (обучающая цифра LADDER_FIGURE_DIGIT, одна на прогон)
    example_index: int | None = None  # её номер в наборе MNIST


@dataclass(frozen=True)
class LadderLevelTrace:
    """Итог одного прохода цикла: окна, их коды, память уровня, детекторы, срабатывания и детали."""

    number: int
    unit: float  # шаг мест элементов на плоскости детекторов места, в долях её единицы
    seen: float  # какую долю углопозиций своей картинки окно видит в среднем
    labels: np.ndarray  # (N,) цифра каждой картинки
    # Окна внимания (П.2): картинка, центр (y, x) и радиус поля на картинке в клетках.
    keyhole_picture: np.ndarray
    keyhole_centre: np.ndarray
    keyhole_extent: np.ndarray
    # Вхождения элементов в окна и сами элементы уровня (углопозиции либо детали прошлого уровня).
    entry_keyhole: np.ndarray
    entry_element: np.ndarray
    entry_offset: np.ndarray  # (E, 2) смещение элемента от центра окна (dy, dx), в клетках
    element_picture: np.ndarray
    element_y: np.ndarray
    element_x: np.ndarray
    # Коды окон (П.4–П.5): сегмент места и сегмент содержания, по строке на окно.
    keyhole_codes: np.ndarray
    code_length: int
    place_length: int
    content_length: int
    place_encoder: "Encoder"  # детекторы места — круги на плоскости окна (§4.3, П.5)
    # Память уровня (§5): ячейка (y, x), код и цифра каждой точки карты.
    map_cells: np.ndarray
    map_size: tuple[int, int]
    map_codes: np.ndarray
    map_labels: np.ndarray
    # Энергии точек (§5.7) и порог μ_e, ниже которого точка в срабатывание детектора не входит (§6.5).
    energies: np.ndarray
    energy_threshold: float
    # Детекторы (§6.3) по слоям: центр (y, x), радиус, порог слоя λ_d, собственная энергия e_d.
    detectors: np.ndarray
    layer_index: np.ndarray
    layer_thresholds: tuple[float, ...]
    activation_lambda: float  # λ_a (§6.1, §7.1.11)
    detector_threshold: float  # μ_d (§6.5)
    # Срабатывания: окно, детектор, уровень активации E(d).
    fired_keyhole: np.ndarray
    fired_detector: np.ndarray
    fired_level: np.ndarray
    # Детали — эмбеддинги окон (§6.5): код, его длина и уровень активации детали (П.1).
    detail_codes: np.ndarray
    detail_length: int
    detail_strength: np.ndarray
    # Отклик картинки на уровне: вектор «сколько раз сработал каждый детектор», единичной длины (мерка, вне статьи).
    response: np.ndarray
    # Секунды шагов прохода: окна, коды окон, память, детекторы, срабатывания, детали (из кеша — почти ноль).
    seconds: dict = field(default_factory=dict)


@dataclass(frozen=True)
class LadderTopTrace:
    """Детали верхнего уровня: по ним голосуется класс картинки (§2.2.5); карта верхних деталей — их память."""

    level: int
    picture: np.ndarray
    y: np.ndarray
    x: np.ndarray
    codes: np.ndarray  # по строке на деталь
    code_length: int
    # Карта верхних деталей (§5): ячейка (y, x) и цифра каждой точки; None — карта ещё не разложена.
    map_cells: np.ndarray | None = None
    map_size: tuple[int, int] | None = None
    map_labels: np.ndarray | None = None


# Части прохода уровня по порядку счёта: имя части → этап пайплайна, к которому она относится (вкладка Rerun).
# Значения части — простые массивы под именами полей LadderLevelTrace (keyhole_picture, map_cells, …).
LEVEL_PARTS = {
    "keyholes": 6,  # окна уровня (П.2, П.3): окно растёт от уровня к уровню — этап 6
    "codes": 2,  # коды окон «что и где» (П.4–П.7)
    "memory": 3,  # карта памяти уровня (§5)
    "detectors": 4,  # детекторы над картой (§6.4)
    "firing": 4,  # срабатывания (§6.1, §6.5)
    "details": 5,  # детали — эмбеддинги окон (§6.5, П.1)
    "recognition": 7,  # узнавание по уровню (§2.2.5; мерка вне статьи)
}


class LadderObserver:
    """Наблюдатель прогона цикла; по умолчанию ничего не делает."""

    def start(self, trace: LadderRunTrace) -> None:
        """Перед первым проходом: картинки и их углопозиции."""

    def stage(self, number: int, stage: int) -> None:
        """Начался этап stage (1–7; 0 — прогон закончен) на уровне number (0 — до первого уровня): для показа хода."""

    def begin(self, number: int) -> None:
        """Перед проходом уровня number (этап 2 — начало цикла)."""

    def part(self, number: int, name: str, values: dict) -> None:
        """Готова часть name прохода уровня number (LEVEL_PARTS) — сразу, не дожидаясь конца прохода."""

    def level(self, trace: LadderLevelTrace) -> None:
        """После каждого прохода цикла."""

    def top(self, trace: LadderTopTrace) -> None:
        """После последнего прохода: детали верхнего уровня и их карта."""

    def finish(self, reports: list, accuracy: dict | None) -> None:
        """В конце прогона: итоги уровней (LadderLevelReport) и голосование верхних деталей."""


class LadderObservers(LadderObserver):
    """Несколько наблюдателей сразу (например, рисунки README и окно Rerun): каждый получает всё по порядку."""

    def __init__(self, *observers: LadderObserver) -> None:
        self.observers = tuple(observers)

    def start(self, trace: LadderRunTrace) -> None:
        for observer in self.observers:
            observer.start(trace)

    def stage(self, number: int, stage: int) -> None:
        for observer in self.observers:
            observer.stage(number, stage)

    def begin(self, number: int) -> None:
        for observer in self.observers:
            observer.begin(number)

    def part(self, number: int, name: str, values: dict) -> None:
        for observer in self.observers:
            observer.part(number, name, values)

    def level(self, trace: LadderLevelTrace) -> None:
        for observer in self.observers:
            observer.level(trace)

    def top(self, trace: LadderTopTrace) -> None:
        for observer in self.observers:
            observer.top(trace)

    def finish(self, reports: list, accuracy: dict | None) -> None:
        for observer in self.observers:
            observer.finish(reports, accuracy)
