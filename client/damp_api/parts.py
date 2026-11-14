"""Части цикла: результаты «кубиков» этапов, из которых клиент собирает свой проход.

Часть — результат одного вызова фреймворка. Она держит системный объект внутри, отдаёт простые значения и
передаётся в следующий вызов. Каждая часть помнит настройки того `Damp`, который её сделал (`Session`), и её
методы работают с ними же.

Цепочка частей одного уровня цикла (в скобках — этап и вызов, что её делает):

| Часть | Этап | Что делает её | Что в ней |
|---|---|---|---|
| `Digits` | 1 | `Damp.stimuli.digits` | картинки MNIST и их углопозиции |
| `Elements` | 2, 5 | `Damp.coding.sticks`, `Damp.embeddings.details` | элементы уровня: штрихи или детали |
| `Keyholes` | 6 | `Damp.cycle.keyholes` | окна внимания уровня (П.2) |
| `KeyholeCodes` | 2 | `Damp.coding.keyholes` | код «что и где» каждого окна (П.4–П.7) |
| `Memory` | 3 | `Damp.layout.memory` | карта памяти уровня (§5) |
| `MapDetectors` | 4 | `Damp.detectors.build` | детекторы над картой (§6.4) |
| `Firing` | 4 | `MapDetectors.fire` | какие детекторы сработали на окнах (§6.5) |
| `Level` | 2–6 | `Damp.cycle.step` | весь проход уровня разом |

Example:
    >>> digits = damp.stimuli.digits(train=100, test=500)
    >>> keyholes = damp.cycle.keyholes(damp.coding.sticks(digits))
    >>> codes = damp.coding.keyholes(keyholes)
    >>> keyholes.picture_seen, codes.length
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from ._system import system
from .reports import Picture, Stimulus


class Session:
    """Настройки одного `Damp`: файл настроек и поверх него `changes` (имя ключа → значение).

    Служебный класс: части и этапы берут из него настройки для своих вызовов. Клиенту нужен редко;
    значения настроек читаются через `Damp.settings`.

    Attributes:
        changes: изменения поверх файла настроек, имя ключа → значение.
    """

    def __init__(self, changes: Mapping[str, object] | None = None) -> None:
        self.changes = dict(changes or {})
        system.check_changes(self.changes)

    def merged(self, changes: Mapping[str, object]) -> "Session":
        """Новая сессия: нынешние изменения и поверх них `changes`."""
        return Session({**self.changes, **dict(changes)})

    @contextmanager
    def active(self) -> Iterator[None]:
        """Контекст: пока он открыт, системный слой читает настройки этой сессии."""
        with system.session(self.changes):
            yield


def _ones(packed: "np.ndarray") -> "np.ndarray":
    return np.unpackbits(np.ascontiguousarray(packed, dtype=np.uint8), axis=1).sum(axis=1)


class _Part:
    """Основа частей: системный объект внутри (не для клиента) и настройки своего `Damp`."""

    def __init__(self, session: Session) -> None:
        self._session = session


class Digits(_Part):
    """Этап 1. Цифры прогона: сначала обучающие, затем проверочные и их сдвиги.

    Память уровня строится только по обучающим картинкам; проверочные лишь читаются ею. Часть делает
    `Damp.stimuli.digits`, принимают `Damp.coding.sticks` и `Damp.recognition.*`.

    Example:
        >>> digits = damp.stimuli.digits(train=100, test=500)
        >>> digits.train, digits.test
        (100, 500)
        >>> digits.groups["train"]       # (первая картинка, после последней)
        (0, 100)
        >>> five = digits.picture(0)      # картинка, цифра и углопозиции
    """

    def __init__(self, session: Session, groups) -> None:  # noqa: ANN001
        super().__init__(session)
        self._groups = groups

    @property
    def pictures(self) -> int:
        """Сколько картинок всего."""
        return int(self._groups.pictures)

    @property
    def train(self) -> int:
        """Сколько обучающих картинок (они первые)."""
        return int(self._groups.train_pictures)

    @property
    def test(self) -> int:
        """Сколько проверочных картинок (без сдвигов)."""
        first, last = self._groups.ranges.get("test", (0, 0))
        return int(last - first)

    @property
    def groups(self) -> dict[str, tuple[int, int]]:
        """Группа → (первая картинка, после последней): "train", "test", сдвиги "dy,dx"."""
        return {key if isinstance(key, str) else f"{key[0]},{key[1]}": value for key, value in self._groups.ranges.items()}

    @property
    def labels(self) -> tuple[int, ...]:
        """Цифра каждой картинки по номеру картинки, 0…9."""
        return tuple(int(label) for label in self._groups.labels)

    @property
    def sticks(self) -> int:
        """Сколько углопозиций у всех картинок."""
        return int(self._groups.atoms.picture.shape[0])

    def picture(self, index: int) -> Picture:
        """Картинка прогона с цифрой, группой и углопозициями.

        Args:
            index: номер картинки, от 0 до `pictures - 1`; обучающие идут первыми.

        Returns:
            `Picture`: цифра, группа, сама картинка 28×28 (0…255) и её углопозиции.
        """
        atoms = self._groups.atoms
        first, last = self._groups.sticks(int(index))
        group = next(name for name, (begin, end) in self.groups.items() if begin <= index < end)
        stimuli = tuple(Stimulus(float(atoms.angle[i]), int(atoms.x[i]), int(atoms.y[i])) for i in range(first, last))
        return Picture(index=int(index), digit=int(atoms.labels[index]), group=group, image=atoms.images[index], stimuli=stimuli)


class Elements(_Part):
    """Элементы уровня: углопозиции (уровень 0) либо детали уровня. Вход этапа 2 следующего прохода цикла.

    Каждый элемент — «что» (разреженный код) и «где» (клетка и картинка). Уровень 0 делает
    `Damp.coding.sticks`, уровни выше — `Damp.embeddings.details` (§6.5, §7.1.12; П.1).

    Attributes:
        digits: цифры прогона, к которым относятся элементы.
        level: номер уровня элементов: 0 — штрихи, 1 и выше — детали уровня.
    """

    def __init__(self, session: Session, elements, digits: Digits, level: int) -> None:  # noqa: ANN001
        super().__init__(session)
        self._elements = elements
        self.digits = digits
        self.level = int(level)

    @property
    def count(self) -> int:
        """Сколько элементов всего, по всем картинкам."""
        return int(self._elements.count)

    @property
    def code_length(self) -> int:
        """Длина кода «что» элемента: код направления (уровень 0) либо код детали."""
        return int(self._elements.what_length)

    @property
    def distinct_codes(self) -> int:
        """Сколько различных кодов «что» среди элементов."""
        return int(self._elements.what_packed.shape[0])

    @property
    def ones(self) -> float:
        """Среднее число единиц в коде элемента."""
        return float(_ones(self._elements.what_packed)[self._elements.what_index].mean()) if self.count else 0.0

    @property
    def per_picture(self) -> float:
        """Сколько элементов в среднем на обучающей картинке."""
        return float(np.count_nonzero(self._elements.picture < self.digits.train)) / max(1, self.digits.train)


class Keyholes(_Part):
    """Окна внимания уровня (этап 6, П.2): где стоят, сколько видят. Каждое окно станет одним кодом (этап 2).

    Окно — круг вокруг элемента; его содержимое — один стимул следующей раскладки. Окно растёт от уровня к
    уровню (П.3). Часть делает `Damp.cycle.keyholes`.

    Attributes:
        elements: элементы, на которых стоят окна.
    """

    def __init__(self, session: Session, keyholes, elements: Elements) -> None:  # noqa: ANN001
        super().__init__(session)
        self._keyholes = keyholes
        self.elements = elements

    @property
    def level(self) -> int:
        """Номер уровня окон: на единицу больше уровня их элементов."""
        return int(self._keyholes.number)

    @property
    def count(self) -> int:
        """Сколько окон всего, по всем картинкам."""
        return int(self._keyholes.count)

    @property
    def radius(self) -> float:
        """Сколько клеток картинки видит окно, в среднем по обучающим."""
        return float(self._keyholes.sight)

    @property
    def picture_seen(self) -> float:
        """Какую долю углопозиций своей картинки видит окно, в среднем."""
        return float(self._keyholes.seen)

    @property
    def per_picture(self) -> float:
        """Окон на обучающей картинке, в среднем."""
        return float(self._keyholes.taught) / max(1, self.elements.digits.train)

    @property
    def elements_per_keyhole(self) -> float:
        """Элементов в окне, в среднем по обучающим окнам."""
        pieces = self._keyholes.pieces
        per = np.bincount(pieces.piece, minlength=pieces.count)[pieces.picture < self.elements.digits.train]
        return float(per.mean()) if per.size else 0.0


class KeyholeCodes(_Part):
    """Коды окон уровня (этап 2, П.4–П.7): биты места и содержание под маской своего места, по коду на окно.

    Код окна = биты места (OR) + коды элементов, переставленные маской круга места (П.4, П.5). Часть
    делает `Damp.coding.keyholes`.

    Attributes:
        keyholes: окна, чьи это коды.
    """

    def __init__(self, session: Session, codes, keyholes: Keyholes) -> None:  # noqa: ANN001
        super().__init__(session)
        self._codes = codes
        self.keyholes = keyholes

    @property
    def count(self) -> int:
        """Сколько кодов (по числу окон)."""
        return int(self._codes.count)

    @property
    def length(self) -> int:
        """Длина кода окна в битах: биты места и содержание вместе."""
        return int(self._codes.code_length)

    @property
    def place_length(self) -> int:
        """Сколько бит кода отведено месту."""
        return int(self._codes.place_length)

    @property
    def content_length(self) -> int:
        """Сколько бит кода отведено содержанию (элементам окна)."""
        return int(self._codes.content_length)

    @property
    def ones(self) -> float:
        """Среднее число единиц в коде окна."""
        return float(_ones(self._codes.packed).mean()) if self.count else 0.0


class Memory(_Part):
    """Карта памяти уровня (этап 3, §5): образцы кодов обучающих картинок, похожие — рядом.

    Одна раскладка на все окна уровня (П.2); похожие коды сливаются в образец (§7.1.7). Часть делает
    `Damp.layout.memory`; видеокарту под ней освобождает `release`.

    Attributes:
        codes: коды окон, из которых построена карта.
    """

    def __init__(self, session: Session, memory, codes: KeyholeCodes) -> None:  # noqa: ANN001
        super().__init__(session)
        self._memory = memory
        self.codes = codes

    @property
    def points(self) -> int:
        """Сколько точек на карте (образцов кодов)."""
        return int(self._memory.points)

    @property
    def width(self) -> int:
        """Ширина сетки карты, клеток."""
        return int(self._memory.layout.width)

    @property
    def height(self) -> int:
        """Высота сетки карты, клеток."""
        return int(self._memory.layout.height)

    @property
    def same_label(self) -> float:
        """Доля соседних точек карты с одной цифрой (4 соседа); у случайной карты — около 0.1."""
        return float(self._memory.quality["same_label"])

    def release(self) -> None:
        """Освободить память видеокарты под картой (после работы с её детекторами)."""
        self._memory.release()


@dataclass(frozen=True)
class DetectorLayer:
    """Слой детекторов: порог сходства λ_d и сколько детекторов (§6.4, §7.1.10).

    Attributes:
        threshold: порог сходства λ_d слоя; у широких слоёв он ниже.
        detectors: сколько детекторов в слое.
    """

    threshold: float
    detectors: int


class MapDetectors(_Part):
    """Детекторы над картой памяти (этап 4, §6.4): круги над кластерами похожих точек, по слоям.

    Кластеры ищет DBSCAN, радиус круга — по наибольшему заполнению (§6.4.1, §6.4.3). Часть делает
    `Damp.detectors.build`; срабатывания на кодах окон считает `fire`.

    Attributes:
        memory: карта, над которой стоят детекторы.
    """

    def __init__(self, session: Session, detectors, memory: Memory) -> None:  # noqa: ANN001
        super().__init__(session)
        self._detectors = detectors
        self.memory = memory

    @property
    def count(self) -> int:
        """Сколько детекторов всего, по всем слоям."""
        return len(self._detectors.detectors)

    @property
    def layers(self) -> tuple[DetectorLayer, ...]:
        """Слои от широкого (низкий порог) к узкому."""
        per = np.bincount(self._detectors.layer_index, minlength=len(self._detectors.detection.layer_thresholds))
        return tuple(DetectorLayer(float(t), int(n)) for t, n in zip(self._detectors.detection.layer_thresholds, per))

    @property
    def activation_threshold(self) -> float:
        """Порог λ_a: код зажигает точки карты, похожие на него не меньше (§6.1, §7.1.11)."""
        return float(self._detectors.activation_lambda)

    def fire(self, codes: KeyholeCodes) -> "Firing":
        """Какие детекторы сработали на каждом окне: код окна зажигает точки карты, точки зажигают детекторы (§6.5).

        Args:
            codes: коды окон; обычно те, по которым построена карта.

        Returns:
            `Firing`: на каком окне какой детектор сработал и с каким уровнем активации.
        """
        every = np.arange(codes.count, dtype=np.int64)
        with self._session.active():
            firing = self._detectors.fire(codes._codes.packed, every, every, codes.count)
        return Firing(self._session, firing, self, codes)


class Firing(_Part):
    """Срабатывания детекторов (этап 4, §6.5): на каком окне какой детектор и с каким уровнем активации.

    Часть делает `MapDetectors.fire`; по ней `Damp.embeddings.details` собирает детали (§6.5, §7.1.12).

    Attributes:
        detectors: детекторы, которые срабатывали.
        codes: коды окон, на которых они срабатывали.
    """

    def __init__(self, session: Session, firing, detectors: MapDetectors, codes: KeyholeCodes) -> None:  # noqa: ANN001
        super().__init__(session)
        self._firing = firing
        self.detectors = detectors
        self.codes = codes

    @property
    def keyholes(self) -> int:
        """Сколько окон проверено."""
        return int(self._firing.sets)

    @property
    def fired(self) -> int:
        """Сколько срабатываний всего."""
        return int(self._firing.rows.shape[0])

    @property
    def per_keyhole(self) -> float:
        """Сработавших детекторов на окно, в среднем."""
        return self.fired / max(1, self.keyholes)

    @property
    def silent(self) -> float:
        """Доля окон, на которых не сработал ни один детектор."""
        per = np.bincount(self._firing.rows, minlength=self.keyholes)
        return float(np.mean(per == 0)) if per.size else 0.0


@dataclass(frozen=True)
class Level:
    """Один проход цикла (этапы 2–5): окна, их коды, память, детекторы, срабатывания и детали — элементы следующего.

    Результат `Damp.cycle.step`. Детали `details` — вход следующего шага.

    Attributes:
        keyholes: окна внимания уровня.
        codes: коды окон.
        memory: карта памяти (видеокарта под ней уже освобождена).
        detectors: детекторы над картой.
        firing: срабатывания детекторов на окнах.
        details: детали окон — элементы следующего уровня.
    """

    keyholes: Keyholes
    codes: KeyholeCodes
    memory: Memory
    detectors: MapDetectors
    firing: Firing
    details: Elements

    @property
    def number(self) -> int:
        """Номер уровня (совпадает с `keyholes.level`)."""
        return self.keyholes.level
