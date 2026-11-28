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
| `TrainedLadder` | 6, 7 | `Damp.cycle.trained` | обученный цикл и читатель: путь новой картинки по уровням |

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
from .reports import (
    LevelMap, LevelVotes, Picture, PrimaryMap, PrimarySight, Sight, SightLevel, SnapshotReport, Stimulus, TrainingSticks,
)


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


def _snapshot_report(path: str, meta: Mapping[str, object], *, same: int) -> SnapshotReport:
    """Описание снимка (snapshot.json) → `SnapshotReport`."""
    return SnapshotReport(
        path=str(path), label=str(meta.get("label", "")), version=str(meta.get("version", "")),
        commit=str(meta.get("commit", "")), created=str(meta.get("created", "")),
        train_pictures=int(meta.get("train_pictures", 0)), levels=int(meta.get("levels", 0)),
        megabytes=float(meta.get("megabytes", 0.0)), control=int(meta["control"]["pictures"]), same=int(same),
    )


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
        """Группа → (первая картинка, после последней).

        Имена: `"train"`, `"extra"` (добавочные картинки верхних уровней, LADDER_EXTRA_PICTURES), `"validation"`, `"test"`; сдвиг на клетки сетки — `"dy,dx"`; сдвиг на пиксели —
        `"pixel:dy,dx"` (у разных сдвигов разные имена).
        """
        return {self._group_name(key): value for key, value in self._groups.ranges.items()}

    @staticmethod
    def _group_name(key) -> str:  # noqa: ANN001
        """Имя группы по её ключу: строка — как есть, (dy, dx) — "dy,dx", ("pixel", dy, dx) — "pixel:dy,dx"."""
        if isinstance(key, str):
            return key
        if key[0] == "pixel":
            return f"pixel:{key[1]},{key[2]}"
        return f"{key[0]},{key[1]}"

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
        stimuli = tuple(
            Stimulus(float(atoms.angle[i]), float(atoms.x[i]), float(atoms.y[i]), float(atoms.span[i])) for i in range(first, last)
        )
        return Picture(index=int(index), digit=int(atoms.labels[index]), group=group, image=atoms.images[index], stimuli=stimuli,
                       mnist=int(atoms.index[index]))


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
    def keyhole_radius(self) -> float:
        """Средний радиус окна обучающих картинок, клеток: круг, в котором окно собирает элементы уровня."""
        return float(self._keyholes.radius)

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

    @property
    def merge(self) -> dict[str, float]:
        """Чистота слияния (мерка, вне статьи): одной ли цифры картинки, чьи окна слились в одну точку карты.

        Ключи: `merge_pictures_per_point` — картинок на точку; `merge_shared_points` — доля точек с окнами двух и
        более картинок; `merge_purity` — доля картинок таких точек, чья цифра — главная цифра точки;
        `merge_purity_random` — она же при перемешанных цифрах; `merge_pure_points` — доля общих точек одной цифры.
        """
        return {key: float(value) for key, value in self._memory.quality.items() if key.startswith("merge_")}

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
            `Firing`: на каком окне какой детектор сработал и с каким уровнем активации; при
            `LADDER_LEVEL_NORMALISATION = "train_max"` уровень в [0, 1] (способ — доработка
            автора проекта, вне статьи; §7.1.12 и §9.3 требуют лишь значений в [0, 1]).
        """
        keyholes = codes.keyholes
        with self._session.active():
            # Тот же шаг, что у цикла: уровни нормируются по окнам обучающих картинок (доработка, вне статьи).
            step = system.cycle_step(keyholes.elements.digits._groups, map_points=system.settings.LADDER_MAP_MAX_POINTS)
            firing = step.fire(self._detectors, codes._codes, step.trained(keyholes._keyholes))
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


class TrainedLadder(_Part):
    """Этапы 6–7. Обученный цикл и его линейный читатель (§6.6, §9.3): по ним идёт любая новая картинка 28×28.

    Часть делает `Damp.cycle.trained`. Картинка идёт теми же шагами, что картинка прогона `ladder`: углопозиции
    (§4.4), окна (П.2), их коды (П.4–П.7), срабатывания детекторов карт (§6.5), детали (П.1) — уровень за уровнем;
    всё подобранное по обучающим картинкам берётся из обученных уровней. Поэтому у цифры MNIST путь тот же, что в
    прогоне, цифра в цифру. Так работает живой дашборд (`python client/run.py dashboard`).

    Example:
        >>> ladder = damp.cycle.trained(100)
        >>> sight = ladder.see(picture)          # numpy 28×28, 0…255, белая цифра на чёрном
        >>> sight.digit, [len(level.keyhole_extent) for level in sight.levels]
    """

    def __init__(self, session: Session, ladder, snapshot: SnapshotReport | None = None) -> None:  # noqa: ANN001
        super().__init__(session)
        self._ladder = ladder
        self._snapshot = snapshot

    @property
    def levels(self) -> int:
        """Сколько уровней у обученного цикла."""
        return len(self._ladder.levels)

    @property
    def snapshot(self) -> SnapshotReport | None:
        """Снимок, из которого загружен цикл (`Damp.cycle.load`), со сверкой ответов; `None` — цикл обучен здесь."""
        return self._snapshot

    def save(self, path: str, *, label: str = "", control: int = 100) -> SnapshotReport:
        """Снимок обученного цикла в папку: потом `Damp.cycle.load(path)` даёт тот же цикл без обучения (вне статьи).

        В снимок идут все уровни (карты памяти, детекторы, всё подобранное по обучающим картинкам), читатель, голоса
        памяти и все настройки этого `Damp`; граф объектов — JSON, массивы — npz, без pickle. Перед записью цикл
        отвечает на первые `control` картинок MNIST t10k: при загрузке ответы сверяются. Память видеокарты
        освобождается и строится заново при следующем `see`.

        Args:
            path: папка снимка; новая, пустая либо прежний снимок (он перезаписывается).
            label: подпись снимка, видна при загрузке.
            control: картинок t10k для сверки.

        Returns:
            `SnapshotReport`: что записано и сколько весит.

        Raises:
            FileExistsError: папка не пуста и это не снимок.

        Example:
            >>> ladder = damp.cycle.trained(1000)
            >>> ladder.save("data/dashboard/trained-1000", label="1000 картинок").megabytes
        """
        with self._session.active():
            meta = self._ladder.save(path, values=system.settings_values(), label=label, control=int(control))
        return _snapshot_report(path, meta, same=int(meta["control"]["pictures"]))

    @property
    def train_pictures(self) -> int:
        """Обучающих картинок вместе с видами (§7.1.4)."""
        return int(self._ladder.groups.train_pictures)

    @property
    def decay(self) -> float:
        """Штраф L2 линейного читателя (§9.3)."""
        return float(self._ladder.decay)

    @property
    def knowledge_thresholds(self) -> tuple[float, ...]:
        """Пороги знания по уровням (§7.1.13.3): ниже порога уровень картинку «не знает» (`RECOGNITION_KNOWLEDGE_QUANTILE`)."""
        return tuple(float(value) for value in (self._ladder.memory.thresholds or ()))

    def see(self, image, *, log: bool = False) -> Sight:  # noqa: ANN001
        """Путь картинки через все уровни и её класс: голоса памяти и линейный читатель (§6.5, §7.1.13.3, §9.3; П.1–П.5).

        Args:
            image: картинка 28×28: numpy uint8 0…255 либо float в [0, 1], белая цифра на чёрном, как в MNIST.
            log: записать итог в журнал событием `dashboard.see` (у дашборда — по окончании штриха).

        Returns:
            `Sight`: углопозиции, уровни (кусочки, зажжённые точки, сработавшие детекторы, детали), голоса памяти по
            уровням, ответ памяти (среднее долей голосов) с согласием уровней и вероятности цифр по линейному читателю.

        Raises:
            ValueError: картинка не 28×28.

        Example:
            >>> sight = ladder.see(np.zeros((28, 28), dtype=np.uint8))
            >>> sight.digit, sight.levels                # пустая картинка: (-1, ())
            >>> sight.verdict, sight.memory_digit        # пустая картинка: ("unknown", -1)
        """
        with self._session.active():
            found = self._ladder.see(np.asarray(image), log=log)
        thresholds = self._ladder.memory.thresholds
        return Sight(
            stick_angle=found.stick_angle,
            stick_y=found.stick_y,
            stick_x=found.stick_x,
            stick_strength=found.stick_strength,
            stick_span=found.stick_span,
            levels=tuple(
                SightLevel(
                    number=level.number, keyhole_centre=level.keyhole_centre, keyhole_extent=level.keyhole_extent,
                    entry_keyhole=level.entry_keyhole, entry_element=level.entry_element,
                    lit_keyhole=level.lit_keyhole, lit_point=level.lit_point, lit_value=level.lit_value,
                    fired_keyhole=level.fired_keyhole, fired_detector=level.fired_detector, fired_level=level.fired_level,
                    detail_length=level.detail_length, strength=level.strength, peak=level.peak,
                )
                for level in found.levels
            ),
            probabilities=tuple(float(value) for value in found.probabilities),
            digit=int(found.answer),
            seconds=float(found.seconds),
            votes=tuple(
                LevelVotes(
                    number=int(level.number), votes=tuple(float(value) for value in level.votes), digit=int(level.digit),
                    share=float(level.share), knowledge=float(level.knowledge),
                    threshold=float(thresholds[level.number - 1]) if thresholds else 0.0,
                    knows=bool(level.voted and level.knowledge >= (thresholds[level.number - 1] if thresholds else 0.0)),
                    keyholes=int(level.keyholes), voted=int(level.voted),
                )
                for level in found.votes
            ),
            verdict=str(found.verdict.kind),
            memory_digit=int(found.verdict.digit),
            memory_share=float(found.verdict.share),
            memory_agreed=int(found.verdict.agreed),
            memory_votes=tuple(float(value) for value in found.verdict.votes),
            primary=None if found.primary is None or self._ladder.primary is None else PrimarySight(
                points=int(self._ladder.primary.memory.points), detectors=int(self._ladder.primary.detectors.built),
                lit_stick=found.primary.lit_stick, lit_point=found.primary.lit_point, lit_value=found.primary.lit_value,
                fired_stick=found.primary.fired_stick, fired_detector=found.primary.fired_detector,
                fired_level=found.primary.fired_level,
            ),
        )

    def maps(self) -> tuple[LevelMap, ...]:
        """Карты уровней для показа: точки карты, окно обучающей картинки каждой точки, детекторы по слоям (§5, §6.4).

        Returns:
            `LevelMap` по уровням, от первого к верхнему.
        """
        found = []
        for level in self._ladder.levels:
            layout, detectors = level.memory.layout, level.detectors
            found.append(
                LevelMap(
                    number=int(level.number),
                    height=int(layout.height),
                    width=int(layout.width),
                    cells=np.asarray(layout.cells_array(), dtype=np.int64),
                    labels=np.asarray([int(label) for label in layout.code_table.labels()], dtype=np.int64),
                    energies=np.asarray(level.memory.energies, dtype=np.float64),
                    point_picture=np.asarray(level.point_picture, dtype=np.int64),
                    point_centre=np.asarray(level.point_centre, dtype=np.float64),
                    point_extent=np.asarray(level.point_extent, dtype=np.float64),
                    detectors=np.asarray(
                        [(d.center[0], d.center[1], d.radius, d.lambda_threshold, d.energy) for d in detectors.detectors],
                        dtype=np.float64,
                    ).reshape(-1, 5),
                    layer=np.asarray(detectors.layer_index, dtype=np.int64),
                    layer_thresholds=tuple(float(value) for value in detectors.layer_thresholds),
                    activation_threshold=float(detectors.activation_lambda),
                    field_half=int(max(abs(value) for value in level.field.box)),
                    merge={key: float(value) for key, value in level.memory.quality.items() if key.startswith("merge_")},
                )
            )
        return tuple(found)

    def primary_map(self) -> "PrimaryMap | None":
        """Первичная карта — «уровень 0»: раскладка первичных кодов углопозиций обучающих цифр (§4.6, §5, §5.13.2).

        Returns:
            `PrimaryMap`: точки карты с углом и местом своей углопозиции и детекторы над картой; `None` — карта не
            построена.

        Example:
            >>> primary = ladder.primary_map()
            >>> primary.width, primary.angle_period      # сетка карты и период угла: 180 у штрихов без стрелки
        """
        level = self._ladder.primary
        if level is None:
            return None
        layout, detectors = level.memory.layout, level.detectors
        with self._session.active():
            s = system.settings
            period = (float(s.MNIST_ANGLE_MAX) - float(s.MNIST_ANGLE_MIN)) / (2.0 if level.axial() else 1.0)
        labels = np.asarray(self._ladder.groups.atoms.labels, dtype=np.int64)
        return PrimaryMap(
            height=int(layout.height),
            width=int(layout.width),
            cells=np.asarray(layout.cells_array(), dtype=np.int64),
            angle=np.asarray(level.point_angle, dtype=np.float64),
            y=np.asarray(level.point_y, dtype=np.float64),
            x=np.asarray(level.point_x, dtype=np.float64),
            labels=labels[np.maximum(np.asarray(level.point_picture, dtype=np.int64), 0)],
            angle_period=float(period),
            detectors=np.asarray(
                [(d.center[0], d.center[1], d.radius, d.lambda_threshold, d.energy) for d in detectors.detectors], dtype=np.float64
            ).reshape(-1, 5),
            layer=np.asarray(detectors.layer_index, dtype=np.int64),
            layer_thresholds=tuple(float(value) for value in detectors.layer_thresholds),
            activation_threshold=float(detectors.activation_lambda),
            # Уровень 0 цикла (StrokeLevel): штрихи идут в окна уровня 1 с кодами его деталей (§6.5, §7.1.12).
            in_cycle=isinstance(level, system.stroke_level()),
        )

    def sticks(self) -> TrainingSticks:
        """Углопозиции обучающих картинок (с видами): из них рисуется содержимое точек карт.

        Returns:
            `TrainingSticks`: картинка, клетка и угол каждой углопозиции; цифра и номер MNIST каждой картинки.
        """
        groups = self._ladder.groups
        atoms = groups.atoms
        taught = groups.train_pictures
        rows = atoms.picture < taught
        return TrainingSticks(
            picture=np.asarray(atoms.picture[rows], dtype=np.int64),
            y=np.asarray(atoms.y[rows], dtype=np.float64),
            x=np.asarray(atoms.x[rows], dtype=np.float64),
            angle=np.asarray(atoms.angle[rows], dtype=np.float64),
            span=np.asarray(atoms.span[rows], dtype=np.float64),
            labels=np.asarray(atoms.labels[:taught], dtype=np.int64),
            index=np.asarray(atoms.index[:taught], dtype=np.int64),
            originals=int(groups.original_pictures),
        )
