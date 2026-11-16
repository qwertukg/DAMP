"""Этап 1. Цифры MNIST и их углопозиции одним массивом — вход цикла (вне статьи; что из них кодируется — §4.4, §4.6)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from damp import settings
from damp.article_refs import OUTSIDE_ARTICLE, TOKEN_ENCODING
from damp.logging import LOGGER
from damp.stimuli.MnistSobelAngleMap import MnistSobelAngleMap
from damp.stimuli.mnist import _build_extractor


@dataclass(frozen=True)
class DigitAtoms:
    """Углопозиции набора цифр одним массивом: картинка, угол и узел сетки окон каждой углопозиции (§4.4, §4.6)."""

    picture: "np.ndarray"  # номер картинки каждой углопозиции
    angle: "np.ndarray"
    y: "np.ndarray"  # узел сетки окон Собеля
    x: "np.ndarray"
    labels: "np.ndarray"  # метка каждой картинки
    images: "np.ndarray"  # сами картинки (N × 28 × 28, 0…255) — для рисунков карт
    strength: "np.ndarray"  # сила штриха (модуль среднего градиента), нормированная внутри своей картинки: (0, 1]
    index: "np.ndarray"  # номер каждой картинки в наборе MNIST (int64)

    @property
    def pictures(self) -> int:
        return int(self.labels.shape[0])

    def joined(self, other: "DigitAtoms | None") -> "DigitAtoms":
        """Этот набор и следом за ним other (номера его картинок сдвигаются)."""
        if other is None:
            return self
        return DigitAtoms(
            picture=np.concatenate([self.picture, other.picture + self.pictures]),
            angle=np.concatenate([self.angle, other.angle]),
            y=np.concatenate([self.y, other.y]),
            x=np.concatenate([self.x, other.x]),
            labels=np.concatenate([self.labels, other.labels]),
            images=np.concatenate([self.images, other.images]),
            strength=np.concatenate([self.strength, other.strength]),
            index=np.concatenate([self.index, other.index]),
        )

    def shifted(self, dy: int, dx: int, *, pictures: int) -> "DigitAtoms":
        """Первые pictures картинок набора, сдвинутые на (dy, dx) клеток (мерка устойчивости к сдвигу, вне статьи)."""
        pictures = min(int(pictures), self.pictures)
        rows = self.picture < pictures
        return DigitAtoms(
            picture=self.picture[rows],
            angle=self.angle[rows],
            y=self.y[rows] + int(dy),
            x=self.x[rows] + int(dx),
            labels=self.labels[:pictures],
            images=self.images[:pictures],
            strength=self.strength[rows],
            index=self.index[:pictures],
        )

    def only(self, first: int, last: int) -> "DigitAtoms":
        """Картинки first…last-1 набора, перенумерованные с нуля (вне статьи): одна группа картинок прогона."""
        rows = (self.picture >= first) & (self.picture < last)
        return DigitAtoms(
            picture=self.picture[rows] - int(first),
            angle=self.angle[rows],
            y=self.y[rows],
            x=self.x[rows],
            labels=self.labels[first:last],
            images=self.images[first:last],
            strength=self.strength[rows],
            index=self.index[first:last],
        )

    @classmethod
    def from_rows(
        cls, rows: Sequence["np.ndarray"], labels: Sequence[int], images: Sequence["np.ndarray"], index: Sequence[int]
    ) -> "DigitAtoms":
        """Набор из углопозиций картинок: rows — по массиву на картинку, строки (картинка, угол, x, y, сила штриха).

        Сила штриха нормируется внутри своей картинки: доля от самого сильного штриха этой картинки, (0, 1]. Место —
        дробное при MNIST_SUBCELL_POSITION (там, где край проходит внутри окна Собеля), иначе номер окна (целое).
        """
        data = np.concatenate(rows) if rows else np.zeros((0, 1 + PictureSticks.COLUMNS), dtype=np.float64)
        picture = data[:, 0].astype(np.int64)
        strength = data[:, 4].copy()
        peak = np.zeros(len(labels), dtype=np.float64)
        np.maximum.at(peak, picture, strength)
        peak[peak <= 0.0] = 1.0
        strength = strength / np.maximum(peak[picture], np.finfo(np.float64).tiny)
        fractional = bool(settings.MNIST_SUBCELL_POSITION)
        return cls(
            picture=data[:, 0].astype(np.int64),
            angle=data[:, 1],
            y=data[:, 3].astype(np.float64) if fractional else np.rint(data[:, 3]).astype(np.int64),
            x=data[:, 2].astype(np.float64) if fractional else np.rint(data[:, 2]).astype(np.int64),
            labels=np.asarray(labels, dtype=np.int64),
            images=np.stack(images) if len(images) else np.zeros((0, 28, 28), dtype=np.uint8),
            strength=strength,
            index=np.asarray(index, dtype=np.int64),
        )


class PictureSticks:
    """Кеш углопозиций по картинкам (вне статьи): каждая картинка набора разбирается на штрихи один раз.

    Один файл на набор MNIST и способ извлечения: номера картинок и их углопозиции (угол, x, y, сила штриха);
    новые картинки дописываются. Файлы старого формата (три столбца, без силы) не читаются: ключ другой, а при
    чтении проверяется ширина.
    Ключ — содержимое файлов набора MNIST, настройки и код извлечения (system/damp/cache.py).
    """

    MODULES = ("damp.stimuli.digits", "damp.stimuli.MnistSobelAngleMap")
    COLUMNS = 4  # угол, x, y, сила штриха

    def __init__(self, split: str, raw_files: Sequence[str]) -> None:
        from damp.cache import CACHE

        sources = []
        for path in raw_files:
            with open(path, "rb") as stream:
                sources.append(stream.read())
        key, _ = CACHE.key("stimuli.sticks", {"split": split, "mnist": tuple(np.frombuffer(s, dtype=np.uint8) for s in sources)}, self.MODULES)
        self.path = CACHE.root / "pictures" / f"{split}-{key[:24]}.npz"
        self._found: dict[int, "np.ndarray"] = {}
        self._added = 0
        self.hits = 0
        if self.path.exists():
            with np.load(self.path, allow_pickle=False) as stored:
                starts, values = stored["starts"], stored["values"]
                if values.ndim == 2 and values.shape[1] == self.COLUMNS:
                    for row, index in enumerate(stored["index"].tolist()):
                        self._found[int(index)] = values[starts[row] : starts[row + 1]]

    def get(self, index: int) -> "np.ndarray | None":
        values = self._found.get(int(index))
        if values is not None:
            self.hits += 1
        return values

    def put(self, index: int, values: "np.ndarray") -> None:
        self._found[int(index)] = values
        self._added += 1

    def save(self) -> None:
        """Дописать новые картинки на диск (атомарно: временный файл → переименование)."""
        if not self._added:
            return
        import os

        order = sorted(self._found)
        sizes = np.asarray([self._found[index].shape[0] for index in order], dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        values = np.concatenate([self._found[index] for index in order]) if order else np.zeros((0, self.COLUMNS))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f"{self.path.stem}.{os.getpid()}.tmp.npz")
        np.savez(temporary, index=np.asarray(order, dtype=np.int64), starts=starts, values=values.astype(np.float64))
        os.replace(temporary, self.path)
        self._added = 0


class MnistDigits:
    """Цифры MNIST с углопозициями: первые count картинок набора, у которых углопозиции есть."""

    def __init__(self, extractor: MnistSobelAngleMap | None = None, *, root: str = "./data") -> None:
        self.extractor = extractor if extractor is not None else _build_extractor()
        self._root = str(root)

    def sticks_of(self, pixels: "np.ndarray", label: int = 0) -> "np.ndarray":
        """Углопозиции одной картинки 28×28 (§4.4): строки (угол, x, y, сила штриха); пикселы — float32 в [0, 1],
        белая цифра на чёрном, как у MNIST после ToTensor. Тот же разбор, что у load, — для любой картинки; label —
        цифра картинки, только для журнала разбора (на углопозиции не влияет)."""
        found = self.extractor.extract(pixels, label=int(label), with_magnitude=True).get(int(label), [])
        return np.asarray(found, dtype=np.float64).reshape(-1, PictureSticks.COLUMNS)

    @staticmethod
    def shift_pixels(pixels: "np.ndarray", dy: int, dx: int) -> "np.ndarray":
        """Картинка, сдвинутая на (dy, dx) пикселей; освободившееся место — нулями, ушедшее за край пропадает."""
        moved = np.zeros_like(pixels)
        height, width = pixels.shape
        dy, dx = int(dy), int(dx)
        if abs(dy) >= height or abs(dx) >= width:
            return moved
        moved[max(0, dy) : height + min(0, dy), max(0, dx) : width + min(0, dx)] = pixels[
            max(0, -dy) : height + min(0, -dy), max(0, -dx) : width + min(0, -dx)
        ]
        return moved

    def load(
        self,
        *,
        train: bool,
        count: int,
        start: int = 0,
        shift: tuple[int, int] = (0, 0),
        numbers: "Sequence[int] | None" = None,
    ) -> DigitAtoms:
        """Углопозиции count цифр набора начиная с номера start, у которых они есть; разобранные раньше — из кеша.

        shift — сдвиг всей картинки 28×28 на (dy, dx) пикселей с нулевым заполнением (мерка устойчивости к
        деформации, вне статьи): такие картинки в кеш штрихов не пишутся и из него не берутся.
        numbers — взять ровно эти картинки набора, в этом порядке, не пропуская безштриховые (так сдвинутая копия
        набора совпадает с ним по номерам картинок); count и start тогда не используются.
        """
        from torchvision import transforms
        from torchvision.datasets import MNIST

        from damp.cache import CACHE

        dataset = MNIST(root=self._root, train=train, download=True, transform=transforms.ToTensor())
        split = "train" if train else "test"
        store = None
        if CACHE.enabled:
            prefix = "train" if train else "t10k"
            store = PictureSticks(
                split,
                [f"{dataset.raw_folder}/{prefix}-images-idx3-ubyte", f"{dataset.raw_folder}/{prefix}-labels-idx1-ubyte"],
            )
        rows: list["np.ndarray"] = []
        labels: list[int] = []
        images: list["np.ndarray"] = []
        taken: list[int] = []  # номера взятых картинок в наборе MNIST
        moved = (int(shift[0]), int(shift[1])) != (0, 0)
        if moved:
            store = None
        wanted = None if numbers is None else [int(number) for number in numbers]
        for index in range(int(start), len(dataset)) if wanted is None else wanted:
            if wanted is None and len(labels) >= count:
                break
            image, label = dataset[index]
            pixels = image.squeeze(0).numpy()
            if moved:
                pixels = self.shift_pixels(pixels, shift[0], shift[1])
            values = None if store is None else store.get(index)
            if values is None:
                values = self.sticks_of(pixels, int(label))  # (угол, x, y, сила)
                if store is not None:
                    store.put(index, values)
            if values.shape[0] == 0 and wanted is None:
                continue
            images.append((pixels * 255.0).astype(np.uint8))
            rows.append(np.column_stack([np.full(values.shape[0], len(labels), dtype=np.float64), values]))
            labels.append(int(label))
            taken.append(index)
        if store is not None:
            store.save()
            LOGGER.event(
                "cache.pictures",
                section=OUTSIDE_ARTICLE,
                data={"split": split, "from_cache": store.hits, "path": str(store.path),
                      "note": "углопозиции картинок, разобранных раньше, взяты из кеша; новые дописаны"},
            )
        if not rows:
            raise ValueError("no digits with angle-positions")
        if wanted is not None and len(labels) != len(wanted):
            raise ValueError("pictures were lost")
        atoms = DigitAtoms.from_rows(rows, labels, images, taken)
        LOGGER.event(
            "stimuli.digits",
            section=OUTSIDE_ARTICLE,
            data={
                "split": split,
                "start": int(start),
                "shift": [int(shift[0]), int(shift[1])],
                "pictures": atoms.pictures,
                "atoms": int(atoms.picture.shape[0]),
                "atoms_per_picture": float(atoms.picture.shape[0]) / atoms.pictures,
            },
        )
        return atoms


@dataclass(frozen=True)
class DigitGroups:
    """Картинки прогона одним набором: сначала обучающие, затем отложенные и проверочные и их сдвиги (вне статьи).

    Порядок картинок: обучающие (MNIST train с train_start или ровно train_numbers), отложенные validation (MNIST
    train с validation_start или ровно validation_numbers), проверочные test (набор split, первые или ровно
    test_numbers), сдвиги проверочных на клетки (dy, dx), первые картинки отложенных, сдвинутые на пиксели (ключ
    ("pixel", dy, dx)). Стенд bench передаёт явные номера — случайную выборку из пулов (MNIST не перемешан по
    авторам, сплошной блок нечестен); картинки без углопозиций тогда не пропускаются, а остаются без ответа.
    ranges — группа → (первая картинка, после последней): "train", "validation" (если есть), "test" (если есть),
    сдвиги. Обучающие всегда первые: память уровней строится только по ним, остальные картинки — только
    предъявляются. Отложенная выборка с обучающими по номерам MNIST не пересекается.
    """

    atoms: DigitAtoms
    ranges: dict[object, tuple[int, int]]
    starts: "np.ndarray" = field(repr=False)  # первая углопозиция каждой картинки и после последней

    @classmethod
    def of(cls, atoms: DigitAtoms, ranges: dict[object, tuple[int, int]]) -> "DigitGroups":
        return cls(atoms=atoms, ranges=dict(ranges), starts=np.searchsorted(atoms.picture, np.arange(atoms.pictures + 1)))

    @classmethod
    def load(
        cls,
        digits: MnistDigits,
        *,
        train: int,
        test: int,
        split: str = "test",
        shifts: Sequence[tuple[int, int]] = (),
        shift_pictures: int = 0,
        train_start: int = 0,
        validation: int = 0,
        validation_start: int | None = None,
        pixel_shift: tuple[int, int] | None = None,
        pixel_pictures: int = 0,
        train_numbers: Sequence[int] | None = None,
        validation_numbers: Sequence[int] | None = None,
        test_numbers: Sequence[int] | None = None,
        train_views: Sequence[tuple[int, int]] = (),
    ) -> "DigitGroups":
        """Обучающие, отложенные, проверочные цифры и их сдвиги (на клетки и на пиксели) одним набором.

        train обучающих цифр с номера train_start; validation отложенных с номера validation_start (None — начало
        пула settings.BENCH_VALIDATION_POOL); test проверочных из набора split; сдвиги на клетки — первых
        shift_pictures проверочных; сдвиг на pixel_shift пикселей — первых pixel_pictures отложенных.
        train_numbers, validation_numbers, test_numbers — ровно эти номера MNIST в этом порядке (число картинок
        группы — их число); картинки без углопозиций не пропускаются.
        train_views — ещё виды каждой обучающей цифры, сдвинутые на (dy, dx) пикселей (§7.1.4: слово берётся в
        нескольких разбиениях сразу, чтобы сходство не зависело от разбиения; у картинки «разбиение» — то, в какие
        клетки сетки попали штрихи). Виды идут следом за обучающими и сами обучающие: память строится и по ним.
        """
        if train_numbers is not None:
            taught = digits.load(train=True, count=len(train_numbers), numbers=train_numbers)
        else:
            taught = digits.load(train=True, count=int(train), start=int(train_start))
        views = tuple((int(dy), int(dx)) for dy, dx in train_views if (int(dy), int(dx)) != (0, 0))
        originals_count = taught.pictures
        if views:
            originals = taught
            for dy, dx in views:
                taught = taught.joined(digits.load(train=True, shift=(dy, dx), count=originals.pictures, numbers=originals.index))
            LOGGER.event(
                "stimuli.views",
                section=TOKEN_ENCODING,
                data={"pictures": originals.pictures, "views": [list(view) for view in views], "train_pictures": taught.pictures,
                      "note": "каждая обучающая цифра ещё и в сдвинутых видах — несколько разбиений одной картинки на клетки "
                              "сетки (как несколько разбиений слова на токены, §7.1.4); виды — тоже обучающие"},
            )
        held = None
        if validation_numbers is not None and len(validation_numbers):
            held = digits.load(train=True, count=len(validation_numbers), numbers=validation_numbers)
        elif validation:
            first = int(settings.BENCH_VALIDATION_POOL[0] if validation_start is None else validation_start)
            held = digits.load(train=True, count=int(validation), start=first)
        if held is not None:
            assert not set(held.index.tolist()) & set(taught.index.tolist()), "отложенная выборка пересекается с обучающей"
        checked = None
        if test_numbers is not None and len(test_numbers):
            checked = digits.load(train=split == "train", count=len(test_numbers), numbers=test_numbers)
        elif test:
            checked = digits.load(train=split == "train", count=int(test))
        atoms = taught.joined(held).joined(checked) if held is not None else taught.joined(checked)
        ranges: dict[object, tuple[int, int]] = {"train": (0, taught.pictures)}
        if views:
            ranges["views"] = (originals_count, taught.pictures)  # виды — часть обучающих, следом за исходными цифрами
        if held is not None:
            ranges["validation"] = (taught.pictures, taught.pictures + held.pictures)
        if checked is not None:
            ranges["test"] = (atoms.pictures - checked.pictures, atoms.pictures)
            for dy, dx in shifts:
                moved = checked.shifted(int(dy), int(dx), pictures=int(shift_pictures))
                ranges[(int(dy), int(dx))] = (atoms.pictures, atoms.pictures + moved.pictures)
                atoms = atoms.joined(moved)
        if held is not None and pixel_shift is not None and int(pixel_pictures) > 0:
            pictures = min(int(pixel_pictures), held.pictures)
            dy, dx = int(pixel_shift[0]), int(pixel_shift[1])
            moved = digits.load(train=True, shift=(dy, dx), count=pictures, numbers=held.index[:pictures])
            ranges[("pixel", dy, dx)] = (atoms.pictures, atoms.pictures + moved.pictures)
            atoms = atoms.joined(moved)
        return cls.of(atoms, ranges)

    @property
    def labels(self) -> "np.ndarray":
        return self.atoms.labels

    @property
    def pictures(self) -> int:
        return self.atoms.pictures

    @property
    def train_pictures(self) -> int:
        """Обучающих картинок вместе с видами (§7.1.4): по ним всем строится память."""
        return int(self.ranges["train"][1])

    @property
    def original_pictures(self) -> int:
        """Обучающих цифр без видов: исходные картинки идут первыми, виды — следом (ranges["views"])."""
        return int(self.ranges["views"][0]) if "views" in self.ranges else self.train_pictures

    @property
    def shifts(self) -> tuple[tuple[int, int], ...]:
        return tuple(key for key in self.ranges if isinstance(key, tuple) and key[0] != "pixel")

    @property
    def pixel_shifts(self) -> tuple[tuple[int, int], ...]:
        """Сдвиги отложенных картинок на пиксели (dy, dx)."""
        return tuple((int(key[1]), int(key[2])) for key in self.ranges if isinstance(key, tuple) and key[0] == "pixel")

    def sticks(self, picture: int) -> tuple[int, int]:
        """Углопозиции картинки: первая и после последней."""
        return int(self.starts[picture]), int(self.starts[picture + 1])


class ExampleDigit:
    """Цифра-пример прогона: одна обучающая картинка цифры LADDER_FIGURE_DIGIT, общая для рисунков и Rerun (вне статьи).

    LADDER_FIGURE_PICTURE = "random" — в каждом прогоне другая пятёрка (np.random.default_rng() без зерна);
    число — номер цифры по счёту среди обучающих картинок (нет такой — последняя). Выбор делается один раз за
    прогон и записывается в журнал событием ladder.example.
    """

    @staticmethod
    def choose(groups: "DigitGroups") -> int:
        """Номер картинки-примера в наборе; среди обучающих её цифра — LADDER_FIGURE_DIGIT, иначе — первая обучающая."""
        atoms = groups.atoms
        taught = groups.original_pictures  # среди исходных цифр: сдвинутый вид показал бы не ту картинку, что номер MNIST
        digit = int(settings.LADDER_FIGURE_DIGIT)
        # Только картинки со штрихами: при явных номерах MNIST картинка без углопозиций остаётся в наборе, а у неё
        # нет ни окон, ни деталей — показывать нечего.
        drawn = np.zeros(taught, dtype=bool)
        drawn[np.unique(atoms.picture[atoms.picture < taught])] = True
        found = np.flatnonzero((atoms.labels[:taught] == digit) & drawn)
        wish = settings.LADDER_FIGURE_PICTURE
        if found.size == 0:
            picture, how = 0, "нет такой цифры среди обучающих — первая картинка"
        elif isinstance(wish, str):
            if wish != "random":
                raise ValueError(f"LADDER_FIGURE_PICTURE: число или \"random\", а не {wish!r}")
            picture, how = int(found[np.random.default_rng().integers(found.size)]), "случайная, без зерна"
        else:
            picture, how = int(found[min(int(wish), found.size - 1)]), f"по счёту {int(wish)}"
        LOGGER.event(
            "ladder.example",
            section=OUTSIDE_ARTICLE,
            data={
                "picture": picture,
                "mnist_index": int(atoms.index[picture]),
                "label": int(atoms.labels[picture]),
                "choice": how,
                "note": "пример для рисунков и Rerun: каждый прогон — другая обучающая картинка цифры-примера",
            },
        )
        return picture
