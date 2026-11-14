"""Этап 1. Цифры MNIST и их углопозиции одним массивом — вход цикла (вне статьи; что из них кодируется — §4.4, §4.6)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from damp.article_refs import OUTSIDE_ARTICLE
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
        )


class PictureSticks:
    """Кеш углопозиций по картинкам (вне статьи): каждая картинка набора разбирается на штрихи один раз.

    Один файл на набор MNIST и способ извлечения: номера картинок и их углопозиции; новые картинки дописываются.
    Ключ — содержимое файлов набора MNIST, настройки и код извлечения (system/damp/cache.py).
    """

    MODULES = ("damp.stimuli.digits",)

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
        values = np.concatenate([self._found[index] for index in order]) if order else np.zeros((0, 3))
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

    def load(self, *, train: bool, count: int) -> DigitAtoms:
        """Углопозиции первых count цифр набора, у которых они есть; разобранные раньше картинки — из кеша."""
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
        for index, (image, label) in enumerate(dataset):
            if len(labels) >= count:
                break
            pixels = image.squeeze(0).numpy()
            values = None if store is None else store.get(index)
            if values is None:
                found = self.extractor.extract(pixels, label=int(label)).get(int(label), [])
                values = np.asarray(found, dtype=np.float64).reshape(-1, 3)  # (угол, x, y)
                if store is not None:
                    store.put(index, values)
            if values.shape[0] == 0:
                continue
            images.append((pixels * 255.0).astype(np.uint8))
            rows.append(np.column_stack([np.full(values.shape[0], len(labels), dtype=np.float64), values]))
            labels.append(int(label))
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
        data = np.concatenate(rows)
        atoms = DigitAtoms(
            picture=data[:, 0].astype(np.int64),
            angle=data[:, 1],
            y=np.rint(data[:, 3]).astype(np.int64),
            x=np.rint(data[:, 2]).astype(np.int64),
            labels=np.asarray(labels, dtype=np.int64),
            images=np.stack(images),
        )
        LOGGER.event(
            "stimuli.digits",
            section=OUTSIDE_ARTICLE,
            data={
                "split": split,
                "pictures": atoms.pictures,
                "atoms": int(atoms.picture.shape[0]),
                "atoms_per_picture": float(atoms.picture.shape[0]) / atoms.pictures,
            },
        )
        return atoms


@dataclass(frozen=True)
class DigitGroups:
    """Картинки прогона одним набором: сначала обучающие, затем проверочные и их сдвиги (вне статьи).

    ranges — группа → (первая картинка, после последней): "train", "test" и сдвиги (dy, dx). Обучающие всегда
    первые: память уровней строится только по ним, остальные картинки — только предъявляются.
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
    ) -> "DigitGroups":
        """train обучающих цифр, test проверочных из набора split и сдвиги первых shift_pictures проверочных."""
        taught = digits.load(train=True, count=int(train))
        checked = digits.load(train=split == "train", count=int(test)) if test else None
        atoms = taught.joined(checked)
        ranges: dict[object, tuple[int, int]] = {"train": (0, taught.pictures)}
        if checked is not None:
            ranges["test"] = (taught.pictures, atoms.pictures)
            for dy, dx in shifts:
                moved = checked.shifted(int(dy), int(dx), pictures=int(shift_pictures))
                ranges[(int(dy), int(dx))] = (atoms.pictures, atoms.pictures + moved.pictures)
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
        return int(self.ranges["train"][1])

    @property
    def shifts(self) -> tuple[tuple[int, int], ...]:
        return tuple(key for key in self.ranges if isinstance(key, tuple))

    def sticks(self, picture: int) -> tuple[int, int]:
        """Углопозиции картинки: первая и после последней."""
        return int(self.starts[picture]), int(self.starts[picture + 1])
