"""Эталоны для честного сравнения: kNN по пикселям и обычная свёрточная сеть (вне статьи).

Цель проекта — догнать и обогнать обычную CNN «при прочих равных»: те же обучающие картинки, настройки и ранняя
остановка — только по отложенной выборке validation, проверочные (test) эталоны не видят. Картинки — N × 28 × 28,
значения 0…255 (как DigitAtoms.images) или 0…1; внутри всё приводится к [0, 1].
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from damp import settings
from damp.article_refs import OUTSIDE_ARTICLE
from damp.logging import LOGGER


def unit_pixels(images: "np.ndarray") -> "np.ndarray":
    """Картинки как float32 в [0, 1]: значения больше 1 считаются байтами 0…255 (вне статьи)."""
    pixels = np.asarray(images)
    if pixels.ndim == 4:
        pixels = pixels[:, 0]
    pixels = pixels.astype(np.float32)
    if pixels.size and float(pixels.max()) > 1.0:
        pixels = pixels / 255.0
    return pixels


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class PixelNearest:
    """k ближайших по пикселям: картинка — вектор 784, евклидово расстояние, голос большинства (вне статьи).

    При равенстве голосов — класс с меньшим номером, как у DAMP.
    """

    def __init__(self, images: "np.ndarray", labels: "np.ndarray") -> None:
        self.train = unit_pixels(images).reshape(len(labels), -1)
        self.labels = np.asarray(labels, dtype=np.int64)

    def order(self, images: "np.ndarray", kmax: int) -> "np.ndarray":
        """Метки kmax ближайших обучающих картинок для каждой картинки images, от ближней к дальней."""
        import torch

        device = _device()
        train = torch.as_tensor(self.train, device=device, dtype=torch.float64)
        labels = torch.as_tensor(self.labels, device=device)
        query = torch.as_tensor(unit_pixels(images).reshape(len(images), -1), device=device, dtype=torch.float64)
        kmax = min(int(kmax), train.shape[0])
        found = []
        for first in range(0, query.shape[0], 1024):
            distance = torch.cdist(query[first : first + 1024], train)
            # Равные расстояния — по номеру обучающей картинки (стабильная сортировка).
            index = torch.sort(distance, dim=1, stable=True).indices[:, :kmax]
            found.append(labels[index].cpu())
        return torch.cat(found).numpy() if found else np.zeros((0, kmax), dtype=np.int64)

    @staticmethod
    def vote(neighbours: "np.ndarray", k: int, classes: int = 10) -> "np.ndarray":
        """Голос большинства среди первых k соседей; при равенстве — меньший номер класса."""
        counts = np.zeros((neighbours.shape[0], classes), dtype=np.int64)
        for column in range(min(int(k), neighbours.shape[1])):
            np.add.at(counts, (np.arange(neighbours.shape[0]), neighbours[:, column]), 1)
        return counts.argmax(axis=1)  # argmax берёт первый максимум — меньший номер

    def predict(self, images: "np.ndarray", k: int) -> "np.ndarray":
        return self.vote(self.order(images, k), k)

    def accuracy(self, images: "np.ndarray", labels: "np.ndarray", ks: Sequence[int] | None = None) -> dict[int, float]:
        """Точность для каждого k из ks (по умолчанию BENCH_PIXEL_NEIGHBOURS)."""
        ks = tuple(int(k) for k in (settings.BENCH_PIXEL_NEIGHBOURS if ks is None else ks))
        neighbours = self.order(images, max(ks))
        truth = np.asarray(labels)
        result = {k: float((self.vote(neighbours, k) == truth).mean()) for k in ks}
        LOGGER.event(
            "bench.pixel_nearest.accuracy",
            section=OUTSIDE_ARTICLE,
            data={"train": int(self.labels.shape[0]), "pictures": int(truth.shape[0]), **{f"k{k}": a for k, a in result.items()}},
        )
        return result


class ShiftAugment:
    """Случайный сдвиг каждой картинки на целое число пикселей в пределах ±pixels, нулевое заполнение (вне статьи).

    Сдвиг (dy, dx) — как MnistDigits.shift_pixels: освободившееся место нули, ушедшее за край пропадает.
    """

    def __init__(self, pixels: int) -> None:
        self.pixels = int(pixels)

    def draw(self, count: int, generator=None, device: str = "cpu"):
        """Сдвиги (dy, dx) для count картинок: два тензора целых в [-pixels, pixels]."""
        import torch

        high = self.pixels + 1
        dy = torch.randint(-self.pixels, high, (count,), generator=generator, device=device)
        dx = torch.randint(-self.pixels, high, (count,), generator=generator, device=device)
        return dy, dx

    def apply(self, batch, dy, dx):
        """Картинки batch (B × 1 × H × W) со сдвигами dy, dx (по одному на картинку)."""
        import torch

        m = self.pixels
        if m == 0:
            return batch
        _, _, height, width = batch.shape
        padded = torch.nn.functional.pad(batch, (m, m, m, m))
        rows = torch.arange(height, device=batch.device)[None, :] + (m - dy)[:, None]  # B × H
        cols = torch.arange(width, device=batch.device)[None, :] + (m - dx)[:, None]  # B × W
        index = torch.arange(batch.shape[0], device=batch.device)[:, None, None]
        return padded[:, 0][index, rows[:, :, None], cols[:, None, :]][:, None]

    def __call__(self, batch, generator=None):
        dy, dx = self.draw(batch.shape[0], generator, str(batch.device))
        return self.apply(batch, dy, dx)


@dataclass
class ConvNetResult:
    """Итог обучения одной конфигурации CNN на одном наборе (вне статьи)."""

    name: str
    best_epoch: int
    validation_accuracy: float
    epochs_run: int
    seconds: float
    history: list[dict[str, float]] = field(default_factory=list)  # по эпохам: epoch, loss, validation_accuracy


class ConvNetBaseline:
    """Обычная свёрточная сеть «при прочих равных» (вне статьи): учится на тех же картинках, что и DAMP.

    Гиперпараметры и ранняя остановка — только по отложенной выборке validation. Архитектуры: "lenet5"
    (conv 5×5 6 → maxpool → conv 5×5 16 → maxpool → fc 120 → fc 84 → fc 10) и "wide" (conv 3×3 32 → conv 3×3 32 →
    maxpool → conv 3×3 64 → maxpool → fc 128 → fc 10), ReLU, dropout перед последним слоем. Adam, cross-entropy.
    Детерминизм: зерно на запуск (torch, numpy, cudnn.deterministic).
    """

    ARCHS = ("lenet5", "wide")

    def __init__(
        self,
        arch: str = "lenet5",
        *,
        lr: float | None = None,
        weight_decay: float | None = None,
        dropout: float | None = None,
        augment: str = "none",
        seed: int = 0,
    ) -> None:
        if arch not in self.ARCHS:
            raise ValueError(f"unknown architecture: {arch}")
        if augment not in ("none", "shifts"):
            raise ValueError(f"unknown augmentation: {augment}")
        self.arch = arch
        self.lr = float(settings.BENCH_CNN_LR[0] if lr is None else lr)
        self.weight_decay = float(settings.BENCH_CNN_WEIGHT_DECAY[0] if weight_decay is None else weight_decay)
        self.dropout = float(settings.BENCH_CNN_DROPOUT[0] if dropout is None else dropout)
        self.augment = augment
        self.seed = int(seed)
        self.network = None
        self.result: ConvNetResult | None = None

    @property
    def name(self) -> str:
        """Имя конфигурации — путь в Rerun и поле журнала."""
        return f"{self.arch}-lr{self.lr:g}-wd{self.weight_decay:g}-do{self.dropout:g}-{self.augment}"

    def _build(self):
        from torch import nn

        if self.arch == "lenet5":
            return nn.Sequential(
                nn.Conv2d(1, 6, 5), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(), nn.Linear(16 * 4 * 4, 120), nn.ReLU(),
                nn.Linear(120, 84), nn.ReLU(), nn.Dropout(self.dropout), nn.Linear(84, 10),
            )
        return nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.Conv2d(32, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 5 * 5, 128), nn.ReLU(), nn.Dropout(self.dropout), nn.Linear(128, 10),
        )

    def _seed(self) -> None:
        import torch

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def _tensor(images: "np.ndarray", device: str):
        import torch

        return torch.as_tensor(unit_pixels(images), device=device)[:, None]

    def fit(
        self,
        images: "np.ndarray",
        labels: "np.ndarray",
        validation_images: "np.ndarray",
        validation_labels: "np.ndarray",
        *,
        dataset: str = "set",
        epochs: int | None = None,
        patience: int | None = None,
        batch: int | None = None,
    ) -> ConvNetResult:
        """Учит сеть на images/labels; остановка по точности на validation, остаются веса лучшей эпохи.

        dataset — имя обучающего набора для пути в Rerun (bench/cnn/<конфигурация>/<набор>/…).
        """
        import torch
        from torch import nn

        epochs = int(settings.BENCH_CNN_EPOCHS if epochs is None else epochs)
        patience = int(settings.BENCH_CNN_PATIENCE if patience is None else patience)
        batch = int(settings.BENCH_CNN_BATCH if batch is None else batch)
        device = _device()
        self._seed()
        network = self._build().to(device)
        optimiser = torch.optim.Adam(network.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_function = nn.CrossEntropyLoss()
        train_x = self._tensor(images, device)
        train_y = torch.as_tensor(np.asarray(labels), device=device, dtype=torch.long)
        check_x = self._tensor(validation_images, device)
        check_truth = np.asarray(validation_labels)
        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed)
        shifts = ShiftAugment(settings.BENCH_SHIFT_PIXELS) if self.augment == "shifts" else None
        path = f"bench/cnn/{self.name}/{dataset}"
        history: list[dict[str, float]] = []
        best_accuracy, best_epoch, best_state = -1.0, 0, None
        started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            network.train()
            order = torch.randperm(train_x.shape[0], generator=generator, device=device)
            total = 0.0
            for first in range(0, order.shape[0], batch):
                chosen = order[first : first + batch]
                inputs = train_x[chosen]
                if shifts is not None:
                    inputs = shifts(inputs, generator)
                optimiser.zero_grad()
                loss = loss_function(network(inputs), train_y[chosen])
                loss.backward()
                optimiser.step()
                total += float(loss.detach()) * chosen.shape[0]
            loss_mean = total / max(1, int(train_x.shape[0]))
            accuracy = float((self._predict(network, check_x) == check_truth).mean())
            history.append({"epoch": epoch, "loss": loss_mean, "validation_accuracy": accuracy})
            LOGGER.series(path, {"loss": loss_mean, "validation_accuracy": accuracy}, step=epoch)
            if accuracy > best_accuracy:
                best_accuracy, best_epoch = accuracy, epoch
                best_state = copy.deepcopy(network.state_dict())
            elif epoch - best_epoch >= patience:
                break
        network.load_state_dict(best_state)
        network.eval()
        self.network = network
        self.result = ConvNetResult(
            name=self.name,
            best_epoch=best_epoch,
            validation_accuracy=best_accuracy,
            epochs_run=len(history),
            seconds=time.perf_counter() - started,
            history=history,
        )
        LOGGER.event(
            "bench.cnn.trained",
            section=OUTSIDE_ARTICLE,
            data={
                "config": self.name,
                "dataset": dataset,
                "train": int(train_x.shape[0]),
                "best_epoch": best_epoch,
                "epochs_run": len(history),
                "validation_accuracy": best_accuracy,
                "seconds": self.result.seconds,
            },
        )
        return self.result

    @staticmethod
    def _predict(network, tensor) -> "np.ndarray":
        import torch

        network.eval()
        with torch.no_grad():
            parts = [network(tensor[first : first + 2048]).argmax(dim=1) for first in range(0, tensor.shape[0], 2048)]
        return torch.cat(parts).cpu().numpy() if parts else np.zeros(0, dtype=np.int64)

    def predict(self, images: "np.ndarray") -> "np.ndarray":
        """Классы для любых картинок (после fit)."""
        if self.network is None:
            raise RuntimeError("сначала fit")
        device = next(self.network.parameters()).device
        return self._predict(self.network, self._tensor(images, str(device)))

    def accuracy(self, images: "np.ndarray", labels: "np.ndarray") -> float:
        return float((self.predict(images) == np.asarray(labels)).mean())
