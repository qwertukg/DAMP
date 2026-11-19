"""Этап 7, мерка: чистота слияния на карте памяти уровня (вне статьи).

На карте уровня похожие коды окон сливаются в одну точку (§2.2.4): точка хранит не одно окно, а всё, что на него
похоже. Идеал автора (2026-11-18): «в идеале должно остаться 10 классов на последнем уровне» — точки верхней карты
должны собирать картинки одной цифры. Мерка отвечает на вопрос: одной ли цифры картинки, попавшие в одну точку.

Как считается. Каждое окно обучающей картинки относится к своей ближайшей точке карты — самому похожему образцу
(косинус, §2.2.5). У точки — набор разных картинок, чьи окна к ней отнесены, и цифры этих картинок.

    общая точка        — точка, в которую попали окна двух и более картинок: только в ней есть слияние между картинками;
    чистота            — доля картинок общих точек, чья цифра — главная цифра своей точки;
    чистые точки       — доля общих точек, все картинки которых одной цифры;
    случайная чистота  — та же чистота при цифрах картинок, перемешанных между картинками: с чем сравнивать.

Точка с окнами одной картинки чиста по построению и в чистоту не входит.
"""
from __future__ import annotations

import numpy as np

from damp import settings
from damp.article_refs import OUTSIDE_ARTICLE
from damp.layout.code_dictionary import CodeDictionary
from damp.logging import LOGGER


class MergePurity:
    """Чистота слияния кодов окон в точки карты памяти: одной ли цифры картинки одной точки (мерка, вне статьи)."""

    CHUNK = 2048  # окон в одной части при поиске ближайшей точки

    def __init__(self, *, seed: int = 0, use_gpu: bool | None = None) -> None:
        self._seed = int(seed)
        self._use_gpu = use_gpu

    def nearest(self, packed: "np.ndarray", code_length: int, dictionary: CodeDictionary) -> "np.ndarray":
        """Номер ближайшей точки карты (образца словаря) для каждого кода: наибольший косинус, при равных — меньший
        номер."""
        import torch

        packed = np.ascontiguousarray(packed, dtype=np.uint8)
        use_gpu = settings.ACTIVATION_USE_GPU if self._use_gpu is None else self._use_gpu
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        bank = torch.as_tensor(dictionary.bits(), device=device).to(torch.float32)
        bank_ones = torch.clamp(bank.sum(dim=1), min=1.0)
        found = np.zeros(packed.shape[0], dtype=np.int64)
        for start in range(0, packed.shape[0], self.CHUNK):
            block = torch.as_tensor(
                np.unpackbits(packed[start : start + self.CHUNK], axis=1, count=int(code_length)), device=device
            ).to(torch.float32)
            ones = torch.clamp(block.sum(dim=1), min=1.0)
            cosine = (block @ bank.T) / torch.sqrt(ones[:, None] * bank_ones[None, :])
            found[start : start + self.CHUNK] = cosine.argmax(dim=1).cpu().numpy()
        del bank, bank_ones
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return found

    @staticmethod
    def shares(point: "np.ndarray", digit: "np.ndarray", points: int, classes: int) -> tuple[float, float, float, float, float]:
        """По парам «точка — картинка» (digit — цифра картинки): (картинок на точку, доля общих точек, чистота общих
        точек, доля чистых среди общих, цифр на общую точку). Общих точек нет — чистота 1: смешивать нечего."""
        counts = np.zeros((int(points), int(classes)), dtype=np.int64)
        np.add.at(counts, (np.asarray(point, dtype=np.int64), np.asarray(digit, dtype=np.int64)), 1)
        held = counts.sum(axis=1)
        used = held > 0
        shared = held >= 2
        if not used.any():
            return 0.0, 0.0, 1.0, 1.0, 0.0
        per_point = float(held[used].mean())
        if not shared.any():
            return per_point, 0.0, 1.0, 1.0, 0.0
        kinds = np.count_nonzero(counts[shared], axis=1)
        return (
            per_point,
            float(shared.sum()) / float(used.sum()),
            float(counts[shared].max(axis=1).sum()) / float(held[shared].sum()),
            float(np.mean(kinds == 1)),
            float(kinds.mean()),
        )

    def measure(
        self,
        packed: "np.ndarray",
        code_length: int,
        picture: "np.ndarray",
        labels: "np.ndarray",
        dictionary: CodeDictionary,
        *,
        name: str,
    ) -> dict[str, float]:
        """Чистота слияния на карте name: packed — коды окон обучающих картинок, picture и labels — картинка и цифра
        каждого окна, dictionary — образцы карты. Числа — словарём (ключи merge_*), они же — в журнале."""
        picture = np.asarray(picture, dtype=np.int64)
        labels = np.asarray(labels, dtype=np.int64)
        nearest = self.nearest(packed, code_length, dictionary)
        pairs = np.unique(np.stack([nearest, picture], axis=1), axis=0)  # разные пары «точка — картинка»
        point, owner = pairs[:, 0], pairs[:, 1]
        of_picture = np.zeros(int(picture.max()) + 1 if picture.size else 0, dtype=np.int64)
        of_picture[picture] = labels
        classes = int(labels.max()) + 1 if labels.size else 1
        per_point, shared, purity, pure, kinds = self.shares(point, of_picture[owner], dictionary.count, classes)
        # С чем сравнивать: те же точки и картинки, цифры перемешаны между картинками.
        present = np.unique(picture)
        mixed = of_picture.copy()
        mixed[present] = np.random.default_rng(self._seed).permutation(of_picture[present])
        _, _, random_purity, random_pure, _ = self.shares(point, mixed[owner], dictionary.count, classes)
        found = {
            "merge_pictures_per_point": per_point,
            "merge_shared_points": shared,
            "merge_purity": purity,
            "merge_purity_random": random_purity,
            "merge_pure_points": pure,
            "merge_pure_points_random": random_pure,
            "merge_digits_per_point": kinds,
        }
        LOGGER.event(
            "ladder.merge",
            section=OUTSIDE_ARTICLE,
            data={
                "name": name,
                "keyholes": int(picture.shape[0]),
                "pictures": int(present.shape[0]),
                "points": int(dictionary.count),
                **found,
                "note": "чистота слияния (мерка, вне статьи): окно обучающей картинки относится к ближайшей точке карты "
                "(§2.2.5); общая точка — с окнами двух и более картинок; чистота — доля картинок общих точек, чья цифра "
                "— главная цифра точки; случайная — при перемешанных цифрах картинок. Идеал автора: на последнем уровне "
                "остаётся 10 классов",
            },
        )
        return found
