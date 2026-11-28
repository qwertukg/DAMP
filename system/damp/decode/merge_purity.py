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

    def nearest(self, packed: "np.ndarray", code_length: int, dictionary: CodeDictionary, rows: "np.ndarray | None" = None) -> "np.ndarray":
        """Номер ближайшей точки карты (образца словаря) для каждого кода: наибольший косинус, при равных — меньший
        номер."""
        import torch

        # rows — номера кодов, которые сравниваются (None — все): коды берутся кусками по номерам, без копии всего набора
        # (на 30 000 картинках копия кодов обучающих окон уровня 0 — 3.5 ГБ ОЗУ).
        rows = np.arange(packed.shape[0], dtype=np.int64) if rows is None else np.asarray(rows, dtype=np.int64)
        use_gpu = settings.ACTIVATION_USE_GPU if self._use_gpu is None else self._use_gpu
        device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        if device.type == "cuda" and bool(settings.ACTIVATION_INT8_CODES) and hasattr(torch, "_int_mm"):
            found = self._nearest_int8(torch, device, packed, int(code_length), dictionary, rows)
            if found is not None:
                return found
        bank = torch.as_tensor(dictionary.bits(), device=device).to(torch.float32)
        bank_ones = torch.clamp(bank.sum(dim=1), min=1.0)
        found = np.zeros(rows.shape[0], dtype=np.int64)
        for start in range(0, rows.shape[0], self.CHUNK):
            block = torch.as_tensor(
                np.unpackbits(packed[rows[start : start + self.CHUNK]], axis=1, count=int(code_length)), device=device
            ).to(torch.float32)
            ones = torch.clamp(block.sum(dim=1), min=1.0)
            cosine = (block @ bank.T) / torch.sqrt(ones[:, None] * bank_ones[None, :])
            found[start : start + self.CHUNK] = cosine.argmax(dim=1).cpu().numpy()
        del bank, bank_ones
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return found

    # torch._int_mm: левая матрица — больше 16 строк, длина кода и число точек — кратны 8 (дополняются нулями: нулевой
    # бит общих бит не добавляет, нулевые точки отбрасываются).
    INT_MM_ROWS_MIN = 17
    INT_MM_MULTIPLE = 8

    def _nearest_int8(self, torch, device, packed: "np.ndarray", code_length: int, dictionary: CodeDictionary,  # noqa: ANN001, ANN202
                      rows: "np.ndarray | None" = None):
        """То же, что nearest на видеокарте, с кодами int8: число общих бит — целочисленное произведение int8 × int8 →
        int32 (torch._int_mm), то же целое, что у произведения float32; косинус — та же формула float32 над теми же
        числами, поэтому ближайшие точки те же. Памяти под коды карты вчетверо меньше, умножение быстрее (2026-11-25:
        мерка на 20 000 картинках шла 10 минут, видеокарта — 98 %). Первая часть сверяется с float32: не сошлось —
        None (считает прежний путь)."""
        multiple = self.INT_MM_MULTIPLE
        bits = np.asarray(dictionary.bits(), dtype=np.uint8)
        points, width = int(bits.shape[0]), -(-int(code_length) // multiple) * multiple
        bank = torch.zeros((-(-points // multiple) * multiple, width), dtype=torch.int8, device=device)
        bank[:points, : bits.shape[1]] = torch.as_tensor(bits, device=device).to(torch.int8)
        bank_ones = torch.clamp(bank[:points].sum(dim=1, dtype=torch.int32).to(torch.float32), min=1.0)
        order = np.arange(packed.shape[0], dtype=np.int64) if rows is None else np.asarray(rows, dtype=np.int64)
        found = np.zeros(order.shape[0], dtype=np.int64)
        try:
            for start in range(0, order.shape[0], self.CHUNK):
                rows = np.unpackbits(packed[order[start : start + self.CHUNK]], axis=1, count=int(code_length))
                taken = int(rows.shape[0])
                block = torch.zeros((max(taken, self.INT_MM_ROWS_MIN), width), dtype=torch.int8, device=device)
                block[:taken, : rows.shape[1]] = torch.as_tensor(rows, device=device).to(torch.int8)
                ones = torch.clamp(block[:taken].sum(dim=1, dtype=torch.int32).to(torch.float32), min=1.0)
                common = torch._int_mm(block, bank.T)[:taken, :points].to(torch.float32)
                cosine = common / torch.sqrt(ones[:, None] * bank_ones[None, :])
                found[start : start + taken] = cosine.argmax(dim=1).cpu().numpy()
                if start == 0:
                    # Сверка первой части с прежним произведением float32.
                    plain = torch.as_tensor(rows, device=device).to(torch.float32)
                    exact = (plain @ bank[:points, : rows.shape[1]].to(torch.float32).T) / torch.sqrt(
                        torch.clamp(plain.sum(dim=1), min=1.0)[:, None] * bank_ones[None, :])
                    if not np.array_equal(exact.argmax(dim=1).cpu().numpy(), found[:taken]):
                        return None
                    del plain, exact
                del block, common, cosine
        finally:
            del bank
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
        rows: "np.ndarray | None" = None,
    ) -> dict[str, float]:
        """Чистота слияния на карте name: packed — коды окон обучающих картинок, picture и labels — картинка и цифра
        каждого окна, dictionary — образцы карты; rows — номера кодов в packed, по одному на окно (None — packed по порядку). Числа — словарём (ключи merge_*), они же — в журнале."""
        picture = np.asarray(picture, dtype=np.int64)
        labels = np.asarray(labels, dtype=np.int64)
        nearest = self.nearest(packed, code_length, dictionary, rows)
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
