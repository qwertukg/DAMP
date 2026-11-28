"""Словарь образцов: слияние похожих кодов к образцу перед раскладкой (расширение статьи).

Кодов кусков сотни тысяч, и почти все различны, но большинство отличается от уже встречавшихся на несколько
бит. Раскладывать их все незачем: код присоединяется к самому раннему образцу, с которым его сходство
(§2.2.4) не ниже порога, иначе сам становится образцом. Коды обходятся от частых к редким, поэтому
образцами становятся самые частые из различимых кодов.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from damp.article_refs import OUTSIDE_ARTICLE, SIMILARITY_MEASURES
from damp.logging import LOGGER


@dataclass(frozen=True)
class CodeDictionary:
    """Образцы словаря: их коды и строка входа, где образец встретился первым."""

    packed: "np.ndarray"  # (S, ⌈L/8⌉) коды образцов
    rows: "np.ndarray"  # (S,) номер строки входа с первым вхождением образца
    code_length: int
    distinct: int  # сколько было различных кодов

    @property
    def count(self) -> int:
        return int(self.packed.shape[0])

    def bits(self) -> "np.ndarray":
        return np.unpackbits(self.packed, axis=1, count=self.code_length)


class LeaderDictionary:
    """Слияние кодов к образцу по косинусному сходству (расширение статьи; мера сходства — §2.2.4)."""

    BLOCK = 512
    # Образцов в одной части банка. Банк растёт частями, и блок кодов сравнивается с ним по частям: память
    # устройства определяется числом найденных образцов, а не пределом max_size.
    BANK_PART = 65536

    def __init__(self, *, similarity_threshold: float, max_size: int, use_gpu: bool = True) -> None:
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")
        if max_size < 1:
            raise ValueError("max_size must be positive")
        self.similarity_threshold = float(similarity_threshold)
        self.max_size = int(max_size)
        self._use_gpu = bool(use_gpu)

    # Строк кодов в одном куске при поиске одинаковых кодов (хеш строк): память — кусок, а не весь набор.
    HASH_ROWS = 65536
    # Попыток хеша с разными зёрнами, прежде чем искать одинаковые коды сортировкой копии всего набора.
    HASH_ATTEMPTS = 3

    @staticmethod
    def _mix(words: "np.ndarray") -> "np.ndarray":
        """Перемешивание 64-битных слов (финализатор splitmix64): взаимно однозначно, и каждый бит слова влияет на все
        биты результата. Без него сумма слов с нечётными множителями mod 2⁶⁴ теряет разность старших бит: коды,
        различные только старшими битами двух слов (2⁶³ + 2⁶³ ≡ 0), получали один хеш (на 50 000 картинках так и
        вышло — и отбор образцов уровня 1 ушёл в сортировку копии всех кодов, +6 ГБ ОЗУ)."""
        mixed = words ^ (words >> np.uint64(30))
        mixed *= np.uint64(0xBF58476D1CE4E5B9)
        mixed ^= mixed >> np.uint64(27)
        mixed *= np.uint64(0x94D049BB133111EB)
        mixed ^= mixed >> np.uint64(31)
        return mixed

    @classmethod
    def _groups(cls, packed: "np.ndarray", order: "np.ndarray", attempt: int = 0) -> "tuple[np.ndarray, np.ndarray] | None":
        """Одинаковые коды в обходе order: (первое вхождение каждого различного кода — номер в обходе, сколько раз он
        встретился) — то же, что np.unique по строкам с return_index и return_counts, но без копий всего набора: хеш
        строки — 128 бит (две суммы перемешанных 64-битных слов, у каждой свои нечётные множители; соль слов — своя у
        каждого места в строке), группы по хешу сверяются побайтно. Не сошлось — None. attempt — номер попытки: своё
        зерно соли и множителей."""
        count, width = int(order.shape[0]), int(packed.shape[1])
        words = -(-width // 8)
        rng = np.random.default_rng(20261125 + 7919 * int(attempt))
        factors = rng.integers(1, 2**63, size=(2, words), dtype=np.uint64) | np.uint64(1)
        salt = rng.integers(0, 2**63, size=words, dtype=np.uint64)
        keys = np.empty((count, 2), dtype=np.uint64)
        with np.errstate(over="ignore"):
            for start in range(0, count, cls.HASH_ROWS):
                rows = np.zeros((min(cls.HASH_ROWS, count - start), words * 8), dtype=np.uint8)
                rows[:, :width] = packed[order[start : start + cls.HASH_ROWS]]
                view = rows.view(np.uint64)
                view ^= salt
                mixed = cls._mix(view)
                for half in (0, 1):
                    keys[start : start + rows.shape[0], half] = (mixed * factors[half]).sum(axis=1, dtype=np.uint64)
                del rows, view, mixed
        _, first, inverse, counts = np.unique(keys, axis=0, return_index=True, return_inverse=True, return_counts=True)
        inverse = inverse.reshape(-1)
        # Сверка: каждый не первый член группы побайтно равен первому (иначе хеши двух разных кодов совпали).
        others = np.flatnonzero(first[inverse] != np.arange(count))
        for start in range(0, others.shape[0], cls.HASH_ROWS):
            part = others[start : start + cls.HASH_ROWS]
            if not np.array_equal(packed[order[part]], packed[order[first[inverse[part]]]]):
                return None
        return first.astype(np.int64), counts.astype(np.int64)

    def select(self, packed: "np.ndarray", code_length: int, order: "np.ndarray | None" = None) -> CodeDictionary:
        """Образцы для набора кодов (строки × ⌈L/8⌉ упакованных бит). order — порядок обхода строк (None — по порядку):
        коды берутся по нему кусками, без копии всего набора. Номера в словаре (rows) — места в этом обходе."""
        import torch

        if packed.ndim != 2 or packed.shape[0] == 0:
            raise ValueError("packed codes must be a non-empty matrix")
        if packed.dtype != np.uint8:
            packed = packed.astype(np.uint8)
        order = np.arange(packed.shape[0], dtype=np.int64) if order is None else np.asarray(order, dtype=np.int64)
        if order.shape[0] == 0:
            raise ValueError("packed codes must be a non-empty matrix")
        grouped = None
        for attempt in range(self.HASH_ATTEMPTS):
            grouped = self._groups(packed, order, attempt)
            if grouped is not None:
                break
            LOGGER.event(
                "dictionary.hash_collision",
                section=OUTSIDE_ARTICLE,
                data={"codes": int(order.shape[0]), "attempt": attempt,
                      "note": "у двух разных кодов совпал 128-битный хеш строки: следующая попытка — другое зерно хеша, "
                      "после последней — одинаковые коды сортировкой копии всего набора"},
            )
        if grouped is None:
            # Хеши разных кодов совпали — прежний путь: одинаковые строки по сортировке копии.
            copy = np.ascontiguousarray(packed[order])
            view = copy.view(np.dtype((np.void, copy.shape[1]))).reshape(-1)
            _, first, counts = np.unique(view, return_index=True, return_counts=True)
            del copy, view
        else:
            first, counts = grouped
        # От частых к редким; при равной частоте — по первому вхождению (детерминированно).
        ranking = np.lexsort((first, -counts))
        first = first[ranking]
        int8 = None  # коды int8 на видеокарте: (дополненная длина, блок int8 под умножение)
        device = torch.device("cuda" if self._use_gpu and torch.cuda.is_available() else "cpu")
        part_size = min(self.BANK_PART, self.max_size)
        banks: list["torch.Tensor"] = []  # части банка: (part_size, L) коды образцов
        banks_ones: list["torch.Tensor"] = []  # число единиц в кодах образцов той же части
        chosen: list[int] = []
        fresh = np.zeros(self.BLOCK, dtype=np.int64)
        for start in range(0, first.shape[0], self.BLOCK):
            if len(chosen) >= self.max_size:
                break
            rows = first[start : start + self.BLOCK]
            unpacked = torch.as_tensor(np.unpackbits(packed[order[rows]], axis=1, count=code_length), device=device)
            if int8 is None and start == 0:
                int8 = self._int8_ready(torch, device, unpacked, code_length)
            if int8:
                block = self._int8_rows(torch, unpacked, int8)
                ones = block.sum(dim=1, dtype=torch.int32).to(torch.float32)
            else:
                block = unpacked.to(torch.float32)
                ones = block.sum(dim=1)
            del unpacked
            # Наибольшее сходство кода блока с уже выбранными образцами — по частям банка.
            best = torch.zeros(block.shape[0], dtype=torch.float32, device=device)
            for part, (bank, bank_ones) in enumerate(zip(banks, banks_ones)):
                used = min(part_size, len(chosen) - part * part_size)
                cosine = self._common(torch, block, bank, used, int8) / torch.sqrt(
                    torch.clamp(ones[:, None] * bank_ones[None, :used], min=1.0)
                )
                best = torch.maximum(best, cosine.max(dim=1).values)
            free = (best < self.similarity_threshold).cpu().numpy()
            inner = (
                self._common(torch, block, block, int(block.shape[0]), int8)
                / torch.sqrt(torch.clamp(ones[:, None] * ones[None, :], min=1.0))
                >= self.similarity_threshold
            ).cpu().numpy()
            taken = 0
            for row in np.flatnonzero(free):
                if len(chosen) + taken >= self.max_size:
                    break
                if taken and inner[row, fresh[:taken]].any():
                    continue
                fresh[taken] = row
                taken += 1
            # Новые образцы блока дописываются в банк разом; часть банка заводится, когда прежняя заполнена.
            written = 0
            while written < taken:
                part, offset = divmod(len(chosen), part_size)
                if part == len(banks):
                    # Банк int8 — с длиной и числом строк, дополненными до кратных 8 (нулевые строки и биты общих бит не
                    # добавляют; в сравнение идут только занятые строки).
                    shape = (-(-part_size // 8) * 8, int8) if int8 else (part_size, code_length)
                    banks.append(torch.zeros(shape, dtype=torch.int8 if int8 else torch.float32, device=device))
                    banks_ones.append(torch.ones(part_size, dtype=torch.float32, device=device))
                step = min(taken - written, part_size - offset)
                picked = torch.as_tensor(fresh[written : written + step], device=device)
                banks[part][offset : offset + step] = block[picked]
                banks_ones[part][offset : offset + step] = ones[picked]
                chosen.extend(int(rows[row]) for row in fresh[written : written + step])
                written += step
        del banks, banks_ones
        selected = np.asarray(chosen, dtype=np.int64)
        dictionary = CodeDictionary(
            packed=np.ascontiguousarray(packed[order[selected]]), rows=selected, code_length=int(code_length),
            distinct=int(first.shape[0]),
        )
        LOGGER.event(
            "dictionary.select",
            section=OUTSIDE_ARTICLE,
            data={
                "codes": int(order.shape[0]),
                "int8": bool(int8),
                "distinct": dictionary.distinct,
                "samples": dictionary.count,
                "compression": float(packed.shape[0]) / max(1, dictionary.count),
                "similarity_threshold": self.similarity_threshold,
                "max_size": self.max_size,
                "capped": dictionary.count >= self.max_size,
                "note": "слияние к образцу: код присоединяется к образцу со сходством не ниже порога, иначе сам "
                "становится образцом; обход от частых кодов к редким",
            },
        )
        return dictionary


def _leader_int8_helpers() -> None:
    """Помощники LeaderDictionary для кодов int8 (вынесены, чтобы select читался как прежде)."""

    def int8_ready(self, torch, device, unpacked, code_length: int):  # noqa: ANN001, ANN202
        """Длина кода, дополненная до кратной 8, если коды int8 можно взять: видеокарта, torch._int_mm, LAYOUT_INT8_CODES
        и совпадение числа общих бит первого блока с произведением float32; иначе 0 (прежний счёт float32)."""
        from damp import settings

        if device.type != "cuda" or not hasattr(torch, "_int_mm") or not bool(settings.LAYOUT_INT8_CODES):
            return 0
        width = -(-int(code_length) // 8) * 8
        try:
            block = LeaderDictionary._int8_rows(torch, unpacked, width)
            found = LeaderDictionary._common(torch, block, block, int(block.shape[0]), width)
            plain = unpacked.to(torch.float32)
            exact = plain @ plain.T
        except Exception:  # noqa: BLE001 — нет целочисленного произведения на этом устройстве
            return 0
        return width if bool(torch.equal(found, exact)) else 0

    def int8_rows(torch, unpacked, width: int):  # noqa: ANN001, ANN202
        """Коды блока 0 / 1 → int8 с длиной width (дополнение нулями)."""
        block = torch.zeros((int(unpacked.shape[0]), int(width)), dtype=torch.int8, device=unpacked.device)
        block[:, : unpacked.shape[1]] = unpacked.to(torch.int8)
        return block

    def common(torch, block, bank, used: int, int8):  # noqa: ANN001, ANN202
        """Число общих бит строк блока с первыми used строками банка, float32 (строки × used): float32 — произведение
        матриц, как прежде; int8 — torch._int_mm (int8 × int8 → int32; левая — не меньше 17 строк, правая — кратна 8
        по строкам): то же целое число."""
        if not int8:
            return block @ bank[:used].T
        taken = int(block.shape[0])
        rows = max(taken, 17)
        if rows != taken:
            block = torch.cat((block, block.new_zeros((rows - taken, block.shape[1]))))
        right = -(-int(used) // 8) * 8
        if right > bank.shape[0]:
            bank = torch.cat((bank, bank.new_zeros((right - bank.shape[0], bank.shape[1]))))
        return torch._int_mm(block, bank[:right].T)[:taken, :used].to(torch.float32)

    LeaderDictionary._int8_ready = int8_ready
    LeaderDictionary._int8_rows = staticmethod(int8_rows)
    LeaderDictionary._common = staticmethod(common)


_leader_int8_helpers()


class PairwiseSimilarity:
    """Квантили попарных сходств кодов (§2.2.4): по ним задаётся порог λ раскладки этих кодов (§7.1.7)."""

    _MEMO: dict[tuple, tuple[list[float], int, float]] = {}  # на процесс: (значения, строк выборки, среднее)

    def __init__(self, *, sample: int = 2000, seed: int = 0) -> None:
        self._sample = int(sample)
        self._seed = int(seed)

    def quantiles(self, bits: "np.ndarray", quantiles: Sequence[float], *, name: str) -> list[float]:
        """Значения квантилей косинусного сходства пар кодов (по выборке строк)."""
        bits = np.asarray(bits)
        if bits.shape[0] < 2:
            raise ValueError("at least two codes are required")
        quantiles = tuple(quantiles)
        # Раскладка и детекторы уровня (и повторные раскладки верха) спрашивают одно и то же — по тем же битам, зерну,
        # выборке и квантилям: `sample @ sample.T` считается один раз на процесс, ответ тот же до бита.
        content = np.ascontiguousarray(bits)
        memo_key = (
            hashlib.blake2b(content.tobytes(), digest_size=20).digest(),
            str(content.dtype),
            content.shape,
            self._seed,
            self._sample,
            tuple(float(q) for q in quantiles),
        )
        known = self._MEMO.get(memo_key)
        if known is None:
            rng = np.random.default_rng(self._seed)
            rows = rng.choice(bits.shape[0], size=min(self._sample, bits.shape[0]), replace=False)
            sample = bits[np.sort(rows)].astype(np.float32)
            ones = sample.sum(axis=1)
            cosine = (sample @ sample.T) / np.sqrt(np.maximum(ones[:, None] * ones[None, :], 1.0))
            pairs = cosine[np.triu_indices(sample.shape[0], k=1)]
            values = [float(np.quantile(pairs, q)) for q in quantiles]
            known = (values, int(sample.shape[0]), float(pairs.mean()))
            if len(self._MEMO) >= 64:
                self._MEMO.pop(next(iter(self._MEMO)))
            self._MEMO[memo_key] = known
        values, sampled, mean = known
        values = list(values)
        LOGGER.event(
            "codes.pairwise_similarity",
            section=SIMILARITY_MEASURES,
            data={
                "name": name,
                "codes": int(bits.shape[0]),
                "sampled": sampled,
                "mean": mean,
                "quantiles": {f"{q:g}": value for q, value in zip(quantiles, values)},
            },
        )
        return values
