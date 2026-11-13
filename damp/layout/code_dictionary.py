"""Словарь образцов: слияние похожих кодов к образцу перед раскладкой (расширение статьи).

Кодов кусков сотни тысяч, и почти все различны, но большинство отличается от уже встречавшихся на несколько
бит. Раскладывать их все незачем: код присоединяется к самому раннему образцу, с которым его сходство
(§2.2.4) не ниже порога, иначе сам становится образцом. Коды обходятся от частых к редким, поэтому
образцами становятся самые частые из различимых кодов.
"""
from __future__ import annotations

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

    def __init__(self, *, similarity_threshold: float, max_size: int, use_gpu: bool = True) -> None:
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")
        if max_size < 1:
            raise ValueError("max_size must be positive")
        self.similarity_threshold = float(similarity_threshold)
        self.max_size = int(max_size)
        self._use_gpu = bool(use_gpu)

    def select(self, packed: "np.ndarray", code_length: int) -> CodeDictionary:
        """Образцы для набора кодов (строки × ⌈L/8⌉ упакованных бит)."""
        import torch

        packed = np.ascontiguousarray(packed, dtype=np.uint8)
        if packed.ndim != 2 or packed.shape[0] == 0:
            raise ValueError("packed codes must be a non-empty matrix")
        view = packed.view(np.dtype((np.void, packed.shape[1]))).reshape(-1)
        _, first, counts = np.unique(view, return_index=True, return_counts=True)
        # От частых к редким; при равной частоте — по первому вхождению (детерминированно).
        order = np.lexsort((first, -counts))
        first = first[order]
        device = torch.device("cuda" if self._use_gpu and torch.cuda.is_available() else "cpu")
        bank = torch.zeros((self.max_size, code_length), dtype=torch.float32, device=device)
        bank_ones = torch.ones(self.max_size, dtype=torch.float32, device=device)
        chosen: list[int] = []
        for start in range(0, first.shape[0], self.BLOCK):
            if len(chosen) >= self.max_size:
                break
            rows = first[start : start + self.BLOCK]
            block = torch.as_tensor(np.unpackbits(packed[rows], axis=1, count=code_length), device=device).to(torch.float32)
            ones = block.sum(dim=1)
            if chosen:
                cosine = (block @ bank[: len(chosen)].T) / torch.sqrt(
                    torch.clamp(ones[:, None] * bank_ones[None, : len(chosen)], min=1.0)
                )
                free = (cosine.max(dim=1).values < self.similarity_threshold).cpu().numpy()
            else:
                free = np.ones(block.shape[0], dtype=bool)
            inner = (
                (block @ block.T) / torch.sqrt(torch.clamp(ones[:, None] * ones[None, :], min=1.0))
                >= self.similarity_threshold
            ).cpu().numpy()
            fresh: list[int] = []
            for row in np.flatnonzero(free):
                if len(chosen) >= self.max_size:
                    break
                if any(inner[row, other] for other in fresh):
                    continue
                fresh.append(int(row))
                bank[len(chosen)] = block[row]
                bank_ones[len(chosen)] = ones[row]
                chosen.append(int(rows[row]))
        selected = np.asarray(chosen, dtype=np.int64)
        dictionary = CodeDictionary(
            packed=packed[selected], rows=selected, code_length=int(code_length), distinct=int(first.shape[0])
        )
        LOGGER.event(
            "dictionary.select",
            section=OUTSIDE_ARTICLE,
            data={
                "codes": int(packed.shape[0]),
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


class PairwiseSimilarity:
    """Квантили попарных сходств кодов (§2.2.4): по ним задаётся порог λ раскладки этих кодов (§7.1.7)."""

    def __init__(self, *, sample: int = 2000, seed: int = 0) -> None:
        self._sample = int(sample)
        self._seed = int(seed)

    def quantiles(self, bits: "np.ndarray", quantiles: Sequence[float], *, name: str) -> list[float]:
        """Значения квантилей косинусного сходства пар кодов (по выборке строк)."""
        bits = np.asarray(bits)
        if bits.shape[0] < 2:
            raise ValueError("at least two codes are required")
        rng = np.random.default_rng(self._seed)
        rows = rng.choice(bits.shape[0], size=min(self._sample, bits.shape[0]), replace=False)
        sample = bits[np.sort(rows)].astype(np.float32)
        ones = sample.sum(axis=1)
        cosine = (sample @ sample.T) / np.sqrt(np.maximum(ones[:, None] * ones[None, :], 1.0))
        pairs = cosine[np.triu_indices(sample.shape[0], k=1)]
        values = [float(np.quantile(pairs, q)) for q in quantiles]
        LOGGER.event(
            "codes.pairwise_similarity",
            section=SIMILARITY_MEASURES,
            data={
                "name": name,
                "codes": int(bits.shape[0]),
                "sampled": int(sample.shape[0]),
                "mean": float(pairs.mean()),
                "quantiles": {f"{q:g}": value for q, value in zip(quantiles, values)},
            },
        )
        return values
