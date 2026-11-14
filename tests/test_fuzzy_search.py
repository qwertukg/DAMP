"""Нечёткий поиск по базе эмбеддингов (§2.2.5): векторный расчёт сходства даёт те же значения, что поэлементный."""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("RERUN", "off")

import numpy as np  # noqa: E402

from damp.decode import DecoderConfig, EmbeddingDatabase, LabeledEmbedding  # noqa: E402
from damp.encoding.bitarray import BitArray  # noqa: E402
from damp.logging import LOGGER  # noqa: E402

LOGGER.configure_intervals({}, default_interval=10**9)


class FuzzySearchSimilarityTests(unittest.TestCase):
    def _database(self, similarity: str, length: int, seed: int) -> tuple[EmbeddingDatabase, np.random.Generator]:
        rng = np.random.default_rng(seed)
        items = [
            LabeledEmbedding(
                label=int(rng.integers(0, 10)),
                embedding=BitArray.from_indices(
                    length, rng.choice(length, size=int(rng.integers(1, length // 3)), replace=False).tolist()
                ),
            )
            for _ in range(120)
        ]
        config = DecoderConfig(
            similarity=similarity,
            temperature=0.05,
            min_confidence=0.05,
            expected_classes=tuple(range(10)),
            knn_top_k=40,
            top_k_per_class=5,
        )
        return EmbeddingDatabase(items, config), rng

    def test_packed_similarities_equal_bitwise_ones(self) -> None:
        # Длины, не кратные 8, проверяют хвост последнего байта упакованного кода.
        for similarity in ("jaccard2", "cosine", "cosine2"):
            for length in (61, 256, 1001):
                database, rng = self._database(similarity, length, seed=length)
                for _ in range(10):
                    query = BitArray.from_indices(
                        length, rng.choice(length, size=int(rng.integers(1, length // 3)), replace=False).tolist()
                    )
                    ones = query.count()
                    packed = database._similarities(query, ones)
                    bitwise = np.array(
                        [
                            database._similarity_value(query, ones, stored, stored_ones)
                            for _, stored, stored_ones in database._items
                        ]
                    )
                    np.testing.assert_array_equal(packed, bitwise)

    def test_stored_embedding_is_its_own_nearest_neighbour(self) -> None:
        database, _ = self._database("jaccard2", 256, seed=7)
        label, stored, _ = database._items[17]
        result = database.predict(stored, temperature=0.001)
        self.assertEqual(result.predicted_class, label)

    def test_query_of_another_length_is_rejected(self) -> None:
        database, _ = self._database("jaccard2", 256, seed=9)
        with self.assertRaises(ValueError):
            database.predict(BitArray.from_indices(128, [1, 5]))


if __name__ == "__main__":
    unittest.main()
