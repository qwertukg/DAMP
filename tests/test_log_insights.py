"""Инварианты анализа логов: кодирование строк по §7.1.3–§7.1.4, числа по §4.2.1 и конвейер на синтетическом логе."""
from __future__ import annotations

import itertools
import math
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("RERUN", "off")

import numpy as np  # noqa: E402

from damp.encoding.bitarray import BitArray  # noqa: E402
from damp.encoding.log_encoder import (  # noqa: E402
    FragmentDictionary,
    LineFragmenter,
    LogLine,
    LogLineEncoder,
    LogLineReader,
    NumericValueEncoder,
    TokenAlphabet,
)
from damp.logging import LOGGER  # noqa: E402
import log_insights  # noqa: E402
import main  # noqa: E402

LOGGER.configure_intervals({}, default_interval=10**9)


def _reader(line_words_max: int = main.LOG_LINE_WORDS_MAX) -> LogLineReader:
    return LogLineReader(
        word_min=main.LOG_WORD_MIN,
        word_max=main.LOG_WORD_MAX,
        line_words_max=line_words_max,
        slot_words=main.LOG_NUMBER_SLOT_WORDS,
    )


def _encoder(lines: list[tuple[str, ...]], *, saturation: dict | None = None) -> LogLineEncoder:
    dictionary = FragmentDictionary(lines, fragment_words_max=main.LOG_FRAGMENT_WORDS_MAX)
    return LogLineEncoder(
        TokenAlphabet(code_length=main.LOG_CODE_BITS, token_bits=main.LOG_TOKEN_BITS, seed=main.LOG_SEED),
        LineFragmenter(dictionary, fragments_max=main.LOG_FRAGMENTS_MAX),
        fragment_saturation=saturation or main.LOG_FRAGMENT_SATURATION,
    )


def _numeric() -> NumericValueEncoder:
    return NumericValueEncoder(
        unit=main.LOG_NUMERIC_UNIT,
        decades=main.LOG_NUMERIC_DECADES,
        layers=main.LOG_NUMERIC_LAYERS,
        widest=main.LOG_NUMERIC_WIDEST,
        narrowest=main.LOG_NUMERIC_NARROWEST,
        overlap=main.LOG_NUMERIC_OVERLAP,
        code_length=main.LOG_NUMERIC_CODE_BITS,
        seed=main.LOG_SEED,
    )


def _cosine(a: BitArray, b: BitArray) -> float:
    return a.common(b) / math.sqrt(a.count() * b.count())


class ReaderTests(unittest.TestCase):
    """§7.1.3: строка делится на слова по небуквенным символам, слова — в нижний регистр, длина ограничена."""

    def test_words_are_split_lowercased_and_filtered_by_length(self) -> None:
        words, _ = _reader().parse("Layout.Run_progress modeLong a x verylongwordthatexceedstwentychars ok")
        self.assertEqual(words, ("layout", "run", "progress", "mode", "long", "ok"))

    def test_runs_with_digits_are_values_not_words(self) -> None:
        words, _ = _reader().parse("request id=a0d8f41fba7a sha256 utf8 done")
        self.assertEqual(words, ("request", "id", "done"))

    def test_numbers_get_slots_and_timestamps_and_dotted_numbers_are_skipped(self) -> None:
        _, numbers = _reader().parse("2026-11-12T07:12:05.754669Z energy=0.82 swap_ratio=1e-3 [5.9.1] at 10.0.0.5")
        self.assertEqual(numbers, (("energy", 0.82), ("swap ratio", 0.001)))

    def test_consecutive_numbers_share_a_continuation_slot(self) -> None:
        _, numbers = _reader().parse("counts=[2, 3, 5]")
        self.assertEqual([slot for slot, _ in numbers], ["counts", "counts …", "counts …"])

    def test_line_is_truncated_to_words_max(self) -> None:
        words, _ = _reader(line_words_max=3).parse("alpha beta gamma delta epsilon")
        self.assertEqual(words, ("alpha", "beta", "gamma"))


class DictionaryTests(unittest.TestCase):
    """§7.1.3, §7.1.4: вес фрагмента — способность составлять строки; разбиения покрывают строку без пропусков."""

    def test_fragment_weight_is_sum_of_length_shares_over_distinct_lines(self) -> None:
        lines = [("a", "b", "c"), ("a", "b"), ("a", "b")]
        dictionary = FragmentDictionary(lines, fragment_words_max=8)
        self.assertAlmostEqual(dictionary.weight(("a", "b")), 2 / 3 + 2 / 2)
        self.assertAlmostEqual(dictionary.weight(("a",)), 1 / 3 + 1 / 2)
        self.assertAlmostEqual(dictionary.weight(("c",)), 1 / 3)
        self.assertEqual(dictionary.weight(("z",)), 0.0)

    def test_fragmentations_tile_the_line_and_have_the_best_weight_for_each_count(self) -> None:
        lines = [
            ("db", "query", "start", "table", "users"),
            ("db", "query", "done", "table", "users"),
            ("db", "pool", "acquire"),
            ("http", "request", "start"),
        ]
        dictionary = FragmentDictionary(lines, fragment_words_max=8)
        fragmenter = LineFragmenter(dictionary, fragments_max=4)
        words = lines[0]
        found = fragmenter.fragmentations(words)
        self.assertEqual([len(item) for item in found], [1, 2, 3, 4])
        for fragments in found:
            self.assertEqual(tuple(itertools.chain.from_iterable(fragments)), words)
            parts = len(fragments)
            best = max(
                sum(dictionary.weight(words[a:b]) for a, b in zip((0, *cuts), (*cuts, len(words))))
                for cuts in itertools.combinations(range(1, len(words)), parts - 1)
            )
            self.assertAlmostEqual(sum(dictionary.weight(fragment) for fragment in fragments), best)


class LineCodeTests(unittest.TestCase):
    """§7.1.4.1, §7.1.4.4, §7.1.4.5, §3.2: алфавит, бюджеты фрагментов, акценты, вложенность кодов."""

    def test_alphabet_is_deterministic_and_depends_on_position_and_word(self) -> None:
        first = TokenAlphabet(code_length=256, token_bits=40, seed=0)
        second = TokenAlphabet(code_length=256, token_bits=40, seed=0)
        bits = first.bits(0, "layout")
        self.assertEqual(len(set(bits.tolist())), 40)
        self.assertEqual(bits.tolist(), second.bits(0, "layout").tolist())
        self.assertNotEqual(bits.tolist(), first.bits(1, "layout").tolist())
        self.assertNotEqual(bits.tolist(), first.bits(0, "encoder").tolist())

    def test_smaller_budget_gives_a_subset_of_the_larger_budget_code(self) -> None:
        encoder = _encoder([("alpha", "beta")])
        small = set(encoder._fragment_bits(("alpha", "beta"), 4, "head"))
        large = set(encoder._fragment_bits(("alpha", "beta"), 12, "head"))
        self.assertEqual(len(small), 4)
        self.assertEqual(len(large), 12)
        self.assertTrue(small <= large)

    def test_every_word_of_a_long_fragment_gives_at_least_one_bit(self) -> None:
        fragment = tuple(f"word{chr(97 + i)}" for i in range(10))
        encoder = _encoder([fragment])
        self.assertEqual(len(encoder._fragment_bits(fragment, 3, "head")), 10)

    def test_fragmentation_code_is_the_union_of_fragment_codes(self) -> None:
        encoder = _encoder([("alpha", "beta", "gamma")])
        fragments = (("alpha",), ("beta", "gamma"))
        code = encoder.encode_fragmentation(fragments, "head")
        head, tail = main.LOG_FRAGMENT_SATURATION[2]
        expected = set(encoder._fragment_bits(fragments[0], head, "head")) | set(
            encoder._fragment_bits(fragments[1], tail, "head")
        )
        self.assertEqual(set(code.active_indices()), expected)

    def test_tail_accent_reverses_fragment_budgets(self) -> None:
        encoder = _encoder([("alpha", "beta")], saturation={1: (15,), 2: (15, 5)})
        fragments = (("alpha",), ("beta",))
        alpha = set(TokenAlphabet(code_length=256, token_bits=40, seed=0).bits(0, "alpha").tolist())
        head = set(encoder.encode_fragmentation(fragments, "head").active_indices())
        tail = set(encoder.encode_fragmentation(fragments, "tail").active_indices())
        self.assertGreater(len(head & alpha), len(tail & alpha))
        self.assertNotEqual(head, tail)

    def test_shared_fragment_gives_shared_bits_at_any_position_in_the_line(self) -> None:
        encoder = _encoder([("alpha", "beta")])
        shared = ("connection", "timed", "out")
        left = encoder.encode_fragmentation((("db",), shared), "tail")
        right = encoder.encode_fragmentation((("http", "client"), shared), "tail")
        budget = main.LOG_FRAGMENT_SATURATION[2][0]
        self.assertGreaterEqual(left.common(right), len(set(encoder._fragment_bits(shared, budget, "tail"))))

    def test_similar_lines_are_closer_than_unrelated_ones_and_codes_stay_sparse(self) -> None:
        lines = [
            ("layout", "run", "progress", "step", "mode", "long", "accepted", "swaps"),
            ("layout", "run", "progress", "step", "mode", "short", "accepted", "swaps"),
            ("decoder", "evaluation", "summary", "samples", "accuracy", "errors"),
            tuple(f"word{chr(97 + i % 26)}{chr(97 + i // 26)}" for i in range(main.LOG_LINE_WORDS_MAX)),
        ]
        encoder = _encoder(lines)
        codes = [encoder.encode(words) for words in lines]

        def best(a: int, b: int) -> float:
            return max(_cosine(x.code, y.code) for x in codes[a] for y in codes[b])

        self.assertGreater(best(0, 1), 0.8)
        self.assertLess(best(0, 2), 0.4)
        for line_codes in codes:
            self.assertLessEqual(len(line_codes), 2 * main.LOG_FRAGMENTS_MAX)
            for item in line_codes:
                self.assertLessEqual(item.code.count() / main.LOG_CODE_BITS, 0.25)


class NumericTests(unittest.TestCase):
    """§4.1, §4.2.1: близкие числа получают больше общих бит, чем далёкие."""

    def test_common_bits_decrease_with_distance_on_the_log_scale(self) -> None:
        encoder = _numeric()
        base = set(encoder.bits(1.0))
        near = len(base & set(encoder.bits(1.2)))
        far = len(base & set(encoder.bits(100.0)))
        very_far = len(base & set(encoder.bits(1e9)))
        self.assertGreater(near, far)
        self.assertGreaterEqual(far, very_far)
        self.assertLess(len(base & set(encoder.bits(-1.0))), near)

    def test_new_value_is_reported_and_known_value_is_not(self) -> None:
        def line(source: str, number: int, value: float) -> LogLine:
            return LogLine(source=source, number=number, text=f"job done elapsed={value}",
                           words=("job", "done", "elapsed"), numbers=(("elapsed", value),))

        reference = [line("ref.log", i + 1, 0.1 + 0.05 * (i % 17)) for i in range(40)]
        target = [line("new.log", 1, 0.3), line("new.log", 2, 5000.0)]
        corpus = log_insights.LogCorpus(reference, target)

        class _Types:
            types = [type("T", (), {"label": "job done elapsed"})()]

            @staticmethod
            def line_types() -> np.ndarray:
                return np.zeros(len(corpus.lines), dtype=np.int64)

        found = log_insights.NumericNovelty(corpus, _Types(), _numeric()).find()
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["value"], 5000.0)
        self.assertEqual(found[0]["where"], "new.log:2")
        self.assertLess(found[0]["known_ratio"], main.LOG_NUMERIC_KNOWN_RATIO)


class CodeMemoryTests(unittest.TestCase):
    """§2.2.4.1, §2.2.5: дискретный косинус и нечёткий поиск по памяти кодов."""

    def test_search_returns_discrete_cosine_and_zero_for_empty_code(self) -> None:
        codes = [BitArray.from_indices(16, [0, 1, 2, 3]), BitArray.from_indices(16, [2, 3, 4, 5]), BitArray(16)]
        memory = log_insights.CodeMemory(codes)
        found = memory.search(codes[0])
        self.assertAlmostEqual(float(found[0]), 1.0)
        self.assertAlmostEqual(float(found[1]), 0.5)
        self.assertEqual(float(found[2]), 0.0)
        self.assertTrue(np.all(memory.search(codes[2]) == 0.0))


class PipelineTests(unittest.TestCase):
    """Путь «строка → коды → раскладка → детекторы → эмбеддинг» (§7.1, рис. 23) на синтетическом логе."""

    FAMILIES = {
        "db": ["query start table users", "query done table users elapsed", "pool acquire connections",
               "pool release connections", "transaction begin isolation serializable",
               "transaction commit isolation serializable", "index rebuild table orders", "vacuum table orders pages"],
        "http": ["request begin method get path users", "request end method get status bytes",
                 "client connect host example port", "client close host example port",
                 "route match handler profile", "route miss handler fallback", "tls handshake cipher strong",
                 "tls renegotiate cipher strong"],
        "cache": ["hit key profile ttl", "miss key profile ttl", "evict entries reason memory",
                  "warm entries reason startup", "shard rebalance nodes", "shard join nodes",
                  "snapshot write bytes", "snapshot read bytes"],
    }

    KEYS = {"db": "rows", "http": "bytes", "cache": "entries"}

    @classmethod
    def _run_lines(cls, seed: int, rounds: int) -> list[str]:
        rng = np.random.default_rng(seed)
        lines = []
        for _ in range(rounds):
            for family, events in cls.FAMILIES.items():
                for event in events:
                    lines.append(f"{family}.{event.replace(' ', '.')} {cls.KEYS[family]}={int(rng.integers(1, 90))}")
        return lines

    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        root = Path(cls._directory.name)
        reference = []
        for index in range(3):
            path = root / f"reference{index}.log"
            path.write_text("\n".join(cls._run_lines(index, rounds=5)) + "\n", encoding="utf-8")
            reference.append(str(path))
        target_lines = cls._run_lines(7, rounds=5)
        target_lines.insert(60, "kernel panic: unable to mount root filesystem on device")
        target_lines.insert(61, "db.query.done.table.users.elapsed rows=900000000")
        target = root / "target.log"
        target.write_text("\n".join(target_lines) + "\n", encoding="utf-8")
        cls.insights = log_insights.LogInsightsExperiment(reference=reference, target=[str(target)]).run(save=False)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def test_only_the_unseen_line_is_new(self) -> None:
        new_lines = self.insights["new_lines"]
        self.assertEqual(len(new_lines), 1)
        self.assertIn("kernel panic", new_lines[0]["text"])
        self.assertNotEqual(new_lines[0]["status"], log_insights.STATUS_EXACT)
        self.assertTrue(new_lines[0]["problem"])

    def test_incident_contains_the_unseen_line_and_has_similar_reference_windows(self) -> None:
        incidents = self.insights["incidents"]
        self.assertEqual(len(incidents), 1)
        self.assertTrue(incidents[0]["first_signal"].endswith(":61"))
        self.assertTrue(any("kernel panic" in signal["text"] for signal in incidents[0]["signals"]))
        self.assertTrue(incidents[0]["similar"])
        self.assertTrue(incidents[0]["before"])

    def test_out_of_range_number_is_reported(self) -> None:
        values = [item["value"] for item in self.insights["new_numbers"]]
        self.assertIn(900000000.0, values)

    def test_lower_layers_separate_families_and_upper_layers_separate_types(self) -> None:
        summary = self.insights["summary"]
        self.assertGreaterEqual(summary["families"], len(self.FAMILIES))
        self.assertGreater(summary["types"], summary["families"])
        self.assertLessEqual(summary["types"], summary["stimuli"])

    def test_embeddings_follow_primary_code_similarity(self) -> None:
        quality = self.insights["embedding_quality"]
        self.assertGreater(quality["correlation"], 0.5)
        self.assertGreater(quality["embedding_similarity_of_similar"], quality["embedding_similarity_of_others"])

    def test_event_order_gives_rules(self) -> None:
        self.assertTrue(self.insights["rules"])
        for rule in self.insights["rules"]:
            self.assertGreaterEqual(rule["confidence"], main.LOG_RELATION_MIN_CONFIDENCE)
            self.assertGreaterEqual(rule["lift"], main.LOG_RELATION_MIN_LIFT)


if __name__ == "__main__":
    unittest.main()
