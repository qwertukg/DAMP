"""Инварианты анализа логов: кодирование строк по §7.1.3–§7.1.4, числа по §4.2.1 и конвейер на синтетическом логе."""
from __future__ import annotations

import gzip
import itertools
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("RERUN", "off")

import numpy as np  # noqa: E402

from damp.encoding.bitarray import BitArray  # noqa: E402
from damp.encoding.log_encoder import (  # noqa: E402
    VALUE,
    BackgroundWords,
    FragmentDictionary,
    LineFragmenter,
    LogLine,
    LogLineEncoder,
    LogLineReader,
    NumericValueEncoder,
    TokenAlphabet,
    VariableSlots,
    limit_words,
    truncated,
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


def _words(text: str, **options) -> tuple[str, ...]:
    """Стимул строки: слова по порядку, на месте значений — заполнитель."""
    units, numbers = _reader(**options).parse(text)
    return LogLine(source="test", number=1, text=text, units=units, numbers=numbers).words


def _units(*words: str) -> tuple[tuple[str, ...], ...]:
    """Единицы строки из отдельных слов; заполнитель значения — пустая единица."""
    return tuple(() if word == VALUE else (word,) for word in words)


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
        words = _words("Layout.Run_progress modeLong verylongwordthatexceedstwentychars ok")
        self.assertEqual(words, ("layout", "run", "progress", "mode", "long", VALUE, "ok"))

    def test_tokens_without_words_become_one_value_placeholder(self) -> None:
        self.assertEqual(_words("request id=a0d8f41fba7a sha256 utf8 done"), ("request", "id", VALUE, "done"))
        self.assertEqual(_words("42 took 3 5 ms 7"), ("took", VALUE, "ms"))

    def test_numbers_get_slots_and_timestamps_and_dotted_numbers_are_skipped(self) -> None:
        units, numbers = _reader().parse("2026-11-12T07:12:05.754669Z energy=0.82 swap_ratio=1e-3 [5.9.1] at 10.0.0.5")
        self.assertEqual(numbers, (("energy", 0.82), ("swap ratio", 0.001)))
        self.assertEqual(units, (("energy",), (), ("swap", "ratio"), (), ("at",)))

    def test_number_with_a_short_unit_is_a_number(self) -> None:
        _, numbers = _reader().parse("request took=35ms load=12% size 4.5MB")
        self.assertEqual(numbers, (("request took", 35.0), ("load", 12.0), ("size", 4.5)))

    def test_consecutive_numbers_get_numbered_slots(self) -> None:
        _, numbers = _reader().parse("GET /index HTTP 200 2326 0.003")
        self.assertEqual([slot for slot, _ in numbers], ["index http", "index http #1", "index http #2"])

    def test_timestamps_of_common_formats_give_no_words(self) -> None:
        for stamp in (
            "Dec 10 06:55:46",
            "[Sun Dec 04 04:47:44 2005]",
            "2015-07-29 17:41:44,747",
            "[10/Oct/2000:13:55:36 -0700]",
            "12 November 2026 14:05:11",
        ):
            self.assertEqual(_words(f"{stamp} session opened"), ("session", "opened"), stamp)

    def test_letters_of_any_alphabet_are_words(self) -> None:
        self.assertEqual(_words("Қате: дерекқорға қосылу мүмкін емес"), ("қате", "дерекқорға", "қосылу", "мүмкін", "емес"))
        self.assertEqual(_words("HTTPServer қайта іске қосылды"), ("http", "server", "қайта", "іске", "қосылды"))

    def test_line_is_truncated_to_words_max(self) -> None:
        self.assertEqual(_words("alpha beta gamma delta epsilon", line_words_max=3), ("alpha", "beta", "gamma"))

    def test_json_record_is_unfolded_into_keys_and_values(self) -> None:
        text = '{"@timestamp":"2026-11-12T16:51:30.898+05:00","message":"Kafka version: 3.9.1","level":"INFO","level_value":20000}'
        units, numbers, shape = _reader().parse_record(text)
        words = LogLine(source="test", number=1, text=text, units=units, numbers=numbers).words
        self.assertEqual(words, ("timestamp", "message", "kafka", "version", VALUE, "level", "info", "level", "value"))
        self.assertEqual(numbers, (("level value", 20000.0),))
        # Формы записи: «json» и набор ключей объекта — записи с одним набором ключей устроены одинаково.
        self.assertEqual(shape, ("json", "json:@timestamp|level|level_value|message"))
        self.assertEqual(_reader().parse_record("plain text line")[2], ())

    def test_broken_json_record_stays_text_and_keeps_its_keys(self) -> None:
        text = r'{"message":"Config values: \n\tallow.auto = true\n\tretries = 5","level":"INF'
        units, numbers, shape = _reader().parse_record(text)
        words = LogLine(source="test", number=1, text=text, units=units, numbers=numbers).words
        # Экранированные перевод строки и табуляция не склеиваются со словами («tallow», «nretries»).
        self.assertEqual(words, ("message", "config", "values", "allow", "auto", "true", "retries", VALUE, "level", "inf"))
        self.assertEqual(shape, ("json", "json:level|message"))

    def test_identifiers_are_values_not_words(self) -> None:
        self.assertEqual(
            _words("consumer bd869f0e-eefd-4824-a608-79d8dd24e822 joined group"), ("consumer", VALUE, "joined", "group")
        )
        self.assertEqual(_words("device aa:bb:cc:dd:ee:ff added to queue"), ("device", VALUE, "added", "to", "queue"))
        self.assertEqual(
            _words("commit deadbeef12 by cafe-0a1b-22cc feed updated"), ("commit", VALUE, "by", VALUE, "feed", "updated")
        )

    def test_terminal_colour_codes_are_removed(self) -> None:
        self.assertEqual(
            _words("\x1b[32m17:38:36.949\x1b[0;39m \x1b[34mINFO \x1b[0;39m: session opened"), ("info", "session", "opened")
        )
        self.assertEqual(_words("\x1b[1;31mERROR\x1b[0m disk full"), ("error", "disk", "full"))

    def test_time_zone_offset_belongs_to_the_timestamp(self) -> None:
        for stamp in ("2026-11-12T09:35:59.298+0500", "2026-11-12T09:35:59.298+05:00", "2026-11-12 09:35:59Z"):
            units, numbers = _reader().parse(f"{stamp} session opened")
            self.assertEqual((units, numbers), ((("session",), ("opened",)), ()), stamp)


class BackgroundAndVariableTests(unittest.TestCase):
    """Заголовок и переменные места строки находятся по статистике эталона, без знания формата (вне статьи)."""

    @staticmethod
    def _line(*words: str) -> LogLine:
        return LogLine(source="test", number=1, text=" ".join(words), units=_units(*words), numbers=())

    def test_words_present_in_almost_every_line_are_dropped(self) -> None:
        lines = [self._line("web", "sshd", "session", "opened")] * 30 + [self._line("web", "sshd", "disk", "full")] * 30
        background = BackgroundWords(lines, share=0.9, min_lines=50)
        self.assertEqual(background.words, frozenset({"web", "sshd"}))
        self.assertEqual(background.clean(lines)[0].words, ("session", "opened"))
        # Строка из одних фоновых слов остаётся как есть.
        self.assertEqual(background.strip(_units("web", "sshd")), _units("web", "sshd"))

    def test_background_is_not_searched_in_a_small_corpus(self) -> None:
        lines = [self._line("web", "session", "opened")] * 10
        self.assertEqual(BackgroundWords(lines, share=0.9, min_lines=50).words, frozenset())

    def test_slot_with_many_values_in_the_same_context_is_masked(self) -> None:
        lines = [_units("invalid", "user", name, "from") for name in ("admin", "guest", "oracle")]
        lines += [_units("session", "opened"), _units("session", "closed"), _units("session", "expired")]
        variables = VariableSlots(lines, min_values=3, min_context=2)
        self.assertEqual(variables.mask(lines[0]), _units("invalid", "user", VALUE, "from"))
        # Новое значение на известном переменном месте тоже заменяется.
        self.assertEqual(variables.mask(_units("invalid", "user", "newbie", "from")), _units("invalid", "user", VALUE, "from"))
        # В короткой строке окружения мало: слова остаются.
        self.assertEqual(variables.mask(_units("session", "closed")), _units("session", "closed"))
        self.assertEqual(variables.mask(_units("totally", "new", "line")), _units("totally", "new", "line"))

    def test_word_is_masked_where_other_lines_have_a_value(self) -> None:
        lines = [_units("failed", "password", "for", VALUE, "from"), _units("failed", "password", "for", "root", "from")]
        variables = VariableSlots(lines, min_values=3, min_context=2)
        self.assertEqual(variables.mask(lines[1]), lines[0])

    def test_two_values_are_not_enough(self) -> None:
        lines = [_units("mode", "long", "accepted", "swaps"), _units("mode", "short", "accepted", "swaps")]
        variables = VariableSlots(lines, min_values=3, min_context=2)
        self.assertEqual(variables.mask(lines[0]), lines[0])

    def test_background_is_searched_inside_each_record_shape(self) -> None:
        shape = ("json", "json:level|message")

        def record(*words: str) -> LogLine:
            return LogLine(source="test", number=1, text=" ".join(words), units=_units(*words), numbers=(), shape=shape)

        lines = [self._line("disk", "full", "on", "node")] * 300
        lines += [record("message", "session", "opened", "level", "info")] * 30
        lines += [record("message", "queue", "drained", "level", "info")] * 30
        background = BackgroundWords(lines, share=0.9, min_lines=50)
        # Во всём логе слова обёртки редки, но стоят в каждой записи своей формы.
        self.assertEqual(background.words, frozenset())
        self.assertEqual(background.shape_words[shape[1]], frozenset({"message", "level", "info"}))
        cleaned = background.clean(lines)
        self.assertEqual(cleaned[300].words, ("session", "opened"))
        self.assertEqual(cleaned[0].words, ("disk", "full", "on", "node"))

    def test_line_is_cut_to_the_word_limit_after_cleaning(self) -> None:
        units = (("alpha", "beta"), (), ("gamma", "delta", "epsilon"), ("zeta",))
        self.assertEqual(truncated(units, 3), (("alpha", "beta"), (), ("gamma",)))
        line = LogLine(source="test", number=1, text="", units=units, numbers=())
        self.assertEqual(limit_words([line], 3)[0].words, ("alpha", "beta", VALUE, "gamma"))
        self.assertIs(limit_words([line], 10)[0], line)


class AnyInputTests(unittest.TestCase):
    """Любой вход (вне статьи): каталоги, архивы, двоичные файлы, другие кодировки, сверхдлинные строки."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def _read(self, **options) -> tuple[LogLineReader, list[LogLine]]:
        reader = LogLineReader(
            word_min=main.LOG_WORD_MIN, word_max=main.LOG_WORD_MAX, line_words_max=main.LOG_LINE_PARSE_WORDS_MAX,
            slot_words=main.LOG_NUMBER_SLOT_WORDS, **options,
        )
        return reader, reader.read_sources([str(self.root)])

    def test_directory_is_read_with_archives_encodings_and_garbage(self) -> None:
        (self.root / "sub").mkdir()
        (self.root / "plain.log").write_text("service started\n###### 12345 ######\n\nservice stopped\n", encoding="utf-8")
        (self.root / "sub" / "packed.log.gz").write_bytes(gzip.compress("queue drained\n".encode("utf-8")))
        (self.root / "sub" / "wide.log").write_text("служба запущена\n", encoding="utf-16")
        (self.root / "sub" / "old.log").write_bytes("ошибка диска\n".encode("cp1251"))
        (self.root / "binary.bin").write_bytes(bytes(range(256)) * 8)
        (self.root / "broken.gz").write_bytes(b"\x1f\x8b" + b"not an archive")
        (self.root / "empty.log").write_bytes(b"")
        reader, lines = self._read()
        self.assertEqual(
            sorted(line.words for line in lines),
            sorted([("service", "started"), ("service", "stopped"), ("queue", "drained"), ("служба", "запущена"),
                    ("ошибка", "диска")]),
        )
        notes = {Path(item["source"]).name: item for item in reader.files}
        self.assertEqual(len(notes), 7)
        self.assertEqual(notes["plain.log"]["without_words"], 2)
        self.assertIn("двоичный", notes["binary.bin"]["note"])
        self.assertIn("не прочитан", notes["broken.gz"]["note"])
        self.assertEqual(notes["empty.log"]["lines"], 0)

    def test_long_line_is_cut_and_reading_stops_at_the_line_limit(self) -> None:
        (self.root / "a.log").write_text("head word " + "x" * 5000 + " tail\nsecond line\nthird line\n", encoding="utf-8")
        reader, lines = self._read(line_chars_max=100, lines_max=2)
        self.assertEqual([line.words for line in lines], [("head", "word"), ("second", "line")])
        self.assertEqual(reader.files[0]["cut_lines"], 1)
        self.assertIn("предел", reader.files[0]["note"])

    def test_pile_checks_the_end_of_each_file_against_the_rest(self) -> None:
        def lines(source: str, count: int) -> list[LogLine]:
            return [
                LogLine(source=source, number=number, text="x", units=_units("word"), numbers=())
                for number in range(1, count + 1)
            ]

        pile = log_insights.LogPile(
            lines("big.log", 400) + lines("small.log", 50), target_share=0.25, min_lines=200, mark=" (конец)"
        )
        self.assertEqual(len(pile.reference), 350)
        self.assertEqual({line.source for line in pile.target}, {"big.log (конец)"})
        self.assertEqual([line.number for line in pile.target], list(range(301, 401)))
        self.assertEqual(pile.split, {"big.log": 301})

    def test_space_keeps_the_most_frequent_lines_within_the_point_limit(self) -> None:
        frequent = [f"frequent event number {name} happened again" for name in ("one", "two", "three")]
        rare = [f"rare thing {name} occurred once somewhere" for name in ("alpha", "beta", "gamma", "delta")]
        reader = _reader()
        reference = []
        for number, line in enumerate(frequent * 20 + rare, start=1):
            units, numbers = reader.parse(line)
            reference.append(LogLine(source="a.log", number=number, text=line, units=units, numbers=numbers))
        corpus = log_insights.LogCorpus(reference, [])
        encoder = _encoder(corpus.stimuli)
        with mock.patch.object(log_insights, "LOG_SPACE_POINTS_MAX", 30):
            space = log_insights.LogCodeSpace(corpus, encoder)
        self.assertLessEqual(len(space.points), 30)
        self.assertEqual(space.budget["stimuli_inside"], 3)
        self.assertEqual({corpus.stimuli[index][0] for index in space.outside}, {"rare"})
        # Без предела в пространство входят все строки.
        self.assertEqual(log_insights.LogCodeSpace(corpus, encoder).outside, frozenset())


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

    def test_value_placeholders_do_not_enter_the_code(self) -> None:
        encoder = _encoder([("invalid", "user", "from")])
        with_value = encoder.encode(("invalid", "user", VALUE, "from"))
        without = encoder.encode(("invalid", "user", "from"))
        self.assertEqual([item.code for item in with_value], [item.code for item in without])

    def test_views_are_whole_line_head_and_tail(self) -> None:
        encoder = _encoder([("alpha", "beta", "gamma")])
        views = [item.view for item in encoder.encode(("alpha", "beta", "gamma"))]
        self.assertEqual(views.count("whole"), 1)
        self.assertIn("head", views)
        self.assertIn("tail", views)

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
                           units=_units("job", "done", "elapsed"), numbers=(("elapsed", value),))

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


class PileTests(unittest.TestCase):
    """Куча логов без деления на эталон и цель: сигналы ищутся во всех файлах, итог — дашборд (вне статьи)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        root = Path(cls._directory.name)
        (root / "logs").mkdir()
        for index in range(3):
            lines = PipelineTests._run_lines(index, rounds=12)
            if index == 0:
                # Редкое событие в начале файла: новым оно не будет, но останется редким.
                lines.insert(40, "raid controller degraded array rebuild required")
            if index == 2:
                # Новое событие в конце файла.
                lines.insert(len(lines) - 20, "kernel panic: unable to mount root filesystem on device")
            (root / "logs" / f"service{index}.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (root / "logs" / "core.dump").write_bytes(bytes(range(256)) * 16)
        (root / "logs" / "noise.txt").write_text("#### ---- ####\n12345 67890\n", encoding="utf-8")
        cls.out = root / "out"
        cls.insights = log_insights.LogInsightsExperiment(pile=[str(root / "logs")]).run(out_dir=str(cls.out))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def test_garbage_files_are_reported_and_do_not_stop_the_run(self) -> None:
        files = {Path(item["source"]).name: item for item in self.insights["input"]["files"]}
        self.assertEqual(self.insights["input"]["mode"], "pile")
        self.assertIn("двоичный", files["core.dump"]["note"])
        self.assertEqual((files["noise.txt"]["lines"], files["noise.txt"]["without_words"]), (0, 2))
        self.assertEqual(len(self.insights["input"]["pile_split"]), 3)

    def test_new_line_at_the_end_of_a_file_is_found(self) -> None:
        new = [item for item in self.insights["new_lines"] if "kernel panic" in item["text"]]
        self.assertEqual(len(new), 1)
        self.assertIn(main.LOG_PILE_TAIL_MARK, new[0]["where"])

    def test_rare_line_at_the_start_of_a_file_and_new_line_at_the_end_give_incidents(self) -> None:
        signals = [signal["text"] for item in self.insights["incidents"] for signal in item["signals"]]
        self.assertTrue(any("raid controller degraded" in text for text in signals), signals)
        self.assertTrue(any("kernel panic" in text for text in signals), signals)

    def test_timeline_covers_every_line_of_every_file(self) -> None:
        timeline = self.insights["timeline"]
        self.assertEqual(sum(row["lines"] for row in timeline), self.insights["summary"]["lines"])
        for row in timeline:
            self.assertEqual(sum(cell["lines"] for cell in row["cells"]), row["lines"])
            self.assertLessEqual(len(row["cells"]), main.LOG_DASHBOARD_TIMELINE_COLUMNS)
        self.assertGreater(sum(cell["signals"] for row in timeline for cell in row["cells"]), 0)

    def test_dashboard_is_one_self_contained_page(self) -> None:
        page = (self.out / Path(main.LOG_INSIGHTS_DASHBOARD_PATH).name).read_text(encoding="utf-8")
        for part in ("Инсайты из логов", "Главное", "Карта событий", "Лента сигналов", "Инциденты", "Что было на входе",
                     "kernel panic", '<canvas id="map">', "data:image/png;base64,"):
            self.assertIn(part, page)
        # Ни одной внешней ссылки: страница открывается без сети.
        self.assertNotIn("://", page)
        self.assertNotIn("<script src", page)
        self.assertNotIn("<link", page)


if __name__ == "__main__":
    unittest.main()
