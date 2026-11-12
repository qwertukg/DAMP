"""Первичное кодирование строк лога в разреженные битовые коды (§2.3.5, §3.2, §4.2.1, §7.1.3, §7.1.4).

Строка лога играет роль слова из §7.1, слово строки — роль символа:

1. предобработка (§7.1.3): строка делится на слова по небуквенным символам, слова приводятся к нижнему
   регистру, слишком короткие и слишком длинные отбрасываются;
2. словарь фрагментов (§7.1.3): все подряд идущие группы слов всех строк, вес фрагмента — его способность
   составлять целые строки (сумма долей длины по строкам, где он встречается);
3. фрагментации (§7.1.4): одна строка разбивается на фрагменты несколькими способами («несколько точек
   зрения», §7.1.1) — каждое разбиение даёт свою точку кодового пространства;
4. код фрагментации (§7.1.4.1, §7.1.4.4): объединение кодов фрагментов; код фрагмента — биты кодов
   «позиция во фрагменте × слово» в пределах бюджета насыщенности фрагмента (табл. 3);
5. акценты (§7.1.4.5): каждое разбиение кодируется дважды — с бюджетом, убывающим от начала строки
   и от её конца.

Числа из строки в структурный код не входят (§7.1.3: небуквенные символы удаляются); они кодируются
отдельно широкими детекторами общего Encoder (§4.1, §4.2.1).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import glob
import hashlib
import math
import os
import re
from typing import Iterable, Mapping, Sequence

import numpy as np

from damp.article_refs import (
    CHARACTERS_WORDS,
    COLOR_ENCODING,
    INTEGER_ENCODING,
    MORPHOLOGY_ACCENTS,
    MORPHOLOGY_DICTIONARY,
    MORPHOLOGY_FRAGMENT_CODES,
    MORPHOLOGY_TOKEN_ENCODING,
    OUTSIDE_ARTICLE,
    SPARSE_BIT_VECTORS,
    WIDE_DETECTORS,
)
from damp.encoding.bitarray import BitArray
from damp.encoding.damp_encoder import Encoder, OpenedDimension, geometric_detector_layers
from damp.logging import LOGGER


# ---------------------------------------------------------------------------------------------- строки
@dataclass(frozen=True)
class LogLine:
    """Строка лога: слова — стимул структурного кода, числа — значения со слотами."""

    source: str
    number: int
    text: str
    words: tuple[str, ...]
    numbers: tuple[tuple[str, float], ...]


class LogLineReader:
    """Чтение логов и предобработка строк по §7.1.3.

    Строка делится на слова по небуквенным символам; слитные составные имена (camelCase) делятся
    по заглавным буквам; слова приводятся к нижнему регистру, слова короче ``word_min`` и длиннее
    ``word_max`` символов отбрасываются. Группа букв и цифр с цифрой внутри (номер, идентификатор, хеш)
    считается значением, а не словом, и в структурный код не входит (вне статьи: в §7.1.3 цифры —
    разделители, но для логов обрывки идентификаторов дают случайные слова). Строка длиннее
    ``line_words_max`` слов обрезается до первых ``line_words_max``.
    """

    _RUN = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")
    _CAMEL = re.compile(r"[A-ZА-ЯЁ]?[a-zа-яё]+|[A-ZА-ЯЁ]+(?![a-zа-яё])")
    # Порядок альтернатив: метка времени и номер с точками (версия, адрес, ссылка на раздел) — не числа.
    _TOKEN = re.compile(
        r"(?P<stamp>\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)?Z?|\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?)"
        r"|(?P<dotted>(?<![\w.])\d+(?:\.\d+){2,}(?![\w.]))"
        r"|(?P<number>(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w]))"
        r"|(?P<run>[A-Za-zА-Яа-яЁё0-9]+)"
    )

    def __init__(self, *, word_min: int, word_max: int, line_words_max: int, slot_words: int) -> None:
        if word_min <= 0 or word_max < word_min:
            raise ValueError("word length bounds must satisfy 0 < word_min <= word_max")
        if line_words_max <= 0 or slot_words <= 0:
            raise ValueError("line_words_max and slot_words must be positive")
        self._word_min = int(word_min)
        self._word_max = int(word_max)
        self._line_words_max = int(line_words_max)
        self._slot_words = int(slot_words)
        LOGGER.event(
            "logs.reader.init",
            section=MORPHOLOGY_DICTIONARY,
            data={
                "word_min": self._word_min,
                "word_max": self._word_max,
                "line_words_max": self._line_words_max,
                "note": "строка делится на слова по небуквенным символам, слова — в нижний регистр",
            },
        )

    def _run_words(self, run: str) -> list[str]:
        if any(char.isdigit() for char in run):
            return []
        parts = self._CAMEL.findall(run) or [run]
        return [part.lower() for part in parts if self._word_min <= len(part) <= self._word_max]

    def parse(self, text: str) -> tuple[tuple[str, ...], tuple[tuple[str, float], ...]]:
        """Слова строки и её числа со слотами. Слот числа — последние слова перед ним (после прошлого числа)."""
        words: list[str] = []
        numbers: list[tuple[str, float]] = []
        since_number: list[str] = []
        last_slot = ""
        for match in self._TOKEN.finditer(text):
            kind = match.lastgroup
            if kind == "run":
                found = self._run_words(match.group())
                words.extend(found)
                since_number.extend(found)
            elif kind == "number":
                value = float(match.group())
                if not math.isfinite(value):
                    continue
                if since_number:
                    last_slot = " ".join(since_number[-self._slot_words:])
                    slot = last_slot
                else:
                    # Числа подряд (список значений): общий слот продолжения.
                    slot = f"{last_slot} …".strip()
                numbers.append((slot, value))
                since_number = []
        return tuple(words[: self._line_words_max]), tuple(numbers)

    @staticmethod
    def _decode(data: bytes) -> str:
        for encoding in ("utf-8", "cp1251"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")

    def read(self, path: str) -> list[LogLine]:
        """Строки файла, в которых есть хотя бы одно слово; файл — один запуск."""
        with open(path, "rb") as handle:
            text = self._decode(handle.read())
        source = os.path.normpath(path).replace("\\", "/")
        lines: list[LogLine] = []
        skipped = 0
        for number, raw in enumerate(text.splitlines(), start=1):
            words, numbers = self.parse(raw)
            if not words:
                skipped += 1
                continue
            lines.append(LogLine(source=source, number=number, text=raw.rstrip(), words=words, numbers=numbers))
        LOGGER.event(
            "logs.reader.file",
            section=OUTSIDE_ARTICLE,
            data={"source": source, "lines": len(lines), "skipped_without_words": skipped},
        )
        return lines

    def read_sources(self, patterns: Iterable[str]) -> list[LogLine]:
        """Все файлы по шаблонам glob в порядке шаблонов; каждый файл — отдельный запуск."""
        paths: list[str] = []
        for pattern in patterns:
            for path in sorted(glob.glob(pattern)):
                if os.path.isfile(path) and path not in paths:
                    paths.append(path)
        lines: list[LogLine] = []
        for path in paths:
            lines.extend(self.read(path))
        LOGGER.event(
            "logs.reader.sources",
            section=OUTSIDE_ARTICLE,
            data={"patterns": list(patterns), "files": len(paths), "lines": len(lines)},
        )
        return lines


# ---------------------------------------------------------------------------------------------- словарь
class FragmentDictionary:
    """Словарь фрагментов строк с весами (§7.1.3).

    Рекурсивное деление строки на начало и конец даёт все её подряд идущие группы слов. Вес фрагмента —
    сумма по строкам, где он встречается, доли его длины в длине строки: фрагменты ранжируются не по
    частоте, а по способности составлять целые строки. Каждая различная строка учитывается один раз.
    Фрагменты длиннее ``fragment_words_max`` слов в словарь не входят (вес 0).
    """

    def __init__(self, lines: Iterable[Sequence[str]], *, fragment_words_max: int) -> None:
        if fragment_words_max <= 0:
            raise ValueError("fragment_words_max must be positive")
        self._fragment_words_max = int(fragment_words_max)
        weights: dict[tuple[str, ...], float] = defaultdict(float)
        # Порядок обхода — порядок первого появления: суммы весов и выбор разбиений не зависят от хеш-зерна Python.
        unique = list(dict.fromkeys(tuple(words) for words in lines if words))
        for words in unique:
            length = len(words)
            fragments = dict.fromkeys(
                words[start:stop]
                for start in range(length)
                for stop in range(start + 1, min(length, start + self._fragment_words_max) + 1)
            )
            for fragment in fragments:
                weights[fragment] += len(fragment) / length
        self._weights = dict(weights)
        self._lines = len(unique)
        LOGGER.event(
            "logs.dictionary",
            section=MORPHOLOGY_DICTIONARY,
            data={
                "lines": self._lines,
                "fragments": len(self._weights),
                "words": sum(1 for fragment in self._weights if len(fragment) == 1),
                "fragment_words_max": self._fragment_words_max,
                "top": [
                    f"{' '.join(fragment)}={weight:.1f}"
                    for fragment, weight in sorted(self._weights.items(), key=lambda item: -item[1])[:6]
                ],
            },
        )

    def __len__(self) -> int:
        return len(self._weights)

    @property
    def fragment_words_max(self) -> int:
        return self._fragment_words_max

    def weight(self, fragment: Sequence[str]) -> float:
        return self._weights.get(tuple(fragment), 0.0)

    def known(self, fragment: Sequence[str]) -> bool:
        return tuple(fragment) in self._weights


class LineFragmenter:
    """Несколько удачных разбиений строки на фрагменты (§7.1.4, табл. 1).

    Вес разбиения — сумма весов его фрагментов (формула 1). Для каждого числа фрагментов от 1 до
    ``fragments_max`` берётся разбиение наибольшего веса: так строка получает и крупные, и мелкие
    разбиения. Статья отбирает разбиения по медиане веса среди всех; здесь отбор — по лучшему на каждое
    число фрагментов, потому что сумма весов всегда растёт с числом фрагментов и порог по медиане оставил бы
    только самые мелкие разбиения.
    """

    def __init__(self, dictionary: FragmentDictionary, *, fragments_max: int) -> None:
        if fragments_max <= 0:
            raise ValueError("fragments_max must be positive")
        self._dictionary = dictionary
        self._fragments_max = int(fragments_max)

    def fragmentations(self, words: Sequence[str]) -> list[tuple[tuple[str, ...], ...]]:
        words = tuple(words)
        length = len(words)
        if length == 0:
            return []
        weight = self._dictionary.weight
        parts_max = min(self._fragments_max, length)
        # best[k][i] — наибольший вес разбиения первых i слов на k фрагментов; cut[k][i] — начало последнего.
        best = [[-math.inf] * (length + 1) for _ in range(parts_max + 1)]
        cut = [[0] * (length + 1) for _ in range(parts_max + 1)]
        best[0][0] = 0.0
        for parts in range(1, parts_max + 1):
            for stop in range(parts, length + 1):
                for start in range(parts - 1, stop):
                    previous = best[parts - 1][start]
                    if previous == -math.inf:
                        continue
                    value = previous + weight(words[start:stop])
                    if value > best[parts][stop]:
                        best[parts][stop] = value
                        cut[parts][stop] = start
        result: list[tuple[tuple[str, ...], ...]] = []
        for parts in range(1, parts_max + 1):
            fragments: list[tuple[str, ...]] = []
            stop = length
            for level in range(parts, 0, -1):
                start = cut[level][stop]
                fragments.append(words[start:stop])
                stop = start
            fragmentation = tuple(reversed(fragments))
            if fragmentation not in result:
                result.append(fragmentation)
        return result


# ---------------------------------------------------------------------------------------------- алфавит
class TokenAlphabet:
    """Алфавит 𝒜 = {f(a) : a ∈ ℕ × слова} (§7.1.4.1): код слова на позиции внутри фрагмента.

    Код — ``token_bits`` различных случайных бит (§2, §2.0.1), упорядоченных по «цвету» (§3.2): цвет бита —
    его приоритет при объединении, в код попадают первые σ бит по порядку. Поэтому коды одного слова при
    разных бюджетах вложены друг в друга и остаются сравнимыми. Функция детерминирована.
    """

    def __init__(self, *, code_length: int, token_bits: int, seed: int) -> None:
        if code_length <= 0 or not 0 < token_bits <= code_length:
            raise ValueError("token_bits must be in (0, code_length]")
        self.code_length = int(code_length)
        self.token_bits = int(token_bits)
        self._seed = int(seed)
        self._cache: dict[tuple[int, str], np.ndarray] = {}
        LOGGER.event(
            "logs.alphabet.init",
            section=MORPHOLOGY_TOKEN_ENCODING,
            data={"code_length": self.code_length, "token_bits": self.token_bits, "seed": self._seed},
        )
        LOGGER.event(
            "logs.alphabet.colours",
            section=COLOR_ENCODING,
            data={
                "coloured_bits": self.token_bits,
                "note": "цвет бита — его приоритет: в код входят первые σ бит кода «позиция × слово»",
            },
        )

    def bits(self, position: int, word: str) -> np.ndarray:
        """Биты f(position, word) в порядке приоритета."""
        key = (int(position), word)
        cached = self._cache.get(key)
        if cached is None:
            digest = hashlib.blake2b(f"{self._seed}|{key[0]}|{word}".encode("utf-8"), digest_size=8).digest()
            rng = np.random.default_rng(int.from_bytes(digest, "little"))
            cached = rng.permutation(self.code_length)[: self.token_bits]
            self._cache[key] = cached
        return cached


# ---------------------------------------------------------------------------------------------- код строки
@dataclass(frozen=True)
class LineCode:
    """Код одного разбиения строки: точка кодового пространства (§7.1.6)."""

    code: BitArray
    fragments: tuple[tuple[str, ...], ...]
    accent: str

    def describe(self) -> str:
        return " ⋅ ".join(" ".join(fragment) for fragment in self.fragments)


class LogLineEncoder:
    """Коды разбиений строки: по фрагментам, с акцентами на начало и на конец (§7.1.4.4, §7.1.4.5).

    Бюджет насыщенности фрагмента берётся из таблицы «число фрагментов → бюджеты по номеру фрагмента»
    (табл. 3); при акценте на конец номер фрагмента считается с конца (§7.1.4.5). Бюджет делится между
    словами фрагмента поровну по кругу, но каждое слово даёт хотя бы один бит: код короткого фрагмента —
    объединение кодов его слов, код длинного — по биту от слова, то есть хеш (§7.1.4.4).
    """

    ACCENTS = ("head", "tail")

    def __init__(
        self,
        alphabet: TokenAlphabet,
        fragmenter: LineFragmenter,
        *,
        fragment_saturation: Mapping[int, Sequence[int]],
    ) -> None:
        self._alphabet = alphabet
        self._fragmenter = fragmenter
        self._saturation = {int(parts): tuple(int(v) for v in budgets) for parts, budgets in fragment_saturation.items()}
        for parts, budgets in self._saturation.items():
            if len(budgets) != parts or any(value <= 0 for value in budgets):
                raise ValueError("fragment_saturation[k] must hold k positive budgets")
        self.code_length = alphabet.code_length
        self._encoded = 0
        LOGGER.event(
            "logs.encoder.init",
            section=MORPHOLOGY_FRAGMENT_CODES,
            data={
                "code_length": self.code_length,
                "fragment_saturation": {str(k): list(v) for k, v in sorted(self._saturation.items())},
            },
        )
        LOGGER.event(
            "logs.encoder.accents",
            section=MORPHOLOGY_ACCENTS,
            data={"accents": list(self.ACCENTS), "note": "бюджет фрагментов убывает от начала и от конца строки"},
        )

    def _budgets(self, parts: int, accent: str) -> tuple[int, ...]:
        if parts not in self._saturation:
            raise ValueError(f"no saturation budgets for {parts} fragments")
        budgets = self._saturation[parts]
        return budgets if accent == "head" else tuple(reversed(budgets))

    def _fragment_bits(self, fragment: Sequence[str], budget: int, accent: str) -> list[int]:
        """Биты фрагмента: слово i даёт первые q_i бит кода f(i, слово); Σ q_i = max(budget, число слов)."""
        count = len(fragment)
        quota = [1] * count
        order = range(count) if accent == "head" else range(count - 1, -1, -1)
        extra = max(0, int(budget) - count)
        while extra > 0:
            progressed = False
            for index in order:
                if extra == 0:
                    break
                if quota[index] < self._alphabet.token_bits:
                    quota[index] += 1
                    extra -= 1
                    progressed = True
            if not progressed:
                break
        bits: list[int] = []
        for position, word in enumerate(fragment):
            bits.extend(int(bit) for bit in self._alphabet.bits(position, word)[: quota[position]])
        return bits

    def encode_fragmentation(self, fragments: Sequence[Sequence[str]], accent: str) -> BitArray:
        """Код разбиения — объединение кодов фрагментов (формула 1, §2.2.1)."""
        if accent not in self.ACCENTS:
            raise ValueError("accent must be 'head' or 'tail'")
        budgets = self._budgets(len(fragments), accent)
        active: set[int] = set()
        for fragment, budget in zip(fragments, budgets):
            active.update(self._fragment_bits(fragment, budget, accent))
        return BitArray.from_indices(self.code_length, sorted(active))

    def encode(self, words: Sequence[str]) -> list[LineCode]:
        """Все коды строки W(s) ∪ W⁻¹(s): каждое разбиение с акцентом на начало и на конец (§7.1.6)."""
        codes: list[LineCode] = []
        seen: set[bytes] = set()
        for fragments in self._fragmenter.fragmentations(words):
            for accent in self.ACCENTS:
                code = self.encode_fragmentation(fragments, accent)
                key = bytes(code._bits)
                if key in seen:
                    continue
                seen.add(key)
                codes.append(LineCode(code=code, fragments=tuple(fragments), accent=accent))
        self._encoded += 1
        LOGGER.event(
            "logs.encoder.line",
            section=CHARACTERS_WORDS,
            data={
                "words": len(words),
                "codes": len(codes),
                "ones_mean": float(np.mean([item.code.count() for item in codes])) if codes else 0.0,
                "density_max": max((item.code.count() for item in codes), default=0) / self.code_length,
                "fragmentation": codes[-1].describe() if codes else "",
            },
        )
        return codes

    def log_density(self, codes: Sequence[BitArray]) -> None:
        """Плотность кодов пространства: статья рекомендует не более 25 % единиц (§2)."""
        ones = [code.count() for code in codes]
        LOGGER.event(
            "logs.encoder.density",
            section=SPARSE_BIT_VECTORS,
            data={
                "codes": len(ones),
                "code_length": self.code_length,
                "ones_min": min(ones, default=0),
                "ones_mean": float(np.mean(ones)) if ones else 0.0,
                "ones_max": max(ones, default=0),
                "density_max": max(ones, default=0) / self.code_length,
            },
        )


# ---------------------------------------------------------------------------------------------- числа
class NumericValueEncoder:
    """Число → код широких детекторов на знаковой логарифмической шкале (§4.1, §4.2.1, рис. 3).

    t(x) = sign(x) · log10(1 + |x| / unit), обрезанное до ±decades; отрезок кодирует общий Encoder:
    слои от самого широкого к самому узкому, каждому детектору — один случайный бит цветовой полосы
    своего слоя (§3.2). Близкие по порядку величины числа получают общие биты, далёкие — только биты
    широких слоёв.
    """

    def __init__(
        self,
        *,
        unit: float,
        decades: float,
        layers: int,
        widest: int,
        narrowest: int,
        overlap: float,
        code_length: int,
        seed: int,
    ) -> None:
        if unit <= 0.0 or decades <= 0.0:
            raise ValueError("unit and decades must be positive")
        self._unit = float(unit)
        self._decades = float(decades)
        self._dimension = OpenedDimension(
            "magnitude",
            (-self._decades, self._decades),
            geometric_detector_layers(
                layers_count=layers, widest_detectors=widest, narrowest_detectors=narrowest, overlap=overlap
            ),
            code_length=code_length,
        )
        self._encoder = Encoder(self._dimension, random_seed=seed)
        self.code_length = self._encoder.code_length
        self.layers = int(layers)
        self._bands = [self._dimension.layer_band(layer) for layer in range(self.layers)]
        LOGGER.event(
            "logs.numeric.encoder",
            section=INTEGER_ENCODING,
            data={
                "scale": "sign(x)·log10(1 + |x|/unit)",
                "unit": self._unit,
                "decades": self._decades,
                "layers": self.layers,
                "code_length": self.code_length,
            },
        )
        LOGGER.event(
            "logs.numeric.detectors",
            section=WIDE_DETECTORS,
            data={"detectors": len(self._encoder.detectors), "overlap": overlap},
        )

    def transform(self, value: float) -> float:
        scaled = math.copysign(math.log10(1.0 + abs(value) / self._unit), value)
        return max(-self._decades, min(self._decades, scaled))

    def bits(self, value: float) -> tuple[int, ...]:
        """Биты активных детекторов значения."""
        return self._encoder.dimension_bits(0, self.transform(float(value)))

    def layer_of(self, bit: int) -> int:
        """Слой детекторов, которому принадлежит бит: 0 — самый широкий."""
        for layer, (start, length) in enumerate(self._bands):
            if start <= bit < start + length:
                return layer
        raise ValueError("bit is outside the code")

    def layer_width(self, layer: int) -> float:
        """Ширина рецептивного поля детектора слоя в порядках величины."""
        detectors = [d for d in self._encoder.detectors if d.layer_index == layer]
        return 2.0 * detectors[0].half_width if detectors else 2.0 * self._decades


__all__ = [
    "FragmentDictionary",
    "LineCode",
    "LineFragmenter",
    "LogLine",
    "LogLineEncoder",
    "LogLineReader",
    "NumericValueEncoder",
    "TokenAlphabet",
]
