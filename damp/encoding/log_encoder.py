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

Формат лога заранее не известен, правил под формат нет. Метки времени распространённых видов распознаются
при разборе. Слова, стоящие почти в каждой строке (хост, процесс, уровень), находятся по статистике эталона
и в код не входят (BackgroundWords). Переменные места строки — имя пользователя, хост — тоже находятся
по статистике и заменяются заполнителем значения (VariableSlots).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from functools import cached_property
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
# Заполнитель значения в последовательности слов строки: число, идентификатор, адрес или переменное слово.
VALUE = "*"

Units = tuple[tuple[str, ...], ...]


def content(words: Sequence[str]) -> tuple[str, ...]:
    """Слова строки без заполнителей значений — то, что входит в структурный код."""
    return tuple(word for word in words if word != VALUE)


def _collapsed(units: Sequence[tuple[str, ...]]) -> Units:
    """Единицы строки без идущих подряд заполнителей."""
    result: list[tuple[str, ...]] = []
    for unit in units:
        if not unit and result and not result[-1]:
            continue
        result.append(tuple(unit))
    return tuple(result)


def _normalised(units: Sequence[tuple[str, ...]]) -> Units:
    """Единицы строки без идущих подряд заполнителей и без заполнителей по краям."""
    result = list(_collapsed(units))
    while result and not result[0]:
        result.pop(0)
    while result and not result[-1]:
        result.pop()
    return tuple(result)


@dataclass(frozen=True)
class LogLine:
    """Строка лога.

    units — единицы строки по порядку: слова одного токена либо пустой кортеж на месте значения
    (числа, идентификатора, переменного слова). numbers — числа строки со слотами.
    """

    source: str
    number: int
    text: str
    units: Units
    numbers: tuple[tuple[str, float], ...]

    @cached_property
    def words(self) -> tuple[str, ...]:
        """Стимул: слова строки по порядку, на месте значений — заполнитель."""
        return tuple(word for unit in self.units for word in (unit or (VALUE,)))


class LogLineReader:
    """Чтение логов и предобработка строк по §7.1.3, без знания формата лога.

    Строка делится на токены по пробелам, знакам равенства, запятым, скобкам и кавычкам; токен — на слова
    по небуквенным символам; буквы — любого алфавита (§7.1.4.1: кодовые точки Unicode). Слитные составные
    имена (camelCase) делятся по заглавным буквам; слова приводятся к нижнему регистру, слова короче
    ``word_min`` и длиннее ``word_max`` символов отбрасываются. Группа букв и цифр с цифрой внутри (номер,
    идентификатор, хеш) словом не считается (вне статьи: в §7.1.3 цифры — разделители, но для логов
    обрывки идентификаторов дают случайные слова).

    Токен без слов — значение: на его месте в строке остаётся заполнитель. Число (в том числе с короткой
    единицей измерения: 35ms, 12%) записывается со слотом. Метка времени — ISO, syslog, asctime, журнала
    веб-сервера или просто время — убирается до деления на токены: названия месяцев и дней недели в ней
    словами не становятся. Слова строки после ``line_words_max`` отбрасываются.
    """

    _MONTH = (
        r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?"
        r"|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    )
    _WEEKDAY = r"(?:Mon|Tue(?:s)?|Wed(?:nes)?|Thu(?:rs)?|Fri|Sat(?:ur)?|Sun)(?:day)?"
    _TIME = r"\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?"
    _STAMP = re.compile(
        rf"\d{{4}}-\d{{2}}-\d{{2}}(?:[T ]{_TIME})?Z?"
        rf"|\d{{1,2}}/{_MONTH}/\d{{4}}(?::{_TIME})?(?:\s[+-]\d{{4}})?"
        rf"|(?<![^\W\d_])(?:{_WEEKDAY},?\s+)?{_MONTH}\s+\d{{1,2}}(?!\d)(?:\s+{_TIME})?(?:\s+\d{{4}}(?!\d))?"
        rf"|(?<![\w.])\d{{1,2}}\s+{_MONTH}\s+\d{{4}}(?!\d)(?:\s+{_TIME})?"
        rf"|(?<![\w.:]){_TIME}(?![\w:])"
    )
    _SEPARATORS = re.compile(r"[\s=,;()\[\]{}\"'<>|]+")
    _RUN = re.compile(r"[^\W_]+")
    # Число, возможно с короткой единицей измерения.
    _MEASURE = re.compile(r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?:[^\W\d_]{1,3}|%)?")
    _EDGES = ".:!?#*/\\"

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
                "note": "строка делится на токены и слова по небуквенным символам, слова — в нижний регистр",
            },
        )

    @staticmethod
    def _camel_parts(run: str) -> list[str]:
        """Части слитного имени: граница перед заглавной буквой после строчной и перед последней заглавной
        в цепочке заглавных, за которой идёт строчная (HTTPServer → HTTP, Server)."""
        parts: list[str] = []
        start = 0
        for index in range(1, len(run)):
            previous, current = run[index - 1], run[index]
            following = run[index + 1] if index + 1 < len(run) else ""
            if current.isupper() and (previous.islower() or (previous.isupper() and following.islower())):
                parts.append(run[start:index])
                start = index
        parts.append(run[start:])
        return parts

    def _run_words(self, run: str) -> list[str]:
        if any(char.isdigit() for char in run):
            return []
        return [part.lower() for part in self._camel_parts(run) if self._word_min <= len(part) <= self._word_max]

    def parse(self, text: str) -> tuple[Units, tuple[tuple[str, float], ...]]:
        """Единицы строки и её числа со слотами.

        Слот числа — последние слова перед ним (после прошлого числа). Числа, идущие подряд без слов между
        ними (список значений, колонки без подписей), получают слот прошлого числа с порядковым номером.
        """
        units: list[tuple[str, ...]] = []
        numbers: list[tuple[str, float]] = []
        since_number: list[str] = []
        last_slot = ""
        run_length = 0
        words_total = 0
        for token in self._SEPARATORS.split(self._STAMP.sub(" ", text)):
            token = token.strip(self._EDGES)
            if not token:
                continue
            measure = self._MEASURE.fullmatch(token)
            if measure:
                value = float(measure.group(1))
                if math.isfinite(value):
                    if since_number:
                        last_slot = " ".join(since_number[-self._slot_words:])
                        slot = last_slot
                        run_length = 0
                    else:
                        run_length += 1
                        slot = f"{last_slot} #{run_length}".strip()
                    numbers.append((slot, value))
                    since_number = []
                units.append(())
                continue
            words = [word for run in self._RUN.findall(token) for word in self._run_words(run)]
            words = words[: max(0, self._line_words_max - words_total)]
            if words:
                words_total += len(words)
                units.append(tuple(words))
                since_number.extend(words)
            elif any(char.isalnum() for char in token):
                units.append(())
        return _normalised(units), tuple(numbers)

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
            units, numbers = self.parse(raw)
            if not any(units):
                skipped += 1
                continue
            lines.append(LogLine(source=source, number=number, text=raw.rstrip(), units=units, numbers=numbers))
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


class BackgroundWords:
    """Фоновые слова: стоят почти в каждой строке лога и строк не различают (вне статьи).

    Имя хоста, процесса, уровень записи повторяются в заголовке каждой строки. В коде они дали бы всем
    строкам общие биты: сходство любых двух строк выросло бы, а порог λ перестал бы отделять кластеры —
    то самое паразитное сходство, из-за которого статья меняла коды (§7.1.8.3, §7.1.8.4). Слово считается
    фоновым, если встречается не меньше чем в доле ``share`` строк эталона; формат лога для этого знать
    не нужно. Строка, целиком состоящая из фоновых слов, остаётся как есть.
    """

    def __init__(self, lines: Iterable[LogLine], *, share: float, min_lines: int) -> None:
        if not 0.0 < share <= 1.0:
            raise ValueError("share must be in (0, 1]")
        counts: Counter[str] = Counter()
        total = 0
        for line in lines:
            total += 1
            counts.update(list(dict.fromkeys(content(line.words))))
        # На нескольких строках доля ничего не говорит: фон ищется только в достаточно большом эталоне.
        found = [word for word, count in counts.items() if total >= min_lines and count >= share * total]
        self.words = frozenset(found)
        LOGGER.event(
            "logs.background",
            section=OUTSIDE_ARTICLE,
            data={"lines": total, "share": share, "words": sorted(found)},
        )

    def strip(self, units: Sequence[tuple[str, ...]]) -> Units:
        """Единицы строки без фоновых слов; единица из одних фоновых слов уходит целиком."""
        kept = [tuple(word for word in unit if word not in self.words) for unit in units]
        result = _normalised([unit for unit, original in zip(kept, units) if unit or not original])
        return result if any(result) else tuple(units)

    def clean(self, lines: Sequence[LogLine]) -> list[LogLine]:
        """Строки без фоновых слов."""
        if not self.words:
            return list(lines)
        return [replace(line, units=self.strip(line.units)) for line in lines]


class VariableSlots:
    """Переменные места строк, найденные по эталону (вне статьи).

    Единица строки (слова одного токена) считается переменной, если при том же окружении на её месте
    встречается не меньше ``min_values`` разных единиц либо в какой-то строке на этом месте стоит значение.
    Переменная единица заменяется заполнителем значения: «invalid user admin from» и «invalid user guest
    from» становятся одной строкой «invalid user * from». Окружение должно содержать не меньше
    ``min_context`` единиц со словами, поэтому в коротких строках («session opened», «session closed»)
    слова не заменяются. Правило не зависит от формата лога и применяется повторно, пока находятся новые
    места: у строки их может быть несколько.

    Статья оставляет различие вариантов сходству кодов (§7.1.4); для строк логов этого мало: переменное
    слово в короткой строке меняет код сильнее, чем различие двух событий в длинной.
    """

    ROUNDS = 8

    def __init__(self, lines: Iterable[Sequence[tuple[str, ...]]], *, min_values: int, min_context: int) -> None:
        if min_values < 2 or min_context < 1:
            raise ValueError("min_values must be >= 2 and min_context >= 1")
        self._min_values = int(min_values)
        self._min_context = int(min_context)
        self._patterns: set[Units] = set()
        current = {units: units for units in dict.fromkeys(_normalised(line) for line in lines)}
        for _ in range(self.ROUNDS):
            distinct = list(dict.fromkeys(current.values()))
            present = set(distinct)
            fillers: dict[Units, dict[tuple[str, ...], None]] = defaultdict(dict)
            for units in distinct:
                if sum(1 for unit in units if unit) - 1 < self._min_context:
                    continue
                for index, unit in enumerate(units):
                    if unit:
                        fillers[_collapsed(units[:index] + ((),) + units[index + 1:])][unit] = None
            found = {key for key, values in fillers.items() if len(values) >= self._min_values or key in present}
            self._patterns |= found
            changed = False
            for original, units in current.items():
                masked = self.mask(units)
                if masked != units:
                    current[original] = masked
                    changed = True
            if not changed:
                break
        distinct_after = len(set(current.values()))
        LOGGER.event(
            "logs.variables",
            section=OUTSIDE_ARTICLE,
            data={
                "lines_before": len(current),
                "lines_after": distinct_after,
                "patterns": len(self._patterns),
                "min_values": self._min_values,
                "min_context": self._min_context,
            },
        )

    def mask(self, units: Sequence[tuple[str, ...]]) -> Units:
        """Единицы строки, где переменные заменены заполнителем значения."""
        units = _normalised(units)
        while True:
            if sum(1 for unit in units if unit) - 1 < self._min_context:
                return units
            for index, unit in enumerate(units):
                if unit and _collapsed(units[:index] + ((),) + units[index + 1:]) in self._patterns:
                    units = _normalised(units[:index] + ((),) + units[index + 1:])
                    break
            else:
                return units

    def clean(self, lines: Sequence[LogLine]) -> list[LogLine]:
        """Строки с заполнителями на переменных местах."""
        if not self._patterns:
            return list(lines)
        cache: dict[Units, Units] = {}
        result = []
        for line in lines:
            masked = cache.get(line.units)
            if masked is None:
                masked = cache[line.units] = self.mask(line.units)
            result.append(line if masked == line.units else replace(line, units=masked))
        return result


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
        unique = [words for words in dict.fromkeys(content(words) for words in lines) if words]
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

    @property
    def view(self) -> str:
        """Взгляд на строку: «whole» — строка целиком одним фрагментом, иначе акцент разбиения."""
        return "whole" if len(self.fragments) == 1 else self.accent

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
        """Все коды строки W(s) ∪ W⁻¹(s): каждое разбиение с акцентом на начало и на конец (§7.1.6).

        Заполнители значений в код не входят и мест в строке не занимают.
        """
        words = content(words)
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
    "VALUE",
    "BackgroundWords",
    "FragmentDictionary",
    "LineCode",
    "LineFragmenter",
    "LogLine",
    "LogLineEncoder",
    "LogLineReader",
    "NumericValueEncoder",
    "TokenAlphabet",
    "VariableSlots",
    "content",
]
