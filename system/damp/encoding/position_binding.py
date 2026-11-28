"""Код окна внимания: детекторы позиции с битами места и маской содержания (продолжение статьи, П.4–П.7).

Источник — продолжение статьи, docs/DAMP_continuation.md («Кодирование признаков и их положения в иерархической
модели зрения»); номера П.1…П.9 — его разделы по порядку заголовков (damp/article_refs.py). Задача (П.4): в окне
внимания лежит несколько признаков в разных местах, и код окна должен хранить, какой признак где стоит.
Объединение кодов признаков и кодов мест операцией OR (§2.2.1) эту связь теряет: набор {(a₁, p₁), (a₂, p₂)}
даёт тот же код, что {(a₁, p₂), (a₂, p₁)}.

Связь держит детектор позиции. Это широкий детектор-круг на плоскости окна (§4.3) — берётся готовым из
Encoder, — у которого две кодирующие составляющие:

    - позиция: сработавший детектор добавляет в код окна свой случайный бит операцией OR (§2.2.1, §4.2.1) —
      знак того, что в его области есть активность;
    - содержание: детектор преобразует коды признаков, попавших в его область, своей маской через XOR и
      добавляет результат в код окна.

Перед преобразованием детектор отбирает, что брать из своей области (пулинг): «max» — максимум по битам кодов
(объединение), «min» — минимум по битам (общие биты), «strongest» — код самого сильного признака.

Маска применяется одним из двух способов:

    - «address» — XOR маски с адресом каждого единичного бита кода: бит i признака уходит в бит i ⊕ m_d. Для
      одного детектора это перестановка битов, поэтому число единиц кода признака сохраняется точно, а разные
      детекторы разносят один и тот же код по разным битам;
    - «bits» — XOR маски с самим битовым вектором кода, буквально. У разрежённого кода с k единицами и маски с
      m единицами выходит k + m − 2·(общих) единиц: биты признака остаются на своих местах при любом детекторе,
      а плотность растёт на вес маски. Оставлен для проверки, которую требует П.5 («для XOR это требование
      необходимо проверять на выбранных масках и входных кодах»).

Признак в соседних местах окна делит часть детекторов (круги перекрываются), поэтому код окна меняется плавно
при сдвиге признака, а один и тот же признак в далёких местах даёт разные биты и места, и содержания.

Насыщенность кода окна держится двумя способами (§2, §3.3): длина сегмента содержания подбирается под среднюю
плотность, а если бит в окне всё равно больше бюджета, лишние отбрасываются слиянием цветов (colour merge):
цвет бита — слой детектора, который его дал; первыми уходят биты узких слоёв, биты широких остаются («to
obtain a description that makes sense as a whole, the short-wavelength bits can be discarded», §3.3.2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from damp.article_refs import (
    COLOR_MERGE,
    CONTINUATION_BINDING,
    CONTINUATION_LOCAL_DETECTOR,
    CONTINUATION_MERGING,
    SPARSE_BIT_VECTORS,
)
from damp.encoding.damp_encoder import Encoder
from damp.logging import LOGGER


@dataclass(frozen=True)
class KeyholeCodes:
    """Коды окон внимания: сегмент битов места и сегмент преобразованного содержания (§2.2.3)."""

    packed: "np.ndarray"  # (W, ⌈L/8⌉) коды окон
    code_length: int
    place_length: int  # длина сегмента битов места
    content_length: int  # длина сегмента содержания

    @property
    def count(self) -> int:
        return int(self.packed.shape[0])

    def bits(self) -> "np.ndarray":
        return np.unpackbits(self.packed, axis=1, count=self.code_length)


class PositionBindingEncoder:
    """Код окна из признаков и их мест: детектор позиции = бит места (OR) + маска содержания (XOR).

    place — кодер двумерной позиции (§4.3) на плоскости смещений от центра окна: его круги и есть детекторы
    позиции, случайный бит круга — бит места. Смещение меряется в шагах решётки элементов (unit клеток), поэтому
    один кодер места служит всем уровням.
    """

    POOLINGS = ("max", "min", "strongest", "enumerate")
    MASKS = ("address", "bits")
    # Ячеек плотной матрицы «окна × биты» за один проход.
    CHUNK_CELLS = 8_000_000
    # Строк за одну распаковку кодов признаков.
    CODE_ROWS = 8192
    _UNSET = 255  # цвет невыставленного бита

    def __init__(
        self,
        place: Encoder,
        *,
        pooling: str,
        mask: str,
        mask_bits: int,
        target_density: float,
        saturation: float,
        max_length: int,
        seed: int,
        use_gpu: bool = True,
        name: str = "keyholes",
        concepts: int | None = None,
        place_bits: int = 1,
        pooling_tie: float = 0.0,
        content_layers: int | None = None,
    ) -> None:
        if concepts is not None and int(concepts) < 1:
            raise ValueError("concepts must be positive or None")
        if len(place.dimensions) != 1 or place.dimensions[0].arity != 2:
            raise ValueError("place must be an encoder of one two-dimensional position (§4.3)")
        if pooling not in self.POOLINGS:
            raise ValueError(f"pooling must be one of {self.POOLINGS}")
        if mask not in self.MASKS:
            raise ValueError(f"mask must be one of {self.MASKS}")
        if mask_bits < 0:
            raise ValueError("mask_bits must be non-negative")
        if not 0.0 < target_density < 1.0:
            raise ValueError("target_density must be in (0, 1)")
        if not 0.0 < saturation <= 1.0:
            raise ValueError("saturation must be in (0, 1]")
        if max_length < 8 or max_length & (max_length - 1):
            raise ValueError("max_length must be a power of two")
        self._place = place
        self._pooling = str(pooling)
        # Допуск при выборе элемента круга (план v8.3, П.5 — способ пулинга открыт): > 0 — к выбранному элементу круга
        # («strongest», «enumerate») добавляется лучший из остальных кандидатов его области, если его сила не меньше
        # (1 − pooling_tie) от сильнейшего; малая разница сил почерков не меняет код круга целиком. 0 — как было.
        if not 0.0 <= float(pooling_tie) < 1.0:
            raise ValueError("pooling_tie must be in [0, 1)")
        self._tie = float(pooling_tie)
        self._mask = str(mask)
        self._mask_bits = int(mask_bits)
        self._target = float(target_density)
        self._saturation = float(saturation)
        self._max_length = int(max_length)
        self._seed = int(seed)
        self._name = str(name)
        # П.7: окно объединяет около 5–7 понятий; коды остальных элементов окна отбрасываются целиком. None — все.
        self._concepts = None if concepts is None else int(concepts)
        # Раскраска битов, слияние цветов и упаковка кодов окон идут на видеокарте, если она есть; действия целые,
        # результат тот же, что на процессоре.
        self._torch = None
        self._device = None
        if use_gpu:
            try:
                import torch

                if torch.cuda.is_available():
                    self._torch, self._device = torch, torch.device("cuda")
            except ImportError:  # pragma: no cover - без torch всё считается на numpy
                pass
        self._layers = np.asarray(place.detector_layer_indices(), dtype=np.int64)
        # Бит места на круг (автор 2026-11-20: «биты позиции должны перевешивать биты угла»): сегмент места — place_bits
        # копий кода кругов подряд; в копии j у круга свой случайный бит той же цветовой полосы (перестановка полосы,
        # своя у копии). Плотность каждой копии та же, что у одного кода места (§2), а вес места в сходстве кодов окон
        # растёт в place_bits раз. Копия 0 — сам код кругов (place.detector_bits()).
        if int(place_bits) < 1:
            raise ValueError("place_bits must be positive")
        # Грубое связывание (стратегия v3, 2026-11-21, docs/CLUSTERS.md, Б4): содержание кладут только круги места слоёв
        # 0…content_layers−1 (широкие); биты места — от всех слоёв. На окнах во всю цифру узкие круги различают почерк, а не
        # цифру: §3.3.2 «if the goal is… a description that makes sense as a whole, the short-wavelength bits can be
        # discarded»; П.7 — мелкие детали уступают место. None — все слои.
        if content_layers is not None and int(content_layers) < 1:
            raise ValueError("content_layers must be positive or None")
        self._content_layers = None if content_layers is None else int(content_layers)
        self._place_copies = int(place_bits)
        base = np.asarray(place.detector_bits(), dtype=np.int64)
        copies = [base]
        dimension = place.dimensions[0]
        for copy in range(1, self._place_copies):
            rng = np.random.default_rng([self._seed, 3, copy])
            moved = base.copy()
            for layer in range(len(dimension.detector_layers)):
                start, length = dimension.layer_band(layer)
                permutation = rng.permutation(int(length))
                inside = (base >= start) & (base < start + length)
                moved[inside] = start + permutation[base[inside] - start]
            copies.append(moved + copy * int(place.code_length))
        self._place_bits = np.stack(copies, axis=1)  # (круги, копии)
        if int(self._layers.max()) >= self._UNSET:
            raise ValueError("too many detector layers")
        # Маска адреса у детектора одна на все длины кода: от неё берутся младшие биты.
        self._address_masks = np.random.default_rng([self._seed, 0]).integers(
            0, 1 << 30, size=self._layers.shape[0], dtype=np.int64
        )
        self._content_length = 0
        LOGGER.event(
            "position_binding.init",
            section=CONTINUATION_LOCAL_DETECTOR,
            data={
                "name": self._name,
                "detectors": int(self._layers.shape[0]),
                "detectors_per_layer": np.bincount(self._layers).tolist(),
                "place_length": self.place_length,
                "place_bits": self._place_copies,
                "content_layers": self._content_layers,
                "pooling": self._pooling,
                "mask": self._mask,
                "mask_bits": self._mask_bits,
                "note": "детектор позиции — круг на плоскости окна (§4.3) с двумя составляющими (П.5): его "
                "случайный бит входит в код окна операцией OR (§2.2.1), а коды признаков из его области после "
                "пулинга преобразуются его маской через XOR",
            },
        )

    @property
    def place(self) -> Encoder:
        return self._place

    @property
    def place_length(self) -> int:
        """Длина сегмента места: код кругов места × бит места на круг (копий)."""
        return int(self._place.code_length) * self._place_copies

    @property
    def content_length(self) -> int:
        """Длина сегмента содержания; 0 — ещё не подобрана (подбирается первым вызовом encode)."""
        return self._content_length

    def fix_content_length(self, length: int) -> None:
        """Длина сегмента содержания, подобранная раньше по окнам обучающих картинок (§6.6, шаг 3): encode её не
        подбирает. Так кодируется новая картинка обученного уровня — тем же кодом, что в прогоне."""
        if int(length) <= 0:
            raise ValueError("content length must be positive")
        self._content_length = int(length)

    # ------------------------------------------------------------------ маски
    def address_masks(self, length: int) -> "np.ndarray":
        """Маска адреса каждого детектора для сегмента содержания длины length (степень двойки)."""
        return self._address_masks & (int(length) - 1)

    def bit_masks(self, length: int) -> "np.ndarray":
        """Маска-вектор каждого детектора: mask_bits различных случайных битов кода длины length."""
        count = min(self._mask_bits, int(length))
        draws = np.random.default_rng([self._seed, 1, int(length)]).random((self._layers.shape[0], int(length)))
        return np.argsort(draws, axis=1)[:, :count].astype(np.int64)

    def _priority(self, length: int) -> "np.ndarray":
        """Очерёдность битов одного цвета при слиянии цветов: случайная, но одна и та же для всех окон."""
        return np.random.default_rng([self._seed, 2, int(length)]).permutation(int(length)).astype(np.int32)

    # ------------------------------------------------------------------ подготовка входа
    def _code_bits(self, packed: "np.ndarray", length: int) -> tuple["np.ndarray", "np.ndarray"]:
        """Единичные биты каждого кода признака: начало его битов в общем списке и сам список."""
        counts = np.zeros(packed.shape[0], dtype=np.int64)
        parts: list["np.ndarray"] = []
        for start in range(0, packed.shape[0], self.CODE_ROWS):
            rows, bits = np.nonzero(np.unpackbits(packed[start : start + self.CODE_ROWS], axis=1, count=int(length)))
            counts[start : start + self.CODE_ROWS] = np.bincount(rows, minlength=min(self.CODE_ROWS, packed.shape[0] - start))
            parts.append(bits.astype(np.int64))
        starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        return starts, np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

    def _place_detectors(
        self, offsets: "np.ndarray", offset_index: "np.ndarray", unit: float
    ) -> tuple["np.ndarray", "np.ndarray"]:
        """Детекторы позиции каждого смещения: начало его детекторов в общем списке и сам список."""
        # Спрашиваются только встретившиеся смещения: остальные могут лежать вне плоскости мест.
        met = np.unique(offset_index)
        rows, ids = self._place.active_detectors(0, offsets[met] / float(unit))
        order = np.argsort(rows, kind="stable")
        counts = np.zeros(offsets.shape[0], dtype=np.int64)
        counts[met] = np.bincount(rows, minlength=met.shape[0])
        starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        return starts, ids[order].astype(np.int64)

    @staticmethod
    def _expand(owners: "np.ndarray", starts: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
        """Развёртка списков: номер строки-владельца и место в общем списке для каждого элемента его списка."""
        counts = starts[owners + 1] - starts[owners]
        rows = np.repeat(np.arange(owners.shape[0], dtype=np.int64), counts)
        within = np.arange(rows.shape[0], dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
        return rows, starts[owners][rows] + within

    # ------------------------------------------------------------------ код блока окон
    def _block(self, source: dict, first: int, last: int, length: int) -> tuple["np.ndarray", "np.ndarray", dict]:
        """Окна first…last−1: плотные биты места, цвет каждого бита содержания (слой детектора) и счётчики."""
        place, pair, bit, active, stats = self._passes(source, first, last)
        return place, self._paint(source, pair, bit, active, last - first, length), stats

    def _passes(
        self, source: dict, first: int, last: int
    ) -> tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray", dict]:
        """Проходы признаков через детекторы позиции в окнах first…last−1, до применения маски.

        Возвращает плотные биты места, для каждого прошедшего бита признака — пару «окно × детектор» и номер бита,
        список сработавших пар и счётчики. От длины кода это не зависит: при подборе длины считается один раз.
        """
        detectors = int(self._layers.shape[0])
        entries = np.arange(int(source["starts"][first]), int(source["starts"][last]), dtype=np.int64)
        keyholes = last - first
        place = np.zeros((keyholes, self.place_length), dtype=np.uint8)
        empty = np.zeros(0, dtype=np.int64)
        stats = {"entries": int(entries.shape[0]), "members": 0, "pairs": 0}
        if entries.size == 0:
            return place, empty, empty, empty, stats
        # Вхождение элемента в окно × детекторы позиции, в чью область попало его место.
        rows, spot = self._expand(source["offset_index"][entries], source["place_starts"])
        entry, detector = entries[rows], source["place_flat"][spot]
        keyhole = source["groups"][entry] - first
        stats["members"] = int(entry.shape[0])
        # Позиция: бит сработавшего детектора входит в код окна операцией OR (§2.2.1).
        place[keyhole[:, None], self._place_bits[detector]] = 1
        pair = keyhole * detectors + detector
        active = np.unique(pair)
        stats["pairs"] = int(active.shape[0])
        if self._content_layers is not None:
            coarse = self._layers[detector] < self._content_layers
            entry, detector, keyhole, pair = entry[coarse], detector[coarse], keyhole[coarse], pair[coarse]
        if self._concepts is not None:
            # П.7: окно объединяет около 5–7 понятий. В содержание идут только самые сильные элементы окна, коды
            # остальных отбрасываются целиком («отбрасывать … целые коды»); их биты места уже выставлены выше.
            keep = np.isin(entry, self._strongest_entries(source, entry, first))
            entry, detector, keyhole, pair = entry[keep], detector[keep], keyhole[keep], pair[keep]
        candidates = (entry, detector, keyhole, pair)
        if self._pooling == "strongest":
            # Перечисление выбранных: из области детектора берётся один, самый сильный признак.
            order = np.lexsort((entry, -source["strength"][entry], pair))
            head = np.ones(order.shape[0], dtype=bool)
            head[1:] = pair[order][1:] != pair[order][:-1]
            entry, detector, keyhole, pair = entry[order][head], detector[order][head], keyhole[order][head], pair[order][head]
        elif self._pooling == "enumerate":
            pick = self._enumerate(entry, detector, keyhole, pair, source["strength"])
            entry, detector, keyhole, pair = entry[pick], detector[pick], keyhole[pick], pair[pick]
        if self._tie > 0.0 and self._pooling in ("strongest", "enumerate") and source["strength"] is not None:
            entry, detector, keyhole, pair = self._with_ties(candidates, (entry, detector, keyhole, pair), source["strength"])
        # Единичные биты кодов признаков, попавших в область детектора.
        rows, spot = self._expand(source["what_index"][entry], source["bit_starts"])
        bit = source["bit_flat"][spot]
        if self._pooling == "min":
            # Минимум по битам: остаются биты, общие всем признакам области детектора.
            pairs, member, size = np.unique(pair, return_inverse=True, return_counts=True)
            keys, hits = np.unique(member.reshape(-1)[rows] * int(source["what_length"]) + bit, return_counts=True)
            common = hits == size[keys // int(source["what_length"])]
            pair, bit = pairs[keys[common] // int(source["what_length"])], keys[common] % int(source["what_length"])
        else:
            pair = pair[rows]
        return place, pair, bit, active, stats

    def _with_ties(self, candidates: tuple, chosen: tuple, strength: "np.ndarray") -> tuple:
        """План v8.3: к выбранному элементу каждого круга (пары окно–круг) — лучший из остальных кандидатов области, если
        его сила не меньше (1 − tie) от сильнейшего кандидата. Возвращает выбранные и добавленные вхождения."""
        entry, detector, keyhole, pair = candidates
        if entry.size == 0 or chosen[0].size == 0:
            return chosen
        level = strength[entry]
        order = np.lexsort((entry, -level, pair))
        e, d, k, p, lv = entry[order], detector[order], keyhole[order], pair[order], level[order]
        head = np.ones(p.shape[0], dtype=bool)
        head[1:] = p[1:] != p[:-1]
        starts = np.flatnonzero(head)
        ends = np.append(starts[1:], p.shape[0])
        best = lv[starts]
        # Выбранный элемент каждого круга: пары выбранных отсортированы так же, как кандидаты (по номеру пары).
        c_entry, c_pair = chosen[0], chosen[3]
        c_order = np.argsort(c_pair, kind="stable")
        c_pair, c_entry = c_pair[c_order], c_entry[c_order]
        at = np.searchsorted(c_pair, p[starts])
        picked = np.where((at < c_pair.shape[0]) & (c_pair[np.minimum(at, c_pair.shape[0] - 1)] == p[starts]),
                          c_entry[np.minimum(at, c_pair.shape[0] - 1)], -1)
        # Лучший из остальных: первый кандидат круга, если его не взяли, иначе второй.
        first_taken = e[starts] == picked
        second = np.where(first_taken, starts + 1, starts)
        valid = second < ends
        second = np.minimum(second, p.shape[0] - 1)
        add = valid & (e[second] != picked) & (lv[second] >= (1.0 - self._tie) * best)
        rows = second[add]
        return (np.concatenate([chosen[0], e[rows]]), np.concatenate([chosen[1], d[rows]]),
                np.concatenate([chosen[2], k[rows]]), np.concatenate([chosen[3], p[rows]]))

    def _enumerate(self, entry: "np.ndarray", detector: "np.ndarray", keyhole: "np.ndarray", pair: "np.ndarray",
                   strength: "np.ndarray") -> "np.ndarray":
        """Перечисление выбранных (П.5; задача #17): в каждом окне и слое круги места по очереди — от круга с самым
        сильным элементом области к слабым — берут сильнейший элемент своей области, ещё не взятый другим кругом этого
        слоя; если все взяты — свой сильнейший (как «strongest»). Возвращает номера выбранных вхождений (по одному на
        круг). Ничьи — по номеру вхождения, порядок полностью определён."""
        if entry.size == 0:
            return np.zeros(0, dtype=np.int64)
        level = strength[entry]
        layer = self._layers[detector]
        # Кандидаты каждого круга — по убыванию силы; круги группы (окно, слой) — по силе своего лучшего кандидата.
        order = np.lexsort((entry, -level, pair))
        head = np.ones(order.shape[0], dtype=bool)
        head[1:] = pair[order][1:] != pair[order][:-1]
        starts = np.flatnonzero(head)
        ends = np.append(starts[1:], order.shape[0])
        best = level[order][starts]
        groups = keyhole[order][starts].astype(np.int64) * (int(layer.max()) + 1) + layer[order][starts]
        circles = np.lexsort((pair[order][starts], -best, groups))
        picks = np.empty(starts.shape[0], dtype=np.int64)
        taken: set = set()
        group = None
        for index in circles:
            if groups[index] != group:
                group, taken = groups[index], set()
            span = order[starts[index]:ends[index]]
            choice = next((row for row in span if int(entry[row]) not in taken), span[0])
            taken.add(int(entry[choice]))
            picks[index] = choice
        return np.sort(picks)

    def _strongest_entries(self, source: dict, entry: "np.ndarray", first: int) -> "np.ndarray":
        """Вхождения-понятия окна (П.7): в каждом окне — self._concepts самых сильных элементов (ничья — по номеру
        вхождения). Без уровней активации все элементы равны, и берутся первые по порядку."""
        mine = np.unique(entry)
        keyhole = source["groups"][mine] - first
        strength = source["strength"][mine] if source["strength"] is not None else np.zeros(mine.shape[0])
        order = np.lexsort((mine, -strength, keyhole))
        ranked = keyhole[order]
        rank = np.arange(order.shape[0]) - np.searchsorted(ranked, ranked, side="left")
        return mine[order][rank < self._concepts]

    def _paint(
        self, source: dict, pair: "np.ndarray", bit: "np.ndarray", active: "np.ndarray", keyholes: int, length: int
    ) -> "np.ndarray":
        """Маска детектора и цвет: для каждого окна — цвет (слой детектора) каждого бита сегмента содержания."""
        detectors = int(self._layers.shape[0])
        colour = np.full((keyholes, int(length)), self._UNSET, dtype=np.uint8)
        if pair.size == 0 and (self._mask == "address" or active.size == 0):
            return colour
        if self._mask == "address":
            # XOR маски с адресом бита: перестановка битов кода признака, своя у каждого детектора.
            address = bit ^ source["address_masks"][pair % detectors]
        else:
            # XOR маски с самим кодом: симметрическая разность битов кода и битов маски в области детектора.
            width = int(source["what_length"])
            table = source["bit_masks"]
            masks = (active[:, None] * width + table[active % detectors]).reshape(-1)
            keys, hits = np.unique(np.concatenate([np.unique(pair * width + bit), masks]), return_counts=True)
            odd = keys[hits == 1]
            pair, address = odd // width, odd % width
        keyhole, layer = pair // detectors, self._layers[pair % detectors]
        # Цвет бита — слой детектора (§3.2): от узких слоёв к широким, чтобы у общего бита остался цвет широкого.
        for number in np.unique(layer)[::-1]:
            mine = layer == number
            colour[keyhole[mine], address[mine]] = number
        return colour

    def _on_device(self, values):  # noqa: ANN001, ANN202
        """Массив numpy — на видеокарту (тензор — как есть)."""
        torch = self._torch
        return values if torch.is_tensor(values) else torch.as_tensor(np.ascontiguousarray(values), device=self._device)

    def _paint_device(self, source: dict, pair, bit, keyholes: int, length: int):  # noqa: ANN001, ANN202
        """То же, что _paint для маски адреса, на видеокарте: тензор цветов «окна × биты содержания». pair и bit —
        массивы numpy или уже тензоры на видеокарте (подбор длины кода раскрашивает один кусок окон восемью длинами —
        данные куска передаются один раз). Адреса, слои и номера клеток считаются на видеокарте теми же целочисленными
        действиями, что прежде на процессоре (2026-11-25: numpy здесь занимал половину времени кодов окон)."""
        torch = self._torch
        detectors = int(self._layers.shape[0])
        colour = torch.full((keyholes, int(length)), self._UNSET, dtype=torch.uint8, device=self._device)
        if int(pair.shape[0]) == 0:
            return colour
        pair, bit = self._on_device(pair), self._on_device(bit)
        detector = pair % detectors
        # XOR маски с адресом бита: перестановка битов кода признака, своя у каждого детектора.
        cell = (pair // detectors) * int(length) + (bit ^ self._on_device(source["address_masks"])[detector])
        layers = self.__dict__.get("_layers_device")
        if layers is None:
            layers = self._layers_device = torch.as_tensor(self._layers, device=self._device)
        layer_device = layers[detector]
        del detector
        flat = colour.view(-1)
        # Цвет бита — слой детектора (§3.2): от узких слоёв к широким, чтобы у общего бита остался цвет широкого. Слои
        # без бит пропускаются — тот же порядок записей, что по встретившимся слоям.
        for number in range(int(self._layers.max()), -1, -1):
            mine = cell[layer_device == number]
            if int(mine.shape[0]):
                flat[mine] = number
        return colour

    def _merge_device(self, colour, budget: int, priority) -> int:  # noqa: ANN001
        """То же, что _merge, на видеокарте: в окнах сверх бюджета остаются биты самых широких слоёв (§3.3)."""
        torch = self._torch
        over = torch.nonzero((colour != self._UNSET).sum(dim=1) > budget).view(-1)
        if over.numel() == 0:
            return 0
        # Сначала широкие слои; внутри слоя — по общей для всех окон очерёдности битов. Ключи в строке различны,
        # поэтому budget наименьших ключей — один и тот же набор при любом способе отбора.
        keys = colour[over].to(torch.int32) * int(colour.shape[1]) + priority[None, :]
        kept = torch.topk(keys, budget, dim=1, largest=False).indices
        keep = torch.zeros(keys.shape, dtype=torch.bool, device=self._device)
        keep.scatter_(1, kept, True)
        trimmed = colour[over]
        trimmed[~keep] = self._UNSET
        colour[over] = trimmed
        return int(over.numel())

    def _merge(self, colour: "np.ndarray", budget: int, priority: "np.ndarray") -> int:
        """Слияние цветов (§3.3): в окнах, где бит больше бюджета, остаются биты самых широких слоёв."""
        count = (colour != self._UNSET).sum(axis=1)
        over = np.flatnonzero(count > budget)
        if over.size == 0:
            return 0
        # Сначала широкие слои; внутри слоя — по общей для всех окон очерёдности битов.
        keys = colour[over].astype(np.int32) * np.int32(colour.shape[1]) + priority[None, :]
        kept = np.argpartition(keys, budget - 1, axis=1)[:, :budget]
        keep = np.zeros(keys.shape, dtype=bool)
        np.put_along_axis(keep, kept, True, axis=1)
        trimmed = colour[over]
        trimmed[~keep] = self._UNSET
        colour[over] = trimmed
        return int(over.shape[0])

    def _fit_length(self, source: dict, keyholes: int) -> int:
        """Наименьшая длина сегмента содержания (степень двойки) со средней плотностью не выше заданной (§2)."""
        length = 8
        while length < int(source["what_length"]):
            length *= 2
        if self._mask == "bits":
            # Маска-вектор живёт в битах самого кода признака: длина сегмента равна длине кода.
            return int(source["what_length"])
        # Длины-кандидаты по возрастанию; проходы признаков через детекторы считаются один раз на блок окон,
        # для каждой длины заново применяется только маска.
        lengths = [length]
        while lengths[-1] < self._max_length:
            lengths.append(lengths[-1] * 2)
        ones = dict.fromkeys(lengths, 0)
        step = max(1, self.CHUNK_CELLS // (self.place_length + lengths[-1]))
        for first in range(0, keyholes, step):
            last = min(keyholes, first + step)
            _, pair, bit, active, _ = self._passes(source, first, last)
            if self._device is not None:
                pair, bit = self._on_device(pair), self._on_device(bit)  # кусок — на видеокарту один раз на все длины
            for size in lengths:
                source["address_masks"] = self.address_masks(size)
                if self._device is not None:
                    ones[size] += int((self._paint_device(source, pair, bit, last - first, size) != self._UNSET).sum().item())
                else:
                    ones[size] += int((self._paint(source, pair, bit, active, last - first, size) != self._UNSET).sum())
        tried: list[tuple[int, float]] = []
        for length in lengths:
            density = ones[length] / (max(1, keyholes) * length)
            tried.append((length, density))
            if density <= self._target:
                break
        LOGGER.event(
            "position_binding.code_size",
            section=SPARSE_BIT_VECTORS,
            data={
                "name": self._name,
                "calibration_keyholes": int(keyholes),
                "target_density": self._target,
                "tried": [{"length": size, "density": value} for size, value in tried],
                "content_length": length,
                "max_length": self._max_length,
                "note": "длина сегмента содержания — наименьшая степень двойки, при которой средняя плотность кодов "
                "окон не выше заданной (§2: разрежённые коды работают при плотности не выше 25 %)",
            },
        )
        return length

    # ------------------------------------------------------------------ кодирование
    def encode(
        self,
        what_packed: "np.ndarray",
        what_length: int,
        what_index: "np.ndarray",
        offsets: "np.ndarray",
        offset_index: "np.ndarray",
        groups: "np.ndarray",
        group_count: int,
        *,
        unit: float,
        strength: "np.ndarray | None" = None,
        calibration: int | None = None,
    ) -> KeyholeCodes:
        """Коды окон 0…group_count−1.

        Вхождение i — признак с кодом what_packed[what_index[i]] на смещении offsets[offset_index[i]] от центра
        окна groups[i] (по возрастанию); strength[i] — уровень его активации (нужен пулингу «strongest»). Длина
        сегмента содержания подбирается один раз, по первым calibration окнам.
        """
        what_packed = np.ascontiguousarray(what_packed, dtype=np.uint8)
        what_index = np.asarray(what_index, dtype=np.int64)
        offset_index = np.asarray(offset_index, dtype=np.int64)
        groups = np.asarray(groups, dtype=np.int64)
        if unit <= 0:
            raise ValueError("unit must be positive")
        if what_packed.ndim != 2 or what_packed.shape[1] != -(-int(what_length) // 8):
            raise ValueError("what codes must be rows × ceil(what_length / 8) packed bytes")
        if not (what_index.shape == offset_index.shape == groups.shape) or what_index.ndim != 1:
            raise ValueError("what_index, offset_index and groups must be one-dimensional arrays of the same length")
        if what_index.size == 0:
            raise ValueError("no elements")
        if np.any(np.diff(groups) < 0) or groups[0] < 0 or groups[-1] >= group_count:
            raise ValueError("groups must be ascending keyhole numbers below group_count")
        if self._pooling in ("strongest", "enumerate") and strength is None:
            raise ValueError(f"pooling '{self._pooling}' needs the activation level of every element")
        offsets = np.asarray(offsets, dtype=np.float64).reshape(-1, 2)
        bit_starts, bit_flat = self._code_bits(what_packed, int(what_length))
        place_starts, place_flat = self._place_detectors(offsets, offset_index, float(unit))
        source = {
            "what_length": int(what_length),
            "what_index": what_index,
            "offset_index": offset_index,
            "groups": groups,
            "starts": np.searchsorted(groups, np.arange(int(group_count) + 1)),
            "strength": None if strength is None else np.asarray(strength, dtype=np.float64),
            "bit_starts": bit_starts,
            "bit_flat": bit_flat,
            "place_starts": place_starts,
            "place_flat": place_flat,
        }
        if self._mask == "bits":
            source["bit_masks"] = self.bit_masks(int(what_length))
        if self._content_length == 0:
            sample = int(group_count) if calibration is None else max(1, min(int(calibration), int(group_count)))
            self._content_length = self._fit_length(source, sample)
        length = self._content_length
        if self._mask == "address":
            if length < int(what_length):
                raise ValueError("content length is shorter than the feature code: use a new encoder for these codes")
            source["address_masks"] = self.address_masks(length)
        elif length != int(what_length):
            raise ValueError("mask 'bits' keeps the feature code length: use a new encoder for these codes")
        budget = max(1, int(self._saturation * length))
        priority = self._priority(length)
        total = self.place_length + length
        packed = np.zeros((int(group_count), -(-total // 8)), dtype=np.uint8)
        step = max(1, self.CHUNK_CELLS // total)
        layers = int(self._layers.max()) + 1
        raw = np.zeros(layers, dtype=np.int64)
        kept = np.zeros(layers, dtype=np.int64)
        entries = members = pairs = place_ones = trimmed = 0
        on_device = self._device is not None and self._mask == "address"
        if on_device:
            torch = self._torch
            priority_device = torch.as_tensor(priority, device=self._device)
            weights = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=self._device)
        for first in range(0, int(group_count), step):
            last = min(int(group_count), first + step)
            if on_device:
                place, pair, bit, _, stats = self._passes(source, first, last)
                colour = self._paint_device(source, pair, bit, last - first, length)
                raw += torch.bincount(colour[colour != self._UNSET].to(torch.int64), minlength=layers)[:layers].cpu().numpy()
                trimmed += self._merge_device(colour, budget, priority_device)
                kept += torch.bincount(colour[colour != self._UNSET].to(torch.int64), minlength=layers)[:layers].cpu().numpy()
                # Упаковка по 8 бит в байт, старший бит первым — как np.packbits.
                bits = torch.cat([torch.as_tensor(place, device=self._device) > 0, colour != self._UNSET], dim=1)
                spare = (-bits.shape[1]) % 8
                if spare:
                    bits = torch.nn.functional.pad(bits, (0, spare))
                packed[first:last] = (
                    (bits.view(last - first, -1, 8).to(torch.uint8) * weights).sum(dim=2, dtype=torch.uint8).cpu().numpy()
                )
            else:
                place, colour, stats = self._block(source, first, last, length)
                raw += np.bincount(colour[colour != self._UNSET], minlength=layers)[:layers]
                trimmed += self._merge(colour, budget, priority)
                kept += np.bincount(colour[colour != self._UNSET], minlength=layers)[:layers]
                packed[first:last] = np.packbits(np.concatenate([place, colour != self._UNSET], axis=1), axis=1)
            place_ones += int(place.sum())
            entries, members, pairs = entries + stats["entries"], members + stats["members"], pairs + stats["pairs"]
        keyholes = max(1, int(group_count))
        LOGGER.event(
            "position_binding.merge",
            section=COLOR_MERGE,
            data={
                "name": self._name,
                "budget_bits": budget,
                "saturation": self._saturation,
                "keyholes_over_budget": trimmed / keyholes,
                "bits_by_layer": (raw / keyholes).tolist(),
                "kept_by_layer": (kept / keyholes).tolist(),
                "dropped_share": 1.0 - float(kept.sum()) / max(1, int(raw.sum())),
                "note": "слияние цветов: цвет бита — слой детектора позиции, который его дал; в окне, где бит "
                "содержания больше бюджета, уходят биты узких слоёв (ближний порядок), остаются широкие (§3.3.2); "
                "так объединение понятий не перенасыщается, а код окна получает свойства хеширования (П.7)",
            },
        )
        LOGGER.event(
            "position_binding.density",
            section=CONTINUATION_MERGING,
            data={
                "name": self._name,
                "keyholes": int(group_count),
                "what_bits_mean": float((bit_starts[1:] - bit_starts[:-1])[what_index].mean()),
                "content_bits_per_element": float(raw.sum()) / max(1, entries),
                "content_bits_per_detector_pass": float(raw.sum()) / max(1, members),
                "content_density_before_merge": float(raw.sum()) / (keyholes * length),
                "content_density": float(kept.sum()) / (keyholes * length),
                "mask": self._mask,
                "note": "проверка плотности после XOR (П.5: «необходимо проверять на выбранных масках и входных "
                "кодах»): сколько бит содержания даёт один признак и один проход признака через детектор; у маски "
                "адреса проход сохраняет число единиц кода, у маски-вектора добавляет вес маски",
            },
        )
        LOGGER.event(
            "position_binding.encode",
            section=CONTINUATION_BINDING,
            data={
                "name": self._name,
                "keyholes": int(group_count),
                "elements": entries,
                "elements_per_keyhole": entries / keyholes,
                "detectors_per_element": members / max(1, entries),
                "active_detectors_per_keyhole": pairs / keyholes,
                "elements_per_active_detector": members / max(1, pairs),
                "what_codes": int(what_packed.shape[0]),
                "what_length": int(what_length),
                "what_bits_mean": float((bit_starts[1:] - bit_starts[:-1])[what_index].mean()),
                "pooling": self._pooling,
                "mask": self._mask,
                "place_length": self.place_length,
                "content_length": length,
                "code_length": total,
                "place_bits_per_keyhole": place_ones / keyholes,
                "content_bits_per_keyhole": float(kept.sum()) / keyholes,
                "place_density": place_ones / (keyholes * self.place_length),
                "content_density": float(kept.sum()) / (keyholes * length),
                "unit": float(unit),
                "note": "связывание признака с позицией (П.4): код окна — OR вкладов сработавших детекторов позиции "
                "(§2.2.1): бит места детектора и коды признаков из его области, преобразованные его маской; "
                "признак записан в координатах окна, а не картинки (§7.1.4.1)",
            },
        )
        return KeyholeCodes(packed=packed, code_length=total, place_length=self.place_length, content_length=length)
