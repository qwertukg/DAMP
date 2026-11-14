"""Начальная расстановка большой раскладки по раскладке случайного подмножества её точек (вне статьи).

Статья начинает раскладку со случайной расстановки (§5.2) и собирает глобальный порядок шагами с большим
радиусом выбора пар (§5.3, §7.1.7); энергия пары — сумма по точкам V (§5.5). На миллионах точек эти шаги
стоят N·r² и остаются посильными только при оценке энергий по подмножеству (§5.9.4), шум которой на больших
радиусах портит глобальный порядок. Здесь глобальный порядок берётся из точной раскладки подмножества:

1. случайное подмножество B точек («затравка») раскладывается обычным алгоритмом статьи целиком;
2. каждая точка V получает «родителя» — самую похожую на неё точку затравки (мера сходства §2.2.4);
3. сетка затравки растягивается до сетки большой раскладки: ячейке затравки отвечает блок ячеек, цель точки —
   случайное место блока её родителя; точки расставляются по ячейкам блоков взаимно-однозначно делением
   точек и ячеек пополам по медианам координат (порядок сохраняется, сдвиги — доли блока);
4. дальше идёт обычная раскладка, но с малого начального радиуса: порядок крупнее блока уже есть.

Затравка — та же идея, что подмножество B для оценки энергий (§5.9.4), применённая к порядку раскладки.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from damp.article_refs import OPTIM_SUBSET, OUTSIDE_ARTICLE, SIMILARITY_MEASURES
from damp.layout.code_table import LayoutCodeTable
from damp.layout.device_kernel import DeviceKernel
from damp.logging import LOGGER


class SeededPlacement:
    """Расстановка точек таблицы кодов по раскладке случайного подмножества этих точек."""

    # Предел элементов (точки × родители) на один проход поиска родителей на устройстве.
    MATCH_ELEMENTS = 64_000_000
    # Число уровней квантования координаты в ключе сортировки деления пополам (20 бит на координату).
    KEY_LEVELS = 1 << 20
    # То же для расчёта на процессоре (numpy): в памяти держится порция × родители × слова кода.
    MATCH_ELEMENTS_CPU = 2_000_000
    # Поиск родителя в два приёма, когда родителей больше стольких (многоуровневая затравка): перебор
    # «каждая точка × каждый родитель» на сотнях тысяч родителей занял бы десятки минут. Родители делятся
    # на списки по самой похожей из PIVOTS опорных точек (случайные родители); точке подбираются PROBES самых
    # похожих опорных точек, и родитель ищется в их списках (до LIST_MAX случайных родителей в списке).
    # 882 тыс. точек, 163 тыс. родителей: у 99.9 % точек найден родитель с тем же сходством, что у перебора
    # (среднее сходство с родителем 0.9943 в обоих случаях).
    PIVOT_MIN_PARENTS = 65536
    PIVOTS = 8192
    PROBES = 8
    LIST_MAX = 128
    # Предел элементов (точки × кандидаты в родители) на один проход второго приёма: операциями torch
    # (на элемент — слова кода кандидата) и ядром устройства (на элемент — номер кандидата и число общих бит).
    CANDIDATE_ELEMENTS = 2_000_000
    CANDIDATE_ELEMENTS_KERNEL = 16_000_000
    _CANDIDATE_KERNELS: dict[int, Any] = {}
    # Место точки внутри блока родителя (режим статьи): точка сдвигается от середины блока к другим похожим
    # на неё родителям — взвешенное по сходству среднее смещение до NEIGHBOUR_PARENTS самых похожих родителей,
    # лежащих не дальше NEIGHBOUR_RADIUS ячеек затравки от самого похожего, — и получает случайную добавку:
    # нормальную, со стандартным отклонением NEIGHBOUR_SPREAD блока по каждой оси. Порядок внутри блока тем
    # самым продолжает порядок раскладки затравки. Добавка изотропна и шире блока: если точки родителя
    # занимают ровно его блок, карта уровня — это карта затравки, увеличенная квадратами, и «лесенка» её
    # границ превращается в прямые отрезки вдоль осей сетки длиной в блок и больше. Измерено на 10 000 цифр
    # (уровни 5000 → 66 тыс. → 883 тыс., ближняя фаза уровня без ранней отсечки; доля длины границ областей
    # на прямых отрезках ≥ 8 / ≥ 16 ячеек и Ē(3)): равномерно в 0.75 блока — 9.4 % / 1.2 % и 0.888;
    # σ = 0.5 — 5.5 % / 0.22 % и 0.889; σ = 0.7 — 5.1 % / 0.04 % и 0.887; σ = 1.0 — 4.5 % / 0.13 % и 0.881
    # (у точных раскладок 100–1000 цифр — 1.5–2 % / 0).
    NEIGHBOUR_PARENTS = 8
    NEIGHBOUR_RADIUS = 3.0
    NEIGHBOUR_SPREAD = 0.7
    # Точка затравки годится в родители, если её связная область раскладки затравки (связность по 8 соседним
    # ячейкам) содержит не меньше этой доли точек затравки: одиночные точки и мелкие островки вне общего
    # массива утянули бы за собой по блоку точек каждая.
    PARENT_COMPONENT_FRACTION = 0.01

    def __init__(self, table: LayoutCodeTable, *, seed_count: int, seed: int, similarity: str = "cosine") -> None:
        if similarity not in ("cosine", "jaccard"):
            raise ValueError("similarity must be 'cosine' or 'jaccard'")
        if not 0 < seed_count < table.count:
            raise ValueError("seed_count must be positive and smaller than the number of points")
        self._table = table
        self._measure = similarity
        # Свой генератор: потоки случайных чисел самой раскладки не меняются.
        self._rng = np.random.default_rng([int(seed), 2])
        self._indices = np.sort(self._rng.choice(table.count, size=int(seed_count), replace=False))
        LOGGER.event(
            "layout.seed.subset",
            section=OPTIM_SUBSET,
            data={
                "points": table.count,
                "seed_points": int(seed_count),
                "note": "случайное подмножество B точек раскладывается целиком; остальные точки ставятся возле "
                "самых похожих точек B (расширение §5.9.4: подмножество задаёт глобальный порядок); пока "
                "раскладывается B, на карте хода раскладки только его точки — остальные появляются после расстановки",
            },
        )

    @property
    def seed_indices(self) -> "np.ndarray":
        """Номера точек затравки в таблице (по возрастанию)."""
        return self._indices

    @property
    def seed_count(self) -> int:
        return int(self._indices.size)

    def seed_table(self) -> LayoutCodeTable:
        """Коды точек затравки (в порядке номеров точек)."""
        return self._table.take(self._indices)

    def block_side(self) -> float:
        """Сторона блока ячеек, приходящегося на одну точку затравки: √(N / затравка)."""
        return math.sqrt(self._table.count / float(self.seed_count))

    # ------------------------------------------------------------------ родители
    @staticmethod
    def components(occupied: "np.ndarray") -> "np.ndarray":
        """Метки связных областей занятых ячеек (связность по 8 соседям): наименьший линейный номер ячейки
        области; у пустых ячеек — height·width."""
        height, width = occupied.shape
        empty = height * width
        try:
            # scipy размечает области за один проход; метка области приводится к тому же виду —
            # наименьшему линейному номеру её ячейки (сетка 774²: 0.05 с вместо 15 с распространения меток).
            from scipy import ndimage

            marked, regions = ndimage.label(occupied, structure=np.ones((3, 3), dtype=bool))
            if regions == 0:
                return np.full((height, width), empty, dtype=np.int64)
            linear = np.arange(empty, dtype=np.int64).reshape(height, width)
            lowest = np.asarray(ndimage.minimum(linear, marked, index=np.arange(1, regions + 1)), dtype=np.int64)
            return np.where(occupied, np.concatenate(([empty], lowest))[marked], empty)
        except ImportError:
            pass
        labels = np.where(occupied, np.arange(empty, dtype=np.int64).reshape(height, width), empty)
        while True:
            padded = np.pad(labels, 1, constant_values=empty)
            lowest = labels
            for dy in (0, 1, 2):
                for dx in (0, 1, 2):
                    lowest = np.minimum(lowest, padded[dy : dy + height, dx : dx + width])
            lowest = np.where(occupied, lowest, empty)
            if np.array_equal(lowest, labels):
                return labels
            labels = lowest

    def _parent_mask(self, seed_cells: "np.ndarray", seed_shape: tuple[int, int]) -> "np.ndarray":
        """Точки затравки, годные в родители: точки больших связных областей её раскладки."""
        occupied = np.zeros(seed_shape, dtype=bool)
        occupied[seed_cells[:, 0], seed_cells[:, 1]] = True
        labels = self.components(occupied)[seed_cells[:, 0], seed_cells[:, 1]]
        values, inverse, sizes = np.unique(labels, return_inverse=True, return_counts=True)
        large = sizes >= max(1.0, self.PARENT_COMPONENT_FRACTION * self.seed_count)
        if not bool(large.any()):
            large = sizes == sizes.max()
        return large[inverse.reshape(-1)]

    def _similarity(self, common: Any, ones_rows: Any, ones_cols: Any, xp: Any) -> Any:
        """Сходство по числу общих бит и числу единиц кодов (§2.2.4); нулевой код — 0."""
        if self._measure == "cosine":
            denominator = xp.sqrt(ones_rows * ones_cols)
        else:
            denominator = ones_rows + ones_cols - common
        return xp.where(denominator > 0, common / xp.where(denominator > 0, denominator, 1.0), 0.0)

    def _distinct_codes(self) -> tuple["np.ndarray", "np.ndarray"]:
        """Различные коды таблицы: номер точки-представителя каждого кода и номер кода каждой точки.

        Коды сравниваются по 64-битной свёртке слов, затем проверяется, что код каждой точки побитово равен
        коду её представителя; если свёртки разных кодов совпали, различные коды находит сортировка самих
        кодов (на 5.3 млн точек — 13 с вместо 2.5 с). Каждое слово сначала перемешивается (умножения
        и сдвиги по модулю 2⁶⁴), и только потом слова складываются с нечётными множителями: сумма самих слов
        с множителями совпадает у кодов, которые отличаются только старшими битами слов.
        """
        words = self._table.words64()
        multipliers = np.random.default_rng(0x5EED).integers(0, 2**63, size=words.shape[1], dtype=np.uint64) * np.uint64(2)
        multipliers += np.uint64(1)
        mixed = words ^ (words >> np.uint64(33))
        mixed *= np.uint64(0xFF51AFD7ED558CCD)
        mixed ^= mixed >> np.uint64(33)
        mixed *= np.uint64(0xC4CEB9FE1A85EC53)
        mixed ^= mixed >> np.uint64(33)
        mixed *= multipliers[None, :]
        keys = mixed.sum(axis=1, dtype=np.uint64)
        del mixed
        _, representatives, code_of_point = np.unique(keys, return_index=True, return_inverse=True)
        code_of_point = code_of_point.reshape(-1)
        if not bool((words == words[representatives[code_of_point]]).all()):
            _, representatives, code_of_point = np.unique(words, axis=0, return_index=True, return_inverse=True)
            code_of_point = code_of_point.reshape(-1)
        return representatives.astype(np.int64), code_of_point.astype(np.int64)

    def _match_device(
        self, parents: "np.ndarray", top: int = 1, queries: "np.ndarray | None" = None
    ) -> "tuple[np.ndarray, np.ndarray] | None":
        """Самые похожие родители точек queries (None — всех точек) на CUDA: номера (строки × top, по убыванию
        сходства) и сходства; None — устройство недоступно.

        Число общих бит (§2.2.4) — произведение кодов порции точек на коды родителей (целые до длины кода,
        в float32 точны); биты порции распаковываются из упакованных кодов таблицы.
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return None
        except Exception:  # noqa: BLE001
            return None
        table = self._table
        device = torch.device("cuda")
        with torch.no_grad():
            # Коды — 64-битными словами на устройстве; биты порции распаковываются там же.
            words = torch.as_tensor(np.ascontiguousarray(table.words64()).view(np.int64), device=device)
            bit = np.arange(table.code_length, dtype=np.int64)
            # Бит i лежит в байте i // 8 (старший бит байта — меньший номер), байты слова — от младшего.
            word_of_bit = torch.as_tensor(bit // 64, device=device)
            shift_of_bit = torch.as_tensor(8 * ((bit // 8) % 8) + (7 - bit % 8), device=device).unsqueeze(0)

            def codes(rows: Any) -> Any:
                return ((rows[:, word_of_bit] >> shift_of_bit) & 1).to(torch.float32)

            ones = torch.as_tensor(table.ones.astype(np.float32), device=device)
            parent_device = torch.as_tensor(parents, device=device)
            rows = (
                torch.arange(table.count, device=device)
                if queries is None
                else torch.as_tensor(np.asarray(queries, dtype=np.int64), device=device)
            )
            row_count = int(rows.numel())
            top = max(1, min(int(top), int(parents.size)))
            best = torch.empty((row_count, top), dtype=torch.int64, device=device)
            best_similarity = torch.empty((row_count, top), dtype=torch.float32, device=device)
            if parents.size > self.PIVOT_MIN_PARENTS:
                self._match_pivots(torch, words, codes, ones, parent_device, rows, best, best_similarity)
            else:
                parent_codes = codes(words[parent_device]).T.contiguous()
                parent_ones = ones[parent_device].unsqueeze(0)
                chunk = max(1, self.MATCH_ELEMENTS // max(1, parents.size))
                for start in range(0, row_count, chunk):
                    stop = min(row_count, start + chunk)
                    part = rows[start:stop]
                    similarity = self._similarity(
                        codes(words[part]) @ parent_codes, ones[part].unsqueeze(1), parent_ones, torch
                    )
                    if top == 1:
                        best[start:stop, 0] = similarity.argmax(dim=1)
                        best_similarity[start:stop] = torch.gather(similarity, 1, best[start:stop])
                    else:
                        best_similarity[start:stop], best[start:stop] = torch.topk(similarity, top, dim=1)
                del parent_codes
            result = best.cpu().numpy(), best_similarity.cpu().numpy()
        del best, best_similarity, words
        torch.cuda.empty_cache()
        return result

    def _match_pivots(
        self, torch: Any, words: Any, codes: Any, ones: Any, parents: Any, rows: Any, best: Any, best_similarity: Any
    ) -> None:
        """Самые похожие родители точек rows в два приёма (вне статьи; ср. подмножество §5.9.4): пишет
        номера и сходства в best, best_similarity (строки × top).

        1. PIVOTS случайных родителей — опорные точки; каждый родитель попадает в список самой похожей на него
           опорной точки (в списке — до LIST_MAX случайных родителей).
        2. Для точки находятся PROBES самых похожих опорных точек (произведение кодов, как в полном переборе),
           а родитель выбирается из их списков по числу общих бит кодов (§2.2.4).
        Мера сходства та же; родитель — самый похожий среди кандидатов, а не среди всех родителей.
        """
        device = words.device
        count = int(rows.numel())
        parent_count = int(parents.numel())
        pivots = min(self.PIVOTS, parent_count)
        rng = self._rng
        pivot_slots = torch.as_tensor(np.sort(rng.choice(parent_count, size=pivots, replace=False)), device=device)
        pivot_points = parents[pivot_slots]
        pivot_codes = codes(words[pivot_points]).T.contiguous()
        pivot_ones = ones[pivot_points].unsqueeze(0)
        # Приём 1: список каждого родителя — по самой похожей опорной точке.
        assign = torch.empty(parent_count, dtype=torch.int64, device=device)
        chunk = max(1, self.MATCH_ELEMENTS // pivots)
        for start in range(0, parent_count, chunk):
            listed = parents[start : start + chunk]
            similarity = self._similarity(
                codes(words[listed]) @ pivot_codes, ones[listed].unsqueeze(1), pivot_ones, torch
            )
            assign[start : start + chunk] = similarity.argmax(dim=1)
        # Родители в случайном порядке, затем устойчиво по спискам: первые LIST_MAX списка — случайная выборка.
        shuffled = torch.as_tensor(rng.permutation(parent_count), device=device)
        sorted_assign, order = torch.sort(assign[shuffled], stable=True)
        members = shuffled[order]
        sizes = torch.bincount(assign, minlength=pivots)
        starts = torch.cumsum(sizes, dim=0) - sizes
        rank = torch.arange(parent_count, device=device) - starts[sorted_assign]
        list_max = int(min(self.LIST_MAX, int(sizes.max().item())))
        kept = rank < list_max
        lists = torch.full((pivots, list_max), -1, dtype=torch.int64, device=device)
        lists[sorted_assign[kept], rank[kept]] = members[kept]
        dropped = int(parent_count - int(kept.sum().item()))
        # Приём 2: PROBES самых похожих опорных точек и лучший родитель из их списков.
        parent_words = words[parents]
        parent_ones = ones[parents]
        probes = min(self.PROBES, pivots)
        candidates_per_point = probes * list_max
        kernel = self._candidate_kernel(int(words.shape[1]))
        candidate_elements = self.CANDIDATE_ELEMENTS if kernel is None else self.CANDIDATE_ELEMENTS_KERNEL
        chunk = max(1, min(self.MATCH_ELEMENTS // pivots, candidate_elements // candidates_per_point))
        top = int(best.shape[1])
        for start in range(0, count, chunk):
            stop = min(count, start + chunk)
            part = rows[start:stop]
            point_words = words[part]
            point_ones = ones[part].unsqueeze(1)
            similarity = self._similarity(codes(point_words) @ pivot_codes, point_ones, pivot_ones, torch)
            nearest = torch.topk(similarity, probes, dim=1).indices
            candidates = lists[nearest].reshape(stop - start, candidates_per_point)
            safe = torch.clamp(candidates, min=0)
            # Общие биты с кандидатами — popcount(код точки & код родителя) по 64-битным словам (§2.2.4).
            if kernel is not None:
                table = candidates.to(torch.int32).contiguous()
                rows_index = torch.arange(stop - start, dtype=torch.int32, device=device).unsqueeze(1)
                slot_index = torch.arange(candidates_per_point, dtype=torch.int32, device=device).unsqueeze(0)
                common = kernel(
                    rows_index, slot_index,
                    cand_ptr=table.data_ptr(), qwords_ptr=point_words.data_ptr(), pwords_ptr=parent_words.data_ptr(),
                    slots=candidates_per_point,
                ).clamp(min=0).to(torch.float32)
            else:
                bits = parent_words[safe] & point_words.unsqueeze(1)
                bits = bits - ((bits >> 1) & 0x5555555555555555)
                bits = (bits & 0x3333333333333333) + ((bits >> 2) & 0x3333333333333333)
                bits = (bits + (bits >> 4)) & 0x0F0F0F0F0F0F0F0F
                common = (((bits * 0x0101010101010101) >> 56) & 0xFF).sum(dim=2).to(torch.float32)
            candidate_similarity = self._similarity(common, point_ones, parent_ones[safe], torch)
            candidate_similarity = torch.where(
                candidates >= 0, candidate_similarity, torch.full_like(candidate_similarity, -1.0)
            )
            if top == 1:
                choice = candidate_similarity.argmax(dim=1, keepdim=True)
                chosen_similarity = torch.gather(candidate_similarity, 1, choice)
            else:
                # Кандидатов может быть меньше top: недостающие места занимает лучший родитель.
                chosen_similarity, choice = torch.topk(candidate_similarity, min(top, candidates_per_point), dim=1)
                if int(choice.shape[1]) < top:
                    pad = top - int(choice.shape[1])
                    choice = torch.cat([choice, choice[:, :1].expand(-1, pad)], dim=1)
                    chosen_similarity = torch.cat([chosen_similarity, chosen_similarity[:, :1].expand(-1, pad)], dim=1)
            chosen = torch.gather(candidates, 1, choice)
            missing = chosen < 0
            chosen = torch.where(missing, chosen[:, :1].expand_as(chosen), chosen)
            chosen_similarity = torch.where(missing, chosen_similarity[:, :1].expand_as(chosen_similarity), chosen_similarity)
            best[start:stop] = chosen
            best_similarity[start:stop] = chosen_similarity
        LOGGER.event(
            "layout.seed.parents.pivots",
            section=OPTIM_SUBSET,
            data={
                "points": count,
                "parents": parent_count,
                "pivots": pivots,
                "probes": probes,
                "list_mean": float(sizes.to(torch.float64).mean().item()),
                "list_max": int(sizes.max().item()),
                "list_kept": list_max,
                "parents_outside_lists": dropped,
                "note": "родитель ищется среди родителей из списков самых похожих опорных точек, а не перебором "
                "всех родителей (ср. подмножество §5.9.4); мера сходства та же (§2.2.4)",
            },
        )

    @classmethod
    def _candidate_kernel(cls, words: int) -> Any:
        """Ядро устройства «число общих бит точки и кандидата в родители» (§2.2.4); None — jiterator недоступен.

        Поток читает номер кандидата точки (−1 — места в списке нет) и считает popcount(код точки & код
        кандидата) по словам кода: слова кодов кандидатов не собираются в промежуточный тензор.
        """
        if words not in cls._CANDIDATE_KERNELS:
            kernel = None
            try:
                from torch.cuda.jiterator import _create_jit_fn

                popcount = "".join(f"  common += __popcll(a[{index}] & b[{index}]);\n" for index in range(words))
                source = (
                    f"template <typename T> T damp_candidate_common_{words}(T row, T slot, long long cand_ptr,\n"
                    "    long long qwords_ptr, long long pwords_ptr, long long slots) {\n"
                    "  int candidate = ((const int*)cand_ptr)[(long long)row * slots + (long long)slot];\n"
                    "  if (candidate < 0) return (T)(-1);\n"
                    f"  const long long* a = (const long long*)qwords_ptr + {words} * (long long)row;\n"
                    f"  const long long* b = (const long long*)pwords_ptr + {words} * (long long)candidate;\n"
                    "  int common = 0;\n" + popcount + "  return (T)common;\n"
                    "}\n"
                )
                kernel = DeviceKernel(_create_jit_fn(source, cand_ptr=0, qwords_ptr=0, pwords_ptr=0, slots=1))
            except Exception:  # noqa: BLE001
                kernel = None
            cls._CANDIDATE_KERNELS[words] = kernel
        return cls._CANDIDATE_KERNELS[words]

    def _match_host(
        self, parents: "np.ndarray", top: int = 1, queries: "np.ndarray | None" = None
    ) -> tuple["np.ndarray", "np.ndarray"]:
        """Самые похожие родители точек queries (None — всех точек) на процессоре (numpy): номера и сходства."""
        table = self._table
        words = table.words64()
        ones = table.ones.astype(np.float64)
        parent_words = words[parents]
        parent_ones = ones[parents][None, :]
        points = np.arange(table.count, dtype=np.int64) if queries is None else np.asarray(queries, dtype=np.int64)
        top = max(1, min(int(top), int(parents.size)))
        best = np.empty((points.size, top), dtype=np.int64)
        best_similarity = np.empty((points.size, top), dtype=np.float32)
        chunk = max(1, self.MATCH_ELEMENTS_CPU // max(1, parents.size))
        for start in range(0, points.size, chunk):
            part = points[start : start + chunk]
            rows = words[part]
            common = np.bitwise_count(rows[:, None, :] & parent_words[None, :, :]).sum(axis=2, dtype=np.int64)
            similarity = self._similarity(common.astype(np.float64), ones[part, None], parent_ones, np)
            if top == 1:
                order = similarity.argmax(axis=1)[:, None]
            else:
                # По убыванию сходства; при равенстве — меньший номер родителя (устойчивая сортировка).
                order = np.argsort(-similarity, axis=1, kind="stable")[:, :top]
            best[start : start + chunk] = order
            best_similarity[start : start + chunk] = np.take_along_axis(similarity, order, axis=1)
        return best, best_similarity

    def parents(self, seed_cells: "np.ndarray", seed_shape: tuple[int, int]) -> tuple["np.ndarray", "np.ndarray"]:
        """Родитель каждой точки: номера точек затравки, годных в родители, и номер родителя (в этом списке).

        Родитель — самая похожая точка затравки (§2.2.4); точка затравки, годная в родители, — родитель самой
        себе (её место в раскладке затравки сохраняется).
        """
        parent_seeds, nearest, _ = self.similar_parents(seed_cells, seed_shape, top=1)
        return parent_seeds, nearest[:, 0].copy()

    def similar_parents(
        self, seed_cells: "np.ndarray", seed_shape: tuple[int, int], *, top: int, balanced: bool = False
    ) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
        """Самые похожие родители каждой точки: номера точек затравки, годных в родители, номера top самых
        похожих родителей точки в этом списке (N × top, по убыванию сходства) и их сходства (§2.2.4).

        Первый столбец — родитель точки; точка затравки, годная в родители, — родитель самой себе.
        balanced — точки делятся поровну между родителями с одним и тем же кодом (_balanced_parents).
        """
        mask = self._parent_mask(seed_cells, seed_shape)
        parent_seeds = np.flatnonzero(mask)
        parent_points = self._indices[parent_seeds]
        # Сходство — функция кодов (§2.2.4): у точек с одинаковым кодом родители одни и те же. Родители ищутся
        # для одной точки каждого различного кода и переносятся на остальные точки этого кода.
        representatives, code_of_point = self._distinct_codes()
        matched = self._match_device(parent_points, top, representatives)
        device = matched is not None
        if matched is None:
            matched = self._match_host(parent_points, top, representatives)
        nearest = matched[0][code_of_point]
        similarity = matched[1][code_of_point]
        del matched
        nearest[parent_points, 0] = np.arange(parent_points.size, dtype=np.int64)
        similarity[parent_points, 0] = 1.0
        children_single = np.bincount(nearest[:, 0], minlength=parent_points.size)
        if balanced:
            nearest[:, 0] = self._balanced_parents(
                nearest[:, 0], code_of_point, parent_points, seed_cells[parent_seeds]
            )
        best = nearest[:, 0]
        children = np.bincount(best, minlength=parent_points.size)
        LOGGER.event(
            "layout.seed.parents",
            section=SIMILARITY_MEASURES,
            data={
                "points": self._table.count,
                "seed_points": self.seed_count,
                "parents": int(parent_points.size),
                "distinct_codes": int(representatives.size),
                "parent_codes": int(np.unique(code_of_point[parent_points]).size),
                "children_mean": float(children.mean()),
                "children_p99": float(np.quantile(children, 0.99)),
                "children_max": int(children.max()),
                "children_max_single": int(children_single.max()),
                "balanced": bool(balanced),
                "device": "cuda" if device else "cpu",
                "similarity": self._measure,
                "note": "родитель точки — самая похожая на неё точка затравки (§2.2.4); одиночные точки и мелкие "
                "островки затравки в родители не идут; children_max_single — детей у родителя, если бы все точки "
                "одного кода шли к одному из равных родителей",
            },
        )
        return parent_seeds, nearest, similarity

    @staticmethod
    def _morton(cell_y: "np.ndarray", cell_x: "np.ndarray") -> "np.ndarray":
        """Номер ячейки вдоль Z-кривой (биты y и x вперемежку): близкие номера — близкие ячейки."""

        def spread(values: "np.ndarray") -> "np.ndarray":
            bits = values.astype(np.uint64) & np.uint64(0xFFFFFFFF)
            bits = (bits | (bits << np.uint64(16))) & np.uint64(0x0000FFFF0000FFFF)
            bits = (bits | (bits << np.uint64(8))) & np.uint64(0x00FF00FF00FF00FF)
            bits = (bits | (bits << np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
            bits = (bits | (bits << np.uint64(2))) & np.uint64(0x3333333333333333)
            return (bits | (bits << np.uint64(1))) & np.uint64(0x5555555555555555)

        return (spread(cell_y) << np.uint64(1)) | spread(cell_x)

    def _balanced_parents(
        self,
        best: "np.ndarray",
        code_of_point: "np.ndarray",
        parent_points: "np.ndarray",
        parent_cells: "np.ndarray",
    ) -> "np.ndarray":
        """Родитель каждой точки среди родителей с тем же кодом, что у самого похожего на неё: всем поровну.

        Сходство — функция кодов (§2.2.4): родители с одинаковым кодом похожи на точку одинаково, а в раскладке
        затравки они занимают целую область. Если отдать все точки одному из равных родителей, на его блок
        придётся в десятки раз больше точек, чем в нём ячеек (в наборе MNIST на код приходится 12 точек,
        у частых кодов — тысячи), и расстановка раздвинет их в прямоугольник с прямыми границами. Поэтому
        точки, выбравшие родителей одного кода, делятся между всеми родителями этого кода поровну: область
        кода на большой сетке повторяет его область в раскладке затравки.

        Родители одного кода идут вдоль Z-кривой по ячейкам затравки, точки — по своим кодам: точки одного
        кода попадают к одному родителю или к соседним. Точка затравки остаётся родителем самой себе.
        """
        parent_count = int(parent_points.size)
        parent_code = code_of_point[parent_points]
        order = np.lexsort((self._morton(parent_cells[:, 0], parent_cells[:, 1]), parent_code))
        ordered_code = parent_code[order]
        first = np.flatnonzero(np.concatenate(([True], ordered_code[1:] != ordered_code[:-1])))
        sizes = np.diff(np.concatenate((first, [parent_count])))
        group_of_parent = np.empty(parent_count, dtype=np.int64)
        group_of_parent[order] = np.repeat(np.arange(first.size, dtype=np.int64), sizes)
        is_parent = np.zeros(best.size, dtype=bool)
        is_parent[parent_points] = True
        points = np.flatnonzero(~is_parent)
        group = group_of_parent[best[points]]
        point_order = np.lexsort((code_of_point[points], group))
        points = points[point_order]
        group = group[point_order]
        group_first = np.searchsorted(group, np.arange(first.size, dtype=np.int64))
        group_size = np.diff(np.concatenate((group_first, [points.size])))
        rank = np.arange(points.size, dtype=np.int64) - group_first[group]
        slot = (rank * sizes[group]) // np.maximum(group_size[group], 1)
        result = best.copy()
        result[points] = order[first[group] + slot]
        return result

    # ------------------------------------------------------------------ расстановка
    @staticmethod
    def _dilated(region: "np.ndarray") -> "np.ndarray":
        """Область, расширенная на одну ячейку по четырём сторонам."""
        grown = region.copy()
        grown[1:, :] |= region[:-1, :]
        grown[:-1, :] |= region[1:, :]
        grown[:, 1:] |= region[:, :-1]
        grown[:, :-1] |= region[:, 1:]
        return grown

    def _region(
        self,
        parent_cells: "np.ndarray",
        seed_shape: tuple[int, int],
        shape: tuple[int, int],
    ) -> "np.ndarray":
        """Ячейки большой сетки под точки: блоки ячеек затравки, занятых родителями, — ровно по числу точек.

        Ячейке затравки отвечает блок ячеек большой сетки. Блоков родителей чуть меньше или больше, чем точек:
        область наращивается кольцами по границе (или теряет случайные граничные ячейки) до числа точек.
        """
        height, width = shape
        count = self._table.count
        rng = self._rng
        occupied = np.zeros(seed_shape, dtype=bool)
        occupied[parent_cells[:, 0], parent_cells[:, 1]] = True
        block_y = np.minimum((np.arange(height) * seed_shape[0]) // height, seed_shape[0] - 1)
        block_x = np.minimum((np.arange(width) * seed_shape[1]) // width, seed_shape[1] - 1)
        region = occupied[block_y[:, None], block_x[None, :]]
        cells = int(region.sum())
        while cells < count:
            ring = self._dilated(region) & ~region
            candidates = np.flatnonzero(ring)
            if candidates.size == 0:
                raise ValueError("grid is too small for the number of points")
            if candidates.size > count - cells:
                candidates = rng.choice(candidates, size=count - cells, replace=False)
            region.reshape(-1)[candidates] = True
            cells += int(candidates.size)
        while cells > count:
            border = np.flatnonzero(region & self._dilated(~region))
            if border.size == 0:
                border = np.flatnonzero(region)
            if border.size > cells - count:
                border = rng.choice(border, size=cells - count, replace=False)
            region.reshape(-1)[border] = False
            cells -= int(border.size)
        return region

    def _neighbour_targets(
        self,
        parent_cells: "np.ndarray",
        nearest: "np.ndarray",
        similarity: "np.ndarray",
        threshold: float,
        seed_shape: tuple[int, int],
    ) -> "np.ndarray":
        """Цели точек в ячейках затравки (N × 2, дробные): середина блока родителя со сдвигом к похожим родителям.

        В дальней энергии статьи (§5.5.1) точку тянут все похожие на неё точки: φ = Σ sim_λ·d. Здесь это
        притяжение считается по подмножеству (§5.9.4) — по самым похожим родителям точки: сдвиг от середины
        блока родителя — взвешенное по сходству среднее смещение до родителей с sim ≥ threshold (§5.4), лежащих
        не дальше NEIGHBOUR_RADIUS ячеек затравки от самого похожего (дальние похожие родители — это другое место
        карты, их середина лежала бы в чужой области). Сам родитель входит в среднее с нулевым смещением.
        Нормальная добавка (σ = NEIGHBOUR_SPREAD блока) разводит точки с одинаковыми кодами и размывает
        границы блоков: точки соседних родителей перемешиваются на ширине блока, и границу областей
        проводит ближняя фаза уровня, а не сетка блоков.
        """
        rng = self._rng
        count = int(nearest.shape[0])
        base = parent_cells[nearest[:, 0]].astype(np.float32)
        shift = np.zeros((count, 2), dtype=np.float32)
        total = np.maximum(similarity[:, 0], np.float32(1e-6)).astype(np.float32)
        radius_sq = np.float32(self.NEIGHBOUR_RADIUS * self.NEIGHBOUR_RADIUS)
        moved = np.zeros(count, dtype=bool)
        for column in range(1, int(nearest.shape[1])):
            offset = parent_cells[nearest[:, column]].astype(np.float32) - base
            weight = similarity[:, column].astype(np.float32)
            near = (weight >= np.float32(threshold)) & ((offset * offset).sum(axis=1) <= radius_sq)
            weight = np.where(near, weight, np.float32(0.0))
            shift += offset * weight[:, None]
            total += weight
            moved |= near & (offset != 0).any(axis=1)
        shift /= total[:, None]
        jitter = rng.standard_normal((count, 2)).astype(np.float32) * np.float32(self.NEIGHBOUR_SPREAD)
        target = base + np.float32(0.5) + shift + jitter
        target[:, 0] = np.clip(target[:, 0], 0.0, np.nextafter(np.float32(seed_shape[0]), np.float32(0.0)))
        target[:, 1] = np.clip(target[:, 1], 0.0, np.nextafter(np.float32(seed_shape[1]), np.float32(0.0)))
        length = np.sqrt((shift * shift).sum(axis=1))
        LOGGER.event(
            "layout.seed.neighbours",
            section=OPTIM_SUBSET,
            data={
                "points": count,
                "neighbour_parents": int(nearest.shape[1]),
                "threshold": threshold,
                "radius": self.NEIGHBOUR_RADIUS,
                "shifted_share": float(moved.mean()),
                "shift_mean": float(length.mean()),
                "shift_p99": float(np.quantile(length, 0.99)),
                "note": "место точки в блоке родителя сдвинуто к другим похожим на неё родителям (sim ≥ λ, §5.4): "
                "притяжение дальней энергии §5.5.1 по подмножеству — самым похожим точкам затравки (§5.9.4)",
            },
        )
        return target.astype(np.float64)

    @staticmethod
    def _halved(group: "np.ndarray", first: "np.ndarray", second: "np.ndarray") -> "np.ndarray":
        """Каждая группа делится пополам по координате first (при равенстве — по second): номера новых групп.

        first, second — координаты, квантованные до KEY_LEVELS уровней: порядок «группа, first, second» даёт
        одна сортировка составного ключа (номер группы меньше 2²³, по 20 бит на координату).
        """
        count = int(group.size)
        order = np.argsort((group << 40) | (first << 20) | second)
        ordered = group[order]
        starts = np.flatnonzero(np.concatenate(([True], ordered[1:] != ordered[:-1])))
        sizes = np.diff(np.concatenate((starts, [count])))
        rank = np.arange(count, dtype=np.int64) - np.repeat(starts, sizes)
        upper = rank >= np.repeat(sizes // 2, sizes)
        result = np.empty(count, dtype=np.int64)
        result[order] = ordered * 2 + upper
        return result

    @staticmethod
    def _halved_device(torch: Any, group: Any, first: Any, second: Any) -> Any:
        """_halved на устройстве: те же группы; при равных ключах раньше идёт элемент с меньшим номером."""
        count = int(group.numel())
        order = torch.argsort((group << 40) | (first << 20) | second, stable=True)
        ordered = group[order]
        boundary = torch.ones(count, dtype=torch.bool, device=group.device)
        boundary[1:] = ordered[1:] != ordered[:-1]
        starts = torch.nonzero(boundary).view(-1)
        sizes = torch.diff(torch.cat([starts, torch.tensor([count], device=group.device)]))
        rank = torch.arange(count, device=group.device) - torch.repeat_interleave(starts, sizes)
        upper = rank >= torch.repeat_interleave(torch.div(sizes, 2, rounding_mode="floor"), sizes)
        result = torch.empty(count, dtype=torch.int64, device=group.device)
        result[order] = ordered * 2 + upper.to(torch.int64)
        return result

    @classmethod
    def match_cells(
        cls,
        target_y: "np.ndarray",
        target_x: "np.ndarray",
        cell_y: "np.ndarray",
        cell_x: "np.ndarray",
        *,
        on_device: bool = False,
    ) -> "np.ndarray":
        """Взаимно-однозначное соответствие точек (с целями target) ячейкам: номер ячейки каждой точки.

        Точки и ячейки делятся пополам по медиане одной и той же координаты, поочерёдно по y и x, пока
        в группе не останется один элемент: точка получает ячейку своей группы. Порядок по обеим осям
        сохраняется, избыток точек в одном месте раздвигает соседей на доли блока, а не уносится на край.

        on_device — сортировки на CUDA (5.3 млн точек: 3 с вместо 30 с); при равных ключах порядок элементов
        может отличаться от сортировки numpy, соответствие остаётся взаимно-однозначным и сохраняет порядок.
        """
        count = int(target_y.size)
        if int(cell_y.size) != count:
            raise ValueError("the number of cells must equal the number of points")
        levels = max(1, math.ceil(math.log2(max(2, count))))
        if levels > 23:
            raise ValueError("too many points for the bisection key")
        # Координаты — целыми уровнями общего масштаба: порядок точек и ячеек по оси сравним.
        extent = float(max(target_y.max(), target_x.max(), cell_y.max() + 1, cell_x.max() + 1, 1.0))
        scale = cls.KEY_LEVELS / extent

        def quantised(values: "np.ndarray") -> "np.ndarray":
            return np.clip((np.asarray(values, dtype=np.float64) * scale).astype(np.int64), 0, cls.KEY_LEVELS - 1)

        target_y, target_x = quantised(target_y), quantised(target_x)
        cell_y, cell_x = quantised(cell_y), quantised(cell_x)
        if on_device:
            try:
                import torch

                available = torch.cuda.is_available()
            except Exception:  # noqa: BLE001
                available = False
            if available:
                device = torch.device("cuda")
                with torch.no_grad():
                    point_y, point_x, grid_y, grid_x = (
                        torch.as_tensor(values, device=device) for values in (target_y, target_x, cell_y, cell_x)
                    )
                    point_group = torch.zeros(count, dtype=torch.int64, device=device)
                    cell_group = torch.zeros(count, dtype=torch.int64, device=device)
                    for level in range(levels):
                        if level % 2 == 0:
                            point_group = cls._halved_device(torch, point_group, point_y, point_x)
                            cell_group = cls._halved_device(torch, cell_group, grid_y, grid_x)
                        else:
                            point_group = cls._halved_device(torch, point_group, point_x, point_y)
                            cell_group = cls._halved_device(torch, cell_group, grid_x, grid_y)
                    match = torch.empty(count, dtype=torch.int64, device=device)
                    match[torch.argsort(point_group, stable=True)] = torch.argsort(cell_group, stable=True)
                    result = match.cpu().numpy()
                del point_y, point_x, grid_y, grid_x, point_group, cell_group, match
                torch.cuda.empty_cache()
                return result
        point_group = np.zeros(count, dtype=np.int64)
        cell_group = np.zeros(count, dtype=np.int64)
        for level in range(levels):
            if level % 2 == 0:
                point_group = cls._halved(point_group, target_y, target_x)
                cell_group = cls._halved(cell_group, cell_y, cell_x)
            else:
                point_group = cls._halved(point_group, target_x, target_y)
                cell_group = cls._halved(cell_group, cell_x, cell_y)
        match = np.empty(count, dtype=np.int64)
        match[np.argsort(point_group, kind="stable")] = np.argsort(cell_group, kind="stable")
        return match

    def cells(
        self,
        seed_cells: "np.ndarray",
        seed_shape: tuple[int, int],
        shape: tuple[int, int],
        *,
        neighbour_threshold: float | None = None,
    ) -> "np.ndarray":
        """Ячейки (y, x) всех точек на сетке shape по раскладке затравки seed_cells на сетке seed_shape.

        seed_cells — ячейки точек затравки в порядке seed_indices. Каждой ячейке затравки отвечает блок
        ячеек большой сетки; цель точки — случайное место блока её родителя. Точки расставляются по ячейкам
        области блоков родителей взаимно-однозначно (match_cells): одна точка в ячейке (§5.1.2).

        neighbour_threshold — порог сходства λ раскладки затравки (§5.4). Если он задан, цель точки —
        не случайное место блока, а середина блока родителя, сдвинутая к другим родителям, похожим на точку
        не меньше порога (_neighbour_targets): порядок внутри блока продолжает порядок затравки.
        """
        seed_cells = np.asarray(seed_cells, dtype=np.int64)
        if seed_cells.shape != (self.seed_count, 2):
            raise ValueError("seed_cells must hold one cell per seed point")
        height, width = int(shape[0]), int(shape[1])
        seed_shape = (int(seed_shape[0]), int(seed_shape[1]))
        count = self._table.count
        if height * width < count:
            raise ValueError("grid is too small for the number of points")
        scale_y = height / float(seed_shape[0])
        scale_x = width / float(seed_shape[1])
        rng = self._rng
        if neighbour_threshold is None:
            parent_seeds, parent = self.parents(seed_cells, seed_shape)
            parent_cells = seed_cells[parent_seeds]
            # Цель точки — случайное место блока родителя.
            target_y = (parent_cells[parent, 0] + rng.random(count)) * scale_y
            target_x = (parent_cells[parent, 1] + rng.random(count)) * scale_x
        else:
            parent_seeds, nearest, similarity = self.similar_parents(
                seed_cells, seed_shape, top=self.NEIGHBOUR_PARENTS, balanced=True
            )
            parent_cells = seed_cells[parent_seeds]
            target = self._neighbour_targets(parent_cells, nearest, similarity, float(neighbour_threshold), seed_shape)
            target_y = target[:, 0] * scale_y
            target_x = target[:, 1] * scale_x
        region = self._region(parent_cells, seed_shape, (height, width))
        cell_y, cell_x = np.nonzero(region)
        match = self.match_cells(target_y, target_x, cell_y, cell_x, on_device=neighbour_threshold is not None)
        cells = np.stack([cell_y[match], cell_x[match]], axis=1).astype(np.int64)
        shift = np.hypot(cells[:, 0] + 0.5 - target_y, cells[:, 1] + 0.5 - target_x)
        LOGGER.event(
            "layout.seed.placement",
            section=OUTSIDE_ARTICLE,
            data={
                "points": count,
                "grid": (height, width),
                "seed_grid": seed_shape,
                "block": (scale_y, scale_x),
                "parents": int(parent_seeds.size),
                "shift_mean": float(shift.mean()),
                "shift_p99": float(np.quantile(shift, 0.99)),
                "shift_max": float(shift.max()),
                "note": "точка ставится в ячейку области блоков родителей (блок — ячейка затравки, растянутая "
                "до большой сетки) рядом с блоком своего родителя; дальше — обычная раскладка с малого радиуса",
            },
        )
        return cells
