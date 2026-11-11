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
from damp.logging import LOGGER


class SeededPlacement:
    """Расстановка точек таблицы кодов по раскладке случайного подмножества этих точек."""

    # Предел элементов (точки × родители) на один проход поиска родителей на устройстве.
    MATCH_ELEMENTS = 64_000_000
    # Число уровней квантования координаты в ключе сортировки деления пополам (20 бит на координату).
    KEY_LEVELS = 1 << 20
    # То же для расчёта на процессоре (numpy): в памяти держится порция × родители × слова кода.
    MATCH_ELEMENTS_CPU = 2_000_000
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
                "самых похожих точек B (расширение §5.9.4: подмножество задаёт глобальный порядок)",
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

    def _match_device(self, parents: "np.ndarray") -> "np.ndarray | None":
        """Самый похожий родитель каждой точки на CUDA; None — устройство недоступно.

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
            parent_codes = codes(words[parent_device]).T.contiguous()
            parent_ones = ones[parent_device].unsqueeze(0)
            best = torch.empty(table.count, dtype=torch.int64, device=device)
            chunk = max(1, self.MATCH_ELEMENTS // max(1, parents.size))
            for start in range(0, table.count, chunk):
                stop = min(table.count, start + chunk)
                similarity = self._similarity(
                    codes(words[start:stop]) @ parent_codes, ones[start:stop].unsqueeze(1), parent_ones, torch
                )
                best[start:stop] = similarity.argmax(dim=1)
            result = best.cpu().numpy()
        del parent_codes, best, words
        torch.cuda.empty_cache()
        return result

    def _match_host(self, parents: "np.ndarray") -> "np.ndarray":
        """Самый похожий родитель каждой точки на процессоре (numpy)."""
        table = self._table
        words = table.words64()
        ones = table.ones.astype(np.float64)
        parent_words = words[parents]
        parent_ones = ones[parents][None, :]
        best = np.empty(table.count, dtype=np.int64)
        chunk = max(1, self.MATCH_ELEMENTS_CPU // max(1, parents.size))
        for start in range(0, table.count, chunk):
            rows = words[start : start + chunk]
            common = np.bitwise_count(rows[:, None, :] & parent_words[None, :, :]).sum(axis=2, dtype=np.int64)
            similarity = self._similarity(common.astype(np.float64), ones[start : start + chunk, None], parent_ones, np)
            best[start : start + chunk] = similarity.argmax(axis=1)
        return best

    def parents(self, seed_cells: "np.ndarray", seed_shape: tuple[int, int]) -> tuple["np.ndarray", "np.ndarray"]:
        """Родитель каждой точки: номера точек затравки, годных в родители, и номер родителя (в этом списке).

        Родитель — самая похожая точка затравки (§2.2.4); точка затравки, годная в родители, — родитель самой
        себе (её место в раскладке затравки сохраняется).
        """
        mask = self._parent_mask(seed_cells, seed_shape)
        parent_seeds = np.flatnonzero(mask)
        parent_points = self._indices[parent_seeds]
        best = self._match_device(parent_points)
        device = best is not None
        if best is None:
            best = self._match_host(parent_points)
        best[parent_points] = np.arange(parent_points.size, dtype=np.int64)
        children = np.bincount(best, minlength=parent_points.size)
        LOGGER.event(
            "layout.seed.parents",
            section=SIMILARITY_MEASURES,
            data={
                "points": self._table.count,
                "seed_points": self.seed_count,
                "parents": int(parent_points.size),
                "children_mean": float(children.mean()),
                "children_max": int(children.max()),
                "device": "cuda" if device else "cpu",
                "similarity": self._measure,
                "note": "родитель точки — самая похожая на неё точка затравки (§2.2.4); одиночные точки и мелкие "
                "островки затравки в родители не идут",
            },
        )
        return parent_seeds, best

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

    @classmethod
    def match_cells(
        cls,
        target_y: "np.ndarray",
        target_x: "np.ndarray",
        cell_y: "np.ndarray",
        cell_x: "np.ndarray",
    ) -> "np.ndarray":
        """Взаимно-однозначное соответствие точек (с целями target) ячейкам: номер ячейки каждой точки.

        Точки и ячейки делятся пополам по медиане одной и той же координаты, поочерёдно по y и x, пока
        в группе не останется один элемент: точка получает ячейку своей группы. Порядок по обеим осям
        сохраняется, избыток точек в одном месте раздвигает соседей на доли блока, а не уносится на край.
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
    ) -> "np.ndarray":
        """Ячейки (y, x) всех точек на сетке shape по раскладке затравки seed_cells на сетке seed_shape.

        seed_cells — ячейки точек затравки в порядке seed_indices. Каждой ячейке затравки отвечает блок
        ячеек большой сетки; цель точки — случайное место блока её родителя. Точки расставляются по ячейкам
        области блоков родителей взаимно-однозначно (match_cells): одна точка в ячейке (§5.1.2).
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
        parent_seeds, parent = self.parents(seed_cells, seed_shape)
        parent_cells = seed_cells[parent_seeds]
        rng = self._rng
        # Цель точки — случайное место блока родителя.
        target_y = (parent_cells[parent, 0] + rng.random(count)) * scale_y
        target_x = (parent_cells[parent, 1] + rng.random(count)) * scale_x
        region = self._region(parent_cells, seed_shape, (height, width))
        cell_y, cell_x = np.nonzero(region)
        match = self.match_cells(target_y, target_x, cell_y, cell_x)
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
