"""Энергии пар (§5.5.1, §5.5.2) и точек (§5.7) по разреженным строкам сходства при жёстком пороге λ (§5.4, §7.1.7).

Математика. При η = ∞ функция порога обнуляет слабые связи: sim_λ(a, v) = sim(a, v), если sim(a, v) ≥ λ,
иначе 0. В суммах по всему V

    φ_c = Σ_v s1·d1 + s2·d2,        φ_s = Σ_v s2·d1 + s1·d2        (§5.5.1)

ненулевые слагаемые дают только точки v, похожие на a или на b. Таких точек k ≪ |V| (проценты и доли
процента), поэтому значения sim_λ предвычисляются один раз (§5.9.5; хранятся только ненулевые элементы
строк), а слагаемые считаются только по ним: O(p·k) операций на шаг вместо O(p·|V|) (§5.11).

Точность. Обмен (§5.6) решает знак φ_c − φ_s, обе суммы считаются в float32, а сумма float32 зависит
от порядка сложения. Чтобы раскладка шаг в шаг совпадала с плотным проходом
TorchPairEnergyEngine.pair_energies, его результат воспроизводится побитово:

  1. sim_λ берётся той же функцией движка — те же операции float32 над теми же числами;
  2. каждое слагаемое считается теми же операциями: d = sqrt(dy² + dx²), s·d, сумма вкладов a и b;
  3. слагаемые раскладываются в нулевой буфер той же формы [порция, |V|], что и у плотного прохода,
     и суммируются той же редукцией sum(dim=1). Прибавление нуля не меняет частичную сумму, поэтому
     редукция даёт то же число, что и на плотной строке.

Отсев по оценке ошибки. Сумма float32 из k неотрицательных слагаемых при любом порядке сложения
отличается от точной не больше чем на γ_k·S, γ_k = k·u / (1 − k·u), u = 2⁻²⁴, S — сумма слагаемых
(оценка ошибки суммирования, Higham, «Accuracy and Stability of Numerical Algorithms», §4.2). Поэтому
если разность сумм тех же слагаемых в float64 по модулю больше γ_k·(S_c + S_s), знак φ_c − φ_s
в float32 известен без плотной редукции. Она нужна только остальным парам; каждая такая пара ставится
в буфер на ту же строку той же формы, что и в плотном проходе, поэтому её суммы — те же числа.

Окрестность. Энергии окрестности пары (§5.5.2) и точек (§5.7) — суммы s/d по точкам круга радиуса r;
ненулевые слагаемые снова дают только похожие точки. Они берутся из тех же строк сходства и ставятся
в буфер формы [порция, ячейки окна] на место ячейки точки в окне — как в проходе по окну ячеек.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np

from damp.article_refs import (
    ENERGY_LONG,
    ENERGY_SHORT,
    GPU_IMPLEMENTATION,
    OPTIM_SIM_MATRIX,
    POINT_ENERGY,
    POINT_EXCHANGE,
    SIMILARITY_THRESHOLD,
)
from damp.layout.pair_energy_gpu import TorchPairEnergyEngine
from damp.logging import LOGGER


class SimilarityRows:
    """Ненулевые элементы строк sim_λ всех точек V при жёстком пороге λ (§5.4): CSR на устройстве (§5.9.5).

    Строка a хранит точки v ≠ a с sim(a, v) ≥ λ и сами значения сходства в том виде, в каком их считает
    движок для строки «точка пары × все точки V». При росте λ (§7.1.7) строки прореживаются на месте.
    За последним элементом лежит запас в max_degree нулей: строки читаются прямоугольными блоками
    шириной до max_degree, и чтение за концом строки не выходит за массив.
    """

    # Предел элементов (строки × точки) на блок плотного расчёта сходства при построении.
    BLOCK_ELEMENTS = 16_000_000
    # Число случайных строк для оценки числа ненулевых элементов до построения.
    PROBE_ROWS = 256
    # Запас ёмкости массивов относительно оценки числа элементов.
    CAPACITY_MARGIN = 1.15
    # Байт на элемент: столбец int32 и значение float32.
    BYTES_PER_EDGE = 8

    def __init__(
        self, engine: TorchPairEnergyEngine, *, lambda_threshold: float, degree: Any, col: Any, val: Any
    ) -> None:
        self._engine = engine
        self._torch = engine.torch_mod
        self._lambda = float(lambda_threshold)
        self._col = col
        self._val = val
        self._set_degree(degree)

    def _set_degree(self, degree: Any) -> None:
        torch = self._torch
        row_ptr = torch.zeros(int(degree.numel()) + 1, dtype=torch.long, device=degree.device)
        row_ptr[1:] = torch.cumsum(degree, dim=0)
        self._row_ptr = row_ptr
        self._degree = degree
        self._degree_np = degree.cpu().numpy()
        self._row_ptr_np = row_ptr.cpu().numpy()
        self._edges = int(self._degree_np.sum())
        self._max_degree = int(self._degree_np.max()) if self._degree_np.size else 0

    @classmethod
    def build(
        cls, engine: TorchPairEnergyEngine, *, lambda_threshold: float, max_edges: int
    ) -> "SimilarityRows | None":
        """Строки sim_λ всех точек; None, если ненулевых элементов больше max_edges (тогда нужен плотный проход)."""
        torch = engine.torch_mod
        device = engine.torch_device
        count = engine.point_count
        if count == 0 or max_edges <= 0:
            return None
        with torch.no_grad():
            probe_generator = torch.Generator(device="cpu")
            probe_generator.manual_seed(0)
            probe = torch.randperm(count, generator=probe_generator)[: cls.PROBE_ROWS].to(device)
            probe_sim = engine.similarity_rows(probe, lambda_threshold=lambda_threshold, eta=None)
            probe_degree = torch.count_nonzero(probe_sim, dim=1)
            probe_edges = int(probe_degree.sum().item()) - int(probe.numel())
            probe_max = int(probe_degree.max().item())
            estimate = max(0, probe_edges) / max(1, int(probe.numel())) * count
            del probe_sim
            if estimate > max_edges:
                LOGGER.event(
                    "layout.sparse.rows.skipped",
                    section=OPTIM_SIM_MATRIX,
                    data={
                        "points": count,
                        "lambda_threshold": lambda_threshold,
                        "edges_estimate": int(estimate),
                        "max_edges": int(max_edges),
                        "note": "ненулевых sim_λ слишком много: память под строки больше, чем экономия (§5.9.5)",
                    },
                )
                return None
            capacity = int(estimate * cls.CAPACITY_MARGIN) + 4 * probe_max + count
            col = torch.zeros(capacity, dtype=torch.int32, device=device)
            val = torch.zeros(capacity, dtype=torch.float32, device=device)
            degree = torch.zeros(count, dtype=torch.long, device=device)
            edges = 0
            max_degree = 0
            block_rows = max(1, cls.BLOCK_ELEMENTS // count)
            for start in range(0, count, block_rows):
                rows = torch.arange(start, min(count, start + block_rows), device=device)
                sim = engine.similarity_rows(rows, lambda_threshold=lambda_threshold, eta=None)
                # Сама точка в сумму энергии не входит (§5.5): диагональ не хранится.
                sim[torch.arange(rows.numel(), device=device), rows] = 0.0
                row_index, col_index = torch.nonzero(sim, as_tuple=True)
                found = int(row_index.numel())
                block_degree = torch.bincount(row_index, minlength=int(rows.numel()))
                max_degree = max(max_degree, int(block_degree.max().item()))
                if edges + found > max_edges:
                    LOGGER.event(
                        "layout.sparse.rows.skipped",
                        section=OPTIM_SIM_MATRIX,
                        data={
                            "points": count,
                            "lambda_threshold": lambda_threshold,
                            "edges_estimate": int(estimate),
                            "edges_found": edges + found,
                            "rows_done": int(start),
                            "max_edges": int(max_edges),
                        },
                    )
                    return None
                if edges + found + max_degree > capacity:
                    capacity = int((edges + found) * 1.5) + max_degree + count
                    grown_col = torch.zeros(capacity, dtype=torch.int32, device=device)
                    grown_val = torch.zeros(capacity, dtype=torch.float32, device=device)
                    grown_col[:edges] = col[:edges]
                    grown_val[:edges] = val[:edges]
                    col, val = grown_col, grown_val
                col[edges : edges + found] = col_index.to(torch.int32)
                val[edges : edges + found] = sim[row_index, col_index]
                degree[rows] = block_degree
                edges += found
                del sim, row_index, col_index
        rows_object = cls(engine, lambda_threshold=lambda_threshold, degree=degree, col=col, val=val)
        LOGGER.event(
            "layout.sparse.rows.build",
            section=OPTIM_SIM_MATRIX,
            data={
                "points": count,
                "lambda_threshold": lambda_threshold,
                "edges": edges,
                "degree_mean": edges / count,
                "degree_max": rows_object.max_degree,
                "density": edges / float(count * count),
                "memory_mb": capacity * cls.BYTES_PER_EDGE / 1e6,
                "note": "ненулевые sim_λ(a, v) строк всех точек: sim ≥ λ при жёстком пороге (§5.4), "
                "значения предвычислены один раз (§5.9.5)",
            },
        )
        return rows_object

    @property
    def lambda_threshold(self) -> float:
        return self._lambda

    @property
    def edges(self) -> int:
        return self._edges

    @property
    def max_degree(self) -> int:
        return self._max_degree

    @property
    def row_ptr(self) -> Any:
        return self._row_ptr

    @property
    def degree(self) -> Any:
        return self._degree

    @property
    def degree_np(self) -> "np.ndarray":
        """Степени строк на хосте: учёт пар и порций шага идёт без обращения к устройству."""
        return self._degree_np

    @property
    def row_ptr_np(self) -> "np.ndarray":
        """Начала строк на хосте (N + 1 значение)."""
        return self._row_ptr_np

    @property
    def col(self) -> Any:
        return self._col

    @property
    def val(self) -> Any:
        return self._val

    def prune(self, lambda_threshold: float) -> bool:
        """Порог вырос до λ (§7.1.7): в строках остаются элементы с sim ≥ λ. False — λ ниже порога строк."""
        if lambda_threshold == self._lambda:
            return True
        if lambda_threshold < self._lambda:
            return False
        torch = self._torch
        count = int(self._degree.numel())
        edges_before = self._edges
        with torch.no_grad():
            new_degree = torch.zeros_like(self._degree)
            written = 0
            row_start = 0
            ends = self._row_ptr[1:].cpu().numpy()
            while row_start < count:
                first = int(ends[row_start - 1]) if row_start else 0
                # Блок строк с не более чем BLOCK_ELEMENTS элементами (хотя бы одна строка).
                row_stop = int(np.searchsorted(ends, first + self.BLOCK_ELEMENTS, side="right"))
                row_stop = min(count, max(row_start + 1, row_stop))
                last = int(ends[row_stop - 1])
                if last > first:
                    values = self._val[first:last]
                    # То же сравнение, что в пороге движка: элемент отбрасывается при sim < λ (§5.4).
                    keep = ~(values < lambda_threshold)
                    kept_before = torch.zeros(last - first + 1, dtype=torch.long, device=values.device)
                    kept_before[1:] = torch.cumsum(keep, dim=0)
                    bounds = self._row_ptr[row_start : row_stop + 1] - first
                    new_degree[row_start:row_stop] = kept_before[bounds[1:]] - kept_before[bounds[:-1]]
                    kept = int(kept_before[-1].item())
                    # Запись идёт позади чтения (written ≤ first), правая часть вычислена до присваивания.
                    self._col[written : written + kept] = self._col[first:last][keep]
                    self._val[written : written + kept] = values[keep]
                    written += kept
                row_start = row_stop
            self._set_degree(new_degree)
            tail = written + self._max_degree + 1
            if tail * 2 < int(self._col.numel()):
                # Освободить память под отброшенные элементы.
                self._col = self._col[:tail].clone()
                self._val = self._val[:tail].clone()
                if self._engine.torch_device.type == "cuda":
                    torch.cuda.empty_cache()
            # Запас за последним элементом читается блоками строк: значения в нём нулевые.
            self._col[written:] = 0
            self._val[written:] = 0.0
        self._lambda = float(lambda_threshold)
        LOGGER.event(
            "layout.sparse.rows.prune",
            section=SIMILARITY_THRESHOLD,
            data={
                "lambda_threshold": lambda_threshold,
                "edges_before": edges_before,
                "edges": written,
                "degree_mean": written / max(1, count),
                "degree_max": self._max_degree,
                "note": "порог λ вырос (§7.1.7): в строках остаются sim ≥ λ (§5.4)",
            },
        )
        return True


class _PairRows:
    """Строки сходства точек пар шага: по одной на непустую точку пары (§5.5); учёт ведётся на хосте.

    table — по строке на точку: начало строки в CSR, степень, вторая точка пары (−1 — её нет), ячейка
    самой точки (y, x), ячейка второй точки (y, x), номер пары. Строка a даёт s1·d1 в φ_c и s1·d2 в φ_s,
    строка b — s2·d2 в φ_c и s2·d1 в φ_s (§5.5.1): в обоих случаях это вклад с расстоянием до своей
    ячейки и вклад с расстоянием до ячейки второй точки; в окрестности §5.5.2 вместо d стоит вес 1/d.
    """

    # Во сколько раз падает степень между соседними кандидатами в границы блоков.
    _CUT_RATIO = 0.7

    def __init__(
        self,
        idx_a: "np.ndarray",
        idx_b: "np.ndarray",
        pos_a: "np.ndarray",
        pos_b: "np.ndarray",
        rows: SimilarityRows,
        select: "np.ndarray | None" = None,
    ) -> None:
        if select is None:
            select = np.arange(idx_a.size, dtype=np.int64)
        a = idx_a[select]
        b = idx_b[select]
        has_b = b >= 0
        select_b = select[has_b]
        point = np.concatenate([a, b[has_b]])
        self.pair = np.concatenate([select, select_b])
        self.degree = rows.degree_np[point]
        table = np.empty((point.size, 8), dtype=np.int64)
        table[:, 0] = rows.row_ptr_np[point]
        table[:, 1] = self.degree
        table[:, 2] = np.concatenate([b, a[has_b]])
        split = int(select.size)
        table[:split, 3:5] = pos_a[select]
        table[:split, 5:7] = pos_b[select]
        table[split:, 3:5] = pos_b[select_b]
        table[split:, 5:7] = pos_a[select_b]
        table[:, 7] = self.pair
        self.table = table
        # Строки по убыванию степени: соседние строки читаются одним прямоугольным блоком почти без пустот.
        order = np.argsort(-self.degree, kind="stable")
        self.order = order[: int(np.count_nonzero(self.degree))]

    def batches(self, budget: int, overhead: int) -> list["np.ndarray"]:
        """Индексы строк блоками: строки × наибольшая степень блока ≤ budget, пустот как можно меньше.

        Блок читается прямоугольником шириной в наибольшую степень, поэтому его цена — overhead (запуск
        операций блока, в элементах) плюс строки × ширина. Границы выбираются динамическим
        программированием по местам, где степень падает в 1/_CUT_RATIO раз.
        """
        order = self.order
        total = int(order.size)
        if total == 0:
            return []
        degree = self.degree[order]
        if total * int(degree[0]) <= overhead:
            return [order]
        levels = []
        level = float(degree[0]) * self._CUT_RATIO
        while level >= 1.0:
            levels.append(level)
            level *= self._CUT_RATIO
        # degree убывает: первая строка со степенью ниже уровня.
        positions = np.searchsorted(-degree, -np.asarray(levels, dtype=np.float64), side="right") if levels else []
        cuts = np.unique(np.concatenate([[0], np.asarray(positions, dtype=np.int64), [total]])).tolist()
        widths = [int(degree[cut]) for cut in cuts[:-1]]
        best = [0.0] * len(cuts)
        previous = [0] * len(cuts)
        for stop in range(1, len(cuts)):
            best_cost = None
            for start in range(stop):
                cost = best[start] + overhead + (cuts[stop] - cuts[start]) * widths[start]
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    previous[stop] = start
            best[stop] = best_cost
        bounds = []
        stop = len(cuts) - 1
        while stop > 0:
            bounds.append((cuts[previous[stop]], cuts[stop]))
            stop = previous[stop]
        result: list["np.ndarray"] = []
        for start, end in reversed(bounds):
            rows = max(1, budget // max(1, int(degree[start])))
            for first in range(start, end, rows):
                result.append(order[first : min(end, first + rows)])
        return result


@dataclass(frozen=True)
class _TermsCache:
    """Слагаемые единственного блока строк шага (отсев → плотная редукция без пересчёта)."""

    pair_rows: _PairRows
    batch: "np.ndarray"
    column: Any
    terms: Any


@dataclass(frozen=True)
class _Reduction:
    """Что и как суммируется: форма порции редукции исходного прохода и расчёт слагаемых блока строк.

    width — длина суммируемой строки (|V| в проходе по всему V, число ячеек окна в окрестности);
    chunk — пар в порции редукции; sides — 2 (φ_c и φ_s) или 1 (только φ_c);
    terms(pair_rows, batch) → (место слагаемого в строке [строки, ширина блока], вклады [строки, ширина, sides]).
    """

    width: int
    chunk: int
    sides: int
    terms: Callable[[_PairRows, "np.ndarray"], tuple[Any, Any]]


class SparseLongEnergy:
    """Энергии пар по строкам SimilarityRows — те же числа, что у проходов TorchPairEnergyEngine.

    pair_energies — дальняя фаза по всему V (§5.5.1); neighbourhood_energies — окрестность пары (§5.5.2)
    и энергии точек (§5.7). В обоих случаях ненулевые слагаемые дают только похожие точки (sim ≥ λ).
    """

    # Запас поверх оценки ошибки суммирования float32 (учитывает ошибку сумм float64 и самой оценки).
    BOUND_SAFETY = 2.0
    _UNIT_ROUNDOFF = 2.0 ** -24
    # Цена запуска операций одного блока строк, выраженная в элементах блока: на малых блоках время шага
    # определяет число запусков ядер (≈ 1 мс на блок), а не объём данных (≈ 0.3 нс на элемент). Отдельный
    # блок выделяется, только если экономит на пустотах больше.
    BATCH_OVERHEAD_ELEMENTS = 3_000_000
    # Обнулить буфер целиком дешевле, чем записать нули по индексам, пока элементов буфера не больше
    # стольких на одно слагаемое (измерено: 0.01 нс на элемент против 0.5 нс на индекс).
    ZERO_FILL_RATIO = 48
    # Окрестность считается по строкам сходства, пока слагаемых не больше стольких на ячейку окна:
    # ячейка окна стоит подсчёта общих бит кода и трёх десятков операций, слагаемое строки — десятка.
    WINDOW_CELL_COST = 8

    def __init__(
        self,
        engine: TorchPairEnergyEngine,
        rows: SimilarityRows,
        *,
        buffer_bytes: int,
        entry_budget: int,
        screen_min_chunks: int,
    ) -> None:
        if buffer_bytes <= 0 or entry_budget <= 0 or screen_min_chunks <= 0:
            raise ValueError("buffer_bytes, entry_budget and screen_min_chunks must be positive")
        self._engine = engine
        self._rows = rows
        self._torch = engine.torch_mod
        self._device = engine.torch_device
        self._buffer_bytes = int(buffer_bytes)
        self._entry_budget = int(entry_budget)
        self._screen_min_chunks = int(screen_min_chunks)
        # Буфер плотной редукции: sides × slots участков по stride элементов — участки φ_c, затем φ_s.
        self._buffer: Any = None
        self._buffer_key: tuple[int, int, int] = (0, 0, 0)
        self._offsets: Any = None
        self._sides: Any = None
        LOGGER.event(
            "layout.sparse.energy.init",
            section=GPU_IMPLEMENTATION,
            data={
                "points": engine.point_count,
                "chunk_rows": engine.dense_chunk_rows(),
                "local_chunk_rows": engine.local_chunk_rows(),
                "buffer_mb": self._buffer_bytes / 1e6,
                "entry_budget": self._entry_budget,
                "screen_min_chunks": self._screen_min_chunks,
                "note": "φ_c, φ_s по ненулевым sim_λ (§5.5.1, §5.5.2); суммирование — та же редукция float32 "
                "той же формы, что у прохода по всему V или по окну ячеек (§5.12)",
            },
        )

    @property
    def rows(self) -> SimilarityRows:
        return self._rows

    # ------------------------------------------------------------------ слагаемые
    def _column_offsets(self, width: int) -> Any:
        if self._offsets is None or int(self._offsets.numel()) < width:
            self._offsets = self._torch.arange(max(width, 256), device=self._device)
        return self._offsets[:width]

    def _row_entries(self, pair_rows: _PairRows, batch: "np.ndarray") -> tuple[Any, Any, Any, Any, Any]:
        """Элементы блока строк: (таблица строк, столбцы v, сходство, не-пустота, ячейки точек v).

        Строки читаются прямоугольником шириной в наибольшую степень блока; за концом строки и на второй
        точке пары элемент не учитывается: сами точки пары в сумму не входят (§5.5).
        """
        torch = self._torch
        rows = self._rows
        width = int(pair_rows.degree[batch[0]])
        table = torch.as_tensor(pair_rows.table[batch], device=self._device)
        offsets = self._column_offsets(width)
        position = table[:, 0:1] + offsets
        column = rows.col[position]
        keep = (offsets < table[:, 1:2]) & (column != table[:, 2:3])
        return table, column, rows.val[position], keep, self._engine.positions[column]

    def _long_terms(self, pair_rows: _PairRows, batch: "np.ndarray", *, euclidean: bool) -> tuple[Any, Any]:
        """Слагаемые дальней энергии (§5.5.1): s·d до своей ячейки (в φ_c) и до ячейки второй точки (в φ_s).

        d — евклидово (§5.5) или d² (§5.9.2). Каждое число получается теми же операциями float32, что
        в плотном проходе. Место слагаемого в строке суммы — номер точки v.
        """
        torch = self._torch
        table, column, similarity, keep, cell = self._row_entries(pair_rows, batch)
        similarity = similarity * keep
        # Разности координат: [строки, ширина, ячейка (своя, второй точки), (y, x)].
        delta = table[:, 3:7].to(torch.float32).view(-1, 1, 2, 2) - cell.unsqueeze(2)
        distance = (delta * delta).sum(dim=3)
        if euclidean:
            distance = torch.sqrt(distance)
        return column, similarity.unsqueeze(2) * distance

    def _neighbourhood_terms(
        self,
        pair_rows: _PairRows,
        batch: "np.ndarray",
        *,
        centre: Any,
        radius_sq: Any,
        base: Any | None,
        reach: int,
        sides: int,
        kernel: str,
        euclidean: bool,
        distance_eps: float,
        kernel_width: float,
    ) -> tuple[Any, Any]:
        """Слагаемые окрестности (§5.5.2, §5.7): s·w до своей ячейки (в φ_c) и до ячейки второй точки (в φ_s).

        Точка v входит в сумму, если ‖v − c‖ ≤ r; w = 1/(d + ε) (d — евклидово или d², §5.9.2) либо
        exp(−d²/(2σ²)) (расширение статьи). Место слагаемого в строке суммы — номер ячейки точки v в окне
        (2R+1)² вокруг центра пары, как в проходе по окну ячеек; base = None — окно не меньше всего V,
        исходный проход идёт по всем точкам, и место слагаемого — номер точки v.
        """
        torch = self._torch
        table, column, similarity, keep, cell = self._row_entries(pair_rows, batch)
        pair = table[:, 7]
        offset = cell - centre[pair].unsqueeze(1)
        keep = keep & ((offset * offset).sum(dim=2) <= radius_sq[pair].unsqueeze(1))
        similarity = similarity * keep
        delta = table[:, 3 : 3 + 2 * sides].to(torch.float32).view(-1, 1, sides, 2) - cell.unsqueeze(2)
        d_sq = (delta * delta).sum(dim=3)
        if kernel == "gauss":
            sig2 = 2.0 * kernel_width * kernel_width * radius_sq[pair].view(-1, 1, 1)
            weight = torch.exp(-d_sq / sig2)
        else:
            distance = torch.sqrt(d_sq) if euclidean else d_sq
            weight = 1.0 / (distance + distance_eps)
        if base is None:
            return column, similarity.unsqueeze(2) * weight
        # Номер ячейки окна: (v_y − base_y + R)·(2R + 1) + (v_x − base_x + R); у неучтённых элементов — 0.
        window = cell.to(torch.long) - base[pair].unsqueeze(1) + reach
        place = (window[:, :, 0] * (2 * reach + 1) + window[:, :, 1]) * keep
        return place, similarity.unsqueeze(2) * weight

    # ------------------------------------------------------------------ плотная редукция
    def _ensure_buffer(self, slots: int, stride: int, sides: int) -> int:
        """Буфер не меньше чем на slots участков; возвращает число участков на одну сторону."""
        torch = self._torch
        have_slots, have_stride, have_sides = self._buffer_key
        if have_slots >= slots and have_stride == stride and have_sides == sides:
            return have_slots
        self._buffer = None
        self._buffer = torch.zeros((sides * slots, stride), dtype=torch.float32, device=self._device)
        self._buffer_key = (slots, stride, sides)
        # Смещения участков φ_c и φ_s в развёрнутом буфере.
        self._sides = torch.arange(sides, device=self._device) * (slots * stride)
        return slots

    def _row_blocks(
        self, reduction: _Reduction, pair_rows: _PairRows, place: "np.ndarray"
    ) -> Iterator[tuple[Any, Any, Any]]:
        """Блоки слагаемых строк pair_rows: (смещение строки пары в буфере, места в строке, вклады)."""
        base_of_row = place[pair_rows.pair]
        for batch in pair_rows.batches(self._entry_budget, self.BATCH_OVERHEAD_ELEMENTS):
            column, terms = reduction.terms(pair_rows, batch)
            yield self._torch.as_tensor(base_of_row[batch], device=self._device), column, terms

    def _cached_block(
        self, cache: _TermsCache, pending: "np.ndarray", place: "np.ndarray"
    ) -> Iterator[tuple[Any, Any, Any]]:
        """Тот же блок слагаемых, что считал отсев, но только строки пар pending (слагаемые не пересчитываются)."""
        pairs = cache.pair_rows.pair[cache.batch]
        chosen = np.zeros(place.size, dtype=bool)
        chosen[pending] = True
        rows = np.flatnonzero(chosen[pairs])
        if rows.size == 0:
            return
        table = self._torch.as_tensor(np.stack([rows, place[pairs[rows]]], axis=1), device=self._device)
        yield table[:, 1], cache.column[table[:, 0]], cache.terms[table[:, 0]]

    def _reduce(
        self, reduction: _Reduction, blocks: Iterable[tuple[Any, Any, Any]], slot_rows: Sequence[int]
    ) -> "np.ndarray":
        """Суммы той же редукцией, что в исходном проходе: по строкам буферов slot_rows.

        Блок задаёт смещение строки каждой пары в развёрнутом буфере, места и вклады её слагаемых. Буфер
        slot имеет форму [slot_rows[slot], width] — ту же, что порция исходного прохода, в которой пара стояла
        на той же строке. Возвращает суммы подряд по буферам: строки φ_c, затем (при sides = 2) строки φ_s.
        """
        torch = self._torch
        width = reduction.width
        slots, stride, sides = self._buffer_key
        flat = self._buffer.view(-1)
        entries = 0
        touched: list[Any] = []
        for base, column, terms in blocks:
            index = (base.unsqueeze(1) + column).unsqueeze(2) + self._sides
            # В элемент буфера попадает не больше двух ненулевых вкладов (строки a и строки b), первый —
            # в точный ноль, поэтому результат не зависит от порядка: fl(s1·d1 + s2·d2), как в исходном проходе.
            flat.index_add_(0, index.reshape(-1), terms.reshape(-1))
            touched.append(index)
            entries += int(index.numel())
        sums = []
        for slot_index, rows in enumerate(slot_rows):
            for side in range(sides):
                sums.append(self._buffer[side * slots + slot_index, : rows * width].view(rows, width).sum(dim=1))
        values = torch.cat(sums).cpu().numpy()
        # Буфер возвращается к нулям: целиком, если это дешевле записи нулей по тем же индексам.
        if sides * slots * stride <= self.ZERO_FILL_RATIO * entries:
            self._buffer.zero_()
        else:
            for index in touched:
                flat.index_fill_(0, index.reshape(-1), 0.0)
        return values

    # ------------------------------------------------------------------ отсев по оценке ошибки
    def _screen(
        self,
        reduction: _Reduction,
        pair_rows: _PairRows,
        idx_a: "np.ndarray",
        idx_b: "np.ndarray",
        *,
        energy_c: "np.ndarray",
        energy_s: "np.ndarray",
    ) -> tuple["np.ndarray", _TermsCache | None]:
        """Пары, знак φ_c − φ_s которых не следует из оценки ошибки суммирования float32.

        Остальным парам записываются суммы float64 тех же слагаемых, округлённые до float32 с сохранением
        знака разности. Если все строки шага уместились в один блок, его слагаемые возвращаются вторым
        значением: плотная редукция оставшихся пар берёт их оттуда, не пересчитывая.
        """
        torch = self._torch
        pair_count = int(idx_a.size)
        degree = self._rows.degree_np
        sum_c = np.zeros(pair_count, dtype=np.float64)
        sum_s = np.zeros(pair_count, dtype=np.float64)
        batches = pair_rows.batches(self._entry_budget, self.BATCH_OVERHEAD_ELEMENTS)
        cache: _TermsCache | None = None
        for batch in batches:
            column, terms = reduction.terms(pair_rows, batch)
            sums = terms.sum(dim=1, dtype=torch.float64).cpu().numpy()
            pairs = pair_rows.pair[batch]
            sum_c += np.bincount(pairs, weights=sums[:, 0], minlength=pair_count)
            sum_s += np.bincount(pairs, weights=sums[:, 1], minlength=pair_count)
            if len(batches) == 1:
                cache = _TermsCache(pair_rows, batch, column, terms)
        terms_count = (degree[idx_a] + np.where(idx_b >= 0, degree[np.maximum(idx_b, 0)], 0)).astype(np.float64)
        # Ошибка суммы float32 из k неотрицательных слагаемых при любом порядке сложения ≤ γ_k·S.
        gamma = terms_count * self._UNIT_ROUNDOFF / (1.0 - terms_count * self._UNIT_ROUNDOFF)
        total = sum_c + sum_s
        difference = sum_c - sum_s
        # Все слагаемые нулевые (у точек пары нет похожих) — обе суммы точные нули.
        certain = (np.abs(difference) > self.BOUND_SAFETY * gamma * total) | (total == 0.0)
        value_c = sum_c.astype(np.float32)
        value_s = sum_s.astype(np.float32)
        # Округление до float32 не должно менять знак разности: совпавшие значения разводятся на один шаг сетки.
        tie = certain & (value_c == value_s) & (difference != 0.0)
        if tie.any():
            toward = np.where(difference[tie] > 0.0, -np.inf, np.inf).astype(np.float32)
            value_s[tie] = np.nextafter(value_s[tie], toward)
        energy_c[certain] = value_c[certain]
        energy_s[certain] = value_s[certain]
        return np.flatnonzero(~certain), cache

    # ------------------------------------------------------------------ общий ход расчёта
    def _energies(
        self,
        reduction: _Reduction,
        ia: "np.ndarray",
        ib: "np.ndarray",
        pa: "np.ndarray",
        pb: "np.ndarray",
        *,
        signs_only: bool,
    ) -> tuple["np.ndarray", "np.ndarray", int, bool]:
        """(φ_c, φ_s, число пар с плотной редукцией, был ли отсев) для заданной формы редукции."""
        pair_count = int(ia.size)
        energy_c = np.zeros(pair_count, dtype=np.float32)
        energy_s = np.zeros(pair_count, dtype=np.float32)
        chunk = reduction.chunk
        width = reduction.width
        sides = reduction.sides
        chunks = -(-pair_count // chunk)
        last_rows = pair_count - (chunks - 1) * chunk
        # Шаг буфера кратен 128 элементам (512 байт): начало каждого участка выровнено так же, как начало
        # отдельного тензора, поэтому редукция по участку идёт тем же порядком, что по порции исходного прохода.
        stride = -(-(chunk * width) // 128) * 128
        max_slots = max(1, self._buffer_bytes // (sides * 4 * stride))
        screened = signs_only and sides == 2 and chunks >= self._screen_min_chunks
        all_rows = _PairRows(ia, ib, pa, pb, self._rows)
        cache: _TermsCache | None = None
        if screened:
            pending, cache = self._screen(reduction, all_rows, ia, ib, energy_c=energy_c, energy_s=energy_s)
        else:
            pending = np.arange(pair_count, dtype=np.int64)
        if pending.size:
            chunk_of = pending // chunk
            row_of = pending - chunk_of * chunk
            short = (chunk_of == chunks - 1) & (last_rows != chunk)
            if screened:
                # Пары с одной строкой и одной формой порции расходятся по разным буферам: номер буфера —
                # порядковый номер пары среди пар с той же (формой, строкой).
                key = row_of + np.where(short, chunk, 0)
                order = np.argsort(key, kind="stable")
                sorted_key = key[order]
                group_start = np.flatnonzero(np.concatenate([[True], sorted_key[1:] != sorted_key[:-1]]))
                group_size = np.diff(np.concatenate([group_start, [sorted_key.size]]))
                rank = np.empty(pending.size, dtype=np.int64)
                rank[order] = np.arange(sorted_key.size) - np.repeat(group_start, group_size)
                full_slots = int(rank[~short].max()) + 1 if (~short).any() else 0
                short_slots = int(rank[short].max()) + 1 if short.any() else 0
                slot = np.where(short, full_slots + rank, rank)
            else:
                full_slots = chunks - 1 if last_rows != chunk else chunks
                short_slots = chunks - full_slots
                slot = chunk_of
            slot_rows = np.asarray([chunk] * full_slots + [last_rows] * short_slots, dtype=np.int64)
            slots = self._ensure_buffer(min(int(slot_rows.size), max_slots), stride, sides)
            place = np.zeros(pair_count, dtype=np.int64)
            for first in range(0, int(slot_rows.size), slots):
                inside = (slot >= first) & (slot < first + slots)
                if not inside.any():
                    continue
                select = pending[inside]
                local_slot = slot[inside] - first
                local_row = row_of[inside]
                group_rows = slot_rows[first : first + slots]
                place[select] = local_slot * stride + local_row * width
                if cache is not None:
                    blocks = self._cached_block(cache, select, place)
                elif select.size == pair_count:
                    blocks = self._row_blocks(reduction, all_rows, place)
                else:
                    blocks = self._row_blocks(reduction, _PairRows(ia, ib, pa, pb, self._rows, select), place)
                values = self._reduce(reduction, blocks, group_rows.tolist())
                starts = np.zeros(group_rows.size, dtype=np.int64)
                np.cumsum(sides * group_rows[:-1], out=starts[1:])
                energy_c[select] = values[starts[local_slot] + local_row]
                if sides == 2:
                    energy_s[select] = values[starts[local_slot] + group_rows[local_slot] + local_row]
        return energy_c, energy_s, int(pending.size), screened

    @staticmethod
    def _pair_arrays(
        idx_a: Sequence[int], idx_b: Sequence[int], pos_a: Sequence[Sequence[float]], pos_b: Sequence[Sequence[float]]
    ) -> tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:
        ia = np.asarray(idx_a, dtype=np.int64)
        ib = np.asarray(idx_b, dtype=np.int64)
        pa = np.asarray(pos_a).reshape(-1, 2).astype(np.int64, copy=False)
        pb = np.asarray(pos_b).reshape(-1, 2).astype(np.int64, copy=False)
        return ia, ib, pa, pb

    # ------------------------------------------------------------------ энергии пар дальней фазы
    def pair_energies(
        self,
        idx_a: Sequence[int],
        idx_b: Sequence[int],
        pos_a: Sequence[Sequence[float]],
        pos_b: Sequence[Sequence[float]],
        *,
        euclidean: bool,
        signs_only: bool,
    ) -> tuple[Any, Any]:
        """(φ_c, φ_s) пар дальней фазы (§5.5.1) по всему V; numpy float32.

        signs_only = False: оба массива побитово равны результату TorchPairEnergyEngine.pair_energies.
        signs_only = True (жадное правило §5.6, нужен только знак φ_c − φ_s): у пар, знак разности которых
        следует из оценки ошибки суммирования, возвращаются суммы float64, округлённые до float32, — знак
        разности тот же, что у плотного прохода; у остальных пар — побитово те же суммы.
        """
        ia, ib, pa, pb = self._pair_arrays(idx_a, idx_b, pos_a, pos_b)
        if ia.size == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
        reduction = _Reduction(
            width=self._engine.point_count,
            chunk=self._engine.dense_chunk_rows(),
            sides=2,
            terms=lambda pair_rows, batch: self._long_terms(pair_rows, batch, euclidean=euclidean),
        )
        with self._torch.no_grad():
            energy_c, energy_s, reduced, screened = self._energies(reduction, ia, ib, pa, pb, signs_only=signs_only)
        LOGGER.event(
            "layout.sparse.energy.pairs",
            section=ENERGY_LONG,
            data={
                "pairs": int(ia.size),
                "points": self._engine.point_count,
                "screened": screened,
                "reduced_pairs": reduced,
                "lambda_threshold": self._rows.lambda_threshold,
                "note": "φ_c = Σ s1·d1 + s2·d2, φ_s = Σ s2·d1 + s1·d2 по ненулевым sim_λ (§5.5.1); "
                "reduced_pairs — пары, суммы которых взяты плотной редукцией float32",
            },
        )
        return energy_c, energy_s

    # ------------------------------------------------------------------ энергии окрестности
    def neighbourhood_energies(
        self,
        idx_a: Sequence[int],
        idx_b: Sequence[int],
        pos_a: Sequence[Sequence[float]],
        pos_b: Sequence[Sequence[float]],
        *,
        local_radius: float | None,
        kernel: str,
        euclidean: bool,
        distance_eps: float,
        kernel_width: float,
        signs_only: bool,
        need_swapped: bool = True,
    ) -> tuple[Any, Any] | None:
        """(φ_c, φ_s) окрестности пар (§5.5.2) и энергии точек (§5.7, idx_b = −1); numpy float32.

        Те же числа, что даёт TorchPairEnergyEngine (при signs_only — с тем же знаком разности): путь по окну
        ячеек, а если окно не меньше всего V (у пары далёкая вторая ячейка) — проход по всем точкам.
        None — расчёт по строкам здесь невыгоден (слагаемых больше, чем стоят ячейки окна): окрестность
        считает сам движок.
        """
        torch = self._torch
        ia, ib, pa, pb = self._pair_arrays(idx_a, idx_b, pos_a, pos_b)
        pair_count = int(ia.size)
        if pair_count == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
        engine = self._engine
        degree = self._rows.degree_np
        entries = int(degree[ia].sum() + degree[ib[ib >= 0]].sum())
        with torch.no_grad():
            pair_cells = torch.as_tensor(np.concatenate([pa, pb], axis=1), device=self._device).to(torch.float32)
            centre, radius_sq, reach = engine.neighbourhood_geometry(pair_cells[:, 0:2], pair_cells[:, 2:4], local_radius)
            cells = (2 * reach + 1) ** 2
            # Окно не меньше всего V: исходный проход суммирует по всем точкам порциями плотного прохода.
            whole_space = cells >= engine.point_count
            if not whole_space and entries > self.WINDOW_CELL_COST * pair_count * cells:
                return None
            base = None if whole_space else torch.floor(centre).to(torch.long)
            sides = 2 if need_swapped else 1
            reduction = _Reduction(
                width=engine.point_count if whole_space else cells,
                chunk=engine.dense_chunk_rows() if whole_space else engine.local_chunk_rows(),
                sides=sides,
                terms=lambda pair_rows, batch: self._neighbourhood_terms(
                    pair_rows, batch, centre=centre, radius_sq=radius_sq, base=base, reach=reach, sides=sides,
                    kernel=kernel, euclidean=euclidean, distance_eps=distance_eps, kernel_width=kernel_width,
                ),
            )
            energy_c, energy_s, reduced, screened = self._energies(reduction, ia, ib, pa, pb, signs_only=signs_only)
        LOGGER.event(
            "layout.sparse.energy.neighbourhood",
            section=ENERGY_SHORT if need_swapped else POINT_ENERGY,
            data={
                "pairs": pair_count,
                "window_cells": cells,
                "whole_space": whole_space,
                "entries": entries,
                "screened": screened,
                "reduced_pairs": reduced,
                "kernel": kernel,
                "lambda_threshold": self._rows.lambda_threshold,
                "note": "φ = Σ s/d по точкам окрестности с sim ≥ λ (§5.5.2, §5.7): слагаемые по строкам сходства "
                "вместо всех ячеек окна, суммы те же",
            },
        )
        return energy_c, energy_s

    # ------------------------------------------------------------------ сверка
    def matches(
        self,
        sparse_exact: tuple[Any, Any],
        sparse_signs: tuple[Any, Any] | None,
        dense: tuple[Any, Any],
        *,
        kind: str,
        step: int | None,
        swap_when_greater: bool,
    ) -> bool:
        """Сверка с исходным проходом на тех же парах: суммы побитово и знаки разности после отсева.

        swap_when_greater — обмен выгоден при φ_s > φ_c (окрестность §5.5.2), иначе при φ_s < φ_c (§5.5.1).
        """
        dense_c = np.asarray(dense[0])
        dense_s = np.asarray(dense[1])
        mismatches = int((sparse_exact[0] != dense_c).sum() + (sparse_exact[1] != dense_s).sum())
        exact = mismatches == 0
        signs = True
        if sparse_signs is not None:
            sign_mismatches = int((np.sign(dense_c - dense_s) != np.sign(sparse_signs[0] - sparse_signs[1])).sum())
            signs = sign_mismatches == 0
            mismatches += sign_mismatches
        LOGGER.event(
            "layout.sparse.energy.verify",
            section=POINT_EXCHANGE,
            data={
                "step": -1 if step is None else step,
                "kind": kind,
                "pairs": int(dense_c.size),
                "bitwise_equal": exact,
                "signs_equal": signs,
                "mismatches": mismatches,
                "favourable": int(((dense_s > dense_c) if swap_when_greater else (dense_s < dense_c)).sum()),
                "lambda_threshold": self._rows.lambda_threshold,
                "note": "φ_c, φ_s по ненулевым sim_λ сверены с исходным проходом (§5.5.1, §5.5.2): "
                "обмены §5.6 совпадают при побитовом равенстве сумм",
            },
            force=True,
        )
        return exact and signs
