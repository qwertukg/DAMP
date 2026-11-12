"""Знак φ_c − φ_s пар шага раскладки (§5.5.1, §5.5.2, §5.6) одним проходом ядра по строкам sim_λ (§5.12).

Что нужно шагу. Жадное правило §5.6 меняет точки пары местами, когда энергия после обмена лучше: в дальней
фазе — φ_s < φ_c (§5.5.1), в ближней — φ_s > φ_c (§5.5.2). Самих энергий правилу не нужно, нужен знак их
разности. При жёстком пороге (η = ∞, §7.1.7) ненулевые слагаемые сумм дают только точки, похожие на точки
пары, — элементы строк SimilarityRows (§5.9.5).

Как считается. Строка точки пары режется на отрезки по SEGMENT элементов; каждый отрезок считает свой поток
устройства (§5.12: «каждый поток считает…»): читает номер похожей точки, её сходство и ячейку, считает
слагаемые s·d (или s/(d + ε) в окрестности) теми же операциями float32, что исходный проход
TorchPairEnergyEngine, и складывает их в частичную сумму float64. Второе ядро складывает частичные суммы
пары и выносит решение. Весь шаг — два запуска ядер, одна загрузка таблицы пар и одна выгрузка решений:
время шага определяет объём слагаемых, а не число операций torch над промежуточными тензорами.

Почему результат тот же. Исходный проход суммирует те же слагаемые float32 редукцией, порядок сложения
которой неизвестен. Сумма float32 из n неотрицательных слагаемых при любом порядке сложения отличается
от точной не больше чем на γ_{n−1}·S, γ_k = k·u / (1 − k·u), u = 2⁻²⁴ (Higham, «Accuracy and Stability
of Numerical Algorithms», §4.2). Суммы ядра — float64 тех же слагаемых, их собственная ошибка на девять
порядков меньше. Решение выносится, только если |Σ_c − Σ_s| > γ_{k+TERM_ALLOWANCE}·(Σ_c + Σ_s), k — число
элементов строк пары: тогда знак разности у исходного прохода заведомо тот же. Остальные пары (их единицы
на шаг) помечаются «не определено».

Пары «не определено». Их суммы нужны такими, какими их даёт редукция float32 исходного прохода. Порядок
сложения у неё зависит от ширины строки тензора, выравнивания начала строки (16 байт) и — если строк
меньше MINI_ROWS — от числа строк. Поэтому слагаемые такой пары раскладываются (тем же ядром слагаемых)
в нулевой участок [2·MINI_ROWS, ширина]: φ_c — на строку первой половины с тем же выравниванием, что у строки
пары в порции исходного прохода, φ_s — на ту же строку второй половины; участок суммирует та же редукция
torch sum(dim=1). Совпадение сумм малого участка с суммами порции исходной формы проверяется на случайных
строках для каждой встреченной формы порции (calibrated); первые расчёты и затем каждый CHECK_EVERY-й
сверяются с расчётом SparseLongEnergy на тех же парах. Раскладка поэтому совпадает с исходной шаг в шаг.

Пары точек с одинаковыми кодами. У таких точек строки sim_λ совпадают (сходство — функция кодов, §2.2.4),
поэтому в исходном проходе элементы сумм φ_c и φ_s равны поэлементно: s1 = s2 и s1·d1 + s2·d2 = s2·d1 + s1·d2.
Одна и та же редукция по одинаковым тензорам даёт одно и то же число, разность — точный ноль, обмена нет
(§5.6). Такие пары получают статус EQUAL без расчёта: в таблице строк их точки помечены как отсутствующие.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from damp.article_refs import ENERGY_LONG, ENERGY_SHORT, GPU_IMPLEMENTATION, POINT_EXCHANGE
from damp.layout.device_kernel import DeviceKernel, StagedTable
from damp.layout.pair_energy_gpu import TorchPairEnergyEngine
from damp.layout.potential_bounds import PotentialBounds
from damp.layout.sparse_long_energy import SimilarityRows
from damp.logging import LOGGER


class PairSignKernel:
    """Знак φ_c − φ_s пар по строкам SimilarityRows: +1, −1, EQUAL (обе суммы нули) или 0 — «не определено».

    status_from_table — решение по парам шага: дальняя фаза по всему V (§5.5.1) или окрестность пары (§5.5.2).
    reference_sums — суммы φ_c, φ_s пар «не определено» редукцией float32 исходной формы. Решение выносится
    только там, где оно заведомо совпадает с исходным проходом TorchPairEnergyEngine.
    """

    # Обе суммы — точные нули (у точек пары нет похожих): разность равна нулю и у исходного прохода.
    EQUAL = 2
    # φ_c < φ_s следует из нижней границы потенциала точек пары (PotentialBounds): строки пары не считались.
    BOUNDED = -3
    # Множитель оценки ошибки суммирования float32: сама оценка γ строгая, запас задаёт TERM_ALLOWANCE.
    BOUND_SAFETY = 1.0
    # Запас в числе слагаемых оценки γ_{k + запас}: покрывает ошибку сумм float64 ядра и по одному шагу
    # сетки float32 на слагаемое, если операция устройства округлит иначе, чем у исходного прохода.
    TERM_ALLOWANCE = 8
    # Столбцы таблицы строк (int32): точка строки, вторая точка пары (−1 — её нет), своя ячейка (y, x),
    # ячейка второй точки (y, x).
    ROW_COLUMNS = 6
    # Строк в половине малого участка редукции (φ_c — первая половина, φ_s — вторая): с этого числа строк
    # порядок сложения редукции torch внутри строки от числа строк тензора не зависит (проверяется
    # на случайных строках, calibrated).
    MINI_ROWS = 16
    # Выравнивание начала строки, от которого зависит порядок сложения редукции: 16 байт = 4 элемента float32.
    ALIGN_ELEMENTS = 4
    # Начало каждого участка буфера кратно стольким элементам (512 байт), как начало отдельного тензора.
    SLOT_ALIGN = 128
    # Столько первых расчётов пар «не определено» и затем каждый CHECK_EVERY-й сверяются со SparseLongEnergy.
    CHECK_FIRST = 8
    CHECK_EVERY = 256
    # Случайных строк на проверку формы участка и ненулевых элементов в строке.
    CALIBRATION_ROWS = 12
    CALIBRATION_NONZERO = 384

    _KERNEL_CACHE: dict[Any, DeviceKernel] = {}

    def __init__(self, engine: TorchPairEnergyEngine, rows: SimilarityRows, *, segment: int) -> None:
        if segment <= 0:
            raise ValueError("segment must be positive")
        self._engine = engine
        self._rows = rows
        self._torch = engine.torch_mod
        self._device = engine.torch_device
        self._segment = int(segment)
        self._row_index: Any = None
        self._part_index: Any = None
        self._pair_index: Any = None
        # Таблицы шага в закреплённой памяти: пары шага и пары «не определено».
        self._pair_stage = StagedTable(self._torch, self._device)
        self._reference_stage = StagedTable(self._torch, self._device, capacity=1 << 12)
        # Формы порций исходного прохода (строк, ширина), для которых малый участок даёт те же суммы.
        self._calibration: dict[tuple[int, int], bool] = {}
        self._slots_enabled = True
        self._reference_calls = 0
        LOGGER.event(
            "layout.pair_kernel.init",
            section=GPU_IMPLEMENTATION,
            data={
                "points": engine.point_count,
                "segment": self._segment,
                "term_allowance": self.TERM_ALLOWANCE,
                "mini_rows": self.MINI_ROWS,
                "note": "знак φ_c − φ_s пар (§5.6) по ненулевым sim_λ: поток на отрезок строки сходства (§5.12), "
                "частичные суммы float64 слагаемых float32, решение по оценке ошибки суммирования; пары у нуля "
                "разности — редукцией исходного прохода по малым участкам той же формы строки",
            },
        )

    # ------------------------------------------------------------------ применимость
    @classmethod
    def supported(cls, engine: TorchPairEnergyEngine) -> bool:
        """Ядра компилируются только для CUDA (jiterator)."""
        if engine.torch_device.type != "cuda":
            return False
        try:
            from torch.cuda import jiterator  # noqa: F401
        except Exception:  # noqa: BLE001
            return False
        return True

    @property
    def rows(self) -> SimilarityRows:
        return self._rows

    @property
    def segment(self) -> int:
        return self._segment

    # ------------------------------------------------------------------ тексты ядер
    @staticmethod
    def _row_source(kind: str, euclidean: bool, segment: int, sink: str) -> tuple[str, str]:
        """Имя и текст ядра отрезка строки. В теле нет знака «больше»: по нему jiterator ищет конец заголовка.

        kind — «long» (§5.5.1: s·d) или «neighbourhood» (§5.5.2: s/(d + ε) по точкам круга). sink — куда идут
        слагаемые: «partial» — частичные суммы отрезка (float64), «scatter» — элементы участков буфера редукции
        (место — номер точки v либо номер её ячейки в окне пары). Операции над float32 — отдельными
        округлениями (__fmul_rn, __fadd_rn, __fdiv_rn), без слияния умножения и сложения: каждое слагаемое
        побитово то же, что у исходного прохода.
        """
        metric = "sqrtf({0})" if euclidean else "{0}"
        if kind == "long":
            # §5.5.1: s·d до своей ячейки (в φ_c) и до ячейки второй точки (в φ_s); d — евклидово или d² (§5.9.2).
            prologue = ""
            terms = (
                "        float t0 = __fmul_rn(s, " + metric.format("q0") + ");\n"
                "        float t1 = __fmul_rn(s, " + metric.format("q1") + ");\n"
            )
        elif kind == "neighbourhood":
            # §5.5.2: точки круга ‖v − c‖ ≤ r, c — середина пары, r = max(⌈‖p1 − p2‖⌉ + 1, local_radius);
            # вес 1/(d + ε), d — евклидово или d² (§5.9.2).
            prologue = (
                "      float cy = __fmul_rn(__fadd_rn(y0, y1), 0.5f);\n"
                "      float cx = __fmul_rn(__fadd_rn(x0, x1), 0.5f);\n"
                "      float aby = y0 - y1;\n"
                "      float abx = x0 - x1;\n"
                "      float radius = ceilf(sqrtf(__fadd_rn(__fmul_rn(aby, aby), __fmul_rn(abx, abx)))) + 1.0f;\n"
                "      if (radius < (float)local_radius) radius = (float)local_radius;\n"
                "      float radius_sq = __fmul_rn(radius, radius);\n"
            )
            terms = (
                "        float oy = vy - cy;\n"
                "        float ox = vx - cx;\n"
                "        if (radius_sq < __fadd_rn(__fmul_rn(oy, oy), __fmul_rn(ox, ox))) continue;\n"
                "        float t0 = __fmul_rn(s, __fdiv_rn(1.0f, __fadd_rn(" + metric.format("q0") + ", (float)eps)));\n"
                "        float t1 = __fmul_rn(s, __fdiv_rn(1.0f, __fadd_rn(" + metric.format("q1") + ", (float)eps)));\n"
            )
        else:
            raise ValueError("kind must be 'long' or 'neighbourhood'")
        name = f"damp_{kind}_{sink}_{'e' if euclidean else 'q'}{segment}"
        common = (
            "T row, T part, long long rows_ptr, long long rowptr_ptr, long long col_ptr, long long val_ptr,\n"
            "    long long pos_ptr, double local_radius, double eps"
        )
        if sink == "partial":
            # Частичные суммы отрезка пишутся по адресу (строка, отрезок) → (Σ_c, Σ_s) float64; строки
            # отключённых пар (точка −1) не считаются и не пишутся.
            header = "template <typename T> T " + name + "(" + common + ", long long out_ptr, long long parts) {\n"
            setup = "  double acc_c = 0.0;\n  double acc_s = 0.0;\n"
            store = "        acc_c += (double)t0;\n        acc_s += (double)t1;\n"
            row_end = (
                "    double* out = (double*)out_ptr + 2 * ((long long)row * parts + (long long)part);\n"
                "    out[0] = acc_c;\n"
                "    out[1] = acc_s;\n"
            )
            before_loop = ""
        elif sink == "scatter":
            header = (
                "template <typename T> T " + name + "(" + common + ", long long base_ptr, long long c_ptr,\n"
                "    long long s_ptr, long long reach) {\n"
            )
            setup = ""
            # Место слагаемого в строке участка: номер точки v (сумма по всему V) либо номер её ячейки в окне
            # (2R + 1)² вокруг ячейки середины пары, как в проходе по окну ячеек.
            before_loop = (
                "      long long base = ((const int*)base_ptr)[(long long)row];\n"
                "      long long corner_y = ((long long)item[2] + (long long)item[4]) / 2 - reach;\n"
                "      long long corner_x = ((long long)item[3] + (long long)item[5]) / 2 - reach;\n"
            )
            store = (
                "        long long place = v;\n"
                "        if (0 <= reach) place = ((long long)vy - corner_y) * (2 * reach + 1) + ((long long)vx - corner_x);\n"
                "        atomicAdd((float*)c_ptr + base + place, t0);\n"
                # s_ptr = 0 — нужна только φ_c (энергии точек §5.7: второй точки нет).
                "        if (s_ptr != 0) atomicAdd((float*)s_ptr + base + place, t1);\n"
            )
            row_end = ""
        else:
            raise ValueError("sink must be 'partial' or 'scatter'")
        source = (
            header
            + "  const int* item = (const int*)rows_ptr + 6 * (long long)row;\n"
            "  const long long* rowptr = (const long long*)rowptr_ptr;\n"
            "  const int* col = (const int*)col_ptr;\n"
            "  const float* val = (const float*)val_ptr;\n"
            "  const float* pos = (const float*)pos_ptr;\n"
            "  int point = item[0];\n"
            + setup
            + "  if (0 <= point) {\n"
            "    long long start = rowptr[point];\n"
            "    long long degree = rowptr[point + 1] - start;\n"
            "    long long first = (long long)part * " + str(segment) + ";\n"
            "    if (first < degree) {\n"
            "      long long last = first + " + str(segment) + ";\n"
            "      if (degree < last) last = degree;\n"
            "      int other = item[1];\n"
            "      float y0 = (float)item[2];\n"
            "      float x0 = (float)item[3];\n"
            "      float y1 = (float)item[4];\n"
            "      float x1 = (float)item[5];\n"
            + prologue
            + before_loop
            + "      for (long long j = start + first; j < start + last; ++j) {\n"
            "        int v = col[j];\n"
            "        if (v == other) continue;\n"
            "        float s = val[j];\n"
            "        float vy = pos[2 * v];\n"
            "        float vx = pos[2 * v + 1];\n"
            "        float dy0 = y0 - vy;\n"
            "        float dx0 = x0 - vx;\n"
            "        float dy1 = y1 - vy;\n"
            "        float dx1 = x1 - vx;\n"
            "        float q0 = __fadd_rn(__fmul_rn(dy0, dy0), __fmul_rn(dx0, dx0));\n"
            "        float q1 = __fadd_rn(__fmul_rn(dy1, dy1), __fmul_rn(dx1, dx1));\n"
            + terms
            + store
            + "      }\n"
            "    }\n"
            + row_end
            + "  }\n"
            "  return (T)0;\n"
            "}\n"
        )
        return name, source

    @staticmethod
    def _decide_source() -> str:
        """Ядро решения: суммы пары по частичным суммам двух её строк (float64) и знак по оценке ошибки.

        Пара, помеченная границей потенциала, получает BOUNDED; пара с отключёнными строками (одинаковые
        коды) — EQUAL; у пары без второй точки читается только строка первой.
        """
        return (
            "template <typename T> T damp_pair_decide(T index, long long out_ptr, long long rows_ptr, long long terms_ptr,\n"
            "    long long bounded_ptr, long long pairs, long long parts, double bound, double extra) {\n"
            "  long long i = (long long)index;\n"
            "  if (bounded_ptr != 0) {\n"
            "    if (((const int*)bounded_ptr)[i] != 0) return (T)(-3);\n"
            "  }\n"
            "  const int* rows = (const int*)rows_ptr;\n"
            "  if (rows[6 * i] < 0) return (T)2;\n"
            "  const double* partial = (const double*)out_ptr;\n"
            "  const int* terms = (const int*)terms_ptr;\n"
            "  double sum_c = 0.0;\n"
            "  double sum_s = 0.0;\n"
            "  for (long long side = 0; side < 2; ++side) {\n"
            "    long long row = i + side * pairs;\n"
            "    if (rows[6 * row] < 0) continue;\n"
            "    const double* base = partial + 2 * row * parts;\n"
            "    for (long long g = 0; g < parts; ++g) {\n"
            "      sum_c += base[2 * g];\n"
            "      sum_s += base[2 * g + 1];\n"
            "    }\n"
            "  }\n"
            "  double total = sum_c + sum_s;\n"
            "  double diff = sum_c - sum_s;\n"
            "  double k = ((double)terms[i] + extra) * 5.9604644775390625e-08;\n"
            "  double limit = bound * (k / (1.0 - k)) * total;\n"
            "  int status = 0;\n"
            "  if (total == 0.0) status = 2;\n"
            "  else if (limit < diff) status = 1;\n"
            "  else if (diff < -limit) status = -1;\n"
            "  return (T)status;\n"
            "}\n"
        )

    @classmethod
    def _row_kernel(cls, kind: str, euclidean: bool, segment: int, sink: str) -> DeviceKernel:
        key = (kind, bool(euclidean), int(segment), sink)
        cached = cls._KERNEL_CACHE.get(key)
        if cached is None:
            from torch.cuda.jiterator import _create_jit_fn

            _, source = cls._row_source(kind, euclidean, segment, sink)
            scalars = dict(rows_ptr=0, rowptr_ptr=0, col_ptr=0, val_ptr=0, pos_ptr=0, local_radius=0.0, eps=0.0)
            if sink == "partial":
                function = _create_jit_fn(source, **scalars, out_ptr=0, parts=1)
            else:
                # Порядок скаляров — как в заголовке ядра: jiterator передаёт их по порядку объявления здесь.
                function = _create_jit_fn(source, **scalars, base_ptr=0, c_ptr=0, s_ptr=0, reach=-1)
            cached = DeviceKernel(function)
            cls._KERNEL_CACHE[key] = cached
        return cached

    @classmethod
    def _decide_kernel(cls) -> DeviceKernel:
        cached = cls._KERNEL_CACHE.get("decide")
        if cached is None:
            from torch.cuda.jiterator import _create_jit_fn

            cached = DeviceKernel(
                _create_jit_fn(
                    cls._decide_source(), out_ptr=0, rows_ptr=0, terms_ptr=0, bounded_ptr=0, pairs=0, parts=0,
                    bound=1.0, extra=0.0,
                )
            )
            cls._KERNEL_CACHE["decide"] = cached
        return cached

    # ------------------------------------------------------------------ решение по парам шага
    def _indices(self, rows: int, parts: int) -> tuple[Any, Any, Any]:
        """Номера строк, отрезков и пар на устройстве (входы ядер); растут по мере надобности."""
        torch = self._torch
        if self._row_index is None or int(self._row_index.shape[0]) < rows:
            size = max(rows, 1024)
            self._row_index = torch.arange(size, dtype=torch.int32, device=self._device).unsqueeze(1)
            self._pair_index = torch.arange(size, dtype=torch.int32, device=self._device)
        if self._part_index is None or int(self._part_index.shape[1]) < parts:
            self._part_index = torch.arange(max(parts, 256), dtype=torch.int32, device=self._device).unsqueeze(0)
        return self._row_index[:rows], self._part_index[:, :parts], self._pair_index[: rows // 2]

    def stage(self, pairs: int) -> "np.ndarray":
        """Место под таблицу строк пар шага (int32, 13 значений на пару) в закреплённой памяти хоста."""
        return self._pair_stage.host((2 * self.ROW_COLUMNS + 1) * pairs)

    def pair_table(
        self,
        idx_a: "np.ndarray",
        idx_b: "np.ndarray",
        pos_a: "np.ndarray",
        pos_b: "np.ndarray",
        code_class: "np.ndarray | None" = None,
    ) -> tuple["np.ndarray", int]:
        """Таблица строк пар для ядра (int32) и наибольшая степень строки.

        Сначала строки точек a всех пар, затем строки точек b (точка −1 — второй точки нет); за ними —
        число элементов строк каждой пары k = deg a + deg b для оценки ошибки суммирования. code_class —
        номер класса одинаковых кодов каждой точки: у пары точек одного класса суммы равны точно, её
        строки помечаются как отсутствующие (статус EQUAL без расчёта).
        """
        pairs = int(idx_a.size)
        columns = self.ROW_COLUMNS
        table = np.empty(2 * pairs * columns + pairs, dtype=np.int32)
        body = table[: 2 * pairs * columns].reshape(2 * pairs, columns)
        body[:pairs, 0] = idx_a
        body[:pairs, 1] = idx_b
        body[:pairs, 2:4] = pos_a
        body[:pairs, 4:6] = pos_b
        body[pairs:, 0] = idx_b
        body[pairs:, 1] = idx_a
        body[pairs:, 2:4] = pos_b
        body[pairs:, 4:6] = pos_a
        degree = self._rows.degree_np
        degree_a = degree[idx_a]
        degree_b = np.where(idx_b >= 0, degree[np.maximum(idx_b, 0)], 0)
        table[2 * pairs * columns :] = degree_a + degree_b
        if code_class is not None and pairs:
            same = (idx_b >= 0) & (code_class[idx_a] == code_class[np.maximum(idx_b, 0)])
            if same.any():
                body[:pairs, 0][same] = -1
                body[pairs:, 0][same] = -1
                table[2 * pairs * columns :][same] = 0
        widest = int(max(degree_a.max(), degree_b.max())) if pairs else 0
        return table, widest

    def status_from_table(
        self,
        table: "np.ndarray",
        pairs: int,
        widest: int,
        *,
        kind: str,
        euclidean: bool,
        local_radius: float = 0.0,
        distance_eps: float = 0.0,
        bounds: PotentialBounds | None = None,
    ) -> "np.ndarray":
        """Статусы пар по готовой таблице строк (pair_table): +1, −1, EQUAL, BOUNDED или 0 — «не определено».

        bounds — действующие границы потенциала (дальняя фаза): пары, у которых φ_c < φ_s следует из границы,
        получают BOUNDED, их строки не считаются.
        """
        if pairs == 0:
            return np.zeros(0, dtype=np.int32)
        torch = self._torch
        rows = self._rows
        segment = self._segment
        parts = max(1, -(-widest // segment))
        with torch.no_grad():
            size = int(table.size)
            if table is not self._pair_stage.view:
                # Таблица собрана не в закреплённой памяти (stage): переносится туда.
                self._pair_stage.host(size)[:] = table
            device_table = self._pair_stage.upload(size)
            row_index, part_index, pair_index = self._indices(2 * pairs, parts)
            base = device_table.data_ptr()
            bounded = None if bounds is None else bounds.certify(base, pairs)
            # Частичные суммы: на (строку, отрезок) — Σ_c и Σ_s; пишутся только строки неотключённых пар.
            partial = torch.empty(2 * 2 * pairs * parts, dtype=torch.float64, device=self._device)
            self._row_kernel(kind, euclidean, segment, "partial")(
                row_index,
                part_index,
                rows_ptr=base,
                rowptr_ptr=rows.row_ptr.data_ptr(),
                col_ptr=rows.col.data_ptr(),
                val_ptr=rows.val.data_ptr(),
                pos_ptr=self._engine.positions.data_ptr(),
                local_radius=float(local_radius),
                eps=float(distance_eps),
                out_ptr=partial.data_ptr(),
                parts=parts,
            )
            status = self._decide_kernel()(
                pair_index,
                out_ptr=partial.data_ptr(),
                rows_ptr=base,
                terms_ptr=base + 4 * 2 * pairs * self.ROW_COLUMNS,
                bounded_ptr=0 if bounded is None else bounded.data_ptr(),
                pairs=pairs,
                parts=parts,
                bound=self.BOUND_SAFETY,
                extra=float(self.TERM_ALLOWANCE),
            )
            # Выгрузка ждёт все ядра: таблица, отметки границы и частичные суммы живут до этого места.
            return status.cpu().numpy()

    # ------------------------------------------------------------------ суммы редукцией исходной формы
    def _calibrated(self, rows_full: int, width: int) -> bool:
        """Дают ли строки малого участка [2·MINI_ROWS, width] те же суммы, что строки порции [rows_full, width].

        Проверяется на случайных разреженных строках: суммы sum(dim=1) порции исходной формы и малого участка
        со строкой на месте с тем же выравниванием должны совпасть побитово. Разный порядок сложения на таких
        данных даёт разные числа, поэтому совпадение на всех пробных строках означает один и тот же порядок.
        """
        key = (int(rows_full), int(width))
        known = self._calibration.get(key)
        if known is not None:
            return known
        torch = self._torch
        mini = self.MINI_ROWS
        matched = rows_full >= mini
        if matched:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(1_000_003 * rows_full + width)
            tested = sorted(
                {
                    row % rows_full
                    for row in (0, 1, 2, 3, 4, 5, 6, 7, rows_full // 2, rows_full // 3,
                                rows_full - 1, rows_full - 2, rows_full - 3, rows_full - 4)
                }
            )[: self.CALIBRATION_ROWS]
            nonzero = min(width, self.CALIBRATION_NONZERO)
            with torch.no_grad():
                full = torch.zeros((rows_full, width), dtype=torch.float32, device=self._device)
                samples = []
                for row in tested:
                    columns = torch.randperm(width, generator=generator)[:nonzero].to(self._device)
                    # Слагаемые разного порядка величины: округление суммы чувствительно к порядку сложения.
                    values = (torch.rand(nonzero, generator=generator) * 4.0).exp().to(torch.float32).to(self._device)
                    full[row, columns] = values
                    samples.append((row, columns, values))
                expected = full.sum(dim=1)
                del full
                for row, columns, values in samples:
                    # Первая и последняя строки малого участка с тем же выравниванием, что у строки порции.
                    candidates = self._aligned_rows(row, width)
                    for place in {candidates[0], candidates[-1]}:
                        small = torch.zeros((2 * mini, width), dtype=torch.float32, device=self._device)
                        small[place, columns] = values
                        small[mini + place, columns] = values
                        sums = small.sum(dim=1)
                        if not bool(((sums[place] == expected[row]) & (sums[mini + place] == expected[row])).item()):
                            matched = False
                    if not matched:
                        break
        self._calibration[key] = matched
        LOGGER.event(
            "layout.pair_kernel.calibration",
            section=GPU_IMPLEMENTATION,
            data={
                "rows": int(rows_full),
                "width": int(width),
                "mini_rows": mini,
                "identical": matched,
                "note": "суммы строки в малом участке редукции и в порции исходной формы (§5.12): identical — "
                "порядок сложения float32 тот же, пары у нуля разности считаются по малым участкам",
            },
        )
        return matched

    def _aligned_rows(self, row: int, width: int) -> list[int]:
        """Строки половины малого участка, выровненные так же, как строка row порции исходного прохода.

        Вторая половина участка начинается через MINI_ROWS строк: MINI_ROWS кратно ALIGN_ELEMENTS, поэтому
        строка place + MINI_ROWS выровнена так же, как строка place.
        """
        align = self.ALIGN_ELEMENTS
        target = (row * width) % align
        return [place for place in range(self.MINI_ROWS) if (place * width) % align == target]

    def reference_geometry(
        self, pos_a: "np.ndarray", pos_b: "np.ndarray", *, kind: str, local_radius: float | None
    ) -> tuple[int, int, int]:
        """Форма редукции исходного прохода для пар шага: (ширина строки, пар в порции, полуширина окна или −1).

        Дальняя фаза и окрестность с окном не меньше всего V суммируются по всем точкам порциями плотного
        прохода; окрестность — по окну (2R + 1)² ячеек вокруг середины пары порциями прохода по окну.
        """
        engine = self._engine
        if kind == "long":
            return engine.point_count, engine.dense_chunk_rows(), -1
        delta = (np.asarray(pos_a) - np.asarray(pos_b)).astype(np.float32)
        distance = np.sqrt((delta * delta).sum(axis=1, dtype=np.float32))
        radius = np.ceil(distance) + np.float32(1.0)
        if local_radius is not None:
            radius = np.maximum(radius, np.float32(local_radius))
        reach = int(math.ceil(float(radius.max()))) + 1
        cells = (2 * reach + 1) ** 2
        if cells >= engine.point_count:
            return engine.point_count, engine.dense_chunk_rows(), -1
        return cells, engine.local_chunk_rows(), reach

    def reference_sums(
        self,
        idx_a: "np.ndarray",
        idx_b: "np.ndarray",
        pos_a: "np.ndarray",
        pos_b: "np.ndarray",
        pending: "np.ndarray",
        *,
        kind: str,
        euclidean: bool,
        local_radius: float | None = None,
        distance_eps: float = 0.0,
    ) -> tuple["np.ndarray", "np.ndarray"] | None:
        """(φ_c, φ_s) пар pending — те же числа float32, что у исходного прохода; None — малые участки неприменимы.

        Слагаемые пары ставятся в нулевой участок [2·MINI_ROWS, ширина]: φ_c — на строку первой половины
        с тем же выравниванием, что у строки пары в порции исходного прохода (пара i стоит на строке
        i mod порция), φ_s — на ту же строку второй половины; участок суммирует та же редукция sum(dim=1).
        В элемент участка попадает не больше двух слагаемых (строки a и строки b), первое — в точный ноль,
        поэтому элемент равен fl(s1·d1 + s2·d2), как в исходном проходе.
        """
        if not self._slots_enabled or pending.size == 0:
            return None
        torch = self._torch
        pair_count = int(idx_a.size)
        width, chunk, reach = self.reference_geometry(pos_a, pos_b, kind=kind, local_radius=local_radius)
        chunk_of = pending // chunk
        row_of = pending - chunk_of * chunk
        chunks = -(-pair_count // chunk)
        last_rows = pair_count - (chunks - 1) * chunk
        rows_full = np.where(chunk_of == chunks - 1, last_rows, chunk)
        for shape_rows in np.unique(rows_full).tolist():
            if not self._calibrated(int(shape_rows), width):
                return None
        mini = self.MINI_ROWS
        stride = -(-(2 * mini * width) // self.SLOT_ALIGN) * self.SLOT_ALIGN
        # Место пары: участок и строка в нём среди строк с нужным выравниванием.
        taken: dict[int, int] = {}
        slot = np.empty(pending.size, dtype=np.int64)
        place = np.empty(pending.size, dtype=np.int64)
        align = self.ALIGN_ELEMENTS
        for index, row in enumerate(row_of.tolist()):
            candidates = self._aligned_rows(row, width)
            key = (row * width) % align
            used = taken.get(key, 0)
            taken[key] = used + 1
            slot[index] = used // len(candidates)
            place[index] = candidates[used % len(candidates)]
        slots = int(slot.max()) + 1
        if slots * stride >= 2**31:
            return None
        select_a = idx_a[pending]
        select_b = idx_b[pending]
        count = int(pending.size)
        columns = self.ROW_COLUMNS
        # Одна загрузка: таблица строк пар и за ней место строки каждой пары в буфере участков.
        packed = self._reference_stage.host(2 * count * (columns + 1))
        table = packed[: 2 * count * columns].reshape(2 * count, columns)
        table[:count, 0] = select_a
        table[:count, 1] = select_b
        table[:count, 2:4] = pos_a[pending]
        table[:count, 4:6] = pos_b[pending]
        table[count:, 0] = select_b
        table[count:, 1] = select_a
        table[count:, 2:4] = pos_b[pending]
        table[count:, 4:6] = pos_a[pending]
        packed[2 * count * columns :] = np.tile(slot * stride + place * width, 2)
        degree = self._rows.degree_np
        widest = int(max(degree[select_a].max(), degree[np.maximum(select_b, 0)].max()))
        segment = self._segment
        parts = max(1, -(-widest // segment))
        rows = self._rows
        with torch.no_grad():
            device_table = self._reference_stage.upload(int(packed.size))
            buffer = torch.zeros((slots, stride), dtype=torch.float32, device=self._device)
            row_index, part_index, _ = self._indices(2 * count, parts)
            table_address = device_table.data_ptr()
            self._row_kernel(kind, euclidean, segment, "scatter")(
                row_index,
                part_index,
                rows_ptr=table_address,
                rowptr_ptr=rows.row_ptr.data_ptr(),
                col_ptr=rows.col.data_ptr(),
                val_ptr=rows.val.data_ptr(),
                pos_ptr=self._engine.positions.data_ptr(),
                local_radius=0.0 if local_radius is None else float(local_radius),
                eps=float(distance_eps),
                base_ptr=table_address + 4 * 2 * count * columns,
                c_ptr=buffer.data_ptr(),
                # φ_s — вторая половина участка: те же места через MINI_ROWS строк.
                s_ptr=buffer.data_ptr() + 4 * mini * width,
                reach=int(reach),
            )
            sums = [
                buffer[index, : 2 * mini * width].view(2 * mini, width).sum(dim=1) for index in range(slots)
            ]
            values = (sums[0] if slots == 1 else torch.cat(sums)).cpu().numpy()
        energy_c = values[slot * 2 * mini + place]
        energy_s = values[slot * 2 * mini + mini + place]
        return energy_c, energy_s

    def point_energies(self, cells_of_points: "np.ndarray", radius: float, *, distance_eps: float) -> "np.ndarray | None":
        """E(c, r) = Σ sim_λ(c, v) / (d + ε) по точкам круга радиуса r (§5.7) для всех точек — numpy float32.

        Те же числа, что у прохода TorchPairEnergyEngine по окну ячеек: ядро раскладывает слагаемые строки
        сходства точки по ячейкам её окна (2R + 1)², блоки строк той же формы суммирует та же редукция порциями.
        cells_of_points — ячейки точек (N × 2, целые). None — окно не меньше всего V (там проход идёт по всем
        точкам) или блок не помещается в адресацию int32.
        """
        torch = self._torch
        engine = self._engine
        count = engine.point_count
        reach = int(math.ceil(float(radius))) + 1
        cells = (2 * reach + 1) ** 2
        if count == 0 or cells >= count:
            return None
        chunk = engine.local_chunk_rows()
        block = engine.window_block_rows(cells)
        if block * cells >= 2**31:
            return None
        rows = self._rows
        segment = self._segment
        parts = max(1, -(-rows.max_degree // segment))
        columns = self.ROW_COLUMNS
        packed = np.empty(count * (columns + 1), dtype=np.int32)
        table = packed[: count * columns].reshape(count, columns)
        table[:, 0] = np.arange(count)
        table[:, 1] = -1
        table[:, 2:4] = cells_of_points
        table[:, 4:6] = cells_of_points
        # Место строки точки в буфере её блока: (номер в блоке)·(ячейки окна).
        packed[count * columns :] = (np.arange(count) % block) * cells
        sums = []
        with torch.no_grad():
            device_table = torch.as_tensor(packed, device=self._device)
            address = device_table.data_ptr()
            kernel = self._row_kernel("neighbourhood", True, segment, "scatter")
            for start in range(0, count, block):
                stop = min(count, start + block)
                buffer = torch.zeros((stop - start, cells), dtype=torch.float32, device=self._device)
                row_index, part_index, _ = self._indices(count, parts)
                kernel(
                    row_index[start:stop],
                    part_index,
                    rows_ptr=address,
                    rowptr_ptr=rows.row_ptr.data_ptr(),
                    col_ptr=rows.col.data_ptr(),
                    val_ptr=rows.val.data_ptr(),
                    pos_ptr=engine.positions.data_ptr(),
                    local_radius=float(radius),
                    eps=float(distance_eps),
                    base_ptr=address + 4 * count * columns,
                    c_ptr=buffer.data_ptr(),
                    s_ptr=0,
                    reach=reach,
                )
                sums.append(engine.sum_rows_by_chunks(buffer, chunk))
            return (sums[0] if len(sums) == 1 else torch.cat(sums)).cpu().numpy()

    def reference_check_due(self) -> bool:
        """Пора ли сверить суммы малых участков с расчётом SparseLongEnergy: первые расчёты и затем периодически."""
        calls = self._reference_calls
        self._reference_calls = calls + 1
        return calls < self.CHECK_FIRST or calls % self.CHECK_EVERY == 0

    def reference_checked(self, matched: bool, *, kind: str, pairs: int) -> None:
        """Итог сверки сумм малых участков; при расхождении малые участки отключаются до конца работы."""
        LOGGER.event(
            "layout.pair_kernel.reference.verify",
            section=POINT_EXCHANGE,
            data={
                "kind": kind,
                "pairs": pairs,
                "call": self._reference_calls - 1,
                "bitwise_equal": matched,
                "note": "суммы φ_c, φ_s пар у нуля разности по малым участкам редукции сверены с расчётом по порциям "
                "исходной формы (§5.5.1, §5.5.2): обмены §5.6 совпадают при побитовом равенстве сумм",
            },
            force=True,
        )
        if not matched:
            self._slots_enabled = False

    def log_status(self, status: "np.ndarray", *, kind: str) -> None:
        """Итог расчёта шага в журнал: сколько пар решено ядром и сколько осталось редукции исходного прохода."""
        event = "layout.pair_kernel.status"
        if not LOGGER.should_log(event):
            return
        LOGGER.event(
            event,
            section=ENERGY_LONG if kind == "long" else ENERGY_SHORT,
            data={
                "kind": kind,
                "pairs": int(status.size),
                "greater": int((status == 1).sum()),
                "less": int((status == -1).sum()),
                "bounded": int((status == self.BOUNDED).sum()),
                "equal": int((status == self.EQUAL).sum()),
                "undecided": int((status == 0).sum()),
                "segment": self._segment,
                "lambda_threshold": self._rows.lambda_threshold,
                "note": "знак φ_c − φ_s по ненулевым sim_λ (§5.5.1, §5.5.2): greater — φ_c > φ_s, less — φ_c < φ_s; "
                "bounded — φ_c < φ_s по нижней границе потенциала точек пары, строки не считались; "
                "undecided — разность в пределах ошибки суммирования float32, суммы берёт редукция исходного прохода",
            },
            force=True,
        )
