"""Строгая нижняя граница потенциала точки (§5.5.1): пары, которым обмен заведомо невыгоден, без суммы по строкам.

Потенциал. При жёстком пороге (η = ∞, §7.1.7) дальняя энергия пары (§5.5.1) складывается из потенциалов её точек:

    U_a(p) = Σ_v sim_λ(a, v)·‖p − p_v‖        (сумма по строке сходства точки a),
    φ_c − φ_s = [U_a(p_a) − U_a(p_b)] + [U_b(p_b) − U_b(p_a)] − 2·sim_λ(a, b)·‖p_a − p_b‖.

Обмен выгоден при φ_c > φ_s (§5.6). В разложенном пространстве почти все пары шага невыгодны с большим запасом
(доля обменов падает до процента и ниже, §7.1.7), а цена пары — сумма по всем похожим точкам (§5.11).

Граница. Расстояние ‖p − q‖ — выпуклая функция q, поэтому для любой группы точек с массами m_v
(неравенство Йенсена)

    Σ_v m_v·‖p − p_v‖ ≥ ‖M·p − P‖,        M = Σ m_v,  P = Σ m_v·p_v.

Похожие точки a делятся на группы по ячейкам грубой сетки GRID × GRID; для каждой точки хранятся массы M
и моменты P её групп. Тогда U_a(p) ≥ L_a(p) = Σ_ячейки ‖M·p − P‖ для любой ячейки p — за GRID² операций
вместо длины строки. Собственная энергия E_a = U_a(p_a) хранится точно. Отсюда

    φ_c − φ_s ≤ [E_a − L_a(p_b)] + [E_b − L_b(p_a)] + поправки,

и если правая часть отрицательна с запасом на ошибку суммирования float32 исходного прохода (γ_k·(φ_c + φ_s),
см. PairSignKernel), то и у исходного прохода φ_c < φ_s: обмена нет, строки пары можно не считать.
Граница односторонняя: она только снимает заведомо невыгодные пары, остальные считаются как раньше,
поэтому раскладка совпадает с исходной шаг в шаг.

Массы. m_v — сходство sim_λ(a, v), округлённое вниз до целого числа долей 2^−MASS_BITS (меньшая масса только
ослабляет границу). Масса считается из числа общих бит кодов и чисел их единиц (§2.2.4) симметрично по паре
точек, поэтому у строки a для точки v и у строки v для точки a она одна и та же: после обмена суммы точки v
у всех её похожих точек обновляются проходом по строке самой v. Целые суммы M, P обновляются точно, без
накопления ошибки. Остаток R_a = Σ_v (sim_λ − m_v) ≥ 0 входит в границу как поправка R_a·‖p_a − p_b‖
(неравенство треугольника).

Обновление. После обменов шага для каждой сдвинутой точки x (ячейка q → q′) один проход по её строке:
у каждой похожей точки y масса x переносится из группы ячейки q в группу ячейки q′, энергия E_y меняется
на m·(‖p_y − q′‖ − ‖p_y − q‖), а энергия самой x считается заново. Расстояния в этом проходе считаются
в float32 (корень двойной точности на устройстве на порядок медленнее): ошибка одного обновления энергии
точки y не больше DISTANCE_ERROR·диагональ·Σ_v sim_λ(y, v), поэтому в границу входит запас
(число обновлений с последнего построения + 1)·DISTANCE_ERROR·диагональ·(Σ sim_λ точек пары). Суммы строятся
заново (в float64) при смене λ (§7.1.7), при включении и каждые REBUILD_EVERY обновлений.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from damp.article_refs import ENERGY_LONG, GPU_IMPLEMENTATION, LAYOUT_PARAMETERS
from damp.layout.device_kernel import DeviceKernel, StagedTable
from damp.layout.pair_energy_gpu import TorchPairEnergyEngine
from damp.layout.sparse_long_energy import SimilarityRows
from damp.logging import LOGGER


class PotentialBounds:
    """Массы и моменты групп похожих точек каждой точки, её собственная энергия и проверка пар по границе Йенсена.

    apply_moves — обновление после обменов шага; certify — пометить пары, у которых φ_c < φ_s заведомо
    (их строки исключаются из расчёта PairSignKernel). Применимо к дальней фазе с евклидовым расстоянием
    (§5.5.1, §5.5): у d² (§5.9.2) граница другая.
    """

    # Масса — целое число долей 2^−MASS_BITS сходства.
    MASS_BITS = 20
    # Относительное уменьшение массы перед округлением вниз: сходство строки посчитано в float32 другим
    # порядком операций, чем симметричное значение из общих бит (расхождение — единицы 10⁻⁷).
    MASS_SHRINK = 2e-6
    # Масса отлична от нуля, только если симметричное сходство не ниже λ·(1 + LAMBDA_GUARD): тогда элемент
    # заведомо есть в строках обеих точек пары.
    LAMBDA_GUARD = 4e-6
    # Запас на ошибку вычисления самой границы в float64, в долях φ_c + φ_s.
    FLOAT_SLACK = 1e-9
    # Запас в числе слагаемых оценки ошибки суммирования float32 — тот же, что у PairSignKernel.
    TERM_ALLOWANCE = 8
    # Суммы строятся заново не реже чем раз в столько обновлений: запас на ошибку энергий растёт с их числом.
    REBUILD_EVERY = 1024
    # Относительная ошибка пары расстояний float32 в одном обновлении энергии (2·2⁻²⁴ с запасом).
    DISTANCE_ERROR = 1.25e-7
    # Нижняя граница L считается в float32 и уменьшается на эту долю: ошибка корня и суммы до 2·GRID² слагаемых.
    LOWER_SHRINK = 1e-5
    # Отрезок строки сдвинутой точки на поток ядра обновления (короче отрезка расчёта: строк мало, а на элемент
    # приходится шесть атомарных сложений).
    MOVE_SEGMENT = 4
    # Столбцы таблицы обменов (int32): точка, старая ячейка, новая ячейка (y·ширина + x).
    MOVE_COLUMNS = 3

    _KERNEL_CACHE: dict[Any, DeviceKernel] = {}

    def __init__(
        self,
        engine: TorchPairEnergyEngine,
        rows: SimilarityRows,
        *,
        grid: int,
        segment: int,
        similarity: str,
        grid_shape: tuple[int, int],
    ) -> None:
        if grid <= 0 or segment <= 0:
            raise ValueError("grid and segment must be positive")
        if similarity not in ("cosine", "jaccard"):
            raise ValueError("similarity must be 'cosine' or 'jaccard'")
        torch = engine.torch_mod
        self._engine = engine
        self._rows = rows
        self._torch = torch
        self._device = engine.torch_device
        self._grid = int(grid)
        self._segment = int(segment)
        self._jaccard = 1 if similarity == "jaccard" else 0
        self._height, self._width = int(grid_shape[0]), int(grid_shape[1])
        self._cell_h = -(-self._height // self._grid)
        self._cell_w = -(-self._width // self._grid)
        self._cells = self._grid * self._grid
        self._diagonal = math.hypot(self._height, self._width)
        count = engine.point_count
        # Группы похожих точек каждой точки: [точка, ячейка грубой сетки, (масса, момент y, момент x)].
        self._groups = torch.zeros((count, self._cells, 3), dtype=torch.int64, device=self._device)
        self._energy = torch.zeros(count, dtype=torch.float64, device=self._device)
        self._residual = torch.zeros(count, dtype=torch.float64, device=self._device)
        self._similarity_sum = torch.zeros(count, dtype=torch.float64, device=self._device)
        # Номер обновления, в котором точка сдвигалась последний раз.
        self._stamp = torch.full((count,), -1, dtype=torch.int32, device=self._device)
        self._row_index: Any = None
        self._part_index: Any = None
        self._flat_index: Any = None
        # Таблица обменов шага в закреплённой памяти хоста.
        self._stage = StagedTable(torch, self._device, capacity=1 << 12)
        self._update = 0
        self._updates_since_rebuild = 0
        self._lambda: float | None = None
        self._valid = False
        LOGGER.event(
            "layout.bounds.init",
            section=ENERGY_LONG,
            data={
                "points": count,
                "grid": self._grid,
                "cell": (self._cell_h, self._cell_w),
                "mass_bits": self.MASS_BITS,
                "memory_mb": (count * self._cells * 24 + count * 28) / 1e6,
                "note": "нижняя граница потенциала точки U_a(p) = Σ sim_λ·‖p − p_v‖ (§5.5.1) по группам похожих "
                "точек на грубой сетке (неравенство Йенсена): пары, у которых φ_c < φ_s заведомо, не считаются "
                "по строкам сходства; остальные пары — как раньше",
            },
        )

    # ------------------------------------------------------------------ применимость и состояние
    @classmethod
    def supported(cls, engine: TorchPairEnergyEngine, grid_shape: tuple[int, int]) -> bool:
        """Ядра компилируются только для CUDA; номера ячеек и точек должны помещаться в int32."""
        if engine.torch_device.type != "cuda":
            return False
        if int(grid_shape[0]) * int(grid_shape[1]) >= 2**31 or engine.point_count >= 2**31 - 2:
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
    def valid(self) -> bool:
        """Суммы соответствуют нынешним позициям точек на устройстве и порогу строк сходства."""
        return self._valid and self._lambda == self._rows.lambda_threshold

    def invalidate(self) -> None:
        """Позиции на устройстве изменились в обход apply_moves: суммы надо строить заново."""
        self._valid = False

    def _mass_scalars(self) -> dict[str, Any]:
        return {
            "lambda_guard": float(np.float32(self._rows.lambda_threshold * (1.0 + self.LAMBDA_GUARD))),
            "scale": float((1.0 - self.MASS_SHRINK) * 2.0**self.MASS_BITS),
            "unit": 2.0 ** -self.MASS_BITS,
            "jaccard": self._jaccard,
        }

    # ------------------------------------------------------------------ тексты ядер
    # Масса элемента строки: симметричное сходство из числа общих бит (§2.2.4), округлённое вниз.
    # В текстах ядер нет знака «больше»: по нему jiterator ищет конец заголовка функции.
    _MASS = (
        "    float root_ = sqrtf(__fmul_rn(na, nv));\n"
        "    float sum_ = __fadd_rn(na, nv);\n"
        "    float common_ = jaccard != 0 ? rintf(__fdiv_rn(__fmul_rn(s, sum_), __fadd_rn(1.0f, s)))\n"
        "                                 : rintf(__fmul_rn(s, root_));\n"
        "    float sym_ = jaccard != 0 ? __fdiv_rn(common_, __fsub_rn(sum_, common_)) : __fdiv_rn(common_, root_);\n"
        "    long long mu = 0;\n"
        "    if ((float)lambda_guard <= sym_) mu = (long long)floor((double)sym_ * scale);\n"
    )
    _ROW_ARGUMENTS = (
        "long long rowptr_ptr, long long col_ptr, long long val_ptr, long long pos_ptr, long long ones_ptr,\n"
        "    long long groups_ptr, long long energy_ptr, long long cells, long long cell_h, long long cell_w,\n"
        "    long long grid, double lambda_guard, double scale, double unit, long long jaccard"
    )

    @classmethod
    def _build_source(cls, segment: int) -> str:
        """Ядро построения сумм: отрезок строки точки a — массы и моменты её групп, энергия, остаток, сумма сходств."""
        return (
            "template <typename T> T damp_bound_build_" + str(segment) + "(T row, T part, " + cls._ROW_ARGUMENTS + ",\n"
            "    long long residual_ptr, long long total_ptr) {\n"
            "  long long a = (long long)row;\n"
            "  const long long* rowptr = (const long long*)rowptr_ptr;\n"
            "  long long start = rowptr[a];\n"
            "  long long degree = rowptr[a + 1] - start;\n"
            "  long long first = (long long)part * " + str(segment) + ";\n"
            "  if (!(first < degree)) return (T)0;\n"
            "  long long last = first + " + str(segment) + ";\n"
            "  if (degree < last) last = degree;\n"
            "  const int* col = (const int*)col_ptr;\n"
            "  const float* val = (const float*)val_ptr;\n"
            "  const float* pos = (const float*)pos_ptr;\n"
            "  const float* ones = (const float*)ones_ptr;\n"
            "  unsigned long long* groups = (unsigned long long*)groups_ptr + 3 * cells * a;\n"
            "  float ay = pos[2 * a];\n"
            "  float ax = pos[2 * a + 1];\n"
            "  float na = ones[a];\n"
            "  double acc_e = 0.0;\n"
            "  double acc_r = 0.0;\n"
            "  double acc_s = 0.0;\n"
            "  for (long long j = start + first; j < start + last; ++j) {\n"
            "    int v = col[j];\n"
            "    float s = val[j];\n"
            "    float nv = ones[v];\n"
            + cls._MASS
            + "    double mass = (double)mu * unit;\n"
            "    acc_s += (double)s;\n"
            "    acc_r += (double)s - mass;\n"
            "    if (0 < mu) {\n"
            "      long long vy = (long long)pos[2 * v];\n"
            "      long long vx = (long long)pos[2 * v + 1];\n"
            "      unsigned long long* group = groups + 3 * ((vy / cell_h) * grid + vx / cell_w);\n"
            "      atomicAdd(group, (unsigned long long)mu);\n"
            "      atomicAdd(group + 1, (unsigned long long)(mu * vy));\n"
            "      atomicAdd(group + 2, (unsigned long long)(mu * vx));\n"
            "      double dy = (double)ay - (double)vy;\n"
            "      double dx = (double)ax - (double)vx;\n"
            "      acc_e += mass * sqrt(dy * dy + dx * dx);\n"
            "    }\n"
            "  }\n"
            "  atomicAdd((double*)energy_ptr + a, acc_e);\n"
            "  atomicAdd((double*)residual_ptr + a, acc_r);\n"
            "  atomicAdd((double*)total_ptr + a, acc_s);\n"
            "  return (T)0;\n"
            "}\n"
        )

    @classmethod
    def _move_source(cls, segment: int) -> str:
        """Ядро обмена: отрезок строки сдвинутой точки x — перенос её массы у похожих точек и пересчёт энергий."""
        return (
            "template <typename T> T damp_bound_move_" + str(segment) + "(T row, T part, " + cls._ROW_ARGUMENTS + ",\n"
            "    long long moves_ptr, long long stamp_ptr, long long width, long long update) {\n"
            "  const int* move = (const int*)moves_ptr + 3 * (long long)row;\n"
            "  long long x = move[0];\n"
            "  const long long* rowptr = (const long long*)rowptr_ptr;\n"
            "  long long start = rowptr[x];\n"
            "  long long degree = rowptr[x + 1] - start;\n"
            "  long long first = (long long)part * " + str(segment) + ";\n"
            "  if (!(first < degree)) return (T)0;\n"
            "  long long last = first + " + str(segment) + ";\n"
            "  if (degree < last) last = degree;\n"
            "  const int* col = (const int*)col_ptr;\n"
            "  const float* val = (const float*)val_ptr;\n"
            "  const float* pos = (const float*)pos_ptr;\n"
            "  const float* ones = (const float*)ones_ptr;\n"
            "  const int* stamp = (const int*)stamp_ptr;\n"
            "  double* energy = (double*)energy_ptr;\n"
            "  long long old_y = move[1] / width;\n"
            "  long long old_x = move[1] % width;\n"
            "  long long new_y = move[2] / width;\n"
            "  long long new_x = move[2] % width;\n"
            "  long long old_group = 3 * ((old_y / cell_h) * grid + old_x / cell_w);\n"
            "  long long new_group = 3 * ((new_y / cell_h) * grid + new_x / cell_w);\n"
            "  float na = ones[x];\n"
            "  double acc_e = 0.0;\n"
            "  for (long long j = start + first; j < start + last; ++j) {\n"
            "    int y = col[j];\n"
            "    float s = val[j];\n"
            "    float nv = ones[y];\n"
            + cls._MASS
            + "    if (!(0 < mu)) continue;\n"
            "    double mass = (double)mu * unit;\n"
            "    unsigned long long* groups = (unsigned long long*)groups_ptr + 3 * cells * (long long)y;\n"
            "    atomicAdd(groups + old_group, (unsigned long long)(-mu));\n"
            "    atomicAdd(groups + old_group + 1, (unsigned long long)(-mu * old_y));\n"
            "    atomicAdd(groups + old_group + 2, (unsigned long long)(-mu * old_x));\n"
            "    atomicAdd(groups + new_group, (unsigned long long)mu);\n"
            "    atomicAdd(groups + new_group + 1, (unsigned long long)(mu * new_y));\n"
            "    atomicAdd(groups + new_group + 2, (unsigned long long)(mu * new_x));\n"
            "    float py = pos[2 * y];\n"
            "    float px = pos[2 * y + 1];\n"
            "    float ny = py - (float)new_y;\n"
            "    float nx = px - (float)new_x;\n"
            "    double to_new = (double)sqrtf(ny * ny + nx * nx);\n"
            "    if (stamp[y] != (int)update) {\n"
            "      float oy = py - (float)old_y;\n"
            "      float ox = px - (float)old_x;\n"
            "      atomicAdd(energy + y, mass * (to_new - (double)sqrtf(oy * oy + ox * ox)));\n"
            "    }\n"
            "    acc_e += mass * to_new;\n"
            "  }\n"
            "  atomicAdd(energy + x, acc_e);\n"
            "  return (T)0;\n"
            "}\n"
        )

    @staticmethod
    def _cells_source() -> str:
        """Ядро «в ячейке теперь эта точка» с отметкой сдвинутых точек: их энергия обнуляется под пересчёт."""
        return (
            "template <typename T> T damp_bound_cells(T index, long long table_ptr, long long grid_ptr, long long pos_ptr,\n"
            "    long long stamp_ptr, long long energy_ptr, long long width, long long update) {\n"
            "  const int* item = (const int*)table_ptr + 2 * (long long)index;\n"
            "  long long cell = item[0];\n"
            "  int occupant = item[1];\n"
            "  ((long long*)grid_ptr)[cell] = occupant;\n"
            "  if (0 <= occupant) {\n"
            "    float* place = (float*)pos_ptr + 2 * (long long)occupant;\n"
            "    place[0] = (float)(cell / width);\n"
            "    place[1] = (float)(cell % width);\n"
            "    ((int*)stamp_ptr)[occupant] = (int)update;\n"
            "    ((double*)energy_ptr)[occupant] = 0.0;\n"
            "  }\n"
            "  return (T)0;\n"
            "}\n"
        )

    @staticmethod
    def _lower_source() -> str:
        """Ядро нижней границы: строка грубой сетки групп точки строки таблицы пар → часть L(ячейка второй точки).

        Строка таблицы пар — (точка, вторая точка, своя ячейка, ячейка второй точки): для неё нужна граница
        потенциала точки в ячейке второй точки, L = Σ_групп ‖M·p − P‖. Поток считает GRID групп одной строки
        грубой сетки (float32) и пишет частичную сумму по адресу; у отключённой строки (точка −1) — ноль.
        """
        return (
            "template <typename T> T damp_bound_lower(T row, T part, long long rows_ptr, long long groups_ptr,\n"
            "    long long out_ptr, long long cells, long long grid, double unit) {\n"
            "  const int* item = (const int*)rows_ptr + 6 * (long long)row;\n"
            "  int point = item[0];\n"
            "  float sum = 0.0f;\n"
            "  if (0 <= point) {\n"
            "    long long y = item[4];\n"
            "    long long x = item[5];\n"
            "    const long long* group = (const long long*)groups_ptr + 3 * (cells * (long long)point + grid * (long long)part);\n"
            "    for (long long c = 0; c < grid; ++c) {\n"
            "      long long mass = group[3 * c];\n"
            "      if (mass == 0) continue;\n"
            "      float vy = (float)(mass * y - group[3 * c + 1]) * (float)unit;\n"
            "      float vx = (float)(mass * x - group[3 * c + 2]) * (float)unit;\n"
            "      sum += sqrtf(vy * vy + vx * vx);\n"
            "    }\n"
            "  }\n"
            "  ((float*)out_ptr)[(long long)row * grid + (long long)part] = sum;\n"
            "  return (T)0;\n"
            "}\n"
        )

    @staticmethod
    def _certify_source() -> str:
        """Ядро проверки пары: φ_c − φ_s ≤ [E_a − L_a(p_b)] + [E_b − L_b(p_a)] + поправки.

        Если граница отрицательна с запасом на ошибку суммирования float32 исходного прохода, пара помечается
        (результат 1), а её строки в таблице пар отключаются (точка −1): расчёт по строкам их пропустит.
        Частичные суммы L двух строк пары посчитало ядро нижней границы.
        """
        return (
            "template <typename T> T damp_bound_certify(T index, long long rows_ptr, long long terms_ptr, long long pairs,\n"
            "    long long lower_ptr, long long energy_ptr, long long residual_ptr, long long total_ptr,\n"
            "    long long grid, double diagonal, double extra, double slack, double shrink, double drift) {\n"
            "  long long i = (long long)index;\n"
            "  int* item = (int*)rows_ptr + 6 * i;\n"
            "  int a = item[0];\n"
            "  if (a < 0) return (T)0;\n"
            "  int b = item[1];\n"
            "  const double* energy = (const double*)energy_ptr;\n"
            "  const double* residual = (const double*)residual_ptr;\n"
            "  const double* total = (const double*)total_ptr;\n"
            "  const float* part = (const float*)lower_ptr;\n"
            "  double own = energy[a];\n"
            "  double rest = residual[a];\n"
            "  double weight = total[a];\n"
            "  float sum = 0.0f;\n"
            "  for (long long g = 0; g < grid; ++g) sum += part[i * grid + g];\n"
            "  if (0 <= b) {\n"
            "    own += energy[b];\n"
            "    rest += residual[b];\n"
            "    weight += total[b];\n"
            "    for (long long g = 0; g < grid; ++g) sum += part[(i + pairs) * grid + g];\n"
            "  }\n"
            "  double lower = (double)sum * shrink;\n"
            "  double dy = (double)(item[2] - item[4]);\n"
            "  double dx = (double)(item[3] - item[5]);\n"
            "  double apart = sqrt(dy * dy + dx * dx);\n"
            # Верхняя граница φ_c − φ_s и верхняя граница φ_c + φ_s (для оценки ошибки суммирования float32).
            "  double upper = own - lower + rest * apart + drift * weight;\n"
            "  double both = 2.0 * own + 2.0 * diagonal * rest + weight * apart + 2.0 * drift * weight;\n"
            "  double k = ((double)((const int*)terms_ptr)[i] + extra) * 5.9604644775390625e-08;\n"
            "  double limit = (k / (1.0 - k) + slack) * both;\n"
            "  if (!(upper + limit < 0.0)) return (T)0;\n"
            "  item[0] = -1;\n"
            "  ((int*)rows_ptr)[6 * (i + pairs)] = -1;\n"
            "  return (T)1;\n"
            "}\n"
        )

    @classmethod
    def _kernel(cls, name: str, segment: int = 0) -> DeviceKernel:
        key = (name, int(segment))
        cached = cls._KERNEL_CACHE.get(key)
        if cached is None:
            from torch.cuda.jiterator import _create_jit_fn

            # Порядок скаляров — как в заголовке ядра: jiterator передаёт их по порядку объявления здесь.
            row_scalars = dict(
                rowptr_ptr=0, col_ptr=0, val_ptr=0, pos_ptr=0, ones_ptr=0, groups_ptr=0, energy_ptr=0,
                cells=1, cell_h=1, cell_w=1, grid=1, lambda_guard=0.0, scale=1.0, unit=1.0, jaccard=0,
            )
            if name == "build":
                function = _create_jit_fn(cls._build_source(segment), **row_scalars, residual_ptr=0, total_ptr=0)
            elif name == "move":
                function = _create_jit_fn(
                    cls._move_source(segment), **row_scalars, moves_ptr=0, stamp_ptr=0, width=1, update=0
                )
            elif name == "cells":
                function = _create_jit_fn(
                    cls._cells_source(), table_ptr=0, grid_ptr=0, pos_ptr=0, stamp_ptr=0, energy_ptr=0, width=1, update=0
                )
            elif name == "lower":
                function = _create_jit_fn(
                    cls._lower_source(), rows_ptr=0, groups_ptr=0, out_ptr=0, cells=1, grid=1, unit=1.0
                )
            elif name == "certify":
                function = _create_jit_fn(
                    cls._certify_source(), rows_ptr=0, terms_ptr=0, pairs=0, lower_ptr=0, energy_ptr=0,
                    residual_ptr=0, total_ptr=0, grid=1, diagonal=1.0, extra=0.0, slack=0.0,
                    shrink=1.0, drift=0.0,
                )
            else:
                raise ValueError(f"unknown kernel {name}")
            cached = DeviceKernel(function)
            cls._KERNEL_CACHE[key] = cached
        return cached

    # ------------------------------------------------------------------ расчёт
    def _indices(self, rows: int, parts: int) -> tuple[Any, Any]:
        torch = self._torch
        if self._row_index is None or int(self._row_index.shape[0]) < rows:
            self._row_index = torch.arange(max(rows, 1024), dtype=torch.int32, device=self._device).unsqueeze(1)
        if self._part_index is None or int(self._part_index.shape[1]) < parts:
            self._part_index = torch.arange(max(parts, 256), dtype=torch.int32, device=self._device).unsqueeze(0)
        return self._row_index[:rows], self._part_index[:, :parts]

    def _flat(self, count: int) -> Any:
        torch = self._torch
        if self._flat_index is None or int(self._flat_index.numel()) < count:
            self._flat_index = torch.arange(max(count, 4096), dtype=torch.int32, device=self._device)
        return self._flat_index[:count]

    def _row_scalars(self) -> dict[str, Any]:
        rows = self._rows
        engine = self._engine
        return dict(
            rowptr_ptr=rows.row_ptr.data_ptr(),
            col_ptr=rows.col.data_ptr(),
            val_ptr=rows.val.data_ptr(),
            pos_ptr=engine.positions.data_ptr(),
            ones_ptr=engine.ones.data_ptr(),
            groups_ptr=self._groups.data_ptr(),
            energy_ptr=self._energy.data_ptr(),
            cells=self._cells,
            cell_h=self._cell_h,
            cell_w=self._cell_w,
            grid=self._grid,
            **self._mass_scalars(),
        )

    def rebuild(self) -> None:
        """Построить суммы заново по нынешним позициям точек на устройстве и нынешним строкам сходства."""
        torch = self._torch
        rows = self._rows
        count = self._engine.point_count
        parts = max(1, -(-rows.max_degree // self._segment))
        with torch.no_grad():
            self._groups.zero_()
            self._energy.zero_()
            self._residual.zero_()
            self._similarity_sum.zero_()
            row_index, part_index = self._indices(count, parts)
            self._kernel("build", self._segment)(
                row_index,
                part_index,
                **self._row_scalars(),
                residual_ptr=self._residual.data_ptr(),
                total_ptr=self._similarity_sum.data_ptr(),
            )
        self._lambda = rows.lambda_threshold
        self._updates_since_rebuild = 0
        self._valid = True
        event = "layout.bounds.rebuild"
        if LOGGER.should_log(event):
            LOGGER.event(
                event,
                section=LAYOUT_PARAMETERS,
                data={
                    "points": count,
                    "lambda_threshold": rows.lambda_threshold,
                    "edges": rows.edges,
                    "grid": self._grid,
                    "note": "массы и моменты групп похожих точек и собственные энергии построены заново (§5.5.1): "
                    "при включении, смене λ (§7.1.7) и периодически",
                },
                force=True,
            )

    def apply_moves(self, cells: "np.ndarray", occupants: "np.ndarray", moves: "np.ndarray") -> None:
        """Обмены шага: ячейки cells теперь заняты occupants (−1 — пусто), точки moves[:, 0] сдвинуты.

        moves — строки (точка, старая ячейка, новая ячейка), по одной на сдвинутую точку. Сетка и позиции
        движка обновляются здесь же (вместо TorchPairEnergyEngine.update_cells): одна загрузка и два ядра.
        """
        torch = self._torch
        engine = self._engine
        changed = int(cells.size)
        moved = int(moves.shape[0])
        if changed == 0:
            return
        self._update += 1
        self._updates_since_rebuild += 1
        size = 2 * changed + self.MOVE_COLUMNS * moved
        table = self._stage.host(size)
        head = table[: 2 * changed].reshape(changed, 2)
        head[:, 0] = cells
        head[:, 1] = occupants
        table[2 * changed :] = moves.reshape(-1)
        widest = int(self._rows.degree_np[moves[:, 0]].max()) if moved else 0
        parts = max(1, -(-widest // self.MOVE_SEGMENT))
        with torch.no_grad():
            device_table = self._stage.upload(size)
            address = device_table.data_ptr()
            self._kernel("cells")(
                self._flat(changed),
                table_ptr=address,
                grid_ptr=engine.grid.data_ptr(),
                pos_ptr=engine.positions.data_ptr(),
                stamp_ptr=self._stamp.data_ptr(),
                energy_ptr=self._energy.data_ptr(),
                width=self._width,
                update=self._update,
            )
            if moved:
                row_index, part_index = self._indices(moved, parts)
                self._kernel("move", self.MOVE_SEGMENT)(
                    row_index,
                    part_index,
                    **self._row_scalars(),
                    moves_ptr=address + 4 * 2 * changed,
                    stamp_ptr=self._stamp.data_ptr(),
                    width=self._width,
                    update=self._update,
                )

    @property
    def rebuild_due(self) -> bool:
        """Суммы пора строить заново: они недействительны либо накопилось много обновлений."""
        return not self.valid or self._updates_since_rebuild >= self.REBUILD_EVERY

    def certify(self, table_address: int, pairs: int) -> Any:
        """Пометить пары таблицы строк (PairSignKernel.pair_table), у которых φ_c < φ_s заведомо.

        Возвращает тензор устройства int32 длины pairs: 1 — пара помечена, её строки в таблице отключены.
        """
        columns = 6
        torch = self._torch
        # Нижние границы: поток на строку грубой сетки групп каждой строки таблицы пар.
        lower = torch.empty(2 * pairs * self._grid, dtype=torch.float32, device=self._device)
        row_index, part_index = self._indices(2 * pairs, self._grid)
        self._kernel("lower")(
            row_index,
            part_index,
            rows_ptr=table_address,
            groups_ptr=self._groups.data_ptr(),
            out_ptr=lower.data_ptr(),
            cells=self._cells,
            grid=self._grid,
            unit=2.0 ** -self.MASS_BITS,
        )
        return self._kernel("certify")(
            self._flat(pairs),
            rows_ptr=table_address,
            terms_ptr=table_address + 4 * 2 * pairs * columns,
            pairs=pairs,
            lower_ptr=lower.data_ptr(),
            energy_ptr=self._energy.data_ptr(),
            residual_ptr=self._residual.data_ptr(),
            total_ptr=self._similarity_sum.data_ptr(),
            grid=self._grid,
            diagonal=self._diagonal,
            extra=float(self.TERM_ALLOWANCE),
            slack=self.FLOAT_SLACK,
            shrink=1.0 - self.LOWER_SHRINK,
            drift=(self._updates_since_rebuild + 1) * self.DISTANCE_ERROR * self._diagonal,
        )

    def snapshot(self) -> dict[str, "np.ndarray"]:
        """Суммы на хосте (для проверок): группы, энергии, остатки, суммы сходств."""
        return {
            "groups": self._groups.cpu().numpy(),
            "energy": self._energy.cpu().numpy(),
            "residual": self._residual.cpu().numpy(),
            "similarity_sum": self._similarity_sum.cpu().numpy(),
        }

    def log_disabled(self, *, reason: str) -> None:
        LOGGER.event(
            "layout.bounds.disabled",
            section=GPU_IMPLEMENTATION,
            data={"reason": reason, "note": "пары шага считаются по строкам сходства без отсева по границе (§5.5.1)"},
            force=True,
        )
