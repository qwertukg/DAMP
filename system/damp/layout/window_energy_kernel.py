"""Энергии окрестности пар (§5.5.2) и точек (§5.7) больших раскладок одним ядром по ячейкам окна (§5.12).

Зачем. На миллионах точек строки сходства SimilarityRows не помещаются в память, а проход «операциями над
тензорами» (TorchPairEnergyEngine, FusedPairEnergy) держит на устройстве по нескольку десятков байт
на каждое слагаемое (пара, ячейка окна) и суммирует их отдельной редукцией. Здесь слагаемые нигде
не хранятся: поток устройства считает отрезок ячеек окна одной пары (§5.12: «каждый поток считает…»)
и отдаёт две частичные суммы — время шага определяет только число слагаемых.

Что считается — то же, что в TorchPairEnergyEngine._pair_energies_local (§5.5.2):

- окрестность пары — круг с центром в середине отрезка пары c = (p1 + p2) / 2 и радиусом
  r = max(⌈‖p1 − p2‖⌉ + 1, local_radius) (r ≥ ‖p1 − p2‖, §5.5.2); в сумму входят точки v круга,
  кроме самих точек пары;
- s1 = sim_λ(v, a), s2 = sim_λ(v, b) — мера сходства по общим битам кодов (§2.2.4), жёсткий порог λ (§5.4,
  §7.1.7); общие биты — popcount(код ячейки & код точки пары) по 64-битным словам;
- вес w = 1 / (d + ε), d — евклидово расстояние или его квадрат (§5.9.2);
- φ_c = Σ s1·w1 + s2·w2, φ_s = Σ s2·w1 + s1·w2 (§5.5.3).

Энергия точки E(c, r) (§5.7) — тот же расчёт для «пары» из точки и пустой второй ячейки на том же месте.

Коды лежат на устройстве дважды: по точкам (для точек пары) и по ячейкам сетки (для точек окна: ячейки
окна читаются подряд). Таблица по ячейкам обновляется вместе с обменами шага (update).

Сверка. Суммы ядра — float32 по отрезкам и float64 между отрезками; исходный проход суммирует те же
слагаемые редукцией float32. Числа совпадают с точностью сложения float32 (verify): это режим больших
раскладок, где хода «шаг в шаг» с исходным проходом нет.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from damp.article_refs import ENERGY_SHORT, GPU_IMPLEMENTATION, POINT_ENERGY
from damp.layout.device_kernel import DeviceKernel
from damp.logging import LOGGER


class WindowEnergyKernel:
    """φ_c, φ_s пар и E точек по ячейкам окна окрестности: сходство — из слов кодов ячеек на лету."""

    # Ячеек окна на один поток устройства. Поток ждёт каждое чтение кода ячейки, поэтому короткие отрезки
    # быстрее длинных (882 тыс. точек, окно 3 357 ячеек: 4 ячейки — 0.35 нс на слагаемое, 64 — 0.72 нс).
    SEGMENT = 4
    # Предел потоков одного запуска ядра (пары × отрезки окна): ограничивает память под частичные суммы.
    BLOCK_THREADS = 4_000_000
    # Допуск сверки с исходным проходом: суммы float32 в другом порядке сложения.
    VERIFY_RTOL = 2e-4
    VERIFY_ATOL = 1e-5

    _KERNEL_CACHE: dict[Any, DeviceKernel] = {}

    def __init__(self, engine: Any, *, segment: int | None = None) -> None:
        torch = engine.torch_mod
        self._engine = engine
        self._torch = torch
        self._device = engine.torch_device
        self._count = int(engine.point_count)
        self._height, self._width = engine.grid_shape
        self._similarity = engine.similarity
        self._segment = int(segment or self.SEGMENT)
        if self._segment <= 0:
            raise ValueError("segment must be positive")
        cells = self._height * self._width
        with torch.no_grad():
            # Коды точек 64-битными словами — общие с движком; «точки нет» (номер N) — нулевой код.
            self._point_words = engine.packed_codes()
            self._word_count = int(self._point_words.shape[1])
            self._zero_words = torch.zeros(self._word_count, dtype=torch.int64, device=self._device)
            ones = engine.ones
            norm = torch.zeros(self._count + 1, dtype=torch.float32, device=self._device)
            if self._similarity == "cosine":
                # Тот же множитель n^−½, что в TorchPairEnergyEngine._sim_lambda_from_common (нулевой код — 0).
                norm[: self._count] = torch.where(
                    ones > 0, torch.rsqrt(torch.clamp(ones, min=1.0)), torch.zeros_like(ones)
                )
            else:
                norm[: self._count] = ones
            self._point_norm = norm
            # Коды и нормировки по ячейкам сетки; последняя ячейка — «за сеткой» (нули).
            self._cell_words = torch.zeros((cells + 1, self._word_count), dtype=torch.int64, device=self._device)
            self._cell_norm = torch.zeros(cells + 1, dtype=torch.float32, device=self._device)
        self._offsets_cache: dict[int, tuple[Any, int, Any]] = {}
        self._weights_cache: dict[tuple[int, bool, float], Any] = {}
        self._index_cache: dict[tuple[int, int], tuple[Any, Any]] = {}
        LOGGER.event(
            "layout.window_kernel.init",
            section=GPU_IMPLEMENTATION,
            data={
                "points": self._count,
                "grid": (self._height, self._width),
                "segment": self._segment,
                "code_words": self._word_count,
                "memory_mb": ((cells + 1) * (self._word_count * 8 + 4) + (self._count + 1) * 4) / 1e6,
                "note": "энергии окрестности пар (§5.5.2) и точек (§5.7): поток на отрезок ячеек окна, сходство "
                "по общим битам слов кода ячейки и точки пары (§2.2.4), слагаемые не хранятся (§5.12)",
            },
        )

    # ------------------------------------------------------------------ применимость
    @classmethod
    def supported(cls, engine: Any) -> bool:
        """Ядро компилируется только для CUDA; номера ячеек и точек — int32."""
        if engine.torch_device.type != "cuda":
            return False
        height, width = engine.grid_shape
        if height * width + 1 >= 2**31 or engine.point_count + 1 >= 2**31:
            return False
        try:
            from torch.cuda import jiterator  # noqa: F401
        except Exception:  # noqa: BLE001
            return False
        return engine.similarity in ("cosine", "jaccard")

    # ------------------------------------------------------------------ расстановка
    def load(self, grid_points: Any) -> None:
        """Коды по ячейкам из расстановки: grid_points — точка каждой ячейки (N — пусто), длина ячейки + 1."""
        torch = self._torch
        with torch.no_grad():
            points = grid_points.to(torch.int64)
            step = 1 << 20
            for start in range(0, int(points.numel()), step):
                part = points[start : start + step]
                self._cell_words[start : start + step] = self._words_of(part)
                self._cell_norm[start : start + step] = self._point_norm[part]
            self._cell_words[-1] = 0
            self._cell_norm[-1] = 0.0

    def update(self, cells: Any, points: Any) -> None:
        """В ячейках cells (линейные номера) теперь точки points (N — пусто): коды ячеек после обменов шага."""
        self._cell_words[cells] = self._words_of(points)
        self._cell_norm[cells] = self._point_norm[points]

    def _words_of(self, points: Any) -> Any:
        """Слова кодов точек points (N — «точки нет»: нули)."""
        torch = self._torch
        present = points < self._count
        words = self._point_words[torch.clamp(points, max=self._count - 1)]
        return torch.where(present.unsqueeze(1), words, torch.zeros_like(words))

    def pair_similarity(self, point_a: Any, point_b: Any) -> Any:
        """sim(a, b) пар без порога (§2.2.4) — для ранней отсечки пар (§5.9.1); пустая вторая ячейка — 0."""
        torch = self._torch
        bits = self._words_of(point_a) & self._words_of(point_b)
        # popcount по 64-битным словам: параллельный подсчёт бит (SWAR).
        bits = bits - ((bits >> 1) & 0x5555555555555555)
        bits = (bits & 0x3333333333333333) + ((bits >> 2) & 0x3333333333333333)
        bits = (bits + (bits >> 4)) & 0x0F0F0F0F0F0F0F0F
        common = (((bits * 0x0101010101010101) >> 56) & 0xFF).sum(dim=1).to(torch.float32)
        norm_a = self._point_norm[point_a]
        norm_b = self._point_norm[point_b]
        if self._similarity == "cosine":
            return common * norm_a * norm_b
        denominator = norm_a + norm_b - common
        positive = (denominator > 0) & (norm_a > 0) & (norm_b > 0)
        return torch.where(positive, common / torch.where(positive, denominator, torch.ones_like(denominator)), torch.zeros_like(common))

    # ------------------------------------------------------------------ ядро
    @classmethod
    def _source(cls, similarity: str, words: int) -> tuple[str, str]:
        """Имя и текст ядра отрезка окна. В теле нет знака «больше»: по нему jiterator ищет конец заголовка."""
        if similarity == "cosine":
            # cos = common · n_a^−½ · n_v^−½ — те же два умножения float32, что у исходного прохода.
            measure = (
                "    float s1 = __fmul_rn(__fmul_rn((float)common_a, norm_a), norm_v);\n"
                "    float s2 = __fmul_rn(__fmul_rn((float)common_b, norm_b), norm_v);\n"
            )
        else:
            # Жаккар: common / (n_a + n_v − common); нулевой код — 0.
            measure = (
                "    float den_a = __fsub_rn(__fadd_rn(norm_a, norm_v), (float)common_a);\n"
                "    float den_b = __fsub_rn(__fadd_rn(norm_b, norm_v), (float)common_b);\n"
                "    float s1 = (0.0f < den_a && 0.0f < norm_a && 0.0f < norm_v) ? __fdiv_rn((float)common_a, den_a) : 0.0f;\n"
                "    float s2 = (0.0f < den_b && 0.0f < norm_b && 0.0f < norm_v) ? __fdiv_rn((float)common_b, den_b) : 0.0f;\n"
            )
        popcount = "".join(
            f"    {{ long long v = cw[{index}]; common_a += __popcll(v & wa[{index}]); "
            f"common_b += __popcll(v & wb[{index}]); }}\n"
            for index in range(words)
        )
        name = f"damp_window_{similarity}_{words}"
        source = (
            "template <typename T> T " + name + "(T pair, T part, long long table_ptr, long long words_ptr,\n"
            "    long long norm_ptr, long long cwords_ptr, long long cnorm_ptr, long long off_ptr, long long near_ptr,\n"
            "    long long weight_ptr, long long out_ptr, long long parts, long long cells, long long segment,\n"
            "    long long width, long long height, long long count, long long zero_ptr, double lam,\n"
            "    double local_radius) {\n"
            "  const int* item = (const int*)table_ptr + 4 * (long long)pair;\n"
            "  int a = item[0];\n"
            "  int b = item[1];\n"
            "  int cell_a = item[2];\n"
            "  int cell_b = item[3];\n"
            "  int grid_width = (int)width;\n"
            "  int grid_height = (int)height;\n"
            "  int ya = cell_a / grid_width;\n"
            "  int xa = cell_a - ya * grid_width;\n"
            "  int yb = cell_b / grid_width;\n"
            "  int xb = cell_b - yb * grid_width;\n"
            # Окрестность пары (§5.5.2): центр — середина отрезка, r = max(⌈‖p1 − p2‖⌉ + 1, local_radius).
            "  float cy = __fmul_rn((float)(ya + yb), 0.5f);\n"
            "  float cx = __fmul_rn((float)(xa + xb), 0.5f);\n"
            "  int aby = ya - yb;\n"
            "  int abx = xa - xb;\n"
            "  float radius = ceilf(sqrtf((float)(aby * aby + abx * abx))) + 1.0f;\n"
            "  if (radius < (float)local_radius) radius = (float)local_radius;\n"
            "  float radius_sq = __fmul_rn(radius, radius);\n"
            "  float* out = (float*)out_ptr + 2 * ((long long)pair * parts + (long long)part);\n"
            # Отрезок окна целиком вне круга пары: ближайшая его ячейка дальше r от любого возможного центра.
            "  if (radius_sq < ((const float*)near_ptr)[(int)part]) {\n"
            "    out[0] = 0.0f;\n"
            "    out[1] = 0.0f;\n"
            "    return (T)0;\n"
            "  }\n"
            "  int base_y = (ya + yb) / 2;\n"
            "  int base_x = (xa + xb) / 2;\n"
            f"  const long long* wa = (const long long*)words_ptr + {words} * (long long)a;\n"
            # Второй точки нет (номер N: ячейка пуста) — нулевой код, её сходство с любой точкой — 0.
            f"  const long long* wb = (b < (int)count) ? (const long long*)words_ptr + {words} * (long long)b\n"
            "      : (const long long*)zero_ptr;\n"
            "  float norm_a = ((const float*)norm_ptr)[a];\n"
            "  float norm_b = ((const float*)norm_ptr)[b];\n"
            "  const short* off = (const short*)off_ptr;\n"
            "  const float* cnorm = (const float*)cnorm_ptr;\n"
            "  const float* weight = (const float*)weight_ptr;\n"
            "  float threshold = (float)lam;\n"
            "  float acc_c = 0.0f;\n"
            "  float acc_s = 0.0f;\n"
            "  int first = (int)part * (int)segment;\n"
            "  int last = first + (int)segment;\n"
            "  if ((int)cells < last) last = (int)cells;\n"
            "  int outside = grid_height * grid_width;\n"
            # Тело цикла без ветвлений: потоки варпа идут одним путём. Ячейка за сеткой читается как последняя
            # ячейка таблицы (нулевой код), вклад ячеек вне круга и самих точек пары обнуляется.
            "  for (int k = first; k < last; ++k) {\n"
            "    int y = base_y + off[2 * k];\n"
            "    int x = base_x + off[2 * k + 1];\n"
            "    bool inside = (0 <= y) && (y < grid_height) && (0 <= x) && (x < grid_width);\n"
            "    int cell = inside ? y * grid_width + x : outside;\n"
            "    float oy = (float)y - cy;\n"
            "    float ox = (float)x - cx;\n"
            # В сумму входят точки круга ‖v − c‖ ≤ r, кроме самих точек пары (§5.5.2).
            "    bool valid = inside && (cell != cell_a) && (cell != cell_b)\n"
            "        && (__fadd_rn(__fmul_rn(oy, oy), __fmul_rn(ox, ox)) <= radius_sq);\n"
            f"    const long long* cw = (const long long*)cwords_ptr + {words} * (long long)cell;\n"
            "    float norm_v = cnorm[cell];\n"
            "    int common_a = 0;\n"
            "    int common_b = 0;\n"
            + popcount
            + measure
            # Жёсткий порог λ (§5.4, η = ∞ в §7.1.7): sim < λ — ноль.
            + "    s1 = (valid && threshold <= s1) ? s1 : 0.0f;\n"
            "    s2 = (valid && threshold <= s2) ? s2 : 0.0f;\n"
            # Вес 1 / (d + ε) — из таблицы по целому квадрату расстояния (§5.9.2): те же числа float32.
            "    int d1y = ya - y;\n"
            "    int d1x = xa - x;\n"
            "    int d2y = yb - y;\n"
            "    int d2x = xb - x;\n"
            "    float w1 = weight[d1y * d1y + d1x * d1x];\n"
            "    float w2 = weight[d2y * d2y + d2x * d2x];\n"
            "    acc_c += __fadd_rn(__fmul_rn(s1, w1), __fmul_rn(s2, w2));\n"
            "    acc_s += __fadd_rn(__fmul_rn(s2, w1), __fmul_rn(s1, w2));\n"
            "  }\n"
            "  out[0] = acc_c;\n"
            "  out[1] = acc_s;\n"
            "  return (T)0;\n"
            "}\n"
        )
        return name, source

    def _kernel(self) -> DeviceKernel:
        key = (self._similarity, self._word_count)
        kernel = self._KERNEL_CACHE.get(key)
        if kernel is None:
            from torch.cuda.jiterator import _create_jit_fn

            _, source = self._source(self._similarity, self._word_count)
            kernel = DeviceKernel(
                _create_jit_fn(
                    source,
                    table_ptr=0, words_ptr=0, norm_ptr=0, cwords_ptr=0, cnorm_ptr=0, off_ptr=0, near_ptr=0,
                    weight_ptr=0, out_ptr=0, parts=1, cells=1, segment=1, width=1, height=1, count=1, zero_ptr=0,
                    lam=0.0, local_radius=0.0,
                )
            )
            self._KERNEL_CACHE[key] = kernel
        return kernel

    def _offsets(self, reach: int) -> tuple[Any, int, Any]:
        """Смещения ячеек окна полуширины reach, которые могут попасть в окрестность пары (§5.5.2): int16 K × 2,
        их число и для каждого отрезка окна — квадрат расстояния от центра пары до ближайшей ячейки отрезка.

        Окрестность — круг радиуса не больше reach − 1 вокруг центра пары; центр отстоит от своей ячейки
        не больше чем на полячейки по каждой оси. Ячейка со смещением (dy, dx) лежит от центра не ближе
        √((|dy| − ½)₊² + (|dx| − ½)₊²): смещения, где это больше reach − 1, в окрестность не попадают.
        Смещения идут построчно: ячейки отрезка окна лежат в памяти подряд.
        """
        cached = self._offsets_cache.get(reach)
        if cached is None:
            span = np.arange(-reach, reach + 1, dtype=np.int64)
            dy, dx = np.meshgrid(span, span, indexing="ij")
            near_y = np.maximum(np.abs(dy) - 0.5, 0.0)
            near_x = np.maximum(np.abs(dx) - 0.5, 0.0)
            inside = near_y * near_y + near_x * near_x <= float(reach - 1) ** 2
            table = np.stack([dy[inside], dx[inside]], axis=1).astype(np.int16)
            # Нижняя граница расстояния от центра пары до ячеек отрезка: отрезок, у которого она больше радиуса
            # окрестности пары, поток пропускает целиком.
            near = (near_y * near_y + near_x * near_x)[inside].astype(np.float32)
            parts = -(-near.size // self._segment)
            padded = np.full(parts * self._segment, np.inf, dtype=np.float32)
            padded[: near.size] = near
            part_near = padded.reshape(parts, self._segment).min(axis=1)
            cached = (
                self._torch.as_tensor(np.ascontiguousarray(table), device=self._device),
                int(table.shape[0]),
                self._torch.as_tensor(np.ascontiguousarray(part_near), device=self._device),
            )
            if len(self._offsets_cache) > 64:
                self._offsets_cache.clear()
            self._offsets_cache[reach] = cached
        return cached

    def _weights(self, reach: int, euclidean: bool, distance_eps: float) -> Any:
        """Веса 1 / (d + ε) по целому квадрату расстояния от точки пары до ячейки окна (§5.5.2, §5.9.2): float32.

        d — евклидово расстояние или его квадрат; те же действия float32, что у исходного прохода. Точка пары
        и ячейка окна лежат не дальше 3·reach друг от друга по каждой оси.
        """
        torch = self._torch
        key = (int(reach), bool(euclidean), float(distance_eps))
        cached = self._weights_cache.get(key)
        if cached is None:
            span = 3 * int(reach) + 2
            squares = torch.arange(2 * span * span + 1, dtype=torch.float32, device=self._device)
            distance = torch.sqrt(squares) if euclidean else squares
            cached = 1.0 / (distance + float(distance_eps))
            if len(self._weights_cache) > 64:
                self._weights_cache.clear()
            self._weights_cache[key] = cached
        return cached

    def _indices(self, rows: int, parts: int) -> tuple[Any, Any]:
        """Номера пар блока (столбец) и отрезков окна (строка) — входы поэлементного ядра."""
        torch = self._torch
        cached = self._index_cache.get((rows, parts))
        if cached is None:
            cached = (
                torch.arange(rows, dtype=torch.int32, device=self._device).unsqueeze(1),
                torch.arange(parts, dtype=torch.int32, device=self._device).unsqueeze(0),
            )
            if len(self._index_cache) > 16:
                self._index_cache.clear()
            self._index_cache[(rows, parts)] = cached
        return cached

    @staticmethod
    def reach(max_pair_distance: float, local_radius: float | None) -> int:
        """Полуширина окна ячеек, покрывающего окрестности всех пар (как TorchPairEnergyEngine.window_reach)."""
        radius_max = float(math.ceil(max_pair_distance)) + 1.0
        if local_radius is not None:
            radius_max = max(radius_max, float(np.float32(local_radius)))
        return int(math.ceil(radius_max)) + 1

    # ------------------------------------------------------------------ энергии
    def pair_energies(
        self,
        point_a: Any,
        point_b: Any,
        cell_a: Any,
        cell_b: Any,
        *,
        reach: int,
        local_radius: float | None,
        euclidean: bool,
        distance_eps: float,
        lambda_threshold: float,
    ) -> tuple[Any, Any]:
        """φ_c, φ_s пар (§5.5.2) — тензоры устройства float64.

        point_a, point_b — точки пар (N — второй ячейки нет), cell_a, cell_b — линейные номера их ячеек
        (тензоры устройства); reach — полуширина окна ячеек, покрывающего окрестности всех пар.
        """
        torch = self._torch
        rows = int(point_a.numel())
        offsets, cells, part_near = self._offsets(int(reach))
        parts = -(-cells // self._segment)
        weights = self._weights(int(reach), bool(euclidean), float(distance_eps))
        kernel = self._kernel()
        with torch.no_grad():
            table = torch.stack([point_a, point_b, cell_a, cell_b], dim=1).to(torch.int32).contiguous()
            energy_c = torch.empty(rows, dtype=torch.float64, device=self._device)
            energy_s = torch.empty(rows, dtype=torch.float64, device=self._device)
            block = max(1, self.BLOCK_THREADS // parts)
            out = torch.empty((min(rows, block), parts, 2), dtype=torch.float32, device=self._device)
            for first in range(0, rows, block):
                last = min(rows, first + block)
                size = last - first
                pair_index, part_index = self._indices(size, parts)
                kernel(
                    pair_index,
                    part_index,
                    table_ptr=table.data_ptr() + 16 * first,
                    words_ptr=self._point_words.data_ptr(),
                    norm_ptr=self._point_norm.data_ptr(),
                    cwords_ptr=self._cell_words.data_ptr(),
                    cnorm_ptr=self._cell_norm.data_ptr(),
                    off_ptr=offsets.data_ptr(),
                    near_ptr=part_near.data_ptr(),
                    weight_ptr=weights.data_ptr(),
                    out_ptr=out.data_ptr(),
                    parts=parts,
                    cells=cells,
                    segment=self._segment,
                    width=self._width,
                    height=self._height,
                    count=self._count,
                    zero_ptr=self._zero_words.data_ptr(),
                    # Порог λ (§5.4): sim < λ — ноль; при λ = 0 остаются sim > 0 (наименьшее положительное float32).
                    lam=float(lambda_threshold) if lambda_threshold > 0.0 else 1e-45,
                    local_radius=0.0 if local_radius is None else float(local_radius),
                )
                # Частичные суммы отрезков float32 складываются в float64 (без промежуточной копии).
                sums = out[:size].sum(dim=1, dtype=torch.float64)
                energy_c[first:last] = sums[:, 0]
                energy_s[first:last] = sums[:, 1]
            # Таблица пар и смещения должны жить до конца расчёта: ядро читает их по адресу.
            del table
        return energy_c, energy_s

    def point_energies(
        self,
        cell_of_point: Any,
        radius: float,
        *,
        distance_eps: float,
        lambda_threshold: float,
    ) -> Any:
        """E(c, r) = Σ_{v ≠ c, ‖v − c‖ ≤ r} sim_λ(c, v) / (d + ε) всех точек (§5.7) — тензор устройства float64.

        cell_of_point — линейный номер ячейки каждой точки (тензор устройства).
        """
        torch = self._torch
        energies = torch.empty(self._count, dtype=torch.float64, device=self._device)
        reach = self.reach(0.0, radius)
        step = 1 << 20
        with torch.no_grad():
            for first in range(0, self._count, step):
                last = min(self._count, first + step)
                points = torch.arange(first, last, dtype=torch.int64, device=self._device)
                cells = cell_of_point[first:last]
                energy_c, _ = self.pair_energies(
                    points,
                    torch.full_like(points, self._count),
                    cells,
                    cells,
                    reach=reach,
                    local_radius=float(radius),
                    euclidean=True,
                    distance_eps=distance_eps,
                    lambda_threshold=lambda_threshold,
                )
                energies[first:last] = energy_c
        return energies

    # ------------------------------------------------------------------ сверка
    def verify(
        self,
        energy_c: "np.ndarray",
        energy_s: "np.ndarray",
        expected_c: "np.ndarray",
        expected_s: "np.ndarray",
        *,
        kind: str,
    ) -> bool:
        """Суммы ядра против исходного прохода движка на тех же парах: равны с точностью сложения float32."""
        matched = bool(
            np.allclose(energy_c, expected_c, rtol=self.VERIFY_RTOL, atol=self.VERIFY_ATOL)
            and np.allclose(energy_s, expected_s, rtol=self.VERIFY_RTOL, atol=self.VERIFY_ATOL)
        )
        difference = 0.0
        if len(energy_c):
            difference = float(max(np.abs(energy_c - expected_c).max(), np.abs(energy_s - expected_s).max()))
        LOGGER.event(
            "layout.window_kernel.verify",
            section=POINT_ENERGY if kind == "points" else ENERGY_SHORT,
            data={
                "kind": kind,
                "pairs": int(len(energy_c)),
                "matched": matched,
                "max_difference": difference,
                "note": "φ_c, φ_s ядра по ячейкам окна равны исходному проходу движка с точностью сложения "
                "float32 (§5.5.2, §5.7, §5.12)",
            },
            force=True,
        )
        return matched
