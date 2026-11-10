"""Энергии пар и точек раскладки на GPU (torch): сходство считается из кодов, без матрицы N×N.

§5.12: каждый поток считает одну пару p ∈ P, проходя по точкам V (или подмножеству B ⊆ V, §5.9.4);
коды пары читаются один раз. §5.9.5: матрица сходства предвычисляется, только если это дешевле
пересчёта — для сотен тысяч точек она не помещается в память, поэтому сходство sim(a, v) считается
на лету матричным произведением кодов (общие биты) и нормировкой (§2.2.4). Позиции точек хранятся
на устройстве и обновляются после обменов; на хост возвращаются только энергии пар.

Формулы повторяют шейдер и CPU-путь Layout:
  дальняя энергия (§5.5.1): φ_c = Σ s1·d1 + s2·d2, φ_s = Σ s2·d1 + s1·d2 (d евклидово или d², §5.9.2);
  окрестность (§5.5.2): точки v с ‖v − c‖ ≤ r, c — середина пары, r ≥ ‖p1 − p2‖ (+1), либо заданный
  радиус; вес 1/(d + ε) (inverse) или exp(−d²/(2σ²)), σ = width·r (gauss, расширение статьи).

Окрестность считается по окну ячеек вокруг центра пары. Общие биты кода точки пары нужны только с точками
окна: при большом V они считаются по упакованным словам кода только для K ячеек окна (p·K·L/64 операций
над словами) вместо строки произведения на коды всех N точек (p·N·L). Число общих бит — целое, оба способа
дают одно и то же число, поэтому энергии побитово те же. Слагаемые окна считаются одним блоком по многим
парам, а суммируются порциями прежней формы: сумма float32 зависит от порядка сложения, порядок — от формы
тензора редукции.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from damp.article_refs import GPU_IMPLEMENTATION, OPTIM_SIM_MATRIX, OPTIM_SUBSET
from damp.layout.device_kernel import DeviceKernel
from damp.logging import LOGGER


class TorchPairEnergyEngine:
    """Энергии пар и точек на torch-устройстве (cuda/mps) без предвычисленной матрицы сходства."""

    # Предел элементов (пары × точки) на один проход: ограничивает память под промежуточные тензоры.
    CHUNK_ELEMENTS = 4_000_000
    # В локальном пути (окно ячеек) крупных тензоров p × N меньше — порция пар крупнее.
    LOCAL_CHUNK_ELEMENTS = 16_000_000
    # Предел элементов (пары × ячейки окна) на один блок арифметики пути по окну ячеек: в блоке — целое число
    # порций редукции LOCAL_CHUNK_ELEMENTS // N.
    LOCAL_BLOCK_ELEMENTS = 4_000_000
    # Предел элементов (пары × ячейки окна × 64-битные слова кода) на одну порцию подсчёта общих бит по окну.
    WINDOW_WORD_ELEMENTS = 16_000_000
    # Пока точек не больше стольких на ячейку окна, строка произведения кодов «точка × все точки V» дешевле
    # побитового подсчёта по окну (измерено: N = 8 693, окно 441 ячейка — 1.9 мс против 2.5 мс;
    # N = 88 317 — 139 мс против 25 мс). Оба способа дают одно и то же целое число общих бит.
    WINDOW_FULL_PRODUCT_RATIO = 32

    def __init__(
        self,
        *,
        torch_mod: Any,
        device: Any,
        code_matrix: Any,
        ones: Any,
        similarity: str,
        grid_shape: tuple[int, int],
    ) -> None:
        self._torch = torch_mod
        self._device = device
        self._similarity = similarity
        self._codes = torch_mod.as_tensor(code_matrix, dtype=torch_mod.float32, device=device)
        self._ones = torch_mod.as_tensor(ones, dtype=torch_mod.float32, device=device)
        self._point_count = int(code_matrix.shape[0])
        self._code_length = int(code_matrix.shape[1])
        self._codes_packed: Any = None
        self._height, self._width = int(grid_shape[0]), int(grid_shape[1])
        self._pos = torch_mod.zeros((self._point_count, 2), dtype=torch_mod.float32, device=device)
        self._grid = torch_mod.full((self._height, self._width), -1, dtype=torch_mod.long, device=device)
        self._all_indices = torch_mod.arange(self._point_count, device=device)
        self._cell_index: Any = None
        LOGGER.event(
            "layout.gpu.pair_energy.init",
            section=GPU_IMPLEMENTATION,
            data={
                "device": str(device),
                "points": self._point_count,
                "grid": (self._height, self._width),
                "code_length": int(code_matrix.shape[1]),
                "similarity": similarity,
                "chunk_elements": self.CHUNK_ELEMENTS,
                "note": "сходство из кодов на лету, матрица N×N не строится (§5.9.5, §5.12)",
            },
        )

    @classmethod
    def create(
        cls, *, code_matrix: Any, ones: Any, similarity: str, grid_shape: tuple[int, int]
    ) -> "TorchPairEnergyEngine | None":
        try:
            import torch
        except Exception as exc:  # noqa: BLE001
            LOGGER.event(
                "layout.gpu.pair_energy.unavailable",
                section=GPU_IMPLEMENTATION,
                data={"error": str(exc)},
            )
            return None
        if code_matrix is None or ones is None:
            LOGGER.event(
                "layout.gpu.pair_energy.unavailable",
                section=GPU_IMPLEMENTATION,
                data={"error": "code matrix missing"},
            )
            return None
        device = None
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        if device is None:
            LOGGER.event(
                "layout.gpu.pair_energy.unavailable",
                section=GPU_IMPLEMENTATION,
                data={"error": "no gpu backend"},
            )
            return None
        try:
            return cls(
                torch_mod=torch,
                device=device,
                code_matrix=code_matrix,
                ones=ones,
                similarity=similarity,
                grid_shape=grid_shape,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.event(
                "layout.gpu.pair_energy.unavailable",
                section=GPU_IMPLEMENTATION,
                data={"error": str(exc)},
            )
            return None

    @property
    def device(self) -> str:
        return str(self._device)

    @property
    def point_count(self) -> int:
        return self._point_count

    @property
    def torch_mod(self) -> Any:
        return self._torch

    @property
    def torch_device(self) -> Any:
        return self._device

    @property
    def positions(self) -> Any:
        """Позиции точек на устройстве (N×2 float32, (y, x))."""
        return self._pos

    @property
    def ones(self) -> Any:
        """Число единиц кода каждой точки на устройстве (N, float32)."""
        return self._ones

    @property
    def grid(self) -> Any:
        """Сетка ячеек на устройстве: номер точки в ячейке или −1 (высота × ширина, int64)."""
        return self._grid

    def local_chunk_rows(self) -> int:
        """Число пар в одной порции редукции пути по окну ячеек (форма порции [пары, ячейки окна])."""
        return max(1, self.LOCAL_CHUNK_ELEMENTS // max(1, self._point_count))

    def neighbourhood_geometry(self, pa: Any, pb: Any, local_radius: float | None) -> tuple[Any, Any, int]:
        """Окрестность пар (§5.5.2): центры c, квадраты радиусов r² и полуширина окна ячеек, покрывающего их все.

        r ≥ |p1 − p2|; +1 гарантирует, что окрестность содержит точки помимо самой пары.
        """
        torch = self._torch
        centre = (pa + pb) * 0.5
        dist_ab = torch.sqrt(((pa - pb) ** 2).sum(dim=1))
        base_radius = torch.ceil(dist_ab) + 1.0
        radius = base_radius if local_radius is None else torch.clamp(base_radius, min=float(local_radius))
        radius_sq = radius * radius
        window_reach = int(math.ceil(float(radius.max().item()))) + 1
        return centre, radius_sq, window_reach

    def dense_chunk_rows(self) -> int:
        """Число пар в одной порции плотного прохода по всему V (форма порции [пары, |V|])."""
        return max(1, self.CHUNK_ELEMENTS // max(1, self._point_count))

    def similarity_rows(self, row_indices: Any, *, lambda_threshold: float, eta: float | None) -> Any:
        """Строки sim_λ(a, v) точек row_indices со всеми точками V — те же значения, что в плотном проходе (§5.4)."""
        return self._similarity_lambda(
            row_indices, self._codes, self._ones, lambda_threshold=lambda_threshold, eta=eta
        )

    def free_memory_bytes(self) -> int | None:
        """Свободная память устройства вместе с кешем аллокатора torch; None — устройство её не сообщает."""
        torch = self._torch
        if self._device.type != "cuda":
            return None
        free, _ = torch.cuda.mem_get_info(self._device)
        cached = torch.cuda.memory_reserved(self._device) - torch.cuda.memory_allocated(self._device)
        return int(free + max(0, cached))

    def _window_common_bits(self, points: Any, window: Any) -> Any:
        """Число общих бит кода точки points[i] с кодами точек window[i, k] (§2.2.4): тензор p × K float32.

        То же целое, что стоит в столбцах window[i] строки произведения «код точки × коды всех точек V»:
        суммы произведений бит не превосходят длины кода и в float точны при любом порядке сложения.
        Стоимость — p·K·L вместо p·N·L.
        """
        torch = self._torch
        pair_count, cells = int(window.shape[0]), int(window.shape[1])
        common = torch.empty((pair_count, cells), dtype=torch.float32, device=self._device)
        if self._point_count <= self.WINDOW_FULL_PRODUCT_RATIO * cells:
            # Точек мало: строка произведения на коды всех точек дешевле побитового счёта по окну.
            step = max(1, self.LOCAL_CHUNK_ELEMENTS // max(1, self._point_count))
            for start in range(0, pair_count, step):
                stop = min(pair_count, start + step)
                common[start:stop] = torch.gather(
                    self._codes[points[start:stop]] @ self._codes.T, 1, window[start:stop]
                )
            return common
        packed = self._packed_codes()
        words = int(packed.shape[1])
        step = max(1, self.WINDOW_WORD_ELEMENTS // max(1, cells * words))
        for start in range(0, pair_count, step):
            stop = min(pair_count, start + step)
            # popcount(a & v) по 64-битным словам: параллельный подсчёт бит (SWAR) целочисленными операциями.
            bits = packed[window[start:stop]] & packed[points[start:stop]].unsqueeze(1)  # p × K × слова
            bits = bits - ((bits >> 1) & 0x5555555555555555)
            bits = (bits & 0x3333333333333333) + ((bits >> 2) & 0x3333333333333333)
            bits = (bits + (bits >> 4)) & 0x0F0F0F0F0F0F0F0F
            bits = ((bits * 0x0101010101010101) >> 56) & 0xFF
            common[start:stop] = bits.sum(dim=2)
        return common

    def _packed_codes(self) -> Any:
        """Коды точек 64-битными словами на устройстве (N × ⌈L/64⌉, int64) для подсчёта общих бит (§2.2.4)."""
        if self._codes_packed is None:
            import numpy as np

            codes = (self._codes != 0).to(self._torch.uint8).cpu().numpy()
            words = -(-self._code_length // 64)
            padded = np.zeros((self._point_count, words * 64), dtype=np.uint8)
            padded[:, : self._code_length] = codes
            packed = np.ascontiguousarray(np.packbits(padded, axis=1)).view(np.int64)
            self._codes_packed = self._torch.as_tensor(packed, device=self._device)
        return self._codes_packed

    def update_positions(self, pos_y: Sequence[int], pos_x: Sequence[int]) -> None:
        """Все позиции точек на устройство (после обменов шага; N×2 float32)."""
        torch = self._torch
        import numpy as np

        pos = np.empty((self._point_count, 2), dtype=np.float32)
        pos[:, 0] = pos_y
        pos[:, 1] = pos_x
        self._pos = torch.as_tensor(pos, dtype=torch.float32, device=self._device)
        # Сетка ячеек → индекс точки (−1 — пусто): для локального сбора окрестности (§5.5.2).
        grid = np.full((self._height, self._width), -1, dtype=np.int64)
        pos_y_np = np.asarray(pos_y, dtype=np.int64)
        pos_x_np = np.asarray(pos_x, dtype=np.int64)
        valid = (pos_y_np >= 0) & (pos_x_np >= 0) & (pos_y_np < self._height) & (pos_x_np < self._width)
        grid[pos_y_np[valid], pos_x_np[valid]] = np.flatnonzero(valid)
        self._grid = torch.as_tensor(grid, dtype=torch.long, device=self._device)

    def _sim_lambda_from_common(
        self,
        common: Any,
        ones_rows: Any,
        ones_cols: Any,
        *,
        lambda_threshold: float,
        eta: float | None,
    ) -> Any:
        """sim_λ из числа общих битов и числа единиц кодов (§2.2.4, §5.4); формы транслируются."""
        torch = self._torch
        if self._similarity == "cosine":
            # cos = common / sqrt(n_a·n_v) = common · n_a^−½ · n_v^−½: два прохода по матрице вместо пяти;
            # нулевой код даёт множитель 0 (а не деление на ноль).
            inv_rows = torch.where(ones_rows > 0, torch.rsqrt(torch.clamp(ones_rows, min=1.0)), torch.zeros_like(ones_rows))
            inv_cols = torch.where(ones_cols > 0, torch.rsqrt(torch.clamp(ones_cols, min=1.0)), torch.zeros_like(ones_cols))
            sim = common * inv_rows
            sim.mul_(inv_cols)
        else:
            denominator = ones_rows + ones_cols - common
            positive = (denominator > 0) & (ones_rows > 0) & (ones_cols > 0)
            sim = torch.where(positive, common / torch.where(positive, denominator, torch.ones_like(denominator)), torch.zeros_like(common))
        # sim ≥ 0 по построению (общие биты ≥ 0); порог λ (§5.4): жёсткий или сигмоида η.
        if eta is None:
            sim.masked_fill_(sim < lambda_threshold if lambda_threshold > 0.0 else sim <= 0.0, 0.0)
            return sim
        return sim / (1.0 + torch.exp(-eta * (sim - lambda_threshold)))

    def _similarity_lambda(
        self,
        row_indices: Any,
        col_codes: Any,
        col_ones: Any,
        *,
        lambda_threshold: float,
        eta: float | None,
    ) -> Any:
        """sim_λ(a, v) для кодов a (по индексам) и столбцов v: общие биты через матричное произведение (§2.2.4, §5.4)."""
        common = self._codes[row_indices] @ col_codes.T
        return self._sim_lambda_from_common(
            common, self._ones[row_indices][:, None], col_ones[None, :], lambda_threshold=lambda_threshold, eta=eta
        )

    def update_points(
        self,
        indices: Sequence[int],
        pos_y: Sequence[int],
        pos_x: Sequence[int],
        cells: Sequence[tuple[int, int, int]],
    ) -> None:
        """Позиции только сдвинутых точек и изменившиеся ячейки сетки (после обменов шага).

        Точки (индекс, y, x) и ячейки (y, x, точка или −1) уходят на устройство одним массивом.
        """
        torch = self._torch
        import numpy as np

        moved = len(indices)
        changed = len(cells)
        if moved + changed == 0:
            return
        table = np.empty((moved + changed, 3), dtype=np.int64)
        if moved:
            table[:moved, 0] = indices
            table[:moved, 1] = pos_y
            table[:moved, 2] = pos_x
        if changed:
            table[moved:] = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
        device_table = torch.as_tensor(table, device=self._device)
        if moved:
            self._pos[device_table[:moved, 0]] = device_table[:moved, 1:3].to(torch.float32)
        if changed:
            self._grid[device_table[moved:, 0], device_table[moved:, 1]] = device_table[moved:, 2]

    # Ядро обновления ячеек (CUDA): компилируется при первом обновлении, общее для всех движков процесса.
    _CELL_KERNEL: Any = None

    @classmethod
    def _cell_kernel(cls) -> DeviceKernel:
        """Ядро «в ячейке теперь эта точка»: пишет жильца в сетку и позицию жильца в таблицу позиций."""
        if cls._CELL_KERNEL is None:
            from torch.cuda.jiterator import _create_jit_fn

            source = (
                "template <typename T> T damp_cell_update(T index, long long table_ptr, long long grid_ptr,\n"
                "    long long pos_ptr, long long width) {\n"
                "  const int* item = (const int*)table_ptr + 2 * (long long)index;\n"
                "  long long cell = item[0];\n"
                "  int occupant = item[1];\n"
                "  ((long long*)grid_ptr)[cell] = occupant;\n"
                "  if (0 <= occupant) {\n"
                "    float* place = (float*)pos_ptr + 2 * (long long)occupant;\n"
                "    place[0] = (float)(cell / width);\n"
                "    place[1] = (float)(cell % width);\n"
                "  }\n"
                "  return (T)0;\n"
                "}\n"
            )
            cls._CELL_KERNEL = DeviceKernel(_create_jit_fn(source, table_ptr=0, grid_ptr=0, pos_ptr=0, width=1))
        return cls._CELL_KERNEL

    def update_cells(self, cells: Any, occupants: Any) -> None:
        """Изменившиеся ячейки после обменов шага: в ячейке cells[i] (y·ширина + x) теперь точка occupants[i].

        −1 — ячейка пуста. Позиция точки-жильца меняется вместе с ячейкой; ячейки в списке не повторяются.
        На CUDA — одна загрузка таблицы и один запуск ядра вместо нескольких операций над индексами.
        """
        torch = self._torch
        import numpy as np

        count = int(cells.size)
        if count == 0:
            return
        if self._device.type == "cuda" and self._height * self._width < 2**31:
            table = np.empty((count, 2), dtype=np.int32)
            table[:, 0] = cells
            table[:, 1] = occupants
            device_table = torch.as_tensor(table, device=self._device)
            if self._cell_index is None or int(self._cell_index.numel()) < count:
                self._cell_index = torch.arange(max(count, 4096), dtype=torch.int32, device=self._device)
            self._cell_kernel()(
                self._cell_index[:count],
                table_ptr=device_table.data_ptr(),
                grid_ptr=self._grid.data_ptr(),
                pos_ptr=self._pos.data_ptr(),
                width=self._width,
            )
            return
        device_cells = torch.as_tensor(np.asarray(cells, dtype=np.int64), device=self._device)
        device_occupants = torch.as_tensor(np.asarray(occupants, dtype=np.int64), device=self._device)
        self._grid.view(-1)[device_cells] = device_occupants
        moved = device_occupants >= 0
        points = device_occupants[moved]
        places = device_cells[moved]
        self._pos[points, 0] = torch.div(places, self._width, rounding_mode="floor").to(torch.float32)
        self._pos[points, 1] = (places % self._width).to(torch.float32)

    def pair_energies(
        self,
        idx_a: Sequence[int],
        idx_b: Sequence[int],
        pos_a: Sequence[Sequence[float]],
        pos_b: Sequence[Sequence[float]],
        *,
        neighbourhood: bool,
        local_radius: float | None,
        kernel: str,
        euclidean: bool,
        distance_eps: float,
        kernel_width: float,
        lambda_threshold: float,
        eta: float | None,
        eval_indices: Sequence[int] | None = None,
        need_swapped: bool = True,
    ) -> tuple[Any, Any]:
        """(φ_c, φ_s) для всех пар одним проходом по V или B (§5.5, §5.9.4, §5.12); numpy float32.

        idx_b = −1 — вторая ячейка пуста (точка переносится в пустую ячейку, §5.9.1).
        need_swapped = False — φ_s вызывающему не нужна (энергии точек §5.7): в пути по окну ячеек
        её суммы не считаются и возвращаются нули.
        """
        torch = self._torch
        import numpy as np

        pair_count = len(idx_a)
        if pair_count == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
        with torch.no_grad():
            device = self._device
            ia = torch.as_tensor(np.asarray(idx_a, dtype=np.int64), device=device)
            ib = torch.as_tensor(np.asarray(idx_b, dtype=np.int64), device=device)
            pa = torch.as_tensor(np.asarray(pos_a, dtype=np.float32), device=device)
            pb = torch.as_tensor(np.asarray(pos_b, dtype=np.float32), device=device)
            if eval_indices is None:
                eval_idx = self._all_indices
            else:
                eval_idx = torch.as_tensor(np.asarray(eval_indices, dtype=np.int64), device=device)
            eval_pos = self._pos[eval_idx]
            eval_codes = self._codes[eval_idx]
            eval_ones = self._ones[eval_idx]
            eval_count = int(eval_idx.shape[0])
            chunk = max(1, self.CHUNK_ELEMENTS // max(1, eval_count))
            energy_c = torch.zeros(pair_count, dtype=torch.float32, device=device)
            energy_s = torch.zeros(pair_count, dtype=torch.float32, device=device)
            if neighbourhood:
                centre, radius_sq, window_reach = self.neighbourhood_geometry(pa, pb, local_radius)
                # Окно (2R+1)² ячеек вокруг центра покрывает окрестность; если оно меньше всего V, сходства,
                # расстояния и веса считаются только по окну.
                window_cells = (2 * window_reach + 1) ** 2
                if eval_indices is None and window_cells < eval_count:
                    return self._pair_energies_local(
                        ia, ib, pa, pb, centre, radius_sq, window_reach,
                        kernel=kernel, euclidean=euclidean, distance_eps=distance_eps, kernel_width=kernel_width,
                        lambda_threshold=lambda_threshold, eta=eta, need_swapped=need_swapped,
                    )
            for start in range(0, pair_count, chunk):
                stop = min(pair_count, start + chunk)
                a = ia[start:stop]
                b = ib[start:stop]
                has_b = b >= 0
                s1 = self._similarity_lambda(a, eval_codes, eval_ones, lambda_threshold=lambda_threshold, eta=eta)
                s2 = self._similarity_lambda(
                    torch.clamp(b, min=0), eval_codes, eval_ones, lambda_threshold=lambda_threshold, eta=eta
                )
                s2 = torch.where(has_b[:, None], s2, torch.zeros_like(s2))
                # Сами точки пары в сумму не входят.
                self_mask = (eval_idx[None, :] == a[:, None]) | (eval_idx[None, :] == b[:, None])
                s1 = torch.where(self_mask, torch.zeros_like(s1), s1)
                s2 = torch.where(self_mask, torch.zeros_like(s2), s2)
                dy1 = pa[start:stop, 0:1] - eval_pos[None, :, 0]
                dx1 = pa[start:stop, 1:2] - eval_pos[None, :, 1]
                dy2 = pb[start:stop, 0:1] - eval_pos[None, :, 0]
                dx2 = pb[start:stop, 1:2] - eval_pos[None, :, 1]
                d1_sq = dy1 * dy1 + dx1 * dx1
                d2_sq = dy2 * dy2 + dx2 * dx2
                if neighbourhood:
                    dyc = eval_pos[None, :, 0] - centre[start:stop, 0:1]
                    dxc = eval_pos[None, :, 1] - centre[start:stop, 1:2]
                    inside = (dyc * dyc + dxc * dxc) <= radius_sq[start:stop, None]
                    s1 = torch.where(inside, s1, torch.zeros_like(s1))
                    s2 = torch.where(inside, s2, torch.zeros_like(s2))
                    if kernel == "gauss":
                        sig2 = 2.0 * kernel_width * kernel_width * radius_sq[start:stop, None]
                        w1 = torch.exp(-d1_sq / sig2)
                        w2 = torch.exp(-d2_sq / sig2)
                    else:
                        d1 = torch.sqrt(d1_sq) if euclidean else d1_sq
                        d2 = torch.sqrt(d2_sq) if euclidean else d2_sq
                        w1 = 1.0 / (d1 + distance_eps)
                        w2 = 1.0 / (d2 + distance_eps)
                    energy_c[start:stop] = (s1 * w1 + s2 * w2).sum(dim=1)
                    energy_s[start:stop] = (s2 * w1 + s1 * w2).sum(dim=1)
                else:
                    d1 = torch.sqrt(d1_sq) if euclidean else d1_sq
                    d2 = torch.sqrt(d2_sq) if euclidean else d2_sq
                    energy_c[start:stop] = (s1 * d1 + s2 * d2).sum(dim=1)
                    energy_s[start:stop] = (s2 * d1 + s1 * d2).sum(dim=1)
            return energy_c.cpu().numpy(), energy_s.cpu().numpy()

    def _pair_energies_local(
        self,
        ia: Any,
        ib: Any,
        pa: Any,
        pb: Any,
        centre: Any,
        radius_sq: Any,
        window_reach: int,
        *,
        kernel: str,
        euclidean: bool,
        distance_eps: float,
        kernel_width: float,
        lambda_threshold: float,
        eta: float | None,
        need_swapped: bool = True,
    ) -> tuple[Any, Any]:
        """Энергии окрестности (§5.5.2) по окну ячеек сетки вокруг центра пары.

        Та же сумма, что и в полном проходе: точки v с ‖v − c‖ ≤ r (все они лежат в окне), без самой пары;
        сходство, расстояния и веса считаются только для K = (2R+1)² ячеек окна вместо N точек (§5.12).

        Слагаемые считаются блоками из многих порций пар (значение элемента от размера тензора не зависит),
        а суммируются прежними порциями: сумма float32 зависит от порядка сложения, порядок — от формы
        тензора, поэтому форма [порция, K] у редукции та же, что и раньше (_sum_rows_by_chunks).
        need_swapped = False — φ_s не нужна (энергии точек §5.7), её суммы не считаются.
        """
        torch = self._torch
        device = self._device
        pair_count = int(ia.shape[0])
        offsets = torch.arange(-window_reach, window_reach + 1, device=device)
        dy_off, dx_off = torch.meshgrid(offsets, offsets, indexing="ij")
        dy_off = dy_off.reshape(-1)
        dx_off = dx_off.reshape(-1)
        # Порция редукции (пар на одну сумму по окну) и блок арифметики из целого числа порций.
        chunk = self.local_chunk_rows()
        cells = int(dy_off.numel())
        block = self.window_block_rows(cells)
        energy_c = torch.zeros(pair_count, dtype=torch.float32, device=device)
        energy_s = torch.zeros(pair_count, dtype=torch.float32, device=device)
        base_y = torch.floor(centre[:, 0]).to(torch.long)
        base_x = torch.floor(centre[:, 1]).to(torch.long)
        for start in range(0, pair_count, block):
            stop = min(pair_count, start + block)
            a = ia[start:stop]
            b = ib[start:stop]
            has_b = b >= 0
            cell_y = base_y[start:stop, None] + dy_off[None, :]
            cell_x = base_x[start:stop, None] + dx_off[None, :]
            inside_grid = (cell_y >= 0) & (cell_y < self._height) & (cell_x >= 0) & (cell_x < self._width)
            idx = self._grid[cell_y.clamp(0, self._height - 1), cell_x.clamp(0, self._width - 1)]
            idx = torch.where(inside_grid, idx, torch.full_like(idx, -1))
            valid = (idx >= 0) & (idx != a[:, None]) & (idx != b[:, None])
            safe_idx = idx.clamp(min=0)
            point_pos = self._pos[safe_idx]  # p × K × 2
            dyc = point_pos[:, :, 0] - centre[start:stop, 0:1]
            dxc = point_pos[:, :, 1] - centre[start:stop, 1:2]
            valid &= (dyc * dyc + dxc * dxc) <= radius_sq[start:stop, None]
            # Общие биты точки пары нужны только с точками окна (§5.12); порог λ и нормировка — как раньше.
            ones_cols = self._ones[safe_idx]
            common_a = self._window_common_bits(a, safe_idx)
            s1 = self._sim_lambda_from_common(
                common_a, self._ones[a][:, None], ones_cols, lambda_threshold=lambda_threshold, eta=eta
            )
            s1 = torch.where(valid, s1, torch.zeros_like(s1))
            if bool(has_b.any().item()):
                safe_b = torch.clamp(b, min=0)
                common_b = self._window_common_bits(safe_b, safe_idx)
                s2 = self._sim_lambda_from_common(
                    common_b, self._ones[safe_b][:, None], ones_cols, lambda_threshold=lambda_threshold, eta=eta
                )
                s2 = torch.where(valid & has_b[:, None], s2, torch.zeros_like(s2))
            else:
                # Вторая ячейка всех пар пуста (энергии точек §5.7): s2 = 0.
                s2 = torch.zeros_like(s1)
            dy1 = pa[start:stop, 0:1] - point_pos[:, :, 0]
            dx1 = pa[start:stop, 1:2] - point_pos[:, :, 1]
            dy2 = pb[start:stop, 0:1] - point_pos[:, :, 0]
            dx2 = pb[start:stop, 1:2] - point_pos[:, :, 1]
            d1_sq = dy1 * dy1 + dx1 * dx1
            d2_sq = dy2 * dy2 + dx2 * dx2
            if kernel == "gauss":
                sig2 = 2.0 * kernel_width * kernel_width * radius_sq[start:stop, None]
                w1 = torch.exp(-d1_sq / sig2)
                w2 = torch.exp(-d2_sq / sig2)
            else:
                d1 = torch.sqrt(d1_sq) if euclidean else d1_sq
                d2 = torch.sqrt(d2_sq) if euclidean else d2_sq
                w1 = 1.0 / (d1 + distance_eps)
                w2 = 1.0 / (d2 + distance_eps)
            energy_c[start:stop] = self._sum_rows_by_chunks(s1 * w1 + s2 * w2, chunk)
            if need_swapped:
                energy_s[start:stop] = self._sum_rows_by_chunks(s2 * w1 + s1 * w2, chunk)
        return energy_c.cpu().numpy(), energy_s.cpu().numpy()

    def window_block_rows(self, cells: int) -> int:
        """Число строк одного блока арифметики пути по окну ячеек: целое число порций редукции."""
        chunk = self.local_chunk_rows()
        return chunk * max(1, self.LOCAL_BLOCK_ELEMENTS // max(1, chunk * cells))

    def sum_rows_by_chunks(self, terms: Any, chunk: int) -> Any:
        """Суммы строк блока слагаемых [строки, ячейки окна] той же редукцией, что в проходе по окну ячеек."""
        return self._sum_rows_by_chunks(terms, chunk)

    def _sum_rows_by_chunks(self, terms: Any, chunk: int) -> Any:
        """Суммы по строкам тензора [пары, K] порциями по chunk пар — те же числа float32, что даёт sum(dim=1)
        отдельного тензора [chunk, K] каждой порции.

        Редукция суммирует в порядке, который зависит от формы тензора и выравнивания его начала в памяти.
        Порции берутся участками terms; если участок начинается не на границе 512 байт (как начало
        отдельного тензора), порции переносятся в буфер с шагом, кратным 128 элементам.
        """
        torch = self._torch
        rows, width = int(terms.shape[0]), int(terms.shape[1])
        if rows <= chunk:
            return terms.sum(dim=1)
        full = rows // chunk
        elements = chunk * width
        padding = (-elements) % 128
        body = terms[: full * chunk].reshape(full, elements)
        if padding:
            body = torch.nn.functional.pad(body, (0, padding))
        sums = [body[index, :elements].view(chunk, width).sum(dim=1) for index in range(full)]
        if rows > full * chunk:
            sums.append(terms[full * chunk :].clone().sum(dim=1))
        return torch.cat(sums)

    def point_energies(
        self,
        radius: float,
        *,
        lambda_threshold: float,
        eta: float | None,
        distance_eps: float,
    ) -> Any:
        """E(c, r) = Σ_{v ≠ c, ‖v − c‖ ≤ r} sim_λ(c, v) / (d + ε) для всех точек (§5.7)."""
        import numpy as np

        pos = self._pos.cpu().numpy()
        indices = np.arange(self._point_count, dtype=np.int64)
        energy_c, _ = self.pair_energies(
            indices,
            np.full(self._point_count, -1, dtype=np.int64),
            pos,
            pos,
            neighbourhood=True,
            local_radius=float(radius),
            kernel="inverse",
            euclidean=True,
            distance_eps=distance_eps,
            kernel_width=1.0,
            lambda_threshold=lambda_threshold,
            eta=eta,
            need_swapped=False,
        )
        return energy_c

    def log_subset(self, subset_size: int | None) -> None:
        LOGGER.event(
            "layout.gpu.pair_energy.subset",
            section=OPTIM_SUBSET if subset_size is not None else OPTIM_SIM_MATRIX,
            data={
                "points": self._point_count,
                "eval_points": self._point_count if subset_size is None else int(subset_size),
                "mode": "full_space" if subset_size is None else "subset",
            },
        )
