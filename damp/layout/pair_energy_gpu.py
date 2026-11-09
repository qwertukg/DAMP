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
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from damp.article_refs import GPU_IMPLEMENTATION, OPTIM_SIM_MATRIX, OPTIM_SUBSET
from damp.logging import LOGGER


class TorchPairEnergyEngine:
    """Энергии пар и точек на torch-устройстве (cuda/mps) без предвычисленной матрицы сходства."""

    # Предел элементов (пары × точки) на один проход: ограничивает память под промежуточные тензоры.
    CHUNK_ELEMENTS = 4_000_000
    # В локальном пути (окно ячеек) крупных тензоров p × N меньше — порция пар крупнее.
    LOCAL_CHUNK_ELEMENTS = 16_000_000

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
        self._height, self._width = int(grid_shape[0]), int(grid_shape[1])
        self._pos = torch_mod.zeros((self._point_count, 2), dtype=torch_mod.float32, device=device)
        self._grid = torch_mod.full((self._height, self._width), -1, dtype=torch_mod.long, device=device)
        self._all_indices = torch_mod.arange(self._point_count, device=device)
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
        """Позиции только сдвинутых точек и изменившиеся ячейки сетки (после обменов шага)."""
        torch = self._torch
        import numpy as np

        if len(indices):
            idx = torch.as_tensor(np.asarray(indices, dtype=np.int64), device=self._device)
            self._pos[idx, 0] = torch.as_tensor(np.asarray(pos_y, dtype=np.float32), device=self._device)
            self._pos[idx, 1] = torch.as_tensor(np.asarray(pos_x, dtype=np.float32), device=self._device)
        if len(cells):
            cell_array = np.asarray(cells, dtype=np.int64).reshape(-1, 3)
            cy = torch.as_tensor(cell_array[:, 0], device=self._device)
            cx = torch.as_tensor(cell_array[:, 1], device=self._device)
            self._grid[cy, cx] = torch.as_tensor(cell_array[:, 2], device=self._device)

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
    ) -> tuple[Any, Any]:
        """(φ_c, φ_s) для всех пар одним проходом по V или B (§5.5, §5.9.4, §5.12); numpy float32.

        idx_b = −1 — вторая ячейка пуста (точка переносится в пустую ячейку, §5.9.1).
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
                centre = (pa + pb) * 0.5
                dist_ab = torch.sqrt(((pa - pb) ** 2).sum(dim=1))
                # §5.5.2: r ≥ |p1 − p2|; +1 гарантирует, что окрестность содержит точки помимо самой пары.
                base_radius = torch.ceil(dist_ab) + 1.0
                radius = base_radius if local_radius is None else torch.clamp(base_radius, min=float(local_radius))
                radius_sq = radius * radius
                # Окно (2R+1)² ячеек вокруг центра покрывает окрестность; если оно меньше всего V, расстояния и
                # веса считаются только по окну (сходства — по-прежнему одним матричным произведением по V).
                window_reach = int(math.ceil(float(radius.max().item()))) + 1
                window_cells = (2 * window_reach + 1) ** 2
                if eval_indices is None and window_cells < eval_count:
                    return self._pair_energies_local(
                        ia, ib, pa, pb, centre, radius_sq, window_reach,
                        kernel=kernel, euclidean=euclidean, distance_eps=distance_eps, kernel_width=kernel_width,
                        lambda_threshold=lambda_threshold, eta=eta,
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
    ) -> tuple[Any, Any]:
        """Энергии окрестности (§5.5.2) по окну ячеек сетки вокруг центра пары.

        Та же сумма, что и в полном проходе: точки v с ‖v − c‖ ≤ r (все они лежат в окне), без самой пары;
        sim_λ берётся из строк матричного произведения кодов пары на все коды V (§5.12), а расстояния и веса
        считаются только для K = (2R+1)² ячеек окна вместо N точек.
        """
        torch = self._torch
        device = self._device
        pair_count = int(ia.shape[0])
        offsets = torch.arange(-window_reach, window_reach + 1, device=device)
        dy_off, dx_off = torch.meshgrid(offsets, offsets, indexing="ij")
        dy_off = dy_off.reshape(-1)
        dx_off = dx_off.reshape(-1)
        # Здесь крупный тензор только один — общие биты пар со всеми точками (p × N) на время gather,
        # поэтому порция пар больше, чем в полном проходе: меньше запусков ядер на шаг.
        chunk = max(1, self.LOCAL_CHUNK_ELEMENTS // max(1, self._point_count))
        energy_c = torch.zeros(pair_count, dtype=torch.float32, device=device)
        energy_s = torch.zeros(pair_count, dtype=torch.float32, device=device)
        base_y = torch.floor(centre[:, 0]).to(torch.long)
        base_x = torch.floor(centre[:, 1]).to(torch.long)
        for start in range(0, pair_count, chunk):
            stop = min(pair_count, start + chunk)
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
            # Общие биты пары со всеми точками — одно матричное произведение (§5.12); порог λ и нормировка
            # считаются только для K столбцов окна, а не для всех N.
            ones_cols = self._ones[safe_idx]
            common_a = torch.gather(self._codes[a] @ self._codes.T, 1, safe_idx)
            s1 = self._sim_lambda_from_common(
                common_a, self._ones[a][:, None], ones_cols, lambda_threshold=lambda_threshold, eta=eta
            )
            safe_b = torch.clamp(b, min=0)
            common_b = torch.gather(self._codes[safe_b] @ self._codes.T, 1, safe_idx)
            s2 = self._sim_lambda_from_common(
                common_b, self._ones[safe_b][:, None], ones_cols, lambda_threshold=lambda_threshold, eta=eta
            )
            s1 = torch.where(valid, s1, torch.zeros_like(s1))
            s2 = torch.where(valid & has_b[:, None], s2, torch.zeros_like(s2))
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
            energy_c[start:stop] = (s1 * w1 + s2 * w2).sum(dim=1)
            energy_s[start:stop] = (s2 * w1 + s1 * w2).sum(dim=1)
        return energy_c.cpu().numpy(), energy_s.cpu().numpy()

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
