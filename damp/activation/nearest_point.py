"""Место кода на раскладке: точка с самым похожим кодом (§6.1, правило «самые похожие»)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from damp.article_refs import CODE_SPACE_ACTIVATION
from damp.layout.damp_layout import Layout
from damp.logging import LOGGER


@dataclass(frozen=True)
class NearestPoints:
    """Для каждого кода — зажжённая точка раскладки: центр её ячейки (y, x) и сходство с её кодом."""

    cells: "np.ndarray"  # (m, 2) центры ячеек
    similarity: "np.ndarray"  # (m,)


class NearestPointActivation:
    """Код-стимул зажигает одну точку раскладки — с самым похожим кодом (косинусное сходство, §2.2.4, §6.1).

    Так новый код получает место на готовой карте: похожие коды попадают в одну точку или в соседние.
    """

    BLOCK = 4096

    def __init__(self, layout: Layout, *, use_gpu: bool = True, name: str = "layout") -> None:
        import torch

        table = layout.code_table
        self._length = int(table.code_length)
        self._row_bytes = -(-self._length // 8)
        self._torch = torch
        self._device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self._bits = self._to_device(np.ascontiguousarray(table.packed))
        self._ones = self._bits.sum(dim=1)
        self._cells = layout.cells_array().astype(np.float64) + 0.5
        self._name = str(name)

    def _to_device(self, packed: "np.ndarray"):  # noqa: ANN202
        bits = np.unpackbits(packed, axis=1, count=self._length)
        return self._torch.as_tensor(bits, device=self._device).to(self._torch.float32)

    def place(self, packed: "np.ndarray", *, similarity_threshold: float) -> NearestPoints:
        """Места кодов (строки × ⌈L/8⌉); similarity_threshold нужен только для сводки в логе."""
        torch = self._torch
        packed = np.ascontiguousarray(packed, dtype=np.uint8)
        if packed.ndim != 2 or packed.shape[1] != self._row_bytes:
            raise ValueError("codes must be rows × ceil(code_length / 8) packed bytes")
        # Одинаковые коды зажигают одну и ту же точку: ищем только для различных.
        view = packed.view(np.dtype((np.void, packed.shape[1]))).reshape(-1)
        _, first, inverse = np.unique(view, return_index=True, return_inverse=True)
        inverse = inverse.reshape(-1)
        index, value = [], []
        for start in range(0, first.shape[0], self.BLOCK):
            block = self._to_device(packed[first[start : start + self.BLOCK]])
            cosine = (block @ self._bits.T) / torch.sqrt(
                torch.clamp(block.sum(dim=1)[:, None] * self._ones[None, :], min=1.0)
            )
            best, where = cosine.max(dim=1)
            index.append(where.cpu().numpy())
            value.append(best.cpu().numpy())
        index = np.concatenate(index)[inverse]
        value = np.concatenate(value)[inverse].astype(np.float64)
        LOGGER.event(
            "activation.nearest_point",
            section=CODE_SPACE_ACTIVATION,
            data={
                "name": self._name,
                "codes": int(packed.shape[0]),
                "distinct": int(first.shape[0]),
                "points": int(self._cells.shape[0]),
                "similarity_mean": float(value.mean()),
                "similarity_min": float(value.min()),
                "exact": float(np.mean(value >= 1.0 - 1e-6)),
                "similarity_threshold": float(similarity_threshold),
                "at_least_threshold": float(np.mean(value >= similarity_threshold)),
                "note": "код зажигает точку раскладки с самым похожим кодом; at_least_threshold — доля кодов, "
                "у которых эта точка похожа не меньше порога",
            },
        )
        return NearestPoints(cells=self._cells[index], similarity=value)
