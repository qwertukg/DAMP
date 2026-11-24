"""Коды точек int8 на устройстве (LAYOUT_INT8_CODES) не меняют результат раскладки.

Число общих бит кодов (§2.2.4) — целое: его считает либо произведение матриц float32, либо целочисленное произведение
int8 × int8 → int32 (torch._int_mm). Оба дают одно и то же число, поэтому сходства, энергии пар (§5.5) и сама раскладка
совпадают до бита. Проверяются дополнения нулями (число точек и длина кода не кратны 8, порции строк короче 17),
подмножество столбцов, проход по окну ячеек, слитое ядро и отказ на коды float32, когда сверка со словами не сошлась.
"""
from __future__ import annotations

import os
import random
import unittest

os.environ.setdefault("RERUN", "off")

import numpy as np  # noqa: E402

from damp.encoding.bitarray import BitArray  # noqa: E402
from damp.layout.damp_layout import AdaptiveLayoutConfig, Layout  # noqa: E402
from damp.layout.pair_energy_gpu import TorchPairEnergyEngine  # noqa: E402
from damp.layout.payload import LayoutPayloadBuilder  # noqa: E402
from damp.logging import LOGGER  # noqa: E402

LOGGER.configure_intervals({}, default_interval=10**9)

_CODE_LENGTH = 99  # не кратно 8: код на устройстве дополняется нулевыми битами
_GROUPS = 12
_POINTS = 501  # не кратно 8: коды дополняются нулевыми точками
_LAYOUT_PARAMS = dict(
    similarity="cosine", lambda_threshold=0.6, eta=10.0, use_gpu=True, seed=5,
    long_distance="euclidean", short_distance="squared", long_kernel="gauss", long_kernel_width=1.0,
)


def _table(points: int = _POINTS, seed: int = 21):
    rng = random.Random(seed)
    prototypes = [rng.sample(range(_CODE_LENGTH), 20) for _ in range(_GROUPS)]
    builder = LayoutPayloadBuilder(similarity=None, lambda_threshold=None, eta=None, metadata={"encoder": "тест"})
    for index in range(points):
        group = (index * 5) % _GROUPS
        bits = list(prototypes[group])
        for _ in range(rng.randrange(0, 5)):
            bits[rng.randrange(len(bits))] = rng.randrange(_CODE_LENGTH)
        builder.add_code(label=str(index % 10), hue=float(group) * 7.5,
                         code=BitArray.from_indices(_CODE_LENGTH, sorted(set(bits))))
    return builder.code_table()


def _bits(values: "np.ndarray") -> "np.ndarray":
    return np.ascontiguousarray(values).view(np.int32)


class Int8CodesTests(unittest.TestCase):
    """Коды int8 и float32 дают одни и те же общие биты, энергии и раскладку."""

    def _layouts(self, **kwargs) -> tuple[Layout, Layout]:
        plain = Layout(_table(), **{**_LAYOUT_PARAMS, **kwargs})
        fast = Layout(_table(), int8_codes=True, **{**_LAYOUT_PARAMS, **kwargs})
        if plain._torch_engine is None or fast._torch_engine is None or not fast._torch_engine.int8_codes:
            self.skipTest("коды int8 недоступны (нужен torch с CUDA и torch._int_mm)")
        return plain, fast

    def test_int8_codes_are_padded_and_hold_the_same_bits(self) -> None:
        plain, fast = self._layouts()
        a, b = plain._torch_engine, fast._torch_engine
        torch = a.torch_mod
        self.assertFalse(a.int8_codes)
        self.assertTrue(a.has_dense_codes and b.has_dense_codes)
        self.assertEqual(tuple(b.codes.shape), (_POINTS, 104))
        self.assertEqual(b.codes.dtype, torch.int8)
        self.assertTrue(bool(torch.equal(b.codes[:, :_CODE_LENGTH].to(torch.float32), a.codes)))
        self.assertEqual(int(b.codes[:, _CODE_LENGTH:].abs().sum()), 0)
        self.assertTrue(bool(torch.equal(a.packed_codes(), b.packed_codes())))

    def test_common_bits_equal_for_any_block_shape(self) -> None:
        plain, fast = self._layouts()
        a, b = plain._torch_engine, fast._torch_engine
        torch = a.torch_mod
        rng = np.random.default_rng(3)
        for rows in (1, 5, 16, 17, 40, _POINTS):
            index = torch.as_tensor(rng.choice(_POINTS, size=rows, replace=rows > _POINTS), device=a.torch_device)
            # Со всеми точками V.
            expected = a.common_bits(a.codes[index], a.codes)
            found = b.common_bits(b.codes[index], b.codes)
            self.assertEqual(found.dtype, torch.float32)
            self.assertTrue(bool(torch.equal(expected, found)), f"все точки, строк {rows}")
            # С подмножеством столбцов любого размера (кратного 8 и нет).
            for cols in (1, 7, 8, 9, 96):
                subset = torch.as_tensor(rng.choice(_POINTS, size=cols, replace=False), device=a.torch_device)
                expected = a.common_bits(a.codes[index], a.codes[subset])
                found = b.common_bits(b.codes[index], b.codes[subset])
                self.assertTrue(bool(torch.equal(expected, found)), f"строк {rows}, столбцов {cols}")
        empty = torch.zeros(0, dtype=torch.long, device=a.torch_device)
        self.assertEqual(tuple(b.common_bits(b.codes[empty], b.codes).shape), (0, _POINTS))

    def test_similarity_rows_and_window_bits_equal(self) -> None:
        plain, fast = self._layouts()
        a, b = plain._torch_engine, fast._torch_engine
        torch = a.torch_mod
        rows = torch.arange(0, _POINTS, 3, device=a.torch_device)
        for eta in (10.0, None):
            expected = a.similarity_rows(rows, lambda_threshold=0.6, eta=eta)
            found = b.similarity_rows(rows, lambda_threshold=0.6, eta=eta)
            self.assertTrue(np.array_equal(_bits(expected.cpu().numpy()), _bits(found.cpu().numpy())), f"eta={eta}")
        rng = np.random.default_rng(4)
        points = torch.as_tensor(rng.integers(0, _POINTS, size=37), device=a.torch_device)
        window = torch.as_tensor(rng.integers(0, _POINTS, size=(37, 25)), device=a.torch_device)
        self.assertTrue(bool(torch.equal(a.window_common_bits(points, window), b.window_common_bits(points, window))))

    def test_pair_energies_equal(self) -> None:
        """Исходный проход (по всему V, по подмножеству B, по окну ячеек) и слитое ядро — те же φ_c, φ_s."""
        plain, fast = self._layouts(fused_table_max_bytes=0, fused_similarity_table=False)
        rng = np.random.default_rng(6)
        for layout in (plain, fast):
            layout._sync_arrays()
            layout._torch_engine.update_positions(layout._pos_np[:, 0], layout._pos_np[:, 1])
        self.assertTrue(np.array_equal(plain._pos_np, fast._pos_np))
        idx_a = rng.permutation(_POINTS)[:200]
        idx_b = rng.permutation(_POINTS)[:200]
        keep = idx_a != idx_b
        idx_a, idx_b = idx_a[keep], idx_b[keep]
        pos_a, pos_b = plain._pos_np[idx_a], plain._pos_np[idx_b]
        subset = random.Random(1).sample(range(_POINTS), 97)
        cases = (
            dict(neighbourhood=False, local_radius=None, kernel="inverse", euclidean=True),
            dict(neighbourhood=False, local_radius=None, kernel="inverse", euclidean=True, eval_indices=subset),
            dict(neighbourhood=True, local_radius=5, kernel="gauss", euclidean=False),
            dict(neighbourhood=True, local_radius=40, kernel="gauss", euclidean=False),
        )
        for case in cases:
            for eta in (10.0, None):
                arguments = dict(distance_eps=plain._distance_eps, kernel_width=1.0, lambda_threshold=0.6, eta=eta, **case)
                expected = plain._torch_engine._pair_energies_reference(idx_a, idx_b, pos_a, pos_b, **arguments)
                found = fast._torch_engine._pair_energies_reference(idx_a, idx_b, pos_a, pos_b, **arguments)
                label = f"{case} eta={eta}"
                self.assertTrue(np.array_equal(_bits(expected[0]), _bits(found[0])), label)
                self.assertTrue(np.array_equal(_bits(expected[1]), _bits(found[1])), label)
                self.assertGreater(float(np.abs(expected[0]).max()), 0.0, label)
                fused = fast._torch_engine._fused_engine()
                if fused is not None:
                    merged = fused.pair_energies(idx_a, idx_b, pos_a, pos_b, positions=fast._pos_np, grid=fast._grid_np,
                                                 **arguments)
                    if merged is not None:
                        self.assertTrue(np.array_equal(_bits(merged[0]), _bits(expected[0])), "слитое ядро: " + label)
                        self.assertTrue(np.array_equal(_bits(merged[1]), _bits(expected[1])), "слитое ядро: " + label)

    def test_layout_run_equals(self) -> None:
        """Дальняя фаза с таблицами слитого ядра и без них: расстановка та же шаг в шаг."""
        for options in ({}, dict(fused_table_max_bytes=0, fused_similarity_table=False)):
            plain, fast = self._layouts(**options)
            for layout in (plain, fast):
                adaptive = AdaptiveLayoutConfig(
                    start_radius=layout.width, end_radius=1, swap_ratio_trigger=0.0, lambda_step=None,
                    radius_schedule="geometric", swap_window=3, temperature=0.1,
                    temperature_radius_max=10, temperature_radius_min=3,
                )
                layout.run(steps=120, pairs_per_step=400, pair_radius=layout.width, mode="long",
                           min_swap_ratio=0.0, adaptive_params=adaptive)
            self.assertTrue(np.array_equal(plain.cells_array(), fast.cells_array()), str(options))

    def test_failed_check_falls_back_to_float32(self) -> None:
        plain, _ = self._layouts()
        check = TorchPairEnergyEngine._int8_matches_words
        TorchPairEnergyEngine._int8_matches_words = lambda self: False
        try:
            layout = Layout(_table(), int8_codes=True, **_LAYOUT_PARAMS)
        finally:
            TorchPairEnergyEngine._int8_matches_words = check
        engine = layout._torch_engine
        self.assertFalse(engine.int8_codes)
        self.assertTrue(engine.has_dense_codes)
        self.assertTrue(bool(engine.torch_mod.equal(engine.codes, plain._torch_engine.codes)))


if __name__ == "__main__":
    unittest.main()
