# -*- coding: utf-8 -*-
"""Кто чужие соседи в пространстве кодов: по выгрузке — для уровней 3–5 доля класса среди 10 ближайших по коду точек других
картинок (строка — класс точки, столбец — класс соседа). Только numpy, процессор."""
import sys

import numpy as np

K = 10
z = np.load(sys.argv[1])
for n in sorted(int(k[1:].split("_")[0]) for k in z.files if k.endswith("_bits") and int(k[1:].split("_")[0]) >= 2):
    bits = np.unpackbits(z[f"L{n}_bits"], axis=1)[:, : int(z[f"L{n}_length"])].astype(np.float32)
    labels, picture = z[f"L{n}_labels"], z[f"L{n}_picture"]
    x = bits / np.maximum(np.sqrt(bits.sum(axis=1, keepdims=True)), 1e-6)
    table = np.zeros((10, 10))
    own_picture = np.zeros(10)
    for start in range(0, x.shape[0], 512):
        sim = x[start:start + 512] @ x.T
        rows = np.arange(start, min(start + 512, x.shape[0]))
        sim[picture[rows][:, None] == picture[None, :]] = -1.0
        near = np.argpartition(-sim, K, axis=1)[:, :K]
        for r, q in enumerate(rows):
            np.add.at(table[labels[q]], labels[near[r]], 1.0)
    table /= table.sum(axis=1, keepdims=True)
    print(f"\nур. {n}: точек {x.shape[0]}, единиц в коде {bits.sum(axis=1).mean():.0f} из {bits.shape[1]}; чистота кода {np.mean([table[d, d] for d in range(10)]):.3f}")
    print("     " + " ".join(f"{d:5d}" for d in range(10)) + "   точек  на картинку")
    for d in range(10):
        own = labels == d
        per = own.sum() / max(1, np.unique(picture[own]).size)
        cells = " ".join(("  .  " if table[d, e] < 0.03 else f"{table[d, e]:5.2f}") for e in range(10))
        print(f"  {d}: {cells}   {own.sum():5d}  {per:5.1f}")
