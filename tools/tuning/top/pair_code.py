# -*- coding: utf-8 -*-
"""Код пар «часть — часть — смещение» против мешка частей и места частей — без стенда, на выгрузке (CPU).

Части картинки — точки карты ур. N (по умолчанию 4) этой картинки: код окна (L{N}_bits) и центр окна (L{N}_centre). Из
них строятся коды картинки длины L:
  мешок   — OR по частям: hash(понятие части);
  место   — OR по частям: hash(понятие, клетка места части относительно центра цифры);
  пары    — OR по упорядоченным парам частей: hash(понятие A, понятие B, клетка смещения B − A);
  место+пары — OR двух предыдущих.
Понятие части — m бит её кода с наименьшим случайным приоритетом (min-hash: у похожих кодов они чаще совпадают).
Место и смещение — в долях K · √(var y + var x) центров частей картинки, клетка — round(Δ / шаг), зажим ±R.
Мерка: у какой доли картинок ближайшая (косинус) — своего класса, и доля своих среди 5 ближайших, по цифрам.

    python tools/tuning/top/pair_code.py <выгрузка.npz> [уровень=4] [m=4] [шаг=0.5]
"""
import sys

import numpy as np

z = np.load(sys.argv[1])
level = int(sys.argv[2]) if len(sys.argv) > 2 else 4
M = int(sys.argv[3]) if len(sys.argv) > 3 else 4
STEP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
K, R, L = 2.0, 3, 8192
rng = np.random.default_rng(0)

bits = np.unpackbits(z[f"L{level}_bits"], axis=1)[:, : int(z[f"L{level}_length"])].astype(bool)
centre, picture, labels = z[f"L{level}_centre"], z[f"L{level}_picture"], z[f"L{level}_labels"]
priority = rng.permutation(bits.shape[1])
concept = np.full((bits.shape[0], M), -1, dtype=np.int64)
for i in range(bits.shape[0]):
    on = np.flatnonzero(bits[i])
    if on.size:
        take = on[np.argsort(priority[on])[:M]]
        concept[i, : take.size] = take


def mix(*values: "np.ndarray") -> "np.ndarray":
    """Хеш нескольких целых → номер бита (одинаковые входы — один бит)."""
    h = np.full(np.broadcast(*values).shape, 1469598103934665603, dtype=np.uint64)
    for v in values:
        h ^= np.asarray(v, dtype=np.int64).astype(np.uint64)
        h *= np.uint64(1099511628211)
        h ^= h >> np.uint64(29)
    return (h % np.uint64(L)).astype(np.int64)


ids = np.unique(picture)
label_of = np.array([labels[picture == p][0] for p in ids])
codes = {name: np.zeros((ids.size, L), dtype=bool) for name in ("мешок", "место", "пары", "место+пары")}
for row, p in enumerate(ids):
    own = np.flatnonzero(picture == p)
    xy = centre[own]
    scale = K * max(float(np.sqrt(np.var(xy[:, 0]) + np.var(xy[:, 1]))), 1e-6)
    cell = np.clip(np.round((xy - xy.mean(axis=0)) / scale / STEP), -R, R).astype(np.int64)
    c = concept[own]                                                    # части × M
    valid = c >= 0
    codes["мешок"][row, mix(c[valid])] = True
    place = mix(c, cell[:, 0:1], cell[:, 1:2])
    codes["место"][row, place[valid]] = True
    if own.size > 1:
        a, b = np.meshgrid(np.arange(own.size), np.arange(own.size), indexing="ij")
        keep = a != b
        a, b = a[keep], b[keep]
        delta = np.clip(np.round((xy[b] - xy[a]) / scale / STEP), -2 * R, 2 * R).astype(np.int64)
        pair = mix(c[a], c[b], delta[:, 0:1], delta[:, 1:2])           # пары × M (i-е понятие A с i-м понятием B)
        ok = valid[a] & valid[b]
        codes["пары"][row, pair[ok]] = True
    codes["место+пары"][row] = codes["место"][row] | codes["пары"][row]

print(f"ур. {level}: картинок {ids.size}, частей на картинку {picture.size / ids.size:.1f}; m {M}, шаг {STEP}, длина {L}")
print("код          единиц  ближайшая своя  свои из 5   по цифрам (свои из 5): 0 … 9")
for name, x in codes.items():
    f = x.astype(np.float32)
    f /= np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-6)
    sim = f @ f.T
    np.fill_diagonal(sim, -1)
    near = np.argsort(-sim, axis=1)[:, :5]
    hit = label_of[near] == label_of[:, None]
    by = [hit[label_of == d].mean() for d in range(10)]
    print(f"{name:11s} {x.sum(axis=1).mean():7.0f}  {hit[:, 0].mean():14.3f}  {hit.mean():9.3f}   " + " ".join(f"{v:.2f}" for v in by))
