# -*- coding: utf-8 -*-
"""Верхний уровень, пространство кодов: (1) чистота при выровненном числе точек класса; (2) сходства: та же картинка, свой
класс, чужой; (3) на уровне картинок: ближайшая картинка по среднему сходству точек — свой класс? Процессор."""
import sys

import numpy as np

K = 10
z = np.load(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
bits = np.unpackbits(z[f"L{n}_bits"], axis=1)[:, : int(z[f"L{n}_length"])].astype(np.float32)
labels, picture = z[f"L{n}_labels"], z[f"L{n}_picture"]
x = bits / np.maximum(np.sqrt(bits.sum(axis=1, keepdims=True)), 1e-6)
sim = x @ x.T
same_picture = picture[:, None] == picture[None, :]
same_class = labels[:, None] == labels[None, :]
print(f"ур. {n}: точек {x.shape[0]}, картинок {np.unique(picture).size}")

# (1) чистота: как есть и при выровненных классах (по m точек на класс, 40 повторов)
def purity(index):
    s = sim[np.ix_(index, index)].copy()
    s[same_picture[np.ix_(index, index)]] = -1.0
    near = np.argpartition(-s, K, axis=1)[:, :K]
    hit = labels[index][near] == labels[index][:, None]
    return np.array([hit[labels[index] == d].mean() for d in range(10)])

full = purity(np.arange(x.shape[0]))
rng = np.random.default_rng(0)
m = min(int((labels == d).sum()) for d in range(10))
even = np.mean([purity(np.concatenate([rng.choice(np.flatnonzero(labels == d), m, replace=False) for d in range(10)])) for _ in range(40)], axis=0)
print("цифра            " + " ".join(f"{d:5d}" for d in range(10)) + "  среднее")
print("точек            " + " ".join(f"{int((labels == d).sum()):5d}" for d in range(10)))
print("чистота как есть " + " ".join(f"{v:5.2f}" for v in full) + f"  {full.mean():.3f}")
print(f"поровну (по {m:3d}) " + " ".join(f"{v:5.2f}" for v in even) + f"  {even.mean():.3f}")

# (2) сходства
off = ~np.eye(x.shape[0], dtype=bool)
print(f"сходство: та же картинка {sim[same_picture & off].mean():.3f}; свой класс, другая картинка {sim[same_class & ~same_picture].mean():.3f}; "
      f"чужой класс {sim[~same_class].mean():.3f}")
for d in range(10):
    own = labels == d
    a = sim[np.ix_(own, own)][~same_picture[np.ix_(own, own)]].mean()
    b = sim[np.ix_(own, ~own)].mean()
    top = np.sort(sim[np.ix_(own, ~own)], axis=1)[:, -K:].mean()
    top_own = np.sort(np.where(same_picture[np.ix_(own, own)], -1, sim[np.ix_(own, own)]), axis=1)[:, -K:].mean()
    print(f"  {d}: свои {a:.3f}, чужие {b:.3f}, отрыв {a - b:+.3f}; 10 ближайших своих {top_own:.3f}, 10 ближайших чужих {top:.3f}")

# (3) картинки: среднее по точкам картинки лучшего сходства с точками другой картинки
ids = np.unique(picture)
label_of = np.array([labels[picture == p][0] for p in ids])
member = np.stack([picture == p for p in ids]).astype(np.float32)
best = np.stack([sim[:, picture == p].max(axis=1) for p in ids], axis=1)        # точка × картинка: лучшее сходство
pair = (member @ best) / member.sum(axis=1, keepdims=True)                      # картинка × картинка
pair = (pair + pair.T) / 2
np.fill_diagonal(pair, -1)
order = np.argsort(-pair, axis=1)
first = label_of[order[:, 0]] == label_of
five = (label_of[order[:, :5]] == label_of[:, None]).mean(axis=1)
print("картинки: ближайшая картинка своего класса — " + " ".join(f"{d}:{first[label_of == d].mean():.2f}" for d in range(10)) + f"  среднее {first.mean():.3f}")
print("картинки: свои среди 5 ближайших      — " + " ".join(f"{d}:{five[label_of == d].mean():.2f}" for d in range(10)) + f"  среднее {five.mean():.3f}")
