# -*- coding: utf-8 -*-
"""Распадается ли класс в пространстве кодов верхнего уровня (процессор, данные — из кеша шагов).

    python tools/tuning/top/codes_purity.py <картинок> [сколько последних записей кеша=3]

Берёт из кеша шагов записи coding.keyholes верхнего уровня (одно окно на картинку: packed [картинок, байт]) —
последние по времени; метки — первые <картинок> меток MNIST train. По каждой цифре:
  — доля «своих» среди k ближайших соседей точки по косинусу кодов (k = 10);
  — на сколько частей распадается класс в графе «точка — её 5 ближайших своего класса среди 15 ближайших всех»
    (та же мерка, что tools/tuning/top/class_split.py) и доля класса в главной части;
  — лучшее деление класса надвое (по знаку главного собственного вектора центрированной матрицы сходств класса):
    размеры частей, среднее сходство внутри частей и между ними."""
import json
import pathlib
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path[:0] = ["client", "system"]
from damp import settings  # noqa: E402

ROOT = pathlib.Path(settings.CACHE_DIR)  # кеш шагов (client/config.toml: CACHE_DIR)


def records(count: int, last: int) -> list:
    found = []
    for path in (ROOT / "steps" / "coding.keyholes").rglob("*"):
        if not path.is_file():
            continue
        entry = json.loads(path.read_text(encoding="utf-8"))
        packed = entry.get("outputs", {}).get("packed", {})
        if packed.get("shape", [0])[0] == count and entry["outputs"].get("place_length", {}).get("value") is not None:
            found.append((entry.get("created", ""), packed["blob"], int(entry["outputs"]["code_length"]["value"])))
    return sorted(found)[-last:]


def labels_of(count: int) -> "np.ndarray":
    raw = open("data/MNIST/raw/train-labels-idx1-ubyte", "rb").read()
    return np.frombuffer(raw, dtype=np.uint8, offset=8)[:count].astype(np.int64)


def study(bits: "np.ndarray", labels: "np.ndarray") -> list:
    ones = bits.sum(axis=1, dtype=np.float64)
    common = bits.astype(np.float32) @ bits.astype(np.float32).T
    sim = common / np.sqrt(np.maximum(ones[:, None] * ones[None, :], 1.0))
    np.fill_diagonal(sim, -1.0)
    order = np.argsort(-sim, axis=1)
    rows = []
    for digit in range(10):
        own = np.flatnonzero(labels == digit)
        near10 = labels[order[own, :10]] == digit
        # Граф класса: 5 ближайших своего класса среди 15 ближайших всех.
        src, dst = [], []
        position = {int(p): i for i, p in enumerate(own)}
        for p in own:
            mates = [int(q) for q in order[p, :15] if labels[q] == digit][:5]
            src += [position[int(p)]] * len(mates)
            dst += [position[q] for q in mates]
        graph = coo_matrix((np.ones(len(src)), (src, dst)), shape=(own.size, own.size))
        parts, part = connected_components(graph, directed=False)
        main = np.bincount(part).max() / own.size
        block = sim[np.ix_(own, own)].astype(np.float64)
        np.fill_diagonal(block, np.nan)
        inside = np.nanmean(block)
        filled = np.where(np.isnan(block), inside, block)
        centred = filled - filled.mean(axis=0, keepdims=True) - filled.mean(axis=1, keepdims=True) + filled.mean()
        _, vectors = np.linalg.eigh(centred)
        side = vectors[:, -1] > 0
        small = min(int(side.sum()), int((~side).sum()))
        if small >= 2:
            within = np.nanmean(np.concatenate([block[np.ix_(side, side)].ravel(), block[np.ix_(~side, ~side)].ravel()]))
            between = float(np.mean(block[np.ix_(side, ~side)]))
        else:
            within, between = inside, inside
        others = sim[np.ix_(own, np.flatnonzero(labels != digit))].mean()
        rows.append((digit, own.size, near10.mean(), parts, main, small / own.size, inside, within, between, others))
    return rows


def main() -> None:
    count = int(sys.argv[1])
    last = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    labels = labels_of(count)
    total = None
    found = records(count, last)
    for created, blob, length in found:
        packed = np.load(ROOT / "blobs" / blob[:2] / f"{blob}.npy", allow_pickle=False)
        bits = np.unpackbits(packed, axis=1, count=length)
        rows = study(bits, labels)
        print(f"=== запись {created}: кодов {bits.shape[0]}, длина {length}, единиц в среднем {bits.sum(axis=1).mean():.0f}")
        print("цифра | точек | своих среди 10 ближайших | частей графа | доля в главной | меньшая половина | сходство: внутри класса | внутри половин | между половинами | с чужими")
        for row in rows:
            print(f"{row[0]} | {row[1]} | {row[2]:.2f} | {row[3]} | {row[4]:.2f} | {row[5]:.2f} | {row[6]:.3f} | {row[7]:.3f} | {row[8]:.3f} | {row[9]:.3f}")
        table = np.array([row[1:] for row in rows], dtype=np.float64)
        total = table if total is None else total + table
        print(f"Итого | {int(table[:, 0].sum())} | {table[:, 1].mean():.2f} | {table[:, 2].mean():.1f} | {table[:, 3].mean():.2f} | {table[:, 4].mean():.2f} | "
              f"{table[:, 5].mean():.3f} | {table[:, 6].mean():.3f} | {table[:, 7].mean():.3f} | {table[:, 8].mean():.3f}")
    if total is not None and len(found) > 1:
        total /= len(found)
        print(f"=== среднее по {len(found)} записям")
        print("цифра | своих среди 10 ближайших | частей графа | доля в главной | меньшая половина | внутри половин − между")
        for digit in range(10):
            print(f"{digit} | {total[digit, 1]:.2f} | {total[digit, 2]:.1f} | {total[digit, 3]:.2f} | {total[digit, 4]:.2f} | {total[digit, 6] - total[digit, 7]:+.3f}")


if __name__ == "__main__":
    main()
