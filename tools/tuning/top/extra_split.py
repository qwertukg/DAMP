# -*- coding: utf-8 -*-
"""Почему на большой верхней карте цифра распадается: код или раскладка (к tools/tuning/top/extra_pictures.py).

    python tools/tuning/top/extra_split.py <зёрна=6,1> [новых=0,500,1500,3500] [раскладки=0,1,2]

По кодам верхних окон (data/top-stand/extra-seedN.npz) для каждого набора «свои 500 + новые M»:
- в пространстве кодов (без раскладки): чистота 10 ближайших по сходству кодов, главный чужой сосед, на сколько связных
  частей распадается класс в графе 5 ближайших своих соседей — это свойства кода;
- на карте (раскладка настройками опоры): доли двух самых больших пятен цифры и цифра, чьих точек больше всего на
  границе между ними, — это свойство раскладки.
Класс цел в пространстве кодов, а на карте в двух пятнах — виновата раскладка; распался уже в кодах — код.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "client"), os.path.join(ROOT, "system"), os.path.dirname(os.path.abspath(__file__))]
os.environ.setdefault("DAMP_LOG_CONSOLE", "heartbeat")
os.environ.setdefault("RERUN", "off")
os.environ.setdefault("DAMP_RERUN_VIEWER", "off")
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

import extra_pictures as E  # noqa: E402

OUT = os.path.join(E.FOLDER, "extra-split.jsonl")
EIGHT = np.ones((3, 3), bool)


def code_space(bits: "np.ndarray", labels: "np.ndarray", k: int = 10, own: int = 5) -> dict:
    """Чистота k ближайших по сходству кодов, главный чужой сосед и число связных частей класса (граф own своих соседей)."""
    codes = bits.astype(np.float32)
    norm = np.sqrt(codes.sum(axis=1))
    similarity = (codes @ codes.T) / np.maximum(1e-9, norm[:, None] * norm[None, :])
    np.fill_diagonal(similarity, -1.0)
    near = np.argpartition(-similarity, k, axis=1)[:, :k]
    purity, foreign, parts, largest = [], [], [], []
    for digit in range(10):
        rows = np.flatnonzero(labels == digit)
        if rows.size == 0:
            purity.append(0.0), foreign.append(-1), parts.append(0), largest.append(0.0)
            continue
        neighbours = labels[near[rows]]
        purity.append(float((neighbours == digit).mean()))
        others = neighbours[neighbours != digit]
        foreign.append(int(np.bincount(others, minlength=10).argmax()) if others.size else -1)
        inner = similarity[np.ix_(rows, rows)]
        take = min(own, rows.size - 1)
        if take <= 0:
            parts.append(1), largest.append(1.0)
            continue
        index = np.argpartition(-inner, take, axis=1)[:, :take]
        graph = csr_matrix((np.ones(index.size), (np.repeat(np.arange(rows.size), take), index.ravel())), shape=(rows.size, rows.size))
        count, component = connected_components(graph, directed=False)
        parts.append(int(count))
        largest.append(float(np.bincount(component).max() / rows.size))
    return {"purity": purity, "foreign": foreign, "parts": parts, "largest_part": largest}


def patches(cells: "np.ndarray", labels: "np.ndarray", height: int, width: int) -> dict:
    """Доли двух самых больших пятен каждой цифры и цифра, чьих точек больше всего рядом со вторым пятном."""
    grid = np.full((int(height), int(width)), -1, dtype=np.int64)
    grid[cells[:, 0], cells[:, 1]] = labels
    empty = grid < 0
    first, second, between = [], [], []
    for digit in range(10):
        mask = grid == digit
        if not mask.any():
            first.append(0.0), second.append(0.0), between.append(-1)
            continue
        parts, _ = ndimage.label(mask | (empty & ndimage.binary_dilation(mask, structure=EIGHT)), structure=EIGHT)
        sizes = np.bincount(parts[mask])
        sizes[0] = 0
        order = np.argsort(-sizes)
        first.append(float(sizes[order[0]] / mask.sum()))
        if sizes[order[1]] == 0:
            second.append(0.0), between.append(-1)
            continue
        second.append(float(sizes[order[1]] / mask.sum()))
        # Кто окружает второе пятно: чужие цифры в двух клетках от него.
        ring = ndimage.binary_dilation((parts == order[1]) & mask, structure=EIGHT, iterations=2) & (grid >= 0) & (grid != digit)
        between.append(int(np.bincount(grid[ring], minlength=10).argmax()) if ring.any() else -1)
    return {"first": first, "second": second, "between": between}


def main() -> None:
    seeds = [int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "6,1").split(",")]
    extras = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "0,500,1500,3500").split(",")]
    layouts = [int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else "0,1,2").split(",")]
    base = E.base_settings()
    found = []
    for seed in seeds:
        bits, labels, _, new, level = E.load_codes(seed)
        old_rows, new_rows = np.flatnonzero(~new), np.flatnonzero(new)
        for extra in extras:
            chosen = np.concatenate([old_rows, new_rows[:extra]])
            space = code_space(bits[chosen], labels[chosen])
            for s in layouts:
                cells, got, _, size, _ = E.lay(bits[chosen], labels[chosen], level, {**base, **E.LIGHT}, s)
                row = {"seed": seed, "extra": extra, "points": int(chosen.size), "layout_seed": s, **space,
                       **patches(cells, got, size[0], size[1])}
                found.append(row)
                with open(OUT, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"зерно {seed}, новых {extra}: готово", flush=True)
    print("точек | мерка | цифры 0…9")
    for extra in extras:
        rows = [r for r in found if r["extra"] == extra]
        points = rows[0]["points"]

        def mean(key):  # noqa: ANN001, ANN202
            return " ".join(f"{v:.2f}" for v in np.mean([r[key] for r in rows], axis=0))

        print(f"{points} | код: чистота 10 ближайших | {mean('purity')}")
        print(f"{points} | код: доля класса в главной связной части | {mean('largest_part')}")
        print(f"{points} | код: связных частей класса | " + " ".join(f"{v:.1f}" for v in np.mean([r['parts'] for r in rows], axis=0)))
        print(f"{points} | код: главный чужой сосед | " + " ".join(str(np.bincount(np.maximum(0, [r['foreign'][d] for r in rows]), minlength=10).argmax()) for d in range(10)))
        print(f"{points} | карта: главное пятно (М1) | {mean('first')}")
        print(f"{points} | карта: второе пятно | {mean('second')}")
        print(f"{points} | карта: кто рядом со вторым пятном | " + " ".join(
            str(np.bincount([r['between'][d] for r in rows if r['between'][d] >= 0], minlength=10).argmax()) if any(r['between'][d] >= 0 for r in rows) else "-"
            for d in range(10)))
    print("Итого: класс цел в коде (главная связная часть ≈ 1), а второе пятно большое — распад делает раскладка")


if __name__ == "__main__":
    main()
