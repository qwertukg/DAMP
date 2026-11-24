"""Код или карта: по выгрузке dump-seedN.npz — чистота соседей в пространстве кодов и на карте, сходства своих и чужих,
картинка на верху. Только numpy/scipy. python codespace.py <папка> <зёрна через запятую>"""
import json
import sys

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

EIGHT = np.ones((3, 3), bool)
K = 10


def patches(cells, labels, height, width):
    """Номер пятна каждой точки (пятна своей цифры, 8 соседей, пустая клетка не рвёт) и М1 по цифрам."""
    grid = np.full((int(height), int(width)), -1, dtype=np.int64)
    grid[cells[:, 0], cells[:, 1]] = labels
    empty = grid < 0
    patch = np.zeros(labels.shape[0], dtype=np.int64)
    m1 = {}
    offset = 0
    for d in range(10):
        mask = grid == d
        if not mask.any():
            continue
        joined = mask | (empty & ndimage.binary_dilation(mask, structure=EIGHT))
        parts, count = ndimage.label(joined, structure=EIGHT)
        own = labels == d
        patch[own] = parts[cells[own, 0], cells[own, 1]] + offset
        offset += count + 1
        sizes = np.bincount(parts[mask])[1:]
        m1[d] = float(sizes.max() / mask.sum())
    return patch, m1


def unit(bits):
    x = bits.astype(np.float32)
    x /= np.maximum(np.sqrt(x.sum(axis=1, keepdims=True)), 1e-6)
    return x


def code_purity(x, labels, picture, queries):
    """Честная чистота K ближайших по коду (соседи — с других картинок) для точек queries."""
    pure = np.zeros(queries.shape[0])
    for start in range(0, queries.shape[0], 500):
        q = queries[start:start + 500]
        sim = x[q] @ x.T
        sim[picture[q][:, None] == picture[None, :]] = -1.0
        near = np.argpartition(-sim, K, axis=1)[:, :K]
        pure[start:start + 500] = (labels[near] == labels[q][:, None]).mean(axis=1)
    return pure


def map_purity(cells, labels, picture, queries):
    tree = cKDTree(cells.astype(np.float64))
    _, near = tree.query(cells[queries].astype(np.float64), k=min(K * 6, cells.shape[0]))
    pure = np.zeros(queries.shape[0])
    for row, q in enumerate(queries):
        cand = near[row]
        cand = cand[picture[cand] != picture[q]][:K]
        pure[row] = (labels[cand] == labels[q]).mean() if cand.size else 0.0
    return pure


def main():
    folder, seeds = sys.argv[1], [int(v) for v in sys.argv[2].split(",")]
    rng = np.random.default_rng(0)
    out = {}
    for seed in seeds:
        z = np.load(f"{folder}/dump-seed{seed}.npz")
        levels = sorted(int(k[1:].split("_")[0]) for k in z.files if k.endswith("_bits"))
        report = {}
        for n in levels:
            bits = np.unpackbits(z[f"L{n}_bits"], axis=1)[:, : int(z[f"L{n}_length"])]
            labels, picture, cells = z[f"L{n}_labels"], z[f"L{n}_picture"], z[f"L{n}_cells"]
            height, width = z[f"L{n}_size"]
            count = labels.shape[0]
            queries = np.arange(count) if count <= 4000 else np.sort(rng.choice(count, 3000, replace=False))
            x = unit(bits)
            cp, mp = code_purity(x, labels, picture, queries), map_purity(cells, labels, picture, queries)
            patch, m1 = patches(cells, labels, height, width)
            row = {"points": int(count), "density": float(bits.mean()), "code_purity": float(cp.mean()), "map_purity": float(mp.mean()),
                   "code_purity_digit": [float(cp[labels[queries] == d].mean()) for d in range(10)],
                   "map_purity_digit": [float(mp[labels[queries] == d].mean()) for d in range(10)],
                   "m1_digit": [m1.get(d, 0.0) for d in range(10)], "m1u": float(np.mean([m1.get(d, 0.0) for d in range(10)]))}
            if n == levels[-1] or n == levels[-2]:
                sim = x @ x.T
                same_pic = picture[:, None] == picture[None, :]
                same_cls = labels[:, None] == labels[None, :]
                eye = np.eye(count, dtype=bool)
                row["sim_same_picture"] = float(sim[same_pic & ~eye].mean())
                row["sim_own_other_picture"] = float(sim[same_cls & ~same_pic].mean())
                row["sim_foreign"] = float(sim[~same_cls].mean())
                row["sim_own_digit"] = [float(sim[(labels[:, None] == d) & (labels[None, :] == d) & ~same_pic].mean()) for d in range(10)]
                row["sim_foreign_digit"] = [float(sim[(labels[:, None] == d) & (labels[None, :] != d)].mean()) for d in range(10)]
                # Картинка: сколько у неё точек и в скольких пятнах они лежат
                pics = np.unique(picture)
                per = [int((picture == p).sum()) for p in pics]
                spread = [int(np.unique(patch[picture == p]).size) for p in pics]
                row["points_per_picture"] = float(np.mean(per))
                row["patches_per_picture"] = float(np.mean(spread))
                row["pictures_in_one_patch"] = float(np.mean(np.asarray(spread) == 1))
                # Сходство картинок: среднее по парам их окон; чистота K ближайших картинок
                index = {p: np.flatnonzero(picture == p) for p in pics}
                member = np.zeros((pics.size, count), dtype=np.float32)
                for i, p in enumerate(pics):
                    member[i, index[p]] = 1.0 / index[p].size
                pic_sim = member @ sim @ member.T
                np.fill_diagonal(pic_sim, -1.0)
                pic_label = np.asarray([np.bincount(labels[index[p]], minlength=10).argmax() for p in pics])
                k = 5
                near = np.argpartition(-pic_sim, k, axis=1)[:, :k]
                pp = (pic_label[near] == pic_label[:, None]).mean(axis=1)
                row["picture_knn5_purity"] = float(pp.mean())
                row["picture_knn5_purity_digit"] = [float(pp[pic_label == d].mean()) for d in range(10)]
                # то же для окон при k = 5 — для сравнения с картинками
                s2 = sim.copy()
                s2[same_pic] = -1.0
                near = np.argpartition(-s2, k, axis=1)[:, :k]
                row["window_knn5_purity"] = float((labels[near] == labels[:, None]).mean())
                # лучшая пара окон двух картинок (max) → чистота ближайших картинок
                best = np.full((pics.size, pics.size), -1.0, dtype=np.float32)
                for i, p in enumerate(pics):
                    block = sim[index[p]]
                    for j, q in enumerate(pics):
                        if i != j:
                            best[i, j] = block[:, index[q]].max()
                near = np.argpartition(-best, k, axis=1)[:, :k]
                row["picture_best_window_knn5_purity"] = float((pic_label[near] == pic_label[:, None]).mean())
            report[n] = row
            print(f"зерно {seed} ур. {n}: точек {count}, плотность {row['density']:.3f}, чистота код {row['code_purity']:.3f} / карта {row['map_purity']:.3f}, М1u {row['m1u']:.3f}", flush=True)
        out[seed] = report
    json.dump(out, open(f"{folder}/codespace.json", "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()
