"""Эталон: те же коды уровня раскладываются t-SNE / UMAP (без меток) и сажаются на ту же сетку — М1u и честная чистота
против нашей раскладки. Только процессор. python reference.py <папка> <зёрна> <уровень>"""
import json, sys, warnings
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
warnings.filterwarnings("ignore")
EIGHT = np.ones((3, 3), bool)

def m1_by_digit(cells, labels, height, width):
    grid = np.full((int(height), int(width)), -1, dtype=np.int64)
    grid[cells[:, 0], cells[:, 1]] = labels
    empty = grid < 0
    out, count = [], 0
    for d in range(10):
        mask = grid == d
        joined = mask | (empty & ndimage.binary_dilation(mask, structure=EIGHT))
        parts, _ = ndimage.label(joined, structure=EIGHT)
        sizes = np.bincount(parts[mask])[1:]
        out.append(float(sizes.max() / mask.sum())); count += int((sizes >= 5).sum())
    return out, count

def purity(cells, labels, picture, k=10):
    tree = cKDTree(cells.astype(np.float64))
    _, near = tree.query(cells.astype(np.float64), k=min(k * 6, cells.shape[0]))
    pure = np.zeros(cells.shape[0])
    for q in range(cells.shape[0]):
        cand = near[q]; cand = cand[picture[cand] != picture[q]][:k]
        pure[q] = (labels[cand] == labels[q]).mean() if cand.size else 0.0
    return float(pure.mean())

def to_grid(xy, height, width):
    xy = (xy - xy.min(axis=0)) / (xy.max(axis=0) - xy.min(axis=0) + 1e-9)
    xy = xy * np.array([height - 1, width - 1])
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    cell = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(np.float64)
    cost = ((xy[:, None, :] - cell[None, :, :]) ** 2).sum(axis=2)
    rows, cols = linear_sum_assignment(cost)
    out = np.zeros((xy.shape[0], 2), dtype=np.int64); out[rows] = cell[cols].astype(np.int64)
    return out

folder, seeds, level = sys.argv[1], [int(v) for v in sys.argv[2].split(",")], int(sys.argv[3])
from sklearn.manifold import TSNE
import umap
for seed in seeds:
    z = np.load(f"{folder}/dump-seed{seed}.npz")
    bits = np.unpackbits(z[f"L{level}_bits"], axis=1)[:, : int(z[f"L{level}_length"])].astype(np.float32)
    labels, picture, cells = z[f"L{level}_labels"], z[f"L{level}_picture"], z[f"L{level}_cells"]
    height, width = [int(v) for v in z[f"L{level}_size"]]
    x = bits / np.sqrt(bits.sum(axis=1, keepdims=True))
    dist = np.clip(1.0 - x @ x.T, 0.0, None); np.fill_diagonal(dist, 0.0)
    m1, patches = m1_by_digit(cells, labels, height, width)
    print(f"зерно {seed} ур. {level}: НАША раскладка М1u {np.mean(m1):.3f} честная {purity(cells, labels, picture):.3f} пятен {patches} | " + " ".join(f"{v:.2f}" for v in m1), flush=True)
    import os
    for name in os.environ.get("REF_METHODS", "tsne30,tsne50,umap15,umap40").split(","):
        for state in [int(v) for v in os.environ.get("REF_STATES", "0,1,2").split(",")]:
            if name.startswith("tsne"):
                xy = TSNE(n_components=2, metric="precomputed", perplexity=int(name[4:]), init="random", random_state=state).fit_transform(dist)
            else:
                xy = umap.UMAP(n_components=2, metric="precomputed", n_neighbors=int(name[4:]), min_dist=0.3, random_state=state).fit_transform(dist)
            got = to_grid(xy, height, width)
            m1, patches = m1_by_digit(got, labels, height, width)
            row = {"seed": seed, "level": level, "method": name, "state": state, "m1u": float(np.mean(m1)), "m1": [round(v, 3) for v in m1], "honest": purity(got, labels, picture), "patches": patches}
            open(f"{folder}/reference.jsonl", "a", encoding="utf-8").write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  {name} #{state}: М1u {row['m1u']:.3f} честная {row['honest']:.3f} пятен {patches} | " + " ".join(f"{v:.2f}" for v in m1), flush=True)
