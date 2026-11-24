"""Какой вес пары согласуется с целыми классами: энергия Σ w_ij·exp(−d²/2σ²) у нашей карты и у эталонных (t-SNE, UMAP на ту
же сетку). Вес хорош, если даёт эталонным картам энергию выше нашей (отношение > 1) и ранжирует карты как М1u."""
import sys, warnings, json
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
from sklearn.manifold import TSNE
import umap
EIGHT = np.ones((3, 3), bool)
def m1u(cells, labels, height, width):
    grid = np.full((int(height), int(width)), -1, dtype=np.int64); grid[cells[:, 0], cells[:, 1]] = labels
    empty = grid < 0; out = []
    for d in range(10):
        mask = grid == d
        parts, _ = ndimage.label(mask | (empty & ndimage.binary_dilation(mask, structure=EIGHT)), structure=EIGHT)
        out.append(float(np.bincount(parts[mask])[1:].max() / mask.sum()))
    return float(np.mean(out))
def to_grid(xy, height, width):
    xy = (xy - xy.min(axis=0)) / (xy.max(axis=0) - xy.min(axis=0) + 1e-9) * np.array([height - 1, width - 1])
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    cell = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(np.float64)
    rows, cols = linear_sum_assignment(((xy[:, None, :] - cell[None, :, :]) ** 2).sum(axis=2))
    out = np.zeros((xy.shape[0], 2), dtype=np.int64); out[rows] = cell[cols].astype(np.int64)
    return out
def knn_mask(sim, k):
    near = np.argpartition(-sim, k, axis=1)[:, :k]
    m = np.zeros_like(sim, dtype=bool); m[np.arange(sim.shape[0])[:, None], near] = True
    return m
def weights(sim, lam):
    n = sim.shape[0]; out = {}
    out["сейчас: s·сигм(10(s−λ0.82))"] = sim / (1.0 + np.exp(-10.0 * (sim - lam)))
    for q in (0.82, 0.90, 0.95, 0.975, 0.99):
        t = np.quantile(sim[np.triu_indices(n, 1)], q)
        out[f"жёсткий порог квантиль {q}"] = np.where(sim >= t, sim, 0.0)
    for p in (2, 4, 8):
        out[f"степень s^{p}"] = sim ** p
    mu = sim[np.triu_indices(n, 1)].mean()
    out["центрированный (s − среднее)+"] = np.clip(sim - mu, 0.0, None)
    row_mu = (sim.sum(axis=1) / (n - 1))
    out["по строкам: (s − ср_i − ср_j + ср)+"] = np.clip(sim - row_mu[:, None] - row_mu[None, :] + mu, 0.0, None)
    for k in (10, 20, 40, 80):
        m = knn_mask(sim, k)
        out[f"взаимные {k} ближайших (и)"] = (m & m.T).astype(np.float64)
        out[f"{k} ближайших (или)"] = (m | m.T).astype(np.float64)
        snn = (m.astype(np.float32) @ m.astype(np.float32).T) / k
        out[f"общие соседи из {k}"] = snn
    # порог по строке: s ≥ квантиль своей строки (у каждой точки своя λ_i), вес s
    for q in (0.95, 0.98):
        t = np.quantile(sim, q, axis=1)
        out[f"свой порог точки, квантиль {q} (или)"] = np.where((sim >= t[:, None]) | (sim >= t[None, :]), sim, 0.0)
    return out
folder = sys.argv[1]; seeds = [int(v) for v in sys.argv[2].split(",")]
table = {}
for seed in seeds:
    z = np.load(f"{folder}/dump-seed{seed}.npz")
    bits = np.unpackbits(z["L5_bits"], axis=1)[:, : int(z["L5_length"])].astype(np.float32)
    labels, cells, lam = z["L5_labels"], z["L5_cells"], float(z["L5_lambda"])
    height, width = [int(v) for v in z["L5_size"]]
    x = bits / np.sqrt(bits.sum(axis=1, keepdims=True)); sim = (x @ x.T).astype(np.float64); np.fill_diagonal(sim, 0.0)
    dist = np.clip(1.0 - sim, 0.0, None); np.fill_diagonal(dist, 0.0)
    maps = [("наша", cells)]
    for state in (0, 1):
        maps.append((f"tsne{state}", to_grid(TSNE(n_components=2, metric="precomputed", perplexity=30, init="random", random_state=state).fit_transform(dist), height, width)))
        maps.append((f"umap{state}", to_grid(umap.UMAP(n_components=2, metric="precomputed", n_neighbors=15, min_dist=0.3, random_state=state).fit_transform(dist), height, width)))
    rng = np.random.default_rng(seed)
    maps.append(("случайная", to_grid(rng.random((labels.size, 2)), height, width)))
    quality = [m1u(c, labels, height, width) for _, c in maps]
    kernels = {}
    for sigma in (3.0, 6.0):
        kernels[sigma] = [np.exp(-((c[:, None, :] - c[None, :, :]) ** 2).sum(axis=2) / (2.0 * sigma * sigma)) for _, c in maps]
    for name, w in weights(sim, lam).items():
        np.fill_diagonal(w, 0.0)
        for sigma in (3.0, 6.0):
            e = np.array([float((w * k).sum()) for k in kernels[sigma]])
            ratio = e[1:5].mean() / e[0]            # эталонные к нашей
            rho = spearmanr(e[:5], quality[:5]).correlation
            table.setdefault((name, sigma), []).append((ratio, rho))
    print(f"зерно {seed}: М1u карт " + " ".join(f"{n} {q:.3f}" for (n, _), q in zip(maps, quality)), flush=True)
print(f"\n{'вес пары':42s} σ ядра | эталон / наша (среднее по зёрнам; > 1 — цель за целые классы) | согласие с М1u (Спирмен)")
for (name, sigma), v in table.items():
    r = np.array(v)
    print(f"{name:42s} {sigma:4.0f}   | {r[:,0].mean():.3f}  ({' '.join(f'{x:.3f}' for x in r[:,0])}) | {r[:,1].mean():+.2f}")
