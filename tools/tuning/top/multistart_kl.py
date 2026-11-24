# -*- coding: utf-8 -*-
"""Несколько гладких стартов и выбор без меток: для кодов ур. 5 шести зёрен — t-SNE с разными случайными началами, у каждого
расхождение KL (критерий без меток) и М1u после посадки на сетку. Вопрос: меньше KL — больше М1u? Системный Python, процессор."""
import json
import os
import sys
import warnings

import numpy as np
from scipy import ndimage
from scipy.stats import spearmanr
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")
sys.setrecursionlimit(100000)
sys.path.insert(0, os.path.join("tools", "tuning", "top"))
from three_maps import m1u, snap  # noqa: E402

FOLDER = os.path.join("data", "top-stand")
STATES = list(range(8))
out = open(sys.argv[1], "a", encoding="utf-8")
total = []
for seed in (1, 3, 0, 2, 4, 5):
    z = np.load(os.path.join(FOLDER, f"dump-seed{seed}.npz"))
    bits = np.unpackbits(z["L5_bits"], axis=1)[:, : int(z["L5_length"])].astype(np.float32)
    labels = z["L5_labels"]
    height, width = [int(v) for v in z["L5_size"]]
    kl, value = [], []
    for state in STATES:
        model = TSNE(n_components=2, metric="cosine", perplexity=30, init="random", random_state=state, n_jobs=2)
        xy = model.fit_transform(bits)
        kl.append(float(model.kl_divergence_))
        value.append(m1u(snap(xy, height, width), labels, height, width)[0])
        out.write(json.dumps({"seed": seed, "state": state, "kl": kl[-1], "m1u": value[-1]}) + "\n")
        out.flush()
    kl, value = np.array(kl), np.array(value)
    rho = spearmanr(kl, value).statistic
    total.append((value.mean(), value[kl.argmin()], value.max(), value.std(ddof=1), rho))
    print(f"зерно {seed}: М1u среднее {value.mean():.3f} (SD {value.std(ddof=1):.3f}), у наименьшего KL {value[kl.argmin()]:.3f}, лучший {value.max():.3f}, "
          f"Спирмен KL~М1u {rho:+.2f}; KL {kl.min():.3f}…{kl.max():.3f}", flush=True)
t = np.array(total)
print(f"Итого: среднее {t[:, 0].mean():.3f}, выбор по KL {t[:, 1].mean():.3f} (Δ {t[:, 1].mean() - t[:, 0].mean():+.3f}), лучший из 8 {t[:, 2].mean():.3f}, "
      f"SD внутри набора {t[:, 3].mean():.3f}, Спирмен {t[:, 4].mean():+.2f}")
