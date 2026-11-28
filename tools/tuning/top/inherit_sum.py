# -*- coding: utf-8 -*-
"""Итог проверки v19.10: старт с карты ниже против случайного старта (опора стенда верха) на тех же зёрнах кодов."""
import collections
import json
import os

import numpy as np

FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "top-stand")
rows = [json.loads(line) for line in open(os.path.join(FOLDER, "inherit.jsonl"), encoding="utf-8")]
base = collections.defaultdict(list)
for line in open(os.path.join(FOLDER, "layout-series.jsonl"), encoding="utf-8"):
    r = json.loads(line)
    if r["variant"] == "base" and r["level"] == 5:
        base[r["seed"]].append(r)
found = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    found[r["variant"]][r["seed"]].append(r)
print(f"{'вариант':46s} зёрен  М1u    Δ к случайному старту (t по зёрнам)  честная (Δ)  пятен | М1 по цифрам 0…9")
for name, by_seed in found.items():
    seeds = sorted(s for s in by_seed if s in base)
    m1u = np.array([np.mean([r["m1u"] for r in by_seed[s]]) for s in seeds])
    ref = np.array([np.mean([r["m1u"] for r in base[s]]) for s in seeds])
    honest = np.mean([r["honest"] for s in seeds for r in by_seed[s]])
    honest_ref = np.mean([r["honest"] for s in seeds for r in base[s]])
    d = m1u - ref
    t = d.mean() / (d.std(ddof=1) / np.sqrt(d.size) + 1e-9) if d.size > 2 else 0.0
    digits = np.mean([r["m1"] for s in seeds for r in by_seed[s]], axis=0)
    patches = np.mean([r["patches"] for s in seeds for r in by_seed[s]])
    print(f"{name:46s} {len(seeds):3d}   {m1u.mean():.3f}  {d.mean():+.3f} ({t:+.1f})                        {honest:.3f} ({honest - honest_ref:+.3f})  {patches:5.1f} | "
          + " ".join(f"{x:.2f}" for x in digits))
seeds = sorted(base)
print(f"{'случайный старт (опора стенда верха)':46s} {len(seeds):3d}   {np.mean([np.mean([r['m1u'] for r in base[s]]) for s in seeds]):.3f}"
      f"                                       {np.mean([r['honest'] for s in seeds for r in base[s]]):.3f}")
