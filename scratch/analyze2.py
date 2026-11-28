# -*- coding: utf-8 -*-
import json, math, statistics
from collections import defaultdict, Counter

rows = [json.loads(l) for l in open("tools/tuning/stand/tune2_log.jsonl", encoding="utf-8") if l.strip()]

print("count x levels:")
tab = Counter((d["count"], d["levels"]) for d in rows)
for (c, lv), n in sorted(tab.items()):
    print(f"  count={c} levels={lv}: {n}")

# одинаковые changes на разных count?
by_cfg = defaultdict(list)
for d in rows:
    key = json.dumps(d["changes"], sort_keys=True, ensure_ascii=False)
    by_cfg[key].append(d)
print("\nконфиги, встречающиеся на разных count:")
found = False
for key, ds in by_cfg.items():
    cs = sorted({d["count"] for d in ds})
    if len(cs) > 1:
        found = True
        print(" count-ы:", cs, "| прогонов:", len(ds), "| label:", ds[0]["label"][:60])
if not found:
    print("  нет")

# точки по уровням: медианы для count=100 и count=250 (все прогоны)
print("\nмедианы точек по уровням (все прогоны):")
for c in (100, 250):
    ds = [d for d in rows if d["count"] == c]
    maxlv = max(len(d["points"]) for d in ds)
    meds = []
    for i in range(maxlv):
        vals = [d["points"][i] for d in ds if len(d["points"]) > i]
        meds.append(statistics.median(vals) if vals else None)
    print(f"  count={c} (n={len(ds)}):", " ".join("--" if m is None else f"{m:.0f}" for m in meds))

# нижняя карта (points[0]) у count=100: разброс
b100 = [d["points"][0] for d in rows if d["count"] == 100]
b250 = [d["points"][0] for d in rows if d["count"] == 250]
print(f"\npoints[0]: count=100 медиана {statistics.median(b100):.0f} (min {min(b0 if False else b100):.0f} max {max(b100):.0f}), "
      f"count=250 медиана {statistics.median(b250):.0f} (min {min(b250):.0f} max {max(b250):.0f})")
r = statistics.median(b250) / statistics.median(b100)
print(f"отношение точек нижней карты 250/100: x{r:.2f}; n·log n -> x{(250*math.log2(250))/(100*math.log2(100)):.2f}; n^2 -> x{6.25:.2f}")

# время: медианы по count (все прогоны) и по уровням
print("\nмедиана train_seconds по count (все прогоны):")
for c in (100, 250):
    ds = [d for d in rows if d["count"] == c]
    print(f"  count={c}: {statistics.median([d['train_seconds'] for d in ds]):.1f} с (n={len(ds)})")
print("медиана train_seconds по (count, levels):")
for (c, lv), n in sorted(tab.items()):
    ds = [d for d in rows if d["count"] == c and d["levels"] == lv]
    print(f"  count={c} levels={lv}: {statistics.median([d['train_seconds'] for d in ds]):.1f} с (n={n})")
