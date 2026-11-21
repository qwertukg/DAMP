# -*- coding: utf-8 -*-
import json, math
from collections import defaultdict

rows = [json.loads(l) for l in open("tools/tuning/stand/tune2_log.jsonl", encoding="utf-8") if l.strip()]

by_cfg = defaultdict(list)
for d in rows:
    key = json.dumps(d["changes"], sort_keys=True, ensure_ascii=False)
    by_cfg[key].append(d)

print("пары одинаковых конфигов на разных count:")
for key, ds in by_cfg.items():
    cs = sorted({d["count"] for d in ds})
    if len(cs) < 2:
        continue
    print("=" * 70)
    print("label:", ds[0]["label"])
    for d in sorted(ds, key=lambda x: x["count"]):
        print(f"  run {d['run']} count={d['count']} levels={d['levels']} "
              f"train={d['train_seconds']:.1f}с points={d['points']} "
              f"test_count={d.get('test_count')} date={d.get('date','')[:10]}")
    if len(ds) == 2:
        a, b = sorted(ds, key=lambda x: x["count"])
        r_t = b["train_seconds"] / a["train_seconds"]
        r_p0 = b["points"][0] / a["points"][0]
        n = b["count"] / a["count"]
        print(f"  -> время x{r_t:.2f}, points[0] x{r_p0:.2f} при n x{n:.2f} "
              f"(n·log n x{(n*math.log2(n))/(1*math.log2(1)) if False else (b['count']*math.log2(b['count']))/(a['count']*math.log2(a['count'])):.2f}, n² x{n**2:.2f})")

# сводка: все пары (count=100 vs 250) по времени и точкам
print()
print("для справки: медианы по count при levels==3 (самая частая конфигурация):")
for c in (100, 250):
    ds = [d for d in rows if d["count"] == c and d["levels"] == 3]
    import statistics
    print(f"  count={c}: время {statistics.median([d['train_seconds'] for d in ds]):.1f}с, "
          f"points[0] {statistics.median([d['points'][0] for d in ds]):.0f} (n={len(ds)})")
