# -*- coding: utf-8 -*-
import json
from collections import defaultdict

rows = [json.loads(l) for l in open("tools/tuning/stand/tune2_log.jsonl", encoding="utf-8") if l.strip()]

# все count=100: run, дата, levels, время
print("count=100 прогоны:")
for d in rows:
    if d["count"] == 100:
        print(f"  run {d['run']} {d.get('date','')[:10]} levels={d['levels']} train={d['train_seconds']:7.1f}с x={d.get('x')} image={d.get('image')} test={d.get('test_count')}")

# даты и x/image по count=250 (уникальные)
print("\ncount=250: даты и x/image:")
seen = defaultdict(list)
for d in rows:
    if d["count"] == 250:
        seen[(d.get('date','')[:10], d.get('x'), d.get('image'))].append(d['run'])
for k, v in sorted(seen.items()):
    print(f"  {k}: {len(v)} прогонов, runs {min(v)}..{max(v)}")

# полная запись одной levels=5 для контекста
d5 = [d for d in rows if d["levels"] == 5][0]
print("\nпример levels=5 записи (run", d5["run"], "):")
for k in ("run","label","count","levels","train_seconds","points","test_count","date","x","image","funnel","merge"):
    print(f"  {k}: {d5.get(k)}")
