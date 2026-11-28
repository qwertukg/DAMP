# -*- coding: utf-8 -*-
"""Разбор журнала прогонов стенда tune2_log.jsonl.

По прогонам с levels == 5: медиана train_seconds по каждому count,
медиана точек верхней карты (points[-1]), и оценка множителя времени
для 500 цифр против 250.
"""
import json
import math
import statistics
from collections import defaultdict

LOG = "tools/tuning/stand/tune2_log.jsonl"

rows = []
total = 0
with open(LOG, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        total += 1
        d = json.loads(line)
        if d.get("levels") != 5:
            continue
        rows.append(d)

print(f"всего строк: {total}, с levels==5: {len(rows)}")

by_count = defaultdict(list)
for d in rows:
    by_count[d["count"]].append(d)

print()
print("count | прогонов | медиана train_seconds | min..max | медиана точек верхней карты")
for count in sorted(by_count):
    rs = by_count[count]
    secs = [d["train_seconds"] for d in rs]
    tops = [d["points"][-1] for d in rs]
    print(
        f"{count:>5} | {len(rs):>8} | {statistics.median(secs):>10.1f} с "
        f"| {min(secs):7.1f}..{max(secs):7.1f} | {statistics.median(tops):>8.0f}"
    )

counts = sorted(by_count)
if len(counts) >= 2:
    c1, c2 = counts[0], counts[-1]
    t1 = statistics.median([d["train_seconds"] for d in by_count[c1]])
    t2 = statistics.median([d["train_seconds"] for d in by_count[c2]])
    p1 = statistics.median([d["points"][-1] for d in by_count[c1]])
    p2 = statistics.median([d["points"][-1] for d in by_count[c2]])
    n1, n2 = float(c1), float(c2)
    lin = n2 / n1
    g_nlogn = (n2 * math.log2(n2)) / (n1 * math.log2(n1))
    g_n2 = (n2 / n1) ** 2
    print()
    print(f"эмпирика {c1} -> {c2}: время x{t2/t1:.2f}, точки верхней карты x{p2/p1:.2f}, "
          f"n·log n предсказал бы x{g_nlogn:.2f}, n² предсказал бы x{g_n2:.2f}")
    for name, g in (("n·log n", g_nlogn), ("n²", g_n2)):
        share = (t2 / t1 - lin) / (g - lin)
        share = max(0.0, min(1.0, share))
        mult = share * g + (1 - share) * lin
        print(f"если раскладка ~{name}: доля времени на раскладку ~{share:.0%}, "
              f"множитель 250->500 ~ x{mult:.1f}")

if len(counts) > 2:
    print()
    print("по парам соседних count:")
    for a, b in zip(counts, counts[1:]):
        ta = statistics.median([d["train_seconds"] for d in by_count[a]])
        tb = statistics.median([d["train_seconds"] for d in by_count[b]])
        pa = statistics.median([d["points"][-1] for d in by_count[a]])
        pb = statistics.median([d["points"][-1] for d in by_count[b]])
        print(f"  {a}->{b}: время x{tb/ta:.2f}, точки x{pb/pa:.2f}")
