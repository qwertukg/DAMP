# -*- coding: utf-8 -*-
import json, math, statistics

rows = [json.loads(l) for l in open("tools/tuning/stand/tune2_log.jsonl", encoding="utf-8") if l.strip()]

# уровни=5 на 250: медианы points[0], суммы, верхняя карта
d5 = [d for d in rows if d["levels"] == 5]
print(f"levels=5 (n={len(d5)}): count={sorted({d['count'] for d in d5})}, "
      f"points[0] медиана {statistics.median([d['points'][0] for d in d5]):.0f}, "
      f"сумма точек медиана {statistics.median([sum(d['points']) for d in d5]):.0f}, "
      f"верхняя карта медиана {statistics.median([d['points'][-1] for d in d5]):.0f}")

# внутри count=250, levels=3: время против points[0] — половины по точкам
ds = [d for d in rows if d["count"] == 250 and d["levels"] == 3]
ds.sort(key=lambda d: d["points"][0])
h = len(ds)//2
lo, hi = ds[:h], ds[h:]
print(f"\ncount=250 levels=3 (n={len(ds)}): время vs точки")
print(f"  нижняя половина по points[0]: медиана points[0]={statistics.median([d['points'][0] for d in lo]):.0f}, "
      f"время {statistics.median([d['train_seconds'] for d in lo]):.1f}с")
print(f"  верхняя половина по points[0]: медиана points[0]={statistics.median([d['points'][0] for d in hi]):.0f}, "
      f"время {statistics.median([d['train_seconds'] for d in hi]):.1f}с")
r_p = statistics.median([d['points'][0] for d in hi]) / statistics.median([d['points'][0] for d in lo])
r_t = statistics.median([d['train_seconds'] for d in hi]) / statistics.median([d['train_seconds'] for d in lo])
print(f"  -> точки x{r_p:.2f}, время x{r_t:.2f}")

# то же для count=100 levels=3
ds1 = [d for d in rows if d["count"] == 100 and d["levels"] == 3]
ds1.sort(key=lambda d: d["points"][0])
h1 = len(ds1)//2
lo1, hi1 = ds1[:h1], ds1[h1:]
r_p1 = statistics.median([d['points'][0] for d in hi1]) / statistics.median([d['points'][0] for d in lo1])
r_t1 = statistics.median([d['train_seconds'] for d in hi1]) / statistics.median([d['train_seconds'] for d in lo1])
print(f"count=100 levels=3 (n={len(ds1)}): точки x{r_p1:.2f}, время x{r_t1:.2f}")

# итоговая оценка 500/250
n1, n2 = 250.0, 500.0
g_nlogn = (n2*math.log(n2))/(n1*math.log(n1))
g_n2 = (n2/n1)**2
print(f"\n250->500: n·log n -> x{g_nlogn:.2f}, n² -> x{g_n2:.2f}, линейно -> x2.00")
print(f"эмпирический показатель роста точек (100->250): {math.log(2.88)/math.log(2.5):.2f} (n·log n ~ 1.13 в этом диапазоне... log-фактор {math.log(2.5)/math.log(2.5):.2f})")
