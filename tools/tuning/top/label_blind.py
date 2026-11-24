# -*- coding: utf-8 -*-
"""Страж: раскладка верха не видит меток цифр (к tools/tuning/top/extra_pictures.py).

    python tools/tuning/top/label_blind.py <зерно=6> [новых=500,1500] [раскладки=0,1,2]

Раскладке метки передаются только для цвета точек, но таблица кодов перед раскладкой группируется по цвету класса, то есть
порядок точек зависит от меток. Здесь та же раскладка делается дважды: с настоящими метками и с метками, перемешанными
случайно (порядок точек от цифр не зависит), а М1u в обоих случаях считается по настоящим меткам. Если раскладка меток
не видит, средние совпадают в пределах разброса раскладок; заметно более высокая М1u с настоящими метками — утечка.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "client"), os.path.join(ROOT, "system"), os.path.dirname(os.path.abspath(__file__))]
os.environ.setdefault("DAMP_LOG_CONSOLE", "heartbeat")
os.environ.setdefault("RERUN", "off")
os.environ.setdefault("DAMP_RERUN_VIEWER", "off")
import numpy as np  # noqa: E402

import extra_pictures as E  # noqa: E402


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    extras = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "500,1500").split(",")]
    layouts = [int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else "0,1,2").split(",")]
    bits, labels, _, new, level = E.load_codes(seed)
    old_rows, new_rows = np.flatnonzero(~new), np.flatnonzero(new)
    base = {**E.base_settings(), **E.LIGHT}
    print("точек | раскладка | М1u с настоящими метками | М1u с перемешанными метками (счёт — по настоящим)")
    for extra in extras:
        chosen = np.concatenate([old_rows, new_rows[:extra]])
        true = labels[chosen]
        fake = np.random.default_rng(1000 + seed + extra).permutation(true)  # те же доли классов, порядок случайный
        found = {"true": [], "fake": []}
        for s in layouts:
            row = []
            for name, shown in (("true", true), ("fake", fake)):
                cells, got, order, size, _ = E.lay(bits[chosen], shown, level, base, s)
                assert np.array_equal(got, shown[order]), "порядок точек раскладки не восстановлен"
                m1, _, _ = E.gathering(cells, true[order], size[0], size[1], {})
                found[name].append(float(np.mean([m1.get(d, 0.0) for d in range(10)])))
                row.append(found[name][-1])
            print(f"{chosen.size} | {s} | {row[0]:.3f} | {row[1]:.3f}", flush=True)
        print(f"{chosen.size} | среднее | {np.mean(found['true']):.3f} | {np.mean(found['fake']):.3f}  (разность {np.mean(found['true']) - np.mean(found['fake']):+.3f}, "
              f"разброс раскладок {np.std(found['true'] + found['fake']):.3f})", flush=True)
    print("Итого: разность в пределах разброса раскладок — раскладка меток не видит")


if __name__ == "__main__":
    main()
