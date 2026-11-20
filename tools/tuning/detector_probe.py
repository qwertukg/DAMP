# -*- coding: utf-8 -*-
"""Разбор детекторов обученного цикла (§6.4, §6.5, §7.1.10, §7.1.12): из чего на самом деле сделан эмбеддинг.

    detector_probe.py <файл с изменениями JSON (как у стенда)> <обучающих цифр> <выход JSON>

Цикл берётся из кеша шагов (настройки как у прогона стенда). По каждой карте уровня 1+:
  1. детекторы по слоям: число, порог λ_d, радиус r_d (медиана, 10 / 90 %), радиус активации r_a;
  2. кластер, который детектор «должен обхватывать» (§6.4.1–§6.4.3): активация карты кодом точки у центра детектора
     в круге r_a при λ_d, DBSCAN, кластер с центром; его точек и доля кластера внутри круга детектора;
  3. чистота детектора по цифре: доля самой частой цифры среди точек внутри круга;
  4. эмбеддинг (§7.1.12): из каких слоёв σ сильнейших детекторов окна, сколько из них взяты при равенстве уровней;
  5. срабатывания по слоям: сколько детекторов слоя срабатывает на окно, средний уровень.
"""
import json
import os
import sys

sys.path[:0] = [os.path.join(os.getcwd(), "client"), os.path.join(os.getcwd(), "system")]

import numpy as np  # noqa: E402

from damp_api import Damp  # noqa: E402

CAPTURED: dict = {}


def capture_embed() -> None:
    from damp.cycle.step import CycleStep

    embed = CycleStep.embed

    def remembered(self, firing, detectors, keyholes):  # noqa: ANN001, ANN202
        CAPTURED[int(keyholes.number)] = {"firing": firing, "detectors": detectors, "keyholes": keyholes}
        return embed(self, firing, detectors, keyholes)

    CycleStep.embed = remembered


def quantiles(values) -> dict:  # noqa: ANN001
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {}
    return {"p10": round(float(np.quantile(values, 0.1)), 2), "median": round(float(np.median(values)), 2),
            "p90": round(float(np.quantile(values, 0.9)), 2), "mean": round(float(values.mean()), 2)}


def main() -> None:
    changes = json.load(open(sys.argv[1], encoding="utf-8"))["changes"]
    count = int(sys.argv[2])
    out = sys.argv[3]
    capture_embed()
    damp = Damp(changes={**changes, "DASHBOARD_FILE_PREFIX": "data/tune/ladder"})
    ladder = damp.cycle.trained(count)
    from damp.activation.activation import DbscanClusterer
    from damp.activation.set_embedding import DetectorEmbeddings

    report = {"changes": changes, "levels": {}}
    s = damp.settings
    sigma = int(s["ACTIVATION_SATURATION_LIMIT"])
    specs = s["ACTIVATION_DETECTOR_LAYER_SPECS"]
    eps, min_samples = float(s["ACTIVATION_DBSCAN_EPS"]), int(s["ACTIVATION_DBSCAN_MIN_SAMPLES"])
    normalisation = s["LADDER_LEVEL_NORMALISATION"]
    clusterer = DbscanClusterer(eps=eps, min_samples=min_samples)
    rng = np.random.default_rng(0)
    for level in ladder._ladder.levels:
        number = int(level.number)
        layout = level.memory.layout
        detectors = level.detectors
        found = list(detectors.detectors)
        layer = np.asarray(detectors.layer_index, dtype=np.int64)
        layers = sorted(set(layer.tolist()))
        cells = np.asarray(layout.cells_array(), dtype=np.int64).reshape(-1, 2)
        labels = np.asarray([int(v) for v in layout.code_table.labels()], dtype=np.int64)
        grid = np.full((int(layout.height), int(layout.width)), -1, dtype=np.int64)
        grid[cells[:, 0], cells[:, 1]] = np.arange(cells.shape[0])
        extent = max(1, min(int(layout.width), int(layout.height)))
        energies = np.asarray(level.memory.energies, dtype=np.float64)
        entry = {"points": int(cells.shape[0]), "grid": [int(layout.height), int(layout.width)], "layers": []}
        for k in layers:
            mine = [d for d, l in zip(found, layer) if l == k]
            radius = np.asarray([d.radius for d in mine])
            spec = specs[min(len(specs) - 1, k * len(specs) // len(layers))]
            activation_radius = max(1, int(round(extent * float(spec[1]))))
            # Кластер, который детектор должен обхватывать: активация кодом точки у центра в круге r_a при λ_d.
            sample = rng.choice(len(mine), size=min(60, len(mine)), replace=False) if mine else []
            cluster_sizes, inside_share, purity, cluster_radius = [], [], [], []
            for i in sample:
                d = mine[int(i)]
                cy, cx = d.center
                yy, xx = int(round(cy)), int(round(cx))
                yy, xx = min(max(yy, 0), grid.shape[0] - 1), min(max(xx, 0), grid.shape[1] - 1)
                centre = grid[yy, xx]
                if centre < 0:
                    near = np.argmin(np.hypot(cells[:, 0] - cy, cells[:, 1] - cx))
                    centre = int(near)
                dist = np.hypot(cells[:, 0] - cy, cells[:, 1] - cx)
                field = np.nonzero(dist <= activation_radius)[0]
                sim = layout._pair_similarity_np(np.full(field.size, centre, dtype=np.int64), field)
                lit = field[sim >= float(d.lambda_threshold)]
                if lit.size < min_samples:
                    continue
                seed_point = lit[np.argmin(np.hypot(cells[lit, 0] - cy, cells[lit, 1] - cx))]
                mask = clusterer.cluster_mask(cells[lit, 0], cells[lit, 1], (int(cells[seed_point, 0]), int(cells[seed_point, 1])))
                members = lit[np.asarray(mask, dtype=bool)]
                if members.size == 0:
                    continue
                cluster_sizes.append(members.size)
                cluster_radius.append(float(np.sqrt(members.size / np.pi)))
                inside_share.append(float((np.hypot(cells[members, 0] - cy, cells[members, 1] - cx) <= d.radius).mean()))
                within = np.nonzero(dist <= d.radius)[0]
                if within.size:
                    purity.append(float(np.bincount(labels[within]).max() / within.size))
            entry["layers"].append({
                "layer": int(k), "detectors": len(mine), "lambda_d": round(float(mine[0].lambda_threshold), 3) if mine else None,
                "activation_radius": activation_radius, "min_points": int(spec[2]),
                "radius": quantiles(radius), "cluster_points": quantiles(cluster_sizes),
                "cluster_equal_radius": quantiles(cluster_radius), "cluster_inside_detector": quantiles(inside_share),
                "digit_purity": quantiles(purity),
                "map_share_covered": round(float(np.mean([np.any(np.hypot(cells[p, 0] - np.asarray([d.center[0] for d in mine]),
                                                                          cells[p, 1] - np.asarray([d.center[1] for d in mine]))
                                                                 <= np.asarray([d.radius for d in mine]))
                                                          for p in rng.choice(cells.shape[0], 400, replace=False)])), 3) if mine else 0.0,
            })
        # Эмбеддинг: из каких слоёв σ сильнейших (на окнах обучающих картинок, после нормировки, как в цикле).
        got = CAPTURED.get(number)
        if got is not None:
            firing = got["firing"]
            rows, columns, levels = np.asarray(firing.rows), np.asarray(firing.columns), np.asarray(firing.levels, dtype=np.float64)
            built_layer = np.asarray(got["detectors"].layer_index, dtype=np.int64)
            kept_rows, kept_columns = DetectorEmbeddings(detectors=int(got["detectors"].built), saturation_limit=sigma,
                                                         target_density=0.1, min_length=64, max_length=8192, seed=0,
                                                         name="probe").strongest(rows, columns, levels)
            sets = max(1, int(firing.sets))
            fired_layer = np.bincount(built_layer[columns], minlength=len(layers)) / sets
            kept_layer = np.bincount(built_layer[kept_columns], minlength=len(layers)) / sets
            mean_level = [round(float(levels[built_layer[columns] == k].mean()), 3) if np.any(built_layer[columns] == k) else 0.0
                          for k in range(len(layers))]
            # Сколько взятых — с уровнем ровно σ-го (равенство, отбор по номеру детектора: широкие слои раньше).
            at_one = float((levels >= 1.0 - 1e-9).mean())
            entry["embedding"] = {"normalisation": normalisation, "sigma": sigma,
                                  "fired_per_window_by_layer": [round(float(v), 2) for v in fired_layer],
                                  "kept_per_window_by_layer": [round(float(v), 2) for v in kept_layer],
                                  "mean_level_by_layer": mean_level, "share_of_firings_at_level_1": round(at_one, 3)}
        report["levels"][number] = entry
        print(json.dumps({number: entry}, ensure_ascii=False))
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
