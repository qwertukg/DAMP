"""Выгрузка карт уровней обученного цикла опоры (из кеша): python dump.py <папка> <зерно> [JSON правок] [метка]."""
import json, os, sys, time
sys.path[:0] = [os.path.join(os.getcwd(), "client"), os.path.join(os.getcwd(), "system")]
os.environ.setdefault("DAMP_LOG_CONSOLE", "heartbeat")
os.environ.setdefault("RERUN", "off")
os.environ.setdefault("DAMP_RERUN_VIEWER", "off")  # без окна Rerun: 2026-11-24 выгрузка оставила висеть окно (ОЗУ)
import numpy as np
from damp_api import Damp

out, seed = sys.argv[1], int(sys.argv[2])
extra = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
tag = sys.argv[4] if len(sys.argv) > 4 else ""
base = json.load(open("tools/tuning/stand/base.json", encoding="utf-8"))
light = {"LAYOUT_LOG_VISUALS": False, "ENCODING_WORKERS": 4, "ACTIVATION_PARALLEL_WORKERS": 2, "LADDER_CLASS_MAP": False}
damp = Damp(changes={**base, **extra, **light, "ENCODER_RANDOM_SEED": seed, "DASHBOARD_FILE_PREFIX": "data/tune/ladder"})
t = time.time()
ladder = damp.cycle.trained(250)
run = ladder._ladder
data = {"seconds": time.time() - t, "picture_label": np.asarray(run.groups.labels, dtype=np.int64)}
if run.primary is not None:  # уровень 0: первичная карта штрихов (цвет на рисунках — угол штриха)
    layout = run.primary.memory.layout
    bits = np.asarray(layout.code_table.bits(), dtype=np.uint8)
    picture0 = np.asarray(run.primary.point_picture, dtype=np.int64)
    data["L0_bits"] = np.packbits(bits, axis=1)
    data["L0_length"] = bits.shape[1]
    data["L0_cells"] = np.asarray(layout.cells_array(), dtype=np.int64)
    data["L0_size"] = np.asarray([layout.height, layout.width])
    data["L0_lambda"] = float(layout.lambda_threshold)
    data["L0_picture"] = picture0
    data["L0_labels"] = np.asarray(run.groups.atoms.labels, dtype=np.int64)[np.maximum(picture0, 0)]
    data["L0_angle"] = np.asarray(run.primary.point_angle, dtype=np.float64)
    with ladder._session.active():
        from damp import settings as live
        data["L0_period"] = (float(live.MNIST_ANGLE_MAX) - float(live.MNIST_ANGLE_MIN)) / (2.0 if run.primary.axial() else 1.0)
    print(f"уровень 0: точек {bits.shape[0]}, длина кода {bits.shape[1]}, карта {layout.height}×{layout.width}", flush=True)
for level in run.levels:
    n, layout = int(level.number), level.memory.layout
    bits = np.asarray(layout.code_table.bits(), dtype=np.uint8)
    data[f"L{n}_bits"] = np.packbits(bits, axis=1)
    data[f"L{n}_length"] = bits.shape[1]
    data[f"L{n}_labels"] = np.asarray([int(v) for v in layout.code_table.labels()], dtype=np.int64)
    data[f"L{n}_cells"] = np.asarray(layout.cells_array(), dtype=np.int64)
    data[f"L{n}_size"] = np.asarray([layout.height, layout.width])
    data[f"L{n}_lambda"] = float(layout.lambda_threshold)
    data[f"L{n}_picture"] = np.asarray(level.point_picture, dtype=np.int64)
    data[f"L{n}_centre"] = np.asarray(level.point_centre, dtype=np.float64)
    data[f"L{n}_extent"] = np.asarray(level.point_extent, dtype=np.float64)
    data[f"L{n}_rows"] = np.asarray(level.memory.rows, dtype=np.int64)
    print(f"уровень {n}: точек {bits.shape[0]}, длина кода {bits.shape[1]}, карта {layout.height}×{layout.width}, λ {float(layout.lambda_threshold):.3f}", flush=True)
    if n >= len(run.levels) - 1:
        try:
            # Срабатывания детекторов карты уровня на коды её же точек (эмбеддинг точки, §6.5–6.6)
            from damp.activation.details import DetailCoder
            with ladder._session.active():
                firing, _ = level.detectors.fire_and_light(np.packbits(bits, axis=1))
                if level.scale is not None:
                    firing = firing.scaled(level.scale)
                centre, radius = DetailCoder.geometry(level.detectors)
            data[f"L{n}_fire_row"] = np.asarray(firing.rows, dtype=np.int64)
            data[f"L{n}_fire_det"] = np.asarray(firing.columns, dtype=np.int64)
            data[f"L{n}_fire_level"] = np.asarray(firing.levels, dtype=np.float32)
            data[f"L{n}_det_layer"] = np.asarray(level.detectors.layer_index, dtype=np.int64)
            data[f"L{n}_det_centre"] = centre
            data[f"L{n}_det_radius"] = radius
            print(f"  детекторов {int(level.detectors.built)}, срабатываний на точку {firing.rows.size / bits.shape[0]:.1f}", flush=True)
        except Exception as error:  # noqa: BLE001
            print("срабатывания не выгружены:", repr(error), flush=True)
np.savez_compressed(os.path.join(out, f"dump-seed{seed}{tag}.npz"), **data)
print(f"зерно {seed}: {time.time() - t:.0f} с")
