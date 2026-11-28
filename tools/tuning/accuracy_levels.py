# -*- coding: utf-8 -*-
"""Точность распознавания на опоре стенда: линейный читатель (§9.3), ответ памяти по всем уровням и голос каждого уровня.

    python tools/tuning/accuracy_levels.py [зёрна=6] [проверочных=1000] [обучающих=500]

Цикл учится настройками опоры (tools/tuning/stand/base.json, шаги — из кеша прогона опоры на зерне), затем проверочные
картинки MNIST t10k (первые N) идут через обученные уровни тем же путём, что рисунок дашборда (TrainedLadder.see_many).
По каждой картинке: ответ читателя, ответ памяти (цифра по среднему долей голосов уровней и согласие уровней,
§2.2.5, §7.1.13.3) и победитель голосов каждого уровня. Итог дописывается в data/tune/accuracy-levels.jsonl.
Стенд на это время не должен считать прогон (видеокарта).
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "client"), os.path.join(ROOT, "system")]
os.environ["RERUN"] = "off"
os.environ["DAMP_RERUN_VIEWER"] = "off"
os.environ.setdefault("DAMP_LOG_CONSOLE", "heartbeat")
import numpy as np  # noqa: E402

LIGHT = {"LAYOUT_LOG_VISUALS": False, "ENCODING_WORKERS": 4, "ACTIVATION_PARALLEL_WORKERS": 2, "LADDER_CLASS_MAP": False}
OUT = os.path.join(ROOT, "data", "tune", "accuracy-levels.jsonl")


def measure(seed: int, tests: int, count: int) -> dict:
    from damp_api import Damp
    from damp.cycle.trained import TrainedLadder as SystemLadder

    base = json.load(open(os.path.join(ROOT, "tools", "tuning", "stand", "base.json"), encoding="utf-8-sig"))
    damp = Damp(changes={**base, **LIGHT, "ENCODER_RANDOM_SEED": seed, "DASHBOARD_FILE_PREFIX": "data/tune/ladder"})
    started = time.time()
    ladder = damp.cycle.trained(count)
    trained = time.time() - started
    levels = [int(level.number) for level in ladder._ladder.levels]
    points = [int(level.memory.points) for level in ladder._ladder.levels]
    digits = damp.stimuli.digits(train=1, test=tests, train_views=())
    first, last = digits.groups["test"]
    atoms = digits._groups.atoms
    pictures = [digits.picture(i) for i in range(first, last)]
    started = time.time()
    with ladder._session.active():
        sights = ladder._ladder.see_many([np.asarray(picture.image, dtype=np.uint8) for picture in pictures],
                                         sticks=[SystemLadder.atom_sticks(atoms, i) for i in range(first, last)])
    seconds = time.time() - started
    truth = np.asarray([int(picture.digit) for picture in pictures], dtype=np.int64)
    reader = np.asarray([int(sight.answer) for sight in sights], dtype=np.int64)
    memory = np.asarray([int(sight.verdict.digit) for sight in sights], dtype=np.int64)
    kind = np.asarray([str(sight.verdict.kind) for sight in sights])
    by_level = np.full((len(sights), len(levels)), -1, dtype=np.int64)
    for row, sight in enumerate(sights):
        for votes in sight.votes:
            by_level[row, levels.index(int(votes.number))] = int(votes.digit)
    total = int(truth.shape[0])
    found = {
        "seed": seed, "count": count, "tests": total, "levels": levels, "points": points,
        "train_seconds": round(trained, 1), "see_ms": round(1000.0 * seconds / max(1, total), 1),
        "reader": float((reader == truth).mean()),
        "memory": float((memory == truth).mean()), "memory_answered": float((memory >= 0).mean()),
        "level": [float((by_level[:, n] == truth).mean()) for n in range(len(levels))],
        "level_answered": [float((by_level[:, n] >= 0).mean()) for n in range(len(levels))],
        "kinds": {name: {"share": float((kind == name).mean()),
                         "right": float((memory[kind == name] == truth[kind == name]).mean()) if (kind == name).any() else None,
                         "reader_right": float((reader[kind == name] == truth[kind == name]).mean()) if (kind == name).any() else None}
                  for name in sorted(set(kind.tolist()))},
        "both_agree": float((reader == memory).mean()),
        "both_agree_right": float((reader[reader == memory] == truth[reader == memory]).mean()) if (reader == memory).any() else None,
        "either_right": float(((reader == truth) | (memory == truth)).mean()),
        "reader_by_digit": [float((reader[truth == d] == d).mean()) if (truth == d).any() else None for d in range(10)],
        "memory_by_digit": [float((memory[truth == d] == d).mean()) if (truth == d).any() else None for d in range(10)],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(found, ensure_ascii=False) + "\n")
    return found


def show(found: dict) -> None:
    print(f"=== зерно {found['seed']}: обучающих {found['count']}, проверочных {found['tests']}; точек карт {found['points']}; "
          f"обучение {found['train_seconds']} с, картинка {found['see_ms']} мс", flush=True)
    print(f"  линейный читатель: {found['reader']:.3f}")
    print(f"  память, ответ по всем уровням: {found['memory']:.3f} (ответила на {found['memory_answered']:.3f})")
    for number, right, answered in zip(found["levels"], found["level"], found["level_answered"]):
        print(f"  голос уровня {number}: {right:.3f} (ответил на {answered:.3f})")
    for name, value in found["kinds"].items():
        right = "—" if value["right"] is None else f"{value['right']:.3f}"
        print(f"  согласие уровней «{name}»: доля картинок {value['share']:.3f}, память права {right}, читатель прав {value['reader_right']:.3f}")
    print(f"  читатель и память назвали одно: {found['both_agree']:.3f} картинок, из них верно {found['both_agree_right']:.3f}; прав хотя бы один: {found['either_right']:.3f}")
    print("  читатель по цифрам:", " ".join("—" if v is None else f"{v:.2f}" for v in found["reader_by_digit"]))
    print("  память по цифрам:  ", " ".join("—" if v is None else f"{v:.2f}" for v in found["memory_by_digit"]), flush=True)


if __name__ == "__main__":
    seeds = [int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "6").split(",")]
    tests = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    for seed in seeds:
        show(measure(seed, tests, count))
