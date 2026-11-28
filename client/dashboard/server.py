"""Сервер живого дашборда: страница с холстом и путь рисунка через обученный цикл (только через damp_api).

    python client/run.py dashboard                 # учит цикл (шаги — из кеша), открывает http://127.0.0.1:8765/

Запросы страницы:

| Запрос | Что отдаёт |
|---|---|
| `GET /` | страница `static/index.html` |
| `GET /static/<файл>` | её стили и код (только файлы папки `static/`) |
| `GET /api/model` | один раз: первичная карта («уровень 0»: точки с углом и местом штриха, детекторы), карты уровней (точки, их окна, детекторы по слоям), углопозиции обучающих картинок, кисть |
| `GET /api/example` | пример: случайная пятёрка из первых `DASHBOARD_EXAMPLES` картинок MNIST t10k и её номер в MNIST |
| `GET /api/status` | сколько рабочих процессов готово считать рисунки (страница шлёт столько запросов сразу) |
| `POST /api/see` | рисунок 28×28 → путь по уровням: кусочки, зажжённые точки, сработавшие детекторы, детали (у штриха — его деталь уровня 0: сильнейшая точка первичной карты и сила); класс — ответ памяти (среднее долей голосов кусочков по уровням и согласие уровней, §7.1.13.3), голоса по уровням и вероятности линейного читателя (§9.3); с выбранной точкой карты — ещё и её кусочки |
| `POST /api/point` | рисунок и выбранная точка карты → все кусочки рисунка, которые её зажгли, по убыванию активации |

Рисунки считают рабочие процессы (`DASHBOARD_WORKERS`): у каждого свой обученный цикл, поэтому несколько рисунков
идут параллельно и страница обновляется чаще. Рисунок, которым штрих закончен, считает главный процесс: его итог
пишется в журнал. Пока рабочие процессы загружаются (и когда их нет), всё считает главный.

Массивы идут строкой base64 своих байтов (little-endian, как у numpy и у типизированных массивов браузера); тип
каждого поля задан здесь и на странице одинаково. Сервер слушает только 127.0.0.1.
"""
from __future__ import annotations

import base64
import json
import multiprocessing
import os
import queue
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from damp_api import Damp, LevelMap, PrimaryMap, PrimarySight, Sight, TrainedLadder, TrainingSticks

STATIC = Path(__file__).resolve().parent / "static"
KINDS = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}
SIDE = 28  # сторона рисунка, как у MNIST
BODY_LIMIT = 65536  # рисунок 28×28 в base64 — около килобайта


def packed(values, dtype: str) -> str:  # noqa: ANN001
    """Массив строкой base64 его байтов в типе dtype."""
    return base64.b64encode(np.ascontiguousarray(np.asarray(values), dtype=dtype).tobytes()).decode("ascii")


class DashboardData:
    """Ответы сервера в JSON: карты обученного цикла (один раз) и путь рисунка (на каждый штрих)."""

    @staticmethod
    def primary_map(primary: "PrimaryMap | None") -> "dict | None":
        """Первичная карта — «уровень 0»: точки с углом и местом своей углопозиции и детекторы над картой."""
        if primary is None:
            return None
        return {
            "height": primary.height, "width": primary.width, "points": int(np.asarray(primary.angle).shape[0]),
            "cells": packed(primary.cells, "<i2"), "angle": packed(primary.angle, "<f4"), "y": packed(primary.y, "<f4"),
            "x": packed(primary.x, "<f4"), "labels": packed(primary.labels, "u1"), "period": primary.angle_period,
            "detectors": packed(primary.detectors, "<f4"), "count": int(np.asarray(primary.layer).shape[0]),
            "layer": packed(primary.layer, "u1"), "layerThresholds": list(primary.layer_thresholds),
            "activation": primary.activation_threshold,
            # Уровень 0 в цикле: элементы уровня 1 — его детали, страница рисует их, как детали уровней выше.
            "inCycle": bool(primary.in_cycle),
        }

    @staticmethod
    def primary_sight(found: "PrimarySight | None", sticks: int) -> "dict | None":
        """Отклик первичной карты на рисунок: у точки — её сильнейшая активация и сколько штрихов её зажгли, у
        детектора — сильнейший уровень и сколько штрихов в него набилось. У штриха — деталь уровня 0 (П.1): точка
        карты, которую он зажёг сильнее всего (−1 — ни одной), и уровень активации детали — сильнейший из сработавших
        на нём детекторов (§6.5); так же страница рисует детали уровней выше."""
        if found is None:
            return None
        lit_stick = np.asarray(found.lit_stick, dtype=np.int64)
        lit_point, lit_value = np.asarray(found.lit_point), np.asarray(found.lit_value, dtype=np.float32)
        rising = np.argsort(lit_value, kind="stable")
        activation = np.zeros(found.points, dtype=np.float32)
        activation[lit_point[rising]] = lit_value[rising]
        # Пары идут от слабых к сильным: у штриха остаётся последняя запись — сильнейшая зажжённая им точка.
        stick_point = np.full(int(sticks), -1, dtype=np.int32)
        stick_point[lit_stick[rising]] = lit_point[rising]
        fired_stick = np.asarray(found.fired_stick, dtype=np.int64)
        fired_detector, fired_level = np.asarray(found.fired_detector), np.asarray(found.fired_level, dtype=np.float32)
        rising = np.argsort(fired_level, kind="stable")
        strongest = np.zeros(found.detectors, dtype=np.float32)
        strongest[fired_detector[rising]] = fired_level[rising]
        stick_strength = np.zeros(int(sticks), dtype=np.float32)
        stick_strength[fired_stick[rising]] = fired_level[rising]
        return {
            "sticks": int(sticks),
            "pointActivation": packed(activation, "<f4"),
            "pointCount": packed(np.minimum(np.bincount(lit_point, minlength=found.points), 65535), "<u2"),
            "litPairs": int(lit_point.shape[0]), "fired": int(fired_detector.shape[0]),
            "detectorCount": packed(np.minimum(np.bincount(fired_detector, minlength=found.detectors), 65535), "<u2"),
            "detectorLevel": packed(strongest, "<f4"),
            "stickPoint": packed(stick_point, "<i4"), "stickStrength": packed(stick_strength, "<f4"),
        }

    @staticmethod
    def model(ladder: TrainedLadder, maps: tuple[LevelMap, ...], sticks: TrainingSticks, settings) -> dict:  # noqa: ANN001
        return {
            # Уровень 0 — первичная карта углопозиций (пинвилы); None — карты нет.
            "primary": DashboardData.primary_map(ladder.primary_map()),
            "train": ladder.train_pictures,
            "originals": sticks.originals,
            "decay": ladder.decay,
            "workers": int(settings["DASHBOARD_WORKERS"]),
            # Штрихи по середине линии: угол без стрелки, 0…180° (0° и 180° — одна линия).
            "axial": bool(settings["MNIST_STROKE_DETECTORS"]) and bool(settings["MNIST_STROKE_MIDLINE"]),
            "topClusters": int(settings["DASHBOARD_TOP_CLUSTERS"]),
            "litPerPiece": int(settings["DASHBOARD_LIT_PER_PIECE"]),
            "ownClusters": int(settings["DASHBOARD_OWN_CLUSTERS"]),
            "litFloor": float(settings["DASHBOARD_LIT_FLOOR"]),
            "brush": {"width": float(settings["DASHBOARD_BRUSH_WIDTH"]), "sigma": float(settings["DASHBOARD_BRUSH_SIGMA"]),
                      "centre": float(settings["DASHBOARD_CENTRE"])},
            "grid": {"cell": float(settings["MNIST_PATCH_SIZE"]) / float(settings["MNIST_SOBEL_UPSCALE"])
                     if settings["MNIST_SOBEL_UPSCALE_ENABLED"] else float(settings["MNIST_PATCH_SIZE"])},
            "sticks": {
                "picture": packed(sticks.picture, "<i4"), "y": packed(sticks.y, "<f4"), "x": packed(sticks.x, "<f4"),
                "angle": packed(sticks.angle, "<f4"), "span": packed(sticks.span, "<f4"),
                "labels": packed(sticks.labels, "u1"), "index": packed(sticks.index, "<i4"),
                "pictures": int(np.asarray(sticks.labels).shape[0]),
            },
            "levels": [
                {
                    "number": level.number, "height": level.height, "width": level.width,
                    "points": int(np.asarray(level.labels).shape[0]),
                    "cells": packed(level.cells, "<i2"), "labels": packed(level.labels, "u1"),
                    "energies": packed(level.energies, "<f4"),
                    "pointPicture": packed(level.point_picture, "<i4"), "pointCentre": packed(level.point_centre, "<f4"),
                    "pointExtent": packed(level.point_extent, "<f4"),
                    "detectors": packed(level.detectors, "<f4"), "count": int(np.asarray(level.layer).shape[0]),
                    "layer": packed(level.layer, "u1"), "layerThresholds": list(level.layer_thresholds),
                    "activation": level.activation_threshold, "fieldHalf": level.field_half,
                }
                for level in maps
            ],
        }

    @staticmethod
    def sight(sight: Sight, maps: tuple[LevelMap, ...], lit_per_piece: int) -> dict:
        """Путь рисунка для страницы. Зажжённых пар «кусочек — точка» сотни тысяч: по точке идёт её сильнейшая
        активация и число зажёгших её кусочков, по кусочку — lit_per_piece его сильнейших точек; срабатывания — все."""
        levels = []
        for level, built in zip(sight.levels, maps):
            points, detectors = int(np.asarray(built.labels).shape[0]), int(np.asarray(built.layer).shape[0])
            keyholes = int(np.asarray(level.keyhole_extent).shape[0])
            lit_keyhole, lit_point = np.asarray(level.lit_keyhole), np.asarray(level.lit_point)
            lit_value = np.asarray(level.lit_value, dtype=np.float32)
            # Сильнейшая активация точки: пары идут от слабых к сильным, при повторе точки остаётся последняя запись —
            # наибольшая. То же, что np.maximum.at, но в разы быстрее (пар — десятки тысяч на уровень).
            rising = np.argsort(lit_value, kind="stable")
            activation = np.zeros(points, dtype=np.float32)
            activation[lit_point[rising]] = lit_value[rising]
            order = np.lexsort((-lit_value, lit_keyhole))  # по кусочку, внутри — сильнейшие первыми
            ordered = lit_keyhole[order]
            rank = np.arange(order.shape[0]) - np.searchsorted(ordered, np.arange(keyholes))[ordered]
            top = order[rank < int(lit_per_piece)]
            fired_detector = np.asarray(level.fired_detector)
            fired_level = np.asarray(level.fired_level, dtype=np.float32)
            rising = np.argsort(fired_level, kind="stable")
            strongest = np.zeros(detectors, dtype=np.float32)
            strongest[fired_detector[rising]] = fired_level[rising]
            levels.append({
                "number": level.number, "keyholes": keyholes,
                "centre": packed(level.keyhole_centre, "<f4"), "extent": packed(level.keyhole_extent, "<f4"),
                "entryKeyhole": packed(level.entry_keyhole, "<u2"), "entryElement": packed(level.entry_element, "<u2"),
                "pointActivation": packed(activation, "<f4"),
                "pointCount": packed(np.minimum(np.bincount(lit_point, minlength=points), 65535), "<u2"),
                "litKeyhole": packed(lit_keyhole[top], "<u2"), "litPoint": packed(lit_point[top], "<u4"),
                "litValue": packed(lit_value[top], "<f4"),
                # Сколько точек зажёг каждый кусочек всего (сильнейших из них страница получает lit_per_piece).
                "litCount": packed(np.bincount(lit_keyhole, minlength=keyholes), "<u4"),
                "firedKeyhole": packed(level.fired_keyhole, "<u2"), "firedDetector": packed(fired_detector, "<u2"),
                "firedLevel": packed(fired_level, "<f4"),
                "detectorCount": packed(np.minimum(np.bincount(fired_detector, minlength=detectors), 65535), "<u2"),
                "detectorLevel": packed(strongest, "<f4"),
                "litPairs": int(lit_point.shape[0]), "fired": int(fired_detector.shape[0]),
                "strength": packed(level.strength, "<f4"), "detailLength": level.detail_length,
            })
        return {
            # Ответ памяти (§2.2.5, §7.1.13.3): цифра по среднему долей голосов уровней, согласие уровней и сами голоса.
            "memory": {"kind": sight.verdict, "digit": sight.memory_digit, "share": sight.memory_share,
                       "agreed": sight.memory_agreed, "votes": list(sight.memory_votes)},
            "votes": [
                {"number": level.number, "votes": list(level.votes), "digit": level.digit, "share": level.share,
                 "knowledge": level.knowledge, "threshold": level.threshold, "knows": level.knows,
                 "keyholes": level.keyholes, "voted": level.voted}
                for level in sight.votes
            ],
            # Уровень 0: отклик первичной карты на штрихи рисунка.
            "primary": DashboardData.primary_sight(sight.primary, int(np.asarray(sight.stick_angle).shape[0])),
            # Линейный читатель (§9.3) — мерка для целых цифр.
            "digit": sight.digit,
            "probabilities": list(sight.probabilities),
            "ms": sight.seconds * 1000.0,
            "sticks": {"angle": packed(sight.stick_angle, "<f4"), "y": packed(sight.stick_y, "<f4"), "x": packed(sight.stick_x, "<f4"),
                       "strength": packed(sight.stick_strength, "<f4"), "span": packed(sight.stick_span, "<f4"),
                       "count": int(np.asarray(sight.stick_angle).shape[0])},
            "levels": levels,
        }


    @staticmethod
    def point(sight: Sight, level: int, index: int) -> dict:
        """Кусочки рисунка, зажёгшие точку карты index уровня level (с нуля), по убыванию её активации. Страница
        получает у кусочка только его сильнейшие точки (DASHBOARD_LIT_PER_PIECE), поэтому для выбранной точки карты
        список кусочков идёт отдельно и целиком."""
        if not 0 <= int(level) < len(sight.levels):
            return {"keyholes": packed([], "<u2"), "values": packed([], "<f4")}
        found = sight.levels[int(level)]
        chosen = np.flatnonzero(np.asarray(found.lit_point) == int(index))
        values = np.asarray(found.lit_value, dtype=np.float32)[chosen]
        order = np.argsort(-values, kind="stable")
        return {"keyholes": packed(np.asarray(found.lit_keyhole)[chosen][order], "<u2"), "values": packed(values[order], "<f4")}


def encoded(payload: dict) -> bytes:
    """Ответ страницы байтами JSON."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class SightAnswers:
    """Ответы на рисунок в одном процессе: путь через обученный цикл и его упаковка для страницы.

    Последний путь помнится вместе со своим рисунком: запрос точки того же рисунка его не пересчитывает.
    """

    def __init__(self, ladder: TrainedLadder, maps: tuple[LevelMap, ...], lit_per_piece: int) -> None:
        self._ladder = ladder
        self._maps = maps
        self._lit = int(lit_per_piece)
        self._image: bytes = b""
        self._sight: Sight | None = None
        self._lock = threading.Lock()  # один рисунок за раз: системный слой и видеокарта общие

    def _path(self, image: bytes, final: bool) -> Sight:
        with self._lock:
            if self._sight is None or self._image != image or final:
                pixels = np.frombuffer(image, dtype=np.uint8).reshape(SIDE, SIDE)
                self._sight = self._ladder.see(pixels, log=final)
                self._image = image
            return self._sight

    def see(self, image: bytes, final: bool, point: "tuple[int, int] | None") -> bytes:
        """Путь рисунка для страницы; point — выбранная точка карты (уровень, номер): её кусочки идут в том же ответе."""
        sight = self._path(image, final)
        answer = {"ok": True, **DashboardData.sight(sight, self._maps, self._lit)}
        if point is not None:
            answer["point"] = DashboardData.point(sight, int(point[0]), int(point[1]))
        return encoded(answer)

    def point(self, image: bytes, level: int, index: int) -> bytes:
        return encoded({"ok": True, **DashboardData.point(self._path(image, False), level, index)})


def worker(connection, changes: dict, count: "int | None", map_points: "int | None", load: "str | None" = None) -> None:  # noqa: ANN001
    """Рабочий процесс: свой обученный цикл (шаги — из кеша; load — из снимка, без сверки) и ответы на рисунки по
    запросам главного процесса.

    Запрос — ("see", рисунок, точка) или ("point", рисунок, уровень, номер точки); ответ — ("ok", байты JSON) или
    ("error", причина). Главный процесс закрылся — соединение рвётся, и рабочий выходит сам.
    """
    try:
        damp = Damp(changes=changes or None, log_file=False)
        ladder = damp.cycle.load(load, check=0) if load else damp.cycle.trained(count, map_points=map_points)
        answers = SightAnswers(ladder, ladder.maps(), int(damp.settings["DASHBOARD_LIT_PER_PIECE"]))
        connection.send(("ready", ""))
    except Exception as error:  # noqa: BLE001 — причину видит главный процесс
        connection.send(("failed", f"{type(error).__name__}: {error}"))
        return
    while True:
        try:
            request = connection.recv()
        except (EOFError, OSError):
            return
        try:
            if request[0] == "see":
                connection.send(("ok", answers.see(request[1], False, request[2])))
            else:
                connection.send(("ok", answers.point(request[1], request[2], request[3])))
        except Exception as error:  # noqa: BLE001 — странице нужна причина
            connection.send(("error", f"{type(error).__name__}: {error}"))


class SightWorkers:
    """Рабочие процессы пути рисунка: в каждом свой обученный цикл, рисунки считаются параллельно.

    Процессы поднимаются по одному в фоне (цикл каждого — из кеша шагов; рабочие файлы у них общие, поэтому не
    разом). Они — копии главного цикла: в окно Rerun и в журнал не пишут. Процессы — не «демоны»: раскладка малой
    карты заводит свои параллельные процессы, а «демону» это запрещено. Поэтому сервер закрывает их сам (stop); если
    главный процесс убит, соединение рвётся и рабочие выходят сами.
    """

    def __init__(self, count: int, changes: dict, train: "int | None", map_points: "int | None",
                 load: "str | None" = None) -> None:
        self.count = int(count)
        self._load = load
        self.ready = 0
        self.failed = ""
        self._changes = dict(changes)
        self._train = train
        self._map_points = map_points
        self._idle: "queue.Queue[int]" = queue.Queue()
        self._connections: dict[int, object] = {}
        self._processes: list = []
        self._stopped = False

    def start(self) -> None:
        threading.Thread(target=self._launch, daemon=True).start()

    def _launch(self) -> None:
        context = multiprocessing.get_context("spawn")
        for number in range(self.count):
            if self._stopped:
                return
            parent, child = context.Pipe()
            before = os.environ.get("RERUN")
            os.environ["RERUN"] = "off"  # окно Rerun — у главного процесса
            try:
                process = context.Process(target=worker, args=(child, self._changes, self._train, self._map_points, self._load),
                                          daemon=False)
                process.start()
                self._processes.append((process, parent))
            finally:
                if before is None:
                    os.environ.pop("RERUN", None)
                else:
                    os.environ["RERUN"] = before
            child.close()
            try:
                status, note = parent.recv()
            except (EOFError, OSError):
                status, note = "failed", "рабочий процесс закрылся при запуске"
            if status != "ready":
                self.failed = str(note)
                continue
            self._connections[number] = parent
            self._idle.put(number)
            self.ready += 1

    def stop(self) -> None:
        """Закрыть рабочие процессы: соединение рвётся — рабочий выходит; кто ещё загружает цикл — снимается."""
        self._stopped = True
        self.ready = 0
        for process, connection in self._processes:
            try:
                connection.close()
            except OSError:
                pass
        for process, _ in self._processes:
            process.join(timeout=3.0)
            if process.is_alive():
                process.terminate()

    def ask(self, request: tuple) -> "bytes | None":
        """Ответ свободного рабочего процесса; None — готовых процессов нет (считает главный)."""
        if not self.ready:
            return None
        number = self._idle.get()
        connection = self._connections[number]
        try:
            connection.send(request)
            status, payload = connection.recv()
        except (EOFError, OSError):
            self.ready -= 1  # процесс закрылся: в очередь свободных он не возвращается
            return None
        self._idle.put(number)
        if status != "ok":
            raise RuntimeError(str(payload))
        return payload


class DashboardServer:
    """Обученный цикл и сервер страницы на этом компьютере (адрес 127.0.0.1: снаружи недоступен)."""

    def __init__(self, damp: Damp, *, count: int | None = None, map_points: int | None = None, port: int | None = None,
                 load: str | None = None, save: str | None = None, label: str = "", workers: int | None = None) -> None:
        """load — папка снимка обученного цикла: цикл берётся из неё без обучения (damp — с настройками снимка,
        `Damp.from_snapshot`); save — записать снимок после обучения (label — его подпись); workers — рабочих
        процессов, None — DASHBOARD_WORKERS."""
        self._damp = damp
        self._count = count
        self._map_points = map_points
        self._load = load
        self._save = save
        self._label = str(label)
        self._worker_count = workers
        self.saved = None  # SnapshotReport записанного снимка (save)
        self._host = str(damp.settings["DASHBOARD_HOST"])
        self._port = int(damp.settings["DASHBOARD_PORT"] if port is None else port)
        self._ladder: TrainedLadder | None = None
        self._answers: SightAnswers | None = None  # ответы главного процесса
        self._workers: SightWorkers | None = None  # рабочие процессы: поднимаются в serve
        self._maps: tuple[LevelMap, ...] = ()
        self._lit = int(damp.settings["DASHBOARD_LIT_PER_PIECE"])
        self._model: bytes = b""
        self._examples = None  # Digits: картинки MNIST t10k для кнопки «пример»
        self._fives: list[int] = []

    @property
    def address(self) -> str:
        return f"http://{self._host}:{self._port}/"

    def prepare(self) -> TrainedLadder:
        """Обучить цикл (шаги — из кеша, если такой цикл уже был) либо взять его из снимка (load), при save — записать
        снимок, и собрать карты уровней для страницы."""
        if self._load:
            self._ladder = self._damp.cycle.load(self._load)
        else:
            self._ladder = self._damp.cycle.trained(self._count, map_points=self._map_points)
        if self._save:
            self.saved = self._ladder.save(self._save, label=self._label)
        self._maps = self._ladder.maps()
        self._answers = SightAnswers(self._ladder, self._maps, self._lit)
        model = DashboardData.model(self._ladder, self._maps, self._ladder.sticks(), self._damp.settings)
        self._model = json.dumps(model, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        # Примеры — проверочные картинки MNIST t10k (в обучении их нет); цифра — та же, что у рисунков проекта.
        settings = self._damp.settings
        self._examples = self._damp.stimuli.digits(train=1, test=int(settings["DASHBOARD_EXAMPLES"]), train_views=())
        first, last = self._examples.groups["test"]
        labels = self._examples.labels
        self._fives = [i for i in range(first, last) if labels[i] == int(settings["LADDER_FIGURE_DIGIT"])]
        return self._ladder

    def example(self) -> dict:
        """Случайная пятёрка MNIST t10k — каждый раз другая — и её номер в MNIST."""
        if not self._fives:
            raise ValueError("среди примеров нет нужной цифры")
        picture = self._examples.picture(random.choice(self._fives))
        image = np.asarray(picture.image, dtype=np.uint8)
        return {"image": packed(image, "u1"), "mnist": picture.mnist, "digit": picture.digit}

    @staticmethod
    def _image(body: dict) -> bytes:
        raw = base64.b64decode(str(body["image"]))
        if len(raw) != SIDE * SIDE:
            raise ValueError(f"рисунок должен быть {SIDE}×{SIDE} байт")
        return raw

    def see(self, body: dict) -> bytes:
        """Путь рисунка из запроса страницы: {"image": base64 784 байт, "final": штрих закончен, "point": выбранная
        точка карты [уровень с нуля, номер] или её нет}. Ответ — байты JSON.

        Рисунок посреди штриха считает свободный рабочий процесс; последний рисунок штриха (final) — главный: его
        итог пишется в журнал. Нет готовых рабочих процессов — всё считает главный.
        """
        raw, final = self._image(body), bool(body.get("final", False))
        point = None if body.get("point") is None else (int(body["point"][0]), int(body["point"][1]))
        if not final and self._workers is not None:
            answer = self._workers.ask(("see", raw, point))
            if answer is not None:
                return answer
        return self._answers.see(raw, final, point)

    def point(self, body: dict) -> bytes:
        """Кусочки рисунка, зажёгшие точку карты: {"image": рисунок, "level": уровень с нуля, "index": номер точки}."""
        raw, level, index = self._image(body), int(body["level"]), int(body["index"])
        if self._workers is not None:
            answer = self._workers.ask(("point", raw, level, index))
            if answer is not None:
                return answer
        return self._answers.point(raw, level, index)

    def status(self) -> dict:
        """Сколько рабочих процессов готово, сколько задано и причина, если процесс не поднялся."""
        workers = self._workers
        if workers is None:
            return {"ready": 0, "workers": 0, "failed": ""}
        return {"ready": workers.ready, "workers": workers.count, "failed": workers.failed}

    def serve(self) -> None:
        """Обслуживать страницу, пока процесс не остановят (Ctrl+C). Рабочие процессы поднимаются в фоне."""
        count = int(self._damp.settings["DASHBOARD_WORKERS"] if self._worker_count is None else self._worker_count)
        if count > 0:
            self._workers = SightWorkers(count, self._damp.settings.changes, self._count, self._map_points, self._load)
            self._workers.start()
        http = self._http()
        try:
            http.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            http.server_close()
            if self._workers is not None:
                self._workers.stop()

    def snapshot(self, path: str, *, query: str = "example&pin=2:30", width: int = 1920, height: int = 1080) -> str:
        """Снимок страницы для README: пример-пятёрка (каждый раз другая) и выбранный кусочек уровня 2 со следом.

        Сервер поднимается на время снимка; браузер — Chrome через playwright (без окна; необязательная зависимость).
        """
        import importlib

        sync_playwright = importlib.import_module("playwright.sync_api").sync_playwright
        http = self._http()
        worker = threading.Thread(target=http.serve_forever, daemon=True)
        worker.start()
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(channel="chrome", headless=True)
                page = browser.new_page(viewport={"width": int(width), "height": int(height)})
                page.goto(f"{self.address}?{query}")
                page.wait_for_function("document.querySelector('#digit').textContent !== '—'", timeout=60000)
                page.wait_for_timeout(1500)  # вспышка новых кластеров гаснет за 0.9 с
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(path))
                browser.close()
        finally:
            http.shutdown()
            http.server_close()
        return str(path)

    def _http(self) -> ThreadingHTTPServer:
        """HTTP-сервер страницы на адресе DASHBOARD_HOST:порт."""
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # noqa: ANN002
                """Запросов за рисованием десятки в секунду — в консоль они не пишутся; итог штриха пишет журнал системы."""

            def _send(self, status: int, body: bytes, kind: str) -> None:
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, payload: dict, status: int = 200) -> None:
                self._send(status, json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
                           "application/json; charset=utf-8")

            def do_GET(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path == "/api/model":
                    self._send(200, server._model, "application/json; charset=utf-8")
                    return
                if path == "/api/status":
                    self._json({"ok": True, **server.status()})
                    return
                if path == "/api/example":
                    try:
                        self._json({"ok": True, **server.example()})
                    except Exception as error:  # noqa: BLE001
                        self._json({"ok": False, "error": f"{type(error).__name__}: {error}"}, 500)
                    return
                if path == "/favicon.ico":
                    self._send(204, b"", "image/x-icon")  # значка нет: пустой ответ, без ошибки в консоли браузера
                    return
                name = "index.html" if path == "/" else path[len("/static/"):] if path.startswith("/static/") else ""
                file = STATIC / Path(name).name if name else None
                kind = KINDS.get(file.suffix) if file is not None else None
                if kind is None or Path(name).name != name or not file.is_file():
                    self._send(404, "нет такой страницы".encode("utf-8"), "text/plain; charset=utf-8")
                    return
                self._send(200, file.read_bytes(), kind)

            def do_POST(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path not in ("/api/see", "/api/point"):
                    self._send(404, b"", "text/plain")
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = 0
                if length <= 0 or length > BODY_LIMIT:
                    self._json({"ok": False, "error": "пустой или слишком большой запрос"}, 400)
                    return
                try:
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                    self._send(200, server.see(body) if path == "/api/see" else server.point(body), "application/json; charset=utf-8")
                except Exception as error:  # noqa: BLE001 — странице нужна причина, а не оборванное соединение
                    self._json({"ok": False, "error": f"{type(error).__name__}: {error}"}, 500)

        http = ThreadingHTTPServer((self._host, self._port), Handler)
        http.daemon_threads = True
        return http
