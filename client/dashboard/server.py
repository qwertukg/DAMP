"""Сервер живого дашборда: страница с холстом и путь рисунка через обученный цикл (только через damp_api).

    python client/run.py dashboard                 # учит цикл (шаги — из кеша), открывает http://127.0.0.1:8765/

Запросы страницы:

| Запрос | Что отдаёт |
|---|---|
| `GET /` | страница `static/index.html` |
| `GET /static/<файл>` | её стили и код (только файлы папки `static/`) |
| `GET /api/model` | один раз: карты уровней (точки, их окна, детекторы по слоям), углопозиции обучающих картинок, кисть |
| `GET /api/example` | пример: случайная пятёрка из первых `DASHBOARD_EXAMPLES` картинок MNIST t10k и её номер в MNIST |
| `POST /api/see` | рисунок 28×28 → путь по уровням: кусочки, зажжённые точки, сработавшие детекторы, детали, вероятности |

Массивы идут строкой base64 своих байтов (little-endian, как у numpy и у типизированных массивов браузера); тип
каждого поля задан здесь и на странице одинаково. Сервер слушает только 127.0.0.1.
"""
from __future__ import annotations

import base64
import json
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from damp_api import Damp, LevelMap, Sight, TrainedLadder, TrainingSticks

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
    def model(ladder: TrainedLadder, maps: tuple[LevelMap, ...], sticks: TrainingSticks, settings) -> dict:  # noqa: ANN001
        return {
            "train": ladder.train_pictures,
            "originals": sticks.originals,
            "decay": ladder.decay,
            "topClusters": int(settings["DASHBOARD_TOP_CLUSTERS"]),
            "litPerPiece": int(settings["DASHBOARD_LIT_PER_PIECE"]),
            "ownClusters": int(settings["DASHBOARD_OWN_CLUSTERS"]),
            "litFloor": float(settings["DASHBOARD_LIT_FLOOR"]),
            "brush": {"width": float(settings["DASHBOARD_BRUSH_WIDTH"]), "sigma": float(settings["DASHBOARD_BRUSH_SIGMA"]),
                      "centre": float(settings["DASHBOARD_CENTRE"])},
            "grid": {"cell": float(settings["MNIST_PATCH_SIZE"]) / float(settings["MNIST_SOBEL_UPSCALE"])
                     if settings["MNIST_SOBEL_UPSCALE_ENABLED"] else float(settings["MNIST_PATCH_SIZE"])},
            "sticks": {
                "picture": packed(sticks.picture, "<i4"), "y": packed(sticks.y, "<i2"), "x": packed(sticks.x, "<i2"),
                "angle": packed(sticks.angle, "<f4"), "labels": packed(sticks.labels, "u1"), "index": packed(sticks.index, "<i4"),
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
            activation = np.zeros(points, dtype=np.float32)
            np.maximum.at(activation, lit_point, lit_value)
            order = np.lexsort((-lit_value, lit_keyhole))  # по кусочку, внутри — сильнейшие первыми
            ordered = lit_keyhole[order]
            rank = np.arange(order.shape[0]) - np.searchsorted(ordered, np.arange(keyholes))[ordered]
            top = order[rank < int(lit_per_piece)]
            fired_detector = np.asarray(level.fired_detector)
            fired_level = np.asarray(level.fired_level, dtype=np.float32)
            strongest = np.zeros(detectors, dtype=np.float32)
            np.maximum.at(strongest, fired_detector, fired_level)
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
            "digit": sight.digit,
            "probabilities": list(sight.probabilities),
            "ms": sight.seconds * 1000.0,
            "sticks": {"angle": packed(sight.stick_angle, "<f4"), "y": packed(sight.stick_y, "<i2"), "x": packed(sight.stick_x, "<i2"),
                       "strength": packed(sight.stick_strength, "<f4"), "count": int(np.asarray(sight.stick_angle).shape[0])},
            "levels": levels,
        }


class DashboardServer:
    """Обученный цикл и сервер страницы на этом компьютере (адрес 127.0.0.1: снаружи недоступен)."""

    def __init__(self, damp: Damp, *, count: int | None = None, map_points: int | None = None, port: int | None = None) -> None:
        self._damp = damp
        self._count = count
        self._map_points = map_points
        self._host = str(damp.settings["DASHBOARD_HOST"])
        self._port = int(damp.settings["DASHBOARD_PORT"] if port is None else port)
        self._ladder: TrainedLadder | None = None
        self._maps: tuple[LevelMap, ...] = ()
        self._lit = int(damp.settings["DASHBOARD_LIT_PER_PIECE"])
        self._model: bytes = b""
        self._examples = None  # Digits: картинки MNIST t10k для кнопки «пример»
        self._fives: list[int] = []
        self._lock = threading.Lock()  # один рисунок за раз: системный слой и видеокарта общие

    @property
    def address(self) -> str:
        return f"http://{self._host}:{self._port}/"

    def prepare(self) -> TrainedLadder:
        """Обучить цикл (шаги — из кеша, если такой цикл уже был) и собрать карты уровней для страницы."""
        self._ladder = self._damp.cycle.trained(self._count, map_points=self._map_points)
        self._maps = self._ladder.maps()
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

    def see(self, body: dict) -> dict:
        """Путь рисунка из запроса страницы: {"image": base64 784 байт, "final": штрих закончен}."""
        pixels = np.frombuffer(base64.b64decode(str(body["image"])), dtype=np.uint8)
        if pixels.shape[0] != SIDE * SIDE:
            raise ValueError(f"рисунок должен быть {SIDE}×{SIDE} байт")
        with self._lock:
            sight = self._ladder.see(pixels.reshape(SIDE, SIDE), log=bool(body.get("final", False)))
        return DashboardData.sight(sight, self._maps, self._lit)

    def serve(self) -> None:
        """Обслуживать страницу, пока процесс не остановят (Ctrl+C)."""
        http = self._http()
        try:
            http.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            http.server_close()

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
                if urlparse(self.path).path != "/api/see":
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
                    self._json({"ok": True, **server.see(json.loads(self.rfile.read(length).decode("utf-8")))})
                except Exception as error:  # noqa: BLE001 — странице нужна причина, а не оборванное соединение
                    self._json({"ok": False, "error": f"{type(error).__name__}: {error}"}, 500)

        http = ThreadingHTTPServer((self._host, self._port), Handler)
        http.daemon_threads = True
        return http
