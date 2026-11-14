"""Единый журнал DAMP: одна запись события — в консоль, в Rerun и в файл JSON Lines с ротацией.

Запись делают только реализации: ``LOGGER.event("область.объект.действие", section=<параграф>, data={...})``.
В конце строки консоли — номер параграфа статьи ``[6.5]``, раздела продолжения ``[П.5]`` или ``[вне статьи]``.
Файл — одна запись на строку (ts, run, level, event, section, data, msg): удобно искать и человеку (rg, jq),
и нейросети.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import rerun as rr

LOG_PATH = "damp/log"


class JsonlLog:
    """Файл журнала JSON Lines с ротацией по размеру (стандартный RotatingFileHandler)."""

    def __init__(self, path: str, *, max_bytes: int, backups: int, array_limit: int) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        self._array_limit = int(array_limit)
        self._run = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{os.getpid()}"
        self._logger = logging.getLogger(f"damp.jsonl.{self._run}")
        self._logger.propagate = False
        self._logger.setLevel(logging.INFO)
        handler = logging.handlers.RotatingFileHandler(self.path, maxBytes=int(max_bytes), backupCount=int(backups), encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)

    @property
    def run(self) -> str:
        return self._run

    def close(self) -> None:
        """Закрыть файл (на Windows открытый файл нельзя удалить или перенести)."""
        for handler in list(self._logger.handlers):
            handler.close()
            self._logger.removeHandler(handler)

    def write(self, event: str, section: str, data: Mapping[str, Any] | None, message: str, *, cycle: int | None = None) -> None:
        """Одна запись; cycle — номер уровня цикла, на котором она сделана (0 — до первого уровня; None — вне цикла)."""
        level = "warning" if event.endswith(".warning") or ".warning." in event else "info"
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "run": self._run,
            "level": level,
            "event": event,
            "section": section,
            "data": self._plain(dict(data or {})),
            "msg": message,
        }
        if cycle is not None:
            record["cycle"] = int(cycle)
        self._logger.info(json.dumps(record, ensure_ascii=False, default=repr))

    def _plain(self, value: Any) -> Any:
        """Значение → JSON: ключи словарей — строки, массивы numpy — списки (длинные — кратко), прочее — repr."""
        if isinstance(value, Mapping):
            return {str(key): self._plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [self._plain(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        try:
            import numpy as np

            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, np.ndarray):
                if value.size <= self._array_limit:
                    return value.tolist()
                numeric = value.size and np.issubdtype(value.dtype, np.number)
                return {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "min": float(value.min()) if numeric else None,
                    "max": float(value.max()) if numeric else None,
                }
        except ImportError:  # pragma: no cover - numpy есть всегда
            pass
        return repr(value)


@dataclass(frozen=True)
class LogVisual:
    path: str
    payload: object
    timeline: str | None = None
    time: int | float | None = None
    static: bool = False


class LogIntervalPolicy:
    def __init__(self, intervals: Mapping[str, int], *, default_interval: int = 1) -> None:
        self._intervals: dict[str, int] = {}
        for event, value in intervals.items():
            interval = int(value)
            if interval <= 0:
                raise ValueError("interval must be positive")
            self._intervals[str(event)] = interval
        self._default_interval = int(default_interval)
        if self._default_interval <= 0:
            raise ValueError("default_interval must be positive")
        self._counters: dict[str, int] = {}

    def should_log(self, event: str) -> bool:
        interval = self._intervals.get(event, self._default_interval)
        if interval <= 1:
            return True
        count = self._counters.get(event, 0)
        self._counters[event] = count + 1
        return count % interval == 0

    def reset(self) -> None:
        self._counters.clear()


class RerunViewerBlueprint:
    """Панели окна Rerun для DAMP: ход раскладки всегда на первом плане.

    Rerun запоминает расположение панелей для каждого приложения и восстанавливает его при следующем
    открытии; если в прошлой сессии панели закрыли или они ссылаются на сущности, которых больше нет,
    окно выглядит пустым. Поэтому логгер отправляет эту раскладку при открытии окна и делает её активной.
    Пути сущностей совпадают с теми, куда пишут Layout, ActivationEngine и DetectorLayerView.
    """

    LAYOUT_MAP = "/layout/angle"
    LAYOUT_CODE_MAP = "/layout/image"
    SWAP_RATIO = "/layout/swap_ratio"
    AVERAGE_ENERGY = "/layout/energy/long"
    TEXT_LOG = f"/{LOG_PATH}"

    @classmethod
    def build(cls):
        import rerun.blueprint as rrb

        progress = rrb.Spatial2DView(name="Ход раскладки: по углу (рис. 16)", origin=cls.LAYOUT_MAP)
        results = rrb.Tabs(
            rrb.Spatial2DView(name="Ход раскладки: псевдоцвета (§5.14)", origin=cls.LAYOUT_CODE_MAP),
            rrb.Spatial2DView(
                name="Итог: псевдоцвета (§5.14)", contents=["/layout/final/image", "/layout/final/points"]
            ),
            rrb.Spatial2DView(
                name="Итог: по углу (рис. 16)",
                contents=["/layout/final/component_image", "/layout/final/component_points"],
            ),
            rrb.Spatial2DView(
                name="Итог: энергия (рис. 18)", contents=["/layout/final/energy_image", "/layout/final/energy_points"]
            ),
            rrb.Spatial2DView(name="Активация", origin="/activation/map"),
            rrb.Spatial2DView(name="Энергия точек", origin="/activation/energy"),
            rrb.Spatial2DView(name="Детекторы кодера", origin="/encoder/detectors"),
        )
        # Вся история: X графиков — время записи, все раскладки прогона подряд, от первой до идущей.
        history = rrb.VisibleTimeRange("log_time", start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.infinite())
        charts = rrb.Vertical(
            rrb.TimeSeriesView(name="Доля обменов (§5.6) · X: время, вся история", origin=cls.SWAP_RATIO, time_ranges=history),
            rrb.TimeSeriesView(name="Качество Ē (§5.8) · X: время, вся история", origin=cls.AVERAGE_ENERGY, time_ranges=history),
            rrb.TextLogView(name="Лог", origin=cls.TEXT_LOG),
            row_shares=[1, 1, 2],
        )
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(progress, results, row_shares=[3, 2]),
                charts,
                column_shares=[3, 2],
            ),
            # Шкала — время записи, окно следует за новыми данными: идущая работа видна без кликов.
            rrb.TimePanel(state="collapsed", timeline="log_time", play_state="following"),
            rrb.SelectionPanel(state="collapsed"),
        )


class DigitSketchBlueprint:
    """Окно digit_sketch.py: оригинальный стимул MNIST рядом с цифрой, собранной из линий его углопозиций.

    Обе панели показывают одну и ту же область side × side пикселей, поэтому линию можно сравнить с местом
    стимула, откуда она взята; сетка окон Собеля видна на обеих. Пути сущностей отсюда же берёт
    AnglePositionSketch, чтобы панели и данные не разошлись.
    """

    def __init__(self, root: str, *, side: float, sobel_root: str | None = None) -> None:
        self.root = "/" + root.strip("/")
        self.side = float(side)
        self.sobel_root = None if sobel_root is None else "/" + sobel_root.strip("/")

    @property
    def stimulus_path(self) -> str:
        return f"{self.root}/stimulus"

    @property
    def grid_path(self) -> str:
        return f"{self.root}/grid"

    @property
    def lines_path(self) -> str:
        return f"{self.root}/lines"

    def build(self):
        import rerun.blueprint as rrb

        bounds = rrb.VisualBounds2D(x_range=[0.0, self.side], y_range=[0.0, self.side])
        stimulus = rrb.Spatial2DView(
            name="Оригинальный стимул",
            origin=self.root,
            contents=[self.stimulus_path, self.grid_path],
            visual_bounds=bounds,
        )
        sketch = rrb.Spatial2DView(
            name="Цифра из линий углопозиций",
            origin=self.root,
            contents=[self.lines_path, self.grid_path],
            background=[0, 0, 0],
            visual_bounds=bounds,
        )
        text_log = rrb.TextLogView(name="Лог", origin=f"/{LOG_PATH}")
        bottom = text_log if self.sobel_root is None else rrb.Horizontal(sobel_views(self.sobel_root), text_log)
        return rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(stimulus, sketch),
                bottom,
                row_shares=[3, 2] if self.sobel_root else [3, 1],
            ),
            rrb.SelectionPanel(state="collapsed"),
        )


def sobel_views(root: str):
    """Вкладки предобработки UpscaledSobel: увеличенная картинка и отклики Собеля (красный +, синий −)."""
    import rerun.blueprint as rrb

    return rrb.Tabs(
        rrb.Spatial2DView(name="Увеличено + контраст", origin=f"{root}/image"),
        rrb.Spatial2DView(name="Собель X (вправо)", origin=f"{root}/gx"),
        rrb.Spatial2DView(name="Собель Y (вверх)", origin=f"{root}/gy"),
    )


class DigitsSheetBlueprint:
    """Окно digits_contour_test.py: таблица «оригинал | цифра из линий углопозиций» по всем цифрам."""

    def __init__(self, root: str, *, sobel_root: str | None = None) -> None:
        self.root = "/" + root.strip("/")
        self.sobel_root = None if sobel_root is None else "/" + sobel_root.strip("/")

    @property
    def sheet_path(self) -> str:
        return f"{self.root}/sheet"

    def build(self):
        import rerun.blueprint as rrb

        sheet = rrb.Spatial2DView(name="Цифры из линий углопозиций", origin=self.root)
        left = sheet if self.sobel_root is None else rrb.Vertical(sheet, sobel_views(self.sobel_root), row_shares=[3, 1])
        return rrb.Blueprint(
            rrb.Horizontal(
                left,
                rrb.TextLogView(name="Лог", origin=f"/{LOG_PATH}"),
                column_shares=[3, 2],
            ),
            rrb.SelectionPanel(state="collapsed"),
        )


class DampLogger:
    # Шкала цикла в Rerun: номер уровня (0 — стимулы, до первого уровня). Порядок записей хранят встроенные шкалы
    # Rerun log_tick и log_time; раскладка панелей цикла делает шкалу cycle активной (TimePanel(timeline=...)).
    CYCLE_TIMELINE = "cycle"

    def __init__(self, app_id: str = "damp", base_path: str = LOG_PATH, *, spawn: bool = True) -> None:
        self._app_id = app_id
        self._base_path = base_path
        self._spawn = spawn
        self._initialized = False
        self._spawned = False
        self._cycle: int | None = None  # номер уровня цикла, которым помечаются записи (None — не цикл)
        self._recording_path: str | None = None
        self._watchers: dict[str, list[Callable[..., None]]] = {}  # ход шагов по мере счёта → подписчики (показ)
        self._interval_policy: LogIntervalPolicy | None = None
        self._blueprint_builder: Callable[[], Any] = RerunViewerBlueprint.build
        self._cycle_blueprint = False  # в окне раскладка прогона цикла (вне цикла — вернуть ход раскладки)
        # True — журнал выключен целиком (рабочие процессы: записи ведёт основной процесс).
        self._muted = False
        self._file: JsonlLog | None = None

    def attach_file(self, sink: JsonlLog | None) -> None:
        """Файл журнала: каждая запись события пишется и туда (None — без файла)."""
        self._file = sink

    @property
    def file(self) -> JsonlLog | None:
        return self._file

    def mute(self) -> None:
        """Выключает журнал этого процесса: ни консоли, ни rerun (для рабочих процессов)."""
        self._muted = True

    @contextmanager
    def quiet(self):
        """Временно выключает журнал: для пробных построений, записи которых повторяли бы записи итогового."""
        previous = self._muted
        self._muted = True
        try:
            yield
        finally:
            self._muted = previous

    def use_blueprint(self, builder: Callable[[], Any], *, cycle: bool = False) -> None:
        """Своя раскладка панелей окна для отдельного скрипта; по умолчанию — ход раскладки DAMP.

        Если окно уже открыто, новая раскладка сразу становится активной. cycle — раскладка прогона цикла: с первым
        событием вне цикла окно само возвращается к ходу раскладки (работа вне цикла видна на нём).
        """
        self._blueprint_builder = builder
        self._cycle_blueprint = bool(cycle)
        if self._spawned:
            self._send_blueprint()

    def _ensure_rerun(self) -> None:
        if not self._initialized:
            rr.init(self._app_id)
            self._initialized = True
        if self._spawn and not self._spawned and rr.is_enabled():
            # DAMP_RERUN_VIEWER=off — без окна: прогон цикла пишется только в файл .rrd (скриншоты, свой UI).
            self._viewer = os.environ.get("DAMP_RERUN_VIEWER", "on").lower() != "off"
            if self._viewer:
                rr.spawn()
            self._spawned = True
            self._send_blueprint()

    # ------------------------------------------------------------------ ход счёта для показа (вне статьи)
    def watch(self, name: str, callback: Callable[..., None]) -> None:
        """Подписка на ход шага name: реализация по ходу счёта зовёт progress(name, ...); подписчиков может быть несколько.

        Подписчик только смотрит (рисует в Rerun) и на счёт не влияет; сам решает, как часто рисовать.
        """
        self._watchers.setdefault(name, [])
        if callback not in self._watchers[name]:
            self._watchers[name].append(callback)

    def unwatch(self, name: str, callback: Callable[..., None]) -> None:
        """Снять подписку callback на ход шага name."""
        callbacks = self._watchers.get(name, [])
        if callback in callbacks:
            callbacks.remove(callback)
        if not callbacks:
            self._watchers.pop(name, None)

    def watched(self, name: str) -> bool:
        """True — на шаг name подписаны: реализации стоит отдать ход счёта (иначе — ни одного лишнего действия)."""
        return not self._muted and bool(self._watchers.get(name))

    def progress(self, name: str, **values: Any) -> None:
        """Ход шага name подписчикам. Ошибка подписчика не роняет счёт: его подписка снимается, в журнале — предупреждение."""
        if self._muted:
            return
        for callback in list(self._watchers.get(name, [])):
            try:
                callback(**values)
            except Exception as error:  # показ — не главное
                self.unwatch(name, callback)
                self.event("rerun.progress.warning", section="вне статьи", data={"step": name, "error": repr(error)})

    def start_recording(self, name: str) -> None:
        """Новая запись Rerun для прогона name: в окно (если оно есть) и в файл RERUN_RECORD_DIR/<время>-<pid>-<name>.rrd
        (RERUN_RECORD; хранятся последние RERUN_RECORD_KEEP). Первая строка записи — на шкале цикла, поэтому окно
        открывается на ней; в файле — только этот прогон, без записей до него."""
        if self._muted:
            return
        self._ensure_rerun()
        if not rr.is_enabled():
            return
        try:
            from damp import settings

            rr.RecordingStream(self._app_id, make_default=True, make_thread_default=True)
            sinks = [rr.GrpcSink()] if getattr(self, "_viewer", False) else []
            if bool(getattr(settings, "RERUN_RECORD", False)):
                folder = Path(str(settings.RERUN_RECORD_DIR))
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}-{name}.rrd"
                sinks.append(rr.FileSink(str(path)))
                self._recording_path = str(path)
                keep = max(1, int(settings.RERUN_RECORD_KEEP))
                for old in sorted(folder.glob("*.rrd"), key=lambda item: item.stat().st_mtime)[:-keep]:
                    old.unlink(missing_ok=True)
            if sinks:
                rr.set_sinks(*sinks)
            rr.reset_time()
            rr.set_time(self.CYCLE_TIMELINE, sequence=0 if self._cycle is None else self._cycle)
            rr.log(f"{self._base_path}", rr.TextLog(f"запись Rerun: прогон {name}"))
            rr.reset_time()
            self._send_blueprint()
            self._console_log(f"rerun.record name={name} path={self._recording_path} [вне статьи]")
        except Exception as exc:  # окно Rerun работает и без отдельной записи
            self._console_log(f"rerun.record.warning error={exc!r} [вне статьи]")

    @property
    def recording_path(self) -> str | None:
        """Файл .rrd этого прогона, если он пишется."""
        return self._recording_path

    def set_cycle(self, level: int | None) -> None:
        """Записи дальше помечаются уровнем цикла level (шкала «cycle» в Rerun, поле cycle в файле); None — снять."""
        self._cycle = None if level is None else int(level)

    @property
    def cycle(self) -> int | None:
        return self._cycle

    def _stamp(self) -> None:
        """Время записи в Rerun: уровень цикла, если идёт цикл; порядок записей Rerun ведёт сам (log_tick)."""
        rr.reset_time()
        if self._cycle is not None:
            rr.set_time(self.CYCLE_TIMELINE, sequence=self._cycle)

    def _send_blueprint(self) -> None:
        """Делает раскладку панелей DAMP активной, перекрывая сохранённую в окне с прошлых сессий.

        DAMP_RERUN_BLUEPRINT=off — запись без раскладки: её задаёт тот, кто открывает файл (свой .rbl, свой UI).
        """
        if os.environ.get("DAMP_RERUN_BLUEPRINT", "on").lower() == "off":
            return
        try:
            rr.send_blueprint(self._blueprint_builder(), make_active=True, make_default=True)
        except Exception as exc:  # окно всё равно получает данные, просто с панелями по умолчанию
            self._console_log(f"rerun.blueprint.error error={exc!r}")

    def _console_log(self, message: str) -> None:
        stream = sys.stdout
        try:
            print(message, file=stream, flush=True)
        except UnicodeEncodeError:
            encoding = getattr(stream, "encoding", None) or "utf-8"
            safe_message = message.encode(encoding, errors="backslashreplace").decode(encoding)
            print(safe_message, file=stream, flush=True)

    @staticmethod
    def _format_value(value: Any, *, max_items: int = 8, max_chars: int = 200) -> str:
        if value is None:
            return "None"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return f"{value:.6g}"
        if isinstance(value, str):
            return value if len(value) <= max_chars else f"{value[:max_chars]}..."
        if isinstance(value, dict):
            items = list(value.items())
            parts = [f"{k}={DampLogger._format_value(v)}" for k, v in items[:max_items]]
            if len(items) > max_items:
                parts.append("...")
            return "{" + ", ".join(parts) + "}"
        if isinstance(value, (list, tuple)):
            seq = list(value)
            parts = [DampLogger._format_value(v) for v in seq[:max_items]]
            if len(seq) > max_items:
                parts.append("...")
            if isinstance(value, list):
                return "[" + ", ".join(parts) + "]"
            return "(" + ", ".join(parts) + ")"
        if isinstance(value, set):
            seq = list(value)
            parts = [DampLogger._format_value(v) for v in seq[:max_items]]
            if len(seq) > max_items:
                parts.append("...")
            return "{" + ", ".join(parts) + "}"
        try:
            from damp.encoding.bitarray import BitArray

            if isinstance(value, BitArray):
                length = len(value)
                ones = value.count()
                if length <= 64:
                    return f"BitArray(len={length}, ones={ones}, bits={value.to01()})"
                return f"BitArray(len={length}, ones={ones})"
        except Exception:
            pass
        text = repr(value)
        return text if len(text) <= max_chars else f"{text[:max_chars]}..."

    @staticmethod
    def _coerce_rr_value(value: Any) -> Any:
        # Числа — всегда дробные: Rerun помнит тип поля, и поле, бывшее то целым, то дробным, он отвергает.
        if value is None:
            return "None"
        if isinstance(value, (bool, float, str)):
            return value
        if isinstance(value, int):
            return float(value)
        if isinstance(value, (list, tuple)):
            if all(isinstance(item, (bool, int, float, str)) for item in value):
                return [float(item) if isinstance(item, int) and not isinstance(item, bool) else item for item in value]
            return DampLogger._format_value(value)
        if isinstance(value, dict):
            return DampLogger._format_value(value)
        return DampLogger._format_value(value)

    def _coerce_anyvalues(self, data: Mapping[str, Any]) -> Mapping[str, Any]:
        """Поля записи для Rerun. Rerun помнит тип поля по имени на весь прогон: если поле с тем же именем раньше было
        другого типа (число и строка), оно пишется под именем с пометкой типа (`step__text`) — без предупреждений."""
        coerced: dict[str, Any] = {}
        kinds = self.__dict__.setdefault("_field_kinds", {})
        for key, value in data.items():
            if value is None:
                continue
            value = str(value) if key == "label" else self._coerce_rr_value(value)
            sample = value[0] if isinstance(value, list) and value else value
            kind = "text" if isinstance(sample, str) else "flag" if isinstance(sample, bool) else "number"
            if isinstance(value, list):
                kind += "s"
            name = key if kinds.setdefault(key, kind) == kind else f"{key}__{kind}"
            coerced[name] = value
        return coerced

    def configure_intervals(self, intervals: Mapping[str, int], *, default_interval: int = 1) -> None:
        self._interval_policy = LogIntervalPolicy(intervals, default_interval=default_interval)

    def should_log(self, event: str) -> bool:
        if self._muted:
            return False
        if self._interval_policy is None:
            return True
        return self._interval_policy.should_log(event)

    def visuals_enabled(self) -> bool:
        """True — картинки записей доходят до Rerun: журнал не выключен и Rerun включён (RERUN=on).

        Нужен там, где сама картинка стоит дорого (карта раскладки из миллионов точек): без Rerun её не строят.
        """
        if self._muted:
            return False
        self._ensure_rerun()
        return bool(rr.is_enabled())

    def event(
        self,
        event: str,
        *,
        section: str,
        data: Mapping[str, Any] | None = None,
        path: str | None = None,
        visuals: Sequence[LogVisual] | None = None,
        force: bool | None = None,
    ) -> None:
        if not section:
            raise ValueError("section must be provided")
        if self._muted:
            return
        if self._cycle_blueprint and self._cycle is None:
            # Прогон цикла закончился, идёт другая работа: окно возвращается к ходу раскладки (вся история по времени).
            self.use_blueprint(RerunViewerBlueprint.build)
        if force is None:
            if not self.should_log(event):
                return
        elif force is False:
            return
        if data:
            details = " ".join(f"{k}={self._format_value(v)}" for k, v in data.items())
            message = f"{event} {details}"
        else:
            message = event
        message = f"{message} [{section}]"
        self._console_log(message)
        if self._file is not None:
            self._file.write(event, section, data, message, cycle=self._cycle)
        self._ensure_rerun()
        self._stamp()
        base_path = path or self._base_path
        rr.log(base_path, rr.TextLog(message))
        if data:
            anyvalues_path = f"{base_path}/{event}"
            anyvalues = self._coerce_anyvalues(data)
            if anyvalues:
                rr.log(anyvalues_path, rr.AnyValues(**anyvalues))
        if visuals:
            for visual in visuals:
                self._stamp()
                # Своя шкала картинки (например, шаг раскладки) — только вне цикла: в цикле шкала одна, «cycle».
                if visual.timeline is not None and visual.time is not None and self._cycle is None:
                    rr.set_time(visual.timeline, sequence=visual.time)
                rr.log(visual.path, visual.payload, static=visual.static)
        rr.reset_time()

    def visual_image(self, path: str, image, *, static: bool = False) -> LogVisual:
        import numpy as np

        return LogVisual(path=path, payload=rr.Image(np.asarray(image)), static=static)

    def visual_points2d(
        self,
        path: str,
        positions: Sequence[Sequence[float]],
        *,
        colors: Sequence[Sequence[int]] | None = None,
        radii: float | Sequence[float] | None = None,
        static: bool = False,
    ) -> LogVisual:
        return LogVisual(
            path=path,
            payload=rr.Points2D(positions, colors=colors, radii=radii),
            static=static,
        )

    def visual_line_strips2d(
        self,
        path: str,
        strips: Sequence[Sequence[Sequence[float]]],
        *,
        colors: Sequence[Sequence[int]] | None = None,
        radii: float | Sequence[float] | None = None,
        static: bool = False,
    ) -> LogVisual:
        return LogVisual(
            path=path,
            payload=rr.LineStrips2D(strips, colors=colors, radii=radii),
            static=static,
        )

    def visual_bar_chart(self, path: str, values: Sequence[int]) -> LogVisual:
        return LogVisual(path=path, payload=rr.BarChart(list(values)))

    def visual_boxes2d(
        self,
        path: str,
        mins: Sequence[Sequence[float]],
        sizes: Sequence[Sequence[float]],
        *,
        colors: Sequence[Sequence[int]] | None = None,
    ) -> LogVisual:
        return LogVisual(path=path, payload=rr.Boxes2D(mins=mins, sizes=sizes, colors=colors))

    def visual_scalars(
        self,
        path: str,
        values: Mapping[str, float | int],
        *,
        step: int | float | None = None,
        timeline: str = "step",
    ) -> list[LogVisual]:
        visuals: list[LogVisual] = []
        for name, value in values.items():
            if value is None:
                continue
            visuals.append(
                LogVisual(
                    path=f"{path}/{name}",
                    payload=rr.Scalars(float(value)),
                    timeline=timeline if step is not None else None,
                    time=step,
                )
            )
        return visuals


LOGGER = DampLogger()
