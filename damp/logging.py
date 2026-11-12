from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import sys
from typing import Any, Callable, Mapping, Sequence

import rerun as rr

LOG_PATH = "damp/log"


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
    # Детекторы-круги над раскладкой (PlaneDetectorSpace): зажжённые точки и активные круги.
    PLANE_DETECTION = "/plane_detection"
    PLANE_DETECTION_MAP = "plane_detection/map"
    PLANE_DETECTION_CIRCLES = "plane_detection/circles"

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
            rrb.Spatial2DView(name="Детекторы-круги над раскладкой (§4.3)", origin=cls.PLANE_DETECTION),
        )
        charts = rrb.Vertical(
            rrb.TimeSeriesView(name="Доля обменов (§5.6)", origin=cls.SWAP_RATIO),
            rrb.TimeSeriesView(name="Качество Ē (§5.8)", origin=cls.AVERAGE_ENERGY),
            rrb.TextLogView(name="Лог", origin=cls.TEXT_LOG),
            row_shares=[1, 1, 2],
        )
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(progress, results, row_shares=[3, 2]),
                charts,
                column_shares=[3, 2],
            ),
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
    def __init__(self, app_id: str = "damp", base_path: str = LOG_PATH, *, spawn: bool = True) -> None:
        self._app_id = app_id
        self._base_path = base_path
        self._spawn = spawn
        self._initialized = False
        self._spawned = False
        self._event_sequence = 0
        self._interval_policy: LogIntervalPolicy | None = None
        self._blueprint_builder: Callable[[], Any] = RerunViewerBlueprint.build
        # True — журнал выключен целиком (рабочие процессы: записи ведёт основной процесс).
        self._muted = False

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

    def use_blueprint(self, builder: Callable[[], Any]) -> None:
        """Своя раскладка панелей окна для отдельного скрипта; по умолчанию — ход раскладки DAMP.

        Если окно уже открыто, новая раскладка сразу становится активной.
        """
        self._blueprint_builder = builder
        if self._spawned:
            self._send_blueprint()

    def _ensure_rerun(self) -> None:
        if not self._initialized:
            rr.init(self._app_id)
            self._initialized = True
        if self._spawn and not self._spawned and rr.is_enabled():
            rr.spawn()
            self._spawned = True
            self._send_blueprint()

    def _send_blueprint(self) -> None:
        """Делает раскладку панелей DAMP активной, перекрывая сохранённую в окне с прошлых сессий."""
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
        if value is None:
            return "None"
        if isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (list, tuple)):
            if all(isinstance(item, (bool, int, float, str)) for item in value):
                return list(value)
            return DampLogger._format_value(value)
        if isinstance(value, dict):
            return DampLogger._format_value(value)
        return DampLogger._format_value(value)

    def _coerce_anyvalues(self, data: Mapping[str, Any]) -> Mapping[str, Any]:
        coerced: dict[str, Any] = {}
        for key, value in data.items():
            if value is None:
                continue
            if key == "label":
                coerced[key] = str(value)
                continue
            coerced[key] = self._coerce_rr_value(value)
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
        self._ensure_rerun()
        self._event_sequence += 1
        rr.reset_time()
        rr.set_time("event", sequence=self._event_sequence)
        base_path = path or self._base_path
        rr.log(base_path, rr.TextLog(message))
        if data:
            anyvalues_path = f"{base_path}/{event}"
            anyvalues = self._coerce_anyvalues(data)
            if anyvalues:
                rr.log(anyvalues_path, rr.AnyValues(**anyvalues))
        if visuals:
            for visual in visuals:
                rr.reset_time()
                if not visual.static:
                    rr.set_time("event", sequence=self._event_sequence)
                if visual.timeline is not None and visual.time is not None:
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
