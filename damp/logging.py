from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Any, Collection, Mapping, Sequence

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


class RerunDashboard:
    """Строит устойчивые рабочие экраны для основных команд DAMP."""

    @staticmethod
    def build(profile: str):
        import rerun.blueprint as rrb

        if profile == "layout":
            dashboard = rrb.Grid(
                rrb.Spatial2DView(
                    origin="layout/progress",
                    contents="layout/progress/**",
                    name="Ход раскладки",
                ),
                rrb.Spatial2DView(
                    origin="layout/final",
                    contents="layout/final/**",
                    name="Итоговая раскладка",
                ),
                rrb.Spatial2DView(
                    origin="activation/latest",
                    contents="activation/latest/**",
                    name="Последняя активация",
                    background=(0, 0, 0, 255),
                ),
                rrb.Spatial2DView(
                    origin="layout/diagnostics/local_energy",
                    contents="layout/diagnostics/local_energy/**",
                    name="Локальная энергия",
                    background=(0, 0, 0, 255),
                ),
                rrb.TimeSeriesView(
                    origin="layout/metrics/swap",
                    contents="layout/metrics/swap/**",
                    name="Принятые свопы",
                ),
                rrb.TimeSeriesView(
                    origin="layout/metrics/energy",
                    contents="layout/metrics/energy/**",
                    name="Динамика энергии",
                ),
                rrb.TimeSeriesView(
                    origin="layout/metrics/parameters",
                    contents="layout/metrics/parameters/**",
                    name="Адаптивные параметры",
                ),
                rrb.TimeSeriesView(
                    origin="layout/metrics/continuity",
                    contents="layout/metrics/continuity/**",
                    name="Непрерывность карты",
                ),
                rrb.Spatial2DView(
                    origin="layout/diagnostics/detectors",
                    contents="layout/diagnostics/detectors/**",
                    name="Иерархия детекторов",
                    background=(0, 0, 0, 255),
                ),
                rrb.TimeSeriesView(
                    origin="activation/metrics/map",
                    contents="activation/metrics/map/**",
                    name="Динамика активации",
                ),
                rrb.TimeSeriesView(
                    origin="activation/metrics/embedding",
                    contents="activation/metrics/embedding/**",
                    name="Динамика эмбеддинга",
                ),
                rrb.TextLogView(
                    origin="damp/log",
                    contents="damp/log",
                    name="Ключевые события",
                ),
                grid_columns=4,
                name="DAMP layout",
            )
            return rrb.Blueprint(
                dashboard,
                auto_views=False,
                collapse_panels=True,
            )

        if profile == "evaluate":
            dashboard = rrb.Grid(
                rrb.Spatial2DView(
                    origin="evaluate/confusion",
                    contents="evaluate/confusion/**",
                    name="Матрица ошибок",
                ),
                rrb.Spatial2DView(
                    origin="evaluate/errors",
                    contents="evaluate/errors/**",
                    name="Ошибочные примеры",
                ),
                rrb.TimeSeriesView(
                    origin="evaluate/class_accuracy",
                    contents="evaluate/class_accuracy/**",
                    name="Точность по классам",
                ),
                rrb.Spatial2DView(
                    origin="activation/latest",
                    contents="activation/latest/**",
                    name="Последняя активация",
                    background=(0, 0, 0, 255),
                ),
                rrb.Spatial2DView(
                    origin="layout/diagnostics/local_energy",
                    contents="layout/diagnostics/local_energy/**",
                    name="Локальная энергия",
                    background=(0, 0, 0, 255),
                ),
                rrb.Spatial2DView(
                    origin="layout/diagnostics/detectors",
                    contents="layout/diagnostics/detectors/**",
                    name="Детекторы",
                ),
                rrb.TimeSeriesView(
                    origin="evaluate/metrics",
                    contents="evaluate/metrics/**",
                    name="Точность и уверенность",
                ),
                rrb.TimeSeriesView(
                    origin="evaluate/confidence",
                    contents="evaluate/confidence/**",
                    name="Динамика уверенности",
                ),
                rrb.TextLogView(
                    origin="damp/log",
                    contents="damp/log",
                    name="Ключевые события",
                ),
                grid_columns=3,
                name="DAMP evaluate",
            )
            return rrb.Blueprint(
                dashboard,
                auto_views=False,
                collapse_panels=True,
            )
        return None


class DampLogger:
    def __init__(self, app_id: str = "damp", base_path: str = LOG_PATH, *, spawn: bool = True) -> None:
        self._app_id = app_id
        self._base_path = base_path
        self._spawn = spawn
        self._initialized = False
        self._spawned = False
        self._event_sequence = 0
        self._interval_policy: LogIntervalPolicy | None = None
        self._rerun_events: set[str] | None = None
        self._dashboard_profile: str | None = None
        self._blueprint_sent = False

    def _ensure_rerun(self) -> None:
        if not self._initialized:
            rr.init(self._app_id)
            self._initialized = True
        if self._spawn and not self._spawned and rr.is_enabled():
            rr.spawn()
            self._spawned = True
        if rr.is_enabled() and not self._blueprint_sent and self._dashboard_profile:
            blueprint = RerunDashboard.build(self._dashboard_profile)
            if blueprint is not None:
                rr.send_blueprint(blueprint)
            self._blueprint_sent = True

    def _console_log(self, message: str) -> None:
        stream = sys.stdout
        try:
            print(message, file=stream)
        except UnicodeEncodeError:
            encoding = getattr(stream, "encoding", None) or "utf-8"
            safe_message = message.encode(encoding, errors="backslashreplace").decode(encoding)
            print(safe_message, file=stream)

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

    def configure_rerun(
        self,
        *,
        events: Collection[str] | None,
        dashboard_profile: str | None,
    ) -> None:
        self._rerun_events = None if events is None else {str(event) for event in events}
        self._dashboard_profile = dashboard_profile
        self._blueprint_sent = False

    def should_log(self, event: str) -> bool:
        if self._interval_policy is None:
            return True
        return self._interval_policy.should_log(event)

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
        if not rr.is_enabled():
            return
        rerun_event = (
            self._rerun_events is None
            or event in self._rerun_events
            or event.endswith((".error", ".failed"))
        )
        if not rerun_event:
            return
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

    def visual_image(
        self,
        path: str,
        image,
        *,
        static: bool = False,
        timeline: str | None = None,
        time: int | float | None = None,
    ) -> LogVisual:
        import numpy as np

        return LogVisual(
            path=path,
            payload=rr.Image(np.asarray(image)),
            timeline=timeline,
            time=time,
            static=static,
        )

    def visual_points2d(
        self,
        path: str,
        positions: Sequence[Sequence[float]],
        *,
        colors: Sequence[Sequence[int]] | None = None,
        radii: float | Sequence[float] | None = None,
        labels: Sequence[str] | None = None,
        show_labels: bool | None = None,
        static: bool = False,
        timeline: str | None = None,
        time: int | float | None = None,
    ) -> LogVisual:
        return LogVisual(
            path=path,
            payload=rr.Points2D(
                positions,
                colors=colors,
                radii=radii,
                labels=labels,
                show_labels=show_labels,
            ),
            timeline=timeline,
            time=time,
            static=static,
        )

    def visual_bar_chart(
        self,
        path: str,
        values: Sequence[int | float],
        *,
        static: bool = False,
    ) -> LogVisual:
        return LogVisual(path=path, payload=rr.BarChart(list(values)), static=static)

    def visual_boxes2d(
        self,
        path: str,
        mins: Sequence[Sequence[float]],
        sizes: Sequence[Sequence[float]],
        *,
        colors: Sequence[Sequence[int]] | None = None,
        static: bool = False,
    ) -> LogVisual:
        return LogVisual(
            path=path,
            payload=rr.Boxes2D(mins=mins, sizes=sizes, colors=colors),
            static=static,
        )

    def visual_scalar_map(
        self,
        path: str,
        values: Sequence[float],
        positions: Sequence[Sequence[int]],
        *,
        height: int,
        width: int,
        static: bool = False,
    ) -> LogVisual:
        import numpy as np

        matrix = np.zeros((height, width), dtype=np.float32)
        for value, position in zip(values, positions):
            y, x = int(position[0]), int(position[1])
            if 0 <= y < height and 0 <= x < width:
                matrix[y, x] = float(value)
        maximum = float(matrix.max()) if matrix.size else 0.0
        if maximum > 0.0:
            matrix /= maximum
        intensity = np.asarray(
            np.clip(matrix * 255.0, 0.0, 255.0), dtype=np.uint8
        )
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[..., 0] = intensity
        return self.visual_image(path, image, static=static)

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
