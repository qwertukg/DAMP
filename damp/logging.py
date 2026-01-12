from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import rerun as rr

LOG_PATH = "damp/log"


@dataclass(frozen=True)
class LogVisual:
    path: str
    payload: object


class DampLogger:
    def __init__(self, app_id: str = "damp", base_path: str = LOG_PATH, *, spawn: bool = True) -> None:
        self._app_id = app_id
        self._base_path = base_path
        self._spawn = spawn
        self._spawned = False

    def _ensure_rerun(self) -> None:
        if rr.is_enabled():
            if self._spawn:
                self._spawned = True
            return
        rr.init(self._app_id)
        if self._spawn:
            rr.spawn()
            self._spawned = True

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
        return {key: self._coerce_rr_value(value) for key, value in data.items()}

    def event(
        self,
        event: str,
        *,
        section: str,
        data: Mapping[str, Any] | None = None,
        path: str | None = None,
        visuals: Sequence[LogVisual] | None = None,
    ) -> None:
        if not section:
            raise ValueError("section must be provided")
        if data:
            details = " ".join(f"{k}={self._format_value(v)}" for k, v in data.items())
            message = f"{event} {details}"
        else:
            message = event
        message = f"{message} [{section}]"
        print(message)
        self._ensure_rerun()
        base_path = path or self._base_path
        rr.log(base_path, rr.TextLog(message))
        if data:
            rr.log(f"{base_path}/{event}", rr.AnyValues(**self._coerce_anyvalues(data)))
        if visuals:
            for visual in visuals:
                rr.log(visual.path, visual.payload)

    def visual_image(self, path: str, image) -> LogVisual:
        import numpy as np

        return LogVisual(path=path, payload=rr.Image(np.asarray(image)))

    def visual_points2d(
        self,
        path: str,
        positions: Sequence[Sequence[float]],
        *,
        colors: Sequence[Sequence[int]] | None = None,
        radii: float | Sequence[float] | None = None,
    ) -> LogVisual:
        return LogVisual(path=path, payload=rr.Points2D(positions, colors=colors, radii=radii))

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


LOGGER = DampLogger()
