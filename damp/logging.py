from __future__ import annotations

from typing import Any, Mapping

import rerun as rr

LOG_PATH = "damp/log"
_DEFAULT_APP_ID = "damp"
_RR_SPAWNED = False


def init_rerun(app_id: str = _DEFAULT_APP_ID, *, spawn: bool = True) -> None:
    global _RR_SPAWNED
    if rr.is_enabled():
        if spawn:
            _RR_SPAWNED = True
        return
    rr.init(app_id)
    if spawn:
        rr.spawn()
        _RR_SPAWNED = True


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
        parts = [f"{k}={_format_value(v)}" for k, v in items[:max_items]]
        if len(items) > max_items:
            parts.append("...")
        return "{" + ", ".join(parts) + "}"
    if isinstance(value, (list, tuple)):
        seq = list(value)
        parts = [_format_value(v) for v in seq[:max_items]]
        if len(seq) > max_items:
            parts.append("...")
        if isinstance(value, list):
            return "[" + ", ".join(parts) + "]"
        return "(" + ", ".join(parts) + ")"
    if isinstance(value, set):
        seq = list(value)
        parts = [_format_value(v) for v in seq[:max_items]]
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


def log_event(
    event: str,
    *,
    section: str,
    data: Mapping[str, Any] | None = None,
    path: str = LOG_PATH,
) -> None:
    if not section:
        raise ValueError("section must be provided")
    if data:
        details = " ".join(f"{k}={_format_value(v)}" for k, v in data.items())
        message = f"{event} {details}"
    else:
        message = event
    message = f"{message} [{section}]"
    print(message)
    init_rerun()
    rr.log(path, rr.TextLog(message))
