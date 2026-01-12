from __future__ import annotations

from collections import defaultdict
import colorsys
from dataclasses import dataclass
from typing import DefaultDict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import rerun as rr

from damp.article_refs import CYCLIC_COORDS, MULTI_DIM_CODES, SPARSE_BIT_VECTORS
from damp.logging import init_rerun, log_event

if TYPE_CHECKING:
    from damp.encoding.damp_encoder import DetectorWindow, Encoder

DEFAULT_LOG_IMAGE_EVERY = 60
PATCH_GRID_SIZE = 7
DEFAULT_PATCH_SIZE = 4

_LAYER_PALETTE = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]


@dataclass
class EncoderState:
    encoder_id: int
    detectors_by_layer: DefaultDict[Tuple[int, int], List[DetectorWindow]]
    boundaries: List[int]


@dataclass
class LogState:
    image_index: int = 0
    last_image_id: int | None = None
    last_logged_image_index: int = 0
    call_index: int = 0


_ENCODER_STATE: Optional[EncoderState] = None
_DEFAULT_LOG_STATE = LogState()


def detector_segments(detector: "DetectorWindow") -> List[Tuple[float, float]]:
    if detector.closed:
        if detector.half_width >= detector.period / 2.0:
            return [(detector.span_min, detector.span_max)]
        low = detector.center - detector.half_width
        high = detector.center + detector.half_width
        if low < detector.span_min:
            return [(detector.span_min, high), (low + detector.period, detector.span_max)]
        if high > detector.span_max:
            return [(low, detector.span_max), (detector.span_min, high - detector.period)]
        return [(low, high)]
    low = max(detector.center - detector.half_width, detector.span_min)
    high = min(detector.center + detector.half_width, detector.span_max)
    return [(low, high)]


def _format_value(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.3f}"


def _normalize_values(encoder: "Encoder", values: Sequence[float]) -> Tuple[float, ...]:
    if len(values) != len(encoder.dimensions):
        raise ValueError("number of values must match number of dimensions")
    normalized: List[float] = []
    for value, dim in zip(values, encoder.dimensions):
        v = float(value)
        if dim.closed:
            v = ((v - dim.min_value) % dim.period) + dim.min_value
        else:
            if not (dim.min_value <= v <= dim.max_value):
                raise ValueError(
                    f"value {v} is outside the open dimension range "
                    f"[{dim.min_value}, {dim.max_value}]"
                )
        normalized.append(v)
    return tuple(normalized)


def _find_dimension_index(encoder: "Encoder", title: str) -> Optional[int]:
    title_lower = title.lower()
    for index, dimension in enumerate(encoder.dimensions):
        dim_title = getattr(dimension, "title", "")
        if str(dim_title).lower() == title_lower:
            return index
    return None


def _ensure_encoder_state(encoder: "Encoder") -> EncoderState:
    global _ENCODER_STATE
    if _ENCODER_STATE is not None and _ENCODER_STATE.encoder_id == id(encoder):
        return _ENCODER_STATE

    detectors_by_layer: DefaultDict[Tuple[int, int], List[DetectorWindow]] = defaultdict(list)
    for detector in encoder.detectors:
        detectors_by_layer[(detector.dimension_index, detector.layer_index)].append(detector)

    boundaries: List[int] = []
    total = 0
    for dimension in encoder.dimensions:
        total += sum(layer.count for layer in dimension.detector_layers)
        boundaries.append(total)

    _ENCODER_STATE = EncoderState(
        encoder_id=id(encoder),
        detectors_by_layer=detectors_by_layer,
        boundaries=boundaries,
    )
    return _ENCODER_STATE


def _ensure_rerun(app_id: str = "damp-encoding") -> None:
    init_rerun(app_id=app_id)


def _set_time_sequence(name: str, sequence: int) -> None:
    if hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence(name, sequence)
    else:
        rr.set_time(name, sequence=sequence)


def _next_log_state(state: LogState, img, log_every: int) -> tuple[int, bool]:
    state.call_index += 1
    if img is None:
        if log_every <= 1:
            return state.call_index, True
        if state.call_index % log_every != 0:
            return state.call_index, False
        return state.call_index, True

    image_id = id(img)
    if image_id != state.last_image_id:
        state.image_index += 1
        state.last_image_id = image_id

    if state.image_index == state.last_logged_image_index:
        return state.image_index, False

    interval = max(log_every, 1)
    if state.image_index % interval != 0:
        return state.image_index, False

    state.last_logged_image_index = state.image_index
    return state.image_index, True


def _infer_patch_grid(img_shape: Sequence[int]) -> tuple[int, int, int]:
    if len(img_shape) < 2:
        return 0, 0, 0
    height = int(img_shape[0])
    width = int(img_shape[1])
    if height <= 0 or width <= 0:
        return 0, 0, 0
    if height % PATCH_GRID_SIZE == 0 and width % PATCH_GRID_SIZE == 0:
        patch_size = min(height // PATCH_GRID_SIZE, width // PATCH_GRID_SIZE)
    else:
        patch_size = DEFAULT_PATCH_SIZE
    if patch_size <= 0:
        return 0, 0, 0
    patches_x = width // patch_size
    patches_y = height // patch_size
    if patches_x <= 0 or patches_y <= 0:
        return 0, 0, 0
    return patch_size, patches_x, patches_y


def _clamp_patch_index(patch_index: Tuple[int, int], patches_x: int, patches_y: int) -> Tuple[int, int]:
    px, py = patch_index
    px = max(0, min(patches_x - 1, px))
    py = max(0, min(patches_y - 1, py))
    return px, py


def _log_patch_grid(path: str, patch_size: int, patches_x: int, patches_y: int) -> None:
    mins: list[tuple[float, float]] = []
    sizes: list[tuple[float, float]] = []
    for py in range(patches_y):
        for px in range(patches_x):
            mins.append((px * patch_size, py * patch_size))
            sizes.append((patch_size, patch_size))
    if not mins:
        rr.log(path, rr.Boxes2D(mins=[], sizes=[]))
        return
    colors = [(255, 255, 255, 80)] * len(mins)
    rr.log(path, rr.Boxes2D(mins=mins, sizes=sizes, colors=colors))


def _angle_to_rgb(angle_deg: float) -> tuple[int, int, int]:
    hue = (angle_deg % 360.0) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def _log_measurements(
    path: str,
    values: Sequence[tuple[float, int, int]],
    patch_size: int,
    patches_x: int,
    patches_y: int,
) -> None:
    if not values:
        rr.log(path, rr.Points2D([]))
        return
    positions: list[tuple[float, float]] = []
    colors: list[tuple[int, int, int]] = []
    center = patch_size / 2.0
    for angle, x, y in values:
        px = max(0, min(patches_x - 1, int(x)))
        py = max(0, min(patches_y - 1, int(y)))
        positions.append((px * patch_size + center, py * patch_size + center))
        colors.append(_angle_to_rgb(float(angle)))
    rr.log(path, rr.Points2D(positions, colors=colors, radii=1.5))


def _log_active_patch(path: str, patch_index: Tuple[int, int], patch_size: int) -> None:
    x0 = patch_index[0] * patch_size
    y0 = patch_index[1] * patch_size
    rr.log(
        path,
        rr.Boxes2D(
            mins=[(x0, y0)],
            sizes=[(patch_size, patch_size)],
            colors=[(255, 0, 0, 200)],
        ),
    )


def _log_image(
    path: str,
    img,
    label: int | None,
    patch_index: Tuple[int, int] | None,
    measurements: Sequence[tuple[float, int, int]] | None,
) -> None:
    img_array = np.asarray(img)
    rr.log(path, rr.Image(img_array))
    if label is not None:
        rr.log(f"{path}/label", rr.AnyValues(label=int(label)))

    patch_size, patches_x, patches_y = _infer_patch_grid(img_array.shape)
    if patch_size:
        _log_patch_grid(f"{path}/patches", patch_size, patches_x, patches_y)
        if measurements is not None:
            _log_measurements(
                f"{path}/measurements",
                measurements,
                patch_size,
                patches_x,
                patches_y,
            )
        elif patch_index is not None:
            px, py = _clamp_patch_index(patch_index, patches_x, patches_y)
            _log_active_patch(f"{path}/active_patch", (px, py), patch_size)


def _layer_color(layer_index: int, active: bool) -> tuple[int, int, int, int]:
    base = _LAYER_PALETTE[layer_index % len(_LAYER_PALETTE)]
    alpha = 200 if active else 70
    return base[0], base[1], base[2], alpha


def _log_dimension_layers(
    path: str,
    dimension_index: int,
    dimension,
    detectors_by_layer: DefaultDict[Tuple[int, int], List[DetectorWindow]],
    active_by_index: Sequence[bool],
    value: float,
) -> None:
    layer_count = len(dimension.detector_layers)
    height = 0.8
    mins: list[tuple[float, float]] = []
    sizes: list[tuple[float, float]] = []
    colors: list[tuple[int, int, int, int]] = []
    for layer_index in range(layer_count):
        detectors = detectors_by_layer.get((dimension_index, layer_index), [])
        for detector in detectors:
            segments = detector_segments(detector)
            is_active = active_by_index[detector.detector_index]
            color = _layer_color(layer_index, is_active)
            for start, end in segments:
                if end <= start:
                    continue
                mins.append((start, layer_index - height / 2.0))
                sizes.append((end - start, height))
                colors.append(color)

    if mins:
        rr.log(f"{path}/segments", rr.Boxes2D(mins=mins, sizes=sizes, colors=colors))
    else:
        rr.log(f"{path}/segments", rr.Boxes2D(mins=[], sizes=[]))

    line = [(value, -0.5), (value, layer_count - 0.5)]
    rr.log(f"{path}/value", rr.LineStrips2D([line], colors=[(0, 0, 0, 200)]))

    status = "closed" if dimension.closed else "open"
    title = getattr(dimension, "title", f"Dimension {dimension_index}")
    rr.log(
        f"{path}/meta",
        rr.AnyValues(
            title=str(title),
            status=status,
            value=_format_value(value),
            min_value=float(dimension.min_value),
            max_value=float(dimension.max_value),
            layers=layer_count,
        ),
    )


def _log_code_bits(path: str, bits: Sequence[int], boundaries: Sequence[int]) -> None:
    bit_values = [int(bit) for bit in bits]
    rr.log(path, rr.BarChart(bit_values))
    rr.log(
        f"{path}/summary",
        rr.AnyValues(length=len(bit_values), ones=sum(bit_values), boundaries=list(boundaries)),
    )


def log_encoding(
    encoder: "Encoder",
    values: Sequence[float],
    codes: Sequence[int],
    img,
    label: int | None,
    *,
    log_every: int,
    state: LogState,
    measurements: Sequence[tuple[float, int, int]] | None = None,
    path_prefix: str = "encoding",
) -> None:
    _ensure_rerun()
    if log_every <= 0:
        return
    normalized_values = _normalize_values(encoder, values)
    if len(codes) != encoder.code_length:
        raise ValueError("codes length must match encoder code length")

    sequence, should_log = _next_log_state(state, img, log_every)
    if not should_log:
        return

    _set_time_sequence("sample", sequence)
    log_event(
        "encoding.visualize.input",
        section=MULTI_DIM_CODES,
        data={"values": values, "label": label, "sequence": sequence},
    )
    log_event(
        "encoding.visualize.normalized",
        section=CYCLIC_COORDS,
        data={"values": normalized_values},
    )
    log_event(
        "encoding.visualize.code",
        section=SPARSE_BIT_VECTORS,
        data={
            "code_length": encoder.code_length,
            "active_bits": sum(int(bit) for bit in codes),
        },
    )

    encoder_state = _ensure_encoder_state(encoder)
    active_by_index = [False] * len(encoder.detectors)
    for detector in encoder.detectors:
        dim_value = normalized_values[detector.dimension_index]
        active_by_index[detector.detector_index] = detector.is_active(dim_value)

    patch_index = None
    x_index = _find_dimension_index(encoder, "X")
    y_index = _find_dimension_index(encoder, "Y")
    if x_index is None or y_index is None:
        if len(normalized_values) >= 3:
            x_index, y_index = 1, 2
        else:
            x_index, y_index = None, None
    if x_index is not None and y_index is not None:
        x_value = int(round(normalized_values[x_index]))
        y_value = int(round(normalized_values[y_index]))
        patch_index = (x_value, y_value)

    if img is not None:
        _log_image(
            f"{path_prefix}/image",
            img,
            label,
            patch_index,
            measurements,
        )
    else:
        rr.log(f"{path_prefix}/label", rr.AnyValues(label=label))

    for dim_index, dimension in enumerate(encoder.dimensions):
        _log_dimension_layers(
            f"{path_prefix}/dimensions/D{dim_index}",
            dim_index,
            dimension,
            encoder_state.detectors_by_layer,
            active_by_index,
            normalized_values[dim_index],
        )

    _log_code_bits(f"{path_prefix}/code", codes, encoder_state.boundaries)


def show(encoder: "Encoder", values: Sequence[float], codes: Sequence[int], img, label: int) -> None:
    log_encoding(
        encoder,
        values,
        codes,
        img,
        label,
        log_every=DEFAULT_LOG_IMAGE_EVERY,
        state=_DEFAULT_LOG_STATE,
    )


def wait_for_close() -> None:
    return
