from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from damp_encoder import ClosedDimension, DetectorWindow, Detectors, Encoder, OpenedDimension

_FIG_STATE: Optional["FigureState"] = None


@dataclass
class FigureState:
    encoder_id: int
    fig: Figure
    axes: List[Axes]
    detectors_by_layer: DefaultDict[Tuple[int, int], List[DetectorWindow]]
    boundaries: List[int]
    layer_colors: Sequence[Tuple[float, float, float]]


def detector_segments(detector: DetectorWindow) -> List[Tuple[float, float]]:
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


def plot_code_bits(ax: Axes, bits: Sequence[int], boundaries: Sequence[int]) -> None:
    bit_values = [int(bit) for bit in bits]
    ax.imshow(
        [bit_values],
        cmap="Greys",
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    for boundary in boundaries[:-1]:
        ax.axvline(boundary - 0.5, color="tab:red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlim(-0.5, len(bit_values) - 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Bit index")
    ax.set_title(f"Code bits (len={len(bit_values)}, ones={sum(bit_values)})")


def plot_image_label(
    ax: Axes,
    img,
    label: int,
    *,
    patch_index: Tuple[int, int] | None = None,
) -> None:
    if img is not None:
        ax.imshow(img, cmap="gray", interpolation="nearest")
        if patch_index is not None:
            px, py = patch_index
            height, width = img.shape[:2]
            patch_size = 4
            if height % 7 == 0 and width % 7 == 0:
                patch_size = min(height // 7, width // 7)
            max_x = max(0, (width // patch_size) - 1)
            max_y = max(0, (height // patch_size) - 1)
            px = max(0, min(max_x, px))
            py = max(0, min(max_y, py))
            x0 = px * patch_size
            y0 = py * patch_size
            if 0 <= x0 < width and 0 <= y0 < height:
                ax.add_patch(
                    Rectangle(
                        (x0, y0),
                        patch_size,
                        patch_size,
                        edgecolor="red",
                        facecolor="none",
                        linewidth=1.5,
                    )
                )
    else:
        ax.text(0.5, 0.5, "No image", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Label: {label}")


def _format_value(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.3f}"


def plot_dimension_layers(
    ax: Axes,
    dimension_index: int,
    dimension,
    detectors_by_layer: DefaultDict[Tuple[int, int], List[DetectorWindow]],
    active_by_index: Sequence[bool],
    value: float,
    layer_colors: Sequence[Tuple[float, float, float]],
) -> None:
    layer_count = len(dimension.detector_layers)
    height = 0.8
    for layer_index in range(layer_count):
        detectors = detectors_by_layer.get((dimension_index, layer_index), [])
        color = layer_colors[layer_index % len(layer_colors)]
        y = layer_index
        for detector in detectors:
            segments = detector_segments(detector)
            alpha = 0.75 if active_by_index[detector.detector_index] else 0.25
            for start, end in segments:
                if end <= start:
                    continue
                ax.add_patch(
                    Rectangle(
                        (start, y - height / 2.0),
                        end - start,
                        height,
                        facecolor=color,
                        edgecolor=color,
                        linewidth=0.6,
                        alpha=alpha,
                    )
                )
    ax.axvline(value, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlim(dimension.min_value, dimension.max_value)
    ax.set_ylim(-0.5, layer_count - 0.5)
    ax.set_yticks(range(layer_count))
    ax.set_yticklabels(
        [f"Layer {index} (n={layer.count})" for index, layer in enumerate(dimension.detector_layers)]
    )
    ax.set_xlabel("Value")
    ax.set_ylabel("Layer")
    status = "closed" if dimension.closed else "open"
    encoded_value = _format_value(value)
    title = getattr(dimension, "title", f"Dimension {dimension_index}")
    ax.set_title(f"{title} ({status}) value={encoded_value}")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.5, alpha=0.6)


def _normalize_values(encoder: Encoder, values: Sequence[float]) -> Tuple[float, ...]:
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


def _find_dimension_index(encoder: Encoder, title: str) -> Optional[int]:
    title_lower = title.lower()
    for index, dimension in enumerate(encoder.dimensions):
        dim_title = getattr(dimension, "title", "")
        if str(dim_title).lower() == title_lower:
            return index
    return None


def _ensure_state(encoder: Encoder) -> FigureState:
    global _FIG_STATE
    if _FIG_STATE is not None and _FIG_STATE.encoder_id == id(encoder):
        return _FIG_STATE

    if _FIG_STATE is not None:
        plt.close(_FIG_STATE.fig)

    detectors_by_layer: DefaultDict[Tuple[int, int], List[DetectorWindow]] = defaultdict(list)
    for detector in encoder.detectors:
        detectors_by_layer[(detector.dimension_index, detector.layer_index)].append(detector)

    boundaries: List[int] = []
    total = 0
    for dimension in encoder.dimensions:
        total += sum(layer.count for layer in dimension.detector_layers)
        boundaries.append(total)

    fig, axes = plt.subplots(
        nrows=len(encoder.dimensions) + 2,
        figsize=(12, 3 * len(encoder.dimensions) + 4),
        constrained_layout=True,
    )
    axes_list = [axes] if isinstance(axes, Axes) else list(axes)

    plt.ion()
    plt.show(block=False)

    _FIG_STATE = FigureState(
        encoder_id=id(encoder),
        fig=fig,
        axes=axes_list,
        detectors_by_layer=detectors_by_layer,
        boundaries=boundaries,
        layer_colors=plt.get_cmap("tab20").colors,
    )
    return _FIG_STATE


def show(encoder: Encoder, values: Sequence[float], codes: Sequence[int], img, label: int) -> None:
    state = _ensure_state(encoder)
    normalized_values = _normalize_values(encoder, values)

    if len(codes) != encoder.code_length:
        raise ValueError("codes length must match encoder code length")

    active_by_index = [False] * len(encoder.detectors)
    for detector in encoder.detectors:
        dim_value = normalized_values[detector.dimension_index]
        active_by_index[detector.detector_index] = detector.is_active(dim_value)

    for ax in state.axes:
        ax.clear()

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

    plot_image_label(state.axes[0], img, label, patch_index=patch_index)

    for dim_index, dimension in enumerate(encoder.dimensions):
        plot_dimension_layers(
            state.axes[dim_index + 1],
            dim_index,
            dimension,
            state.detectors_by_layer,
            active_by_index,
            normalized_values[dim_index],
            state.layer_colors,
        )

    plot_code_bits(state.axes[-1], codes, state.boundaries)
    state.fig.canvas.draw_idle()
    state.fig.canvas.flush_events()
    plt.pause(0.001)


def wait_for_close() -> None:
    state = _FIG_STATE
    if state is None:
        return
    fig_number = state.fig.number
    while plt.fignum_exists(fig_number):
        plt.pause(0.1)
