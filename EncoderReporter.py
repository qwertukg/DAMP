from __future__ import annotations

import math
from typing import Sequence

from BitArray import BitArray
from Encoder import Encoder


class EncoderReporter:
    def __init__(self, encoder: Encoder) -> None:
        self._encoder = encoder

    def visualize_detectors(self) -> None:
        """Show a Matplotlib visualization of detector arcs by layer."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Arc

        dpi = 100
        line_width_px = 1
        even_offset_px = 0.5
        layer_spacing_px = even_offset_px + line_width_px + 1
        min_arc_px = 1.0
        tick_length_px = 50
        tick_gap_px = 1
        label_gap_px = 10
        margin_px = 1

        layers = self._encoder.layers
        max_detectors = max(layer.detectors for layer in layers)
        inner_radius_px = max(
            12, int(math.ceil(min_arc_px * max_detectors / (2 * math.pi)))
        )
        tick_radius_px = max(1, inner_radius_px - tick_gap_px)
        layer_count = len(layers)
        outer_radius_px = (
            inner_radius_px + (layer_count - 1) * layer_spacing_px + even_offset_px
        )
        size_px = int(math.ceil((outer_radius_px + margin_px) * 2))
        line_width = line_width_px * 72.0 / dpi

        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#000000"])
        fig = plt.figure(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_aspect("equal")
        ax.axis("off")
        limit = outer_radius_px + margin_px
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)

        inner_circle = plt.Circle(
            (0.0, 0.0),
            radius=tick_radius_px,
            fill=False,
            linewidth=line_width,
            color="#aaaaaa",
        )
        ax.add_patch(inner_circle)
        font_size = max(4, min(9, int(round(tick_radius_px / 16))))
        label_radius_px = max(1, tick_radius_px - label_gap_px)
        for angle in range(360):
            radians = math.radians(angle)
            cos_a = math.cos(radians)
            sin_a = math.sin(radians)
            x0 = cos_a * tick_radius_px
            y0 = sin_a * tick_radius_px
            x1 = cos_a * (tick_radius_px + tick_length_px)
            y1 = sin_a * (tick_radius_px + tick_length_px)
            ax.plot([x0, x1], [y0, y1], color="#aaaaaa", linewidth=line_width)
            if angle % 10 == 0:
                lx = cos_a * label_radius_px
                ly = sin_a * label_radius_px
                ax.text(
                    lx,
                    ly,
                    str(angle),
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    color="#000000",
                )

        for layer_idx, layer in enumerate(layers):
            base_radius = inner_radius_px + layer_idx * layer_spacing_px
            arc_span = 360.0 / layer.detectors
            arc_length = arc_span * (1.0 + layer.overlap)
            color = colors[layer_idx % len(colors)]
            for detector_idx in range(layer.detectors):
                is_even = (detector_idx + 1) % 2 == 0
                radius = base_radius + (even_offset_px if is_even else 0)
                theta1 = detector_idx * arc_span
                theta2 = theta1 + arc_length
                arc = Arc(
                    (0.0, 0.0),
                    width=2 * radius,
                    height=2 * radius,
                    angle=0.0,
                    theta1=theta1,
                    theta2=theta2,
                    linewidth=line_width,
                    color=color,
                    capstyle="butt",
                )
                ax.add_patch(arc)

        plt.show()

    def print_codes(self, codes: Sequence[tuple[BitArray, float, float]]) -> None:
        """Print 0/1 codes with density and overlap for the given codes."""
        prev_code: BitArray | None = codes[-1][0] if codes else None
        for code, angle, _hue in codes:
            ones = code.count()
            density = (ones / len(code)) * 100
            if prev_code is None:
                common_pct = 0.0
            else:
                prev_ones = prev_code.count()
                common = code.common(prev_code)
                union = ones + prev_ones - common
                common_pct = (common / union) * 100 if union else 0.0
            code_str = code.to01()
            print(
                f"{angle:7.3f}: {code_str} density={density:.2f}% common={common_pct:.2f}%"
            )
            prev_code = code
