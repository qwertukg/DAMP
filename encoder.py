from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal, Sequence

BitAssignment = Literal["random", "detector_index"]


@dataclass(frozen=True)
class LayerConfig:
    detectors: int
    overlap: float


class BitArray:
    def __init__(self, size: int, fill: int = 0) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        if fill not in (0, 1):
            raise ValueError("fill must be 0 or 1")
        self._bits = bytearray([fill] * size)

    def set(self, idx: int, value: int = 1) -> None:
        self._bits[idx] = 1 if value else 0

    def count(self) -> int:
        return sum(self._bits)

    def common(self, other: "BitArray") -> int:
        if len(self) != len(other):
            raise ValueError("BitArray sizes must match")
        return sum(a & b for a, b in zip(self._bits, other._bits))

    def to01(self) -> str:
        return "".join("1" if b else "0" for b in self._bits)

    def __len__(self) -> int:
        return len(self._bits)


class Encoder:
    """Encode angles with layered detectors that overlap by a fraction of their arc length."""

    def __init__(
        self,
        code_bits: int,
        layers: Sequence[LayerConfig],
        *,
        bits_per_detector: int,
        seed: int,
        bit_assignment: BitAssignment = "random",
    ) -> None:
        if code_bits <= 0:
            raise ValueError("code_bits must be positive")
        if bits_per_detector <= 0:
            raise ValueError("bits_per_detector must be positive")
        if bits_per_detector > code_bits:
            raise ValueError("bits_per_detector must be <= code_bits")

        self.code_bits = code_bits
        self.bits_per_detector = bits_per_detector
        self.layers = list(layers)
        if not self.layers:
            raise ValueError("layers must be non-empty")
        if bit_assignment not in ("random", "detector_index"):
            raise ValueError("bit_assignment must be 'random' or 'detector_index'")
        self.bit_assignment = bit_assignment

        self._rng = random.Random(seed)
        self._layer_bits = self._init_layer_bits()

    def encode_sparse(self, angle: float) -> list[int]:
        """Return indices of active bits for the given angle (degrees)."""
        bits: set[int] = set()
        for layer, detector_bits in zip(self.layers, self._layer_bits):
            for detector_index in self._active_detectors(angle, layer):
                bits.update(detector_bits[detector_index])
        return sorted(bits)

    def encode_dense(self, angle: float) -> BitArray:
        """Return a BitArray with only the active bits set."""
        bits = BitArray(self.code_bits)
        for idx in self.encode_sparse(angle):
            bits.set(idx)
        return bits

    def encode(
        self, start: float, end: float, step: float
    ) -> list[tuple[BitArray, float, float]]:
        """Return (code, angle, hue) tuples for angles in the inclusive range [start, end]."""
        if step <= 0:
            raise ValueError("step must be positive")

        angle = start
        epsilon = step * 1e-9
        codes: list[tuple[BitArray, float, float]] = []
        while angle <= end + epsilon:
            code = self.encode_dense(angle)
            hue = angle % 360.0
            codes.append((code, angle, hue))
            angle += step
        return codes

    def visualize_detectors(self) -> None:
        """Show a Matplotlib visualization of detector arcs by layer."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Arc

        dpi = 100
        line_width_px = 1
        even_offset_px = 2
        layer_spacing_px = even_offset_px + line_width_px + 1
        min_arc_px = 2.0
        tick_length_px = 10
        tick_gap_px = 4
        label_gap_px = 10
        margin_px = 4

        max_detectors = max(layer.detectors for layer in self.layers)
        inner_radius_px = max(
            12, int(math.ceil(min_arc_px * max_detectors / (2 * math.pi)))
        )
        tick_radius_px = max(1, inner_radius_px - tick_gap_px)
        layer_count = len(self.layers)
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
            color="#000000",
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
            ax.plot([x0, x1], [y0, y1], color="#000000", linewidth=line_width)
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

        for layer_idx, layer in enumerate(self.layers):
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

    def print_codes(
        self,
        codes: list[tuple[BitArray, float, float]],
    ) -> None:
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

    def _init_layer_bits(self) -> list[list[list[int]]]:
        layers_bits: list[list[list[int]]] = []
        layer_offset = 0
        for layer in self.layers:
            if layer.detectors <= 0:
                raise ValueError("detectors must be positive")
            if layer.overlap < 0:
                raise ValueError("overlap must be >= 0")

            layer_bits: list[list[int]] = []
            for detector_index in range(layer.detectors):
                if self.bit_assignment == "random":
                    bits = self._rng.sample(
                        range(self.code_bits), self.bits_per_detector
                    )
                else:
                    base = layer_offset + detector_index * self.bits_per_detector
                    if base + self.bits_per_detector > self.code_bits:
                        raise ValueError(
                            "code_bits too small for detector_index assignment"
                        )
                    bits = [base + offset for offset in range(self.bits_per_detector)]
                layer_bits.append(bits)
            layers_bits.append(layer_bits)
            if self.bit_assignment == "detector_index":
                layer_offset += layer.detectors * self.bits_per_detector
        return layers_bits

    @staticmethod
    def _active_detectors(angle: float, layer: LayerConfig) -> list[int]:
        detectors = layer.detectors
        overlap = layer.overlap
        normalized = angle % 360.0
        position = normalized / 360.0 * detectors
        span = 1.0 + overlap
        if span <= 0.0:
            return []
        if span >= detectors:
            return list(range(detectors))
        # Detectors cover [j, j + span) in detector units; collect all j that cover position.
        max_idx = math.floor(position + 1e-9)
        start_idx = math.floor(position - span + 1e-12) + 1
        return [(idx % detectors) for idx in range(start_idx, max_idx + 1)]
