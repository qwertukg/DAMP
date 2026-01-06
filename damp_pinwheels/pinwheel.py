from __future__ import annotations

import argparse
import csv
import logging
import math
import os
from dataclasses import dataclass
from typing import Sequence

from .bit_array import BitArray
from .encoder import Encoder
from .layer_config import LayerConfig
from .layout import Layout

_logger = logging.getLogger(__name__)

ANGLE_COMPONENT_LAYERS = [
    LayerConfig(detectors=2, overlap=0.6),
    LayerConfig(detectors=4, overlap=0.4),
    LayerConfig(detectors=8, overlap=0.3),
    LayerConfig(detectors=16, overlap=0.2),
    LayerConfig(detectors=32, overlap=0.15),
    LayerConfig(detectors=64, overlap=0.1),
    LayerConfig(detectors=128, overlap=0.05),
]

RADIUS_LAYERS = [
    LayerConfig(detectors=2, overlap=0.6),
    LayerConfig(detectors=4, overlap=0.4),
    LayerConfig(detectors=8, overlap=0.3),
    LayerConfig(detectors=16, overlap=0.2),
    LayerConfig(detectors=32, overlap=0.1),
]


@dataclass(frozen=True)
class PinwheelPoint:
    index: int
    position_index: int
    angle_deg: float
    radius: float
    code: BitArray


class AnglePositionEncoder:
    """Component-wise encoding for cyclic angle and radial position."""

    def __init__(
        self,
        *,
        angle_layers: Sequence[LayerConfig] = ANGLE_COMPONENT_LAYERS,
        radius_layers: Sequence[LayerConfig] = RADIUS_LAYERS,
        bits_per_detector: int = 1,
        seed: int = 0,
    ) -> None:
        angle_bits = sum(layer.detectors for layer in angle_layers) * bits_per_detector
        radius_bits = sum(layer.detectors for layer in radius_layers) * bits_per_detector

        self._sin_encoder = Encoder(
            code_bits=angle_bits,
            layers=angle_layers,
            bits_per_detector=bits_per_detector,
            seed=seed,
            bit_assignment="detector_index",
            coordinate_mode="linear",
            value_range=(-1.0, 1.0),
        )
        self._cos_encoder = Encoder(
            code_bits=angle_bits,
            layers=angle_layers,
            bits_per_detector=bits_per_detector,
            seed=seed + 1,
            bit_assignment="detector_index",
            coordinate_mode="linear",
            value_range=(-1.0, 1.0),
        )
        self._radius_encoder = Encoder(
            code_bits=radius_bits,
            layers=radius_layers,
            bits_per_detector=bits_per_detector,
            seed=seed + 2,
            bit_assignment="detector_index",
            coordinate_mode="linear",
            value_range=(0.0, 1.0),
        )
        self.code_bits = angle_bits * 2 + radius_bits

    def encode(self, angle_deg: float, radius: float) -> BitArray:
        radians = math.radians(angle_deg)
        sin_value = math.sin(radians)
        cos_value = math.cos(radians)

        code = BitArray(self.code_bits)
        offset = 0
        for idx in self._sin_encoder.encode_sparse(sin_value):
            code.set(offset + idx)
        offset += self._sin_encoder.code_bits
        for idx in self._cos_encoder.encode_sparse(cos_value):
            code.set(offset + idx)
        offset += self._cos_encoder.code_bits
        for idx in self._radius_encoder.encode_sparse(radius):
            code.set(offset + idx)
        return code


def build_pinwheel_points(
    *,
    angle_count: int = 360,
    position_count: int = 5,
    encoder: AnglePositionEncoder | None = None,
) -> list[PinwheelPoint]:
    if angle_count <= 0:
        raise ValueError("angle_count must be positive")
    if position_count <= 0:
        raise ValueError("position_count must be positive")

    encoder = encoder or AnglePositionEncoder()
    if position_count == 1:
        radii = [0.0]
    else:
        radii = [idx / (position_count - 1) for idx in range(position_count)]

    points: list[PinwheelPoint] = []
    index = 0
    for pos_idx, radius in enumerate(radii):
        _logger.info(
            "encoding position %d/%d radius=%.3f",
            pos_idx + 1,
            position_count,
            radius,
        )
        for angle in range(angle_count):
            code = encoder.encode(float(angle), radius)
            points.append(
                PinwheelPoint(
                    index=index,
                    position_index=pos_idx,
                    angle_deg=float(angle),
                    radius=radius,
                    code=code,
                )
            )
            index += 1
    return points


def layout_pinwheels(
    points: Sequence[PinwheelPoint],
    *,
    grid_size: int | None = None,
    seed: int = 0,
    similarity: str = "cosine",
    long_steps: int = 8000,
    short_steps: int = 800,
    log_every: int | None = 100,
    snapshot_path: str | None = None,
    snapshot_every: int | None = None,
) -> Layout:
    codes = [(point.code, float(point.index), point.angle_deg) for point in points]
    layout = Layout(
        codes,
        grid_size=grid_size,
        similarity=similarity,
        lambda_threshold=0.65,
        eta=None,
        seed=seed,
    )
    pair_radius = layout.width // 2
    pairs_per_step = max(600, len(points) // 2)
    _logger.info(
        "layout long-range start steps=%d pairs_per_step=%d radius=%d",
        long_steps,
        pairs_per_step,
        pair_radius,
    )
    layout.run(
        steps=long_steps,
        pairs_per_step=pairs_per_step,
        pair_radius=pair_radius,
        mode="long",
        min_swap_ratio=0.0,
        energy_radius=7,
        energy_check_every=5,
        energy_delta=5e-4,
        energy_patience=4,
        log_every=log_every,
        log_prefix="layout",
        snapshot_path=snapshot_path,
        snapshot_every=snapshot_every,
    )
    _logger.info("layout long-range done steps=%d", layout.last_steps)
    layout.set_similarity_params(lambda_threshold=0.8, eta=None)
    pairs_per_step_short = max(300, len(points) // 4)
    _logger.info(
        "layout short-range start steps=%d pairs_per_step=%d radius=%d",
        short_steps,
        pairs_per_step_short,
        7,
    )
    layout.run(
        steps=short_steps,
        pairs_per_step=pairs_per_step_short,
        pair_radius=7,
        mode="short",
        local_radius=7,
        min_swap_ratio=0.005,
        log_every=log_every,
        log_prefix="layout",
        snapshot_path=snapshot_path,
        snapshot_every=snapshot_every,
    )
    _logger.info("layout short-range done steps=%d", layout.last_steps)
    return layout


def save_layout_csv(
    path: str, layout: Layout, points: Sequence[PinwheelPoint]
) -> None:
    positions = layout.positions()
    _logger.info("writing csv %s", path)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "x",
                "y",
                "position_index",
                "radius",
                "angle_deg",
            ]
        )
        for idx, (y, x) in enumerate(positions):
            point = points[idx]
            writer.writerow(
                [
                    point.index,
                    x,
                    y,
                    point.position_index,
                    f"{point.radius:.6f}",
                    f"{point.angle_deg:.1f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate sparse bit codes for 360 angles at several positions and "
            "layout them into a pinwheel-like map."
        )
    )
    parser.add_argument("--angles", type=int, default=360)
    parser.add_argument("--positions", type=int, default=5)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--long-steps", type=int, default=8000)
    parser.add_argument("--short-steps", type=int, default=800)
    parser.add_argument("--output", default="pinwheel_output")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--snapshot-every", type=int, default=100)
    parser.add_argument(
        "--similarity",
        choices=("cosine", "jaccard"),
        default="cosine",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    os.makedirs(args.output, exist_ok=True)
    ppm_path = os.path.join(args.output, "pinwheel_layout.ppm")
    csv_path = os.path.join(args.output, "pinwheel_layout.csv")

    _logger.info(
        "build codes angles=%d positions=%d",
        args.angles,
        args.positions,
    )
    points = build_pinwheel_points(
        angle_count=args.angles,
        position_count=args.positions,
    )
    _logger.info("build codes done total_points=%d", len(points))
    layout = layout_pinwheels(
        points,
        grid_size=args.grid_size,
        seed=args.seed,
        similarity=args.similarity,
        long_steps=args.long_steps,
        short_steps=args.short_steps,
        log_every=args.log_every,
        snapshot_path=ppm_path,
        snapshot_every=args.snapshot_every,
    )

    _logger.info("writing image %s", ppm_path)
    layout.write_ppm(ppm_path)
    save_layout_csv(csv_path, layout, points)

    _logger.info(
        "done wrote %s and %s with %d points on %dx%d grid",
        ppm_path,
        csv_path,
        len(points),
        layout.width,
        layout.height,
    )


if __name__ == "__main__":
    main()
