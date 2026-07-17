from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Sequence

from main import (
    ENCODER_LOG_EVERY,
    LAYOUT_ADAPTIVE_RADIUS_MIN,
    LAYOUT_ADAPTIVE_RADIUS_START_FACTOR,
    LAYOUT_ADAPTIVE_SWAP_TRIGGER,
    LAYOUT_ENERGY_STABILITY_DELTA,
    LAYOUT_ENERGY_STABILITY_EVERY,
    LAYOUT_ENERGY_STABILITY_MAX_POINTS,
    LAYOUT_ENERGY_STABILITY_WINDOW,
    LAYOUT_LOG_EVERY_LONG,
    LAYOUT_LOG_EVERY_SHORT,
    LAYOUT_LOG_VISUALS,
    LAYOUT_SHORT_ENERGY_EPS,
    LAYOUT_SHORT_ENERGY_MAX_POINTS,
    LAYOUT_SHORT_ENERGY_RECALC_EVERY,
    LAYOUT_SHORT_WEIGHTED_FIRST,
    PINWHEEL_ANGLE_MAX,
    PINWHEEL_ANGLE_MIN,
    PINWHEEL_ANGLE_COUNT,
    PINWHEEL_ADAPTIVE_LAMBDA_STEP,
    PINWHEEL_EMPTY_RATIO,
    PINWHEEL_ETA,
    PINWHEEL_LAMBDA_THRESHOLD,
    PINWHEEL_LAYOUT_PATH,
    PINWHEEL_LONG_JOINT_LAYERS,
    PINWHEEL_LONG_ANGLE_LAYERS,
    PINWHEEL_LONG_LAYOUT_PATH,
    PINWHEEL_LONG_PAIRS_PER_STEP,
    PINWHEEL_LONG_STEPS,
    PINWHEEL_MIN_SWAP_RATIO,
    PINWHEEL_MIN_SWAP_WINDOW,
    PINWHEEL_OUTPUT_PATH,
    PINWHEEL_POSITION_MAX,
    PINWHEEL_POSITION_MIN,
    PINWHEEL_PSEUDO_COLOR_MAX_HUE,
    PINWHEEL_PSEUDO_COLOR_MIN_HUE,
    PINWHEEL_SEED,
    PINWHEEL_RING_RADIUS,
    PINWHEEL_SHORT_JOINT_LAYERS,
    PINWHEEL_SHORT_ANGLE_LAYERS,
    PINWHEEL_SHORT_LOCAL_RADIUS_FACTOR,
    PINWHEEL_SHORT_PAIRS_PER_STEP,
    PINWHEEL_SHORT_PARTITIONS,
    PINWHEEL_SHORT_STEPS,
    PINWHEEL_SIMILARITY,
    PINWHEEL_SINGULARITY_COORDINATES,
    PINWHEEL_SMOOTH_DELTA,
    configure_logging,
)
from damp.article_refs import (
    COMPONENT_ASSIGNMENT,
    LAID_OUT_STRUCTURE,
    LAYOUT_BIO_MOTIVATION,
    MAPPING_CONTINUITY,
)
from damp.encoding.bitarray import BitArray
from damp.encoding.damp_encoder import ClosedDimension, Detectors, Encoder, OpenedDimension
from damp.encoding.joint_encoder import JointDetectorLayer, JointEncoder
from damp.layout.damp_layout import AdaptiveLayoutConfig, Layout, ShortLayoutOptimConfig
from damp.layout.payload import LayoutEncodedCode
from damp.logging import LOGGER


@dataclass(frozen=True)
class RetinotopicSample:
    index: int
    position_index: int
    angle_index: int
    x: float
    y: float
    orientation: float

    @property
    def label(self) -> str:
        return (
            f"sample={self.index};position={self.position_index};"
            f"angle_index={self.angle_index};x={self.x:.9f};y={self.y:.9f};"
            f"orientation={self.orientation:.9f}"
        )


class ContinuousRetinotopicSurface:
    """Nine connected cyclic fibres with 360 orientations each."""

    def __init__(self) -> None:
        self._centres = tuple(float(value) for value in PINWHEEL_SINGULARITY_COORDINATES)
        self._samples = self._build_samples()
        self._positions = {sample.label: (sample.x, sample.y) for sample in self._samples}
        LOGGER.event(
            "pinwheel.surface.init",
            section=MAPPING_CONTINUITY,
            data={
                "samples": len(self._samples),
                "positions": len(self._centres) ** 2,
                "orientations_per_position": PINWHEEL_ANGLE_COUNT,
                "singularities": len(self._centres) ** 2,
                "singularity_coordinates": self._centres,
                "continuous": True,
            },
        )

    @property
    def samples(self) -> Sequence[RetinotopicSample]:
        return self._samples

    def position(self, label: str) -> tuple[float, float]:
        return self._positions[label]

    def _build_samples(self) -> tuple[RetinotopicSample, ...]:
        samples: list[RetinotopicSample] = []
        period = PINWHEEL_ANGLE_MAX - PINWHEEL_ANGLE_MIN
        for row, centre_y in enumerate(self._centres):
            for column, centre_x in enumerate(self._centres):
                position_index = row * len(self._centres) + column
                chirality = 1.0 if (row + column) % 2 == 0 else -1.0
                phase = (column % 2) * period / 2.0
                for angle_index in range(PINWHEEL_ANGLE_COUNT):
                    orientation = (
                        PINWHEEL_ANGLE_MIN
                        + period * angle_index / PINWHEEL_ANGLE_COUNT
                    )
                    topological_orientation = (
                        chirality * orientation + phase
                    ) % period
                    polar_angle = math.radians(2.0 * topological_orientation)
                    x = centre_x + PINWHEEL_RING_RADIUS * math.cos(polar_angle)
                    y = centre_y + PINWHEEL_RING_RADIUS * math.sin(polar_angle)
                    LOGGER.event(
                        "pinwheel.surface.orientation",
                        section=MAPPING_CONTINUITY,
                        data={
                            "position": position_index,
                            "angle_index": angle_index,
                            "orientation": orientation,
                            "topological_orientation": topological_orientation,
                            "chirality": chirality,
                            "phase": phase,
                            "x": x,
                            "y": y,
                        },
                    )
                    samples.append(
                        RetinotopicSample(
                            index=len(samples),
                            position_index=position_index,
                            angle_index=angle_index,
                            x=x,
                            y=y,
                            orientation=orientation,
                        )
                    )
        return tuple(samples)


class RetinotopicEncoder:
    """Union of a joint spatial code and a cyclic orientation code."""

    def __init__(self, spatial_encoder: Encoder, orientation_encoder: Encoder) -> None:
        self._spatial_encoder = spatial_encoder
        self._orientation_encoder = orientation_encoder

    def _encode_components(
        self,
        orientation: float,
        x: float,
        y: float,
    ) -> tuple[BitArray, BitArray]:
        _, spatial_code = self._spatial_encoder.encode(x, y)
        _, orientation_code = self._orientation_encoder.encode(orientation)
        code = BitArray(len(spatial_code) + len(orientation_code))
        for index, bit in enumerate(spatial_code):
            if bit:
                code.set(index)
        offset = len(spatial_code)
        for index, bit in enumerate(orientation_code):
            if bit:
                code.set(offset + index)
        LOGGER.event(
            "pinwheel.encoder.code_union",
            section=COMPONENT_ASSIGNMENT,
            data={
                "orientation": orientation,
                "x": x,
                "y": y,
                "spatial_bits": spatial_code.count(),
                "orientation_bits": orientation_code.count(),
                "active_bits": code.count(),
                "code_length": len(code),
            },
        )
        return code, orientation_code

    def encode(self, orientation: float, x: float, y: float) -> tuple[tuple[float, ...], BitArray]:
        code, _ = self._encode_components(orientation, x, y)
        return (orientation, x, y), code

    def encode_colored(
        self,
        orientation: float,
        x: float,
        y: float,
    ) -> tuple[tuple[float, ...], BitArray, tuple[int, int, int]]:
        code, orientation_code = self._encode_components(orientation, x, y)
        color = Encoder.color_from_code(
            orientation_code,
            min_hue=PINWHEEL_PSEUDO_COLOR_MIN_HUE,
            max_hue=PINWHEEL_PSEUDO_COLOR_MAX_HUE,
        )
        return (orientation, x, y), code, color


class V1PinwheelExperiment:
    """DAMP layout of one continuous retinotopic orientation surface."""

    def __init__(
        self,
        surface: ContinuousRetinotopicSurface,
        long_encoder: RetinotopicEncoder,
        short_encoder: RetinotopicEncoder,
    ) -> None:
        self._surface = surface
        self._long_encoder = long_encoder
        self._short_encoder = short_encoder

    @staticmethod
    def build_encoder(
        layers: Sequence[tuple[tuple[int, ...], tuple[float, ...]]],
        angle_layers: Sequence[tuple[int, float]],
    ) -> RetinotopicEncoder:
        spatial_encoder = JointEncoder(
            OpenedDimension(
                "Retinotopic X",
                (PINWHEEL_POSITION_MIN, PINWHEEL_POSITION_MAX),
                [Detectors(1, 0.0)],
            ),
            OpenedDimension(
                "Retinotopic Y",
                (PINWHEEL_POSITION_MIN, PINWHEEL_POSITION_MAX),
                [Detectors(1, 0.0)],
            ),
            joint_layers=[
                JointDetectorLayer(counts=counts, overlaps=overlaps)
                for counts, overlaps in layers
            ],
            log_every=ENCODER_LOG_EVERY,
        )
        orientation_encoder = Encoder(
            ClosedDimension(
                "Orientation",
                (PINWHEEL_ANGLE_MIN, PINWHEEL_ANGLE_MAX),
                [Detectors(count, overlap) for count, overlap in angle_layers],
            ),
            log_every=ENCODER_LOG_EVERY,
        )
        return RetinotopicEncoder(spatial_encoder, orientation_encoder)

    def _encode_codes(
        self,
        encoder: RetinotopicEncoder,
    ) -> dict[float, list[LayoutEncodedCode]]:
        codes: dict[float, list[LayoutEncodedCode]] = defaultdict(list)
        for sample in self._surface.samples:
            _, code, color = encoder.encode_colored(
                sample.orientation,
                sample.x,
                sample.y,
            )
            codes[sample.orientation].append(
                LayoutEncodedCode(code=code, label=sample.label, color=color)
            )
        return codes

    @staticmethod
    def _adaptive_config(start_radius: int) -> AdaptiveLayoutConfig:
        return AdaptiveLayoutConfig(
            start_radius=start_radius,
            end_radius=LAYOUT_ADAPTIVE_RADIUS_MIN,
            swap_ratio_trigger=LAYOUT_ADAPTIVE_SWAP_TRIGGER,
            lambda_step=PINWHEEL_ADAPTIVE_LAMBDA_STEP,
        )

    @staticmethod
    def _orientation_delta(first: float, second: float) -> float:
        period = PINWHEEL_ANGLE_MAX - PINWHEEL_ANGLE_MIN
        delta = abs(first - second) % period
        return min(delta, period - delta)

    def _log_v1_quality(self, layout: Layout) -> None:
        deltas: list[float] = []
        retinal_distances: list[float] = []
        detected_singularities = 0
        source_step = 2.0 * PINWHEEL_RING_RADIUS * math.sin(
            math.pi / PINWHEEL_ANGLE_COUNT
        )
        for y, row in enumerate(layout.grid):
            for x, point_index in enumerate(row):
                if point_index is None:
                    continue
                for neighbour_y, neighbour_x in ((y, x + 1), (y + 1, x)):
                    if neighbour_y >= layout.height or neighbour_x >= layout.width:
                        continue
                    neighbour_index = layout.grid[neighbour_y][neighbour_x]
                    if neighbour_index is None:
                        continue
                    first = layout._points[point_index]
                    second = layout._points[neighbour_index]
                    deltas.append(self._orientation_delta(first.hue, second.hue))
                    first_position = self._surface.position(first.label)
                    second_position = self._surface.position(second.label)
                    retinal_distances.append(
                        math.hypot(
                            first_position[0] - second_position[0],
                            first_position[1] - second_position[1],
                        )
                    )
        for y in range(layout.height - 1):
            for x in range(layout.width - 1):
                indices = (
                    layout.grid[y][x],
                    layout.grid[y][x + 1],
                    layout.grid[y + 1][x + 1],
                    layout.grid[y + 1][x],
                )
                if any(index is None for index in indices):
                    continue
                phases = [
                    math.radians(2.0 * layout._points[index].hue)
                    for index in indices
                    if index is not None
                ]
                winding = 0.0
                for index, phase in enumerate(phases):
                    delta = phases[(index + 1) % len(phases)] - phase
                    winding += (delta + math.pi) % (2.0 * math.pi) - math.pi
                if abs(winding) > math.pi:
                    detected_singularities += 1
        LOGGER.event(
            "pinwheel.v1.quality",
            section=LAYOUT_BIO_MOTIVATION,
            data={
                "connected_cyclic_surface": True,
                "samples": len(self._surface.samples),
                "positions": len(PINWHEEL_SINGULARITY_COORDINATES) ** 2,
                "orientations_per_position": PINWHEEL_ANGLE_COUNT,
                "pinwheel_singularities": len(PINWHEEL_SINGULARITY_COORDINATES) ** 2,
                "detected_pinwheel_singularities": detected_singularities,
                "edges": len(deltas),
                "mean_orientation_delta": sum(deltas) / len(deltas),
                "max_orientation_delta": max(deltas),
                "sharp_transition_ratio": sum(
                    delta > PINWHEEL_SMOOTH_DELTA for delta in deltas
                )
                / len(deltas),
                "mean_input_distance": sum(retinal_distances) / len(retinal_distances),
                "retinotopic_local_ratio": sum(
                    distance <= math.sqrt(2.0) * source_step * 1.01
                    for distance in retinal_distances
                )
                / len(retinal_distances),
            },
        )

    @staticmethod
    def _save_image(layout: Layout) -> None:
        from PIL import Image

        image = layout.render_image(log=True)
        Image.fromarray(image).save(PINWHEEL_OUTPUT_PATH)
        layout.save_json(PINWHEEL_LAYOUT_PATH)
        LOGGER.event(
            "pinwheel.v1.image.save",
            section=LAID_OUT_STRUCTURE,
            data={
                "path": PINWHEEL_OUTPUT_PATH,
                "layout_path": PINWHEEL_LAYOUT_PATH,
                "width": layout.width,
                "height": layout.height,
            },
        )

    def run(self) -> Layout:
        layout = Layout(
            self._encode_codes(self._long_encoder),
            empty_ratio=PINWHEEL_EMPTY_RATIO,
            similarity=PINWHEEL_SIMILARITY,
            lambda_threshold=PINWHEEL_LAMBDA_THRESHOLD,
            eta=PINWHEEL_ETA,
            seed=PINWHEEL_SEED,
            use_gpu=True,
        )
        step_offset = 1
        long_radius = max(
            LAYOUT_ADAPTIVE_RADIUS_MIN,
            int(layout.width * LAYOUT_ADAPTIVE_RADIUS_START_FACTOR),
        )
        adaptive_long = self._adaptive_config(long_radius)
        layout.run(
            steps=PINWHEEL_LONG_STEPS,
            pairs_per_step=PINWHEEL_LONG_PAIRS_PER_STEP,
            pair_radius=adaptive_long.start_radius,
            mode="long",
            min_swap_ratio=PINWHEEL_MIN_SWAP_RATIO,
            min_swap_window=PINWHEEL_MIN_SWAP_WINDOW,
            log_every=LAYOUT_LOG_EVERY_LONG,
            step_offset=step_offset,
            adaptive_params=adaptive_long,
            energy_radius=None,
            energy_stability_window=LAYOUT_ENERGY_STABILITY_WINDOW,
            energy_stability_delta=LAYOUT_ENERGY_STABILITY_DELTA,
            energy_stability_every=LAYOUT_ENERGY_STABILITY_EVERY,
            energy_stability_max_points=LAYOUT_ENERGY_STABILITY_MAX_POINTS,
            log_visuals=LAYOUT_LOG_VISUALS,
        )
        step_offset += layout.last_steps
        layout.save_json(PINWHEEL_LONG_LAYOUT_PATH)

        layout.replace_codes(self._encode_codes(self._short_encoder))
        short_radius = max(
            LAYOUT_ADAPTIVE_RADIUS_MIN,
            int(round(layout.width * PINWHEEL_SHORT_LOCAL_RADIUS_FACTOR)),
        )
        adaptive_short = self._adaptive_config(short_radius)
        short_optim = ShortLayoutOptimConfig(
            energy_radius=short_radius,
            energy_max_points=LAYOUT_SHORT_ENERGY_MAX_POINTS,
            energy_recalc_every=LAYOUT_SHORT_ENERGY_RECALC_EVERY,
            energy_eps=LAYOUT_SHORT_ENERGY_EPS,
            use_weighted_first_point=LAYOUT_SHORT_WEIGHTED_FIRST,
            similarity_cutoff=PINWHEEL_LAMBDA_THRESHOLD,
            partitions=PINWHEEL_SHORT_PARTITIONS,
        )
        layout.run(
            steps=PINWHEEL_SHORT_STEPS,
            pairs_per_step=PINWHEEL_SHORT_PAIRS_PER_STEP,
            pair_radius=adaptive_short.start_radius,
            mode="short",
            local_radius=short_radius,
            min_swap_ratio=PINWHEEL_MIN_SWAP_RATIO,
            min_swap_window=PINWHEEL_MIN_SWAP_WINDOW,
            log_every=LAYOUT_LOG_EVERY_SHORT,
            step_offset=step_offset,
            adaptive_params=adaptive_short,
            short_optim=short_optim,
            energy_stability_window=LAYOUT_ENERGY_STABILITY_WINDOW,
            energy_stability_delta=LAYOUT_ENERGY_STABILITY_DELTA,
            energy_stability_every=LAYOUT_ENERGY_STABILITY_EVERY,
            energy_stability_max_points=LAYOUT_ENERGY_STABILITY_MAX_POINTS,
            log_visuals=LAYOUT_LOG_VISUALS,
        )
        self._log_v1_quality(layout)
        self._save_image(layout)
        return layout


def main() -> None:
    configure_logging()
    surface = ContinuousRetinotopicSurface()
    long_encoder = V1PinwheelExperiment.build_encoder(
        PINWHEEL_LONG_JOINT_LAYERS,
        PINWHEEL_LONG_ANGLE_LAYERS,
    )
    short_encoder = V1PinwheelExperiment.build_encoder(
        PINWHEEL_SHORT_JOINT_LAYERS,
        PINWHEEL_SHORT_ANGLE_LAYERS,
    )
    V1PinwheelExperiment(surface, long_encoder, short_encoder).run()


if __name__ == "__main__":
    main()
