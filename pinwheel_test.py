from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from main import (
    ENCODER_LOG_EVERY,
    ENCODER_RANDOM_SEED,
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
    PINWHEEL_ADAPTIVE_LAMBDA_STEP,
    PINWHEEL_ANGLE_COUNT,
    PINWHEEL_ANGLE_MAX,
    PINWHEEL_ANGLE_MIN,
    PINWHEEL_COLOR_MAX_HUE,
    PINWHEEL_COLOR_MIN_HUE,
    PINWHEEL_COUNT,
    PINWHEEL_EMPTY_RATIO,
    PINWHEEL_ETA,
    PINWHEEL_LAMBDA_THRESHOLD,
    PINWHEEL_LAYOUT_PATH,
    PINWHEEL_LONG_ANGLE_LAYERS,
    PINWHEEL_LONG_LAYOUT_PATH,
    PINWHEEL_LONG_PAIRS_PER_STEP,
    PINWHEEL_LONG_SPATIAL_JOINT_LAYERS,
    PINWHEEL_LONG_STEPS,
    PINWHEEL_MIN_SWAP_RATIO,
    PINWHEEL_MIN_SWAP_WINDOW,
    PINWHEEL_MIN_UNIQUE_COLORS,
    PINWHEEL_OUTPUT_PATH,
    PINWHEEL_PHASE_STEP,
    PINWHEEL_POSITION_COORDINATES,
    PINWHEEL_POSITION_MAX,
    PINWHEEL_POSITION_MIN,
    PINWHEEL_RING_RADIUS,
    PINWHEEL_RESUME_LONG_CHECKPOINT,
    PINWHEEL_SEED,
    PINWHEEL_SHORT_ANGLE_LAYERS,
    PINWHEEL_SHORT_LOCAL_RADIUS_FACTOR,
    PINWHEEL_SHORT_PAIRS_PER_STEP,
    PINWHEEL_SHORT_PARTITIONS,
    PINWHEEL_SHORT_SPATIAL_JOINT_LAYERS,
    PINWHEEL_SHORT_STEPS,
    PINWHEEL_SIMILARITY,
    PINWHEEL_SMOOTH_DELTA,
    configure_logging,
)
from damp.article_refs import (
    CHROMODYNAMICS,
    CYCLIC_COORDS,
    LAID_OUT_STRUCTURE,
    MAPPING_CONTINUITY,
    MULTI_DIM_CODES,
)
from damp.encoding.bitarray import BitArray
from damp.encoding.damp_encoder import (
    ClosedDimension,
    Detectors,
    Encoder,
    EncoderCodeColorizer,
    OpenedDimension,
)
from damp.encoding.joint_encoder import JointDetectorLayer, JointEncoder
from damp.layout.damp_layout import AdaptiveLayoutConfig, Layout, ShortLayoutOptimConfig
from damp.layout.payload import LayoutEncodedCode
from damp.logging import LOGGER


@dataclass(frozen=True)
class RetinotopicAngleSample:
    index: int
    position_index: int
    center_x: float
    center_y: float
    position_x: float
    position_y: float
    radius_index: int
    radius: float
    angle_index: int
    angle: float
    encoded_angle: float
    chirality: int
    phase: float

    @property
    def label(self) -> str:
        return (
            f"sample={self.index};position={self.position_index};"
            f"center_x={self.center_x:.9f};center_y={self.center_y:.9f};"
            f"x={self.position_x:.9f};y={self.position_y:.9f};"
            f"radius_index={self.radius_index};radius={self.radius:.9f};"
            f"angle_index={self.angle_index};angle={self.angle:.9f};"
            f"encoded_angle={self.encoded_angle:.9f};chirality={self.chirality};"
            f"phase={self.phase:.9f}"
        )


@dataclass(frozen=True)
class AlternatingPinwheelTopology:
    """Согласует входные циклические координаты соседних пинвилов."""

    phase_step: float

    @staticmethod
    def chirality(row: int, column: int) -> int:
        return 1 if (row + column) % 2 == 0 else -1

    def phase(self, column: int) -> float:
        return (column % 2) * self.phase_step

    def encode_angle(self, angle: float, *, row: int, column: int) -> float:
        period = PINWHEEL_ANGLE_MAX - PINWHEEL_ANGLE_MIN
        chirality = self.chirality(row, column)
        phase = self.phase(column)
        encoded = (
            chirality * (angle - PINWHEEL_ANGLE_MIN)
            + phase
        ) % period
        encoded_angle = PINWHEEL_ANGLE_MIN + encoded
        LOGGER.event(
            "pinwheel.topology.angle",
            section=CYCLIC_COORDS,
            data={
                "row": row,
                "column": column,
                "source_angle": angle,
                "chirality": chirality,
                "phase": phase,
                "encoded_angle": encoded_angle,
            },
        )
        return encoded_angle


class RetinotopicAngleSurface:
    """Девять локальных циклических пространств: 9 позиций × 360 углов (§4.4–§4.6)."""

    def __init__(self) -> None:
        self._topology = AlternatingPinwheelTopology(PINWHEEL_PHASE_STEP)
        self._samples = self._build_samples(self._topology)
        self._by_label = {sample.label: sample for sample in self._samples}
        LOGGER.event(
            "pinwheel.surface.init",
            section=MULTI_DIM_CODES,
            data={
                "samples": len(self._samples),
                "positions": PINWHEEL_COUNT,
                "angles_per_position": PINWHEEL_ANGLE_COUNT,
                "topology": "retinotopic_x_y_plus_closed_angle",
                "chirality_pattern": "checkerboard",
                "phase_step": PINWHEEL_PHASE_STEP,
            },
        )

    @property
    def samples(self) -> Sequence[RetinotopicAngleSample]:
        return self._samples

    def sample(self, label: str) -> RetinotopicAngleSample:
        return self._by_label[label]

    @staticmethod
    def _build_samples(
        topology: AlternatingPinwheelTopology,
    ) -> tuple[RetinotopicAngleSample, ...]:
        samples: list[RetinotopicAngleSample] = []
        angle_period = PINWHEEL_ANGLE_MAX - PINWHEEL_ANGLE_MIN
        position_index = 0
        for row, center_y in enumerate(PINWHEEL_POSITION_COORDINATES):
            for column, center_x in enumerate(PINWHEEL_POSITION_COORDINATES):
                chirality = topology.chirality(row, column)
                phase = topology.phase(column)
                for angle_index in range(PINWHEEL_ANGLE_COUNT):
                    angle = PINWHEEL_ANGLE_MIN + (
                        angle_period * angle_index / PINWHEEL_ANGLE_COUNT
                    )
                    encoded_angle = topology.encode_angle(
                        angle,
                        row=row,
                        column=column,
                    )
                    polar_angle = math.radians(2.0 * angle)
                    position_x = center_x + PINWHEEL_RING_RADIUS * math.cos(polar_angle)
                    position_y = center_y + PINWHEEL_RING_RADIUS * math.sin(polar_angle)
                    samples.append(
                        RetinotopicAngleSample(
                            index=len(samples),
                            position_index=position_index,
                            center_x=center_x,
                            center_y=center_y,
                            position_x=position_x,
                            position_y=position_y,
                            radius_index=0,
                            radius=PINWHEEL_RING_RADIUS,
                            angle_index=angle_index,
                            angle=angle,
                            encoded_angle=encoded_angle,
                            chirality=chirality,
                            phase=phase,
                        )
                    )
                position_index += 1
        return tuple(samples)


class RetinotopicAngleEncoder:
    """Объединяет пространственный код кольца и преобразованный угловой код."""

    def __init__(
        self,
        spatial_encoder: JointEncoder,
        angle_encoder: Encoder,
        *,
        color_angle_start: int,
        color_angle_length: int,
    ) -> None:
        self._spatial_encoder = spatial_encoder
        self._angle_encoder = angle_encoder
        self._spatial_code_length = spatial_encoder.code_length
        self.code_length = self._spatial_code_length + angle_encoder.code_length
        self.colorizer = EncoderCodeColorizer(
            min_hue=PINWHEEL_COLOR_MIN_HUE,
            max_hue=PINWHEEL_COLOR_MAX_HUE,
            start_bit=self._spatial_code_length + color_angle_start,
            bit_length=color_angle_length,
        )
        LOGGER.event(
            "pinwheel.encoder.init",
            section=MULTI_DIM_CODES,
            data={
                "spatial_bits": spatial_encoder.code_length,
                "angular_bits": angle_encoder.code_length,
                "color_angle_start": color_angle_start,
                "color_angle_length": color_angle_length,
                "code_length": self.code_length,
                "color_source": "angular_encoder_bits",
            },
        )

    @staticmethod
    def _concatenate(first: BitArray, second: BitArray) -> BitArray:
        code = BitArray(len(first) + len(second))
        for index, bit in enumerate(first):
            if bit:
                code.set(index)
        offset = len(first)
        for index, bit in enumerate(second):
            if bit:
                code.set(offset + index)
        return code

    def encode(self, sample: RetinotopicAngleSample) -> BitArray:
        _, spatial = self._spatial_encoder.encode(sample.position_x, sample.position_y)
        _, angular = self._angle_encoder.encode(sample.encoded_angle)
        code = self._concatenate(spatial, angular)
        LOGGER.event(
            "pinwheel.encoder.code",
            section=MULTI_DIM_CODES,
            data={
                "position": sample.position_index,
                "source_angle": sample.angle,
                "encoded_angle": sample.encoded_angle,
                "chirality": sample.chirality,
                "phase": sample.phase,
                "active_spatial_bits": spatial.count(),
                "active_angular_bits": angular.count(),
                "active_bits": code.count(),
            },
        )
        return code


class PinwheelQualityEvaluator:
    """Использует внешний угол исключительно как эталон проверки непрерывности."""

    def __init__(self, surface: RetinotopicAngleSurface) -> None:
        self._surface = surface

    @staticmethod
    def _angle_delta(first: float, second: float) -> float:
        period = PINWHEEL_ANGLE_MAX - PINWHEEL_ANGLE_MIN
        delta = abs(first - second) % period
        return min(delta, period - delta)

    @staticmethod
    def _signed_angle_delta(first: float, second: float) -> float:
        period = PINWHEEL_ANGLE_MAX - PINWHEEL_ANGLE_MIN
        return (second - first + period / 2.0) % period - period / 2.0

    @staticmethod
    def _winding_component_count(candidates: set[tuple[int, int]]) -> int:
        candidates = set(candidates)
        components = 0
        while candidates:
            components += 1
            queue = deque([candidates.pop()])
            while queue:
                y, x = queue.popleft()
                for delta_y in (-1, 0, 1):
                    for delta_x in (-1, 0, 1):
                        neighbour = (y + delta_y, x + delta_x)
                        if neighbour in candidates:
                            candidates.remove(neighbour)
                            queue.append(neighbour)
        return components

    def _singularity_metrics(self, layout: Layout) -> tuple[dict[int, int], int, int]:
        candidates_by_position: dict[int, set[tuple[int, int]]] = {
            position: set() for position in range(PINWHEEL_COUNT)
        }
        all_winding_cells: set[tuple[int, int]] = set()
        boundary_winding_cells = 0
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
                angles = [
                    self._surface.sample(layout._points[index].label).encoded_angle
                    for index in indices
                    if index is not None
                ]
                winding = sum(
                    self._signed_angle_delta(angles[index], angles[(index + 1) % 4])
                    for index in range(4)
                )
                if abs(winding) >= 0.75 * (PINWHEEL_ANGLE_MAX - PINWHEEL_ANGLE_MIN):
                    all_winding_cells.add((y, x))
                    positions = {
                        self._surface.sample(layout._points[index].label).position_index
                        for index in indices
                        if index is not None
                    }
                    if len(positions) == 1:
                        candidates_by_position[next(iter(positions))].add((y, x))
                    else:
                        boundary_winding_cells += 1

        components_by_position: dict[int, int] = {}
        for position, candidates in candidates_by_position.items():
            components = 0
            while candidates:
                components += 1
                queue = deque([candidates.pop()])
                while queue:
                    y, x = queue.popleft()
                    for neighbour in (
                        (y - 1, x),
                        (y + 1, x),
                        (y, x - 1),
                        (y, x + 1),
                    ):
                        if neighbour in candidates:
                            candidates.remove(neighbour)
                            queue.append(neighbour)
            components_by_position[position] = components
        return (
            components_by_position,
            boundary_winding_cells,
            self._winding_component_count(all_winding_cells),
        )

    def evaluate(
        self,
        layout: Layout,
        encoder: RetinotopicAngleEncoder,
    ) -> Mapping[str, float | int | bool | str]:
        angle_deltas: list[float] = []
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
                    first = self._surface.sample(layout._points[point_index].label)
                    second = self._surface.sample(layout._points[neighbour_index].label)
                    angle_deltas.append(
                        self._angle_delta(first.encoded_angle, second.encoded_angle)
                    )

        first_position = self._surface.samples[:PINWHEEL_ANGLE_COUNT]
        colors = [encoder.colorizer.color(encoder.encode(sample)) for sample in first_position]
        color_steps = [
            math.dist(colors[index], colors[(index + 1) % len(colors)])
            for index in range(len(colors))
        ]
        unique_colors = len(set(colors))
        (
            singularities_by_position,
            boundary_winding_cells,
            singularities,
        ) = self._singularity_metrics(layout)
        positions_with_one = sum(value == 1 for value in singularities_by_position.values())
        metrics: dict[str, float | int | bool | str] = {
            "samples": len(self._surface.samples),
            "positions": PINWHEEL_COUNT,
            "angles_per_position": PINWHEEL_ANGLE_COUNT,
            "unique_code_colors": unique_colors,
            "mean_color_step": sum(color_steps) / len(color_steps),
            "max_color_step": max(color_steps),
            "mean_angle_delta": sum(angle_deltas) / len(angle_deltas),
            "sharp_transition_ratio": sum(value > PINWHEEL_SMOOTH_DELTA for value in angle_deltas)
            / len(angle_deltas),
            "detected_singularities": singularities,
            "singularities_by_position": str(singularities_by_position),
            "positions_with_one_singularity": positions_with_one,
            "boundary_winding_cells": boundary_winding_cells,
            "full_spectrum": unique_colors >= PINWHEEL_MIN_UNIQUE_COLORS,
            "target_singularities": singularities == PINWHEEL_COUNT,
        }
        LOGGER.event("pinwheel.quality", section=MAPPING_CONTINUITY, data=metrics)
        LOGGER.event(
            "pinwheel.color.quality",
            section=CHROMODYNAMICS,
            data={
                "source": "angular_encoder_bits",
                "external_angle_used_for_color": False,
                "unique_colors": unique_colors,
                "full_spectrum": metrics["full_spectrum"],
            },
        )
        return metrics


class PinwheelExperiment:
    def __init__(
        self,
        surface: RetinotopicAngleSurface,
        long_encoder: RetinotopicAngleEncoder,
        short_encoder: RetinotopicAngleEncoder,
    ) -> None:
        self._surface = surface
        self._long_encoder = long_encoder
        self._short_encoder = short_encoder
        self._quality = PinwheelQualityEvaluator(surface)

    @staticmethod
    def build_encoder(
        spatial_layers: Sequence[tuple[tuple[int, ...], tuple[float, ...]]],
        angle_layers: Sequence[tuple[int, float]],
    ) -> RetinotopicAngleEncoder:
        spatial_encoder = JointEncoder(
            OpenedDimension(
                "x",
                (PINWHEEL_POSITION_MIN, PINWHEEL_POSITION_MAX),
                [Detectors(1, 0.0)],
            ),
            OpenedDimension(
                "y",
                (PINWHEEL_POSITION_MIN, PINWHEEL_POSITION_MAX),
                [Detectors(1, 0.0)],
            ),
            joint_layers=[
                JointDetectorLayer(counts=counts, overlaps=overlaps)
                for counts, overlaps in spatial_layers
            ],
            random_bit=False,
            random_seed=ENCODER_RANDOM_SEED,
            log_every=ENCODER_LOG_EVERY,
        )
        angle_encoder = Encoder(
            ClosedDimension(
                "angle",
                (PINWHEEL_ANGLE_MIN, PINWHEEL_ANGLE_MAX),
                [Detectors(count, overlap) for count, overlap in angle_layers],
            ),
            random_bit=False,
            random_seed=ENCODER_RANDOM_SEED,
            log_every=ENCODER_LOG_EVERY,
        )
        LOGGER.event(
            "pinwheel.angle.closed",
            section=CYCLIC_COORDS,
            data={"minimum": PINWHEEL_ANGLE_MIN, "maximum": PINWHEEL_ANGLE_MAX},
        )
        color_angle_start = sum(count for count, _ in angle_layers[:-1])
        color_angle_length = angle_layers[-1][0]
        return RetinotopicAngleEncoder(
            spatial_encoder,
            angle_encoder,
            color_angle_start=color_angle_start,
            color_angle_length=color_angle_length,
        )

    def _encode_codes(
        self,
        encoder: RetinotopicAngleEncoder,
    ) -> Mapping[float, Sequence[LayoutEncodedCode]]:
        codes: dict[float, list[LayoutEncodedCode]] = defaultdict(list)
        for sample in self._surface.samples:
            codes[sample.encoded_angle].append(
                LayoutEncodedCode(code=encoder.encode(sample), label=sample.label)
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
    def _save(layout: Layout) -> None:
        from PIL import Image

        Image.fromarray(layout.render_image(log=True)).save(PINWHEEL_OUTPUT_PATH)
        layout.save_json(PINWHEEL_LAYOUT_PATH)
        LOGGER.event(
            "pinwheel.image.save",
            section=LAID_OUT_STRUCTURE,
            data={"image": PINWHEEL_OUTPUT_PATH, "layout": PINWHEEL_LAYOUT_PATH},
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
            colorizer=self._long_encoder.colorizer,
        )
        long_radius = max(
            LAYOUT_ADAPTIVE_RADIUS_MIN,
            int(layout.width * LAYOUT_ADAPTIVE_RADIUS_START_FACTOR),
        )
        resumed = (
            PINWHEEL_RESUME_LONG_CHECKPOINT
            and layout.restore_positions_json(PINWHEEL_LONG_LAYOUT_PATH)
        )
        if resumed:
            step_offset = 1
        else:
            layout.run(
                steps=PINWHEEL_LONG_STEPS,
                pairs_per_step=PINWHEEL_LONG_PAIRS_PER_STEP,
                pair_radius=long_radius,
                mode="long",
                min_swap_ratio=PINWHEEL_MIN_SWAP_RATIO,
                min_swap_window=PINWHEEL_MIN_SWAP_WINDOW,
                log_every=LAYOUT_LOG_EVERY_LONG,
                step_offset=1,
                adaptive_params=self._adaptive_config(long_radius),
                energy_stability_window=LAYOUT_ENERGY_STABILITY_WINDOW,
                energy_stability_delta=LAYOUT_ENERGY_STABILITY_DELTA,
                energy_stability_every=LAYOUT_ENERGY_STABILITY_EVERY,
                energy_stability_max_points=LAYOUT_ENERGY_STABILITY_MAX_POINTS,
                log_visuals=LAYOUT_LOG_VISUALS,
            )
            step_offset = 1 + layout.last_steps
            layout.save_json(PINWHEEL_LONG_LAYOUT_PATH)
        layout.replace_codes(
            self._encode_codes(self._short_encoder),
            colorizer=self._short_encoder.colorizer,
        )
        short_radius = max(
            LAYOUT_ADAPTIVE_RADIUS_MIN,
            int(round(layout.width * PINWHEEL_SHORT_LOCAL_RADIUS_FACTOR)),
        )
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
            pair_radius=short_radius,
            mode="short",
            local_radius=short_radius,
            min_swap_ratio=PINWHEEL_MIN_SWAP_RATIO,
            min_swap_window=PINWHEEL_MIN_SWAP_WINDOW,
            log_every=LAYOUT_LOG_EVERY_SHORT,
            step_offset=step_offset,
            energy_radius=short_radius,
            energy_stability_window=LAYOUT_ENERGY_STABILITY_WINDOW,
            energy_stability_delta=LAYOUT_ENERGY_STABILITY_DELTA,
            energy_stability_every=LAYOUT_ENERGY_STABILITY_EVERY,
            energy_stability_max_points=LAYOUT_ENERGY_STABILITY_MAX_POINTS,
            adaptive_params=self._adaptive_config(short_radius),
            short_optim=short_optim,
            log_visuals=LAYOUT_LOG_VISUALS,
        )
        self._quality.evaluate(layout, self._short_encoder)
        self._save(layout)
        return layout


def main() -> None:
    configure_logging()
    surface = RetinotopicAngleSurface()
    long_encoder = PinwheelExperiment.build_encoder(
        PINWHEEL_LONG_SPATIAL_JOINT_LAYERS,
        PINWHEEL_LONG_ANGLE_LAYERS,
    )
    short_encoder = PinwheelExperiment.build_encoder(
        PINWHEEL_SHORT_SPATIAL_JOINT_LAYERS,
        PINWHEEL_SHORT_ANGLE_LAYERS,
    )
    PinwheelExperiment(surface, long_encoder, short_encoder).run()


if __name__ == "__main__":
    main()
