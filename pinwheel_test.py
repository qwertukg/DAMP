from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import argparse
import math
import random
from typing import Mapping, Sequence

from main import (
    ENCODER_LOG_EVERY,
    ENCODER_RANDOM_SEED,
    LAYOUT_ADAPTIVE_RADIUS_MIN,
    LAYOUT_ADAPTIVE_RADIUS_START_FACTOR,
    LAYOUT_ADAPTIVE_SWAP_TRIGGER,
    LAYOUT_CONTINUITY_MIN_GAIN,
    LAYOUT_CONTINUITY_STALL_SWEEPS,
    LAYOUT_CONVERGENCE_INITIAL_LIMIT,
    LAYOUT_CONVERGENCE_LIMIT_MULTIPLIER,
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
    PINWHEEL_CONTINUITY_CANDIDATES,
    PINWHEEL_CONTINUITY_DEFECT_POINTS,
    PINWHEEL_CONTINUITY_SWEEPS,
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
    PINWHEEL_POSITION_COORDINATES,
    PINWHEEL_POSITION_GENERATOR_CANDIDATES,
    PINWHEEL_POSITION_GENERATOR_MARGIN,
    PINWHEEL_POSITION_MAX,
    PINWHEEL_POSITION_MIN,
    PINWHEEL_RADIUS_FACTOR,
    PINWHEEL_ISOLATED_RADIUS,
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
    PINWHEEL_SPATIAL_BOUND_MARGIN_FACTOR,
    configure_logging,
)
from damp.article_refs import (
    CHROMODYNAMICS,
    LAID_OUT_STRUCTURE,
    MAPPING_CONTINUITY,
    MULTI_DIM_CODES,
)
from damp.encoding.orientation_encoder import (
    RetinotopicOrientationEncoder,
    RetinotopicOrientationEncoderFactory,
)
from damp.layout.damp_layout import (
    AdaptiveLayoutConfig,
    ContinuityLayoutConfig,
    Layout,
    ShortLayoutOptimConfig,
)
from damp.layout.pinwheel_topology import ContinuousPinwheelTopology
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


class RetinotopicAngleSurface:
    """Локальные циклические пространства: N позиций × 360 углов (§4.4–§4.6)."""

    def __init__(self, centers: Sequence[tuple[float, float]] | None = None) -> None:
        if centers is None:
            centers = tuple(
                (center_x, center_y)
                for center_y in PINWHEEL_POSITION_COORDINATES
                for center_x in PINWHEEL_POSITION_COORDINATES
            )
        self._topology = ContinuousPinwheelTopology(
            centers,
            angle_period=PINWHEEL_ANGLE_MAX - PINWHEEL_ANGLE_MIN,
            radius_factor=PINWHEEL_RADIUS_FACTOR,
            isolated_radius=PINWHEEL_ISOLATED_RADIUS,
        )
        self._samples = self._build_samples(self._topology)
        self._by_label = {sample.label: sample for sample in self._samples}
        LOGGER.event(
            "pinwheel.surface.init",
            section=MULTI_DIM_CODES,
            data={
                "samples": len(self._samples),
                "positions": self.position_count,
                "angles_per_position": PINWHEEL_ANGLE_COUNT,
                "topology": "retinotopic_x_y_plus_closed_angle",
                "chirality_pattern": "gabriel_graph_max_cut",
                "phase_rule": "graph_boundary_angular_synchronization",
            },
        )

    @property
    def samples(self) -> Sequence[RetinotopicAngleSample]:
        return self._samples

    @property
    def topology(self) -> ContinuousPinwheelTopology:
        return self._topology

    @property
    def position_count(self) -> int:
        return len(self._topology.charts)

    @property
    def spatial_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        xs = [sample.position_x for sample in self._samples]
        ys = [sample.position_y for sample in self._samples]
        margin_x = max(xs) - min(xs)
        margin_y = max(ys) - min(ys)
        margin_x = max(1e-9, margin_x * PINWHEEL_SPATIAL_BOUND_MARGIN_FACTOR)
        margin_y = max(1e-9, margin_y * PINWHEEL_SPATIAL_BOUND_MARGIN_FACTOR)
        return (
            (
                min(PINWHEEL_POSITION_MIN, min(xs) - margin_x),
                max(PINWHEEL_POSITION_MAX, max(xs) + margin_x),
            ),
            (
                min(PINWHEEL_POSITION_MIN, min(ys) - margin_y),
                max(PINWHEEL_POSITION_MAX, max(ys) + margin_y),
            ),
        )

    def sample(self, label: str) -> RetinotopicAngleSample:
        return self._by_label[label]

    @staticmethod
    def _build_samples(
        topology: ContinuousPinwheelTopology,
    ) -> tuple[RetinotopicAngleSample, ...]:
        samples: list[RetinotopicAngleSample] = []
        angle_period = PINWHEEL_ANGLE_MAX - PINWHEEL_ANGLE_MIN
        for position_index, chart in enumerate(topology.charts):
            for angle_index in range(PINWHEEL_ANGLE_COUNT):
                angle = PINWHEEL_ANGLE_MIN + (
                    angle_period * angle_index / PINWHEEL_ANGLE_COUNT
                )
                polar_angle = math.radians(2.0 * angle)
                position_x = chart.center_x + chart.radius * math.cos(polar_angle)
                position_y = chart.center_y + chart.radius * math.sin(polar_angle)
                encoded_angle, phase = topology.encode_angle(
                    angle,
                    center_index=position_index,
                    position_x=position_x,
                    position_y=position_y,
                )
                samples.append(
                    RetinotopicAngleSample(
                        index=len(samples),
                        position_index=position_index,
                        center_x=chart.center_x,
                        center_y=chart.center_y,
                        position_x=position_x,
                        position_y=position_y,
                        radius_index=0,
                        radius=chart.radius,
                        angle_index=angle_index,
                        angle=angle,
                        encoded_angle=encoded_angle,
                        chirality=chart.chirality,
                        phase=phase,
                    )
                )
        return tuple(samples)


class PinwheelPositionGenerator:
    """Детерминированно разносит произвольное число тестовых центров по области."""

    def __init__(
        self,
        *,
        minimum: float,
        maximum: float,
        margin: float,
        candidates: int,
    ) -> None:
        if maximum <= minimum:
            raise ValueError("maximum must be greater than minimum")
        if margin < 0.0 or 2.0 * margin >= maximum - minimum:
            raise ValueError("margin leaves no position generation area")
        if candidates <= 0:
            raise ValueError("candidates must be positive")
        self._minimum = minimum + margin
        self._maximum = maximum - margin
        self._candidates = candidates

    def generate(self, count: int, *, seed: int) -> tuple[tuple[float, float], ...]:
        if count <= 0:
            raise ValueError("count must be positive")
        rng = random.Random(seed)
        positions: list[tuple[float, float]] = []
        for _ in range(count):
            candidates = [
                (
                    rng.uniform(self._minimum, self._maximum),
                    rng.uniform(self._minimum, self._maximum),
                )
                for _ in range(self._candidates)
            ]
            if not positions:
                selected = candidates[0]
            else:
                selected = max(
                    candidates,
                    key=lambda candidate: min(
                        math.dist(candidate, existing) for existing in positions
                    ),
                )
            positions.append(selected)
        result = tuple(positions)
        LOGGER.event(
            "pinwheel.positions.generate",
            section=MULTI_DIM_CODES,
            data={"count": count, "seed": seed, "positions": result},
        )
        return result


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
            position: set() for position in range(self._surface.position_count)
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
        encoder: RetinotopicOrientationEncoder,
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
        colors = [
            encoder.colorizer.color(encoder.encode_sample(sample))
            for sample in first_position
        ]
        all_colors = {
            encoder.colorizer.color(encoder.encode_sample(sample))
            for sample in self._surface.samples
        }
        color_steps = [
            math.dist(colors[index], colors[(index + 1) % len(colors)])
            for index in range(len(colors))
        ]
        unique_colors = len(all_colors)
        (
            singularities_by_position,
            boundary_winding_cells,
            singularities,
        ) = self._singularity_metrics(layout)
        positions_with_one = sum(value == 1 for value in singularities_by_position.values())
        metrics: dict[str, float | int | bool | str] = {
            "samples": len(self._surface.samples),
            "positions": self._surface.position_count,
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
            "target_singularities": singularities == self._surface.position_count,
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
        long_encoder: RetinotopicOrientationEncoder,
        short_encoder: RetinotopicOrientationEncoder,
        *,
        output_path: str = PINWHEEL_OUTPUT_PATH,
        layout_path: str = PINWHEEL_LAYOUT_PATH,
        long_layout_path: str = PINWHEEL_LONG_LAYOUT_PATH,
    ) -> None:
        self._surface = surface
        self._long_encoder = long_encoder
        self._short_encoder = short_encoder
        self._quality = PinwheelQualityEvaluator(surface)
        self._output_path = output_path
        self._layout_path = layout_path
        self._long_layout_path = long_layout_path

    @staticmethod
    def build_encoder(
        spatial_layers: Sequence[tuple[tuple[int, ...], tuple[float, ...]]],
        angle_layers: Sequence[tuple[int, float]],
        *,
        spatial_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
        surface: RetinotopicAngleSurface | None = None,
    ) -> RetinotopicOrientationEncoder:
        resolved_surface = surface or RetinotopicAngleSurface()
        bounds = spatial_bounds or (
            (PINWHEEL_POSITION_MIN, PINWHEEL_POSITION_MAX),
            (PINWHEEL_POSITION_MIN, PINWHEEL_POSITION_MAX),
        )
        centers = tuple(
            (chart.center_x, chart.center_y)
            for chart in resolved_surface.topology.charts
        )
        return RetinotopicOrientationEncoderFactory.create(
            centers,
            spatial_bounds=bounds,
            spatial_layers=spatial_layers,
            angle_layers=angle_layers,
            angle_minimum=PINWHEEL_ANGLE_MIN,
            angle_maximum=PINWHEEL_ANGLE_MAX,
            radius_factor=PINWHEEL_RADIUS_FACTOR,
            isolated_radius=PINWHEEL_ISOLATED_RADIUS,
            color_minimum_hue=PINWHEEL_COLOR_MIN_HUE,
            color_maximum_hue=PINWHEEL_COLOR_MAX_HUE,
            random_seed=ENCODER_RANDOM_SEED,
            log_every=ENCODER_LOG_EVERY,
            topology=resolved_surface.topology,
        )

    def _encode_codes(
        self,
        encoder: RetinotopicOrientationEncoder,
    ) -> Mapping[float, Sequence[LayoutEncodedCode]]:
        codes: dict[float, list[LayoutEncodedCode]] = defaultdict(list)
        for sample in self._surface.samples:
            codes[sample.encoded_angle].append(
                LayoutEncodedCode(code=encoder.encode_sample(sample), label=sample.label)
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

    def _save(self, layout: Layout) -> None:
        from PIL import Image

        Image.fromarray(layout.render_image(log=True)).save(self._output_path)
        layout.save_json(self._layout_path)
        LOGGER.event(
            "pinwheel.image.save",
            section=LAID_OUT_STRUCTURE,
            data={"image": self._output_path, "layout": self._layout_path},
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
            and layout.restore_positions_json(self._long_layout_path)
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
                converge_at_min_radius=True,
                max_convergence_steps=max(1, LAYOUT_CONVERGENCE_INITIAL_LIMIT - PINWHEEL_LONG_STEPS),
                convergence_limit_multiplier=LAYOUT_CONVERGENCE_LIMIT_MULTIPLIER,
            )
            step_offset = 1 + layout.last_steps
            layout.save_json(self._long_layout_path)
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
            converge_at_min_radius=True,
            max_convergence_steps=max(1, LAYOUT_CONVERGENCE_INITIAL_LIMIT - PINWHEEL_SHORT_STEPS),
            convergence_limit_multiplier=LAYOUT_CONVERGENCE_LIMIT_MULTIPLIER,
        )
        layout.run_continuity_relaxation(
            ContinuityLayoutConfig(
                sweeps=PINWHEEL_CONTINUITY_SWEEPS,
                defect_points=PINWHEEL_CONTINUITY_DEFECT_POINTS,
                candidates_per_point=PINWHEEL_CONTINUITY_CANDIDATES,
                min_gain=LAYOUT_CONTINUITY_MIN_GAIN,
                stall_sweeps=LAYOUT_CONTINUITY_STALL_SWEEPS,
                cyclic_period=PINWHEEL_ANGLE_MAX - PINWHEEL_ANGLE_MIN,
                cyclic_sharp_threshold=PINWHEEL_SMOOTH_DELTA,
            )
        )
        self._quality.evaluate(layout, self._short_encoder)
        self._save(layout)
        return layout


def main() -> None:
    parser = argparse.ArgumentParser(description="DAMP pinwheel layout test")
    parser.add_argument("--position-count", type=int, default=None)
    parser.add_argument("--position-seed", type=int, default=PINWHEEL_SEED)
    args = parser.parse_args()
    configure_logging()
    centers = None
    stem = "pinwheel_test"
    if args.position_count is not None:
        generator = PinwheelPositionGenerator(
            minimum=PINWHEEL_POSITION_MIN,
            maximum=PINWHEEL_POSITION_MAX,
            margin=PINWHEEL_POSITION_GENERATOR_MARGIN,
            candidates=PINWHEEL_POSITION_GENERATOR_CANDIDATES,
        )
        centers = generator.generate(args.position_count, seed=args.position_seed)
        stem = f"pinwheel_test_{args.position_count}_{args.position_seed}"
    surface = RetinotopicAngleSurface(centers)
    long_encoder = PinwheelExperiment.build_encoder(
        PINWHEEL_LONG_SPATIAL_JOINT_LAYERS,
        PINWHEEL_LONG_ANGLE_LAYERS,
        spatial_bounds=surface.spatial_bounds,
        surface=surface,
    )
    short_encoder = PinwheelExperiment.build_encoder(
        PINWHEEL_SHORT_SPATIAL_JOINT_LAYERS,
        PINWHEEL_SHORT_ANGLE_LAYERS,
        spatial_bounds=surface.spatial_bounds,
        surface=surface,
    )
    PinwheelExperiment(
        surface,
        long_encoder,
        short_encoder,
        output_path=f"{stem}.png",
        layout_path=f"{stem}.json",
        long_layout_path=f"{stem}.long.checkpoint.json",
    ).run()


if __name__ == "__main__":
    main()
