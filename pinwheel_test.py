"""Раскладка двух позиций по 360 углов общим кодером MNIST.

Код Angle || X || Y строится через main._build_encoder() (§2.2.3, §4.6).
Параметры дальней и ближней раскладки общие с MNIST (§5.4–§5.6).
Число углов на позицию, позиции и пути результатов заданы в main.py.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from damp.article_refs import (
    LONG_RANGE_IMPORTANCE,
    MAPPING_CONTINUITY,
    MULTI_DIM_CODES,
    OUTSIDE_ARTICLE,
)
from damp.encoding.damp_encoder import (
    Encoder,
    EncoderCodeColorizer,
)
from damp.layout.damp_layout import AdaptiveLayoutConfig, Layout, ShortLayoutOptimConfig
from damp.layout.payload import LayoutEncodedCode, LayoutPayloadBuilder
from damp.logging import LOGGER
from main import (
    ACTIVATION_POINT_ENERGY_RADIUS,
    ENCODER_SIGNATURE_KEY,
    LAYOUT_ADAPTIVE_LAMBDA_MAX,
    LAYOUT_ADAPTIVE_LAMBDA_STEP,
    LAYOUT_ADAPTIVE_RADIUS_MIN,
    LAYOUT_ADAPTIVE_RADIUS_SCHEDULE,
    LAYOUT_ADAPTIVE_RADIUS_START_FACTOR,
    LAYOUT_ADAPTIVE_RADIUS_TRIGGER_FACTOR,
    LAYOUT_ADAPTIVE_SWAP_TRIGGER,
    LAYOUT_ADAPTIVE_SWAP_WINDOW,
    LAYOUT_ANNEAL_RADIUS_MAX_FACTOR,
    LAYOUT_ANNEAL_RADIUS_MIN,
    LAYOUT_ANNEAL_TEMPERATURE,
    LAYOUT_EMPTY_RATIO,
    LAYOUT_ENERGY_EVAL_EVERY,
    LAYOUT_ENERGY_RADIUS,
    LAYOUT_ENERGY_TARGET,
    LAYOUT_ETA,
    LAYOUT_LAMBDA_THRESHOLD,
    LAYOUT_LOG_EVERY_LONG,
    LAYOUT_LONG_DISTANCE,
    LAYOUT_LONG_KERNEL,
    LAYOUT_LONG_KERNEL_RADIUS_MIN,
    LAYOUT_LONG_KERNEL_WIDTH,
    LAYOUT_LOG_EVERY_SHORT,
    LAYOUT_LOG_VISUALS,
    LAYOUT_MIN_SWAP_RATIO,
    LAYOUT_MIN_SWAP_WINDOW,
    LAYOUT_PSEUDO_COLOR_MAX_HUE,
    LAYOUT_PSEUDO_COLOR_MIN_HUE,
    LAYOUT_RANDOM_SEED,
    LAYOUT_SHORT_ADAPTIVE_RADIUS_FACTOR,
    LAYOUT_SHORT_ADAPTIVE_RADIUS_SCHEDULE,
    LAYOUT_SHORT_DISTANCE,
    LAYOUT_SHORT_ENERGY_EPS,
    LAYOUT_SHORT_PARTITIONS,
    LAYOUT_SHORT_SIMILARITY_CUTOFF,
    LAYOUT_SHORT_WEIGHTED_FIRST,
    LAYOUT_SIMILARITY,
    LAYOUT_USE_GPU,
    MNIST_ANGLE_MAX,
    MNIST_ANGLE_MIN,
    PINWHEEL_ANGLE_COUNT,
    PINWHEEL_COMPONENT_OUTPUT_PATH,
    PINWHEEL_ENERGY_OUTPUT_PATH,
    PINWHEEL_LAYOUT_PATH,
    PINWHEEL_OUTPUT_PATH,
    PINWHEEL_POSITIONS,
    PINWHEEL_SMOOTH_DELTA,
    LayoutParameterTuner,
    TunedLayoutParams,
    _build_encoder,
    configure_logging,
)


class PinwheelEncoderFactory:
    """Тот же кодер Angle || X || Y, что и в конвейере MNIST (§2.2.3, §4.6)."""

    @staticmethod
    def create() -> Encoder:
        return _build_encoder()


@dataclass(frozen=True)
class PinwheelQuality:
    mean_neighbour_delta: float
    sharp_transition_ratio: float
    singularities: int
    long_range_similarity_floor: float

    def as_mapping(self) -> Mapping[str, float | int]:
        return {
            "mean_neighbour_delta": self.mean_neighbour_delta,
            "sharp_transition_ratio": self.sharp_transition_ratio,
            "singularities": self.singularities,
            "long_range_similarity_floor": self.long_range_similarity_floor,
        }


class PinwheelQualityEvaluator:
    """Диагностика раскладки по известному углу точки (ср. рис. 13–16, §5.13.2).

    Угол не участвует в раскладке: он нужен только для проверки, что соседние
    ячейки получили близкие углы; также измеряется число сингулярностей.
    """

    def __init__(self, *, period: float, smooth_delta: float) -> None:
        if period <= 0.0 or not 0.0 < smooth_delta < period / 2.0:
            raise ValueError("invalid pinwheel quality parameters")
        self._period = period
        self._smooth_delta = smooth_delta

    def _signed_delta(self, first: float, second: float) -> float:
        half = self._period / 2.0
        return (second - first + half) % self._period - half

    def evaluate(self, layout: Layout, codes: Sequence) -> PinwheelQuality:
        hues = layout.hues()
        deltas: list[float] = []
        singularities = 0
        for y in range(layout.height):
            for x in range(layout.width):
                index = layout.grid[y][x]
                if index is None:
                    continue
                for ny, nx in ((y, x + 1), (y + 1, x)):
                    if ny >= layout.height or nx >= layout.width:
                        continue
                    neighbour = layout.grid[ny][nx]
                    if neighbour is None:
                        continue
                    deltas.append(abs(self._signed_delta(hues[index], hues[neighbour])))
                if y + 1 < layout.height and x + 1 < layout.width:
                    corners = (
                        layout.grid[y][x],
                        layout.grid[y][x + 1],
                        layout.grid[y + 1][x + 1],
                        layout.grid[y + 1][x],
                    )
                    if any(corner is None for corner in corners):
                        continue
                    winding = sum(
                        self._signed_delta(hues[corners[k]], hues[corners[(k + 1) % 4]])
                        for k in range(4)
                    )
                    if abs(winding) >= 0.75 * self._period:
                        singularities += 1
        half_turn = len(codes) // 2
        floor = sum(
            codes[k].common(codes[(k + half_turn) % len(codes)])
            / math.sqrt(codes[k].count() * codes[(k + half_turn) % len(codes)].count())
            for k in range(len(codes))
        ) / len(codes)
        quality = PinwheelQuality(
            mean_neighbour_delta=sum(deltas) / len(deltas) if deltas else 0.0,
            sharp_transition_ratio=(
                sum(delta > self._smooth_delta for delta in deltas) / len(deltas) if deltas else 0.0
            ),
            singularities=singularities,
            long_range_similarity_floor=floor,
        )
        LOGGER.event(
            "pinwheel.quality",
            section=MAPPING_CONTINUITY,
            data={**quality.as_mapping(), "neighbour_pairs": len(deltas)},
        )
        LOGGER.event(
            "pinwheel.long_range",
            section=LONG_RANGE_IMPORTANCE,
            data={
                "opposite_angle_similarity": floor,
                "lambda_threshold": LAYOUT_LAMBDA_THRESHOLD,
                "note": "Противоположные углы сравниваются внутри каждой позиции по полному коду Angle || X || Y",
            },
        )
        return quality


class PinwheelExperiment:
    """Кодирует по 360 стимулов (угол, X, Y) для каждой позиции (§4.6, §5.2–§5.6)."""

    def __init__(
        self,
        *,
        encoder: Encoder | None = None,
        output_path: str = PINWHEEL_OUTPUT_PATH,
        component_output_path: str = PINWHEEL_COMPONENT_OUTPUT_PATH,
        energy_output_path: str = PINWHEEL_ENERGY_OUTPUT_PATH,
        layout_path: str = PINWHEEL_LAYOUT_PATH,
    ) -> None:
        self._encoder = encoder or PinwheelEncoderFactory.create()
        self._positions = PINWHEEL_POSITIONS
        self._payload: LayoutPayloadBuilder | None = None
        self._layout_tuner = LayoutParameterTuner()
        self._output_path = output_path
        self._component_output_path = component_output_path
        self._energy_output_path = energy_output_path
        self._layout_path = layout_path
        self._quality = PinwheelQualityEvaluator(
            period=MNIST_ANGLE_MAX - MNIST_ANGLE_MIN,
            smooth_delta=PINWHEEL_SMOOTH_DELTA,
        )

    @property
    def encoder(self) -> Encoder:
        return self._encoder

    def encode_codes(self) -> dict[float, list[LayoutEncodedCode]]:
        segments = {dim.title: self._encoder.segment(dim.title) for dim in self._encoder.dimensions}
        self._payload = LayoutPayloadBuilder(
            similarity=LAYOUT_SIMILARITY,
            lambda_threshold=LAYOUT_LAMBDA_THRESHOLD,
            eta=LAYOUT_ETA,
            metadata={
                ENCODER_SIGNATURE_KEY: self._encoder.signature,
                "stimulus_positions": [{"x": x, "y": y} for x, y in self._positions],
                "angles_per_position": PINWHEEL_ANGLE_COUNT,
                "code_segments": segments,
            },
        )
        LOGGER.event(
            "pinwheel.stimulus",
            section=MULTI_DIM_CODES,
            data={
                "positions": self._positions,
                "segments": segments,
                "angles_per_position": PINWHEEL_ANGLE_COUNT,
                "stimuli": PINWHEEL_ANGLE_COUNT * len(self._positions),
            },
        )
        period = MNIST_ANGLE_MAX - MNIST_ANGLE_MIN
        for index in range(PINWHEEL_ANGLE_COUNT):
            angle = MNIST_ANGLE_MIN + period * index / PINWHEEL_ANGLE_COUNT
            for x, y in self._positions:
                _, code = self._encoder.encode(angle, x, y)
                self._payload.add_code(label=f"angle={angle:.3f}, x={x:g}, y={y:g}", hue=angle, code=code)
        LOGGER.event(
            "pinwheel.codes",
            section=MULTI_DIM_CODES,
            data={
                "angles": PINWHEEL_ANGLE_COUNT,
                "positions": len(self._positions),
                "stimuli": len(self._payload.records),
                "code_length": self._encoder.code_length,
                "expected_density": self._encoder.expected_density(),
            },
        )
        return self._payload.codes_by_hue()

    def build_layout(
        self, codes: Mapping[float, Sequence[LayoutEncodedCode]], *, params: TunedLayoutParams | None = None
    ) -> Layout:
        run_params = params or self._layout_tuner.tune(sum(len(bucket) for bucket in codes.values()))
        return Layout(
            codes,
            empty_ratio=LAYOUT_EMPTY_RATIO,
            similarity=LAYOUT_SIMILARITY,
            lambda_threshold=LAYOUT_LAMBDA_THRESHOLD,
            eta=LAYOUT_ETA,
            seed=LAYOUT_RANDOM_SEED,
            use_gpu=LAYOUT_USE_GPU,
            long_subset_size=run_params.long_subset_size,
            long_subset_refresh=run_params.long_subset_refresh,
            colorizer=EncoderCodeColorizer(
                min_hue=LAYOUT_PSEUDO_COLOR_MIN_HUE,
                max_hue=LAYOUT_PSEUDO_COLOR_MAX_HUE,
            ),
            long_distance=LAYOUT_LONG_DISTANCE,
            short_distance=LAYOUT_SHORT_DISTANCE,
            long_kernel=LAYOUT_LONG_KERNEL,
            long_kernel_radius_min=LAYOUT_LONG_KERNEL_RADIUS_MIN,
            long_kernel_width=LAYOUT_LONG_KERNEL_WIDTH,
            component_period=MNIST_ANGLE_MAX - MNIST_ANGLE_MIN,
        )

    def run(self, *, save: bool = True) -> tuple[Layout, PinwheelQuality]:
        codes = self.encode_codes()
        run_params = self._layout_tuner.tune(sum(len(bucket) for bucket in codes.values()))
        layout = self.build_layout(codes, params=run_params)
        long_radius = max(
            LAYOUT_ADAPTIVE_RADIUS_MIN,
            int(layout.width * LAYOUT_ADAPTIVE_RADIUS_START_FACTOR),
        )
        adaptive_long = AdaptiveLayoutConfig(
            start_radius=long_radius,
            end_radius=LAYOUT_ADAPTIVE_RADIUS_MIN,
            swap_ratio_trigger=LAYOUT_ADAPTIVE_SWAP_TRIGGER,
            lambda_step=LAYOUT_ADAPTIVE_LAMBDA_STEP,
            lambda_max=LAYOUT_ADAPTIVE_LAMBDA_MAX,
            radius_schedule=LAYOUT_ADAPTIVE_RADIUS_SCHEDULE,
            radius_factor=LAYOUT_ADAPTIVE_RADIUS_TRIGGER_FACTOR,
            swap_window=LAYOUT_ADAPTIVE_SWAP_WINDOW,
            temperature=LAYOUT_ANNEAL_TEMPERATURE,
            temperature_radius_max=int(layout.width * LAYOUT_ANNEAL_RADIUS_MAX_FACTOR),
            temperature_radius_min=LAYOUT_ANNEAL_RADIUS_MIN,
        )
        layout.run(
            steps=run_params.long_steps,
            pairs_per_step=run_params.long_pairs_per_step,
            pair_radius=long_radius,
            mode="long",
            min_swap_ratio=LAYOUT_MIN_SWAP_RATIO,
            min_swap_window=LAYOUT_MIN_SWAP_WINDOW,
            log_every=LAYOUT_LOG_EVERY_LONG,
            step_offset=1,
            adaptive_params=adaptive_long,
            energy_radius=LAYOUT_ENERGY_RADIUS,
            energy_every=LAYOUT_ENERGY_EVAL_EVERY,
            energy_target=LAYOUT_ENERGY_TARGET,
            log_visuals=LAYOUT_LOG_VISUALS,
        )
        step_offset = 1 + layout.last_steps
        short_local_radius = run_params.short_local_radius
        short_radius_start = max(
            LAYOUT_ADAPTIVE_RADIUS_MIN,
            min(
                int(max(1, round(short_local_radius * LAYOUT_SHORT_ADAPTIVE_RADIUS_FACTOR))),
                int(layout.width * LAYOUT_ADAPTIVE_RADIUS_START_FACTOR),
            ),
        )
        adaptive_short = AdaptiveLayoutConfig(
            start_radius=short_radius_start,
            end_radius=LAYOUT_ADAPTIVE_RADIUS_MIN,
            swap_ratio_trigger=LAYOUT_ADAPTIVE_SWAP_TRIGGER,
            lambda_step=LAYOUT_ADAPTIVE_LAMBDA_STEP,
            lambda_max=LAYOUT_ADAPTIVE_LAMBDA_MAX,
            radius_schedule=LAYOUT_SHORT_ADAPTIVE_RADIUS_SCHEDULE,
            radius_factor=LAYOUT_ADAPTIVE_RADIUS_TRIGGER_FACTOR,
            swap_window=LAYOUT_ADAPTIVE_SWAP_WINDOW,
        )
        short_optim = ShortLayoutOptimConfig(
            energy_radius=run_params.short_energy_radius,
            energy_recalc_every=run_params.short_energy_recalc_every,
            energy_eps=LAYOUT_SHORT_ENERGY_EPS,
            use_weighted_first_point=LAYOUT_SHORT_WEIGHTED_FIRST,
            similarity_cutoff=LAYOUT_SHORT_SIMILARITY_CUTOFF,
            partitions=LAYOUT_SHORT_PARTITIONS,
        )
        layout.run(
            steps=run_params.short_steps,
            pairs_per_step=run_params.short_pairs_per_step,
            pair_radius=short_radius_start,
            mode="short",
            local_radius=short_local_radius,
            min_swap_ratio=LAYOUT_MIN_SWAP_RATIO,
            min_swap_window=LAYOUT_MIN_SWAP_WINDOW,
            log_every=LAYOUT_LOG_EVERY_SHORT,
            step_offset=step_offset,
            adaptive_params=adaptive_short,
            short_optim=short_optim,
            log_visuals=LAYOUT_LOG_VISUALS,
        )
        flat_codes = [entry.code for angle in sorted(codes) for entry in codes[angle]]
        quality = self._quality.evaluate(layout, flat_codes)
        if save:
            self._save(layout)
        return layout, quality

    def _save(self, layout: Layout) -> None:
        from PIL import Image

        if self._payload is None:
            raise RuntimeError("encode_codes must be called before saving the pinwheel layout")
        Image.fromarray(layout.render_image()).save(self._output_path)
        Image.fromarray(
            layout.render_component_image(period=MNIST_ANGLE_MAX - MNIST_ANGLE_MIN)
        ).save(self._component_output_path)
        Image.fromarray(
            layout.render_energy_image(layout.point_energies(ACTIVATION_POINT_ENERGY_RADIUS))
        ).save(self._energy_output_path)
        self._payload.save_base(self._layout_path)
        layout.save_json(self._layout_path)
        LOGGER.event(
            "pinwheel.image.save",
            section=OUTSIDE_ARTICLE,
            data={
                "image": self._output_path,
                "component_image": self._component_output_path,
                "energy_image": self._energy_output_path,
                "layout": self._layout_path,
            },
        )


def main() -> None:
    configure_logging()
    PinwheelExperiment().run()


if __name__ == "__main__":
    main()
