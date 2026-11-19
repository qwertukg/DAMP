"""Тест пинвила: по 360 кодов угла на позицию раскладываются в пинвилы (§4.6, §5.2–§5.6, §5.13.2)."""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from PIL import Image

from damp import settings
from damp.article_refs import LONG_RANGE_IMPORTANCE, MULTI_DIM_CODES
from damp.article_refs import MAPPING_CONTINUITY
from damp.article_refs import OUTSIDE_ARTICLE
from damp.encoding.damp_encoder import Encoder, EncoderCodeColorizer
from damp.encoding.mnist_codes import _build_encoder
from damp.layout.damp_layout import AdaptiveLayoutConfig, Layout, ShortLayoutOptimConfig
from damp.layout.mnist_layout import LayoutParameterTuner, TunedLayoutParams, _sparse_layout_options
from damp.layout.payload import LayoutEncodedCode, LayoutPayloadBuilder
from damp.logging import LOGGER


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
                "lambda_threshold": settings.LAYOUT_LAMBDA_THRESHOLD,
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
        output_path: str | None = None,
        component_output_path: str | None = None,
        energy_output_path: str | None = None,
        layout_path: str | None = None,
    ) -> None:
        self._encoder = encoder or PinwheelEncoderFactory.create()
        self._positions = settings.PINWHEEL_POSITIONS
        self._payload: LayoutPayloadBuilder | None = None
        self._layout_tuner = LayoutParameterTuner()
        # None — путь из настроек в момент создания.
        self._output_path = settings.PINWHEEL_OUTPUT_PATH if output_path is None else output_path
        self._component_output_path = settings.PINWHEEL_COMPONENT_OUTPUT_PATH if component_output_path is None else component_output_path
        self._energy_output_path = settings.PINWHEEL_ENERGY_OUTPUT_PATH if energy_output_path is None else energy_output_path
        self._layout_path = settings.PINWHEEL_LAYOUT_PATH if layout_path is None else layout_path
        self._quality = PinwheelQualityEvaluator(
            period=settings.MNIST_ANGLE_MAX - settings.MNIST_ANGLE_MIN,
            smooth_delta=settings.PINWHEEL_SMOOTH_DELTA,
        )

    @property
    def encoder(self) -> Encoder:
        return self._encoder

    def encode_codes(self) -> dict[float, list[LayoutEncodedCode]]:
        segments = {dim.title: self._encoder.segment(dim.title) for dim in self._encoder.dimensions}
        self._payload = LayoutPayloadBuilder(
            similarity=settings.LAYOUT_SIMILARITY,
            lambda_threshold=settings.LAYOUT_LAMBDA_THRESHOLD,
            eta=settings.LAYOUT_ETA,
            metadata={
                settings.ENCODER_SIGNATURE_KEY: self._encoder.signature,
                "stimulus_positions": [{"x": x, "y": y} for x, y in self._positions],
                "angles_per_position": settings.PINWHEEL_ANGLE_COUNT,
                "code_segments": segments,
            },
        )
        LOGGER.event(
            "pinwheel.stimulus",
            section=MULTI_DIM_CODES,
            data={
                "positions": self._positions,
                "segments": segments,
                "angles_per_position": settings.PINWHEEL_ANGLE_COUNT,
                "stimuli": settings.PINWHEEL_ANGLE_COUNT * len(self._positions),
            },
        )
        period = settings.MNIST_ANGLE_MAX - settings.MNIST_ANGLE_MIN
        for index in range(settings.PINWHEEL_ANGLE_COUNT):
            angle = settings.MNIST_ANGLE_MIN + period * index / settings.PINWHEEL_ANGLE_COUNT
            for x, y in self._positions:
                _, code = self._encoder.encode(angle, x, y)
                self._payload.add_code(label=f"angle={angle:.3f}, x={x:g}, y={y:g}", hue=angle, code=code)
        LOGGER.event(
            "pinwheel.codes",
            section=MULTI_DIM_CODES,
            data={
                "angles": settings.PINWHEEL_ANGLE_COUNT,
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
            empty_ratio=settings.LAYOUT_EMPTY_RATIO,
            similarity=settings.LAYOUT_SIMILARITY,
            lambda_threshold=settings.LAYOUT_LAMBDA_THRESHOLD,
            eta=settings.LAYOUT_ETA,
            seed=settings.LAYOUT_RANDOM_SEED,
            use_gpu=settings.LAYOUT_USE_GPU,
            long_subset_size=run_params.long_subset_size,
            long_subset_refresh=run_params.long_subset_refresh,
            colorizer=EncoderCodeColorizer(
                min_hue=settings.LAYOUT_PSEUDO_COLOR_MIN_HUE,
                max_hue=settings.LAYOUT_PSEUDO_COLOR_MAX_HUE,
            ),
            long_distance=settings.LAYOUT_LONG_DISTANCE,
            short_distance=settings.LAYOUT_SHORT_DISTANCE,
            long_kernel=settings.LAYOUT_LONG_KERNEL,
            long_kernel_radius_min=settings.LAYOUT_LONG_KERNEL_RADIUS_MIN,
            long_kernel_width=settings.LAYOUT_LONG_KERNEL_WIDTH,
            component_period=settings.MNIST_ANGLE_MAX - settings.MNIST_ANGLE_MIN,
            **_sparse_layout_options(),
        )

    def run(self, *, save: bool = True) -> tuple[Layout, PinwheelQuality]:
        codes = self.encode_codes()
        run_params = self._layout_tuner.tune(sum(len(bucket) for bucket in codes.values()))
        layout = self.build_layout(codes, params=run_params)
        long_radius = max(
            settings.LAYOUT_ADAPTIVE_RADIUS_MIN,
            int(layout.width * settings.LAYOUT_ADAPTIVE_RADIUS_START_FACTOR),
        )
        adaptive_long = AdaptiveLayoutConfig(
            start_radius=long_radius,
            end_radius=settings.LAYOUT_ADAPTIVE_RADIUS_MIN,
            swap_ratio_trigger=settings.LAYOUT_ADAPTIVE_SWAP_TRIGGER,
            lambda_step=settings.LAYOUT_ADAPTIVE_LAMBDA_STEP,
            lambda_max=settings.LAYOUT_ADAPTIVE_LAMBDA_MAX,
            radius_schedule=settings.LAYOUT_ADAPTIVE_RADIUS_SCHEDULE,
            radius_factor=settings.LAYOUT_ADAPTIVE_RADIUS_TRIGGER_FACTOR,
            swap_window=settings.LAYOUT_ADAPTIVE_SWAP_WINDOW,
            temperature=settings.LAYOUT_ANNEAL_TEMPERATURE,
            temperature_radius_max=int(layout.width * settings.LAYOUT_ANNEAL_RADIUS_MAX_FACTOR),
            temperature_radius_min=settings.LAYOUT_ANNEAL_RADIUS_MIN,
        )
        layout.run(
            steps=run_params.long_steps,
            pairs_per_step=run_params.long_pairs_per_step,
            pair_radius=long_radius,
            mode="long",
            min_swap_ratio=settings.LAYOUT_MIN_SWAP_RATIO,
            min_swap_window=settings.LAYOUT_MIN_SWAP_WINDOW,
            log_every=settings.LAYOUT_LOG_EVERY_LONG,
            step_offset=1,
            adaptive_params=adaptive_long,
            energy_radius=settings.LAYOUT_ENERGY_RADIUS,
            energy_every=settings.LAYOUT_ENERGY_EVAL_EVERY,
            energy_target=settings.LAYOUT_ENERGY_TARGET,
            log_visuals=settings.LAYOUT_LOG_VISUALS,
        )
        step_offset = 1 + layout.last_steps
        short_local_radius = run_params.short_local_radius
        short_radius_start = max(
            settings.LAYOUT_ADAPTIVE_RADIUS_MIN,
            min(
                int(max(1, round(short_local_radius * settings.LAYOUT_SHORT_ADAPTIVE_RADIUS_FACTOR))),
                int(layout.width * settings.LAYOUT_ADAPTIVE_RADIUS_START_FACTOR),
            ),
        )
        adaptive_short = AdaptiveLayoutConfig(
            start_radius=short_radius_start,
            end_radius=settings.LAYOUT_ADAPTIVE_RADIUS_MIN,
            swap_ratio_trigger=settings.LAYOUT_ADAPTIVE_SWAP_TRIGGER,
            lambda_step=settings.LAYOUT_ADAPTIVE_LAMBDA_STEP,
            lambda_max=settings.LAYOUT_ADAPTIVE_LAMBDA_MAX,
            radius_schedule=settings.LAYOUT_SHORT_ADAPTIVE_RADIUS_SCHEDULE,
            radius_factor=settings.LAYOUT_ADAPTIVE_RADIUS_TRIGGER_FACTOR,
            swap_window=settings.LAYOUT_ADAPTIVE_SWAP_WINDOW,
        )
        short_optim = ShortLayoutOptimConfig(
            energy_radius=run_params.short_energy_radius,
            energy_recalc_every=run_params.short_energy_recalc_every,
            energy_eps=settings.LAYOUT_SHORT_ENERGY_EPS,
            use_weighted_first_point=settings.LAYOUT_SHORT_WEIGHTED_FIRST,
            similarity_cutoff=settings.LAYOUT_SHORT_SIMILARITY_CUTOFF,
            partitions=settings.LAYOUT_SHORT_PARTITIONS,
        )
        layout.run(
            steps=run_params.short_steps,
            pairs_per_step=run_params.short_pairs_per_step,
            pair_radius=short_radius_start,
            mode="short",
            local_radius=short_local_radius,
            min_swap_ratio=settings.LAYOUT_SHORT_MIN_SWAP_RATIO,
            min_swap_window=settings.LAYOUT_MIN_SWAP_WINDOW,
            log_every=settings.LAYOUT_LOG_EVERY_SHORT,
            step_offset=step_offset,
            adaptive_params=adaptive_short,
            short_optim=short_optim,
            log_visuals=settings.LAYOUT_LOG_VISUALS,
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
            layout.render_component_image(period=settings.MNIST_ANGLE_MAX - settings.MNIST_ANGLE_MIN)
        ).save(self._component_output_path)
        Image.fromarray(
            layout.render_energy_image(layout.point_energies(settings.ACTIVATION_POINT_ENERGY_RADIUS))
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
