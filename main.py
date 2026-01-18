import math
from collections import defaultdict
from dataclasses import dataclass

from damp.article_refs import LAYOUT_PARAMETERS, LAYOUT_SYSTEM
from damp.MnistSobelAngleMap import MnistSobelAngleMap
from damp.encoding.damp_encoder import ClosedDimension, Detectors, Encoder, OpenedDimension
from damp.layout.damp_layout import AdaptiveLayoutConfig, Layout, ShortLayoutOptimConfig
from damp.logging import LOGGER

LOG_INTERVAL_DEFAULT = 1
LOG_INTERVAL_INIT = 1
LOG_INTERVAL_SOBEL_IMAGE = 25
LOG_INTERVAL_SOBEL_PATCH = 2000
LOG_INTERVAL_ENCODER_ENCODE = 100
LOG_INTERVAL_ENCODER_IMAGE = 10
LOG_INTERVAL_LAYOUT_ENERGY = 1000
LOG_INTERVAL_LAYOUT_PAIR = 250000
LOG_INTERVAL_LAYOUT_AVG_ENERGY = 200
LOG_INTERVAL_LAYOUT_VISUAL = 1
LOG_INTERVAL_LAYOUT_ADAPTIVE = 200
LOG_INTERVAL_LAYOUT_SWAP_RATIO = 50
LOG_INTERVAL_LAYOUT_EXPORT = 1
LOG_INTERVAL_LAYOUT_ENERGY_PROGRESS = 1

LAYOUT_SIMILARITY = "cosine"
LAYOUT_LAMBDA_THRESHOLD = 0.65
LAYOUT_ETA = 0
LAYOUT_EMPTY_RATIO = 0.5
LAYOUT_USE_GPU = True

LAYOUT_LONG_PAIRS_PER_STEP = 24000
LAYOUT_LONG_STEPS = 14000
LAYOUT_LONG_SUBSET_SIZE = 1024
LAYOUT_LONG_SUBSET_REFRESH = 40

LAYOUT_SHORT_PAIRS_PER_STEP = 100
LAYOUT_SHORT_STEPS = 4000
LAYOUT_SHORT_LOCAL_RADIUS = 7
LAYOUT_SHORT_ENERGY_RADIUS = 7
LAYOUT_SHORT_ENERGY_RECALC_EVERY = 75
LAYOUT_SHORT_ENERGY_MAX_POINTS = 256
LAYOUT_SHORT_ENERGY_EPS = 1e-6
LAYOUT_SHORT_WEIGHTED_FIRST = True
LAYOUT_SHORT_SIMILARITY_CUTOFF = LAYOUT_LAMBDA_THRESHOLD
LAYOUT_SHORT_PARTITIONS = 8

ENCODER_LOG_EVERY = 50
LAYOUT_LOG_EVERY_LONG = 150
LAYOUT_LOG_EVERY_SHORT = 50
LAYOUT_LOG_VISUALS = True
LAYOUT_ADAPTIVE_RADIUS_START_FACTOR = 0.5
LAYOUT_ADAPTIVE_RADIUS_MIN = 1
LAYOUT_ADAPTIVE_SWAP_TRIGGER = 0.01
LAYOUT_ADAPTIVE_LAMBDA_STEP = 0.05
LAYOUT_SHORT_ADAPTIVE_RADIUS_FACTOR = 1.0
LAYOUT_ENERGY_STABILITY_WINDOW = 20
LAYOUT_ENERGY_STABILITY_DELTA = 0.0005
LAYOUT_ENERGY_STABILITY_EVERY = 200
LAYOUT_ENERGY_STABILITY_MAX_POINTS = 128
LAYOUT_MIN_SWAP_RATIO = 0.003
LAYOUT_MIN_SWAP_WINDOW = 50

LAYOUT_TUNE_LONG_PAIR_FACTOR = 0.35
LAYOUT_TUNE_LONG_PAIRS_MIN = 4000
LAYOUT_TUNE_LONG_PAIRS_MAX = 20000
LAYOUT_TUNE_LONG_STEPS_DIVISOR = 25
LAYOUT_TUNE_LONG_STEPS_MIN = 300
LAYOUT_TUNE_LONG_STEPS_MAX = 1500
LAYOUT_TUNE_LONG_SUBSET_MULTIPLIER = 8
LAYOUT_TUNE_LONG_SUBSET_MIN = 512
LAYOUT_TUNE_LONG_SUBSET_MAX = 2048
LAYOUT_TUNE_LONG_SUBSET_REFRESH_DIVISOR = 18
LAYOUT_TUNE_LONG_SUBSET_REFRESH_MIN = 20
LAYOUT_TUNE_LONG_SUBSET_REFRESH_MAX = 80
LAYOUT_TUNE_SHORT_PAIR_FACTOR = 0.2
LAYOUT_TUNE_SHORT_PAIRS_MIN = 1800
LAYOUT_TUNE_SHORT_PAIRS_MAX = 12000
LAYOUT_TUNE_SHORT_STEPS_FACTOR = 1.6
LAYOUT_TUNE_SHORT_STEPS_MIN = 160
LAYOUT_TUNE_SHORT_STEPS_MAX = 900
LAYOUT_TUNE_SHORT_RADIUS_FACTOR = 0.05
LAYOUT_TUNE_SHORT_ENERGY_RADIUS_FACTOR = 0.04
LAYOUT_TUNE_SHORT_ENERGY_RECALC_FLOOR = 30
LOG_INTERVALS = {
    "detectors.init": LOG_INTERVAL_INIT,
    "dimension.closed": LOG_INTERVAL_INIT,
    "dimension.init": LOG_INTERVAL_INIT,
    "encoder.code_length": LOG_INTERVAL_INIT,
    "encoder.detectors": LOG_INTERVAL_INIT,
    "encoder.encode.code": LOG_INTERVAL_ENCODER_ENCODE,
    "encoder.encode.image": LOG_INTERVAL_ENCODER_IMAGE,
    "encoder.encode.input": LOG_INTERVAL_ENCODER_ENCODE,
    "encoder.encode.normalized": LOG_INTERVAL_ENCODER_ENCODE,
    "encoder.init": LOG_INTERVAL_INIT,
    "encoder.layer": LOG_INTERVAL_INIT,
    "encoder.random_bit": LOG_INTERVAL_INIT,
    "layout.codes": LOG_INTERVAL_INIT,
    "layout.distance_eps": LOG_INTERVAL_INIT,
    "layout.energy.average": LOG_INTERVAL_LAYOUT_AVG_ENERGY,
    "layout.energy.long.empty": LOG_INTERVAL_LAYOUT_ENERGY,
    "layout.energy.long.pair": LOG_INTERVAL_LAYOUT_PAIR,
    "layout.energy.long.sim_lambda": LOG_INTERVAL_LAYOUT_ENERGY,
    "layout.energy.long.similarity": LOG_INTERVAL_LAYOUT_ENERGY,
    "layout.energy.long.space": LOG_INTERVAL_LAYOUT_ENERGY,
    "layout.energy.long.threshold": LOG_INTERVAL_LAYOUT_ENERGY,
    "layout.energy.long.tensor": LOG_INTERVAL_LAYOUT_ENERGY,
    "layout.energy.short.tensor": LOG_INTERVAL_LAYOUT_ENERGY,
    "layout.energy.short.tensor.empty_subset": LOG_INTERVAL_LAYOUT_ENERGY,
    "layout.energy.pair.ignore_self": LOG_INTERVAL_LAYOUT_ENERGY,
    "layout.energy.long.progress": LOG_INTERVAL_LAYOUT_ENERGY_PROGRESS,
    "layout.gpu.config": LOG_INTERVAL_INIT,
    "layout.gpu.context_failed": LOG_INTERVAL_INIT,
    "layout.gpu.disabled": LOG_INTERVAL_INIT,
    "layout.gpu.enabled": LOG_INTERVAL_INIT,
    "layout.gpu.import_failed": LOG_INTERVAL_INIT,
    "layout.gpu.init_failed": LOG_INTERVAL_INIT,
    "layout.gpu.tensor.disabled": LOG_INTERVAL_INIT,
    "layout.gpu.tensor.enabled": LOG_INTERVAL_INIT,
    "layout.gpu.tensor.unavailable": LOG_INTERVAL_INIT,
    "layout.gpu.tensor.short.disabled": LOG_INTERVAL_INIT,
    "layout.gpu.tensor.short.sim_failed": LOG_INTERVAL_INIT,
    "layout.gpu.version_too_low": LOG_INTERVAL_INIT,
    "layout.grid": LOG_INTERVAL_INIT,
    "layout.init": LOG_INTERVAL_INIT,
    "layout.parallel": LOG_INTERVAL_INIT,
    "layout.parallel.cpu_count": LOG_INTERVAL_INIT,
    "layout.precompute": LOG_INTERVAL_INIT,
    "layout.precompute_limit": LOG_INTERVAL_INIT,
    "layout.render_image": LOG_INTERVAL_LAYOUT_VISUAL,
    "layout.adaptive.config": LOG_INTERVAL_INIT,
    "layout.adaptive.radius": LOG_INTERVAL_LAYOUT_ADAPTIVE,
    "layout.adaptive.lambda": LOG_INTERVAL_INIT,
    "layout.run.done": LOG_INTERVAL_INIT,
    "layout.run.energy_monitor": LOG_INTERVAL_INIT,
    "layout.run.energy_stability": LOG_INTERVAL_INIT,
    "layout.run.energy_stability.stop": LOG_INTERVAL_INIT,
    "layout.run.mode": LOG_INTERVAL_INIT,
    "layout.run.selection": LOG_INTERVAL_INIT,
    "layout.run.settings": LOG_INTERVAL_INIT,
    "layout.run.swap_monitor": LOG_INTERVAL_INIT,
    "layout.run.swap_ratio": LOG_INTERVAL_LAYOUT_SWAP_RATIO,
    "layout.run.swap_stability.stop": LOG_INTERVAL_INIT,
    "layout.sim_cache.done": LOG_INTERVAL_INIT,
    "layout.sim_cache.gpu": LOG_INTERVAL_INIT,
    "layout.sim_cache.parallel": LOG_INTERVAL_INIT,
    "layout.sim_cache.similarity": LOG_INTERVAL_INIT,
    "layout.sim_cache.start": LOG_INTERVAL_INIT,
    "layout.similarity": LOG_INTERVAL_INIT,
    "layout.similarity_params": LOG_INTERVAL_INIT,
    "layout.thresholds": LOG_INTERVAL_INIT,
    "layout.visual": LOG_INTERVAL_LAYOUT_VISUAL,
    "layout.export.json": LOG_INTERVAL_LAYOUT_EXPORT,
    "layout.export.json.error": LOG_INTERVAL_LAYOUT_EXPORT,
    "sobel_map.extract": LOG_INTERVAL_SOBEL_IMAGE,
    "sobel_map.init": LOG_INTERVAL_INIT,
    "sobel_map.patch.angle": LOG_INTERVAL_SOBEL_PATCH,
    "sobel_map.patch.metrics": LOG_INTERVAL_SOBEL_PATCH,
    "sobel_map.patch_coords": LOG_INTERVAL_SOBEL_IMAGE,
    "sobel_map.patch_grid": LOG_INTERVAL_SOBEL_IMAGE,
}


def configure_logging() -> None:
    LOGGER.configure_intervals(LOG_INTERVALS, default_interval=LOG_INTERVAL_DEFAULT)


@dataclass(frozen=True)
class TunedLayoutParams:
    long_pairs_per_step: int
    long_steps: int
    long_subset_size: int | None
    long_subset_refresh: int
    short_pairs_per_step: int
    short_steps: int
    short_local_radius: int
    short_energy_radius: int
    short_energy_recalc_every: int


class LayoutParameterTuner:
    def __init__(
        self,
        *,
        empty_ratio: float = LAYOUT_EMPTY_RATIO,
        start_radius_factor: float = LAYOUT_ADAPTIVE_RADIUS_START_FACTOR,
    ) -> None:
        self._empty_ratio = empty_ratio
        self._start_radius_factor = start_radius_factor

    def tune(self, total_points: int) -> TunedLayoutParams:
        if total_points <= 0:
            raise ValueError("total_points must be positive for tuning")
        grid_size = int(math.ceil(math.sqrt(total_points * (1.0 + self._empty_ratio))))
        long_pairs = max(
            LAYOUT_TUNE_LONG_PAIRS_MIN,
            min(LAYOUT_TUNE_LONG_PAIRS_MAX, int(total_points * LAYOUT_TUNE_LONG_PAIR_FACTOR)),
        )
        long_steps = max(
            LAYOUT_TUNE_LONG_STEPS_MIN,
            min(LAYOUT_TUNE_LONG_STEPS_MAX, int(total_points / LAYOUT_TUNE_LONG_STEPS_DIVISOR)),
        )
        long_subset_size = min(
            LAYOUT_TUNE_LONG_SUBSET_MAX,
            max(LAYOUT_TUNE_LONG_SUBSET_MIN, int(math.sqrt(total_points) * LAYOUT_TUNE_LONG_SUBSET_MULTIPLIER)),
        )
        long_subset_refresh = max(
            LAYOUT_TUNE_LONG_SUBSET_REFRESH_MIN,
            min(
                LAYOUT_TUNE_LONG_SUBSET_REFRESH_MAX,
                long_steps // LAYOUT_TUNE_LONG_SUBSET_REFRESH_DIVISOR,
            ),
        )
        short_pairs = max(
            LAYOUT_TUNE_SHORT_PAIRS_MIN,
            min(LAYOUT_TUNE_SHORT_PAIRS_MAX, int(total_points * LAYOUT_TUNE_SHORT_PAIR_FACTOR)),
        )
        short_steps = max(
            LAYOUT_TUNE_SHORT_STEPS_MIN,
            min(LAYOUT_TUNE_SHORT_STEPS_MAX, int(grid_size * LAYOUT_TUNE_SHORT_STEPS_FACTOR)),
        )
        short_local_radius = max(
            LAYOUT_ADAPTIVE_RADIUS_MIN, int(grid_size * LAYOUT_TUNE_SHORT_RADIUS_FACTOR)
        )
        short_energy_radius = max(
            LAYOUT_ADAPTIVE_RADIUS_MIN, int(grid_size * LAYOUT_TUNE_SHORT_ENERGY_RADIUS_FACTOR)
        )
        short_energy_recalc_every = max(LAYOUT_TUNE_SHORT_ENERGY_RECALC_FLOOR, short_steps // 6)
        LOGGER.event(
            "layout.tune",
            section=LAYOUT_PARAMETERS,
            data={
                "points": total_points,
                "grid_size_est": grid_size,
                "empty_ratio": self._empty_ratio,
                "start_radius_factor": self._start_radius_factor,
                "long_pairs_per_step": long_pairs,
                "long_steps": long_steps,
                "long_subset_size": long_subset_size,
                "long_subset_refresh": long_subset_refresh,
                "short_pairs_per_step": short_pairs,
                "short_steps": short_steps,
                "short_local_radius": short_local_radius,
                "short_energy_radius": short_energy_radius,
                "short_energy_recalc_every": short_energy_recalc_every,
                "lambda_threshold": LAYOUT_LAMBDA_THRESHOLD,
                "eta": "hard" if LAYOUT_ETA is None else LAYOUT_ETA,
            },
        )
        return TunedLayoutParams(
            long_pairs_per_step=long_pairs,
            long_steps=long_steps,
            long_subset_size=long_subset_size,
            long_subset_refresh=long_subset_refresh,
            short_pairs_per_step=short_pairs,
            short_steps=short_steps,
            short_local_radius=short_local_radius,
            short_energy_radius=short_energy_radius,
            short_energy_recalc_every=short_energy_recalc_every,
        )


def _default_layout_params() -> TunedLayoutParams:
    return TunedLayoutParams(
        long_pairs_per_step=LAYOUT_LONG_PAIRS_PER_STEP,
        long_steps=LAYOUT_LONG_STEPS,
        long_subset_size=LAYOUT_LONG_SUBSET_SIZE,
        long_subset_refresh=LAYOUT_LONG_SUBSET_REFRESH,
        short_pairs_per_step=LAYOUT_SHORT_PAIRS_PER_STEP,
        short_steps=LAYOUT_SHORT_STEPS,
        short_local_radius=LAYOUT_SHORT_LOCAL_RADIUS,
        short_energy_radius=LAYOUT_SHORT_ENERGY_RADIUS,
        short_energy_recalc_every=LAYOUT_SHORT_ENERGY_RECALC_EVERY,
    )


def _build_encoder() -> Encoder:
    return Encoder(
        # Angle
        ClosedDimension(
            "Angle",
            (0.0, 360.0),
            [
                #Detectors(360, 0.4),
                #Detectors(180, 0.4),
                Detectors(90, 0.7),
                Detectors(45, 0.7),
                Detectors(30, 0.7),
                Detectors(10, 0.7),
                #Detectors(5, 0.4),
            ],
        ),
        # X
        OpenedDimension(
            "X",
            (0, 6),
            [
                Detectors(7, 0.7),
                Detectors(6, 0.7),
                Detectors(5, 0.7),
                Detectors(4, 0.7),
                Detectors(3, 0.7),
                Detectors(2, 0.7),
            ],
        ),
        # Y
        OpenedDimension(
            "Y",
            (0, 6),
            [
                Detectors(7, 0.7),
                Detectors(6, 0.7),
                Detectors(5, 0.7),
                Detectors(4, 0.7),
                Detectors(3, 0.7),
                Detectors(2, 0.7),
            ],
        ),
        log_every=ENCODER_LOG_EVERY,
    )


def _collect_codes(
    dataset,
    label: int | None,
    count: int,
    encoder: Encoder,
    extractor: MnistSobelAngleMap,
) -> tuple[dict[float, list], int]:
    codes = defaultdict(list)
    total_codes = 0
    found = 0

    for img_tensor, digit in dataset:
        if label is not None and int(digit) != label:
            continue
        found += 1
        if found > count:
            break
        img = img_tensor.squeeze(0).numpy()
        digit_values = extractor.extract(img, digit)
        measurements = digit_values[int(digit)]
        first = True
        for (a, x, y) in measurements:
            log_image = img if first else None
            log_label = int(digit) if first else None
            log_measurements = measurements if first else None
            _, code = encoder.encode(
                float(a),
                float(x),
                float(y),
                log_image=log_image,
                log_label=log_label,
                log_measurements=log_measurements,
            )
            codes[a].append(code)
            total_codes += 1
            first = False
    return codes, total_codes

SPR = 32
def _run_layout(codes: dict[float, list], params: TunedLayoutParams | None = None) -> Layout:
    run_params = params or _default_layout_params()
    layout = Layout(
        codes,
        empty_ratio=LAYOUT_EMPTY_RATIO,
        similarity=LAYOUT_SIMILARITY,
        lambda_threshold=LAYOUT_LAMBDA_THRESHOLD,
        eta=LAYOUT_ETA,
        seed=0,
        use_gpu=LAYOUT_USE_GPU,
        long_subset_size=run_params.long_subset_size,
        long_subset_refresh=run_params.long_subset_refresh,
    )
    step_offset = 1
    long_radius_start = max(
        LAYOUT_ADAPTIVE_RADIUS_MIN,
        int(layout.width * LAYOUT_ADAPTIVE_RADIUS_START_FACTOR),
    )
    adaptive_long = AdaptiveLayoutConfig(
        start_radius=long_radius_start,
        end_radius=LAYOUT_ADAPTIVE_RADIUS_MIN,
        swap_ratio_trigger=LAYOUT_ADAPTIVE_SWAP_TRIGGER,
        lambda_step=LAYOUT_ADAPTIVE_LAMBDA_STEP,
    )
    layout.run(
        steps=run_params.long_steps,
        pairs_per_step=run_params.long_pairs_per_step,
        pair_radius=adaptive_long.start_radius,
        mode="long",
        min_swap_ratio=LAYOUT_MIN_SWAP_RATIO,
        min_swap_window=LAYOUT_MIN_SWAP_WINDOW,
        log_every=LAYOUT_LOG_EVERY_LONG,
        step_offset=step_offset,
        energy_radius=None,
        energy_stability_window=LAYOUT_ENERGY_STABILITY_WINDOW,
        energy_stability_delta=LAYOUT_ENERGY_STABILITY_DELTA,
        energy_stability_every=LAYOUT_ENERGY_STABILITY_EVERY,
        energy_stability_max_points=LAYOUT_ENERGY_STABILITY_MAX_POINTS,
        log_visuals=LAYOUT_LOG_VISUALS,
        adaptive_params=adaptive_long,
    )
    step_offset += layout.last_steps
    short_radius_max = int(layout.width * LAYOUT_ADAPTIVE_RADIUS_START_FACTOR)
    short_radius_base = int(
        max(
            1,
            round(run_params.short_local_radius * LAYOUT_SHORT_ADAPTIVE_RADIUS_FACTOR),
        )
    )
    short_radius_start = max(
        LAYOUT_ADAPTIVE_RADIUS_MIN,
        min(short_radius_base, short_radius_max),
    )
    adaptive_short = AdaptiveLayoutConfig(
        start_radius=short_radius_start,
        end_radius=LAYOUT_ADAPTIVE_RADIUS_MIN,
        swap_ratio_trigger=LAYOUT_ADAPTIVE_SWAP_TRIGGER,
        lambda_step=LAYOUT_ADAPTIVE_LAMBDA_STEP,
    )
    short_optim = ShortLayoutOptimConfig(
        energy_radius=run_params.short_energy_radius,
        energy_max_points=LAYOUT_SHORT_ENERGY_MAX_POINTS,
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
        local_radius=run_params.short_local_radius,
        min_swap_ratio=LAYOUT_MIN_SWAP_RATIO,
        min_swap_window=LAYOUT_MIN_SWAP_WINDOW,
        log_every=LAYOUT_LOG_EVERY_SHORT,
        step_offset=step_offset,
        adaptive_params=adaptive_short,
        short_optim=short_optim,
    )
    return layout


class MnistLayoutRunner:
    def __init__(self, *, count: int, label: int | None = None) -> None:
        self.count = count
        self.label = label
        configure_logging()
        self.encoder = _build_encoder()
        self.extractor = MnistSobelAngleMap(angle_in_degrees=True, grad_threshold=0.05)

    def _build_dataset(self):
        from torchvision import transforms
        from torchvision.datasets import MNIST

        return MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )

    def _tune_layout(self, total_codes: int) -> TunedLayoutParams:
        tuner = LayoutParameterTuner(
            empty_ratio=LAYOUT_EMPTY_RATIO,
            start_radius_factor=LAYOUT_ADAPTIVE_RADIUS_START_FACTOR,
        )
        return tuner.tune(total_codes)

    def _log_dataset(self, angles: int, total_codes: int) -> None:
        LOGGER.event(
            "mnist.codes",
            section=LAYOUT_SYSTEM,
            data={
                "count": self.count,
                "label": "all" if self.label is None else int(self.label),
                "angles": angles,
                "points": total_codes,
            },
        )

    def _export_layout(self, layout: Layout) -> None:
        from PIL import Image

        image = layout.render_image()
        filename = f"data/{self.count}-{layout.pair_radius}.png"
        Image.fromarray(image).save(filename)
        layout.save_json("data/layout.json")

    def run(self) -> None:
        dataset = self._build_dataset()
        codes, total_codes = _collect_codes(
            dataset,
            self.label,
            self.count,
            self.encoder,
            self.extractor,
        )
        self._log_dataset(len(codes), total_codes)
        tuned_params = self._tune_layout(total_codes)
        layout = _run_layout(codes, tuned_params)
        self._export_layout(layout)


def main() -> None:
    runner = MnistLayoutRunner(count=6000, label=None)
    runner.run()


if __name__ == "__main__":
    main()
