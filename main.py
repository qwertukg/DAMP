from collections import defaultdict

from damp.MnistSobelAngleMap import MnistSobelAngleMap
from damp.encoding.damp_encoder import ClosedDimension, Detectors, Encoder, OpenedDimension
from damp.layout.damp_layout import AdaptiveLayoutConfig, Layout
from damp.logging import LOGGER

LOG_INTERVAL_DEFAULT = 1
LOG_INTERVAL_INIT = 1
LOG_INTERVAL_SOBEL_IMAGE = 25
LOG_INTERVAL_SOBEL_PATCH = 2000
LOG_INTERVAL_ENCODER_ENCODE = 100
LOG_INTERVAL_ENCODER_IMAGE = 10
LOG_INTERVAL_LAYOUT_ENERGY = 200
LOG_INTERVAL_LAYOUT_PAIR = 100000
LOG_INTERVAL_LAYOUT_AVG_ENERGY = 20
LOG_INTERVAL_LAYOUT_VISUAL = 1
LOG_INTERVAL_LAYOUT_ADAPTIVE = 50

LAYOUT_USE_GPU = True

ENCODER_LOG_EVERY = 50
LAYOUT_LOG_EVERY_LONG = 10
LAYOUT_LOG_EVERY_SHORT = 10
LAYOUT_ADAPTIVE_RADIUS_START_FACTOR = 0.5
LAYOUT_ADAPTIVE_RADIUS_MIN = 1
LAYOUT_ADAPTIVE_SWAP_TRIGGER = 0.01
LAYOUT_ADAPTIVE_LAMBDA_STEP = 0.05
LAYOUT_ENERGY_STABILITY_WINDOW = 50
LAYOUT_ENERGY_STABILITY_DELTA = 0.0001
LAYOUT_ENERGY_STABILITY_EVERY = 50
LAYOUT_ENERGY_STABILITY_MAX_POINTS = 512

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
    "layout.energy.pair.ignore_self": LOG_INTERVAL_LAYOUT_ENERGY,
    "layout.gpu.config": LOG_INTERVAL_INIT,
    "layout.gpu.context_failed": LOG_INTERVAL_INIT,
    "layout.gpu.disabled": LOG_INTERVAL_INIT,
    "layout.gpu.enabled": LOG_INTERVAL_INIT,
    "layout.gpu.import_failed": LOG_INTERVAL_INIT,
    "layout.gpu.init_failed": LOG_INTERVAL_INIT,
    "layout.gpu.tensor.disabled": LOG_INTERVAL_INIT,
    "layout.gpu.tensor.enabled": LOG_INTERVAL_INIT,
    "layout.gpu.tensor.unavailable": LOG_INTERVAL_INIT,
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
    "layout.sim_cache.done": LOG_INTERVAL_INIT,
    "layout.sim_cache.gpu": LOG_INTERVAL_INIT,
    "layout.sim_cache.parallel": LOG_INTERVAL_INIT,
    "layout.sim_cache.similarity": LOG_INTERVAL_INIT,
    "layout.sim_cache.start": LOG_INTERVAL_INIT,
    "layout.similarity": LOG_INTERVAL_INIT,
    "layout.similarity_params": LOG_INTERVAL_INIT,
    "layout.thresholds": LOG_INTERVAL_INIT,
    "layout.visual": LOG_INTERVAL_LAYOUT_VISUAL,
    "sobel_map.extract": LOG_INTERVAL_SOBEL_IMAGE,
    "sobel_map.init": LOG_INTERVAL_INIT,
    "sobel_map.patch.angle": LOG_INTERVAL_SOBEL_PATCH,
    "sobel_map.patch.metrics": LOG_INTERVAL_SOBEL_PATCH,
    "sobel_map.patch_coords": LOG_INTERVAL_SOBEL_IMAGE,
    "sobel_map.patch_grid": LOG_INTERVAL_SOBEL_IMAGE,
}


def configure_logging() -> None:
    LOGGER.configure_intervals(LOG_INTERVALS, default_interval=LOG_INTERVAL_DEFAULT)


def _build_encoder() -> Encoder:
    return Encoder(
        # Angle
        ClosedDimension(
            "Angle",
            (0.0, 360.0),
            [
                #Detectors(360, 0.4),
                #Detectors(180, 0.4),
                Detectors(90, 0.4),
                Detectors(45, 0.4),
                Detectors(30, 0.4),
                Detectors(10, 0.4),
                #Detectors(5, 0.4),
            ],
        ),
        # X
        OpenedDimension(
            "X",
            (0, 6),
            [
                Detectors(7, 0.4),
                Detectors(4, 0.4),
                Detectors(2, 0.4),
                Detectors(1, 0.4),
            ],
        ),
        # Y
        OpenedDimension(
            "Y",
            (0, 6),
            [
                Detectors(7, 0.4),
                Detectors(4, 0.4),
                Detectors(2, 0.4),
                Detectors(1, 0.4),
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


def _run_layout(codes: dict[float, list]) -> Layout:
    layout = Layout(
        codes,
        empty_ratio=0.15,
        similarity="cosine",
        lambda_threshold=0.06,
        eta=0.0,
        seed=0,
        use_gpu=LAYOUT_USE_GPU,
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
        steps=22000,
        pairs_per_step=16000,
        pair_radius=adaptive_long.start_radius,
        mode="long",
        min_swap_ratio=0.0,
        log_every=LAYOUT_LOG_EVERY_LONG,
        step_offset=step_offset,
        energy_radius=None,
        energy_stability_window=LAYOUT_ENERGY_STABILITY_WINDOW,
        energy_stability_delta=LAYOUT_ENERGY_STABILITY_DELTA,
        energy_stability_every=LAYOUT_ENERGY_STABILITY_EVERY,
        energy_stability_max_points=LAYOUT_ENERGY_STABILITY_MAX_POINTS,
        adaptive_params=adaptive_long,
    )
    step_offset += layout.last_steps
    short_radius_start = max(
        LAYOUT_ADAPTIVE_RADIUS_MIN,
        int(layout.width * LAYOUT_ADAPTIVE_RADIUS_START_FACTOR),
    )
    adaptive_short = AdaptiveLayoutConfig(
        start_radius=short_radius_start,
        end_radius=LAYOUT_ADAPTIVE_RADIUS_MIN,
        swap_ratio_trigger=LAYOUT_ADAPTIVE_SWAP_TRIGGER,
        lambda_step=LAYOUT_ADAPTIVE_LAMBDA_STEP,
    )
    layout.run(
        steps=900,
        pairs_per_step=16000,
        pair_radius=adaptive_short.start_radius,
        mode="short",
        local_radius=adaptive_short.start_radius,
        min_swap_ratio=0.0,
        log_every=LAYOUT_LOG_EVERY_SHORT,
        step_offset=step_offset,
        adaptive_params=adaptive_short,
        energy_stability_window=LAYOUT_ENERGY_STABILITY_WINDOW,
        energy_stability_delta=LAYOUT_ENERGY_STABILITY_DELTA,
        energy_stability_every=LAYOUT_ENERGY_STABILITY_EVERY,
        energy_stability_max_points=LAYOUT_ENERGY_STABILITY_MAX_POINTS,
    )
    return layout


def main() -> None:
    from PIL import Image
    from torchvision import transforms
    from torchvision.datasets import MNIST

    configure_logging()
    encoder = _build_encoder()
    dataset = MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    extractor = MnistSobelAngleMap(angle_in_degrees=True, grad_threshold=0.05)

    count = 6000
    label = None

    codes, total_codes = _collect_codes(dataset, label, count, encoder, extractor)

    layout = _run_layout(codes)

    image = layout.render_image()
    filename = f"data/{count}-{total_codes}.png"
    Image.fromarray(image).save(filename)


if __name__ == "__main__":
    main()
