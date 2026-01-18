from collections import defaultdict
import random

from main import (
    ENCODER_LOG_EVERY,
    LAYOUT_ADAPTIVE_LAMBDA_STEP,
    LAYOUT_ADAPTIVE_RADIUS_MIN,
    LAYOUT_ADAPTIVE_RADIUS_START_FACTOR,
    LAYOUT_ADAPTIVE_SWAP_TRIGGER,
    LAYOUT_LOG_EVERY_LONG,
    LAYOUT_LOG_EVERY_SHORT,
    LAYOUT_SHORT_ADAPTIVE_RADIUS_FACTOR,
    LAYOUT_LOG_VISUALS,
    LAYOUT_ENERGY_STABILITY_DELTA,
    LAYOUT_ENERGY_STABILITY_WINDOW,
    LAYOUT_ENERGY_STABILITY_EVERY,
    LAYOUT_ENERGY_STABILITY_MAX_POINTS,
    LAYOUT_MIN_SWAP_RATIO,
    LAYOUT_MIN_SWAP_WINDOW,
    LAYOUT_TUNE_SHORT_RADIUS_FACTOR,
    configure_logging,
)
from damp.encoding.damp_encoder import Encoder, ClosedDimension, Detectors
from damp.layout.damp_layout import AdaptiveLayoutConfig, Layout


def main() -> None:
    configure_logging()
    encoder = Encoder(
        ClosedDimension("Angle", (0.0, 360.0), [
            #Detectors(360,  0.7),
            #Detectors(180,  0.7),
            Detectors(90,   0.7),
            Detectors(45,   0.7),
            Detectors(30,   0.7),
            Detectors(10,   0.7),
        ]),
        log_every=ENCODER_LOG_EVERY,
    )

    total_codes = 0
    codes = defaultdict(list)

    for a in range(360):
        values, code = encoder.encode(float(a))
        codes[a].append(code)
        total_codes += 1
    
    for angle_codes in codes.values():
        random.shuffle(angle_codes)

    layout = Layout(
        codes,
        empty_ratio=0.5,
        similarity="cosine",
        lambda_threshold=0.00, # 0.06
        eta=0.0, # 1.0 - выворачивает пинвил в криветку мебиуса (0 - норм)
        seed=0,
        use_gpu=True,
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
        pairs_per_step=1200,
        pair_radius=adaptive_long.start_radius,
        mode="long",
        min_swap_ratio=LAYOUT_MIN_SWAP_RATIO,
        min_swap_window=LAYOUT_MIN_SWAP_WINDOW,
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
    short_local_radius = max(
        LAYOUT_ADAPTIVE_RADIUS_MIN,
        int(layout.width * LAYOUT_TUNE_SHORT_RADIUS_FACTOR),
    )
    short_radius_max = int(layout.width * LAYOUT_ADAPTIVE_RADIUS_START_FACTOR)
    short_radius_base = int(
        max(
            1,
            round(short_local_radius * LAYOUT_SHORT_ADAPTIVE_RADIUS_FACTOR),
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
    layout.run(
        steps=900,
        pairs_per_step=500,
        pair_radius=adaptive_short.start_radius,
        mode="short",
        local_radius=short_local_radius,
        min_swap_ratio=LAYOUT_MIN_SWAP_RATIO,
        min_swap_window=LAYOUT_MIN_SWAP_WINDOW,
        log_every=LAYOUT_LOG_EVERY_SHORT,
        step_offset=step_offset,
        adaptive_params=adaptive_short,
        energy_stability_window=LAYOUT_ENERGY_STABILITY_WINDOW,
        energy_stability_delta=LAYOUT_ENERGY_STABILITY_DELTA,
        energy_stability_every=LAYOUT_ENERGY_STABILITY_EVERY,
        energy_stability_max_points=LAYOUT_ENERGY_STABILITY_MAX_POINTS,
        log_visuals=LAYOUT_LOG_VISUALS,
    )

if __name__ == "__main__":
    main()


#26.18s
