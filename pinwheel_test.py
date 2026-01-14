from collections import defaultdict
import random

from main import (
    ENCODER_LOG_EVERY,
    LAYOUT_LOG_EVERY_LONG,
    LAYOUT_LOG_EVERY_SHORT,
    configure_logging,
)
from damp.encoding.damp_encoder import Encoder, ClosedDimension, Detectors
from damp.layout.damp_layout import Layout


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
        use_gpu=False,
    )
    step_offset = 1
    layout.run(
        steps=22000,
        pairs_per_step=1200,
        pair_radius=layout.width // 2,
        mode="long",
        min_swap_ratio=0.001,
        log_every=LAYOUT_LOG_EVERY_LONG,
        step_offset=step_offset,
        #energy_radius=7,
        #energy_check_every=5,
        #energy_delta=5e-4,
        #energy_patience=4,
    )
    step_offset += layout.last_steps
    layout.run(
        steps=900,
        pairs_per_step=500,
        pair_radius=7,
        mode="short",
        local_radius=7,
        min_swap_ratio=0.001,
        log_every=LAYOUT_LOG_EVERY_SHORT,
        step_offset=step_offset,
    )

if __name__ == "__main__":
    main()


#26.18s
