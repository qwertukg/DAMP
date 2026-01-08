from damp_encoder import Encoder, ClosedDimension, OpenedDimension, Detectors, BitArray as EncoderBitArray
from visualize_detectors import show, wait_for_close
import numpy as np
from MnistSobelAngleMap import MnistSobelAngleMap
from torchvision.datasets import MNIST
from torchvision import transforms
from collections import defaultdict
import json
from pathlib import Path
from damp_layout import Layout
import rerun as rr
from BitArray import BitArray as LayoutBitArray


def main() -> None:
    
    encoder = Encoder(
        ClosedDimension("Angle", (0.0, 360.0), [
            Detectors(360, 0.4),
            Detectors(180, 0.4),
            Detectors(90, 0.4),
            Detectors(45, 0.4),
            Detectors(30, 0.4),
            Detectors(10, 0.4),
            Detectors(5, 0.4),
        ]),
    )

    total_codes = 0
    codes = defaultdict(list)

    for a in range(360):
        values, code = encoder.encode(float(a))
        print(f"Encoded to: {values} -> {code}")
        codes[a].append(code)
        total_codes += 1
        #show(encoder, values, code, None, int(a))
    
    print(f"{total_codes} codes saved to codes.json")

    wait_for_close()

    def to_layout_bitarray(code: EncoderBitArray) -> LayoutBitArray:
        layout_code = LayoutBitArray(len(code))
        for idx, bit in enumerate(code):
            if bit:
                layout_code.set(idx, 1)
        return layout_code

    layout_codes = []
    for angle, angle_codes in codes.items():
        angle_value = 0.0 if angle is None else float(angle)
        for code in angle_codes:
            layout_codes.append((to_layout_bitarray(code), angle_value, angle_value))

    layout = Layout(
        layout_codes,
        grid_size=None,
        similarity="jaccard",
        lambda_threshold=0.06,
        eta=14.0,
        seed=0,
    )
    rr.init("damp-layout")
    rr.spawn()
    layout.log_rerun(step=0)
    step_offset = 1
    layout.run(
        steps=22000,
        pairs_per_step=1200,
        pair_radius=layout.width // 2,
        mode="long",
        min_swap_ratio=0.0,
        log_every=1,
        step_offset=step_offset,
        energy_radius=7,
        energy_check_every=5,
        energy_delta=5e-4,
        energy_patience=4,
    )
    step_offset += layout.last_steps
    layout.set_similarity_params(lambda_threshold=0.16, eta=14.0)
    layout.run(
        steps=900,
        pairs_per_step=500,
        pair_radius=7,
        mode="short",
        local_radius=7,
        min_swap_ratio=0.005,
        log_every=1,
        step_offset=step_offset,
    )


if __name__ == "__main__":
    main()
