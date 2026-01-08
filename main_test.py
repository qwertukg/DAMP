from damp_encoder import Encoder, ClosedDimension, OpenedDimension, Detectors
from visualize_detectors import show, wait_for_close
import numpy as np
from MnistSobelAngleMap import MnistSobelAngleMap
from torchvision.datasets import MNIST
from torchvision import transforms
from collections import defaultdict
import json
from pathlib import Path
from layout import Layout


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
    
    print(f"{total_codes} codes saved to codes.json")


    layout = Layout(
        codes,
        max_codes=70000,
        lambda_start=0.3,
        lambda_end=0.95,
        rr_app_id="mnist_layout",
    )

    layout.layout(
        long_steps=500,
        short_steps=0,
        pairs_per_step=64,
        long_pair_radius=9,
        short_pair_radius=6,
        short_local_radius=6,
        visualize=True,
        visualize_every=1,
        energy_radius=5,
    )

    wait_for_close()



if __name__ == "__main__":
    main()
