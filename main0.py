from encoding.damp_encoder import Encoder, ClosedDimension, OpenedDimension, Detectors
from encoding.visualize_encoding import show, wait_for_close
import numpy as np
from encoding.MnistSobelAngleMap import MnistSobelAngleMap
from torchvision.datasets import MNIST
from torchvision import transforms
from collections import defaultdict
import json
from pathlib import Path
from layout.damp_layout import Layout
import rerun as rr


def main() -> None:
    
    encoder = Encoder(
        # Angle
        ClosedDimension("Angle", (0.0, 360.0), [
            Detectors(360, 0.4),
            Detectors(180, 0.4),
            Detectors(90, 0.4),
            Detectors(45, 0.4),
            Detectors(30, 0.4),
            Detectors(10, 0.4),
            Detectors(5, 0.4),
        ]),
        # X
        OpenedDimension("X", (0, 6), [
            Detectors(7, 0.4),
            Detectors(4, 0.4),
            Detectors(2, 0.4),
            Detectors(1, 0.4),
        ]),
        # Y
        OpenedDimension("Y", (0, 6), [
            Detectors(7, 0.4),
            Detectors(4, 0.4),
            Detectors(2, 0.4),
            Detectors(1, 0.4),
        ]),
    )

    total_codes = 0
    codes = defaultdict(list)
    dataset = MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    extractor = MnistSobelAngleMap(angle_in_degrees=True, grad_threshold=0.05)


    value = 0


    count = 600


    digits = []
    for img_tensor, label in dataset:
        if int(label) == value:
            digits.append((img_tensor, label))
            if len(digits) == count:
                break

    for i in range(len(digits)):
        img_tensor, label = digits[i]
        img = img_tensor.squeeze(0).numpy()

        digitValues = extractor.extract(img, label)

        print("label:", label)
        print("кол-во квадратов:", len(digitValues[label]))

        for (a, x, y) in digitValues[label]:
            print(f"Encoding: {label} -> \t{a}\t{x}\t{y}")
            values, code = encoder.encode(
                float(a), 
                float(x), 
                float(y)
            )
            print(f"Encoded to: {values} -> {code}")
            codes[a].append(code)
            total_codes += 1
            #show(encoder, values, code, img, int(label))
            
            
    print(f"{label}-{count}-{total_codes}")
    


    rr.init("damp-layout")
    rr.spawn()
    layout = Layout(
        codes,
        empty_ratio=0.5,
        similarity="cosine",
        lambda_threshold=0.06,
        eta=0.0,
        seed=0,
    )
    layout.log_rerun(step=0)
    step_offset = 1
    layout.run(
        steps=22000,
        pairs_per_step=1200,
        pair_radius=layout.width // 2,
        mode="long",
        min_swap_ratio=0.001,
        log_every=1,
        step_offset=step_offset,
        energy_radius=7,
        energy_check_every=5,
        energy_delta=5e-4,
        energy_patience=4,
    )
    step_offset += layout.last_steps
    layout.run(
        steps=900,
        pairs_per_step=500,
        pair_radius=7,
        mode="short",
        local_radius=7,
        min_swap_ratio=0.001,
        log_every=1,
        step_offset=step_offset,
    )

    wait_for_close()



if __name__ == "__main__":
    main()
