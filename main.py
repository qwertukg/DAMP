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

    for i in range(70):
        img_tensor, label = dataset[i]
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
            #show(encoder, values, code, img, int(label))
            codes[label].append(code)
            total_codes += 1
            
    print(f"{total_codes} codes saved to codes.json")
    


    layout = Layout(
        codes,
        lambda_start=0.6,
        lambda_end=0.85,
        rr_app_id="mnist_layout",
    )

    layout.layout(
        long_steps=200,
        short_steps=0,
        pairs_per_step=64,
        long_pair_radius=42,
        short_pair_radius=6,
        short_local_radius=6,
        visualize=True,
        visualize_every=1,
        energy_radius=5,
    )

    #data = {str(k): v for k, v in codes.items()}
    #Path("codes.json").write_text(
    #    json.dumps(data, ensure_ascii=False),
    #    encoding="utf-8"
    #)
            
    
    wait_for_close()



if __name__ == "__main__":
    main()
