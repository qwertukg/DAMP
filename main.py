from damp_encoder import Encoder, ClosedDimension, OpenedDimension, Detectors
from visualize_detectors import show, wait_for_close
import numpy as np
from MnistSobelAngleMap import MnistSobelAngleMap
from torchvision.datasets import MNIST
from torchvision import transforms
from collections import defaultdict



def main() -> None:
    
    encoder = Encoder(
        # Angle
        ClosedDimension((0.0, 360.0), [
            Detectors(360, 0.4),
            Detectors(180, 0.4),
            Detectors(90, 0.4),
            Detectors(45, 0.4),
            Detectors(30, 0.4),
            Detectors(10, 0.4),
            Detectors(5, 0.4),
        ]),
        # X
        OpenedDimension((0, 6), [
            Detectors(7, 0.4),
            Detectors(4, 0.4),
            Detectors(2, 0.4),
            Detectors(1, 0.4),
        ]),
        # Y
        OpenedDimension((0, 6), [
            Detectors(7, 0.4),
            Detectors(4, 0.4),
            Detectors(2, 0.4),
            Detectors(1, 0.4),
        ]),
    )

    codes = defaultdict(list)
    ds = MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    extractor = MnistSobelAngleMap(angle_in_degrees=True, grad_threshold=0.05)
    for i in range(100):
        img_tensor, label = ds[i]
        img = img_tensor.squeeze(0).numpy()

        digitValues = extractor.extract(img, label)

        print("label:", label)
        print("кол-во квадратов:", len(digitValues[label]))
        print("первые 10:", digitValues[label][:10])

        for (a, x, y) in digitValues[label]:
            print(f"{label} -> \t{a}\t{x}\t{y}")
            values, code = encoder.encode(
                float(a), 
                float(x), 
                float(y)
            )
            print(f"{values} -> {code}")
            show(encoder, values, code)
            codes[label].extend(code)




                
    
    wait_for_close()



if __name__ == "__main__":
    main()
