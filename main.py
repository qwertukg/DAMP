from collections import defaultdict

from PIL import Image
from torchvision import transforms
from torchvision.datasets import MNIST

from damp.MnistSobelAngleMap import MnistSobelAngleMap
from damp.encoding.damp_encoder import ClosedDimension, Detectors, Encoder, OpenedDimension
from damp.encoding.visualize_encoding import wait_for_close
from damp.layout.damp_layout import Layout


def _build_encoder() -> Encoder:
    return Encoder(
        # Angle
        ClosedDimension(
            "Angle",
            (0.0, 360.0),
            [
                Detectors(360, 0.4),
                Detectors(180, 0.4),
                Detectors(90, 0.4),
                Detectors(45, 0.4),
                Detectors(30, 0.4),
                Detectors(10, 0.4),
                Detectors(5, 0.4),
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
    )


def _collect_codes(
    dataset: MNIST,
    label: int,
    count: int,
    encoder: Encoder,
    extractor: MnistSobelAngleMap,
) -> tuple[dict[float, list], int]:
    codes = defaultdict(list)
    total_codes = 0
    found = 0

    for img_tensor, digit in dataset:
        if int(digit) != label:
            continue
        found += 1
        if found > count:
            break
        img = img_tensor.squeeze(0).numpy()
        digit_values = extractor.extract(img, digit)
        for (a, x, y) in digit_values[digit]:
            _, code = encoder.encode(float(a), float(x), float(y))
            codes[a].append(code)
            total_codes += 1
    return codes, total_codes


def _run_layout(codes: dict[float, list]) -> Layout:
    layout = Layout(
        codes,
        empty_ratio=0.5,
        similarity="cosine",
        lambda_threshold=0.06,
        eta=0.0,
        seed=0,
    )
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
    return layout


def main() -> None:
    encoder = _build_encoder()
    dataset = MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    extractor = MnistSobelAngleMap(angle_in_degrees=True, grad_threshold=0.05)

    count = 60
    label = 8

    codes, total_codes = _collect_codes(dataset, label, count, encoder, extractor)

    layout = _run_layout(codes)

    image = layout.render_image()
    filename = f"data/{label}-{count}-{total_codes}.png"
    Image.fromarray(image).save(filename)

    wait_for_close()


if __name__ == "__main__":
    main()
