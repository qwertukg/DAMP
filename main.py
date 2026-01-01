from __future__ import annotations

from encoder import Encoder, LayerConfig


def main() -> None:
    layers = [
        LayerConfig(detectors=360, window=2),
        LayerConfig(detectors=360, window=2),
        LayerConfig(detectors=360, window=2),
        LayerConfig(detectors=360, window=1),
        LayerConfig(detectors=360, window=1),
    ]
    encoder = Encoder(
        code_bits=32,
        layers=layers,
        bits_per_detector=1,
        seed=0,
    )
    encoder.print_codes(start=0.0, end=360.0, step=1.0)


if __name__ == "__main__":
    main()
