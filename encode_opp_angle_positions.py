from __future__ import annotations

from BitArray import BitArray
from Encoder import Encoder
from LayerConfig import LayerConfig


ANGLE_LAYERS = [
    LayerConfig(detectors=2, overlap=0.9),
    LayerConfig(detectors=4, overlap=0.6),
    LayerConfig(detectors=8, overlap=0.4),
    LayerConfig(detectors=16, overlap=0.3),
    LayerConfig(detectors=32, overlap=0.2),
    LayerConfig(detectors=64, overlap=0.15),
    LayerConfig(detectors=128, overlap=0.1),
    LayerConfig(detectors=180, overlap=0.05),
]

POSITION_LAYERS = [
    LayerConfig(detectors=2, overlap=0.6),
    LayerConfig(detectors=4, overlap=0.4),
    LayerConfig(detectors=8, overlap=0.3),
    LayerConfig(detectors=16, overlap=0.2),
    LayerConfig(detectors=32, overlap=0.1),
]

# Two sample angle-position points (x, y, angle) aligned with OPP's 0-180 orientation range.
ANGLE_POSITIONS: list[tuple[str, float, float, float]] = [
    ("A", 0.0, 0.0, 0.0),
    ("B", 1.0, 0.0, 90.0),
]


def _encode_angle_position(
    angle: float, x: float, y: float, angle_encoder: Encoder, position_encoder: Encoder
) -> BitArray:
    total_bits = angle_encoder.code_bits + position_encoder.code_bits * 2
    code = BitArray(total_bits)
    for idx in angle_encoder.encode_sparse(angle):
        code.set(idx)
    offset = angle_encoder.code_bits
    for idx in position_encoder.encode_sparse(x):
        code.set(offset + idx)
    offset += position_encoder.code_bits
    for idx in position_encoder.encode_sparse(y):
        code.set(offset + idx)
    return code


def main() -> None:
    angle_encoder = Encoder(
        code_bits=sum(layer.detectors for layer in ANGLE_LAYERS),
        layers=ANGLE_LAYERS,
        bits_per_detector=1,
        seed=0,
        bit_assignment="detector_index",
        coordinate_mode="cyclic",
        cycle_length=180.0,
    )
    position_encoder = Encoder(
        code_bits=sum(layer.detectors for layer in POSITION_LAYERS),
        layers=POSITION_LAYERS,
        bits_per_detector=1,
        seed=1,
        bit_assignment="detector_index",
        coordinate_mode="linear",
        value_range=(0.0, 49.0),
    )
    total_bits = angle_encoder.code_bits + position_encoder.code_bits * 2
    for label, x, y, angle in ANGLE_POSITIONS:
        code = _encode_angle_position(angle, x, y, angle_encoder, position_encoder)
        density = (code.count() / total_bits) * 100.0
        print(
            f"{label} x={x:.2f} y={y:.2f} angle={angle:.1f} "
            f"density={density:.2f}% code={code.to01()}"
        )


if __name__ == "__main__":
    main()
