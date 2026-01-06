from __future__ import annotations

import rerun as rr

from BitArray import BitArray
from Encoder import Encoder
from EncoderReporter import EncoderReporter
from LayerConfig import LayerConfig
from Layout import Layout


def main() -> None:
    layers_angle = [
        LayerConfig(detectors=2,    overlap=0.9),
        LayerConfig(detectors=4,    overlap=0.6),
        LayerConfig(detectors=8,    overlap=0.4),
        LayerConfig(detectors=16,   overlap=0.3),
        LayerConfig(detectors=32,   overlap=0.2),
        LayerConfig(detectors=64,   overlap=0.15),
        LayerConfig(detectors=128,  overlap=0.1),
        LayerConfig(detectors=180,  overlap=0.05),
    ]
    layers_position = [
        LayerConfig(detectors=2,    overlap=0.6),
        LayerConfig(detectors=4,    overlap=0.4),
        LayerConfig(detectors=8,    overlap=0.3),
        LayerConfig(detectors=16,   overlap=0.2),
        LayerConfig(detectors=32,   overlap=0.1),
    ]
    angle_encoder = Encoder(
        code_bits=sum(layer.detectors for layer in layers_angle),
        layers=layers_angle,
        bits_per_detector=1,
        seed=0,
        bit_assignment="detector_index",
        coordinate_mode="cyclic",
        cycle_length=180.0,
    )
    position_encoder = Encoder(
        code_bits=sum(layer.detectors for layer in layers_position) * 1,
        layers=layers_position,
        bits_per_detector=1,
        seed=1,
        bit_assignment="detector_index",
        coordinate_mode="linear",
        value_range=(0.0, 100.0),
    )

    positions = [0.0, 100.0]
    total_bits = angle_encoder.code_bits + position_encoder.code_bits
    codes: list[tuple[BitArray, float, float]] = []
    for position_idx, position in enumerate(positions):
        for angle in range(180):
            angle_bits = angle_encoder.encode_sparse(angle)
            position_bits = position_encoder.encode_sparse(position)
            code = BitArray(total_bits)
            for idx in angle_bits:
                code.set(idx)
            offset = angle_encoder.code_bits
            for idx in position_bits:
                code.set(offset + idx)
            angle_label = angle + position_idx * 180
            codes.append((code, float(angle_label), float(angle)))

    reporter = EncoderReporter(angle_encoder)
    reporter.print_codes(codes)
    #reporter.visualize_detectors()

    angle0_bits = angle_encoder.encode_sparse(0.0)
    angle1_bits = angle_encoder.encode_sparse(1.0)
    angle0_code = BitArray(angle_encoder.code_bits)
    angle1_code = BitArray(angle_encoder.code_bits)
    for idx in angle0_bits:
        angle0_code.set(idx)
    for idx in angle1_bits:
        angle1_code.set(idx)
    print(f"angle bits angle 0 pos 0: {angle0_code.to01()}")
    print(f"angle bits angle 1 pos 0: {angle1_code.to01()}")
    common_angle = angle0_code.common(angle1_code)
    union_angle = angle0_code.count() + angle1_code.count() - common_angle
    jaccard_angle = 0.0 if union_angle == 0 else common_angle / union_angle
    print(f"angle bits Jaccard angle 0 vs angle 1: {jaccard_angle:.6f}")
    pos0_bits = position_encoder.encode_sparse(positions[0])
    pos1_bits = position_encoder.encode_sparse(positions[1])
    pos0_code = BitArray(position_encoder.code_bits)
    pos1_code = BitArray(position_encoder.code_bits)
    for idx in pos0_bits:
        pos0_code.set(idx)
    for idx in pos1_bits:
        pos1_code.set(idx)
    print(f"position bits angle 0 pos 0: {pos0_code.to01()}")
    print(f"position bits angle 0 pos 1: {pos1_code.to01()}")
    common = pos0_code.common(pos1_code)
    union = pos0_code.count() + pos1_code.count() - common
    jaccard = 0.0 if union == 0 else common / union
    print(f"position bits Jaccard angle 0 pos 0 vs pos 1: {jaccard:.6f}")

    layout = Layout(
        codes,
        similarity="cosine",
        lambda_threshold=0.65,
        eta=None,
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
    layout.set_similarity_params(lambda_threshold=0.8, eta=None)
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
