from __future__ import annotations

import rerun as rr

from Encoder import Encoder
from EncoderReporter import EncoderReporter
from LayerConfig import LayerConfig
from Layout import Layout


def main() -> None:
    layers = [
        LayerConfig(detectors=4,    overlap=0.7),
        LayerConfig(detectors=8,    overlap=0.7),
        LayerConfig(detectors=16,   overlap=0.7),
        LayerConfig(detectors=32,   overlap=0.7),
        LayerConfig(detectors=64,   overlap=0.7),
        LayerConfig(detectors=128,  overlap=0.7),
        LayerConfig(detectors=256,  overlap=0.7),
    ]
    encoder = Encoder(
        code_bits=sum(layer.detectors for layer in layers),
        layers=layers,
        bits_per_detector=1,
        seed=0,
        bit_assignment="detector_index",
    )
    start = 0.0
    end = 359.0
    step = 1.0
    codes = encoder.encode(start=start, end=end, step=step)

    reporter = EncoderReporter(encoder)
    reporter.print_codes(codes)
    reporter.visualize_detectors()

    layout = Layout(
        codes,
        grid_size=64,
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
