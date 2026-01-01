from __future__ import annotations

import rerun as rr

from encoder import Encoder, LayerConfig
from layout import Layout


def main() -> None:
    layers = [
        LayerConfig(detectors=360, window=1),
        LayerConfig(detectors=180, window=1),
        LayerConfig(detectors=90, window=1),
        LayerConfig(detectors=45, window=1),
        LayerConfig(detectors=22, window=1),
    ]
    encoder = Encoder(
        code_bits=128,
        layers=layers,
        bits_per_detector=2,
        seed=0,
    )
    start = 0.0
    end = 360.0
    step = 1.0
    codes = encoder.encode(start=start, end=end, step=step)
    encoder.print_codes(codes)
    layout = Layout(
        codes,
        grid_size=32,
        similarity="cosine",
        angle_weight=0.7,
        lambda_threshold=0.2,
        eta=8.0,
        seed=0,
    )
    rr.init("damp-layout")
    rr.spawn()
    layout.log_rerun(step=0)
    step_offset = 1
    layout.run(
        steps=10000,
        pairs_per_step=300,
        pair_radius=layout.width // 2,
        mode="long",
        min_swap_ratio=0.0,
        log_every=1,
        step_offset=step_offset,
        energy_radius=5,
        energy_check_every=5,
        energy_delta=1e-3,
        energy_patience=3,
    )
    step_offset += layout.last_steps
    layout.set_similarity_params(lambda_threshold=0.35, eta=8.0, angle_weight=0.5)
    layout.run(
        steps=150,
        pairs_per_step=300,
        pair_radius=5,
        mode="short",
        local_radius=5,
        min_swap_ratio=0.01,
        log_every=1,
        step_offset=step_offset,
    )


if __name__ == "__main__":
    main()
