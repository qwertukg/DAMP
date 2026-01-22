from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image

import main as damp_main
from damp.MnistSobelAngleMap import MnistSobelAngleMap
from damp.article_refs import ENCODING_SYSTEM, LAYOUT_PARAMETERS, LAYOUT_SYSTEM, SLIDING_WINDOW
from damp.layout.damp_layout import Layout
from damp.layout.payload import LayoutPayloadBuilder
from damp.logging import LOGGER
from main import _build_encoder, _run_layout, configure_logging


class SingleImageLayoutRunner:
    def __init__(
        self,
        image_path: Path,
        *,
        label: int = 0,
        output_prefix: Path | None = None,
    ) -> None:
        configure_logging()
        self.image_path = Path(image_path)
        self.label = int(label)
        self.output_prefix = (
            Path(output_prefix) if output_prefix is not None else Path("data/white_circle_layout")
        )
        self._tune_for_small_space()
        self.encoder = _build_encoder()
        self.extractor = MnistSobelAngleMap(angle_in_degrees=True, grad_threshold=0.05)
        LOGGER.event(
            "single_image.init",
            section=ENCODING_SYSTEM,
            data={
                "image_path": str(self.image_path),
                "label": self.label,
                "output_prefix": str(self.output_prefix),
            },
        )

    def _tune_for_small_space(self) -> None:
        damp_main.LAYOUT_USE_GPU = False
        damp_main.LAYOUT_LONG_PAIRS_PER_STEP = 100
        damp_main.LAYOUT_LONG_STEPS = 60
        damp_main.LAYOUT_LONG_SUBSET_SIZE = 36
        damp_main.LAYOUT_LONG_SUBSET_REFRESH = 5
        damp_main.LAYOUT_SHORT_PAIRS_PER_STEP = 40
        damp_main.LAYOUT_SHORT_STEPS = 30
        damp_main.LAYOUT_SHORT_LOCAL_RADIUS = 1
        damp_main.LAYOUT_SHORT_ENERGY_RADIUS = 2
        damp_main.LAYOUT_SHORT_ENERGY_MAX_POINTS = 24
        damp_main.LAYOUT_SHORT_PARTITIONS = 1
        damp_main.LAYOUT_SHORT_SIMILARITY_CUTOFF = None
        damp_main.LAYOUT_SHORT_WEIGHTED_FIRST = False
        damp_main.LAYOUT_MIN_SWAP_RATIO = 0.01
        damp_main.LAYOUT_MIN_SWAP_WINDOW = 5
        damp_main.LAYOUT_ADAPTIVE_RADIUS_START_FACTOR = 0.25
        damp_main.LAYOUT_LOG_VISUALS = False
        damp_main.LAYOUT_LOG_EVERY_LONG = 10
        damp_main.LAYOUT_LOG_EVERY_SHORT = 10
        LOGGER.event(
            "single_image.layout.tune",
            section=LAYOUT_PARAMETERS,
            data={
                "gpu": damp_main.LAYOUT_USE_GPU,
                "long_steps": damp_main.LAYOUT_LONG_STEPS,
                "long_pairs": damp_main.LAYOUT_LONG_PAIRS_PER_STEP,
                "long_subset_size": damp_main.LAYOUT_LONG_SUBSET_SIZE,
                "long_subset_refresh": damp_main.LAYOUT_LONG_SUBSET_REFRESH,
                "short_steps": damp_main.LAYOUT_SHORT_STEPS,
                "short_pairs": damp_main.LAYOUT_SHORT_PAIRS_PER_STEP,
                "short_local_radius": damp_main.LAYOUT_SHORT_LOCAL_RADIUS,
                "short_energy_radius": damp_main.LAYOUT_SHORT_ENERGY_RADIUS,
                "short_energy_max_points": damp_main.LAYOUT_SHORT_ENERGY_MAX_POINTS,
                "short_partitions": damp_main.LAYOUT_SHORT_PARTITIONS,
                "short_similarity_cutoff": damp_main.LAYOUT_SHORT_SIMILARITY_CUTOFF,
                "short_weighted_first": damp_main.LAYOUT_SHORT_WEIGHTED_FIRST,
                "min_swap_ratio": damp_main.LAYOUT_MIN_SWAP_RATIO,
                "min_swap_window": damp_main.LAYOUT_MIN_SWAP_WINDOW,
                "adaptive_radius_factor": damp_main.LAYOUT_ADAPTIVE_RADIUS_START_FACTOR,
                "log_visuals": damp_main.LAYOUT_LOG_VISUALS,
                "log_every_long": damp_main.LAYOUT_LOG_EVERY_LONG,
                "log_every_short": damp_main.LAYOUT_LOG_EVERY_SHORT,
            },
        )

    def _load_image(self) -> np.ndarray:
        if not self.image_path.exists():
            raise FileNotFoundError(f"PNG не найден: {self.image_path}")
        image = Image.open(self.image_path).convert("L")
        width, height = image.size
        array = np.asarray(image, dtype=np.float32)
        LOGGER.event(
            "single_image.load",
            section=SLIDING_WINDOW,
            data={
                "path": str(self.image_path),
                "size": (width, height),
                "expected_shape": MnistSobelAngleMap.EXPECTED_SHAPE,
                "mode": image.mode,
            },
        )
        return array

    def _encode_image(self, image: np.ndarray) -> LayoutPayloadBuilder:
        measurements_map = self.extractor.extract(image, self.label)
        measurements = measurements_map[self.label]
        payload = LayoutPayloadBuilder(
            similarity=None,
            lambda_threshold=None,
            eta=None,
        )
        total_codes = 0
        for idx, (angle, x, y) in enumerate(measurements):
            log_image = image if idx == 0 else None
            log_measurements = measurements if idx == 0 else None
            _, code = self.encoder.encode(
                float(angle),
                float(x),
                float(y),
                log_image=log_image,
                log_label=self.label,
                log_measurements=log_measurements,
            )
            payload.add_code(label=self.label, hue=float(angle), code=code)
            total_codes += 1
        LOGGER.event(
            "single_image.codes",
            section=ENCODING_SYSTEM,
            data={
                "label": self.label,
                "angles": len(payload.codes_by_hue()),
                "codes_total": total_codes,
            },
        )
        return payload

    def _layout_codes(self, payload: LayoutPayloadBuilder) -> Layout:
        codes = payload.codes_by_hue()
        total_codes = len(payload.records)
        LOGGER.event(
            "single_image.layout.start",
            section=LAYOUT_SYSTEM,
            data={
                "codes": total_codes,
                "angles": len(codes),
            },
        )
        layout = _run_layout(
            codes,
            None,
            count=total_codes,
            total_codes=total_codes,
        )
        LOGGER.event(
            "single_image.layout.done",
            section=LAYOUT_SYSTEM,
            data={
                "points": len(layout.positions()),
                "width": layout.width,
                "height": layout.height,
            },
        )
        return layout

    def _export_layout(self, layout: Layout, json_path: Path) -> None:
        image_path = self.output_prefix.with_suffix(".png")
        rendered = layout.render_image()
        Image.fromarray(rendered).save(image_path)
        layout.save_json(str(json_path))
        LOGGER.event(
            "single_image.export",
            section=LAYOUT_SYSTEM,
            data={
                "image_path": str(image_path),
                "json_path": str(json_path),
                "points": len(layout.positions()),
            },
        )

    def run(self) -> None:
        image = self._load_image()
        payload = self._encode_image(image)
        json_path = self.output_prefix.with_suffix(".json")
        payload.save_base(str(json_path))
        layout = self._layout_codes(payload)
        self._export_layout(layout, json_path)


def main() -> None:
    runner = SingleImageLayoutRunner(
        Path("white_circle_28x28_grayscale.png"),
        label=0,
        output_prefix=Path("data/white_circle_layout"),
    )
    runner.run()


if __name__ == "__main__":
    main()
