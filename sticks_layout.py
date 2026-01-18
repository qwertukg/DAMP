from __future__ import annotations

from collections import defaultdict
from io import BytesIO
from pathlib import Path
import re
import zipfile
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

import main as damp_main
from damp.MnistSobelAngleMap import MnistSobelAngleMap
from damp.article_refs import ENCODING_SYSTEM, LAYOUT_PARAMETERS, LAYOUT_SYSTEM, SLIDING_WINDOW
from damp.layout.damp_layout import Layout
from damp.logging import LOGGER
from main import _build_encoder, _run_layout, configure_logging


class SticksLayoutRunner:
    def __init__(
        self,
        zip_path: Path,
        *,
        output_prefix: Path | None = None,
    ) -> None:
        configure_logging()
        self.zip_path = Path(zip_path)
        self.output_prefix = (
            Path(output_prefix) if output_prefix is not None else Path("data/sticks_layout")
        )
        self._tune_for_dataset()
        self.encoder = _build_encoder()
        self.extractor = MnistSobelAngleMap(angle_in_degrees=True, grad_threshold=0.05)
        LOGGER.event(
            "sticks.init",
            section=ENCODING_SYSTEM,
            data={
                "zip_path": str(self.zip_path),
                "output_prefix": str(self.output_prefix),
            },
        )

    def _tune_for_dataset(self) -> None:
        damp_main.LAYOUT_USE_GPU = True
        damp_main.LAYOUT_LONG_PAIRS_PER_STEP = 6000
        damp_main.LAYOUT_LONG_STEPS = 300
        damp_main.LAYOUT_LONG_SUBSET_SIZE = 512
        damp_main.LAYOUT_LONG_SUBSET_REFRESH = 20
        damp_main.LAYOUT_SHORT_PAIRS_PER_STEP = 3000
        damp_main.LAYOUT_SHORT_STEPS = 120
        damp_main.LAYOUT_SHORT_LOCAL_RADIUS = 4
        damp_main.LAYOUT_SHORT_ENERGY_RADIUS = 4
        damp_main.LAYOUT_SHORT_ENERGY_MAX_POINTS = 256
        damp_main.LAYOUT_SHORT_PARTITIONS = 4
        damp_main.LAYOUT_SHORT_SIMILARITY_CUTOFF = 0.03
        damp_main.LAYOUT_SHORT_WEIGHTED_FIRST = True
        damp_main.LAYOUT_MIN_SWAP_RATIO = 0.003
        damp_main.LAYOUT_MIN_SWAP_WINDOW = 50
        damp_main.LAYOUT_ADAPTIVE_RADIUS_START_FACTOR = 0.4
        damp_main.LAYOUT_LOG_VISUALS = True
        damp_main.LAYOUT_LOG_EVERY_LONG = 20
        damp_main.LAYOUT_LOG_EVERY_SHORT = 20
        LOGGER.event(
            "sticks.layout.tune",
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

    def _load_images(self) -> Sequence[Tuple[int, np.ndarray]]:
        if not self.zip_path.exists():
            raise FileNotFoundError(f"ZIP не найден: {self.zip_path}")
        images: list[Tuple[int, np.ndarray]] = []
        with zipfile.ZipFile(self.zip_path) as zf:
            names = sorted(n for n in zf.namelist() if n.lower().endswith(".png"))
            LOGGER.event(
                "sticks.load.start",
                section=SLIDING_WINDOW,
                data={"files": len(names), "zip_path": str(self.zip_path)},
            )
            for name in names:
                with zf.open(name) as fp:
                    img = Image.open(BytesIO(fp.read())).convert("L")
                width, height = img.size
                array = np.asarray(img, dtype=np.float32)
                if array.shape != MnistSobelAngleMap.EXPECTED_SHAPE:
                    raise ValueError(
                        f"Ожидаю {MnistSobelAngleMap.EXPECTED_SHAPE}, получил {array.shape} из {name}"
                    )
                label = self._label_from_name(name)
                images.append((label, array))
                LOGGER.event(
                    "sticks.image.load",
                    section=SLIDING_WINDOW,
                    data={
                        "name": name,
                        "label": label,
                        "size": (width, height),
                    },
                )
        LOGGER.event(
            "sticks.load.done",
            section=SLIDING_WINDOW,
            data={"images": len(images)},
        )
        return images

    @staticmethod
    def _label_from_name(name: str) -> int:
        match = re.search(r"(\d+)", Path(name).stem)
        if not match:
            return 0
        return int(match.group(1))

    def _encode_images(self, images: Sequence[Tuple[int, np.ndarray]]) -> Dict[float, List]:
        codes: dict[float, list] = defaultdict(list)
        total_codes = 0
        for label, image in images:
            measurements_map = self.extractor.extract(image, label)
            measurements = measurements_map[label]
            for idx, (angle, x, y) in enumerate(measurements):
                log_image = image if idx == 0 else None
                log_measurements = measurements if idx == 0 else None
                _, code = self.encoder.encode(
                    float(angle),
                    float(x),
                    float(y),
                    log_image=log_image,
                    log_label=label,
                    log_measurements=log_measurements,
                )
                codes[angle].append(code)
                total_codes += 1
            LOGGER.event(
                "sticks.image.codes",
                section=ENCODING_SYSTEM,
                data={
                    "label": label,
                    "measurements": len(measurements),
                    "codes_total": total_codes,
                },
            )
        LOGGER.event(
            "sticks.codes",
            section=ENCODING_SYSTEM,
            data={
                "angles": len(codes),
                "codes_total": total_codes,
            },
        )
        return codes

    def _layout_codes(self, codes: Dict[float, List]) -> Layout:
        LOGGER.event(
            "sticks.layout.start",
            section=LAYOUT_SYSTEM,
            data={
                "codes": sum(len(bucket) for bucket in codes.values()),
                "angles": len(codes),
            },
        )
        layout = _run_layout(codes)
        LOGGER.event(
            "sticks.layout.done",
            section=LAYOUT_SYSTEM,
            data={
                "points": len(layout.positions()),
                "width": layout.width,
                "height": layout.height,
            },
        )
        return layout

    def _export_layout(self, layout: Layout) -> None:
        image_path = self.output_prefix.with_suffix(".png")
        json_path = self.output_prefix.with_suffix(".json")
        rendered = layout.render_image()
        Image.fromarray(rendered).save(image_path)
        layout.save_json(str(json_path))
        LOGGER.event(
            "sticks.export",
            section=LAYOUT_SYSTEM,
            data={
                "image_path": str(image_path),
                "json_path": str(json_path),
                "points": len(layout.positions()),
            },
        )

    def run(self) -> None:
        images = self._load_images()
        codes = self._encode_images(images)
        layout = self._layout_codes(codes)
        self._export_layout(layout)


def main() -> None:
    runner = SticksLayoutRunner(
        Path("sticks_28x28_360.zip"),
        output_prefix=Path("data/sticks_layout"),
    )
    runner.run()


if __name__ == "__main__":
    main()
