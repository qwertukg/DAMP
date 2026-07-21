from __future__ import annotations

import sys
import tempfile
import unittest
from subprocess import CompletedProcess
from unittest.mock import patch
from pathlib import Path

from PIL import Image, ImageDraw

from test.digit_ui import DigitImageNormalizer, MainPredictCommand


class DigitUiTests(unittest.TestCase):
    def test_normalizer_produces_mnist_shape_and_centered_foreground(self) -> None:
        source = Image.new("L", (280, 280), 0)
        draw = ImageDraw.Draw(source)
        draw.line((40, 30, 80, 220), fill=255, width=22)
        normalized = DigitImageNormalizer(
            output_size=28,
            digit_box_size=20,
        ).normalize(source)
        self.assertEqual(normalized.size, (28, 28))
        bounds = normalized.getbbox()
        self.assertIsNotNone(bounds)
        self.assertLessEqual((bounds or (0, 0, 0, 0))[2], 24)
        self.assertLessEqual((bounds or (0, 0, 0, 0))[3], 24)

    def test_predict_command_uses_existing_main_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = MainPredictCommand(
                project_root=root,
                python_executable=sys.executable,
            )
            image_path = root / "digit.png"
            self.assertEqual(
                command.arguments(count=60000, image_path=image_path),
                [
                    sys.executable,
                    str(root / "main.py"),
                    "predict",
                    "--count",
                    "60000",
                    "--image",
                    str(image_path),
                ],
            )

    def test_predict_command_parses_main_result(self) -> None:
        command = MainPredictCommand(
            project_root=Path("D:/project"),
            python_executable=sys.executable,
        )
        completed = CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "decoder log\n"
                "image=digit.png pred=7 conf=0.8750 "
                "probs=[0.01, 0.02, 0.03, 0.01, 0.01, 0.01, 0.02, 0.875, 0.005, 0.005]\n"
            ),
            stderr="",
        )
        with patch("test.digit_ui.subprocess.run", return_value=completed) as run:
            result = command.run(count=60000, image_path=Path("digit.png"))
        self.assertEqual(result.predicted_class, 7)
        self.assertEqual(result.confidence, 0.875)
        self.assertEqual(len(result.probabilities), 10)
        self.assertEqual(run.call_args.kwargs["env"]["RERUN"], "off")


if __name__ == "__main__":
    unittest.main()
