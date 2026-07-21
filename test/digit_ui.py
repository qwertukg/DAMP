from __future__ import annotations

import argparse
import ast
import os
import queue
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ["RERUN"] = "off"

from main import (
    TEST_UI_BRUSH_WIDTH,
    TEST_UI_CANVAS_SIZE,
    TEST_UI_DEFAULT_COUNT,
    TEST_UI_DIGIT_BOX_SIZE,
    TEST_UI_OUTPUT_SIZE,
    TEST_UI_WINDOW_HEIGHT,
    TEST_UI_WINDOW_WIDTH,
)


@dataclass(frozen=True)
class PredictionViewModel:
    predicted_class: int
    confidence: float
    probabilities: tuple[float, ...]
    output: str


class DigitImageNormalizer:
    """Переводит рисунок мышью в белую MNIST-цифру на чёрном фоне."""

    def __init__(self, *, output_size: int, digit_box_size: int) -> None:
        self._output_size = int(output_size)
        self._digit_box_size = int(digit_box_size)

    def normalize(self, source: Image.Image) -> Image.Image:
        grayscale = source.convert("L")
        bounds = grayscale.getbbox()
        if bounds is None:
            raise ValueError("сначала нарисуйте цифру")
        digit = grayscale.crop(bounds)
        width, height = digit.size
        scale = min(self._digit_box_size / width, self._digit_box_size / height)
        target_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        digit = digit.resize(target_size, Image.Resampling.LANCZOS)
        result = Image.new("L", (self._output_size, self._output_size), 0)
        offset = (
            (self._output_size - target_size[0]) // 2,
            (self._output_size - target_size[1]) // 2,
        )
        result.paste(digit, offset)
        return result


class MainPredictCommand:
    RESULT_PATTERN = re.compile(
        r"pred=(?P<prediction>\d+)\s+conf=(?P<confidence>[0-9.eE+-]+)\s+"
        r"probs=(?P<probabilities>\[[^\]]*\])"
    )

    def __init__(self, *, project_root: Path, python_executable: str) -> None:
        self._project_root = project_root
        self._python_executable = python_executable

    def arguments(self, *, count: int, image_path: Path) -> list[str]:
        return [
            self._python_executable,
            str(self._project_root / "main.py"),
            "predict",
            "--count",
            str(int(count)),
            "--image",
            str(image_path),
        ]

    def run(self, *, count: int, image_path: Path) -> PredictionViewModel:
        completed = subprocess.run(
            self.arguments(count=count, image_path=image_path),
            cwd=self._project_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            env={**os.environ, "RERUN": "off"},
        )
        output = "\n".join(
            part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
        )
        if completed.returncode != 0:
            raise RuntimeError(output or f"predict завершился с кодом {completed.returncode}")
        matches = list(self.RESULT_PATTERN.finditer(output))
        if not matches:
            raise RuntimeError(f"не удалось прочитать результат predict:\n{output}")
        match = matches[-1]
        probabilities = tuple(
            float(value)
            for value in ast.literal_eval(match.group("probabilities"))
        )
        return PredictionViewModel(
            predicted_class=int(match.group("prediction")),
            confidence=float(match.group("confidence")),
            probabilities=probabilities,
            output=output,
        )


class DigitDrawingApplication:
    def __init__(self, *, count: int) -> None:
        import tkinter as tk
        from tkinter import ttk

        self._tk = tk
        self._ttk = ttk
        self._project_root = PROJECT_ROOT
        self._image_path = Path(__file__).resolve().parent / "digit.png"
        self._normalizer = DigitImageNormalizer(
            output_size=TEST_UI_OUTPUT_SIZE,
            digit_box_size=TEST_UI_DIGIT_BOX_SIZE,
        )
        self._command = MainPredictCommand(
            project_root=self._project_root,
            python_executable=sys.executable,
        )
        self._messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self._drawing = Image.new(
            "L", (TEST_UI_CANVAS_SIZE, TEST_UI_CANVAS_SIZE), 0
        )
        self._draw = ImageDraw.Draw(self._drawing)
        self._last_point: tuple[int, int] | None = None

        self.root = tk.Tk()
        self.root.title("DAMP — рисование MNIST")
        self.root.geometry(f"{TEST_UI_WINDOW_WIDTH}x{TEST_UI_WINDOW_HEIGHT}")
        self.root.resizable(False, False)
        self._count = tk.StringVar(value=str(int(count)))
        self._status = tk.StringVar(value="Нарисуйте цифру и нажмите «Распознать»")
        self._result = tk.StringVar(value="—")
        self._probabilities = [tk.DoubleVar(value=0.0) for _ in range(10)]
        self._build_widgets()
        self.root.after(100, self._poll_messages)

    def _build_widgets(self) -> None:
        tk = self._tk
        ttk = self._ttk
        root_frame = ttk.Frame(self.root, padding=16)
        root_frame.pack(fill="both", expand=True)

        left = ttk.Frame(root_frame)
        left.grid(row=0, column=0, sticky="n")
        self._canvas = tk.Canvas(
            left,
            width=TEST_UI_CANVAS_SIZE,
            height=TEST_UI_CANVAS_SIZE,
            background="black",
            highlightthickness=1,
            highlightbackground="#666666",
            cursor="crosshair",
        )
        self._canvas.pack()
        self._canvas.bind("<Button-1>", self._start_stroke)
        self._canvas.bind("<B1-Motion>", self._draw_stroke)
        self._canvas.bind("<ButtonRelease-1>", self._end_stroke)

        controls = ttk.Frame(left, padding=(0, 12, 0, 0))
        controls.pack(fill="x")
        ttk.Button(controls, text="Очистить", command=self.clear).pack(side="left")
        self._predict_button = ttk.Button(
            controls, text="Распознать", command=self.predict
        )
        self._predict_button.pack(side="right")

        right = ttk.Frame(root_frame, padding=(24, 0, 0, 0))
        right.grid(row=0, column=1, sticky="nsew")
        ttk.Label(right, text="Количество обучающих примеров:").grid(
            row=0, column=0, columnspan=3, sticky="w"
        )
        ttk.Entry(right, textvariable=self._count, width=12).grid(
            row=1, column=0, sticky="w", pady=(4, 14)
        )
        ttk.Label(right, text="Результат:").grid(row=2, column=0, sticky="w")
        ttk.Label(right, textvariable=self._result, font=("Segoe UI", 36, "bold")).grid(
            row=3, column=0, sticky="w", pady=(0, 12)
        )
        for digit, probability in enumerate(self._probabilities):
            ttk.Label(right, text=str(digit), width=2).grid(row=4 + digit, column=0)
            ttk.Progressbar(
                right,
                variable=probability,
                maximum=1.0,
                length=250,
            ).grid(row=4 + digit, column=1, padx=6, pady=2)
            ttk.Label(right, textvariable=probability, width=8).grid(
                row=4 + digit, column=2
            )
        ttk.Label(
            root_frame,
            textvariable=self._status,
            wraplength=TEST_UI_WINDOW_WIDTH - 40,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(16, 0))

    def _start_stroke(self, event: object) -> None:
        self._last_point = (int(getattr(event, "x")), int(getattr(event, "y")))
        self._paint_segment(self._last_point, self._last_point)

    def _draw_stroke(self, event: object) -> None:
        point = (int(getattr(event, "x")), int(getattr(event, "y")))
        if self._last_point is not None:
            self._paint_segment(self._last_point, point)
        self._last_point = point

    def _end_stroke(self, _event: object) -> None:
        self._last_point = None

    def _paint_segment(
        self, start: tuple[int, int], end: tuple[int, int]
    ) -> None:
        radius = TEST_UI_BRUSH_WIDTH // 2
        self._canvas.create_line(
            *start,
            *end,
            fill="white",
            width=TEST_UI_BRUSH_WIDTH,
            capstyle=self._tk.ROUND,
            smooth=True,
        )
        self._draw.line(
            (*start, *end),
            fill=255,
            width=TEST_UI_BRUSH_WIDTH,
        )
        for point in (start, end):
            self._draw.ellipse(
                (
                    point[0] - radius,
                    point[1] - radius,
                    point[0] + radius,
                    point[1] + radius,
                ),
                fill=255,
            )

    def clear(self) -> None:
        self._canvas.delete("all")
        self._drawing = Image.new(
            "L", (TEST_UI_CANVAS_SIZE, TEST_UI_CANVAS_SIZE), 0
        )
        self._draw = ImageDraw.Draw(self._drawing)
        self._result.set("—")
        self._status.set("Холст очищен")
        for probability in self._probabilities:
            probability.set(0.0)

    def predict(self) -> None:
        try:
            count = int(self._count.get())
            if count <= 0:
                raise ValueError
            normalized = self._normalizer.normalize(self._drawing)
        except ValueError:
            self._status.set("Нарисуйте цифру и укажите положительный count")
            return
        normalized.save(self._image_path)
        self._predict_button.state(["disabled"])
        self._status.set(
            f"Запуск: python main.py predict --count {count} "
            f"--image {self._image_path}"
        )
        threading.Thread(
            target=self._run_prediction,
            args=(count,),
            daemon=True,
        ).start()

    def _run_prediction(self, count: int) -> None:
        try:
            result = self._command.run(count=count, image_path=self._image_path)
            self._messages.put(("result", result))
        except Exception as error:
            self._messages.put(("error", str(error)))

    def _poll_messages(self) -> None:
        try:
            kind, payload = self._messages.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_messages)
            return
        self._predict_button.state(["!disabled"])
        if kind == "result" and isinstance(payload, PredictionViewModel):
            self._result.set(
                f"{payload.predicted_class}  ({payload.confidence:.1%})"
            )
            for index, variable in enumerate(self._probabilities):
                variable.set(
                    payload.probabilities[index]
                    if index < len(payload.probabilities)
                    else 0.0
                )
            self._status.set(f"Готово. PNG: {self._image_path}")
        else:
            self._status.set(f"Ошибка: {payload}")
        self.root.after(100, self._poll_messages)

    def run(self) -> None:
        self.root.mainloop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UI для проверки DAMP на нарисованной цифре")
    parser.add_argument("--count", type=int, default=TEST_UI_DEFAULT_COUNT)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    DigitDrawingApplication(count=arguments.count).run()


if __name__ == "__main__":
    main()
