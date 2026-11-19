"""Все рисунки README одним запуском: прогон цикла с наблюдателем, тест пинвила, картинки детекторов."""

from __future__ import annotations
import os
import time
from pathlib import Path

from PIL import Image

from damp import settings
from damp.article_refs import OUTSIDE_ARTICLE
from damp.cycle.ladder import MnistLadderRunner
from damp.encoding.detector_view import DetectorLayerView
from damp.encoding.mnist_codes import _build_encoder
from damp.figures.charts import AccuracyChart
from damp.figures.cycle import CycleFigures
from damp.figures.overview import PipelineOverview
from damp.figures.recorder import LadderRecorder
from damp.figures.style import Sheet, crop_cells
from damp.figures.style import figure_style
from damp.layout.pinwheel import PinwheelExperiment
from damp.logging import LOGGER


class ReadmeFigures:
    """Рисунки этапов пайплайна для README."""

    def __init__(self) -> None:
        self.style = figure_style()
        self.docs = settings.README_FIGURES_DIR
        self.work = settings.README_FIGURES_WORK_DIR
        os.makedirs(self.docs, exist_ok=True)
        os.makedirs(self.work, exist_ok=True)
        self.prefix = os.path.join(self.work, "ladder").replace("\\", "/")
        upscale = settings.MNIST_SOBEL_UPSCALE if settings.MNIST_SOBEL_UPSCALE_ENABLED else 1
        self.side_cells = 28.0 * upscale / settings.MNIST_PATCH_SIZE

    def doc(self, name: str) -> str:
        return os.path.join(self.docs, name).replace("\\", "/")

    def views(self, encoder) -> DetectorLayerView:  # noqa: ANN001
        """Картинка детекторов кодера в стиле README: код по 32 клетки в строке."""
        return DetectorLayerView(
            encoder,
            style=self.style,
            min_hue=settings.LAYOUT_PSEUDO_COLOR_MIN_HUE,
            max_hue=settings.LAYOUT_PSEUDO_COLOR_MAX_HUE,
            inactive_alpha=settings.DETECTORS_VIEW_INACTIVE_ALPHA,
            panel=settings.README_FIGURES_PANEL,
            code_columns=32,
            code_cell=12,
        )

    def structure(self) -> str:
        """Рисунок устройства проекта по настоящим папкам (корень проекта — над system/)."""
        from damp.figures.structure import StructureFigure

        return StructureFigure(self.style, str(Path(__file__).resolve().parents[3])).save(self.doc("structure.png"))

    def run(self) -> None:
        self.structure()
        recorder = LadderRecorder()
        runner = MnistLadderRunner(
            count=settings.README_FIGURES_TRAIN,
            evaluate_limit=settings.README_FIGURES_TEST,
            map_max_points=settings.README_FIGURES_MAP_POINTS,
            shifts=(),
            file_prefix=self.prefix,
            observer=recorder,
        )
        started = time.time()
        runner.run()
        LOGGER.event(
            "readme.figures.ladder",
            section=OUTSIDE_ARTICLE,
            data={"seconds": time.time() - started, "levels": len(runner.reports), "top_vote": runner.top_accuracy},
        )
        count = settings.README_FIGURES_TRAIN
        glyphs = {level.number: f"{self.prefix}-level{level.number}-keyholes-{count}.glyphs.png" for level in recorder.levels}
        figures = CycleFigures(
            recorder,
            self.style,
            side_cells=self.side_cells,
            panel=settings.README_FIGURES_PANEL,
            strongest=settings.README_FIGURES_STRONGEST,
            digit=settings.LADDER_FIGURE_DIGIT,
            test_digits=settings.README_FIGURES_TEST_DIGITS,
        )
        figures.stimuli(self.doc("stage1-sticks.png"), self.doc("stage1-digits.png"))
        figures.primary_coding(self.doc("stage2-detectors.png"), self.views)
        if 0 in glyphs:
            # Уровень 0 (LADDER_LEVEL0): первичная раскладка штрихов — место штриха от центра цифры.
            figures.layout_crop(glyphs[0], self.doc("stage3-layout-level0.png"), columns=20, rows=8, number=0)
        figures.layout_crop(glyphs[1], self.doc("stage3-layout.png"), columns=26, rows=12)
        self.pinwheel(self.doc("stage3-pinwheel.png"))
        figures.hierarchy(self.doc("stage4-detectors.png"))
        figures.activation(self.doc("stage4-activation.png"))
        figures.embedding(self.doc("stage5-embedding.png"))
        figures.keyholes_growth(f"{self.prefix}-fields-{count}.png", self.doc("stage6-keyholes.png"))
        for level in recorder.levels:
            figures.step(level.number, self.doc(f"stage6-step-level{level.number}.png"))
        figures.level_maps(glyphs, self.doc("stage6-level-maps.png"), columns=28, rows=11)
        figures.class_clusters(self.doc("stage7-class-clusters.png"))
        figures.recognition(self.doc("stage7-fives.png"))
        figures.top_map(f"{self.prefix}-classes-{count}.glyphs.png", self.doc("stage7-top-map.png"), tile=settings.README_FIGURES_TILE, columns=28, rows=12)
        AccuracyChart(self.style).save(
            [
                {
                    "number": report.number,
                    "seen": report.picture_seen,
                    "alone": report.alone[min(report.alone)],
                    "peak": report.peak[min(report.peak)],
                    "glued": report.glued[min(report.glued)],
                }
                for report in runner.reports
            ],
            runner.top_accuracy[min(runner.top_accuracy)],
            self.doc("stage7-accuracy.png"),
            title=f"Узнавание цифр по уровням цикла: {settings.README_FIGURES_TRAIN} обучающих, {settings.README_FIGURES_TEST} проверочных",
        )
        self.layout_command_detectors(self.doc("layout-command-detectors.png"))
        self.overview(figures, glyphs, self.doc("pipeline-overview.png"))

    def pinwheel(self, path: str) -> None:
        """Тест пинвила текущим кодером: три раскраски одной раскладки (§5.14, рис. 16, рис. 18)."""
        from damp.layout.pinwheel import PinwheelExperiment

        names = ["pinwheel.png", "pinwheel.angle.png", "pinwheel.energy.png"]
        files = [os.path.join(self.work, name) for name in names]
        _, quality = PinwheelExperiment(
            output_path=files[0],
            component_output_path=files[1],
            energy_output_path=files[2],
            layout_path=os.path.join(self.work, "pinwheel.json"),
        ).run()
        panels = [Image.open(name).convert("RGB") for name in files]
        side = panels[0].width
        panels = [panel.resize((panel.width * 8, panel.height * 8), Image.NEAREST) for panel in panels]
        sheet = Sheet(self.style, 3 * panels[0].width + 4 * 20, "Тест пинвила: 4 позиции × 360 углов в одной раскладке")
        bottom = sheet.grid(
            list(
                zip(
                    ["Псевдоцвета битов кода (§5.14)", "Угол стимула", "Яркость — энергия точки (рис. 18)"],
                    [
                        f"сетка {side}×{side}, пиксель — ячейка",
                        f"средняя разница углов соседей {quality.mean_neighbour_delta:.1f}°, сингулярностей {quality.singularities}",
                        f"резких переходов {quality.sharp_transition_ratio:.0%}",
                    ],
                    panels,
                )
            ),
            columns=3,
        )
        sheet.save(path, bottom, note="тест пинвила", data=dict(quality.as_mapping()))

    def layout_command_detectors(self, path: str) -> None:
        """Детекторы кодера команды layout: угол, X и Y картинки, общий код (как detectors_visual.py)."""
        DetectorLayerView(
            _build_encoder(),
            style=self.style,
            min_hue=settings.LAYOUT_PSEUDO_COLOR_MIN_HUE,
            max_hue=settings.LAYOUT_PSEUDO_COLOR_MAX_HUE,
            inactive_alpha=settings.DETECTORS_VIEW_INACTIVE_ALPHA,
            panel=settings.DETECTORS_VIEW_PANEL,
            code_columns=settings.DETECTORS_VIEW_CODE_COLUMNS,
            code_cell=settings.DETECTORS_VIEW_CODE_CELL,
        ).save(path, settings.DETECTORS_VIEW_STIMULUS)

    def overview(self, figures: CycleFigures, glyphs: dict[int, str], path: str) -> None:
        """Полоса этапов с петлёй цикла."""
        picture = figures.picture
        side = 180
        y, x, angle = figures.sticks(picture)
        sticks = figures.view(side * 2, picture, backdrop=0.2)
        for row in range(y.shape[0]):
            sticks.stick(y[row], x[row], angle[row])
        level = figures.recorder.level_trace(1)
        ring = self.views(figures.run.angle_encoder).ring(0, figures.example_angle())
        image = Image.open(glyphs[1]).convert("RGB")
        tile = image.width // int(level.map_size[1])
        layout = crop_cells(image, 5 * tile, 5 * tile, tile=tile, top_skip=image.height - int(level.map_size[0]) * tile)
        panels = figures.keyhole_panels(1)
        recognition = Image.open(self.doc("stage7-fives.png")).convert("RGB")
        votes = Image.new("RGB", (300, 300), self.style.background)
        votes.paste(recognition.crop((436, 112, 736, 302)), (0, 55))
        tiles = [
            ("Картинка", "цифра 28×28", Image.fromarray(figures.run.images[picture]).convert("RGB")),
            ("1. Стимулы", "штрихи по середине линии", sticks.image),
            ("2. Код", "детекторы → биты", ring),
            ("3. Раскладка", "похожие коды рядом", layout),
            ("4. Детекторы", "и активации карты", panels["fired"]),
            ("5. Эмбеддинг", "код детали и её место", panels["detail"]),
            ("7. Распознавание", "голоса верхних деталей", votes),
        ]
        about = {0: "штрихи", 1: "обрывки контура", 2: "части цифр", 3: "цифры целиком"}
        levels = [
            (f"уровень {number}",
             f"{about.get(number, '')}\n" + ("место от центра цифры" if number == 0 else f"окно видит {100 * seen:.0f} % цифры"), thumb)
            for number, seen, thumb in figures.loop_thumbnails(glyphs, 150)
        ]
        PipelineOverview(self.style).save(
            tiles,
            levels,
            path,
            loop_from=5,
            loop_to=2,
            loop_title="6. Цикл: детали уровня — вход следующего",
            loop_notes=(["снова кодируется", "окно: что в нём", "и где"], ["наверху окно видит", "цифру целиком —", "детали голосуют"]),
        )
