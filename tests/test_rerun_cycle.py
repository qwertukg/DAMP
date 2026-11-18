"""Цикл в Rerun: все вкладки этапов 1–7 заполнены, и при повторе из кеша — теми же записями; части уровня приходят
по ходу счёта, по порядку этапов; ход раскладки и слоёв детекторов рисуется не чаще RERUN_LIVE_INTERVAL; сводка
прогона показывает, какой этап идёт; раскладка панелей ссылается на пути, куда пишет вид."""
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path

os.environ.setdefault("RERUN", "off")

from damp import settings  # noqa: E402
from damp.cycle.ladder_trace import LEVEL_PARTS  # noqa: E402
from damp.figures.rerun_cycle import CYCLE, CODING, DETECTORS, EMBEDDINGS, LAYOUT, OVERVIEW, RECOGNITION, STIMULI, CycleBlueprint, CycleRerunView  # noqa: E402
from tests.test_cache import cache_in  # noqa: E402


class FakeSink:
    """Вместо Rerun: запоминает путь, тип записи и уровень цикла; кадры по ходу счёта — отдельно."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str, int | None]] = []
        self.live: list[tuple[str, str, int | None]] = []

    def log(self, path: str, data: object, *, level: int | None = None, live: bool = False) -> None:
        (self.live if live else self.records).append((path, type(data).__name__, level))


class OrderView(CycleRerunView):
    """Вид, который ещё и запоминает порядок этапов и частей."""

    def __init__(self, sink: FakeSink) -> None:
        super().__init__(sink)
        self.order: list[tuple[str, int, object]] = []

    def stage(self, number: int, stage: int) -> None:
        self.order.append(("stage", number, stage))
        super().stage(number, stage)

    def part(self, number: int, name: str, values: dict) -> None:
        self.order.append(("part", number, name))
        super().part(number, name, values)


class CycleRerunViewTest(unittest.TestCase):
    TRAIN, TEST = 30, 20

    def run_view(self, folder: str, *, live: bool = False) -> tuple[FakeSink, object, OrderView]:
        from damp.cycle.ladder import MnistLadderRunner

        sink = FakeSink()
        view = OrderView(sink)
        runner = MnistLadderRunner(
            count=self.TRAIN, evaluate_limit=self.TEST, map_max_points=None, shifts=(),
            file_prefix=str(Path(folder) / "ladder"), observer=view,
        )
        if live:
            view.follow()  # подписки на ход раскладки и слоёв; окна Rerun нет — без смотрителя вкладок
        try:
            runner.run()
        finally:
            view.close()
        return sink, runner, view

    def test_every_stage_is_filled_also_from_cache(self) -> None:
        with tempfile.TemporaryDirectory() as folder, settings.overridden({"LADDER_SAVE_MAPS": False}):
            with cache_in(str(Path(folder) / "cache")):
                first, runner, view = self.run_view(folder)
                repeated, _, _ = self.run_view(folder)  # всё из кеша шагов
        levels = [report.number for report in runner.reports]
        self.assertTrue(levels)
        paths = {path for path, _, _ in first.records}
        # Вкладка «Детекторы»: вместе с картой уровня стираются слои — на новой карте не видно кругов прошлого уровня,
        # пока детекторы этого уровня строятся (баг, найденный автором 2026-11-18).
        layers = int(settings.LADDER_HIERARCHY_LAYERS)
        for number in levels:
            with self.subTest(level=number, check="слои стираются вместе с картой"):
                order = [path for path, _, level in first.records if level == number]
                start = order.index(f"{DETECTORS}/map")
                self.assertEqual(order[start + 1 : start + 1 + layers], [f"{DETECTORS}/layers/l{k}" for k in range(layers)])
        for stage in (OVERVIEW, STIMULI, CODING, LAYOUT, DETECTORS, EMBEDDINGS, CYCLE, RECOGNITION):
            with self.subTest(stage=stage):
                self.assertTrue(any(path.startswith(stage) for path in paths), stage)
                self.assertIn(f"{stage}/about" if stage != OVERVIEW else f"{OVERVIEW}/summary", paths)
        # Каждый уровень несёт картинки этапов 2–6 и числа (шкала cycle); путь пятёрки — от окна до карты.
        for number in levels:
            on_level = {path for path, _, level in first.records if level == number}
            for path in (f"{CODING}/keyhole/circle", f"{CODING}/keyhole/all", f"{CODING}/code/content", f"{LAYOUT}/map",
                         f"{LAYOUT}/map/five", f"{DETECTORS}/activation/lit", f"{EMBEDDINGS}/code", f"{CYCLE}/keyholes/level",
                         f"{RECOGNITION}/accuracy/response_k1"):
                with self.subTest(level=number, path=path):
                    self.assertIn(path, on_level)
        self.assertIn(f"{RECOGNITION}/confusion", paths)
        # Повтор из кеша: те же записи в том же числе (секунды шагов — другие числа, но те же пути).
        self.assertEqual(Counter(first.records), Counter(repeated.records))
        # Части уровня — по ходу счёта и по порядку этапов: окна (6) → коды (2) → карта (3) → детекторы и срабатывания (4)
        # → детали (5) → узнавание (7); перед каждой частью идёт её этап.
        for number in levels:
            parts = [name for kind, level, name in view.order if kind == "part" and level == number]
            self.assertEqual(parts, list(LEVEL_PARTS), number)
            stages = [stage for kind, level, stage in view.order if kind == "stage" and level == number and stage]
            self.assertEqual(stages, [6, 2, 3, 4, 5, 7], number)
        self.assertEqual(view.order[0], ("stage", 0, 1))
        self.assertEqual(view.order[-1], ("stage", levels[-1], 0))

    def test_live_layout_and_detector_layers(self) -> None:
        """Ход раскладки и слоёв детекторов — кадрами live (их нет при повторе из кеша), остальное — как без них."""
        with tempfile.TemporaryDirectory() as folder, settings.overridden({"LADDER_SAVE_MAPS": False, "RERUN_LIVE_INTERVAL": 0.0}):
            with cache_in(str(Path(folder) / "cache")):
                plain, runner, _ = self.run_view(folder)
            with cache_in(str(Path(folder) / "cache-live")):
                live, live_runner, _ = self.run_view(folder, live=True)
        self.assertEqual(Counter(plain.records), Counter(live.records))
        self.assertEqual([report.glued for report in runner.reports], [report.glued for report in live_runner.reports])
        frames = Counter(path for path, _, _ in live.live)
        self.assertGreater(frames[f"{LAYOUT}/map"], 1)
        self.assertGreater(frames[f"{LAYOUT}/live/quality"], 0)
        self.assertGreater(sum(count for path, count in frames.items() if path.startswith(f"{DETECTORS}/layers/l")), 1)
        self.assertGreater(frames[f"{RECOGNITION}/top_map"], 0)

    def test_blueprint_builds_for_every_tab(self) -> None:
        for name in CycleBlueprint.TABS:
            with self.subTest(tab=name):
                self.assertIsNotNone(CycleBlueprint.build(active=name))


class ProgressSummaryTest(unittest.TestCase):
    """Сводка прогона на вкладке «Обзор», пока он идёт: какой этап идёт, что готово, что ждёт."""

    def summary(self, view: CycleRerunView) -> str:
        return view._progress(*view._stage)

    def test_running_stage_is_marked(self) -> None:
        view = CycleRerunView(FakeSink())
        view.stage(2, 3)  # уровень 2, раскладка: окна (6) и коды (2) уже готовы
        text = self.summary(view)
        self.assertIn("▶ Идёт этап 3. Раскладка · уровень 2", text)
        self.assertIn("| **▶ 3. Раскладка** | **идёт**", text)
        for done in ("1. Стимулы", "2. Кодирование ↻", "6. Цикл ↺"):
            self.assertIn(f"| {done} | ✓ готово |", text)
        for waiting in ("4. Детекторы", "5. Эмбеддинги", "7. Распознавание"):
            self.assertIn(f"| {waiting} | ждёт |", text)
        self.assertIn("| **Итого** | готово 2 из 6 этапов уровня |", text)

    def test_before_levels_and_finished_levels(self) -> None:
        view = CycleRerunView(FakeSink())
        view.stage(0, 1)
        self.assertIn("▶ Идёт этап 1. Стимулы · до первого уровня", self.summary(view))
        view.levels[1] = {"points": 5000, "detectors": 1088, "alone": {1: 0.5}, "seconds": {"memory": 7.0, "detectors": 3.0}}
        view.stage(2, 6)
        text = self.summary(view)
        self.assertIn("| 1 | 5000 | 1088 | 50.0% | 10.0 |", text)
        self.assertIn("| **Итого** | готово уровней: 1 | | | 10.0 |", text)


class TabFollowTest(unittest.TestCase):
    """Вкладка идущего этапа открывается сама: та же раскладка панелей с другой открытой вкладкой; в конце — «Обзор»."""

    def opened(self, view: CycleRerunView, number: int, stage: int) -> str | None:
        from unittest import mock

        from damp.logging import LOGGER

        with mock.patch.object(LOGGER, "use_blueprint") as use:
            view.stage(number, stage)
        if not use.called:
            return None
        with mock.patch.object(CycleBlueprint, "build") as build:
            use.call_args[0][0]()
        return build.call_args.kwargs["active"]

    def test_opens_running_stage_not_faster_than_interval(self) -> None:
        view = CycleRerunView(FakeSink())
        view.follow(tabs=True)
        try:
            with settings.overridden({"RERUN_LIVE_INTERVAL": 60.0}):
                self.assertEqual(self.opened(view, 2, 3), "3. Раскладка")
                self.assertIsNone(self.opened(view, 2, 4))  # рано: не чаще раза в RERUN_LIVE_INTERVAL
                self.assertEqual(self.opened(view, 3, 0), "Обзор")  # конец прогона — всегда
        finally:
            view.close()

    def test_off_without_tabs(self) -> None:
        view = CycleRerunView(FakeSink())
        view.follow()
        try:
            self.assertIsNone(self.opened(view, 1, 2))
        finally:
            view.close()


if __name__ == "__main__":
    unittest.main()
