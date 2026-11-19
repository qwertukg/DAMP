"""Наблюдатель прогона цикла, который запоминает итоги всех проходов — для рисунков README."""
from __future__ import annotations

from damp.article_refs import NEURAL_ANALOGY
from damp.cycle.ladder_trace import LadderLevelTrace, LadderObserver, LadderRunTrace, LadderTopTrace
from damp.logging import LOGGER


class LadderRecorder(LadderObserver):
    """Хранит вход прогона, итог каждого прохода цикла (§6.6) и детали верхнего уровня."""

    def __init__(self) -> None:
        self.run: LadderRunTrace | None = None
        self.levels: list[LadderLevelTrace] = []
        self.top_details: LadderTopTrace | None = None

    def start(self, trace: LadderRunTrace) -> None:
        self.run = trace
        self.levels = []
        self.top_details = None

    def level(self, trace: LadderLevelTrace) -> None:
        self.levels.append(trace)
        LOGGER.event(
            "figures.recorder.level",
            section=NEURAL_ANALOGY,
            data={
                "level": trace.number,
                "keyholes": int(trace.keyhole_picture.shape[0]),
                "map_points": int(trace.map_cells.shape[0]),
                "detectors": int(trace.detectors.shape[0]),
                "fired": int(trace.fired_keyhole.shape[0]),
                "note": "итог прохода цикла запомнен для рисунков: окна, память уровня, детекторы, детали",
            },
        )

    def top(self, trace: LadderTopTrace) -> None:
        self.top_details = trace

    def level_trace(self, number: int) -> LadderLevelTrace:
        for trace in self.levels:
            if trace.number == number:
                return trace
        raise KeyError(f"нет прохода {number}")
