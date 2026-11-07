from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import main
from damp.encoding.bitarray import BitArray
from damp.layout.damp_layout import (
    AdaptiveLayoutConfig,
    Layout,
    LayoutRunReport,
    StructuralLongPhaseQualityConfig,
    StructuralLongPhaseQualityEvaluator,
)
from damp.logging import LOGGER


class LayoutPhaseTransitionTests(unittest.TestCase):
    def setUp(self) -> None:
        log_patch = patch.object(LOGGER, "event")
        log_patch.start()
        self.addCleanup(log_patch.stop)

    @staticmethod
    def _codes() -> dict[float, list[BitArray]]:
        # У всех пар сходство 8/9: прирост в 1.5 раза недостижим при любой перестановке.
        return {
            0.0: [BitArray.from_indices(17, (*range(8), 8 + index)) for index in range(9)]
        }

    def _layout(self) -> Layout:
        return Layout(
            self._codes(), grid_size=3, empty_ratio=0.0, use_gpu=False,
            colorizer=main.LAYOUT_COLORIZER,
        )

    @staticmethod
    def _evaluator() -> StructuralLongPhaseQualityEvaluator:
        return StructuralLongPhaseQualityEvaluator(
            StructuralLongPhaseQualityConfig(
                local_similarity_gain_min=main.DETECTOR_SPACE_2_LONG_QUALITY_LOCAL_SIMILARITY_GAIN_MIN,
                max_local_discontinuity_ratio=main.DETECTOR_SPACE_2_LONG_QUALITY_MAX_DISCONTINUITY_RATIO,
                max_spatial_noise_ratio=main.DETECTOR_SPACE_2_LONG_QUALITY_MAX_SPATIAL_NOISE_RATIO,
                global_sample_size=main.DETECTOR_SPACE_2_LONG_QUALITY_GLOBAL_SAMPLE_SIZE,
                minimum_neighbours=main.DETECTOR_SPACE_2_LONG_QUALITY_MINIMUM_NEIGHBOURS,
            )
        )

    @staticmethod
    def _report() -> LayoutRunReport:
        return LayoutRunReport(
            mode="long", final_step=226, steps=226, pair_radius=1,
            swap_ratio=0.0, average_swap_ratio=0.0, energy=0.4,
            energy_max_relative_delta=0.0, converged=True,
            convergence_limit_reached=False,
        )

    def test_unattainable_local_gain_does_not_block_converged_long(self) -> None:
        quality = self._evaluator().evaluate(self._layout(), self._report())
        self.assertEqual(quality.local_discontinuity_ratio, 1.0)
        self.assertEqual(quality.spatial_noise_ratio, 1.0)
        self.assertTrue(quality.accepted)

    def test_structural_quality_still_requires_completed_long(self) -> None:
        layout = self._layout()
        for report in (
            None,
            replace(self._report(), converged=False),
            replace(self._report(), convergence_limit_reached=True),
            replace(self._report(), mode="short"),
        ):
            with self.subTest(report=report):
                self.assertFalse(self._evaluator().evaluate(layout, report).accepted)

    def _resume(self, report: LayoutRunReport, *, run_converges: bool) -> list[str]:
        layout = self._layout()
        modes: list[str] = []
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "c2.json")
            checkpoint = path + main.LAYOUT_LONG_PHASE_SUFFIX
            layout.last_run_report = report
            layout.last_steps = report.steps
            layout.save_json(checkpoint)
            expected_positions = list(layout._positions)
            # Проверяем настоящее восстановление checkpoint, включая позиции и отчёт.
            layout.last_run_report = None
            layout._positions = []

            def run(**kwargs) -> None:
                self.assertEqual(layout._positions, expected_positions)
                modes.append(kwargs["mode"])
                if kwargs["mode"] == "long":
                    self.assertEqual(kwargs["step_offset"], report.final_step + 1)
                layout.last_steps = 25
                layout.last_run_report = replace(
                    self._report(), mode=kwargs["mode"],
                    steps=25, final_step=kwargs["step_offset"] + 24,
                    converged=run_converges,
                    convergence_limit_reached=not run_converges,
                )

            with (
                patch("main._load_layout_from_file", return_value=None),
                patch("main.Layout", return_value=layout),
                patch.object(layout, "run", side_effect=run),
                patch.object(layout, "run_continuity_relaxation", return_value=0),
            ):
                kwargs = dict(
                    count=9, total_codes=9, colorizer=main.LAYOUT_COLORIZER,
                    layout_path=path, use_gpu=False,
                    long_quality_evaluator=self._evaluator(),
                )
                if run_converges:
                    main._run_layout(self._codes(), None, **kwargs)
                    self.assertTrue(Path(path + main.LAYOUT_SHORT_PHASE_SUFFIX).exists())
                else:
                    with self.assertRaisesRegex(RuntimeError, "checkpoint сохранён"):
                        main._run_layout(self._codes(), None, **kwargs)
                    self.assertFalse(Path(path + main.LAYOUT_SHORT_PHASE_SUFFIX).exists())
                    self.assertTrue(layout.restore_positions_json(checkpoint))
                    self.assertFalse(layout.last_run_report.converged)
        return modes

    def test_converged_checkpoint_proceeds_to_short_despite_local_diagnostics(self) -> None:
        self.assertEqual(self._resume(self._report(), run_converges=True), ["short"])

    def test_unfinished_checkpoint_resumes_long_before_short(self) -> None:
        report = replace(self._report(), converged=False, convergence_limit_reached=True)
        self.assertEqual(self._resume(report, run_converges=True), ["long", "short"])

    def test_failed_resume_saves_checkpoint_and_blocks_short(self) -> None:
        report = replace(self._report(), converged=False, convergence_limit_reached=True)
        self.assertEqual(self._resume(report, run_converges=False), ["long"])

    def test_limits_grow_without_restarting_steps_or_stability_monitors(self) -> None:
        for mode in ("long", "short"):
            with self.subTest(mode=mode):
                layout = self._layout()
                indices = []

                def step(*, step_index, **kwargs):
                    indices.append(step_index)
                    layout._last_evaluated_pairs = 1
                    return 1 if step_index < 34 else 0

                with (
                    patch.object(layout, "step", side_effect=step),
                    patch.object(layout, "average_local_energy", return_value=1.0),
                    patch.object(layout, "_log_layout_state"),
                    patch.object(LOGGER, "event") as events,
                ):
                    layout.run(
                        steps=2, pairs_per_step=1, pair_radius=1, mode=mode,
                        min_swap_ratio=0.1, min_swap_window=2,
                        energy_radius=1, energy_stability_window=2,
                        energy_stability_every=1, log_visuals=False,
                        adaptive_params=AdaptiveLayoutConfig(start_radius=1, end_radius=1),
                        converge_at_min_radius=True, max_convergence_steps=1,
                        convergence_limit_multiplier=10,
                    )
                self.assertEqual(indices, list(range(36)))
                self.assertTrue(layout.has_converged_phases(mode))
                extensions = [
                    call.kwargs["data"] for call in events.call_args_list
                    if call.args[0] == "layout.run.convergence.limit.increase"
                ]
                self.assertEqual([e["new_limit"] for e in extensions], [30, 300])
                self.assertEqual([e["steps"] for e in extensions], [3, 30])

    def test_primary_long_requires_convergence_without_quality_evaluator(self) -> None:
        layout = self._layout()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "primary.json")

            def run(**kwargs):
                layout.last_run_report = replace(self._report(), converged=False)

            with (
                patch("main._load_layout_from_file", return_value=None),
                patch("main.Layout", return_value=layout),
                patch.object(layout, "run", side_effect=run) as run_mock,
                patch.object(layout, "run_continuity_relaxation") as continuity,
            ):
                with self.assertRaisesRegex(RuntimeError, "long не сошлась"):
                    main._run_layout(self._codes(), None, count=9, total_codes=9,
                                     colorizer=main.LAYOUT_COLORIZER, layout_path=path, use_gpu=False)
                self.assertEqual(run_mock.call_count, 1)
                continuity.assert_not_called()

    def test_unconverged_short_cannot_reach_continuity(self) -> None:
        layout = self._layout()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "primary.json")

            def run(**kwargs):
                layout.last_run_report = replace(
                    self._report(), mode=kwargs["mode"], converged=kwargs["mode"] == "long")

            with (
                patch("main._load_layout_from_file", return_value=None),
                patch("main.Layout", return_value=layout),
                patch.object(layout, "run", side_effect=run),
                patch.object(layout, "run_continuity_relaxation") as continuity,
            ):
                with self.assertRaisesRegex(RuntimeError, "short не сошлась"):
                    main._run_layout(self._codes(), None, count=9, total_codes=9,
                                     colorizer=main.LAYOUT_COLORIZER, layout_path=path, use_gpu=False)
                continuity.assert_not_called()

    def test_checkpoint_preserves_both_phase_confirmations(self) -> None:
        layout = self._layout()
        layout.phase_reports = {mode: replace(self._report(), mode=mode) for mode in ("long", "short")}
        layout.last_run_report = layout.phase_reports["short"]
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "phases.json")
            layout.save_json(path)
            restored = self._layout()
            self.assertTrue(restored.restore_positions_json(path))
            self.assertTrue(restored.has_converged_phases("long", "short"))


if __name__ == "__main__":
    unittest.main()
