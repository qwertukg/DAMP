"""Честные мерки цикла: код детали и нормировка — только по обучающим окнам, пустые детали не голосуют (вне статьи; §7.1.12, §9.3)."""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("RERUN", "off")

from damp import settings  # noqa: E402
from damp.activation.details import DetailCoder  # noqa: E402
from damp.activation.level_detectors import Firing  # noqa: E402
from damp.activation.set_embedding import DetectorEmbeddings  # noqa: E402
from damp.cycle.step import CycleStep  # noqa: E402
from damp.decode.level_recognition import LevelRecognition  # noqa: E402
from tests.test_cache import cache_in  # noqa: E402


def _firing() -> Firing:
    # Окна 0–3 — обучающие, 4–5 — проверочные. Детектор 0: на обучающих до 2.0; детектор 1: только на проверочном;
    # детектор 2: на обучающих до 0.5.
    return Firing(
        sets=6,
        rows=np.asarray([0, 1, 2, 3, 4, 4, 5, 3]),
        columns=np.asarray([0, 0, 2, 2, 0, 1, 1, 0]),
        levels=np.asarray([1.0, 2.0, 0.25, 0.5, 3.0, 4.0, 0.5, 0.5]),
    )


class _Detectors:
    built = 3

    def __init__(self, firing: Firing) -> None:
        self._firing = firing

    def fire(self, codes, index, group, sets) -> Firing:  # noqa: ANN001
        return self._firing


class _Codes:
    packed = np.zeros((6, 1), dtype=np.uint8)
    count = 6


class NormalisedLevelsTest(unittest.TestCase):
    TRAINED = np.asarray([True, True, True, True, False, False])

    def test_levels_are_in_unit_interval_and_train_maximum_is_one(self) -> None:
        firing = _firing()
        normal = firing.normalised(self.TRAINED, 3)
        self.assertTrue(np.all(normal.levels >= 0.0) and np.all(normal.levels <= 1.0))
        trained = self.TRAINED[normal.rows]
        for detector in (0, 2):
            own = normal.levels[trained & (normal.columns == detector)]
            self.assertAlmostEqual(float(own.max()), 1.0, places=12)
        np.testing.assert_allclose(normal.levels, [0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.25])
        np.testing.assert_array_equal(normal.rows, firing.rows)
        np.testing.assert_array_equal(normal.columns, firing.columns)

    def test_detector_silent_on_taught_windows_is_cut_at_one(self) -> None:
        normal = _firing().normalised(self.TRAINED, 3)
        by_detector_one = normal.levels[normal.columns == 1]
        np.testing.assert_allclose(by_detector_one, [1.0, 0.5])  # min(4.0, 1) и min(0.5, 1)

    def test_validation_windows_do_not_change_the_normalisation(self) -> None:
        firing = _firing()
        louder = Firing(sets=6, rows=firing.rows, columns=firing.columns,
                        levels=np.where(np.isin(firing.rows, (4, 5)), 100.0, firing.levels))
        a, b = firing.normalised(self.TRAINED, 3), louder.normalised(self.TRAINED, 3)
        taught = self.TRAINED[firing.rows]
        np.testing.assert_array_equal(a.levels[taught], b.levels[taught])

    def test_strength_uses_normalised_levels(self) -> None:
        strength = _firing().normalised(self.TRAINED, 3).strength()
        self.assertTrue(np.all(strength <= 1.0))
        self.assertEqual(float(strength[1]), 1.0)

    def test_step_follows_the_setting(self) -> None:
        firing = _firing()
        detectors, codes = _Detectors(firing), _Codes()
        with settings.overridden({"LADDER_LEVEL_NORMALISATION": "train_max"}):
            self.assertLessEqual(float(CycleStep.fire(detectors, codes, self.TRAINED).levels.max()), 1.0)
        with settings.overridden({"LADDER_LEVEL_NORMALISATION": "none"}):
            np.testing.assert_array_equal(CycleStep.fire(detectors, codes, self.TRAINED).levels, firing.levels)
        with settings.overridden({"LADDER_LEVEL_NORMALISATION": "other"}):
            with self.assertRaises(ValueError):
                CycleStep.fire(detectors, codes, self.TRAINED)


class TrainOnlyCalibrationTest(unittest.TestCase):
    @staticmethod
    def _embeddings() -> DetectorEmbeddings:
        return DetectorEmbeddings(
            detectors=40, saturation_limit=0, target_density=0.25, min_length=8, max_length=4096, seed=0
        )

    @staticmethod
    def _sets():
        # Обучающие окна (последние 20) — по одному детектору; проверочные, лежащие РАНЬШЕ по порядку строк, — по тридцати.
        rng = np.random.default_rng(3)
        rows = [np.full(30, row) for row in range(40)] + [np.full(1, row) for row in range(40, 60)]
        columns = [rng.choice(40, 30, replace=False) for _ in range(40)] + [rng.choice(40, 1) for _ in range(20)]
        rows, columns = np.concatenate(rows), np.concatenate(columns)
        trained = np.arange(60) >= 40
        return rows, columns, np.ones(rows.shape[0]), trained

    def test_length_comes_from_taught_windows_by_mask_not_by_order(self) -> None:
        rows, columns, levels, trained = self._sets()
        packed, length = self._embeddings().encode(rows, columns, levels, 60, calibration=trained)
        only = trained[rows]
        shifted_rows = rows[only] - 40
        alone, alone_length = self._embeddings().encode(shifted_rows, columns[only], levels[only], 20)
        self.assertEqual(length, alone_length)
        np.testing.assert_array_equal(packed[40:], alone)
        everyone, wide = self._embeddings().encode(rows, columns, levels, 60)
        self.assertGreater(wide, length)  # плотные проверочные окна раздули бы код

    def test_other_held_out_windows_give_the_same_taught_codes(self) -> None:
        rows, columns, levels, trained = self._sets()
        packed, length = self._embeddings().encode(rows, columns, levels, 60, calibration=trained)
        rng = np.random.default_rng(9)
        more = np.concatenate([np.full(25, row) for row in range(40)])
        extra_columns = np.concatenate([rng.choice(40, 25, replace=False) for _ in range(40)])
        again, same_length = self._embeddings().encode(
            np.concatenate([rows, more]), np.concatenate([columns, extra_columns]), np.ones(rows.shape[0] + more.shape[0]), 60,
            calibration=trained,
        )
        self.assertEqual(length, same_length)
        np.testing.assert_array_equal(packed[40:], again[40:])

    def test_bad_mask_is_refused(self) -> None:
        rows, columns, levels, _ = self._sets()
        with self.assertRaises(ValueError):
            self._embeddings().encode(rows, columns, levels, 60, calibration=np.ones(5, dtype=bool))

    def test_cache_key_follows_the_mask(self) -> None:
        rows, columns, levels, trained = self._sets()
        firing = Firing(sets=60, rows=rows, columns=columns, levels=levels)
        layers = np.zeros(40, dtype=np.int64)
        with tempfile.TemporaryDirectory() as folder, cache_in(folder):
            _, taught = DetailCoder(1).encode(firing, layers, 40, trained)
            _, everyone = DetailCoder(1).encode(firing, layers, 40, np.ones(60, dtype=bool))
            _, again = DetailCoder(1).encode(firing, layers, 40, trained)
        self.assertGreater(everyone, taught)
        self.assertEqual(again, taught)


class EmptyDetailsTest(unittest.TestCase):
    def test_empty_details_do_not_vote_and_pictures_without_votes_are_wrong(self) -> None:
        labels = np.asarray([0, 1, 0, 1, 0, 0])  # картинки 0–3 обучающие, 4–5 проверочные
        length = 16
        a = np.packbits(np.asarray([1, 1, 0, 0] + [0] * 12, dtype=np.uint8))
        b = np.packbits(np.asarray([0, 0, 1, 1] + [0] * 12, dtype=np.uint8))
        empty = np.zeros(2, dtype=np.uint8)
        # (картинка, код): обучающие — a у нуля, b у единицы, пустые детали у обеих; запросы: 4 — только пустая, 5 — a и пустая.
        picture = np.asarray([0, 0, 1, 1, 2, 3, 4, 5, 5])
        codes = np.stack([a, empty, b, empty, a, b, empty, a, empty])
        recognition = LevelRecognition(labels, (0, 4), neighbours=(1, 3))
        accuracy = recognition.by_vote(codes, length, picture, (4, 6))
        self.assertEqual(accuracy[1], 0.5)  # картинка 4 без голосов — неверно, 5 — верно
        self.assertEqual(accuracy[3], 0.5)

    def test_all_details_empty_gives_zero_not_a_crash(self) -> None:
        labels = np.asarray([0, 1, 0, 0])
        codes = np.zeros((4, 2), dtype=np.uint8)
        accuracy = LevelRecognition(labels, (0, 2), neighbours=(1,)).by_vote(codes, 16, np.asarray([0, 1, 2, 3]), (2, 4))
        self.assertEqual(accuracy[1], 0.0)

    def test_picture_with_zero_response_has_no_answer(self) -> None:
        labels = np.asarray([0, 1, 0, 1])
        response = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
        accuracy = LevelRecognition(labels, (0, 2), neighbours=(1,)).by_response(response, (2, 4))
        self.assertEqual(accuracy[1], 0.5)  # картинка 2 верна, картинка 3 без отклика — неверна
        full = LevelRecognition(labels, (0, 2), neighbours=(1,)).by_response(response[:3], (2, 3))
        self.assertEqual(full[1], 1.0)

    def test_taught_picture_with_zero_response_is_not_a_neighbour(self) -> None:
        # Обучающая картинка 1 (метка 1) молчит; запрос [0.0, 1.0] ближе к ней, чем к картинке 0, но она не голосует.
        labels = np.asarray([0, 1, 0, 0])
        response = np.asarray([[1.0, 0.0], [0.0, 0.0], [0.6, 0.8], [0.0, 1.0]])
        recognition = LevelRecognition(labels, (0, 2), neighbours=(1,))
        self.assertEqual(recognition.by_response(response, (2, 4))[1], 1.0)  # оба запроса получают класс 0
        # Все обучающие молчат: ответов нет, ноль, а не падение.
        silent = response.copy()
        silent[0] = 0.0
        self.assertEqual(recognition.by_response(silent, (2, 4)), {1: 0.0})


class TaughtOnlyFieldTest(unittest.TestCase):
    """Поле окна и шаг мест кода задают только обучающие окна и элементы (честная мерка, вне статьи)."""

    TRAIN = 30

    def _field(self, validation: int, test: int, pixels: int, wide: bool):
        from dataclasses import replace

        from damp.cycle.keyholes import KeyholeBuilder
        from damp.encoding.elements import StickCoder
        from damp.stimuli.digits import DigitGroups, MnistDigits

        with settings.overridden({"LADDER_SAVE_MAPS": False}):
            groups = DigitGroups.load(
                MnistDigits(), train=self.TRAIN, test=test, split="test", validation=validation,
                pixel_shift=(1, 1) if pixels else None, pixel_pictures=pixels,
            )
            elements = StickCoder().encode(groups.atoms)
            # Не обучающие элементы видят очень далеко: при выборке по всем окнам ceil() сдвинул бы поле.
            reach = np.where(elements.picture < self.TRAIN, 0.5, 40.0 if wide else 0.5)
            elements = replace(elements, extent=reach)
            return KeyholeBuilder(groups).build(elements, 1), groups

    def test_box_and_codes_of_taught_windows_do_not_depend_on_held_out_pictures(self) -> None:
        base, groups = self._field(5, 0, 0, wide=False)
        for validation, test, pixels in ((12, 0, 0), (12, 10, 5), (25, 10, 10)):
            with self.subTest(validation=validation, test=test, pixels=pixels):
                other, other_groups = self._field(validation, test, pixels, wide=True)
                self.assertEqual(base.field.box, other.field.box)
                self.assertEqual(base.field.inner, other.field.inner)
                own, theirs = base.pieces.picture < self.TRAIN, other.pieces.picture < self.TRAIN
                self.assertEqual(int(own.sum()), int(theirs.sum()))
                np.testing.assert_array_equal(base.pieces.centre[own], other.pieces.centre[theirs])
                np.testing.assert_array_equal(base.extent[own], other.extent[theirs])


class CycleHonestTest(unittest.TestCase):
    """Цикл на 30 обучающих: проверочные картинки на обучение не влияют; уровни в [0, 1]; новые мерки уровня."""

    TRAIN = 30

    @classmethod
    def setUpClass(cls) -> None:
        from damp.cycle.ladder import MnistLadderRunner
        from damp.cycle.ladder_trace import LadderObserver

        def run(validation: int):
            class Catch(LadderObserver):
                def __init__(self) -> None:
                    self.levels = []

                def level(self, trace) -> None:  # noqa: ANN001
                    self.levels.append(trace)

            catch = Catch()
            # Нормировка train_max задаётся явно: класс проверяет и нормированные уровни (по умолчанию — формула §6.5).
            # Без видов обучающих (§7.1.4): честность мерок от них не зависит, а цикл с ними впятеро дольше.
            changes = {"LADDER_SAVE_MAPS": False, "LADDER_LEVEL_NORMALISATION": "train_max", "LADDER_TRAIN_VIEWS": ()}
            with tempfile.TemporaryDirectory() as folder, settings.overridden(changes):
                runner = MnistLadderRunner(
                    count=cls.TRAIN, evaluate_limit=0, map_max_points=None, shifts=(), file_prefix=str(Path(folder) / "ladder"),
                    observer=catch, validation=validation,
                )
                runner.run()
            return catch.levels, runner

        cls.few, cls.few_runner = run(10)
        cls.many, cls.many_runner = run(25)

    def test_there_are_levels(self) -> None:
        self.assertTrue(self.few and len(self.few) == len(self.many))

    def test_held_out_pictures_do_not_change_the_taught_details(self) -> None:
        # Обучающие — исходные цифры и их виды (§7.1.4, LADDER_TRAIN_VIEWS): граница — из самого прогона.
        limit_a, limit_b = self.few_runner.groups.train_pictures, self.many_runner.groups.train_pictures
        self.assertEqual(limit_a, limit_b)
        for a, b in zip(self.few, self.many):
            with self.subTest(level=a.number):
                taught_a, taught_b = a.keyhole_picture < limit_a, b.keyhole_picture < limit_b
                self.assertEqual(a.detail_length, b.detail_length)
                self.assertEqual(int(taught_a.sum()), int(taught_b.sum()))
                # Те же окна, те же срабатывания и те же уровни; коды и сила деталей обучающих окон совпадают бит в бит
                # (пачка уровня детекторов дополняется до постоянного числа строк, поэтому уровень окна не зависит
                # от того, сколько проверочных окон идёт с ним в одной пачке).
                pairs_a = set(zip(a.fired_keyhole[taught_a[a.fired_keyhole]].tolist(), a.fired_detector[taught_a[a.fired_keyhole]].tolist()))
                pairs_b = set(zip(b.fired_keyhole[taught_b[b.fired_keyhole]].tolist(), b.fired_detector[taught_b[b.fired_keyhole]].tolist()))
                self.assertEqual(pairs_a, pairs_b)
                np.testing.assert_array_equal(a.detail_strength[taught_a], b.detail_strength[taught_b])
                np.testing.assert_array_equal(a.detail_codes[taught_a], b.detail_codes[taught_b])
                self.assertEqual(a.detectors.shape, b.detectors.shape)

    def test_levels_are_normalised_by_taught_windows(self) -> None:
        limit = self.few_runner.groups.train_pictures  # исходные цифры и их виды — все обучающие
        for trace in self.few:
            with self.subTest(level=trace.number):
                levels = trace.fired_level.astype(np.float64)
                self.assertTrue(np.all(levels >= 0.0) and np.all(levels <= 1.0))
                taught = trace.keyhole_picture[trace.fired_keyhole] < limit
                for detector in np.unique(trace.fired_detector[taught]):
                    peak = levels[taught & (trace.fired_detector == detector)].max()
                    self.assertAlmostEqual(float(peak), 1.0, places=5)
                self.assertTrue(np.all(trace.detail_strength <= 1.0))

    def test_article_measures_of_levels(self) -> None:
        for report in self.few_runner.reports:
            with self.subTest(level=report.number):
                self.assertTrue(0.0 < report.energy_mean <= 1.0)
                self.assertGreaterEqual(report.code_resolution, 1.0)
                self.assertGreaterEqual(report.code_largest, 1)
                self.assertTrue(0.0 <= report.detail_empty <= 1.0)
                self.assertTrue(0.0 <= report.detail_full <= 1.0)
                self.assertGreater(report.detail_bits, 0.0)


if __name__ == "__main__":
    unittest.main()
