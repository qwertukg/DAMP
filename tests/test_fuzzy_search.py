"""Нечёткий поиск §2.2.5: ε-окрестность кода и голос деталей по ней (быстрый путь сверен с прямым расчётом)."""
import os
import unittest

import numpy as np

os.environ.setdefault("RERUN", "off")

from damp.decode.level_recognition import LevelRecognition  # noqa: E402
from damp.decode.linear_reader import LinearReader  # noqa: E402
from damp.decode.nearest_codes import NearestCodeClassifier  # noqa: E402


class WithinTest(unittest.TestCase):
    def test_counts_match_direct_cosine(self) -> None:
        rng = np.random.default_rng(0)
        base = rng.random((2500, 64)) < 0.3  # больше одной пачки базы (BLOCK = 1000)
        query = rng.random((1300, 64)) < 0.3
        labels = rng.integers(0, 3, base.shape[0])
        levels = (0.2137, 0.4137)  # пороги не совпадают ни с одним возможным косинусом точно
        classifier = NearestCodeClassifier(np.packbits(base, axis=1), labels, 64, use_gpu=False)
        counts = classifier.within(np.packbits(query, axis=1), levels, 3)
        a, b = query.astype(np.float64), base.astype(np.float64)
        cosine = (a @ b.T) / np.sqrt(np.maximum(a.sum(1)[:, None] * b.sum(1)[None, :], 1.0))
        for index, level in enumerate(levels):
            expected = np.stack([((cosine >= level) & (labels[None, :] == c)).sum(axis=1) for c in range(3)], axis=1)
            np.testing.assert_array_equal(counts[:, index, :], expected)


class AnswersWithinTest(unittest.TestCase):
    @staticmethod
    def codes(*rows: str) -> np.ndarray:
        return np.packbits(np.asarray([[int(bit) for bit in row] for row in rows], dtype=np.uint8), axis=1)

    def test_votes_skip_empty_details_and_silence_is_no_answer(self) -> None:
        # Картинки 0 и 1 — обучающие (классы 0 и 1), 2 — проверочная класса 1. У каждой по детали; у картинки 2 ещё
        # пустая деталь — она не голосует.
        labels = np.asarray([0, 1, 1])
        codes = self.codes("11110000", "00001111", "00001110", "00000000")
        picture = np.asarray([0, 1, 2, 2])
        recognition = LevelRecognition(labels, (0, 2), neighbours=(1,))
        found = recognition.answers_within(codes, 8, picture, (2, 3), (0.5, 0.99))
        self.assertEqual(found[0.5].tolist(), [1])  # косинус с «00001111» ≈ 0.87, с «11110000» — 0
        self.assertEqual(found[0.99].tolist(), [-1])  # в окрестности никого: ответа нет

    def test_each_detail_has_one_vote(self) -> None:
        # Деталь A находит трёх обучающих класса 0; деталь B — одну обучающую класса 1, деталь C — тоже класса 1.
        # Один голос на деталь: класс 1 (две детали) побеждает, хотя найденных класса 0 больше.
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        codes = self.codes(
            "11000000", "11000000", "11000000",  # обучающие класса 0 (картинки 0, 1, 2)
            "00110000", "00001100",  # обучающие класса 1 (картинки 3, 4)
            "11000000", "00110000", "00001100",  # проверочная картинка 5: детали A, B, C
        )
        picture = np.asarray([0, 1, 2, 3, 4, 5, 5, 5])
        recognition = LevelRecognition(labels, (0, 5), neighbours=(1,))
        found = recognition.answers_within(codes, 8, picture, (5, 6), (0.9,))
        self.assertEqual(found[0.9].tolist(), [1])


class AnswersLevelsTest(unittest.TestCase):
    """§7.1.12: уровни активности детекторов как эмбеддинг окна; окна голосуют по косинусу."""

    def test_windows_vote_by_cosine_of_activity_levels(self) -> None:
        # Картинки 0 (класс 0) и 1 (класс 1) — обучающие, 2 и 3 — проверочные класса 1. У картинки 2 одно окно
        # говорит, другое молчит (без срабатываний — не голосует); у картинки 3 окно молчит — ответа нет.
        labels = np.asarray([0, 1, 1, 1])
        firing = {
            "rows": np.asarray([0, 0, 1, 1, 2, 2]),
            "columns": np.asarray([0, 1, 2, 3, 2, 3]),
            "levels": np.asarray([0.9, 0.8, 0.7, 2.5, 0.6, 0.9]),  # 2.5 обрезается до 1
            "detectors": 4,
            "picture": np.asarray([0, 1, 2, 2, 3]),  # окна 3 и 4 молчат
        }
        recognition = LevelRecognition(labels, (0, 2), neighbours=(1,))
        found = recognition.answers_levels(firing, (2, 4))
        self.assertEqual(found[1].tolist(), [1, -1])

    def test_train_max_scale(self) -> None:
        # У каждого детектора наибольший уровень на обучающих окнах — 1: деление на него ничего не меняет.
        labels = np.asarray([0, 1, 1])
        firing = {
            "rows": np.asarray([0, 0, 1, 1, 2, 2]),
            "columns": np.asarray([0, 1, 2, 3, 2, 3]),
            "levels": np.asarray([1.0, 1.0, 1.0, 1.0, 0.6, 0.9]),
            "detectors": 4,
            "picture": np.asarray([0, 1, 2]),
        }
        recognition = LevelRecognition(labels, (0, 2), neighbours=(1,))
        clip = recognition.answers_levels(firing, (2, 3))
        peak = recognition.answers_levels(firing, (2, 3), scale="train_max")
        self.assertEqual(clip[1].tolist(), peak[1].tolist())
        with self.assertRaises(ValueError):
            recognition.answers_levels(firing, (2, 3), scale="other")


class AnswersDominanceTest(unittest.TestCase):
    """§8.2: окно голосует профилями классов своих детекторов с весом уровня активности."""

    def test_windows_vote_by_detector_profiles(self) -> None:
        # Детектор 0 — «нули», детектор 1 — почти «единицы». Окно картинки 2 зажгло детектор 1 сильнее (2.0 → 1).
        labels = np.asarray([0, 1, 1, 0])
        firing = {
            "rows": np.asarray([0, 0]), "columns": np.asarray([0, 1]), "levels": np.asarray([0.3, 2.0]),
            "detectors": 2, "picture": np.asarray([2, 3]),  # у картинки 3 окно молчит — ответа нет
            "dominance": np.asarray([[1.0, 0.0], [0.1, 0.9]]),
        }
        recognition = LevelRecognition(labels, (0, 2), neighbours=(1,))
        self.assertEqual(recognition.answers_dominance(firing, (2, 4)).tolist(), [1, -1])


class LinearReaderTest(unittest.TestCase):
    """§9.3: линейный слой с softmax над нормированными уровнями активации."""

    def test_reader_learns_separable_classes_and_silent_pictures_get_no_answer(self) -> None:
        rng = np.random.default_rng(0)
        centres = np.eye(3, 12)  # класс c зажигает свой детектор c
        labels = rng.integers(0, 3, 90)
        features = np.clip(centres[labels] + 0.3 * rng.random((90, 12)), 0.0, None)
        features[60] = 0.0  # картинка без сработавших детекторов
        parts = {"asked": (60, 90)}
        found = LinearReader.answers_for(features, labels, (0, 60), parts, (0.001, 0.1))
        for decay, answers in found.items():
            asked = answers["asked"]
            self.assertEqual(int(asked[0]), -1)
            self.assertTrue(np.array_equal(asked[1:], labels[61:90]), decay)
        again = LinearReader.answers_for(features, labels, (0, 60), parts, (0.001,))
        np.testing.assert_array_equal(again[0.001]["asked"], found[0.001]["asked"])  # старт с нулей — детерминирован

    def test_probabilities_agree_with_answers(self) -> None:
        """Вероятности softmax: сумма 1, самый вероятный класс — ответ; у нулевого эмбеддинга — нули и ответа нет."""
        rng = np.random.default_rng(1)
        labels = rng.integers(0, 3, 60)
        features = np.clip(np.eye(3, 8)[labels] + 0.3 * rng.random((60, 8)), 0.0, None)
        reader = LinearReader(features[:40], labels[:40], 1e-3)
        asked = np.vstack([features[40:], np.zeros((1, 8))])
        chance = reader.probabilities(asked)
        answers = reader.answers(asked)
        np.testing.assert_allclose(chance[:-1].sum(axis=1), 1.0)
        np.testing.assert_array_equal(chance[:-1].argmax(axis=1), answers[:-1])
        self.assertFalse(chance[-1].any())
        self.assertEqual(int(answers[-1]), -1)


if __name__ == "__main__":
    unittest.main()
