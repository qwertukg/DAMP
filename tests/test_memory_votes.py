"""Класс из памяти (system/damp/decode/memory_votes.py; §2.2.5, §7.1.13.3, §8.2): голоса кусочков по уровням, ответ —
среднее долей голосов по уровням, уверенность — согласие уровней."""

from __future__ import annotations

import os
import sys
import unittest


os.environ.setdefault("RERUN", "off")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "system"))

import numpy as np  # noqa: E402

from damp import settings  # noqa: E402
from damp.decode.memory_votes import LevelVotes, MemoryVotes  # noqa: E402


def _memory(thresholds: tuple[float, ...] | None = None) -> MemoryVotes:
    """Память двух уровней: цифра и обучающая картинка каждой точки карты."""
    labels = [np.array([8, 8, 2, 5]), np.array([8, 2, 2])]
    pictures = [np.array([10, 11, 12, 13]), np.array([10, 12, 12])]
    return MemoryVotes(labels, pictures, 10, thresholds)


def _level(number: int, digit: int, share: float, knowledge: float, *, voted: int = 10) -> LevelVotes:
    """Голоса уровня: победитель digit с долей share (не меньше 0.1), остальное — поровну между другими цифрами."""
    votes = np.full(10, (1.0 - share) / 9.0)
    votes[digit] = share
    return LevelVotes(number=number, votes=votes, knowledge=knowledge, keyholes=10, voted=voted)


class LevelVotesTest(unittest.TestCase):
    def test_keyhole_votes_with_its_most_active_point(self) -> None:
        """Голос кусочка — цифра его самой активной точки памяти (ближайшее воспоминание, §2.2.5)."""
        # Кусочек 0 зажёг точки 0 («8», 0.6) и 2 («2», 0.9) — голосует «2»; кусочек 1 — точку 1 («8»); кусочек 2 — точку 3 («5»).
        votes = _memory().level(1, 3, np.array([0, 0, 1, 2]), np.array([0, 2, 1, 3]), np.array([0.6, 0.9, 0.7, 0.5]))
        self.assertEqual(votes.voted, 3)
        np.testing.assert_allclose(votes.votes[[2, 5, 8]], [1 / 3, 1 / 3, 1 / 3])
        self.assertAlmostEqual(float(votes.votes.sum()), 1.0)

    def test_knowledge_is_the_mean_best_activation_of_keyholes(self) -> None:
        """Знание уровня — средняя по кусочкам наибольшая активация точки памяти; молчащий кусочек даёт 0."""
        votes = _memory().level(1, 4, np.array([0, 0, 1]), np.array([0, 2, 1]), np.array([0.6, 0.9, 0.7]))
        self.assertAlmostEqual(votes.knowledge, (0.9 + 0.7 + 0.0 + 0.0) / 4)
        self.assertEqual((votes.keyholes, votes.voted), (4, 2))
        np.testing.assert_allclose(votes.votes[[2, 8]], [0.5, 0.5])

    def test_winner_and_its_share(self) -> None:
        votes = _memory().level(2, 3, np.array([0, 1, 2]), np.array([1, 2, 0]), np.array([0.5, 0.5, 0.5]))
        self.assertEqual(votes.digit, 2)
        self.assertAlmostEqual(votes.share, 2 / 3)

    def test_silent_level_has_no_winner(self) -> None:
        votes = _memory().level(1, 3, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0))
        self.assertEqual((votes.digit, votes.share, votes.knowledge, votes.voted), (-1, 0.0, 0.0, 0))
        self.assertFalse(votes.votes.any())

    def test_own_picture_is_left_out(self) -> None:
        """Мерка порога знания: точки памяти своей обучающей картинки не голосуют и в знание не идут."""
        lit = (np.array([0, 0, 1]), np.array([0, 2, 1]), np.array([1.0, 0.6, 0.7]))
        with_own = _memory().level(1, 2, *lit)
        without = _memory().level(1, 2, *lit, without_picture=10)  # точка 0 — из картинки 10
        self.assertEqual(with_own.digit, 8)
        self.assertAlmostEqual(with_own.knowledge, (1.0 + 0.7) / 2)
        self.assertAlmostEqual(without.knowledge, (0.6 + 0.7) / 2)
        np.testing.assert_allclose(without.votes[[2, 8]], [0.5, 0.5])

    def test_equal_activations_vote_the_same_way_every_time(self) -> None:
        """При равных активациях голосует точка с меньшим номером — от порядка записи пар ответ не зависит."""
        first = _memory().level(1, 1, np.array([0, 0]), np.array([2, 0]), np.array([0.8, 0.8]))
        second = _memory().level(1, 1, np.array([0, 0]), np.array([0, 2]), np.array([0.8, 0.8]))
        self.assertEqual(first.digit, 8)
        np.testing.assert_array_equal(first.votes, second.votes)


class MemoryVerdictTest(unittest.TestCase):
    @staticmethod
    def _three() -> MemoryVotes:
        """Память трёх уровней (сами точки для вердикта не нужны)."""
        return MemoryVotes([np.zeros(1, dtype=np.int64)] * 3, [np.zeros(1, dtype=np.int64)] * 3, 10)

    def test_answer_is_the_digit_with_the_highest_mean_share(self) -> None:
        """Ответ — цифра с наибольшей средней долей голосов по уровням (§2.2.5, §7.1.13.3)."""
        verdict = self._three().verdict([_level(1, 5, 0.40, 0.7), _level(2, 8, 0.41, 0.6), _level(3, 8, 0.63, 0.5)])
        self.assertEqual(verdict.digit, 8)
        self.assertAlmostEqual(verdict.share, (0.60 / 9 + 0.41 + 0.63) / 3)
        self.assertAlmostEqual(float(verdict.votes.sum()), 1.0)
        self.assertEqual(int(verdict.votes.argmax()), 8)

    def test_confident_level_outweighs_two_unsure_ones(self) -> None:
        """Усредняются доли, а не ответы уровней: уверенный верхний уровень весит больше двух неуверенных нижних."""
        verdict = self._three().verdict([_level(1, 8, 0.30, 0.7), _level(2, 8, 0.30, 0.6), _level(3, 3, 0.90, 0.5)])
        self.assertEqual(verdict.digit, 3)  # большинство ответов уровней сказало бы «8»
        self.assertEqual((verdict.kind, verdict.agreed), (MemoryVotes.SPLIT, 1))

    def test_all_levels_naming_the_answer_agree(self) -> None:
        verdict = self._three().verdict([_level(1, 8, 0.3, 0.7), _level(2, 8, 0.5, 0.6), _level(3, 8, 0.9, 0.5)])
        self.assertEqual((verdict.kind, verdict.digit, verdict.agreed), (MemoryVotes.AGREE, 8, 3))

    def test_two_top_levels_naming_the_answer_mostly_agree(self) -> None:
        verdict = self._three().verdict([_level(1, 5, 0.40, 0.7), _level(2, 8, 0.41, 0.6), _level(3, 8, 0.63, 0.5)])
        self.assertEqual((verdict.kind, verdict.agreed), (MemoryVotes.MOSTLY, 2))

    def test_answer_named_by_the_lower_levels_only_is_a_split(self) -> None:
        """Цифру назвали два нижних уровня, верхний — другую: уровни расходятся, сколько бы их ни согласилось."""
        verdict = self._three().verdict([_level(1, 8, 0.9, 0.7), _level(2, 8, 0.9, 0.6), _level(3, 3, 0.4, 0.5)])
        self.assertEqual((verdict.kind, verdict.digit, verdict.agreed), (MemoryVotes.SPLIT, 8, 2))

    def test_picture_that_stopped_below_the_top_is_a_split(self) -> None:
        """Рисунок не дошёл до верхнего уровня: ответ есть, но все уровни согласиться не могли."""
        verdict = self._three().verdict([_level(1, 8, 0.9, 0.95), _level(2, 8, 0.8, 0.9)])
        self.assertEqual((verdict.kind, verdict.digit, verdict.agreed), (MemoryVotes.SPLIT, 8, 2))
        self.assertEqual(_memory().verdict([_level(1, 8, 0.9, 0.95), _level(2, 8, 0.8, 0.9)]).kind, MemoryVotes.AGREE)

    def test_silent_level_does_not_dilute_the_answer(self) -> None:
        """Уровень, где не голосовал ни один кусочек, в среднее не входит."""
        silent = LevelVotes(number=1, votes=np.zeros(10), knowledge=0.0, keyholes=4, voted=0)
        verdict = self._three().verdict([silent, _level(2, 8, 0.6, 0.5), _level(3, 8, 0.8, 0.5)])
        self.assertEqual((verdict.kind, verdict.digit, verdict.agreed), (MemoryVotes.MOSTLY, 8, 2))
        self.assertAlmostEqual(verdict.share, 0.7)

    def test_no_votes_no_answer(self) -> None:
        silent = LevelVotes(number=1, votes=np.zeros(10), knowledge=0.0, keyholes=4, voted=0)
        for levels in ([], [silent]):
            verdict = self._three().verdict(levels)
            self.assertEqual((verdict.kind, verdict.digit, verdict.share, verdict.agreed), (MemoryVotes.UNKNOWN, -1, 0.0, 0))
            self.assertFalse(verdict.votes.any())

    def test_equal_mean_shares_give_the_lower_digit(self) -> None:
        first, second = np.zeros(10), np.zeros(10)
        first[7], second[2] = 1.0, 1.0
        levels = [LevelVotes(1, first, 0.5, 4, 4), LevelVotes(2, second, 0.5, 4, 4)]
        self.assertEqual(_memory().verdict(levels).digit, 2)

    def test_thresholds_are_a_quantile_of_training_knowledge(self) -> None:
        """Порог знания уровня — квантиль знания обучающих картинок (каждая — без своих точек памяти)."""
        knowledge = np.column_stack([np.linspace(0.5, 0.9, 101), np.linspace(0.2, 0.6, 101)])
        with settings.overridden({"RECOGNITION_KNOWLEDGE_QUANTILE": 0.05}):
            memory = _memory().calibrated(knowledge)
        np.testing.assert_allclose(memory.thresholds, [0.52, 0.22])
        with settings.overridden({"RECOGNITION_KNOWLEDGE_QUANTILE": 0.5}):
            np.testing.assert_allclose(_memory().calibrated(knowledge).thresholds, [0.7, 0.4])
        self.assertEqual(_memory().calibrated(np.zeros((0, 2))).thresholds, (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
