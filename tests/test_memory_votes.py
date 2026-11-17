"""Класс из памяти (system/damp/decode/memory_votes.py; §2.2.5, §7.1.13.3, §8.2): голоса кусочков по уровням и вердикт
«знаю целиком / по частям / не знаю»."""

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
    def test_top_level_that_knows_answers_whole(self) -> None:
        """Знаю целиком: верхний уровень знает рисунок (знание не ниже порога) — ответ его голосов (§7.1.13.3)."""
        verdict = _memory((0.7, 0.5)).verdict([_level(1, 3, 0.9, 0.8), _level(2, 8, 0.6, 0.55)])
        self.assertEqual((verdict.kind, verdict.digit, verdict.level), (MemoryVotes.WHOLE, 8, 2))
        self.assertAlmostEqual(verdict.share, 0.6)

    def test_unknown_whole_with_agreeing_parts_answers_by_parts(self) -> None:
        """По частям: верх не знает, а на уровне ниже за одну цифру больше половины кусочков."""
        verdict = _memory((0.7, 0.5)).verdict([_level(1, 8, 0.9, 0.6), _level(2, 2, 0.6, 0.4)])
        self.assertEqual((verdict.kind, verdict.digit, verdict.level), (MemoryVotes.PARTS, 8, 1))

    def test_parts_need_a_majority(self) -> None:
        """Не знаю: верх не знает, и кусочки нижнего уровня не сходятся на одной цифре."""
        memory = _memory((0.7, 0.5))
        self.assertEqual(memory.verdict([_level(1, 8, 0.5, 0.9), _level(2, 2, 0.9, 0.4)]).kind, MemoryVotes.UNKNOWN)
        with settings.overridden({"RECOGNITION_PARTS_MAJORITY": 0.4}):
            self.assertEqual(memory.verdict([_level(1, 8, 0.5, 0.9), _level(2, 2, 0.9, 0.4)]).kind, MemoryVotes.PARTS)

    def test_top_level_does_not_answer_by_parts(self) -> None:
        """Верхний уровень, не знающий рисунка, «по частям» не отвечает, как бы дружно ни голосовал: его окно — целое."""
        verdict = _memory((0.7, 0.5)).verdict([_level(1, 8, 0.3, 0.9), _level(2, 2, 1.0, 0.4)])
        self.assertEqual((verdict.kind, verdict.digit, verdict.level), (MemoryVotes.UNKNOWN, -1, 0))

    def test_picture_that_stopped_below_the_top_is_not_known_whole(self) -> None:
        memory = _memory((0.7, 0.5))
        self.assertEqual(memory.verdict([_level(1, 8, 0.9, 0.95)]).kind, MemoryVotes.PARTS)
        self.assertEqual(memory.verdict([]).kind, MemoryVotes.UNKNOWN)

    def test_silent_top_level_is_not_known(self) -> None:
        silent = LevelVotes(number=2, votes=np.zeros(10), knowledge=0.0, keyholes=4, voted=0)
        self.assertEqual(_memory().verdict([_level(1, 8, 0.4, 0.9), silent]).kind, MemoryVotes.UNKNOWN)

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
