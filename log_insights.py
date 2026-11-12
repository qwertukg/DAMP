"""Инсайты по логам на DAMP: типы событий, новизна, похожие инциденты, связи событий.

Конвейер повторяет путь «стимул → структурный эмбеддинг» из §7.1 (рис. 23), где строка лога — слово:

1. строки эталонных логов делятся на слова, строится словарь фрагментов, каждое разбиение строки
   становится точкой кодового пространства (damp.encoding.log_encoder; §7.1.3, §7.1.4, §7.1.6);
2. пространство раскладывается общим Layout: дальняя и ближняя фазы (§5.5, §7.1.7);
3. над раскладкой строится иерархия детекторов (§6.4, §7.1.10), активность детекторов даёт структурный
   эмбеддинг строки (§6.5, §7.1.12) и окна строк — набора стимулов S (§6.5); нижние слои иерархии
   описывают семейство события, верхние — его тип;
4. инсайты: типы и семейства событий — строки с близкими кодами (§2.2.5); новизна — строка известна точно,
   по частям или неизвестна (§7.1.13.3); новые числа — по кодам широких детекторов (§4.2.1, §2.2.2);
   похожие инциденты — нечёткий поиск по эмбеддингам окон (§2.2.5); связи — память пар соседних
   событий и запросы маской (§2.3.1, §2.3.2).

Все параметры — константы LOG_* в main.py. Запуск: python log_insights.py
Разбор идеи и реализации: docs/LOG_INSIGHTS.md
"""
from __future__ import annotations

import colorsys
from collections import Counter, defaultdict
import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from typing import Sequence

import numpy as np

from damp.activation import (
    ActivatedDetector,
    ActivationEngine,
    DbscanClusterer,
    Detector,
    DetectorHierarchyBuilder,
    DetectorLayerConfig,
    DetectorOutputSizer,
)
from damp.article_refs import (
    ASSOCIATION_CODING,
    CONJUNCTION,
    DETECTOR_LEVELS,
    EMBEDDING_PROPERTIES,
    FUZZY_SEARCH,
    INTERPRETABILITY,
    INTERSECTION,
    LAID_OUT_STRUCTURE,
    LAYOUT_PARAMETERS,
    LISTS,
    MORPHOLOGICAL_EMBEDDINGS,
    MORPHOLOGY_ACTIVATION,
    NEURAL_ANALOGY,
    OUTSIDE_ARTICLE,
    POINT_ENERGY,
    QUALITY_ASSESS,
    SIMILARITY_MEASURES,
    SPACE_FILLING,
    STIMULUS_DETECTION,
    UNKNOWN_WORDS,
)
from damp.encoding.bitarray import BitArray
from damp.encoding.damp_encoder import EncoderCodeColorizer
from damp.encoding.log_encoder import (
    BackgroundWords,
    FragmentDictionary,
    LineCode,
    LineFragmenter,
    LogLine,
    LogLineEncoder,
    LogLineReader,
    NumericValueEncoder,
    TokenAlphabet,
    VariableSlots,
    limit_words,
)
from damp.layout.code_table import LayoutCodeTable
from damp.layout.damp_layout import AdaptiveLayoutConfig, Layout, ShortLayoutOptimConfig
from damp.logging import LOGGER
from log_dashboard import LogInsightsDashboard
from main import (  # общий режим: любой вход, куча логов, предел пространства, дашборд
    LOG_BINARY_PROBE_BYTES,
    LOG_DASHBOARD_TIMELINE_COLUMNS,
    LOG_INCIDENT_SEARCH_MAX,
    LOG_INSIGHTS_DASHBOARD_PATH,
    LOG_INSIGHTS_OUT_DIR,
    LOG_LINE_CHARS_MAX,
    LOG_LINES_MAX,
    LOG_PILE_MIN_LINES,
    LOG_PILE_TAIL_MARK,
    LOG_PILE_TARGET_SHARE,
    LOG_RARE_TYPE_LINES_MIN,
    LOG_RARE_TYPE_SHARE,
    LOG_SPACE_POINTS_MAX,
    LOG_WINDOW_MEMORY_MAX,
)
from main import (
    ACTIVATION_USE_GPU,
    LAYOUT_ENERGY_EVAL_EVERY,
    LAYOUT_LOG_VISUALS,
    LAYOUT_PSEUDO_COLOR_MAX_HUE,
    LAYOUT_PSEUDO_COLOR_MIN_HUE,
    LAYOUT_SHORT_ENERGY_EPS,
    LAYOUT_SHORT_PARTITIONS,
    LAYOUT_USE_GPU,
    LOG_ACTIVATION_DETECTOR_THRESHOLD,
    LOG_ACTIVATION_ENERGY_THRESHOLD,
    LOG_BACKGROUND_MIN_LINES,
    LOG_BACKGROUND_SHARE,
    LOG_CODE_BITS,
    LOG_DBSCAN_EPS,
    LOG_DBSCAN_MIN_SAMPLES,
    LOG_DETECTOR_ATTEMPTS_WITHOUT_CHANGE,
    LOG_DETECTOR_LAYER_SPECS,
    LOG_DETECTOR_RADIUS_MIN,
    LOG_EMBEDDING_BITS_MIN,
    LOG_EMBEDDING_MAX_DENSITY,
    LOG_EMBEDDING_SATURATION_MAX,
    LOG_EMBEDDING_SATURATION_MIN,
    LOG_ENERGY_LAMBDA,
    LOG_ENERGY_RADIUS,
    LOG_EXACT_SIMILARITY,
    LOG_FAMILY_LAMBDA,
    LOG_FAMILY_LAYER_LAMBDA,
    LOG_FRAGMENT_SATURATION,
    LOG_FRAGMENT_WORDS_MAX,
    LOG_FRAGMENTS_MAX,
    LOG_INCIDENT_LAMBDA,
    LOG_INSIGHTS_ENERGY_MAP_PATH,
    LOG_INSIGHTS_JSON_PATH,
    LOG_INSIGHTS_MAP_PATH,
    LOG_INSIGHTS_MAP_SCALE,
    LOG_INSIGHTS_REFERENCE,
    LOG_INSIGHTS_REPORT_PATH,
    LOG_INSIGHTS_TARGET,
    LOG_INSIGHTS_TOP,
    LOG_INSIGHTS_TYPES_MAP_PATH,
    LOG_LAYOUT_EMPTY_RATIO,
    LOG_LAYOUT_ETA,
    LOG_LAYOUT_HUE_PERIOD,
    LOG_LAYOUT_LAMBDA,
    LOG_LAYOUT_LAMBDA_MAX,
    LOG_LAYOUT_LAMBDA_STEP,
    LOG_LAYOUT_LOG_EVERY_LONG,
    LOG_LAYOUT_LOG_EVERY_SHORT,
    LOG_LAYOUT_LONG_DISTANCE,
    LOG_LAYOUT_LONG_KERNEL,
    LOG_LAYOUT_MAX_PRECOMPUTE,
    LOG_LAYOUT_MIN_SWAP_RATIO,
    LOG_LAYOUT_MIN_SWAP_WINDOW,
    LOG_LAYOUT_RADIUS_FACTOR,
    LOG_LAYOUT_RADIUS_START_FACTOR,
    LOG_LAYOUT_SHORT_DISTANCE,
    LOG_LAYOUT_SHORT_MIN_SWAP_RATIO,
    LOG_LAYOUT_SIMILARITY,
    LOG_LAYOUT_SWAP_TRIGGER,
    LOG_LAYOUT_SWAP_WINDOW,
    LOG_LINE_PARSE_WORDS_MAX,
    LOG_LINE_WORDS_MAX,
    LOG_NUMBER_SLOT_WORDS,
    LOG_NUMERIC_CODE_BITS,
    LOG_NUMERIC_DECADES,
    LOG_NUMERIC_KNOWN_RATIO,
    LOG_NUMERIC_LAYERS,
    LOG_NUMERIC_MIN_VALUES,
    LOG_NUMERIC_NARROWEST,
    LOG_NUMERIC_OVERLAP,
    LOG_NUMERIC_UNIT,
    LOG_NUMERIC_WIDEST,
    LOG_PROBLEM_WORDS,
    LOG_RELATION_DEPTH,
    LOG_RELATION_LAMBDA,
    LOG_RELATION_MIN_CONFIDENCE,
    LOG_RELATION_MIN_LIFT,
    LOG_RELATION_MIN_SUPPORT,
    LOG_SEED,
    LOG_TOKEN_BITS,
    LOG_TYPE_LAMBDA,
    LOG_TYPE_LAYER_LAMBDA,
    LOG_VARIABLE_MIN_CONTEXT,
    LOG_VARIABLE_MIN_VALUES,
    LOG_WINDOW_EMBEDDING_SATURATION_MAX,
    LOG_WINDOW_LINES,
    LOG_WINDOW_STRIDE,
    LOG_WORD_MAX,
    LOG_WORD_MIN,
    LayoutParameterTuner,
    _sparse_layout_options,
    configure_logging,
)

STATUS_EXACT = "известна точно"
STATUS_PARTIAL = "по частям"
STATUS_UNKNOWN = "неизвестна"


# ------------------------------------------------------------------------------------------- память кодов
class CodeMemory:
    """Память битовых кодов и нечёткий поиск по ней (§2.2.5) дискретной косинусной мерой (§2.2.4.1)."""

    def __init__(self, codes: Sequence[BitArray]) -> None:
        length = len(codes[0]) if codes else 1
        self.bits = np.zeros((len(codes), length), dtype=np.float32)
        for row, code in enumerate(codes):
            self.bits[row] = np.frombuffer(code._bits, dtype=np.uint8, count=length)
        self.ones = self.bits.sum(axis=1)

    def __len__(self) -> int:
        return int(self.bits.shape[0])

    def search(self, code: BitArray) -> np.ndarray:
        """C(a, b) = |a ∧ b| / √(|a|·|b|) кода со всеми кодами памяти; 0 при пустом коде (§2.2.4.1)."""
        query = np.frombuffer(code._bits, dtype=np.uint8).astype(np.float32)
        ones = float(query.sum())
        if ones == 0.0 or len(self) == 0:
            return np.zeros(len(self), dtype=np.float32)
        denominator = np.sqrt(self.ones * ones)
        return np.divide(self.bits @ query, denominator, out=np.zeros(len(self), dtype=np.float32),
                         where=denominator > 0)

    def pairwise(self) -> np.ndarray:
        denominator = np.sqrt(np.outer(self.ones, self.ones))
        return np.divide(self.bits @ self.bits.T, denominator,
                         out=np.zeros((len(self), len(self)), dtype=np.float32), where=denominator > 0)


# ------------------------------------------------------------------------------------------- корпус
class LogCorpus:
    """Строки эталонных и целевых запусков и их стимулы — различные последовательности слов."""

    def __init__(self, reference: Sequence[LogLine], target: Sequence[LogLine]) -> None:
        if not reference:
            raise ValueError("no reference log lines: check LOG_INSIGHTS_REFERENCE")
        self.reference_sources = list(dict.fromkeys(line.source for line in reference))
        self.target_sources = list(dict.fromkeys(line.source for line in target))
        reference_set = set(self.reference_sources)
        self.lines = list(reference) + [line for line in target if line.source not in reference_set]
        self.sources = list(dict.fromkeys(line.source for line in self.lines))
        index_of: dict[tuple[str, ...], int] = {}
        self.stimuli: list[tuple[str, ...]] = []
        self.example: list[LogLine] = []
        self.stimulus_of = np.empty(len(self.lines), dtype=np.int64)
        for row, line in enumerate(self.lines):
            found = index_of.get(line.words)
            if found is None:
                found = len(self.stimuli)
                index_of[line.words] = found
                self.stimuli.append(line.words)
                self.example.append(line)
            self.stimulus_of[row] = found
        target_set = set(self.target_sources)
        self.is_reference_line = np.array([line.source in reference_set for line in self.lines], dtype=bool)
        self.is_target_line = np.array([line.source in target_set for line in self.lines], dtype=bool)
        count = len(self.stimuli)
        self.count = np.bincount(self.stimulus_of, minlength=count)
        self.reference_count = np.bincount(self.stimulus_of[self.is_reference_line], minlength=count)
        self.target_count = np.bincount(self.stimulus_of[self.is_target_line], minlength=count)
        problem_words = set(LOG_PROBLEM_WORDS)
        self.problem = np.array([bool(problem_words.intersection(words)) for words in self.stimuli], dtype=bool)
        self.run_lines: dict[str, list[int]] = defaultdict(list)
        for row, line in enumerate(self.lines):
            self.run_lines[line.source].append(row)
        self.stimulus_runs: list[set[str]] = [set() for _ in self.stimuli]
        for row, line in enumerate(self.lines):
            self.stimulus_runs[int(self.stimulus_of[row])].add(line.source)
        LOGGER.event(
            "logs.corpus",
            section=OUTSIDE_ARTICLE,
            data={
                "reference_runs": len(self.reference_sources),
                "target_runs": len(self.target_sources),
                "lines": len(self.lines),
                "stimuli": count,
                "reference_stimuli": int((self.reference_count > 0).sum()),
                "target_only_stimuli": int(((self.target_count > 0) & (self.reference_count == 0)).sum()),
                "problem_stimuli": int(self.problem.sum()),
            },
        )

    def text(self, stimulus: int, *, limit: int = 110) -> str:
        """Пример исходной строки стимула."""
        text = self.example[stimulus].text.strip()
        return text if len(text) <= limit else text[: limit - 1] + "…"

    def where(self, stimulus: int) -> str:
        line = self.example[stimulus]
        return f"{line.source}:{line.number}"


class LogPile:
    """Куча логов без деления на эталон и цель (вне статьи).

    Записи файла идут по времени, поэтому конец каждого файла — последняя доля target_share его строк —
    проверяется по всему остальному: по началу этого файла и других файлов. Конец файла становится
    отдельным запуском с пометкой mark в имени. Файл короче min_lines строк не делится и целиком идёт
    в эталон.
    """

    def __init__(self, lines: Sequence[LogLine], *, target_share: float, min_lines: int, mark: str) -> None:
        if not 0.0 <= target_share < 1.0:
            raise ValueError("target_share must be in [0, 1)")
        by_source: dict[str, list[LogLine]] = defaultdict(list)
        for line in lines:
            by_source[line.source].append(line)
        self.reference: list[LogLine] = []
        self.target: list[LogLine] = []
        # Номер строки файла, с которой начинается его проверяемый конец.
        self.split: dict[str, int] = {}
        for source, rows in by_source.items():
            tail = int(round(len(rows) * target_share)) if len(rows) >= min_lines else 0
            head = len(rows) - tail
            self.reference.extend(rows[:head])
            if tail:
                self.split[source] = rows[head].number
                self.target.extend(replace(line, source=source + mark) for line in rows[head:])
        LOGGER.event(
            "logs.pile",
            section=OUTSIDE_ARTICLE,
            data={
                "files": len(by_source),
                "split_files": len(self.split),
                "reference_lines": len(self.reference),
                "target_lines": len(self.target),
                "target_share": target_share,
            },
        )


# ------------------------------------------------------------------------------------------- детекция
class LayeredDetection:
    """Детекция по слоям иерархии детекторов (§6.5, §7.1.10).

    «Каждое значение λ_a задаёт свой слой иерархии детекторов» (§7.1.10): детекторы слоя оцениваются
    по активации пространства A_λ(S) с порогом своего слоя λ_d — тем же, с которым считалась энергия
    детектора e_d при его создании (§6.3), поэтому уровень E(d, A) = Σ a·e / e_d сравним с единицей
    на всех слоях. Нижние слои отвечают на сходство уровня семейства, верхние — только на близкие строки.
    Слой — обычный ActivationEngine со своими детекторами и порогом; номера детекторов — сквозные.
    """

    def __init__(self, layout: Layout, detectors: Sequence[Detector], energies: Sequence[float]) -> None:
        self._detectors = list(detectors)
        self._layers: list[tuple[float, np.ndarray, ActivationEngine]] = []
        for level in sorted({detector.lambda_threshold for detector in self._detectors}):
            indices = np.array([i for i, d in enumerate(self._detectors) if d.lambda_threshold == level])
            engine = ActivationEngine(
                layout=layout,
                detectors=[self._detectors[int(i)] for i in indices],
                lambda_threshold=float(level),
                eta=LOG_LAYOUT_ETA,
                energy_threshold=LOG_ACTIVATION_ENERGY_THRESHOLD,
                detector_threshold=LOG_ACTIVATION_DETECTOR_THRESHOLD,
                saturation_limit=1,
                energy_map=energies,
                use_gpu=ACTIVATION_USE_GPU,
            )
            self._layers.append((float(level), indices, engine))
        LOGGER.event(
            "logs.detection.layers",
            section=DETECTOR_LEVELS,
            data={
                "layers": [(level, int(indices.size)) for level, indices, _ in self._layers],
                "detector_threshold": LOG_ACTIVATION_DETECTOR_THRESHOLD,
                "note": "слой детекторов оценивается по активации пространства с порогом своего слоя",
            },
        )

    @property
    def detectors(self) -> tuple[Detector, ...]:
        return tuple(self._detectors)

    def detect(self, stimuli: Sequence[BitArray]) -> tuple[None, list[ActivatedDetector]]:
        """Активные детекторы D_μd(A) всех слоёв для набора стимулов S (§6.5); тот же вид, что у ActivationEngine."""
        active: list[ActivatedDetector] = []
        for _, indices, engine in self._layers:
            for item in engine.detect(stimuli)[1]:
                index = int(indices[item.index])
                active.append(ActivatedDetector(
                    detector=self._detectors[index],
                    activation_level=item.activation_level,
                    active_points=item.active_points,
                    index=index,
                ))
        return None, active


# ------------------------------------------------------------------------------------------- пространство
@dataclass(frozen=True)
class LineReaction:
    """Реакция пространства на строку (§6.5, §7.1.11, §7.1.13.3).

    embedding — код детекторов, активных на всех кодах строки. type_code — отклик на код строки целиком,
    детекторы верхних слоёв (кластеры и их части, §7.1.10). family_code — отклик на коды с акцентом на начало
    строки (§7.1.4.5), детекторы нижних слоёв (области и гиперкластеры).
    """

    embedding: BitArray
    type_code: BitArray
    family_code: BitArray
    active: tuple[tuple[int, float], ...]
    familiarity: float
    nearest_point: int
    status: str


class LogCodeSpace:
    """Кодовое пространство строк: точки — коды разбиений, раскладка, энергии, иерархия детекторов.

    Пространство заполняется кодами разбиений эталонных строк (§7.1.6), раскладывается дальней и ближней
    фазами Layout с параметрами §7.1.7, по раскладке строится иерархия детекторов (§6.4, §7.1.10). Для каждой
    точки хранятся строки, которые её породили (§8.2).
    """

    def __init__(self, corpus: LogCorpus, encoder: LogLineEncoder) -> None:
        self.corpus = corpus
        self.encoder = encoder
        self.stimulus_codes: list[list[LineCode]] = [encoder.encode(words) for words in corpus.stimuli]
        self.points: list[LineCode] = []
        self.point_stimuli: list[list[int]] = []
        index_of: dict[bytes, int] = {}
        reference = [int(stimulus) for stimulus in np.flatnonzero(corpus.reference_count > 0)]
        self.budget: dict = {}
        self.outside = self._outside(reference)
        for stimulus in reference:
            if stimulus in self.outside:
                continue
            for item in self.stimulus_codes[stimulus]:
                key = bytes(item.code._bits)
                found = index_of.get(key)
                if found is None:
                    found = len(self.points)
                    index_of[key] = found
                    self.points.append(item)
                    self.point_stimuli.append([])
                self.point_stimuli[found].append(int(stimulus))
        encoder.log_density([point.code for point in self.points])
        self.memory = CodeMemory([point.code for point in self.points])
        self.layout: Layout | None = None
        self.energies: np.ndarray | None = None
        self.detectors: list[Detector] = []
        self.detection: LayeredDetection | None = None
        self.coder: ActivationEngine | None = None
        self.window_coder: ActivationEngine | None = None
        self._reactions: dict[int, LineReaction] = {}
        self._active: dict[int, list[ActivatedDetector]] = {}
        self._window_cache: dict[frozenset[int], BitArray] = {}
        LOGGER.event(
            "logs.space.fill",
            section=SPACE_FILLING,
            data={
                "stimuli": int((corpus.reference_count > 0).sum()),
                "points": len(self.points),
                "codes_per_stimulus": len(self.points) / max(1, int((corpus.reference_count > 0).sum())),
                "empty_ratio": LOG_LAYOUT_EMPTY_RATIO,
                "grid_side": int(math.ceil(math.sqrt(len(self.points) * (1.0 + LOG_LAYOUT_EMPTY_RATIO)))),
            },
        )

    # ------------------------------------------------------------------ раскладка
    def _hue(self, point: int) -> float:
        """Оттенок диагностической раскраски: первое слово строки-источника точки (в раскладке не участвует)."""
        word = self.corpus.stimuli[self.point_stimuli[point][0]][0]
        digest = hashlib.blake2b(word.encode("utf-8"), digest_size=4).digest()
        return LOG_LAYOUT_HUE_PERIOD * int.from_bytes(digest, "little") / 2**32

    def lay_out(self) -> None:
        """Раскладка §5.2–§5.6 с расписанием §7.1.7: дальняя фаза, затем ближняя."""
        table = LayoutCodeTable.from_codes(
            [point.code for point in self.points],
            hues=[self._hue(index) for index in range(len(self.points))],
            labels=[f"{point.accent}: {point.describe()}" for point in self.points],
        )
        params = LayoutParameterTuner(
            empty_ratio=LOG_LAYOUT_EMPTY_RATIO, start_radius_factor=LOG_LAYOUT_RADIUS_START_FACTOR
        ).tune(len(self.points))
        layout = Layout(
            table,
            empty_ratio=LOG_LAYOUT_EMPTY_RATIO,
            similarity=LOG_LAYOUT_SIMILARITY,
            lambda_threshold=LOG_LAYOUT_LAMBDA,
            eta=LOG_LAYOUT_ETA,
            max_precompute=LOG_LAYOUT_MAX_PRECOMPUTE,
            seed=LOG_SEED,
            use_gpu=LAYOUT_USE_GPU,
            long_subset_size=params.long_subset_size,
            long_subset_refresh=params.long_subset_refresh,
            colorizer=EncoderCodeColorizer(min_hue=LAYOUT_PSEUDO_COLOR_MIN_HUE, max_hue=LAYOUT_PSEUDO_COLOR_MAX_HUE),
            long_distance=LOG_LAYOUT_LONG_DISTANCE,
            short_distance=LOG_LAYOUT_SHORT_DISTANCE,
            long_kernel=LOG_LAYOUT_LONG_KERNEL,
            component_period=LOG_LAYOUT_HUE_PERIOD,
            **_sparse_layout_options(),
        )
        long_radius = max(1, int(layout.width * LOG_LAYOUT_RADIUS_START_FACTOR))
        LOGGER.event(
            "logs.space.layout.schedule",
            section=LAYOUT_PARAMETERS,
            data={
                "points": len(self.points),
                "width": layout.width,
                "similarity": LOG_LAYOUT_SIMILARITY,
                "lambda_start": LOG_LAYOUT_LAMBDA,
                "lambda_step": LOG_LAYOUT_LAMBDA_STEP,
                "lambda_max": LOG_LAYOUT_LAMBDA_MAX,
                "eta": "hard" if LOG_LAYOUT_ETA is None else LOG_LAYOUT_ETA,
                "radius_start": long_radius,
                "swap_trigger": LOG_LAYOUT_SWAP_TRIGGER,
            },
        )
        layout.run(
            steps=params.long_steps,
            pairs_per_step=params.long_pairs_per_step,
            pair_radius=long_radius,
            mode="long",
            min_swap_ratio=LOG_LAYOUT_MIN_SWAP_RATIO,
            min_swap_window=LOG_LAYOUT_MIN_SWAP_WINDOW,
            log_every=LOG_LAYOUT_LOG_EVERY_LONG,
            step_offset=1,
            energy_radius=LOG_ENERGY_RADIUS,
            energy_every=LAYOUT_ENERGY_EVAL_EVERY,
            log_visuals=LAYOUT_LOG_VISUALS,
            adaptive_params=self._adaptive(long_radius),
        )
        short_radius = max(1, min(params.short_local_radius, long_radius))
        layout.run(
            steps=params.short_steps,
            pairs_per_step=params.short_pairs_per_step,
            pair_radius=short_radius,
            mode="short",
            local_radius=params.short_local_radius,
            min_swap_ratio=LOG_LAYOUT_SHORT_MIN_SWAP_RATIO,
            min_swap_window=LOG_LAYOUT_MIN_SWAP_WINDOW,
            log_every=LOG_LAYOUT_LOG_EVERY_SHORT,
            step_offset=1 + layout.last_steps,
            adaptive_params=self._adaptive(short_radius),
            short_optim=ShortLayoutOptimConfig(
                energy_radius=params.short_energy_radius,
                energy_recalc_every=params.short_energy_recalc_every,
                energy_eps=LAYOUT_SHORT_ENERGY_EPS,
                use_weighted_first_point=True,
                similarity_cutoff=LOG_LAYOUT_LAMBDA,
                partitions=LAYOUT_SHORT_PARTITIONS,
            ),
        )
        # Энергия точки (§5.7) — мера её соответствия окружению — считается с порогом нижнего слоя детекторов:
        # та же матрица Ê отсекает неорганизованные точки при построении детекторов и при детекции (§6.3, §6.5).
        layout.set_similarity_params(lambda_threshold=LOG_ENERGY_LAMBDA)
        self.layout = layout
        self.energies = np.asarray(layout.point_energies(LOG_ENERGY_RADIUS), dtype=np.float64)
        LOGGER.event(
            "logs.space.energy",
            section=POINT_ENERGY,
            data={
                "radius": LOG_ENERGY_RADIUS,
                "lambda": LOG_ENERGY_LAMBDA,
                "energy_mean": float(self.energies.mean()),
                "unorganised_points": int((self.energies < LOG_ACTIVATION_ENERGY_THRESHOLD).sum()),
            },
        )
        LOGGER.event(
            "logs.space.quality",
            section=QUALITY_ASSESS,
            data={"average_energy": float(layout.average_local_energy(LOG_ENERGY_RADIUS)), "radius": LOG_ENERGY_RADIUS},
        )

    @staticmethod
    def _adaptive(start_radius: int) -> AdaptiveLayoutConfig:
        return AdaptiveLayoutConfig(
            start_radius=start_radius,
            end_radius=1,
            swap_ratio_trigger=LOG_LAYOUT_SWAP_TRIGGER,
            lambda_step=LOG_LAYOUT_LAMBDA_STEP,
            lambda_max=LOG_LAYOUT_LAMBDA_MAX,
            radius_schedule="swaps",
            radius_factor=LOG_LAYOUT_RADIUS_FACTOR,
            swap_window=LOG_LAYOUT_SWAP_WINDOW,
        )

    # ------------------------------------------------------------------ детекторы
    def _coder(self, detectors: Sequence[Detector], saturation: int) -> ActivationEngine:
        """Движок, собирающий код из активных детекторов colour merge с насыщенностью ≤ σ (§3.3, §6.5)."""
        return ActivationEngine(
            layout=self.layout,
            detectors=detectors,
            lambda_threshold=LOG_ENERGY_LAMBDA,
            eta=LOG_LAYOUT_ETA,
            energy_threshold=LOG_ACTIVATION_ENERGY_THRESHOLD,
            detector_threshold=LOG_ACTIVATION_DETECTOR_THRESHOLD,
            saturation_limit=saturation,
            energy_map=self.energies.tolist(),
            use_gpu=ACTIVATION_USE_GPU,
        )

    def build_detectors(self) -> None:
        """Иерархия детекторов (§6.4, §7.1.10) и размер выходного кода по числу активных детекторов (§6.6)."""
        if self.layout is None or self.energies is None:
            raise RuntimeError("lay_out must be called before build_detectors")
        extent = max(1, min(self.layout.width, self.layout.height))
        layers = tuple(
            DetectorLayerConfig(
                lambda_threshold=float(level),
                activation_radius=max(LOG_DETECTOR_RADIUS_MIN, int(round(extent * float(factor)))),
                min_points=int(min_points),
            )
            for level, factor, min_points in LOG_DETECTOR_LAYER_SPECS
        )
        LOGGER.event(
            "logs.detectors.layers",
            section=DETECTOR_LEVELS,
            data={"layers": [(layer.lambda_threshold, layer.activation_radius, layer.min_points) for layer in layers]},
        )
        detectors = DetectorHierarchyBuilder(
            layout=self.layout,
            energies=self.energies.tolist(),
            layer_configs=layers,
            radius_min=LOG_DETECTOR_RADIUS_MIN,
            output_bits=LOG_EMBEDDING_BITS_MIN,
            energy_threshold=LOG_ACTIVATION_ENERGY_THRESHOLD,
            eta=LOG_LAYOUT_ETA,
            attempts_without_change=LOG_DETECTOR_ATTEMPTS_WITHOUT_CHANGE,
            seed=LOG_SEED,
            clusterer=DbscanClusterer(eps=LOG_DBSCAN_EPS, min_samples=LOG_DBSCAN_MIN_SAMPLES),
        ).build()
        self.detection = LayeredDetection(self.layout, detectors, self.energies.tolist())
        # §6.6: длина и насыщенность кода строки — по среднему числу детекторов, активных на эталонных строках.
        reference = [int(index) for index in np.flatnonzero(self.corpus.reference_count > 0)]
        size, active_lists = DetectorOutputSizer(
            max_density=LOG_EMBEDDING_MAX_DENSITY,
            min_output_bits=LOG_EMBEDDING_BITS_MIN,
            min_saturation=LOG_EMBEDDING_SATURATION_MIN,
            max_saturation=LOG_EMBEDDING_SATURATION_MAX,
        ).measure_with_results(self.detection, [self._stimuli_codes([index]) for index in reference])
        output_bits = size.output_bits
        while output_bits < len(detectors):
            output_bits *= 2
        self.detectors = self._unique_output_bits(detectors, output_bits)
        self.coder = self._coder(self.detectors, size.saturation_limit)
        for index, active in zip(reference, active_lists):
            self._active[index] = active
        LOGGER.event(
            "logs.detectors.output",
            section=NEURAL_ANALOGY,
            data={
                "detectors": len(self.detectors),
                "output_bits": output_bits,
                "saturation": size.saturation_limit,
                "mean_active_detectors": size.mean_active_detectors,
                "note": "каждому детектору — свой случайный бит: длина кода не меньше числа детекторов",
            },
        )

    @staticmethod
    def _unique_output_bits(detectors: Sequence[Detector], output_bits: int) -> list[Detector]:
        """Выходной код детектора — один случайный бит (§6.3); биты выдаются случайной перестановкой без повторов.

        При случайном бите с повторами несколько детекторов делят бит, и у строк с коротким кодом возникает
        ложное сходство; без повторов код строки точно описывает набор её активных детекторов.
        """
        order = np.random.default_rng(LOG_SEED).permutation(output_bits)[: len(detectors)]
        return [
            Detector(
                center=detector.center,
                radius=detector.radius,
                lambda_threshold=detector.lambda_threshold,
                energy=detector.energy,
                points_count=detector.points_count,
                output_code=BitArray.from_indices(output_bits, [int(bit)]),
                label=detector.label,
            )
            for detector, bit in zip(detectors, order)
        ]

    def size_window_code(self, windows: Sequence[frozenset[int]]) -> None:
        """Код окна (§6.5, набор стимулов S): своя длина и насыщенность по числу детекторов, активных на окнах (§6.6)."""
        if self.detection is None:
            raise RuntimeError("build_detectors must be called before size_window_code")
        unique = list(dict.fromkeys(windows))
        size, active_lists = DetectorOutputSizer(
            max_density=LOG_EMBEDDING_MAX_DENSITY,
            min_output_bits=LOG_EMBEDDING_BITS_MIN,
            min_saturation=LOG_EMBEDDING_SATURATION_MIN,
            max_saturation=LOG_WINDOW_EMBEDDING_SATURATION_MAX,
        ).measure_with_results(self.detection, [self._stimuli_codes(sorted(window)) for window in unique])
        # Биты детекторов те же, что в коде строки; насыщенность — своя, в пределах плотности кода (§2).
        output_bits = len(self.detectors[0].output_code)
        saturation = min(size.saturation_limit, int(output_bits * LOG_EMBEDDING_MAX_DENSITY))
        self.window_coder = self._coder(self.detectors, saturation)
        for window, active in zip(unique, active_lists):
            self._window_cache[window] = self.window_coder.embedding_from_active(active)
        LOGGER.event(
            "logs.windows.output",
            section=NEURAL_ANALOGY,
            data={
                "windows": len(unique),
                "output_bits": output_bits,
                "saturation": saturation,
                "mean_active_detectors": size.mean_active_detectors,
            },
        )

    def primary_code(self, stimulus: int, view: str) -> BitArray:
        """Первичный код строки для взгляда view: код первого разбиения с этим взглядом, иначе код целой строки."""
        codes = self.stimulus_codes[stimulus]
        for item in codes:
            if item.view == view:
                return item.code
        return codes[0].code

    def _stimuli_codes(self, stimuli: Sequence[int]) -> list[BitArray]:
        return [item.code for index in stimuli for item in self.stimulus_codes[index]]

    # ------------------------------------------------------------------ детекция
    def _outside(self, reference: Sequence[int]) -> frozenset[int]:
        """Строки эталона, не вошедшие в пространство из-за предела точек LOG_SPACE_POINTS_MAX (вне статьи).

        Если кодов эталона больше предела, пространство заполняют коды самых частых видов строк. Остальные
        строки считаются известными (в эталоне они были), а к типам и семействам относятся по отклику
        детекторов, как любая строка не из пространства (§7.1.13.3).
        """
        counts = self.corpus.reference_count
        codes = sum(len(self.stimulus_codes[stimulus]) for stimulus in reference)
        inside = set(reference)
        if codes > LOG_SPACE_POINTS_MAX:
            inside = set()
            used = 0
            for stimulus in sorted(reference, key=lambda index: (-int(counts[index]), index)):
                used += len(self.stimulus_codes[stimulus])
                if used > LOG_SPACE_POINTS_MAX:
                    break
                inside.add(stimulus)
        lines = int(sum(int(counts[stimulus]) for stimulus in reference))
        lines_inside = int(sum(int(counts[stimulus]) for stimulus in inside))
        self.budget = {
            "points_max": LOG_SPACE_POINTS_MAX,
            "codes": int(codes),
            "stimuli": len(reference),
            "stimuli_inside": len(inside),
            "lines_inside_share": round(lines_inside / lines, 4) if lines else 0.0,
        }
        LOGGER.event("logs.space.budget", section=OUTSIDE_ARTICLE, data=self.budget)
        return frozenset(reference) - inside

    def reaction(self, stimulus: int) -> LineReaction:
        """Активация пространства кодами разбиений строки и её эмбеддинг (§7.1.11, §7.1.12).

        Знакомость — среднее по кодам строки наибольшего сходства с точками пространства. Строка известна
        точно, если у каждого её кода есть совпадающая точка; известна по частям, если совпадения нет,
        но активны детекторы; иначе неизвестна (§7.1.13.3).
        """
        cached = self._reactions.get(stimulus)
        if cached is not None:
            return cached
        if self.detection is None or self.coder is None:
            raise RuntimeError("build_detectors must be called before reaction")
        active = self._active.get(stimulus)
        if active is None:
            _, active = self.detection.detect(self._stimuli_codes([stimulus]))
            self._active[stimulus] = active
        best = np.zeros(len(self.stimulus_codes[stimulus]), dtype=np.float64)
        nearest = -1
        nearest_value = -1.0
        for row, item in enumerate(self.stimulus_codes[stimulus]):
            similarity = self.memory.search(item.code)
            point = int(np.argmax(similarity))
            best[row] = float(similarity[point])
            if best[row] > nearest_value:
                nearest_value, nearest = best[row], point
        # Строка эталона, не вошедшая в пространство из-за предела точек, тоже известна: в эталоне она была.
        exact = (bool(best.size) and bool((best >= LOG_EXACT_SIMILARITY).all())) or stimulus in self.outside
        status = STATUS_EXACT if exact else (STATUS_PARTIAL if active else STATUS_UNKNOWN)
        fine = [
            item for item in self._view_active(stimulus, "whole")
            if item.detector.lambda_threshold >= LOG_TYPE_LAYER_LAMBDA
        ]
        coarse = [
            item for item in self._view_active(stimulus, "head")
            if item.detector.lambda_threshold <= LOG_FAMILY_LAYER_LAMBDA
        ]
        result = LineReaction(
            embedding=self.coder.embedding_from_active(active),
            type_code=self.coder.embedding_from_active(fine),
            family_code=self.coder.embedding_from_active(coarse),
            active=tuple((item.index, item.activation_level) for item in active),
            familiarity=float(best.mean()) if best.size else 0.0,
            nearest_point=nearest,
            status=status,
        )
        self._reactions[stimulus] = result
        LOGGER.event(
            "logs.line.reaction",
            section=MORPHOLOGY_ACTIVATION,
            data={
                "stimulus": stimulus,
                "codes": int(best.size),
                "active_detectors": len(active),
                "embedding_bits": result.embedding.count(),
                "type_bits": result.type_code.count(),
                "family_bits": result.family_code.count(),
                "familiarity": result.familiarity,
                "status": status,
            },
        )
        return result

    def _view_active(self, stimulus: int, view: str) -> list[ActivatedDetector]:
        """Детекторы, активные на кодах одного взгляда на строку; у строки без таких кодов — на коде целой строки."""
        codes = [item.code for item in self.stimulus_codes[stimulus] if item.view == view]
        if not codes:
            codes = [item.code for item in self.stimulus_codes[stimulus] if item.view == "whole"]
        return self.detection.detect(codes)[1] if codes else []

    def window_embedding(self, stimuli: frozenset[int]) -> BitArray:
        """Эмбеддинг окна: детекция набора стимулов S — a_ji = max по s ∈ S (§6.5)."""
        cached = self._window_cache.get(stimuli)
        if cached is None:
            if self.window_coder is None or self.detection is None:
                raise RuntimeError("size_window_code must be called before window_embedding")
            _, active = self.detection.detect(self._stimuli_codes(sorted(stimuli)))
            cached = self.window_coder.embedding_from_active(active)
            self._window_cache[stimuli] = cached
        return cached

    def point_owner(self, point: int) -> int:
        """Строка-источник точки: самая частая из породивших её (§8.2)."""
        return max(self.point_stimuli[point], key=lambda stimulus: self.corpus.count[stimulus])

    # ------------------------------------------------------------------ картинки
    def save_maps(self, map_path: str, energy_path: str) -> None:
        """Карта пространства в псевдоцветах кодов (§5.14) и композиция V · Ê (рис. 18)."""
        from PIL import Image

        if self.layout is None or self.energies is None:
            raise RuntimeError("lay_out must be called before save_maps")
        scale = LOG_INSIGHTS_MAP_SCALE
        size = (self.layout.width * scale, self.layout.height * scale)
        Image.fromarray(self.layout.render_image()).resize(size, Image.Resampling.NEAREST).save(map_path)
        Image.fromarray(self.layout.render_energy_image(self.energies.tolist())).resize(
            size, Image.Resampling.NEAREST
        ).save(energy_path)
        LOGGER.event(
            "logs.space.maps.save",
            section=LAID_OUT_STRUCTURE,
            data={"map": map_path, "energy_map": energy_path, "width": self.layout.width, "scale": scale},
        )


# ------------------------------------------------------------------------------------------- типы событий
@dataclass
class EventType:
    index: int
    centre: int
    members: list[int]
    label: str
    lines: int
    runs: set[str]
    problem: bool
    family: int = -1


@dataclass
class EventFamily:
    index: int
    members: list[int]
    label: str
    lines: int
    types: int = 0


class EventTypes:
    """Типы и семейства событий: строки с близкими структурными кодами (§2.2.5, §7.1.10, §7.1.13).

    Центр очередного типа — самая частая строка без типа; в тип входят строки без типа, чей код типа
    (отклик на строку целиком, детекторы верхних слоёв) похож на код центра не меньше LOG_TYPE_LAMBDA.
    Семейства строятся так же по коду семейства (отклик на начало строки, детекторы нижних слоёв) с порогом
    LOG_FAMILY_LAMBDA. Строки без активных детекторов нужных слоёв группируются между собой по первичным
    кодам (_with_silent).
    """

    def __init__(self, corpus: LogCorpus, space: LogCodeSpace) -> None:
        self.corpus = corpus
        self.space = space
        count = len(corpus.stimuli)
        reactions = [space.reaction(index) for index in range(count)]
        self.embeddings = [reaction.embedding for reaction in reactions]
        self.codes = [reaction.type_code for reaction in reactions]
        self.memory = CodeMemory(self.codes)
        self.type_of, groups = self._with_silent(
            *self._groups(self.memory, self.codes, LOG_TYPE_LAMBDA), self.codes, "whole", LOG_TYPE_LAYER_LAMBDA
        )
        self.types = [
            EventType(
                index=index,
                centre=members[0],
                members=members,
                label=self._label(members),
                lines=int(corpus.count[members].sum()),
                runs=set().union(*(corpus.stimulus_runs[m] for m in members)),
                problem=bool(corpus.problem[members].any()),
            )
            for index, members in enumerate(groups)
        ]
        family_codes = [reaction.family_code for reaction in reactions]
        self.family_of, groups = self._with_silent(
            *self._groups(CodeMemory(family_codes), family_codes, LOG_FAMILY_LAMBDA),
            family_codes, "head", LOG_FAMILY_LAYER_LAMBDA,
        )
        self.families = [
            EventFamily(index=index, members=members, label=self._label(members, limit=4),
                        lines=int(corpus.count[members].sum()))
            for index, members in enumerate(groups)
        ]
        for event_type in self.types:
            # Семейство типа — семейство большинства его строк.
            event_type.family = Counter(int(self.family_of[m]) for m in event_type.members).most_common(1)[0][0]
            self.families[event_type.family].types += 1
        LOGGER.event(
            "logs.types",
            section=FUZZY_SEARCH,
            data={
                "types": len(self.types),
                "single_line_types": sum(len(t.members) == 1 for t in self.types),
                "without_type_code": sum(self.codes[t.centre].count() == 0 for t in self.types),
                "type_layers_from": LOG_TYPE_LAYER_LAMBDA,
                "lambda": LOG_TYPE_LAMBDA,
            },
        )
        LOGGER.event(
            "logs.families",
            section=DETECTOR_LEVELS,
            data={
                "families": len(self.families),
                "largest": [f"{f.label}: {len(f.members)}" for f in sorted(self.families, key=lambda f: -len(f.members))[:5]],
                "family_layers_to": LOG_FAMILY_LAYER_LAMBDA,
                "lambda": LOG_FAMILY_LAMBDA,
            },
        )

    def _groups(
        self, memory: CodeMemory, codes: Sequence[BitArray], threshold: float, indices: Sequence[int] | None = None
    ) -> tuple[np.ndarray, list[list[int]]]:
        """Группы вокруг центров: центр — самая частая строка без группы, члены — нечёткий поиск по её коду."""
        indices = np.arange(len(codes)) if indices is None else np.asarray(indices, dtype=np.int64)
        group_of = np.full(len(codes), -1, dtype=np.int64)
        groups: list[list[int]] = []
        for centre in np.argsort(-self.corpus.count[indices], kind="stable"):
            centre = int(centre)
            if group_of[centre] >= 0:
                continue
            similar = (memory.search(codes[centre]) >= threshold) & (group_of < 0)
            members = [centre] + [int(i) for i in np.flatnonzero(similar) if int(i) != centre]
            group_of[members] = len(groups)
            groups.append([int(indices[member]) for member in members])
        return group_of, groups

    def _with_silent(
        self, group_of: np.ndarray, groups: list[list[int]], codes: Sequence[BitArray], view: str, threshold: float
    ) -> tuple[np.ndarray, list[list[int]]]:
        """Группы, в которых строки без отклика детекторов собраны по первичным кодам (§2.2.5).

        Строка, на которую не откликнулся ни один детектор нужных слоёв, кода группы не имеет. Таких строк
        много, когда пространство меньше лога: в нём только самые частые виды строк. Между собой они
        сравниваются напрямую — нечётким поиском по первичным кодам взгляда view с порогом нижнего из
        слоёв детекторов, которые дали бы этот код.
        """
        silent = [index for index, code in enumerate(codes) if code.count() == 0]
        if len(silent) < 2:
            return group_of, groups
        primary = [self.space.primary_code(index, view) for index in silent]
        _, found = self._groups(CodeMemory(primary), primary, threshold, silent)
        groups = [members for members in groups if codes[members[0]].count() > 0] + found
        group_of = np.full(len(codes), -1, dtype=np.int64)
        for index, members in enumerate(groups):
            group_of[members] = index
        LOGGER.event(
            "logs.groups.silent",
            section=FUZZY_SEARCH,
            data={"view": view, "stimuli": len(silent), "groups": len(found), "threshold": threshold},
        )
        return group_of, groups

    def _label(self, members: Sequence[int], *, limit: int = 8) -> str:
        """Подпись группы: слова центра, общие для всех её строк, по порядку (§8.2)."""
        words = self.corpus.stimuli[members[0]]
        if len(members) > 1:
            shared = set(words).intersection(*(set(self.corpus.stimuli[m]) for m in members))
            words = tuple(dict.fromkeys(word for word in words if word in shared)) or words[:2]
        text = " ".join(words[:limit]) + (" …" if len(words) > limit else "")
        return text if len(members) == 1 else f"{text} (видов строк: {len(members)})"

    def line_types(self) -> np.ndarray:
        return self.type_of[self.corpus.stimulus_of]

    def quality(self) -> dict:
        """Сохраняет ли эмбеддинг сходство первичных кодов (§6.2: коды наследуют топологию стимулов).

        Сходство двух строк по первичным кодам — наибольшее сходство между их разбиениями; оно сравнивается
        со сходством эмбеддингов по всем парам строк с непустыми эмбеддингами.
        """
        space = self.space
        count = len(self.corpus.stimuli)
        codes = [item.code for index in range(count) for item in space.stimulus_codes[index]]
        owner = np.array([index for index in range(count) for _ in space.stimulus_codes[index]])
        pair = CodeMemory(codes).pairwise()
        starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
        primary = np.maximum.reduceat(np.maximum.reduceat(pair, starts, axis=0), starts, axis=1)
        embedded = CodeMemory(self.embeddings).pairwise()
        filled = np.array([code.count() > 0 for code in self.embeddings])
        rows, cols = np.triu_indices(count, 1)
        keep = filled[rows] & filled[cols]
        a, b = primary[rows[keep], cols[keep]], embedded[rows[keep], cols[keep]]
        correlation = float(np.corrcoef(a, b)[0, 1]) if a.size > 2 and a.std() > 0 and b.std() > 0 else 0.0
        close = a >= LOG_TYPE_LAMBDA
        result = {
            "pairs": int(a.size),
            "correlation": round(correlation, 3),
            "similar_primary_pairs": int(close.sum()),
            "embedding_similarity_of_similar": round(float(b[close].mean()), 3) if close.any() else 0.0,
            "embedding_similarity_of_others": round(float(b[~close].mean()), 3) if (~close).any() else 0.0,
            "stimuli_without_detectors": int((~filled).sum()),
        }
        LOGGER.event("logs.embedding.quality", section=EMBEDDING_PROPERTIES, data=result)
        return result

    def save_map(self, path: str) -> None:
        """Карта семейств: цвет точки — семейство её строки-источника, серый — семейство из одной строки (§8.2)."""
        from PIL import Image, ImageDraw

        layout = self.space.layout
        scale = LOG_INSIGHTS_MAP_SCALE
        image = Image.new("RGB", (layout.width * scale, layout.height * scale), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        golden = (math.sqrt(5.0) - 1.0) / 2.0
        for point, (y, x) in enumerate(layout.cells_array()):
            family = self.families[int(self.family_of[self.space.point_owner(point)])]
            if len(family.members) == 1:
                colour = (70, 70, 70)
            else:
                r, g, b = colorsys.hsv_to_rgb((family.index * golden) % 1.0, 0.7, 0.95)
                colour = (int(r * 255), int(g * 255), int(b * 255))
            draw.rectangle([x * scale, y * scale, (x + 1) * scale - 1, (y + 1) * scale - 1], fill=colour)
        image.save(path)
        LOGGER.event("logs.families.map.save", section=INTERPRETABILITY, data={"path": path})


class DetectorStructure:
    """Слои иерархии детекторов: сколько детекторов, какого размера и сколько из них активирует строка (§7.1.10, §8.2)."""

    def __init__(self, corpus: LogCorpus, space: LogCodeSpace) -> None:
        reference = np.flatnonzero(corpus.reference_count > 0)
        active = Counter()
        for stimulus in reference:
            for detector, _ in space.reaction(int(stimulus)).active:
                active[space.detectors[detector].lambda_threshold] += 1
        self.rows: list[dict] = []
        for level in sorted({detector.lambda_threshold for detector in space.detectors}):
            layer = [detector for detector in space.detectors if detector.lambda_threshold == level]
            self.rows.append({
                "lambda": level,
                "detectors": len(layer),
                "radius_mean": round(float(np.mean([d.radius for d in layer])), 2),
                "points_mean": round(float(np.mean([d.points_count for d in layer])), 1),
                "active_per_line": round(active[level] / max(1, reference.size), 1),
                "role": "семейство" if level <= LOG_FAMILY_LAYER_LAMBDA
                else ("тип" if level >= LOG_TYPE_LAYER_LAMBDA else "промежуточный"),
            })
        LOGGER.event(
            "logs.detectors.structure",
            section=INTERPRETABILITY,
            data={"layers": [(row["lambda"], row["detectors"], row["active_per_line"]) for row in self.rows]},
        )


# ------------------------------------------------------------------------------------------- новизна
class NoveltyFinder:
    """Новые строки целевых логов и неорганизованные строки эталона (§7.1.13.3, §5.14, §5.7)."""

    def __init__(self, corpus: LogCorpus, space: LogCodeSpace, types: EventTypes) -> None:
        self.corpus = corpus
        self.space = space
        self.types = types

    def new_lines(self) -> list[dict]:
        """Строки целевых логов, которых пространство не знает точно: по частям или вовсе (§7.1.13.3)."""
        corpus = self.corpus
        result = []
        for stimulus in np.flatnonzero(corpus.target_count > 0):
            stimulus = int(stimulus)
            reaction = self.space.reaction(stimulus)
            if reaction.status == STATUS_EXACT:
                continue
            nearest = self.space.point_owner(reaction.nearest_point) if reaction.nearest_point >= 0 else -1
            result.append({
                "status": reaction.status,
                "familiarity": round(reaction.familiarity, 2),
                "active_detectors": len(reaction.active),
                "count": int(corpus.target_count[stimulus]),
                "problem": bool(corpus.problem[stimulus]),
                "where": corpus.where(stimulus),
                "text": corpus.text(stimulus),
                # Ближайшая известная строка осмысленна, только когда новая строка узнана по частям.
                "nearest_known": corpus.text(nearest, limit=80)
                if nearest >= 0 and reaction.status == STATUS_PARTIAL else "",
                "stimulus": stimulus,
            })
        result.sort(key=lambda item: (not item["problem"], item["status"] != STATUS_UNKNOWN, item["familiarity"]))
        statuses = Counter(self.space.reaction(int(s)).status for s in np.flatnonzero(corpus.target_count > 0))
        LOGGER.event(
            "logs.insight.novelty",
            section=UNKNOWN_WORDS,
            data={
                "target_stimuli": int((corpus.target_count > 0).sum()),
                "exact": statuses[STATUS_EXACT],
                "partial": statuses[STATUS_PARTIAL],
                "unknown": statuses[STATUS_UNKNOWN],
            },
        )
        return result

    def unorganised(self) -> list[dict]:
        """Эталонные строки, не вошедшие в структуру: у всех их точек низкая энергия (§5.7, §5.14)."""
        corpus = self.corpus
        best = np.zeros(len(corpus.stimuli), dtype=np.float64)
        inside = np.zeros(len(corpus.stimuli), dtype=bool)
        for point, owners in enumerate(self.space.point_stimuli):
            for stimulus in owners:
                best[stimulus] = max(best[stimulus], float(self.space.energies[point]))
                inside[stimulus] = True
        result = []
        for stimulus in np.flatnonzero(inside & (best < LOG_ACTIVATION_ENERGY_THRESHOLD)):
            stimulus = int(stimulus)
            result.append({
                "energy": round(float(best[stimulus]), 3),
                "count": int(corpus.count[stimulus]),
                "runs": len(corpus.stimulus_runs[stimulus]),
                "problem": bool(corpus.problem[stimulus]),
                "where": corpus.where(stimulus),
                "text": corpus.text(stimulus),
            })
        result.sort(key=lambda item: (not item["problem"], item["count"], item["energy"]))
        LOGGER.event(
            "logs.insight.unorganised",
            section=LAID_OUT_STRUCTURE,
            data={"stimuli": len(result), "energy_threshold": LOG_ACTIVATION_ENERGY_THRESHOLD},
        )
        return result


class NumericNovelty:
    """Новые значения чисел в целевых логах (§4.2.1, §2.2.1, §2.2.2).

    Слот числа — тип события и слова перед числом. Все эталонные значения слота объединяются в один код
    (конъюнкция кодов широких детекторов, §2.2.1): он описывает, какие рецептивные поля слот уже занимал.
    Целевое значение проверяется пересечением (§2.2.2): доля бит его кода, уже известных слоту. Самый
    широкий слой с неизвестным битом показывает масштаб отклонения.
    """

    def __init__(self, corpus: LogCorpus, types: EventTypes, encoder: NumericValueEncoder) -> None:
        self.corpus = corpus
        self.types = types
        self.encoder = encoder

    def find(self) -> list[dict]:
        corpus = self.corpus
        line_types = self.types.line_types()
        known: dict[tuple[int, str], set[int]] = defaultdict(set)
        values: dict[tuple[int, str], list[float]] = defaultdict(list)
        for row in np.flatnonzero(corpus.is_reference_line):
            for slot, value in corpus.lines[int(row)].numbers:
                key = (int(line_types[row]), slot)
                known[key].update(self.encoder.bits(value))
                values[key].append(value)
        best: dict[tuple[int, str], dict] = {}
        checked = 0
        for row in np.flatnonzero(corpus.is_target_line & ~corpus.is_reference_line):
            line = corpus.lines[int(row)]
            for slot, value in line.numbers:
                key = (int(line_types[row]), slot)
                if len(values.get(key, ())) < LOG_NUMERIC_MIN_VALUES:
                    continue
                checked += 1
                bits = self.encoder.bits(value)
                unknown = [bit for bit in bits if bit not in known[key]]
                ratio = 1.0 - len(unknown) / max(1, len(bits))
                if ratio >= LOG_NUMERIC_KNOWN_RATIO:
                    continue
                layer = min(self.encoder.layer_of(bit) for bit in unknown)
                item = best.get(key)
                if item is None or ratio < item["known_ratio"]:
                    best[key] = {
                        "type": self.types.types[key[0]].label,
                        "slot": slot,
                        "value": value,
                        "known_ratio": round(ratio, 2),
                        "scale_decades": round(self.encoder.layer_width(layer), 2),
                        "reference_min": min(values[key]),
                        "reference_max": max(values[key]),
                        "reference_values": len(values[key]),
                        "where": f"{line.source}:{line.number}",
                        "count": 0 if item is None else item["count"],
                    }
                best[key]["count"] += 1
        result = sorted(best.values(), key=lambda item: (item["known_ratio"], -item["scale_decades"]))
        LOGGER.event(
            "logs.insight.numeric",
            section=INTERSECTION,
            data={"slots": len(known), "checked_values": checked, "new_values": len(result),
                  "known_ratio": LOG_NUMERIC_KNOWN_RATIO},
        )
        LOGGER.event(
            "logs.insight.numeric.memory",
            section=CONJUNCTION,
            data={"slots": len(known), "bits_per_slot_mean": float(np.mean([len(v) for v in known.values()]))
                  if known else 0.0},
        )
        return result


# ------------------------------------------------------------------------------------------- инциденты
@dataclass(frozen=True)
class LogWindow:
    source: str
    rows: tuple[int, ...]
    stimuli: frozenset[int]

    def span(self, corpus: LogCorpus) -> str:
        return f"{self.source}:{corpus.lines[self.rows[0]].number}-{corpus.lines[self.rows[-1]].number}"


class IncidentFinder:
    """Инциденты целевых логов и похожие на них окна эталонных запусков (§6.5, §2.2.5).

    Запуск режется на окна по LOG_WINDOW_LINES строк. Эмбеддинг окна — детекция набора его строк S
    (a_ji = max по s ∈ S, §6.5). Инцидент — подряд идущие окна целевого запуска со строками-сигналами;
    похожие окна ищутся нечётким поиском по эмбеддингам окон эталона.

    В куче логов (everywhere) инциденты ищутся во всех запусках, а сигналом считается ещё и строка редкого
    типа: событие в начале файла новым не будет, но останется редким (вне статьи).
    """

    def __init__(self, corpus: LogCorpus, space: LogCodeSpace, types: EventTypes, *, everywhere: bool = False) -> None:
        self.corpus = corpus
        self.space = space
        self.types = types
        self._everywhere = bool(everywhere)
        rare_lines = max(LOG_RARE_TYPE_LINES_MIN, LOG_RARE_TYPE_SHARE * len(corpus.lines))
        self._rare = np.array(
            [everywhere and types.types[int(kind)].lines <= rare_lines for kind in types.type_of], dtype=bool
        )
        self._reference_sources = set(corpus.reference_sources)
        self.windows: list[LogWindow] = []
        for source in corpus.sources:
            rows = corpus.run_lines[source]
            for start in range(0, max(1, len(rows) - LOG_WINDOW_LINES + LOG_WINDOW_STRIDE), LOG_WINDOW_STRIDE):
                chunk = rows[start:start + LOG_WINDOW_LINES]
                if chunk:
                    self.windows.append(self._window(source, chunk))
        # Память окон эталона для поиска похожих: не больше LOG_WINDOW_MEMORY_MAX окон, взятых равномерно.
        reference = [i for i, w in enumerate(self.windows) if w.source in self._reference_sources]
        self.reference = reference[:: max(1, math.ceil(len(reference) / LOG_WINDOW_MEMORY_MAX))]
        space.size_window_code([self.windows[i].stimuli for i in self.reference])
        self.embeddings = [space.window_embedding(self.windows[i].stimuli) for i in self.reference]
        self.memory = CodeMemory(self.embeddings)
        LOGGER.event(
            "logs.windows",
            section=STIMULUS_DETECTION,
            data={
                "windows": len(self.windows),
                "reference_windows": len(reference),
                "memory_windows": len(self.reference),
                "lines": LOG_WINDOW_LINES,
                "stride": LOG_WINDOW_STRIDE,
                "bits_mean": float(np.mean([code.count() for code in self.embeddings])) if self.embeddings else 0.0,
            },
        )

    def _window(self, source: str, rows: Sequence[int]) -> LogWindow:
        return LogWindow(
            source=source,
            rows=tuple(int(row) for row in rows),
            stimuli=frozenset(int(self.corpus.stimulus_of[row]) for row in rows),
        )

    def _signal(self, stimulus: int) -> bool:
        """Строка-сигнал: новая для пространства (§7.1.13.3) либо проблемная и редкая.

        Проблемная строка, которая есть в большинстве эталонных запусков, — привычный шум, а не сигнал.
        В куче логов файлы разные, и правило «есть в большинстве запусков» не годится: там сигнал — новая
        строка либо строка редкого типа, а частая строка со словами ошибок — привычный шум своего файла.
        """
        if self.space.reaction(stimulus).status != STATUS_EXACT or bool(self._rare[stimulus]):
            return True
        if self._everywhere:
            return False
        seen = len(self.corpus.stimulus_runs[stimulus] & self._reference_sources)
        return bool(self.corpus.problem[stimulus]) and 2 * seen < len(self._reference_sources)

    def _before(self, source: str, row: int) -> list[str]:
        """Последние известные события перед строкой row её запуска (ближайшее — первым)."""
        rows = self.corpus.run_lines[source]
        position = rows.index(row)
        seen: list[int] = []
        for previous in reversed(rows[:position]):
            stimulus = int(self.corpus.stimulus_of[previous])
            if self._signal(stimulus) or stimulus in seen:
                continue
            seen.append(stimulus)
            if len(seen) == LOG_RELATION_DEPTH:
                break
        return [self.corpus.text(stimulus, limit=90) for stimulus in seen]

    def timeline(self, columns: int) -> list[dict]:
        """Лента каждого запуска: его строки делятся на равные отрезки, в отрезке считаются сигналы.

        Отрезок: номер первой строки, число строк, строк-сигналов и проблемных строк.
        """
        corpus = self.corpus
        signal = np.array([self._signal(stimulus) for stimulus in range(len(corpus.stimuli))], dtype=np.int64)
        problem = corpus.problem.astype(np.int64)
        result = []
        for source in corpus.sources:
            rows = np.asarray(corpus.run_lines[source], dtype=np.int64)
            if not rows.size:
                continue
            stimuli = corpus.stimulus_of[rows]
            edges = np.linspace(0, rows.size, min(int(columns), rows.size) + 1).astype(np.int64)
            starts = edges[:-1]
            result.append({
                "source": source,
                "target": source in set(corpus.target_sources),
                "lines": int(rows.size),
                "cells": [
                    {"line": corpus.lines[int(rows[start])].number, "lines": int(size), "signals": int(found),
                     "problems": int(bad)}
                    for start, size, found, bad in zip(
                        starts, np.diff(edges), np.add.reduceat(signal[stimuli], starts),
                        np.add.reduceat(problem[stimuli], starts),
                    )
                ],
            })
        LOGGER.event("logs.timeline", section=OUTSIDE_ARTICLE, data={"runs": len(result), "columns": int(columns)})
        return result

    def _similar(self, incident: LogWindow) -> list[tuple[float, int]]:
        """До трёх окон памяти эталона из других запусков, похожих на инцидент: нечёткий поиск (§2.2.5)."""
        similarity = self.memory.search(self.space.window_embedding(incident.stimuli))
        best: dict[str, tuple[float, int]] = {}
        for position, value in enumerate(similarity):
            window = self.windows[self.reference[position]]
            if window.source == incident.source or value < LOG_INCIDENT_LAMBDA:
                continue
            if window.source not in best or value > best[window.source][0]:
                best[window.source] = (float(value), self.reference[position])
        return sorted(best.values(), reverse=True)[:3]

    def incidents(self) -> list[dict]:
        corpus = self.corpus
        target = set(corpus.sources) if self._everywhere else set(corpus.target_sources)
        groups: list[list[int]] = []
        for index, window in enumerate(self.windows):
            if window.source not in target or not any(self._signal(s) for s in window.stimuli):
                continue
            if groups and groups[-1][-1] == index - 1 and self.windows[index - 1].source == window.source:
                groups[-1].append(index)
            else:
                groups.append([index])
        found = []
        for group in groups:
            rows = sorted({row for index in group for row in self.windows[index].rows})
            signals = Counter()
            first_signal = None
            for row in rows:
                stimulus = int(corpus.stimulus_of[row])
                if self._signal(stimulus):
                    signals[stimulus] += 1
                    if first_signal is None:
                        first_signal = row
            found.append((self._window(self.windows[group[0]].source, rows), rows, signals, first_signal))
        # Похожие окна ищутся для самых насыщенных сигналами инцидентов: детекция окна долгая.
        ranked = sorted(range(len(found)), key=lambda index: -sum(found[index][2].values()))
        searched = set(ranked[:LOG_INCIDENT_SEARCH_MAX])
        result = []
        for index, (incident, rows, signals, first_signal) in enumerate(found):
            matches = self._similar(incident) if index in searched else []
            problems = [s for s in signals if corpus.problem[s]]
            shown = list(dict.fromkeys(problems + [s for s, _ in signals.most_common()]))[:4]
            item = {
                "span": incident.span(corpus),
                "lines": len(rows),
                "signal_lines": int(sum(signals.values())),
                "signal_kinds": len(signals),
                "first_signal": f"{incident.source}:{corpus.lines[first_signal].number}",
                "signals": [
                    {"count": signals[s], "status": self.space.reaction(s).status, "problem": bool(corpus.problem[s]),
                     "text": corpus.text(s, limit=90)}
                    for s in shown
                ],
                "before": self._before(incident.source, first_signal),
                "similar": [{"span": self.windows[i].span(corpus), "similarity": round(v, 2)} for v, i in matches],
            }
            result.append(item)
            LOGGER.event(
                "logs.insight.incident",
                section=FUZZY_SEARCH,
                data={"span": item["span"], "signal_kinds": len(signals), "similar": len(matches),
                      "best_similarity": matches[0][0] if matches else 0.0},
            )
        return result

    def run_similarity(self) -> dict:
        """Сходство запусков: эмбеддинг запуска — детекция набора всех его строк (§6.5), косинус (§2.2.4.1)."""
        codes = [
            self.space.window_embedding(frozenset(int(self.corpus.stimulus_of[r]) for r in self.corpus.run_lines[s]))
            for s in self.corpus.sources
        ]
        matrix = CodeMemory(codes).pairwise()
        LOGGER.event("logs.insight.run_similarity", section=SIMILARITY_MEASURES, data={"runs": len(codes)})
        return {"runs": self.corpus.sources, "matrix": [[round(float(v), 2) for v in row] for row in matrix]}


# ------------------------------------------------------------------------------------------- связи
class TransitionMemory:
    """Память пар (A, B) кодов соседних событий запуска — кортежи списка (§2.3.2).

    «Что идёт после A» — нечёткий поиск маской (A, ⌀), «что предшествует B» — маской (⌀, B) (§2.3.1,
    §2.2.5): под запрос попадают и пары с событиями, похожими на ключ, поэтому связи похожих событий
    подкрепляют друг друга (§7.1.13.2). Повторы одного типа подряд схлопываются: важна смена событий.
    """

    def __init__(self, corpus: LogCorpus, types: EventTypes, *, depth: int) -> None:
        self.corpus = corpus
        self.types = types
        line_types = types.line_types()
        pairs: Counter[tuple[int, int, int]] = Counter()
        self.occurrences = np.zeros(len(types.types), dtype=np.int64)
        for source in corpus.sources:
            sequence: list[int] = []
            for row in corpus.run_lines[source]:
                kind = int(line_types[row])
                if not sequence or sequence[-1] != kind:
                    sequence.append(kind)
            for position, second in enumerate(sequence):
                self.occurrences[second] += 1
                for lag in range(1, depth + 1):
                    if position - lag >= 0:
                        pairs[(sequence[position - lag], second, lag)] += 1
        self.pairs = list(pairs)
        self.weights = np.array([pairs[pair] for pair in self.pairs], dtype=np.float64)
        self.first = np.array([a for a, _, _ in self.pairs], dtype=np.int64)
        self.second = np.array([b for _, b, _ in self.pairs], dtype=np.int64)
        self.lags = np.array([lag for _, _, lag in self.pairs], dtype=np.int64)
        self.type_memory = CodeMemory([types.codes[t.centre] for t in types.types])
        LOGGER.event(
            "logs.relations.memory",
            section=LISTS,
            data={"pairs": len(self.pairs), "depth": depth, "observations": float(self.weights.sum())},
        )

    def _like(self, kind: int) -> np.ndarray:
        """Типы, похожие на ключ запроса: сам тип и типы с близким кодом центра (§2.2.5)."""
        similar = self.type_memory.search(self.types.codes[self.types.types[kind].centre]) >= LOG_RELATION_LAMBDA
        similar[kind] = True
        return similar

    def rules(self) -> list[dict]:
        """«После A обычно B»: поддержка, уверенность P(B|A) и подъём P(B|A) / P(B)."""
        adjacent = self.lags == 1
        total = self.weights[adjacent].sum()
        base = np.bincount(self.second[adjacent], weights=self.weights[adjacent], minlength=len(self.types.types))
        result = []
        for event_type in self.types.types:
            mask = adjacent & self._like(event_type.index)[self.first]
            support = self.weights[mask].sum()
            if support < LOG_RELATION_MIN_SUPPORT:
                continue
            followers = np.bincount(self.second[mask], weights=self.weights[mask], minlength=len(self.types.types))
            for kind in np.argsort(-followers)[:2]:
                kind = int(kind)
                if kind == event_type.index or followers[kind] <= 0:
                    continue
                confidence = followers[kind] / support
                lift = confidence / (base[kind] / total)
                if confidence >= LOG_RELATION_MIN_CONFIDENCE and lift >= LOG_RELATION_MIN_LIFT:
                    result.append({"after": event_type.label, "then": self.types.types[kind].label,
                                   "support": int(support), "confidence": round(float(confidence), 2),
                                   "lift": round(float(lift), 1)})
        result.sort(key=lambda item: -item["support"] * item["confidence"])
        # Похожие типы отвечают на один и тот же запрос: одинаковые по подписям правила оставляются один раз.
        unique: dict[tuple[str, str], dict] = {}
        for item in result:
            unique.setdefault((item["after"], item["then"]), item)
        result = list(unique.values())
        LOGGER.event("logs.insight.rules", section=ASSOCIATION_CODING,
                     data={"rules": len(result), "lambda": LOG_RELATION_LAMBDA})
        return result

    def precursors(self, kinds: Sequence[int]) -> list[dict]:
        """Что чаще обычного встречается перед событиями kinds: запрос маской (⌀, B) с глубиной памяти.

        Берутся события, которые случались не меньше LOG_RELATION_MIN_SUPPORT раз: по единичному случаю
        связь не оценить.
        """
        total = self.weights.sum()
        base = np.bincount(self.first, weights=self.weights, minlength=len(self.types.types))
        result = []
        for kind in kinds:
            if self.occurrences[kind] < LOG_RELATION_MIN_SUPPORT:
                continue
            mask = self._like(kind)[self.second] & (self.first != kind)
            support = float(self.occurrences[kind])
            before = np.bincount(self.first[mask], weights=self.weights[mask], minlength=len(self.types.types))
            ranked = sorted(
                ((before[k] / support) / (base[k] / total), int(k), before[k])
                for k in np.flatnonzero(before >= LOG_RELATION_MIN_SUPPORT)
            )[::-1]
            result.append({
                "event": self.types.types[kind].label,
                "occurrences": int(self.occurrences[kind]),
                "precursors": [{"type": self.types.types[k].label, "lift": round(float(lift), 1), "times": int(n)}
                               for lift, k, n in ranked[:3]],
            })
        result.sort(key=lambda item: -item["occurrences"])
        LOGGER.event("logs.insight.precursors", section=LISTS, data={"events": len(result)})
        return result


# ------------------------------------------------------------------------------------------- отчёт
class LogInsightsReport:
    """Отчёт: Markdown для чтения, JSON для дальнейшей обработки (вне статьи)."""

    def __init__(self, corpus: LogCorpus, space: LogCodeSpace, types: EventTypes, insights: dict) -> None:
        self.corpus = corpus
        self.space = space
        self.types = types
        self.insights = insights

    @staticmethod
    def _table(header: Sequence[object], rows: Sequence[Sequence[object]]) -> list[str]:
        if not rows:
            return ["_нет_", ""]
        lines = ["| " + " | ".join(str(cell) for cell in header) + " |", "|" + "---|" * len(header)]
        for row in rows:
            lines.append("| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |")
        return lines + [""]

    def markdown(self) -> str:
        ins, corpus, space, top = self.insights, self.corpus, self.space, LOG_INSIGHTS_TOP
        quality = ins["embedding_quality"]
        novelty = Counter(item["status"] for item in ins["new_lines"])
        lines = [
            "# Инсайты по логам (DAMP)",
            "",
            f"Эталон: {', '.join(f'`{s}`' for s in corpus.reference_sources)}.  ",
            f"Цель: {', '.join(f'`{s}`' for s in corpus.target_sources) or '—'}.",
            "",
            f"Строк: **{len(corpus.lines)}**, различных строк (стимулов): **{len(corpus.stimuli)}**, "
            f"точек пространства: **{len(space.points)}** на сетке {space.layout.width}×{space.layout.height}, "
            f"детекторов: **{len(space.detectors)}**, семейств событий: **{len(self.types.families)}**, "
            f"типов событий: **{len(self.types.types)}**.",
            "",
            f"Карты: `{LOG_INSIGHTS_MAP_PATH}` — псевдоцвета кодов (§5.14), `{LOG_INSIGHTS_ENERGY_MAP_PATH}` — "
            f"с яркостью по энергии (рис. 18), `{LOG_INSIGHTS_TYPES_MAP_PATH}` — семейства событий.",
            "",
            f"Проверка эмбеддингов (§6.2): корреляция сходства эмбеддингов со сходством первичных кодов "
            f"**{quality['correlation']}**; среднее сходство эмбеддингов у похожих строк "
            f"{quality['embedding_similarity_of_similar']}, у остальных {quality['embedding_similarity_of_others']}; "
            f"строк без активных детекторов: {quality['stimuli_without_detectors']}.",
            "",
            "## 1. Типы и семейства событий",
            "",
            "### 1.1 Семейства (нижние слои иерархии детекторов: области и гиперкластеры, §7.1.10)",
            "",
        ]
        families = sorted(self.types.families, key=lambda f: -f.lines)
        lines += self._table(
            ["#", "семейство (общие слова)", "видов строк", "типов", "строк"],
            [[f.index, f.label, len(f.members), f.types, f.lines] for f in families[:top]],
        )
        lines += ["### 1.2 Типы (верхние слои иерархии: кластеры и их части)", ""]
        ranked = sorted(self.types.types, key=lambda t: -t.lines)
        lines += self._table(
            ["#", "тип (общие слова)", "семейство", "видов строк", "строк", "запусков", "пример"],
            [[t.index, t.label, t.family, len(t.members), t.lines, len(t.runs), corpus.text(t.centre, limit=70)]
             for t in ranked[:top]],
        )
        lines += ["### 1.3 Слои иерархии детекторов (§7.1.10, §8.2)", ""]
        lines += self._table(
            ["λ_d", "роль", "детекторов", "средний радиус", "точек в поле", "активных на строку"],
            [[row["lambda"], row["role"], row["detectors"], row["radius_mean"], row["points_mean"],
              row["active_per_line"]] for row in ins["structure"]],
        )
        lines += [
            "## 2. Новизна и аномалии",
            "",
            "### 2.1 Новые строки целевых логов (§7.1.13.3)",
            "",
            f"Неизвестных: **{novelty[STATUS_UNKNOWN]}**, известных по частям: **{novelty[STATUS_PARTIAL]}** "
            f"(⚠ — в строке есть проблемное слово).",
            "",
        ]
        lines += self._table(
            ["статус", "знакомость", "детекторов", "раз", "где", "строка", "ближайшая известная"],
            [[n["status"] + (" ⚠" if n["problem"] else ""), n["familiarity"], n["active_detectors"], n["count"],
              n["where"], n["text"], n["nearest_known"] or "—"] for n in ins["new_lines"][:top]],
        )
        if len(ins["new_lines"]) > top:
            lines += [f"_…и ещё {len(ins['new_lines']) - top}; полный список — в JSON._", ""]
        lines += ["### 2.2 Новые значения чисел (§4.2.1, §2.2.2)", ""]
        lines += self._table(
            ["тип", "слот", "значение", "эталон: от", "до", "значений", "известных бит", "масштаб, порядков", "где"],
            [[n["type"], n["slot"], f"{n['value']:g}", f"{n['reference_min']:g}", f"{n['reference_max']:g}",
              n["reference_values"], n["known_ratio"], n["scale_decades"], n["where"]]
             for n in ins["new_numbers"][:top]],
        )
        lines += ["### 2.3 Строки эталона вне структуры: энергия всех их точек ниже порога (§5.7, §5.14)", ""]
        lines += self._table(
            ["энергия", "раз", "запусков", "где", "строка"],
            [[("⚠ " if n["problem"] else "") + str(n["energy"]), n["count"], n["runs"], n["where"], n["text"]]
             for n in ins["unorganised"][:top]],
        )
        if len(ins["unorganised"]) > top:
            lines += [f"_…и ещё {len(ins['unorganised']) - top}._", ""]
        lines += ["## 3. Инциденты и похожие окна эталона (§6.5, §2.2.5)", ""]
        for incident in ins["incidents"]:
            lines += [
                f"### {incident['span']}",
                "",
                f"Строк: {incident['lines']}, из них сигналов: {incident['signal_lines']} "
                f"({incident['signal_kinds']} видов); первый сигнал — `{incident['first_signal']}`.",
                "",
                "Сигналы:",
                "",
            ]
            lines += [f"- {s['status']}{' ⚠' if s['problem'] else ''} ×{s['count']}: `{s['text']}`"
                      for s in incident["signals"]]
            lines += ["", "Перед первым сигналом (ближайшее событие — первым):", ""]
            lines += [f"- `{text}`" for text in incident["before"]] or ["- —"]
            lines += ["", "Похожие окна эталона:", ""]
            lines += [f"- `{s['span']}` — сходство {s['similarity']}" for s in incident["similar"]] or ["- нет"]
            lines += [""]
        if not ins["incidents"]:
            lines += ["_нет_", ""]
        runs = ins["run_similarity"]
        lines += ["### Сходство запусков", ""]
        lines += self._table(
            ["", *range(len(runs["runs"]))],
            [[f"{i}: {name}", *row] for i, (name, row) in enumerate(zip(runs["runs"], runs["matrix"]))],
        )
        lines += ["## 4. Связи событий (§2.3.1, §2.3.2)", "", "### 4.1 «После A обычно идёт B»", ""]
        lines += self._table(
            ["после", "идёт", "поддержка", "уверенность", "подъём"],
            [[r["after"], r["then"], r["support"], r["confidence"], r["lift"]] for r in ins["rules"][:top]],
        )
        lines += [f"### 4.2 Что предшествует повторяющимся проблемным событиям (до {LOG_RELATION_DEPTH} событий назад)", ""]
        lines += self._table(
            ["событие", "раз", "предвестники (подъём, раз)"],
            [[p["event"], p["occurrences"],
              "; ".join(f"{q['type']} (×{q['lift']}, {q['times']})" for q in p["precursors"]) or "—"]
             for p in ins["precursors"][:top]],
        )
        return "\n".join(lines)

    def save(self, markdown_path: str, json_path: str) -> None:
        with open(markdown_path, "w", encoding="utf-8") as handle:
            handle.write(self.markdown() + "\n")
        payload = {
            "reference": self.corpus.reference_sources,
            "target": self.corpus.target_sources,
            "families": [
                {"family": f.index, "label": f.label, "lines": f.lines, "types": f.types,
                 "members": [self.corpus.text(m) for m in f.members]}
                for f in self.types.families
            ],
            "types": [
                {"type": t.index, "label": t.label, "family": t.family, "lines": t.lines, "runs": sorted(t.runs),
                 "problem": t.problem, "members": [self.corpus.text(m) for m in t.members]}
                for t in self.types.types
            ],
            **self.insights,
        }
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        LOGGER.event("logs.report.save", section=OUTSIDE_ARTICLE, data={"markdown": markdown_path, "json": json_path})


# ------------------------------------------------------------------------------------------- эксперимент
class LogInsightsExperiment:
    """Полный конвейер; параметры — константы LOG_* из main.py.

    Вход — либо эталонные и целевые логи, либо куча логов pile: файлы, каталоги и шаблоны без деления
    на эталон и цель (LogPile).
    """

    def __init__(
        self,
        reference: Sequence[str] = LOG_INSIGHTS_REFERENCE,
        target: Sequence[str] = LOG_INSIGHTS_TARGET,
        *,
        pile: Sequence[str] | None = None,
    ) -> None:
        self._reference = tuple(reference)
        self._target = tuple(target)
        self._pile = None if pile is None else tuple(pile)
        self._split: dict[str, int] = {}
        self._reader = LogLineReader(
            word_min=LOG_WORD_MIN,
            word_max=LOG_WORD_MAX,
            line_words_max=LOG_LINE_PARSE_WORDS_MAX,
            slot_words=LOG_NUMBER_SLOT_WORDS,
            line_chars_max=LOG_LINE_CHARS_MAX,
            lines_max=LOG_LINES_MAX,
            binary_probe_bytes=LOG_BINARY_PROBE_BYTES,
        )

    def read_lines(self) -> tuple[list[LogLine], list[LogLine]]:
        """Строки эталонных и целевых логов после предобработки.

        Фоновые слова и переменные места находятся по эталону и убираются из строк обоих наборов; после
        этого строка обрезается до LOG_LINE_WORDS_MAX слов. Обрезка идёт последней: иначе служебные поля
        в конце длинной строки не попадали бы в подсчёт фоновых слов.
        """
        if self._pile is not None:
            pile = LogPile(
                self._reader.read_sources(self._pile),
                target_share=LOG_PILE_TARGET_SHARE,
                min_lines=LOG_PILE_MIN_LINES,
                mark=LOG_PILE_TAIL_MARK,
            )
            reference, target, self._split = pile.reference, pile.target, pile.split
        else:
            reference = self._reader.read_sources(self._reference)
            target = self._reader.read_sources(self._target)
        if not reference:
            raise ValueError("во входных файлах нет ни одной строки со словами")
        background = BackgroundWords(reference, share=LOG_BACKGROUND_SHARE, min_lines=LOG_BACKGROUND_MIN_LINES)
        reference, target = background.clean(reference), background.clean(target)
        variables = VariableSlots(
            (line.units for line in reference),
            min_values=LOG_VARIABLE_MIN_VALUES,
            min_context=LOG_VARIABLE_MIN_CONTEXT,
        )
        return (
            limit_words(variables.clean(reference), LOG_LINE_WORDS_MAX),
            limit_words(variables.clean(target), LOG_LINE_WORDS_MAX),
        )

    def build_space(self) -> tuple[LogCorpus, LogCodeSpace]:
        corpus = LogCorpus(*self.read_lines())
        reference_stimuli = [corpus.stimuli[int(i)] for i in np.flatnonzero(corpus.reference_count > 0)]
        dictionary = FragmentDictionary(reference_stimuli, fragment_words_max=LOG_FRAGMENT_WORDS_MAX)
        encoder = LogLineEncoder(
            TokenAlphabet(code_length=LOG_CODE_BITS, token_bits=LOG_TOKEN_BITS, seed=LOG_SEED),
            LineFragmenter(dictionary, fragments_max=LOG_FRAGMENTS_MAX),
            fragment_saturation=LOG_FRAGMENT_SATURATION,
        )
        space = LogCodeSpace(corpus, encoder)
        space.lay_out()
        space.build_detectors()
        return corpus, space

    def _input(self, space: LogCodeSpace) -> dict:
        """Что подано на вход и что из этого взято: файлы, пределы, деление кучи, предел пространства."""
        return {
            "mode": "pile" if self._pile is not None else "reference_target",
            "files": list(self._reader.files),
            "lines_max": LOG_LINES_MAX,
            "line_chars_max": LOG_LINE_CHARS_MAX,
            "pile_split": dict(self._split),
            "pile_target_share": LOG_PILE_TARGET_SHARE,
            "space": dict(space.budget),
        }

    def run(self, *, save: bool = True, out_dir: str | None = None) -> dict:
        """Весь анализ; out_dir — каталог результатов (по умолчанию пути LOG_INSIGHTS_* как есть)."""
        corpus, space = self.build_space()
        types = EventTypes(corpus, space)
        novelty = NoveltyFinder(corpus, space, types)
        numbers = NumericNovelty(corpus, types, NumericValueEncoder(
            unit=LOG_NUMERIC_UNIT,
            decades=LOG_NUMERIC_DECADES,
            layers=LOG_NUMERIC_LAYERS,
            widest=LOG_NUMERIC_WIDEST,
            narrowest=LOG_NUMERIC_NARROWEST,
            overlap=LOG_NUMERIC_OVERLAP,
            code_length=LOG_NUMERIC_CODE_BITS,
            seed=LOG_SEED,
        ))
        incidents = IncidentFinder(corpus, space, types, everywhere=self._pile is not None)
        relations = TransitionMemory(corpus, types, depth=LOG_RELATION_DEPTH)
        insights = {
            "input": self._input(space),
            "timeline": incidents.timeline(LOG_DASHBOARD_TIMELINE_COLUMNS),
            "summary": {
                "lines": len(corpus.lines),
                "stimuli": len(corpus.stimuli),
                "points": len(space.points),
                "grid": [space.layout.height, space.layout.width],
                "detectors": len(space.detectors),
                "families": len(types.families),
                "types": len(types.types),
            },
            "embedding_quality": types.quality(),
            "structure": DetectorStructure(corpus, space).rows,
            "new_lines": novelty.new_lines(),
            "new_numbers": numbers.find(),
            "unorganised": novelty.unorganised(),
            "incidents": incidents.incidents(),
            "run_similarity": incidents.run_similarity(),
            "rules": relations.rules(),
            "precursors": relations.precursors([t.index for t in types.types if t.problem]),
        }
        if save:
            if out_dir is not None:
                os.makedirs(out_dir, exist_ok=True)

            def path(name: str) -> str:
                return name if out_dir is None else os.path.join(out_dir, os.path.basename(name))

            space.save_maps(path(LOG_INSIGHTS_MAP_PATH), path(LOG_INSIGHTS_ENERGY_MAP_PATH))
            types.save_map(path(LOG_INSIGHTS_TYPES_MAP_PATH))
            LogInsightsReport(corpus, space, types, insights).save(
                path(LOG_INSIGHTS_REPORT_PATH), path(LOG_INSIGHTS_JSON_PATH)
            )
            LogInsightsDashboard(corpus, space, types, insights).save(path(LOG_INSIGHTS_DASHBOARD_PATH))
        LOGGER.event(
            "logs.insights.done",
            section=MORPHOLOGICAL_EMBEDDINGS,
            data={
                "families": len(types.families),
                "types": len(types.types),
                "new_lines": len(insights["new_lines"]),
                "new_numbers": len(insights["new_numbers"]),
                "incidents": len(insights["incidents"]),
                "rules": len(insights["rules"]),
            },
        )
        return insights


def main(arguments: Sequence[str] | None = None) -> None:
    """Без путей — логи из констант main.py; с путями — куча логов: файлы, каталоги, шаблоны."""
    parser = argparse.ArgumentParser(description="Инсайты из логов на DAMP: отчёт и дашборд.")
    parser.add_argument("paths", nargs="*", help="файлы, каталоги или шаблоны логов в любом формате")
    parser.add_argument("--out", default=None, help="каталог результатов")
    options = parser.parse_args(arguments)
    configure_logging()
    if options.paths:
        LogInsightsExperiment(pile=options.paths).run(out_dir=options.out or LOG_INSIGHTS_OUT_DIR)
    else:
        LogInsightsExperiment().run(out_dir=options.out)


if __name__ == "__main__":
    main()
