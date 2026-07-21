from __future__ import annotations

import math
import concurrent.futures
import json
import os
import heapq
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from damp.article_refs import CODE_SPACE_ACTIVATION, SIMILARITY_DEFINITION, SIMILARITY_MEASURES, STIMULUS_DETECTION
from damp.activation import ActivationEngine, ActivationResult
from damp.encoding.bitarray import BitArray
from damp.encoding.damp_encoder import Encoder
from damp.encoding.stimulus_embedding import StimulusEmbeddingBuilder
from damp.MnistSobelAngleMap import MnistSobelAngleMap
from damp.logging import LOGGER


@dataclass(frozen=True)
class DecoderConfig:
    similarity: str
    temperature: float
    min_confidence: float
    expected_classes: Sequence[int]
    knn_top_k: int = 50
    top_k_per_class: int = 0
    min_similarity: float = 0.0
    similarity_batch_size: int = 128


@dataclass(frozen=True)
class ScaledDecoderParameters:
    knn_top_k: int
    temperature: float


class DecoderParameterScaleResolver:
    """Сохраняет относительный k-NN охват при росте базы эмбеддингов."""

    def __init__(
        self,
        *,
        reference_embeddings: int,
        base_top_k: int,
        base_temperature: float,
        max_temperature: float,
    ) -> None:
        if reference_embeddings <= 0 or base_top_k <= 0:
            raise ValueError("decoder scale sizes must be positive")
        if base_temperature <= 0.0 or max_temperature < base_temperature:
            raise ValueError("invalid decoder temperature range")
        self._reference_embeddings = int(reference_embeddings)
        self._base_top_k = int(base_top_k)
        self._base_temperature = float(base_temperature)
        self._max_temperature = float(max_temperature)

    def resolve(self, embeddings: int) -> ScaledDecoderParameters:
        if embeddings <= 0:
            raise ValueError("embeddings must be positive")
        scale = embeddings / self._reference_embeddings
        resolved = ScaledDecoderParameters(
            knn_top_k=min(embeddings, self._base_top_k),
            temperature=self._base_temperature,
        )
        LOGGER.event(
            "decoder.parameters.scale.resolve",
            section=SIMILARITY_MEASURES,
            data={
                "embeddings": embeddings,
                "reference_embeddings": self._reference_embeddings,
                "scale_factor": scale,
                "knn_top_k": resolved.knn_top_k,
                "temperature": resolved.temperature,
            },
        )
        return resolved


@dataclass(frozen=True)
class PredictionResult:
    predicted_class: int
    probabilities: list[float]
    confidence: float


@dataclass(frozen=True)
class LabeledEmbedding:
    label: int
    embedding: BitArray


@dataclass(frozen=True)
class PreparedEmbeddingNeighbors:
    neighbors: tuple[tuple[float, int], ...]
    strategy: str


class EvaluationEmbeddingCache:
    """Хранит эмбеддинги последней выборки без изменения их битового представления."""

    def __init__(self, *, path: str, version: int) -> None:
        self._path = path
        self._version = int(version)

    def load(
        self,
        *,
        key: str,
        labels: Sequence[int],
        code_length: int,
    ) -> list[BitArray] | None:
        try:
            with open(self._path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if (
                payload.get("version") != self._version
                or payload.get("key") != key
                or payload.get("labels") != [int(label) for label in labels]
                or payload.get("code_length") != int(code_length)
            ):
                raise ValueError("cache metadata mismatch")
            raw_embeddings = payload.get("embeddings")
            if not isinstance(raw_embeddings, list):
                raise ValueError("cache embeddings missing")
            embeddings = [self._from01(bits, code_length) for bits in raw_embeddings]
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            LOGGER.event(
                "decoder.evaluation.cache.miss",
                section=CODE_SPACE_ACTIVATION,
                data={"path": self._path, "key": key},
            )
            return None
        LOGGER.event(
            "decoder.evaluation.cache.hit",
            section=CODE_SPACE_ACTIVATION,
            data={"path": self._path, "key": key, "embeddings": len(embeddings)},
        )
        return embeddings

    def save(
        self,
        *,
        key: str,
        labels: Sequence[int],
        embeddings: Sequence[BitArray],
    ) -> None:
        code_length = len(embeddings[0]) if embeddings else 0
        payload = {
            "version": self._version,
            "key": key,
            "labels": [int(label) for label in labels],
            "code_length": code_length,
            "embeddings": [embedding.to01() for embedding in embeddings],
        }
        directory = os.path.dirname(os.path.abspath(self._path))
        os.makedirs(directory, exist_ok=True)
        temporary_path = f"{self._path}.tmp"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
        os.replace(temporary_path, self._path)
        LOGGER.event(
            "decoder.evaluation.cache.save",
            section=CODE_SPACE_ACTIVATION,
            data={"path": self._path, "key": key, "embeddings": len(embeddings)},
        )

    @staticmethod
    def _from01(bits: object, code_length: int) -> BitArray:
        if not isinstance(bits, str) or len(bits) != code_length:
            raise ValueError("invalid cached embedding")
        embedding = BitArray(code_length)
        for index, value in enumerate(bits):
            if value == "1":
                embedding.set(index, 1)
            elif value != "0":
                raise ValueError("invalid cached bit")
        return embedding


class DecoderEncodingWorker:
    """Изолированный OOP-worker Sobel + Encoder для пакетной оценки."""

    _encoder: Encoder | None = None
    _extractor: MnistSobelAngleMap | None = None
    _embedding_builder: StimulusEmbeddingBuilder | None = None

    @classmethod
    def configure(
        cls,
        encoder: Encoder,
        extractor: MnistSobelAngleMap,
        embedding_builder: StimulusEmbeddingBuilder,
    ) -> None:
        LOGGER.configure_worker_silent()
        cls._encoder = encoder
        cls._extractor = extractor
        cls._embedding_builder = embedding_builder

    @classmethod
    def encode(
        cls, task: tuple[Sequence[tuple[float, int, int]], int, bool]
    ) -> tuple[int, BitArray, list[BitArray] | None]:
        measurements, _label, return_codes = task
        if cls._encoder is None or cls._embedding_builder is None:
            raise RuntimeError("decoder encoding worker is not configured")
        if not measurements:
            raise RuntimeError("у изображения нет активных измерений")
        codes = [code for _, code in cls._encoder.encode_many(measurements)]
        return (
            len(codes),
            cls._embedding_builder.build(codes),
            codes if return_codes else None,
        )


class _EvaluationDiagnostics:
    """Собирает диагностические представления классификации (§6.5)."""

    def __init__(
        self,
        expected_classes: Sequence[int],
        *,
        confidence_bins: int,
        error_examples: int,
    ) -> None:
        if confidence_bins <= 0 or error_examples <= 0:
            raise ValueError("evaluation diagnostic sizes must be positive")
        self._classes = tuple(int(value) for value in expected_classes)
        self._class_index = {
            class_label: index for index, class_label in enumerate(self._classes)
        }
        self._confusion = np.zeros(
            (len(self._classes), len(self._classes)), dtype=np.int64
        )
        self._error_limit = int(error_examples)
        self._confidences: list[float] = []
        self._correct_confidences: list[float] = []
        self._wrong_confidences: list[float] = []
        self._errors: list[tuple[np.ndarray, int, int, float]] = []
        self._correct = 0

    def update(
        self,
        *,
        sample_index: int,
        image: np.ndarray,
        expected: int,
        prediction: PredictionResult,
    ) -> None:
        predicted = int(prediction.predicted_class)
        is_correct = expected == predicted
        self._correct += int(is_correct)
        self._confidences.append(float(prediction.confidence))
        target = self._correct_confidences if is_correct else self._wrong_confidences
        target.append(float(prediction.confidence))
        if expected in self._class_index and predicted in self._class_index:
            self._confusion[
                self._class_index[expected], self._class_index[predicted]
            ] += 1
        if not is_correct and len(self._errors) < self._error_limit:
            self._errors.append(
                (np.asarray(image, dtype=np.float32), expected, predicted, prediction.confidence)
            )
        probabilities = sorted(prediction.probabilities, reverse=True)
        margin = (
            probabilities[0] - probabilities[1]
            if len(probabilities) > 1
            else prediction.confidence
        )
        samples = sample_index + 1
        running_accuracy = self._correct / samples
        log_progress = LOGGER.should_log("decoder.evaluation.sample")
        visuals = (
            self._metric_visuals(
                samples=samples,
                running_accuracy=running_accuracy,
                confidence=prediction.confidence,
                margin=margin,
            )
            if log_progress
            else None
        )
        LOGGER.event(
            "decoder.evaluation.sample",
            section=STIMULUS_DETECTION,
            data={
                "sample": samples,
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "running_accuracy": running_accuracy,
                "confidence": prediction.confidence,
                "margin": margin,
            },
            visuals=visuals,
            force=log_progress,
        )

    def log_summary(self) -> None:
        samples = len(self._confidences)
        if samples == 0:
            return
        final_accuracy = self._correct / samples
        row_totals = self._confusion.sum(axis=1)
        per_class_accuracy = np.divide(
            np.diag(self._confusion),
            row_totals,
            out=np.zeros(len(self._classes), dtype=np.float64),
            where=row_totals > 0,
        )
        confusion_positions = [
            (row, column)
            for row in range(len(self._classes))
            for column in range(len(self._classes))
        ]
        visuals = [
            *self._metric_visuals(
                samples=samples,
                running_accuracy=final_accuracy,
                confidence=self._confidences[-1],
                margin=0.0,
            ),
            LOGGER.visual_scalar_map(
                "evaluate/confusion/matrix",
                self._confusion.astype(np.float64).ravel().tolist(),
                confusion_positions,
                height=len(self._classes),
                width=len(self._classes),
                static=True,
            ),
            LOGGER.visual_points2d(
                "evaluate/confusion/axes",
                [
                    *[(-0.5, row + 0.5) for row in range(len(self._classes))],
                    *[(column + 0.5, -0.5) for column in range(len(self._classes))],
                ],
                labels=[
                    *[f"истина {value}" for value in self._classes],
                    *[f"прогноз {value}" for value in self._classes],
                ],
                show_labels=True,
                radii=0.1,
                static=True,
            ),
        ]
        error_montage = self._error_montage()
        if error_montage is not None:
            montage_image, label_positions, labels = error_montage
            visuals.append(
                LOGGER.visual_image(
                    "evaluate/errors/images",
                    montage_image,
                    static=True,
                )
            )
            visuals.append(
                LOGGER.visual_points2d(
                    "evaluate/errors/labels",
                    label_positions,
                    colors=[(255, 64, 64, 255) for _ in labels],
                    radii=0.25,
                    labels=labels,
                    show_labels=True,
                    static=True,
                )
            )
        LOGGER.event(
            "decoder.evaluation.summary",
            section=STIMULUS_DETECTION,
            data={
                "samples": samples,
                "accuracy": final_accuracy,
                "errors": samples - self._correct,
                "mean_confidence": float(np.mean(self._confidences)),
                "per_class_accuracy": {
                    str(label): float(per_class_accuracy[index])
                    for index, label in enumerate(self._classes)
                },
            },
            visuals=visuals,
        )

    def _metric_visuals(
        self,
        *,
        samples: int,
        running_accuracy: float,
        confidence: float,
        margin: float,
    ) -> list[object]:
        row_totals = self._confusion.sum(axis=1)
        class_accuracy = np.divide(
            np.diag(self._confusion),
            row_totals,
            out=np.zeros(len(self._classes), dtype=np.float64),
            where=row_totals > 0,
        )
        class_metrics = {
            f"class_{label}": float(class_accuracy[index])
            for index, label in enumerate(self._classes)
            if row_totals[index] > 0
        }
        confidence_metrics = {
            "current": confidence,
            "mean_all": float(np.mean(self._confidences)),
            "mean_correct": float(np.mean(self._correct_confidences))
            if self._correct_confidences
            else 0.0,
            "mean_wrong": float(np.mean(self._wrong_confidences))
            if self._wrong_confidences
            else 0.0,
            "probability_margin": margin,
        }
        return [
            *LOGGER.visual_scalars(
                "evaluate/metrics",
                {
                    "running_accuracy": running_accuracy,
                    "error_rate": 1.0 - running_accuracy,
                },
                step=samples,
                timeline="sample",
            ),
            *LOGGER.visual_scalars(
                "evaluate/class_accuracy",
                class_metrics,
                step=samples,
                timeline="sample",
            ),
            *LOGGER.visual_scalars(
                "evaluate/confidence",
                confidence_metrics,
                step=samples,
                timeline="sample",
            ),
        ]

    def _error_montage(
        self,
    ) -> tuple[np.ndarray, list[tuple[float, float]], list[str]] | None:
        if not self._errors:
            return None
        columns = int(math.ceil(math.sqrt(len(self._errors))))
        rows = int(math.ceil(len(self._errors) / columns))
        height, width = self._errors[0][0].shape[:2]
        montage = np.zeros((rows * height, columns * width), dtype=np.float32)
        label_positions: list[tuple[float, float]] = []
        labels: list[str] = []
        for index, (image, expected, predicted, confidence) in enumerate(self._errors):
            row, column = divmod(index, columns)
            normalized = np.asarray(image, dtype=np.float32)
            if normalized.max() > 1.0:
                normalized = normalized / 255.0
            montage[
                row * height : (row + 1) * height,
                column * width : (column + 1) * width,
            ] = normalized
            label_positions.append((column * width + 1.0, row * height + 1.0))
            labels.append(f"{expected}→{predicted} ({confidence:.2f})")
        return (
            np.asarray(np.clip(montage * 255.0, 0.0, 255.0), dtype=np.uint8),
            label_positions,
            labels,
        )


class EmbeddingDatabase:
    def __init__(self, embeddings: Sequence[LabeledEmbedding], config: DecoderConfig) -> None:
        if not embeddings:
            raise ValueError("embeddings must be non-empty")
        if config.temperature <= 0:
            raise ValueError("temperature must be positive")
        if config.knn_top_k <= 0:
            raise ValueError("knn_top_k must be positive")
        if config.top_k_per_class < 0:
            raise ValueError("top_k_per_class must be non-negative")
        if config.min_similarity < 0:
            raise ValueError("min_similarity must be non-negative")
        if config.similarity_batch_size <= 0:
            raise ValueError("similarity_batch_size must be positive")
        if config.similarity.lower() not in ("cosine", "cos", "jaccard", "j2"):
            raise ValueError("similarity must be cosine or binary Jaccard/J2")

        self._config = config
        self._embeddings = list(embeddings)

        expected = list(config.expected_classes)
        if not expected:
            expected = sorted({item.label for item in embeddings})
        self._expected_classes = expected

        # Precompute counts once (BitArray.count() can be non-trivial).
        self._items: list[tuple[int, BitArray, int]] = []
        self._by_class: dict[int, list[tuple[BitArray, int]]] = defaultdict(list)
        for item in self._embeddings:
            if item.label not in self._expected_classes:
                continue
            ones = item.embedding.count()
            self._items.append((item.label, item.embedding, ones))
            self._by_class[item.label].append((item.embedding, ones))

        code_lengths = {len(item[1]) for item in self._items}
        if len(code_lengths) != 1:
            raise ValueError("all database embeddings must have equal length")
        self._code_length = next(iter(code_lengths))
        self._matrix = np.empty(
            (len(self._items), self._code_length), dtype=np.float32
        )
        for index, (_, embedding, _) in enumerate(self._items):
            self._matrix[index] = np.frombuffer(
                embedding._bits,
                dtype=np.uint8,
                count=self._code_length,
            )
        self._matrix_t = np.ascontiguousarray(self._matrix.T)
        self._ones_array = np.asarray(
            [item[2] for item in self._items], dtype=np.float32
        )
        self._labels_array = np.asarray(
            [item[0] for item in self._items], dtype=np.int32
        )

        LOGGER.event(
            "decoder.base.init",
            section=SIMILARITY_MEASURES,
            data={
                "embeddings": len(self._embeddings),
                "classes": len(self._expected_classes),
                "similarity": self._config.similarity,
                "temperature": self._config.temperature,
                "knn_top_k": self._config.knn_top_k,
                "top_k_per_class": self._config.top_k_per_class,
                "min_similarity": self._config.min_similarity,
                "similarity_batch_size": self._config.similarity_batch_size,
            },
        )

    @property
    def expected_classes(self) -> Sequence[int]:
        return tuple(self._expected_classes)

    @property
    def config(self) -> DecoderConfig:
        return self._config

    def predict(
        self,
        embedding: BitArray,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_k_per_class: int | None = None,
    ) -> PredictionResult:
        if len(embedding) == 0:
            raise ValueError("embedding must be non-empty")

        temp = temperature if temperature is not None else self._config.temperature
        if temp <= 0:
            raise ValueError("temperature must be positive")

        k_total = int(top_k) if top_k is not None else int(self._config.knn_top_k)
        if k_total <= 0:
            raise ValueError("top_k must be positive")

        k_per_class = (
            int(top_k_per_class)
            if top_k_per_class is not None
            else int(self._config.top_k_per_class)
        )
        if k_per_class < 0:
            raise ValueError("top_k_per_class must be non-negative")

        query_ones = embedding.count()
        if query_ones <= 0:
            uniform = 1.0 / max(len(self._expected_classes), 1)
            return PredictionResult(
                predicted_class=int(self._expected_classes[0]) if self._expected_classes else -1,
                probabilities=[uniform for _ in self._expected_classes],
                confidence=uniform,
            )

        min_sim = float(self._config.min_similarity)

        # Fuzzy search over the full embedding memory (approx. associative recall).
        # Two modes:
        #  - Global top-K (default)
        #  - Balanced per-class top-K then merge (helps when class counts are skewed)
        import heapq

        neighbors: list[tuple[float, int]] = []

        if k_per_class > 0:
            heaps_by_class: dict[int, list[tuple[float, int]]] = {lbl: [] for lbl in self._expected_classes}
            for label, stored, stored_ones in self._items:
                sim = self._similarity_value(embedding, query_ones, stored, stored_ones)
                if sim <= min_sim:
                    continue
                h = heaps_by_class[label]
                if len(h) < k_per_class:
                    heapq.heappush(h, (sim, label))
                elif sim > h[0][0]:
                    heapq.heapreplace(h, (sim, label))

            for h in heaps_by_class.values():
                neighbors.extend(h)

            # Optional final trim to global top-K to keep vote compute stable.
            if len(neighbors) > k_total:
                neighbors = heapq.nlargest(k_total, neighbors, key=lambda pair: pair[0])
            strategy = "per_class_topk"
        else:
            heap: list[tuple[float, int]] = []
            for label, stored, stored_ones in self._items:
                sim = self._similarity_value(embedding, query_ones, stored, stored_ones)
                if sim <= min_sim:
                    continue
                if len(heap) < k_total:
                    heapq.heappush(heap, (sim, label))
                elif sim > heap[0][0]:
                    heapq.heapreplace(heap, (sim, label))
            neighbors = heapq.nlargest(len(heap), heap, key=lambda pair: pair[0])
            strategy = "global_topk"

        if not neighbors:
            uniform = 1.0 / max(len(self._expected_classes), 1)
            probabilities = [uniform for _ in self._expected_classes]
            predicted_label = self._expected_classes[int(np.argmax(probabilities))] if probabilities else -1
            confidence = max(probabilities) if probabilities else 0.0
            LOGGER.event(
                "decoder.base.predict",
                section=SIMILARITY_DEFINITION,
                data={
                    "classes": len(self._expected_classes),
                    "best_label": predicted_label,
                    "confidence": confidence,
                    "temperature": temp,
                    "similarity": self._config.similarity,
                    "top_k": k_total,
                    "top_k_per_class": k_per_class,
                    "neighbors": 0,
                    "strategy": strategy,
                    "min_similarity": min_sim,
                    "low_confidence": confidence < self._config.min_confidence,
                },
            )
            return PredictionResult(
                predicted_class=int(predicted_label),
                probabilities=probabilities,
                confidence=confidence,
            )

        # Stable soft-vote: subtract max sim before exp.
        max_sim = max(sim for sim, _ in neighbors)
        class_weight: dict[int, float] = {label: 0.0 for label in self._expected_classes}
        class_best_sim: dict[int, float] = {label: 0.0 for label in self._expected_classes}
        class_count: dict[int, int] = {label: 0 for label in self._expected_classes}

        for sim, label in neighbors:
            w = math.exp((sim - max_sim) / temp)
            class_weight[label] += w
            class_count[label] += 1
            if sim > class_best_sim[label]:
                class_best_sim[label] = sim

        total_w = sum(class_weight.values())
        if total_w <= 0.0:
            uniform = 1.0 / max(len(self._expected_classes), 1)
            probabilities = [uniform for _ in self._expected_classes]
        else:
            probabilities = [class_weight[label] / total_w for label in self._expected_classes]

        predicted_label = self._expected_classes[int(np.argmax(probabilities))] if probabilities else -1
        confidence = max(probabilities) if probabilities else 0.0

        details = [
            (label, class_weight[label], class_best_sim[label], class_count[label])
            for label in self._expected_classes
        ]
        best_scores = sorted(details, key=lambda item: item[1], reverse=True)[:3]

        LOGGER.event(
            "decoder.base.predict",
            section=SIMILARITY_DEFINITION,
            data={
                "classes": len(self._expected_classes),
                "best_label": predicted_label,
                "confidence": confidence,
                "temperature": temp,
                "similarity": self._config.similarity,
                "top_k": k_total,
                "top_k_per_class": k_per_class,
                "neighbors": len(neighbors),
                "strategy": strategy,
                "min_similarity": min_sim,
                "top_scores": best_scores,
                "low_confidence": confidence < self._config.min_confidence,
            },
        )
        return PredictionResult(
            predicted_class=int(predicted_label),
            probabilities=probabilities,
            confidence=confidence,
        )

    def predict_many(
        self,
        embeddings: Sequence[BitArray],
        *,
        temperature: float | None = None,
    ) -> list[PredictionResult]:
        prepared = self.prepare_many(embeddings)
        return self.predict_prepared(prepared, temperature=temperature)

    def prepare_many(
        self, embeddings: Sequence[BitArray]
    ) -> list[PreparedEmbeddingNeighbors]:
        if not embeddings:
            return []
        prepared: list[PreparedEmbeddingNeighbors] = []
        batch_size = self._config.similarity_batch_size
        for start in range(0, len(embeddings), batch_size):
            batch = embeddings[start : start + batch_size]
            query_matrix = np.empty((len(batch), self._code_length), dtype=np.float32)
            for row, embedding in enumerate(batch):
                if len(embedding) != self._code_length:
                    raise ValueError("embedding sizes must match for similarity")
                query_matrix[row] = np.frombuffer(
                    embedding._bits,
                    dtype=np.uint8,
                    count=self._code_length,
                )
            query_ones = query_matrix.sum(axis=1, dtype=np.float64)
            common = query_matrix @ self._matrix_t
            if (self._config.similarity or "").lower() in ("cosine", "cos"):
                denominator = np.sqrt(
                    query_ones[:, None]
                    * self._ones_array.astype(np.float64, copy=False)[None, :]
                )
            else:
                denominator = (
                    query_ones[:, None]
                    + self._ones_array.astype(np.float64, copy=False)[None, :]
                    - common
                )
            similarities = np.divide(
                common,
                denominator,
                out=np.zeros_like(common, dtype=np.float64),
                where=denominator != 0.0,
            )
            for row in range(len(batch)):
                if query_ones[row] <= 0.0:
                    prepared.append(PreparedEmbeddingNeighbors((), "empty"))
                    continue
                neighbors, strategy = self._select_neighbors_vectorized(
                    similarities[row]
                )
                prepared.append(
                    PreparedEmbeddingNeighbors(tuple(neighbors), strategy)
                )
        LOGGER.event(
            "decoder.base.prepare_many",
            section=SIMILARITY_DEFINITION,
            data={
                "queries": len(embeddings),
                "memory": len(self._items),
                "batch_size": batch_size,
                "similarity": self._config.similarity,
                "top_k": self._config.knn_top_k,
            },
        )
        return prepared

    def predict_prepared(
        self,
        prepared: Sequence[PreparedEmbeddingNeighbors],
        *,
        temperature: float | None = None,
    ) -> list[PredictionResult]:
        temp = self._config.temperature if temperature is None else float(temperature)
        if temp <= 0.0:
            raise ValueError("temperature must be positive")
        results = [
            self._uniform_prediction()
            if item.strategy == "empty"
            else self._prediction_from_neighbors(
                item.neighbors,
                temperature=temp,
                strategy=item.strategy,
            )
            for item in prepared
        ]
        LOGGER.event(
            "decoder.base.predict_many",
            section=SIMILARITY_DEFINITION,
            data={
                "queries": len(prepared),
                "temperature": temp,
                "neighbors_reused": True,
            },
        )
        return results

    def _select_neighbors_vectorized(
        self, similarities: "np.ndarray"
    ) -> tuple[list[tuple[float, int]], str]:
        minimum = float(self._config.min_similarity)
        k_total = min(int(self._config.knn_top_k), len(self._items))
        k_per_class = int(self._config.top_k_per_class)
        if k_per_class <= 0:
            heap: list[tuple[float, int]] = []
            for similarity, label in zip(similarities, self._labels_array):
                value = float(similarity)
                if value <= minimum:
                    continue
                pair = (value, int(label))
                if len(heap) < k_total:
                    heapq.heappush(heap, pair)
                elif value > heap[0][0]:
                    heapq.heapreplace(heap, pair)
            return heapq.nlargest(len(heap), heap, key=lambda pair: pair[0]), "global_topk"

        heaps_by_class: dict[int, list[tuple[float, int]]] = {
            label: [] for label in self._expected_classes
        }
        for similarity, raw_label in zip(similarities, self._labels_array):
            value = float(similarity)
            if value <= minimum:
                continue
            label = int(raw_label)
            heap = heaps_by_class[label]
            pair = (value, label)
            if len(heap) < k_per_class:
                heapq.heappush(heap, pair)
            elif value > heap[0][0]:
                heapq.heapreplace(heap, pair)
        neighbors: list[tuple[float, int]] = []
        for heap in heaps_by_class.values():
            neighbors.extend(heap)
        if len(neighbors) > k_total:
            neighbors = heapq.nlargest(k_total, neighbors, key=lambda pair: pair[0])
        return neighbors, "per_class_topk"

    def _uniform_prediction(self) -> PredictionResult:
        uniform = 1.0 / max(len(self._expected_classes), 1)
        return PredictionResult(
            predicted_class=int(self._expected_classes[0])
            if self._expected_classes
            else -1,
            probabilities=[uniform for _ in self._expected_classes],
            confidence=uniform,
        )

    def _prediction_from_neighbors(
        self,
        neighbors: Sequence[tuple[float, int]],
        *,
        temperature: float,
        strategy: str,
    ) -> PredictionResult:
        if not neighbors:
            return self._uniform_prediction()
        max_sim = max(sim for sim, _ in neighbors)
        class_weight = {label: 0.0 for label in self._expected_classes}
        for similarity, label in neighbors:
            class_weight[label] += math.exp((similarity - max_sim) / temperature)
        total = sum(class_weight.values())
        probabilities = (
            [class_weight[label] / total for label in self._expected_classes]
            if total > 0.0
            else self._uniform_prediction().probabilities
        )
        predicted = (
            self._expected_classes[int(np.argmax(probabilities))]
            if probabilities
            else -1
        )
        return PredictionResult(
            predicted_class=int(predicted),
            probabilities=probabilities,
            confidence=max(probabilities) if probabilities else 0.0,
        )

    def _similarity_value(self, query: BitArray, query_ones: int, stored: BitArray, stored_ones: int) -> float:
        if len(query) != len(stored):
            raise ValueError("embedding sizes must match for similarity")
        if query_ones <= 0 or stored_ones <= 0:
            return 0.0

        common = query.common(stored)
        sim_name = (self._config.similarity or "").lower()

        if sim_name in ("cosine", "cos"):
            denom = math.sqrt(query_ones * stored_ones)
            return 0.0 if denom == 0 else common / denom

        # Для бинарных векторов формула J2 из §2.2.4.2 совпадает с Jaccard.
        union = query_ones + stored_ones - common
        if union == 0:
            return 0.0
        return common / union


class MnistDecoder:
    def __init__(
        self,
        *,
        encoder: Encoder,
        extractor: MnistSobelAngleMap,
        activation_engine: ActivationEngine | None,
        embedding_db: EmbeddingDatabase,
        evaluation_confidence_bins: int,
        evaluation_error_examples: int,
        stimulus_embedding_builder: StimulusEmbeddingBuilder,
        structural_activation_interval: int,
        evaluation_workers: int = 1,
        evaluation_backend: str = "thread",
        evaluation_chunks_per_worker: int = 1,
        evaluation_process_min_samples: int = 1,
    ) -> None:
        self._encoder = encoder
        self._extractor = extractor
        self._activation_engine = activation_engine
        self._embedding_db = embedding_db
        self._evaluation_confidence_bins = int(evaluation_confidence_bins)
        self._evaluation_error_examples = int(evaluation_error_examples)
        self._stimulus_embedding_builder = stimulus_embedding_builder
        if structural_activation_interval <= 0:
            raise ValueError("structural_activation_interval must be positive")
        self._structural_activation_interval = int(structural_activation_interval)
        if evaluation_workers <= 0:
            raise ValueError("evaluation_workers must be positive")
        self._evaluation_workers = int(evaluation_workers)
        if evaluation_backend not in ("thread", "process"):
            raise ValueError("evaluation_backend must be thread or process")
        self._evaluation_backend = evaluation_backend
        if evaluation_chunks_per_worker <= 0:
            raise ValueError("evaluation_chunks_per_worker must be positive")
        self._evaluation_chunks_per_worker = int(evaluation_chunks_per_worker)
        if evaluation_process_min_samples <= 0:
            raise ValueError("evaluation_process_min_samples must be positive")
        self._evaluation_process_min_samples = int(evaluation_process_min_samples)
        self._embedding_sequence = 0
        self._last_activation_values: list[float] | None = None
        code_len = getattr(activation_engine, "_code_length", None)
        LOGGER.event(
            "decoder.init",
            section=CODE_SPACE_ACTIVATION,
            data={
                "code_length": code_len,
                "classes": len(self._embedding_db.expected_classes),
            },
        )

    @property
    def base_temperature(self) -> float:
        return self._embedding_db.config.temperature

    @property
    def expected_classes(self) -> Sequence[int]:
        return self._embedding_db.expected_classes

    @property
    def embedding_code_length(self) -> int:
        return self._encoder.code_length

    def embed_image(
        self,
        image: np.ndarray,
        *,
        label_hint: int | None = None,
        log_image: bool = False,
    ) -> ActivationResult:
        label_value = -1 if label_hint is None else int(label_hint)
        codes, classification_embedding = self._encode_primary(
            image,
            label_value=label_value,
            log_image=log_image,
        )
        return self._finish_embedding(
            codes,
            classification_embedding,
            label_value=label_value,
            log_image=log_image,
        )

    def _encode_primary(
        self,
        image: np.ndarray,
        *,
        label_value: int,
        log_image: bool,
        measurements: Sequence[tuple[float, int, int]] | None = None,
    ) -> tuple[list[BitArray], BitArray]:
        if measurements is None:
            measurements_map = self._extractor.extract(image, label=label_value)
            if label_value not in measurements_map:
                raise RuntimeError("не удалось получить измерения для изображения")
            measurements = measurements_map[label_value]
        if not measurements:
            raise RuntimeError("у изображения нет активных измерений")
        encoded = self._encoder.encode_many(measurements)
        codes = [code for _, code in encoded]
        classification_embedding = self._stimulus_embedding_builder.build(codes)
        return codes, classification_embedding

    def _finish_embedding(
        self,
        codes: Sequence[BitArray],
        classification_embedding: BitArray,
        *,
        label_value: int,
        log_image: bool,
        codes_count: int | None = None,
    ) -> ActivationResult:
        self._embedding_sequence += 1
        evaluate_structure = (
            self._activation_engine is not None
            and (
                log_image
                or self._embedding_sequence == 1
                or self._embedding_sequence % self._structural_activation_interval == 0
            )
        )
        if evaluate_structure:
            structural_result = self._activation_engine.activate(codes)
            activation_result = ActivationResult(
                activation_values=structural_result.activation_values,
                activation_matrix=structural_result.activation_matrix,
                active_detectors=structural_result.active_detectors,
                embedding=structural_result.embedding,
                classification_embedding=classification_embedding,
            )
            self._last_activation_values = list(activation_result.activation_values)
        else:
            activation_result = ActivationResult(
                activation_values=[],
                activation_matrix=[],
                active_detectors=[],
                embedding=classification_embedding,
                classification_embedding=classification_embedding,
            )
        LOGGER.event(
            "decoder.embed",
            section=CODE_SPACE_ACTIVATION,
            data={
                "codes": len(codes) if codes_count is None else codes_count,
                "active_detectors": len(activation_result.active_detectors),
                "embedding_bits": activation_result.embedding.count(),
                "classification_embedding_bits": classification_embedding.count(),
                "structural_evaluated": evaluate_structure,
                "label_hint": label_value,
            },
        )
        return activation_result

    def embed_batch(
        self,
        images: Sequence[np.ndarray],
        labels: Sequence[int] | None = None,
        *,
        log_first: bool = False,
    ) -> list[BitArray]:
        if not images:
            return []

        resolved_labels = [
            int(labels[index])
            if labels is not None and index < len(labels)
            else -1
            for index in range(len(images))
        ]
        measurement_maps = self._extractor.extract_many(
            np.stack(images), resolved_labels
        )

        def encode_item(item: tuple[int, np.ndarray]):
            index, image = item
            label_value = resolved_labels[index]
            return label_value, self._encode_primary(
                image,
                label_value=label_value,
                log_image=log_first and index == 0,
                measurements=measurement_maps[index][label_value],
            )

        workers = min(self._evaluation_workers, len(images))
        use_processes = (
            workers > 1
            and self._evaluation_backend == "process"
            and len(images) >= self._evaluation_process_min_samples
        )
        if use_processes:
            tasks = [
                (
                    measurement_maps[index][resolved_labels[index]],
                    resolved_labels[index],
                    bool(
                        (log_first and index == 0)
                        or self._embedding_sequence + index + 1 == 1
                        or (
                            self._activation_engine is not None
                            and (self._embedding_sequence + index + 1)
                            % self._structural_activation_interval
                            == 0
                        )
                    ),
                )
                for index in range(len(images))
            ]
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                initializer=DecoderEncodingWorker.configure,
                initargs=(
                    self._encoder,
                    self._extractor,
                    self._stimulus_embedding_builder,
                ),
            ) as executor:
                chunk_size = max(
                    1,
                    len(tasks)
                    // (workers * self._evaluation_chunks_per_worker),
                )
                processed = list(
                    executor.map(
                        DecoderEncodingWorker.encode,
                        tasks,
                        chunksize=chunk_size,
                    )
                )
            encoded = [
                (
                    tasks[index][1],
                    codes or (),
                    embedding,
                    codes_count,
                )
                for index, (codes_count, embedding, codes) in enumerate(processed)
            ]
        elif workers == 1:
            sequential = [encode_item(item) for item in enumerate(images)]
            encoded = [
                (label, codes, embedding, None)
                for label, (codes, embedding) in sequential
            ]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                threaded = list(executor.map(encode_item, enumerate(images)))
            encoded = [
                (label, codes, embedding, None)
                for label, (codes, embedding) in threaded
            ]
        embeddings: list[BitArray] = []
        for index, (
            label_value,
            codes,
            classification_embedding,
            codes_count,
        ) in enumerate(encoded):
            self._finish_embedding(
                codes,
                classification_embedding,
                label_value=label_value,
                log_image=log_first and index == 0,
                codes_count=codes_count,
            )
            embeddings.append(classification_embedding)
        LOGGER.event(
            "decoder.embedding.batch",
            section=CODE_SPACE_ACTIVATION,
            data={
                "images": len(images),
                "workers": workers,
                "backend": "process" if use_processes else "thread",
                "code_length": len(embeddings[0]) if embeddings else 0,
            },
        )
        return embeddings

    def predict_image(
        self,
        image: np.ndarray,
        *,
        label_hint: int | None = None,
        log_image: bool = False,
        temperature: float | None = None,
    ) -> PredictionResult:
        activation_result = self.embed_image(image, label_hint=label_hint, log_image=log_image)
        prediction = self._embedding_db.predict(
            activation_result.classification_embedding or activation_result.embedding,
            temperature=temperature,
        )
        LOGGER.event(
            "decoder.predict.image",
            section=STIMULUS_DETECTION,
            data={
                "predicted_class": prediction.predicted_class,
                "confidence": prediction.confidence,
                "label_hint": label_hint if label_hint is not None else -1,
                "temperature": temperature if temperature is not None else self._embedding_db.config.temperature,
            },
        )
        return prediction

    def predict_batch(
        self,
        images: Sequence[np.ndarray],
        labels: Sequence[int] | None = None,
        *,
        temperature: float | None = None,
        log_first: bool = False,
    ) -> tuple[list[PredictionResult], float | None]:
        embeddings = self.embed_batch(images, labels=labels, log_first=log_first)
        return self.predict_embeddings_batch(
            embeddings,
            images=images,
            labels=labels,
            temperature=temperature,
        )

    def predict_embeddings_batch(
        self,
        embeddings: Sequence[BitArray],
        *,
        images: Sequence[np.ndarray],
        labels: Sequence[int] | None = None,
        temperature: float | None = None,
    ) -> tuple[list[PredictionResult], float | None]:
        prepared = self.prepare_embeddings_batch(embeddings)
        return self.predict_prepared_batch(
            prepared,
            images=images,
            labels=labels,
            temperature=temperature,
        )

    def prepare_embeddings_batch(
        self, embeddings: Sequence[BitArray]
    ) -> list[PreparedEmbeddingNeighbors]:
        return self._embedding_db.prepare_many(embeddings)

    def predict_prepared_batch(
        self,
        prepared: Sequence[PreparedEmbeddingNeighbors],
        *,
        images: Sequence[np.ndarray],
        labels: Sequence[int] | None = None,
        temperature: float | None = None,
    ) -> tuple[list[PredictionResult], float | None]:
        results = self._embedding_db.predict_prepared(
            prepared,
            temperature=temperature,
        )
        correct = 0
        diagnostics = (
            _EvaluationDiagnostics(
                self.expected_classes,
                confidence_bins=self._evaluation_confidence_bins,
                error_examples=self._evaluation_error_examples,
            )
            if labels is not None
            else None
        )
        for idx, (image, prediction) in enumerate(zip(images, results)):
            if labels and idx < len(labels):
                if int(labels[idx]) == prediction.predicted_class:
                    correct += 1
                if diagnostics is not None:
                    diagnostics.update(
                        sample_index=idx,
                        image=image,
                        expected=int(labels[idx]),
                        prediction=prediction,
                    )
        accuracy = None
        if labels and results:
            accuracy = correct / len(results)
        if diagnostics is not None:
            diagnostics.log_summary()
        if self._last_activation_values is not None and self._activation_engine is not None:
            self._activation_engine.log_activation_overview(
                self._last_activation_values
            )
        LOGGER.event(
            "decoder.predict.batch",
            section=STIMULUS_DETECTION,
            data={
                "samples": len(results),
                "accuracy": accuracy,
                "temperature": temperature if temperature is not None else self._embedding_db.config.temperature,
            },
        )
        return results, accuracy
