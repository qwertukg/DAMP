from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from damp.article_refs import CODE_SPACE_ACTIVATION, SIMILARITY_DEFINITION, SIMILARITY_MEASURES, STIMULUS_DETECTION
from damp.activation import ActivationEngine, ActivationResult
from damp.encoding.bitarray import BitArray
from damp.encoding.damp_encoder import Encoder
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


@dataclass(frozen=True)
class PredictionResult:
    predicted_class: int
    probabilities: list[float]
    confidence: float


@dataclass(frozen=True)
class LabeledEmbedding:
    label: int
    embedding: BitArray


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
                "classes": self._classes,
                "confusion_matrix": self._confusion.tolist(),
                "error_examples": [
                    {
                        "expected": expected,
                        "predicted": predicted,
                        "confidence": confidence,
                    }
                    for _, expected, predicted, confidence in self._errors
                ],
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
        activation_engine: ActivationEngine,
        embedding_db: EmbeddingDatabase,
        evaluation_confidence_bins: int,
        evaluation_error_examples: int,
    ) -> None:
        self._encoder = encoder
        self._extractor = extractor
        self._activation_engine = activation_engine
        self._embedding_db = embedding_db
        self._evaluation_confidence_bins = int(evaluation_confidence_bins)
        self._evaluation_error_examples = int(evaluation_error_examples)
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

    def embed_image(
        self,
        image: np.ndarray,
        *,
        label_hint: int | None = None,
        log_image: bool = False,
    ) -> ActivationResult:
        label_value = -1 if label_hint is None else int(label_hint)
        measurements_map = self._extractor.extract(image, label=label_value)
        if label_value not in measurements_map:
            raise RuntimeError("не удалось получить измерения для изображения")
        measurements = measurements_map[label_value]
        if not measurements:
            raise RuntimeError("у изображения нет активных измерений")
        codes: list[BitArray] = []
        first = True
        for (angle, x, y) in measurements:
            _, code = self._encoder.encode(
                float(angle),
                float(x),
                float(y),
                log_image=image if log_image and first else None,
                log_label=label_value if log_image and first else None,
                log_measurements=measurements if log_image and first else None,
            )
            codes.append(code)
            first = False
        activation_result = self._activation_engine.activate(codes)
        self._last_activation_values = list(activation_result.activation_values)
        LOGGER.event(
            "decoder.embed",
            section=CODE_SPACE_ACTIVATION,
            data={
                "codes": len(codes),
                "active_detectors": len(activation_result.active_detectors),
                "embedding_bits": activation_result.embedding.count(),
                "label_hint": label_value,
            },
        )
        return activation_result

    def predict_image(
        self,
        image: np.ndarray,
        *,
        label_hint: int | None = None,
        log_image: bool = False,
        temperature: float | None = None,
    ) -> PredictionResult:
        activation_result = self.embed_image(image, label_hint=label_hint, log_image=log_image)
        prediction = self._embedding_db.predict(activation_result.embedding, temperature=temperature)
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
        results: list[PredictionResult] = []
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
        for idx, image in enumerate(images):
            label_hint = labels[idx] if labels and idx < len(labels) else None
            prediction = self.predict_image(
                image,
                label_hint=label_hint,
                log_image=log_first and idx == 0,
                temperature=temperature,
            )
            results.append(prediction)
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
        if self._last_activation_values is not None:
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
