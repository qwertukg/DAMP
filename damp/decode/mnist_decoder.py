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


@dataclass(frozen=True)
class PredictionResult:
    predicted_class: int
    probabilities: list[float]
    confidence: float


@dataclass(frozen=True)
class LabeledEmbedding:
    label: int
    embedding: BitArray


class EmbeddingDatabase:
    def __init__(self, embeddings: Sequence[LabeledEmbedding], config: DecoderConfig) -> None:
        if not embeddings:
            raise ValueError("embeddings must be non-empty")
        if config.temperature <= 0:
            raise ValueError("temperature must be positive")
        self._config = config
        self._embeddings = list(embeddings)
        expected = list(config.expected_classes)
        if not expected:
            expected = sorted({item.label for item in embeddings})
        self._expected_classes = expected
        self._by_class: dict[int, list[BitArray]] = defaultdict(list)
        for item in self._embeddings:
            if item.label not in self._expected_classes:
                continue
            self._by_class[item.label].append(item.embedding)
        LOGGER.event(
            "decoder.base.init",
            section=SIMILARITY_MEASURES,
            data={
                "embeddings": len(self._embeddings),
                "classes": len(self._expected_classes),
                "similarity": self._config.similarity,
                "temperature": self._config.temperature,
            },
        )

    @property
    def expected_classes(self) -> Sequence[int]:
        return tuple(self._expected_classes)

    @property
    def config(self) -> DecoderConfig:
        return self._config

    def predict(self, embedding: BitArray, *, temperature: float | None = None) -> PredictionResult:
        if len(embedding) == 0:
            raise ValueError("embedding must be non-empty")
        query_ones = embedding.count()
        class_scores: dict[int, float] = {}
        class_details: list[tuple[int, float, float, int]] = []
        for label in self._expected_classes:
            sims: list[float] = []
            for stored in self._by_class.get(label, []):
                sims.append(self._similarity_value(embedding, query_ones, stored))
            if sims:
                score = sum(sims) / len(sims)
                best = max(sims)
            else:
                score = float("-inf")
                best = 0.0
            class_scores[label] = score
            class_details.append((label, score, best, len(sims)))
        probs_map = self._to_probabilities(class_scores, temperature=temperature)
        probabilities = [probs_map.get(label, 0.0) for label in self._expected_classes]
        predicted_label = self._expected_classes[int(np.argmax(probabilities))] if probabilities else -1
        confidence = max(probabilities) if probabilities else 0.0
        best_scores = sorted(class_details, key=lambda item: item[1], reverse=True)[:3]
        LOGGER.event(
            "decoder.base.predict",
            section=SIMILARITY_DEFINITION,
            data={
                "classes": len(self._expected_classes),
                "best_label": predicted_label,
                "confidence": confidence,
                "temperature": temperature if temperature is not None else self._config.temperature,
                "similarity": self._config.similarity,
                "top_scores": best_scores,
                "low_confidence": confidence < self._config.min_confidence,
            },
        )
        return PredictionResult(
            predicted_class=int(predicted_label),
            probabilities=probabilities,
            confidence=confidence,
        )

    def _to_probabilities(self, scores: dict[int, float], *, temperature: float | None) -> dict[int, float]:
        temp = temperature if temperature is not None else self._config.temperature
        if temp <= 0:
            raise ValueError("temperature must be positive")
        if not scores:
            return {}
        scaled = {label: score / temp for label, score in scores.items()}
        max_scaled = max(scaled.values())
        exps = {label: math.exp(value - max_scaled) for label, value in scaled.items()}
        total = sum(exps.values()) or 1.0
        probs = {label: value / total for label, value in exps.items()}
        min_conf = self._config.min_confidence
        if min_conf > 0:
            probs = {label: max(prob, 0.0) for label, prob in probs.items()}
        return probs

    def _similarity_value(self, query: BitArray, query_ones: int, stored: BitArray) -> float:
        if len(query) != len(stored):
            raise ValueError("embedding sizes must match for similarity")
        ones_b = stored.count()
        if query_ones <= 0 or ones_b <= 0:
            return 0.0
        common = query.common(stored)
        if self._config.similarity == "cosine":
            denom = math.sqrt(query_ones * ones_b)
            return 0.0 if denom == 0 else common / denom
        union = query_ones + ones_b - common
        return 0.0 if union == 0 else common / union


class MnistDecoder:
    def __init__(
        self,
        *,
        encoder: Encoder,
        extractor: MnistSobelAngleMap,
        activation_engine: ActivationEngine,
        embedding_db: EmbeddingDatabase,
    ) -> None:
        self._encoder = encoder
        self._extractor = extractor
        self._activation_engine = activation_engine
        self._embedding_db = embedding_db
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
        accuracy = None
        if labels and results:
            accuracy = correct / len(results)
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
