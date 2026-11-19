"""Этап 7 по эмбеддингам первого уровня — команды evaluate и predict: нечёткий поиск (§2.2.5)."""

from __future__ import annotations
import os

import numpy as np

from damp import settings
from damp.activation import ActivationEngine
from damp.activation.mnist_activation import EmbeddingRecord, EmbeddingStore, _activation_engine_from_size, _activation_metadata, _calibrated_activation_engine
from damp.article_refs import OUTSIDE_ARTICLE, STIMULUS_DETECTION
from damp.decode import DecoderConfig, EmbeddingDatabase, MnistDecoder, PredictionResult
from damp.encoding.mnist_codes import StimulusCodeCache, _build_colorizer, _build_encoder, _collect_samples
from damp.environment import GPU_ENVIRONMENT, configure_logging
from damp.layout.damp_layout import Layout
from damp.layout.mnist_layout import LayoutCacheFile, LayoutParameterTuner, _layout_cache_paths_for_count
from damp.logging import LOGGER
from damp.stimuli.MnistSobelAngleMap import MnistSobelAngleMap
from damp.stimuli.mnist import _build_extractor, _stimulus_metadata


class MnistDecodeRunner:
    def __init__(self, *, count: int, layout_path: str | None = None) -> None:
        self.count = count
        self.layout_path = layout_path
        self._decoder: MnistDecoder | None = None
        self._temperature_override: float | None = None
        configure_logging()
        GPU_ENVIRONMENT.apply()
        self.encoder = _build_encoder()
        self.colorizer = _build_colorizer()
        self.extractor = _build_extractor()
        self._embedding_store = EmbeddingStore(
            json_key=settings.EMBEDDINGS_JSON_KEY,
            version_key=settings.EMBEDDINGS_VERSION_KEY,
            version=settings.EMBEDDINGS_VERSION,
        )

    def _load_layout(self) -> tuple[Layout, str, int]:
        candidates: list[str] = []
        if self.layout_path:
            candidates.append(self.layout_path)
        candidates.extend(_layout_cache_paths_for_count(self.count))
        seen: set[str] = set()
        for path in candidates:
            if not path or path in seen or not os.path.exists(path):
                continue
            seen.add(path)
            metadata = LayoutCacheFile.inspect(path)
            if metadata is None:
                continue
            total_points, pair_radius, ready = metadata
            if not ready:
                continue
            tuner = LayoutParameterTuner(
                empty_ratio=settings.LAYOUT_EMPTY_RATIO,
                start_radius_factor=settings.LAYOUT_ADAPTIVE_RADIUS_START_FACTOR,
            )
            tuned_params = tuner.tune(total_points)
            layout = LayoutCacheFile.load(
                path,
                tuned_params,
                expected_code_length=self.encoder.code_length,
                colorizer=self.colorizer,
                expected_encoder_signature=self.encoder.signature,
                expected_stimulus=_stimulus_metadata(self.extractor),
            )
            if layout is None:
                continue
            LOGGER.event(
                "decoder.layout.use",
                section=OUTSIDE_ARTICLE,
                data={
                    "path": path,
                    "points": total_points,
                    "pair_radius": pair_radius,
                },
            )
            self.layout_path = path
            return layout, path, total_points
        raise RuntimeError("готовая раскладка не найдена; запустите layout")

    def _ensure_decoder(self) -> MnistDecoder:
        if self._decoder is not None:
            return self._decoder
        layout, layout_path, total_points = self._load_layout()
        energies = layout.point_energies(settings.ACTIVATION_POINT_ENERGY_RADIUS)
        embeddings = self._embedding_store.load(layout_path)
        metadata = self._embedding_store.load_metadata(layout_path)
        activation_engine: ActivationEngine | None = None
        if not embeddings or "output_bits" not in metadata:
            embeddings, activation_engine = self._build_embeddings_for_layout(
                layout,
                layout_path,
                energies,
            )
        labeled_embeddings = EmbeddingStore.to_labeled(embeddings)
        if not labeled_embeddings:
            raise RuntimeError("в раскладке нет базы эмбеддингов")
        if activation_engine is None:
            activation_engine = _activation_engine_from_size(
                layout,
                energies,
                output_bits=int(metadata["output_bits"]),
                saturation_limit=int(metadata["saturation_limit"]),
                stored_detectors=metadata.get(settings.ACTIVATION_DETECTORS_KEY),
            )
        database = EmbeddingDatabase(
            labeled_embeddings,
            DecoderConfig(
                similarity=settings.DECODER_SIMILARITY,
                temperature=settings.DECODER_SOFTMAX_TEMPERATURE,
                min_confidence=settings.DECODER_MIN_CONFIDENCE,
                expected_classes=settings.DECODER_EXPECTED_CLASSES,
                knn_top_k=settings.DECODER_KNN_TOP_K,
                top_k_per_class=settings.DECODER_TOP_K_PER_CLASS,
                min_similarity=settings.DECODER_MIN_SIMILARITY,
            ),
        )
        self._decoder = MnistDecoder(
            encoder=self.encoder,
            extractor=self.extractor,
            activation_engine=activation_engine,
            embedding_db=database,
        )
        LOGGER.event(
            "decoder.build",
            section=OUTSIDE_ARTICLE,
            data={
                "layout_path": layout_path,
                "embeddings": len(labeled_embeddings),
                "points": total_points,
                "temperature": self._decoder.base_temperature,
            },
        )
        return self._decoder

    def _build_embeddings_for_layout(
        self,
        layout: Layout,
        layout_path: str,
        energies: list[float],
    ) -> tuple[list[EmbeddingRecord], ActivationEngine]:
        dataset = self._build_dataset(train=True)
        cache = StimulusCodeCache(dataset, self.encoder, self.extractor, label=None)
        samples = _collect_samples(cache, settings.DECODER_DATASET_LIMIT)
        calibrated = _calibrated_activation_engine(
            layout,
            energies,
            [codes for _, codes in samples[:settings.ACTIVATION_OUTPUT_CALIBRATION_SAMPLES]],
        )
        engine = calibrated.engine
        embeddings: list[EmbeddingRecord] = []
        for idx, (label, codes) in enumerate(samples):
            if idx < len(calibrated.embeddings):
                embedding = calibrated.embeddings[idx]
            else:
                embedding = engine.activate(codes).embedding
            embeddings.append(
                EmbeddingRecord(
                    class_label=str(int(label)),
                    embedding_bits=embedding.to01(),
                )
            )
            LOGGER.event(
                "decoder.base.generate",
                section=STIMULUS_DETECTION,
                data={
                    "index": idx,
                    "label": int(label),
                    "stimuli": len(codes),
                    "bits": embedding.count(),
                },
            )
        self._embedding_store.save(
            layout_path, embeddings, metadata=_activation_metadata(calibrated.size, detectors=engine.detectors)
        )
        LOGGER.event(
            "decoder.base.generated",
            section=STIMULUS_DETECTION,
            data={
                "embeddings": len(embeddings),
                "classes": len({int(item.class_label) for item in embeddings}),
                "path": layout_path,
            },
        )
        return embeddings, engine

    def predict_image(
        self,
        image: "np.ndarray",
        *,
        temperature: float | None = None,
        log_image: bool = False,
    ) -> PredictionResult:
        decoder = self._ensure_decoder()
        temp_value = temperature
        if temp_value is None:
            temp_value = self._temperature_override
        return decoder.predict_image(
            image,
            label_hint=None,
            log_image=log_image,
            temperature=temp_value,
        )

    def predict_image_from_path(
        self,
        path: str,
        *,
        temperature: float | None = None,
        log_image: bool = False,
    ) -> PredictionResult:
        image = MnistSobelAngleMap.load_image(path)
        return self.predict_image(image, temperature=temperature, log_image=log_image)

    def _build_dataset(self, *, train: bool):
        from torchvision import transforms
        from torchvision.datasets import MNIST

        return MNIST(
            root="./data",
            train=train,
            download=True,
            transform=transforms.ToTensor(),
        )

    def predict_dataset(
        self,
        *,
        split: str,
        limit: int,
        temperature: float | None = None,
        log_first: bool = False,
    ) -> tuple[list[PredictionResult], float]:
        decoder = self._ensure_decoder()
        dataset = self._build_dataset(train=split == "train")
        images: list["np.ndarray"] = []
        labels: list[int] = []
        for idx, (img_tensor, label) in enumerate(dataset):
            if idx >= limit:
                break
            images.append(img_tensor.squeeze(0).numpy())
            labels.append(int(label))
        results, accuracy = decoder.predict_batch(
            images,
            labels=labels,
            temperature=temperature if temperature is not None else self._temperature_override,
            log_first=log_first,
        )
        LOGGER.event(
            "decoder.dataset.eval",
            section=OUTSIDE_ARTICLE,
            data={
                "split": split,
                "samples": len(results),
                "accuracy": accuracy,
                "temperature": temperature if temperature is not None else decoder.base_temperature,
            },
        )
        return results, 0.0 if accuracy is None else accuracy

    def calibrate_temperature(self, *, split: str, limit: int) -> tuple[float, float]:
        decoder = self._ensure_decoder()
        dataset = self._build_dataset(train=split == "train")
        images: list["np.ndarray"] = []
        labels: list[int] = []
        for idx, (img_tensor, label) in enumerate(dataset):
            if idx >= limit:
                break
            images.append(img_tensor.squeeze(0).numpy())
            labels.append(int(label))
        best_temp = decoder.base_temperature
        best_acc = -1.0
        if not images:
            self._temperature_override = best_temp
            LOGGER.event(
                "decoder.calibrate",
                section=OUTSIDE_ARTICLE,
                data={
                    "split": split,
                    "samples": 0,
                    "best_temperature": best_temp,
                    "accuracy": 0.0,
                    "candidates": len(settings.DECODER_CALIBRATION_TEMPERATURES),
                    "reason": "empty_dataset",
                },
            )
            return best_temp, 0.0
        for candidate in settings.DECODER_CALIBRATION_TEMPERATURES:
            _, acc = decoder.predict_batch(
                images,
                labels=labels,
                temperature=candidate,
                log_first=False,
            )
            if acc is None:
                continue
            if acc > best_acc:
                best_acc = acc
                best_temp = candidate
        self._temperature_override = best_temp
        LOGGER.event(
            "decoder.calibrate",
            section=OUTSIDE_ARTICLE,
            data={
                "split": split,
                "samples": len(images),
                "best_temperature": best_temp,
                "accuracy": best_acc,
                "candidates": len(settings.DECODER_CALIBRATION_TEMPERATURES),
            },
        )
        return best_temp, best_acc
