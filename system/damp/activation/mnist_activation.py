"""Этапы 4–5 для команды layout: детекторы над раскладкой (§6.4, §7.1.10), размер выходного кода (§6.6),
база эмбеддингов и чекпоинты прогона."""

from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Mapping, Sequence

from damp import settings
from damp.activation import ActivationEngine, DbscanClusterer, Detector, DetectorHierarchyBuilder, DetectorLayerConfig, DetectorOutputSizer, OutputCodeSize, reassign_output_bits
from damp.article_refs import DETECTOR_CONSTRUCTION, OUTSIDE_ARTICLE
from damp.decode import LabeledEmbedding
from damp.encoding.bitarray import BitArray
from damp.layout.damp_layout import Layout
from damp.logging import LOGGER


@dataclass(frozen=True)
class EmbeddingRecord:
    class_label: str
    embedding_bits: str

    def as_payload(self) -> dict[str, str]:
        return {
            "class": self.class_label,
            "embedding": self.embedding_bits,
        }


class EmbeddingStore:
    def __init__(self, json_key: str, *, version_key: str, version: int) -> None:
        if not json_key:
            raise ValueError("json_key must be provided for embedding storage")
        self._json_key = json_key
        self._version_key = str(version_key)
        self._version = int(version)

    def save(
        self,
        path: str,
        embeddings: Sequence[EmbeddingRecord],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if not path:
            raise ValueError("path must be provided to save embeddings")
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload: dict[str, object] = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fp:
                    payload = json.load(fp)
            except Exception as exc:
                LOGGER.event(
                    "activation.embedding.save",
                    section=OUTSIDE_ARTICLE,
                    data={
                        "path": path,
                        "embeddings": len(embeddings),
                        "error": str(exc),
                        "stage": "read",
                    },
                )
                payload = {}
        payload[self._json_key] = [record.as_payload() for record in embeddings]
        payload[self._version_key] = self._version
        if metadata is not None:
            payload[settings.ACTIVATION_METADATA_KEY] = dict(metadata)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=True, indent=2)
        LOGGER.event(
            "activation.embedding.save",
            section=OUTSIDE_ARTICLE,
            data={
                "path": path,
                "embeddings": len(embeddings),
            "json_key": self._json_key,
        },
    )

    def load(self, path: str) -> list["EmbeddingRecord"]:
        """Эмбеддинги базы, сохранённые в JSON раскладки (вне статьи)."""
        if not path:
            raise ValueError("path must be provided to load embeddings")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        stored_version = data.get(self._version_key)
        if stored_version != self._version:
            LOGGER.event(
                "activation.embedding.incompatible",
                section=OUTSIDE_ARTICLE,
                data={
                    "path": path,
                    "stored_version": stored_version,
                    "expected_version": self._version,
                },
            )
            return []
        raw_embeddings = data.get(self._json_key, [])
        embeddings: list[EmbeddingRecord] = []
        for item in raw_embeddings:
            class_label = item.get("class", item.get("label"))
            embedding_bits = item.get("embedding")
            if class_label is None or embedding_bits is None:
                continue
            embeddings.append(
                EmbeddingRecord(
                    class_label=str(class_label),
                    embedding_bits=str(embedding_bits),
                )
            )
        classes = sorted({int(rec.class_label) for rec in embeddings})
        LOGGER.event(
            "activation.embedding.load",
            section=OUTSIDE_ARTICLE,
            data={
                "path": path,
                "embeddings": len(embeddings),
                "classes": classes,
                "json_key": self._json_key,
            },
        )
        return embeddings

    @staticmethod
    def to_labeled(records: Sequence["EmbeddingRecord"]) -> list[LabeledEmbedding]:
        labeled: list[LabeledEmbedding] = []
        for record in records:
            bit_string = record.embedding_bits
            bit_array = BitArray(len(bit_string))
            for idx, char in enumerate(bit_string):
                if char == "1":
                    bit_array.set(idx, 1)
            labeled.append(
                LabeledEmbedding(
                    label=int(record.class_label),
                    embedding=bit_array,
                )
            )
        if labeled:
            LOGGER.event(
                "decoder.base.prepare",
                section=OUTSIDE_ARTICLE,
                data={
                    "embeddings": len(labeled),
                    "classes": len({item.label for item in labeled}),
                    "bit_length": len(labeled[0].embedding),
                },
            )
        return labeled

    def load_metadata(self, path: str) -> dict[str, object]:
        """Размер выходного кода и насыщенность, сохранённые при обучении (§6.6)."""
        if not path or not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if data.get(self._version_key) != self._version:
            return {}
        metadata = data.get(settings.ACTIVATION_METADATA_KEY)
        return dict(metadata) if isinstance(metadata, dict) else {}


class PipelineCheckpointStore:
    """Атомарно сохраняет прогресс этапов и частичные эмбеддинги."""

    def __init__(self, *, suffix: str, version: int) -> None:
        self._suffix = str(suffix)
        self._version = int(version)

    def path_for(self, layout_path: str) -> str:
        return f"{layout_path}{self._suffix}"

    def load(self, layout_path: str) -> dict[str, object]:
        path = self.path_for(layout_path)
        if not os.path.exists(path):
            return self._empty_payload()
        try:
            with open(path, "r", encoding="utf-8") as fp:
                payload = json.load(fp)
        except Exception as exc:
            LOGGER.event(
                "pipeline.checkpoint.load.error",
                section=OUTSIDE_ARTICLE,
                data={"path": path, "error": str(exc)},
            )
            return self._empty_payload()
        if payload.get("version") != self._version:
            LOGGER.event(
                "pipeline.checkpoint.incompatible",
                section=OUTSIDE_ARTICLE,
                data={
                    "path": path,
                    "stored_version": payload.get("version"),
                    "expected_version": self._version,
                },
            )
            return self._empty_payload()
        LOGGER.event(
            "pipeline.checkpoint.load",
            section=OUTSIDE_ARTICLE,
            data={
                "path": path,
                "completed": len(payload.get("embeddings", [])),
                "stages": payload.get("stages", {}),
            },
        )
        return payload

    def save(
        self,
        layout_path: str,
        *,
        stages: dict[str, bool],
        target_samples: int,
        embeddings: Sequence[EmbeddingRecord],
    ) -> None:
        path = self.path_for(layout_path)
        payload = {
            "version": self._version,
            "target_samples": int(target_samples),
            "stages": dict(stages),
            "embeddings": [record.as_payload() for record in embeddings],
        }
        temporary_path = f"{path}.tmp"
        with open(temporary_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=True, indent=2)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(temporary_path, path)
        LOGGER.event(
            "pipeline.checkpoint.save",
            section=OUTSIDE_ARTICLE,
            data={
                "path": path,
                "completed": len(embeddings),
                "target": int(target_samples),
                "stages": stages,
            },
        )

    def reset(self, layout_path: str, *, target_samples: int) -> None:
        """Пустой чекпоинт: этапы и эмбеддинги прошлого запуска не используются (запуск без кеша)."""
        self.save(layout_path, stages={}, target_samples=target_samples, embeddings=[])

    def records(self, payload: dict[str, object]) -> list[EmbeddingRecord]:
        records: list[EmbeddingRecord] = []
        for item in payload.get("embeddings", []):
            if not isinstance(item, dict):
                continue
            class_label = item.get("class")
            embedding = item.get("embedding")
            if class_label is None or embedding is None:
                continue
            records.append(EmbeddingRecord(str(class_label), str(embedding)))
        return records

    def _empty_payload(self) -> dict[str, object]:
        return {"version": self._version, "target_samples": 0, "stages": {}, "embeddings": []}


def _detector_layer_configs(layout: Layout) -> tuple[DetectorLayerConfig, ...]:
    """Слои иерархии: один порог λ_d на слой, r_a и минимум точек (§6.4, §7.1.10)."""
    extent = max(1, min(layout.width, layout.height))
    return tuple(
        DetectorLayerConfig(
            lambda_threshold=float(level),
            activation_radius=max(settings.ACTIVATION_DETECTOR_RADIUS_MIN, int(round(extent * float(radius_factor)))),
            min_points=int(min_points),
        )
        for level, radius_factor, min_points in settings.ACTIVATION_DETECTOR_LAYER_SPECS
    )


def _build_detectors_from_layout(
    layout: Layout,
    energies: list[float],
    *,
    output_bits: int,
) -> list[Detector]:
    return DetectorHierarchyBuilder(
        layout=layout,
        energies=energies,
        layer_configs=_detector_layer_configs(layout),
        radius_min=settings.ACTIVATION_DETECTOR_RADIUS_MIN,
        output_bits=output_bits,
        energy_threshold=settings.ACTIVATION_ENERGY_THRESHOLD,
        eta=settings.ACTIVATION_ETA,
        attempts_without_change=settings.ACTIVATION_DETECTOR_ATTEMPTS_WITHOUT_CHANGE,
        seed=settings.ACTIVATION_DETECTOR_SEED,
        clusterer=DbscanClusterer(
            eps=settings.ACTIVATION_DBSCAN_EPS,
            min_samples=settings.ACTIVATION_DBSCAN_MIN_SAMPLES,
        ),
    ).build()


def _build_activation_engine(
    layout: Layout,
    energies: list[float],
    *,
    detectors: Sequence[Detector] | None = None,
    saturation_limit: int | None = None,
) -> ActivationEngine:
    saturation_limit = settings.ACTIVATION_SATURATION_LIMIT if saturation_limit is None else saturation_limit
    if detectors is None:
        detectors = _build_detectors_from_layout(
            layout,
            energies,
            output_bits=settings.ACTIVATION_OUTPUT_BITS,
        )
    return ActivationEngine(
        layout=layout,
        detectors=detectors,
        lambda_threshold=settings.ACTIVATION_LAMBDA_THRESHOLD,
        eta=settings.ACTIVATION_ETA,
        energy_threshold=settings.ACTIVATION_ENERGY_THRESHOLD,
        detector_threshold=settings.ACTIVATION_DETECTOR_THRESHOLD,
        saturation_limit=saturation_limit,
        energy_map=energies,
        use_gpu=settings.ACTIVATION_USE_GPU,
    )


def _output_sizer() -> DetectorOutputSizer:
    return DetectorOutputSizer(
        max_density=settings.ACTIVATION_MAX_EMBEDDING_DENSITY,
        min_output_bits=settings.ACTIVATION_OUTPUT_BITS,
        min_saturation=settings.ACTIVATION_SATURATION_MIN,
        max_saturation=settings.ACTIVATION_SATURATION_MAX,
    )


@dataclass(frozen=True)
class CalibratedActivation:
    """Движок после калибровки размера выхода и эмбеддинги калибровочных стимулов (§6.6)."""

    engine: ActivationEngine
    size: OutputCodeSize
    embeddings: tuple[BitArray, ...]
    active_counts: tuple[int, ...]


def _calibrated_activation_engine(
    layout: Layout,
    energies: list[float],
    calibration_batches: Sequence[Sequence[BitArray]],
) -> CalibratedActivation:
    """§6.6: строит детекторы, измеряет среднее число активных и задаёт размер выхода.

    Калибровочные стимулы активируются один раз: уровни детекторов не зависят от выходных кодов и σ
    (§6.5), поэтому после перевыдачи битов (§6.3) их эмбеддинги собираются из тех же уровней.
    """
    detectors = _build_detectors_from_layout(layout, energies, output_bits=settings.ACTIVATION_OUTPUT_BITS)
    provisional = _build_activation_engine(layout, energies, detectors=detectors)
    size, active_lists = _output_sizer().measure_with_results(provisional, calibration_batches)
    resized = reassign_output_bits(detectors, output_bits=size.output_bits, seed=settings.ACTIVATION_DETECTOR_SEED)
    engine = provisional.rebind(resized, saturation_limit=size.saturation_limit)
    embeddings = tuple(engine.embedding_from_active(active) for active in active_lists)
    return CalibratedActivation(
        engine=engine,
        size=size,
        embeddings=embeddings,
        active_counts=tuple(len(active) for active in active_lists),
    )


def _activation_engine_from_size(
    layout: Layout,
    energies: list[float],
    *,
    output_bits: int,
    saturation_limit: int,
    stored_detectors: Sequence[Mapping[str, object]] | None = None,
) -> ActivationEngine:
    """Восстанавливает детекторы и выходной код тех же размеров, что при обучении.

    Если детекторы сохранены вместе с эмбеддингами, они читаются из файла; иначе строятся заново
    с тем же seed (§6.4 — построение стохастическое, но воспроизводимое).
    """
    if stored_detectors:
        resized = _detectors_from_payload(stored_detectors, output_bits=output_bits)
        LOGGER.event(
            "activation.detectors.restored",
            section=DETECTOR_CONSTRUCTION,
            data={"detectors": len(resized), "output_bits": output_bits},
        )
    else:
        detectors = _build_detectors_from_layout(layout, energies, output_bits=settings.ACTIVATION_OUTPUT_BITS)
        resized = reassign_output_bits(detectors, output_bits=output_bits, seed=settings.ACTIVATION_DETECTOR_SEED)
    return _build_activation_engine(layout, energies, detectors=resized, saturation_limit=saturation_limit)


def _detectors_payload(detectors: Sequence[Detector]) -> list[dict[str, object]]:
    """Детекторы (§6.3) в JSON: центр, радиус, λ_d, e_d, n_d, единичные биты b_d."""
    return [
        {
            "center": [float(detector.center[0]), float(detector.center[1])],
            "radius": float(detector.radius),
            "lambda": float(detector.lambda_threshold),
            "energy": float(detector.energy),
            "points": int(detector.points_count),
            "bits": [int(bit) for bit in detector.output_code.active_indices()],
            "label": detector.label,
        }
        for detector in detectors
    ]


def _detectors_from_payload(items: Sequence[Mapping[str, object]], *, output_bits: int) -> list[Detector]:
    detectors: list[Detector] = []
    for item in items:
        center = item["center"]
        detectors.append(
            Detector(
                center=(float(center[0]), float(center[1])),
                radius=float(item["radius"]),
                lambda_threshold=float(item["lambda"]),
                energy=float(item["energy"]),
                points_count=int(item["points"]),
                output_code=BitArray.from_indices(output_bits, [int(bit) for bit in item["bits"]]),
                label=None if item.get("label") is None else str(item["label"]),
            )
        )
    return detectors


def _activation_metadata(size: OutputCodeSize, detectors: Sequence[Detector] | None = None) -> dict[str, object]:
    metadata: dict[str, object] = {
        "output_bits": int(size.output_bits),
        "saturation_limit": int(size.saturation_limit),
        "mean_active_detectors": float(size.mean_active_detectors),
        "calibration_samples": int(size.samples),
    }
    if detectors is not None:
        metadata[settings.ACTIVATION_DETECTORS_KEY] = _detectors_payload(detectors)
    return metadata
