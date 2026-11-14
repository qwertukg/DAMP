"""Этапы 2–5 одним проходом без цикла — команда layout: раскладка всех углопозиций обучающих цифр, детекторы,
база эмбеддингов первого уровня (§5, §6)."""

from __future__ import annotations
import os
from typing import Sequence

from PIL import Image

from damp import settings
from damp.activation.mnist_activation import EmbeddingRecord, EmbeddingStore, PipelineCheckpointStore, _activation_metadata, _calibrated_activation_engine
from damp.article_refs import OUTSIDE_ARTICLE, STIMULUS_DETECTION
from damp.encoding.bitarray import BitArray
from damp.encoding.mnist_codes import DigitCodes, EncodedCodesFile, StimulusCodeCache, _build_colorizer, _build_encoder, _collect_codes, _collect_samples
from damp.environment import GPU_ENVIRONMENT, configure_logging
from damp.layout.damp_layout import Layout
from damp.layout.mnist_layout import LayoutCacheFile, LayoutParameterTuner, TunedLayoutParams, _expected_pair_radius, _layout_cache_paths_for_count, _layout_json_path, _run_layout
from damp.logging import LOGGER
from damp.stimuli.mnist import _build_extractor, _stimulus_metadata


class MnistLayoutRunner:
    @staticmethod
    def _save_layout_images(
        layout: Layout,
        *,
        image_path: str,
        energies: Sequence[float] | None = None,
    ) -> None:
        """Карта в псевдоцветах кодов (§5.14), по компоненте угла (рис. 16) и с яркостью Ê (рис. 18)."""
        from PIL import Image

        directory = os.path.dirname(image_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        Image.fromarray(layout.render_image()).save(image_path)
        stem = os.path.splitext(image_path)[0]
        component_path = stem + settings.LAYOUT_COMPONENT_IMAGE_SUFFIX
        Image.fromarray(
            layout.render_component_image(period=settings.MNIST_ANGLE_MAX - settings.MNIST_ANGLE_MIN)
        ).save(component_path)
        energy_path = None
        if energies is not None:
            energy_path = stem + settings.LAYOUT_ENERGY_IMAGE_SUFFIX
            Image.fromarray(layout.render_energy_image(energies)).save(energy_path)
        LOGGER.event(
            "layout.visual.save",
            section=OUTSIDE_ARTICLE,
            data={
                "image": image_path,
                "component_image": component_path,
                "energy_image": energy_path,
                "points": layout.point_count,
            },
        )

    def __init__(
        self,
        *,
        count: int,
        label: int | None = None,
        activation_samples: int | None = None,
        reuse_cache: bool = True,
        layout_only: bool = False,
    ) -> None:
        self.count = count
        self.label = label
        # False (--rebuild) — готовые раскладка, чекпоинт и эмбеддинги игнорируются, всё считается заново
        # и записывается в кеш поверх старого.
        self._reuse_cache = bool(reuse_cache)
        # True (--layout-only) — конвейер заканчивается раскладкой: JSON и картинки сохраняются,
        # активация и эмбеддинги не считаются.
        self._layout_only = bool(layout_only)
        resolved_activation_samples = count if activation_samples is None else int(activation_samples)
        if resolved_activation_samples <= 0:
            raise ValueError("activation_samples must be positive")
        self._activation_samples = resolved_activation_samples
        configure_logging()
        GPU_ENVIRONMENT.apply()
        self.encoder = _build_encoder()
        self.colorizer = _build_colorizer()
        self.extractor = _build_extractor()
        self._codes: StimulusCodeCache | None = None
        self._embedding_store = EmbeddingStore(
            json_key=settings.EMBEDDINGS_JSON_KEY,
            version_key=settings.EMBEDDINGS_VERSION_KEY,
            version=settings.EMBEDDINGS_VERSION,
        )
        self._checkpoint_store = PipelineCheckpointStore(
            suffix=settings.PIPELINE_CHECKPOINT_SUFFIX,
            version=settings.PIPELINE_CHECKPOINT_VERSION,
        )
        self._activation_metadata: dict[str, object] | None = None

    def _layout_cache_paths(self) -> list[str]:
        return _layout_cache_paths_for_count(self.count)

    def _load_layout_metadata(self, path: str) -> tuple[int, int | None, bool] | None:
        return LayoutCacheFile.inspect(path)

    def _try_load_cached_layout(self) -> tuple[Layout, TunedLayoutParams, int, str] | None:
        for path in self._layout_cache_paths():
            metadata = self._load_layout_metadata(path)
            if metadata is None:
                continue
            total_points, pair_radius, ready = metadata
            if not ready:
                LOGGER.event(
                    "layout.cache.incomplete",
                    section=OUTSIDE_ARTICLE,
                    data={
                        "path": path,
                        "points": total_points,
                        "pair_radius": pair_radius,
                    },
                )
                continue
            tuned_params = self._tune_layout(total_points)
            layout = LayoutCacheFile.load(
                path,
                tuned_params,
                expected_code_length=self.encoder.code_length,
                colorizer=self.colorizer,
                expected_encoder_signature=self.encoder.signature,
                expected_stimulus=_stimulus_metadata(self.extractor),
            )
            if layout is None:
                LOGGER.event(
                    "layout.cache.mismatch",
                    section=OUTSIDE_ARTICLE,
                    data={
                        "path": path,
                        "pair_radius": pair_radius,
                        "points": total_points,
                    },
                )
                continue
            LOGGER.event(
                "layout.cache.hit",
                section=OUTSIDE_ARTICLE,
                data={
                    "path": path,
                    "points": total_points,
                    "pair_radius": pair_radius,
                },
            )
            return layout, tuned_params, total_points, path
        LOGGER.event(
            "layout.cache.miss",
            section=OUTSIDE_ARTICLE,
            data={"count": self.count, "label": self.label},
        )
        return None

    def _build_dataset(self):
        from torchvision import transforms
        from torchvision.datasets import MNIST

        return MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )

    def _code_cache(self, dataset) -> StimulusCodeCache:
        """Коды цифр набора — один общий кеш для раскладки и активации."""
        if self._codes is None:
            store = None
            if settings.ENCODING_CODES_CACHE:
                store = EncodedCodesFile(settings.ENCODING_CODES_CACHE_DIR, encoder=self.encoder, split="train", label=self.label)
            self._codes = StimulusCodeCache(dataset, self.encoder, self.extractor, label=self.label, store=store)
        return self._codes

    def _tune_layout(self, total_codes: int) -> TunedLayoutParams:
        tuner = LayoutParameterTuner(
            empty_ratio=settings.LAYOUT_EMPTY_RATIO,
            start_radius_factor=settings.LAYOUT_ADAPTIVE_RADIUS_START_FACTOR,
        )
        return tuner.tune(total_codes)

    def _log_dataset(self, angles: int, total_codes: int) -> None:
        LOGGER.event(
            "mnist.codes",
            section=OUTSIDE_ARTICLE,
            data={
                "count": self.count,
                "label": "all" if self.label is None else int(self.label),
                "angles": angles,
                "points": total_codes,
            },
        )

    def _export_layout(
        self,
        layout: Layout,
        run_params: TunedLayoutParams,
        *,
        layout_path: str | None = None,
        save_json: bool = True,
    ) -> None:
        resolved_layout_path = layout_path or _layout_json_path(self.count, layout.pair_radius)
        if save_json:
            layout.save_json(resolved_layout_path)
        suffix = "auto" if layout.pair_radius is None else str(layout.pair_radius)
        self._save_layout_images(
            layout,
            image_path=f"data/{self.count}-{suffix}.png",
            energies=layout.point_energies(settings.ACTIVATION_POINT_ENERGY_RADIUS),
        )

    def _run_activation(
        self,
        layout: Layout,
        dataset,
        *,
        layout_path: str,
        stages: dict[str, bool],
        initial_embeddings: Sequence[EmbeddingRecord],
    ) -> list[EmbeddingRecord]:
        energies = layout.point_energies(settings.ACTIVATION_POINT_ENERGY_RADIUS)
        samples = _collect_samples(self._code_cache(dataset), self._activation_samples)
        calibrated = _calibrated_activation_engine(
            layout,
            energies,
            [codes for _, codes in samples[:settings.ACTIVATION_OUTPUT_CALIBRATION_SAMPLES]],
        )
        engine = calibrated.engine
        self._activation_metadata = _activation_metadata(calibrated.size, detectors=engine.detectors)
        embeddings = list(initial_embeddings[: len(samples)])
        start_index = len(embeddings)
        # Калибровочные стимулы уже активированы (§6.6): их эмбеддинги готовы, повторно не считаются.
        precomputed: dict[int, tuple[BitArray, int]] = {
            index: (embedding, count)
            for index, (embedding, count) in enumerate(zip(calibrated.embeddings, calibrated.active_counts))
            if index >= start_index
        }
        LOGGER.event(
            "activation.resume",
            section=OUTSIDE_ARTICLE,
            data={
                "completed": start_index,
                "target": len(samples),
                "precomputed": len(precomputed),
                "workers": settings.ACTIVATION_PARALLEL_WORKERS,
                "batch_size": settings.ACTIVATION_BATCH_SIZE,
            },
        )
        for batch_start in range(start_index, len(samples), settings.ACTIVATION_BATCH_SIZE):
            batch = samples[batch_start : batch_start + settings.ACTIVATION_BATCH_SIZE]
            pending = [
                (batch_start + offset, codes)
                for offset, (_, codes) in enumerate(batch)
                if batch_start + offset not in precomputed
            ]
            computed: dict[int, tuple[BitArray, int]] = {}
            if pending:
                results = engine.activate_many(
                    [codes for _, codes in pending],
                    max_workers=settings.ACTIVATION_PARALLEL_WORKERS,
                )
                computed = {
                    index: (result.embedding, len(result.active_detectors))
                    for (index, _), result in zip(pending, results)
                }
            for offset, (label, codes) in enumerate(batch):
                index = batch_start + offset
                embedding, active_count = precomputed.get(index) or computed[index]
                LOGGER.event(
                    "activation.sample",
                    section=STIMULUS_DETECTION,
                    data={
                        "index": index,
                        "sample_label": str(int(label)),
                        "stimuli": len(codes),
                        "active_detectors": active_count,
                        "embedding_bits": embedding.count(),
                        "from_calibration": index in precomputed,
                    },
                )
                embeddings.append(
                    EmbeddingRecord(
                        class_label=str(int(label)),
                        embedding_bits=embedding.to01(),
                    )
                )
            if (
                len(embeddings) % settings.ACTIVATION_CHECKPOINT_EVERY == 0
                or len(embeddings) == len(samples)
            ):
                self._checkpoint_store.save(
                    layout_path,
                    stages=stages,
                    target_samples=self._activation_samples,
                    embeddings=embeddings,
                )
        return embeddings

    def _save_embeddings(self, layout_path: str, embeddings: Sequence[EmbeddingRecord]) -> None:
        if not layout_path:
            raise ValueError("layout_path must be provided to save embeddings")
        self._embedding_store.save(layout_path, embeddings, metadata=self._activation_metadata)

    def _export_stage(
        self,
        layout: Layout,
        run_params: TunedLayoutParams,
        *,
        layout_path: str,
        save_json: bool,
        stages: dict[str, bool],
        embeddings: Sequence[EmbeddingRecord],
    ) -> None:
        """Последний этап конвейера: JSON и картинки раскладки, отметка этапа в чекпоинте (вне статьи)."""
        self._export_layout(layout, run_params, layout_path=layout_path, save_json=save_json)
        stages["export"] = True
        self._checkpoint_store.save(
            layout_path,
            stages=stages,
            target_samples=self._activation_samples,
            embeddings=embeddings,
        )

    def digit_codes(self, count: int) -> list[DigitCodes]:
        """Первые count цифр обучающего набора с их углопозициями и кодами — из общего кеша кодов."""
        return self._code_cache(self._build_dataset()).take_nonempty(count)

    def run(self) -> Layout:
        """Конвейер MNIST: раскладка, затем (без --layout-only) активация и эмбеддинги; возвращает раскладку."""
        if self._reuse_cache:
            cached_layout = self._try_load_cached_layout()
        else:
            cached_layout = None
            LOGGER.event(
                "layout.cache.rebuild",
                section=OUTSIDE_ARTICLE,
                data={"count": self.count, "note": "--rebuild: готовый кеш игнорируется, новый записывается поверх"},
            )
        dataset = self._build_dataset()
        if cached_layout is not None:
            layout, tuned_params, total_codes, layout_path = cached_layout
            angles = layout.hue_count()
            self._log_dataset(angles, total_codes)
            LOGGER.event(
                "mnist.encoding.skip",
                section=OUTSIDE_ARTICLE,
                data={
                    "count": self.count,
                    "label": "all" if self.label is None else int(self.label),
                    "layout_path": layout_path,
                    "points": total_codes,
                    "angles": angles,
                },
            )
            save_json = False
        else:
            payload, total_codes = _collect_codes(
                self._code_cache(dataset),
                self.count,
                self.encoder,
                self.extractor,
            )
            # Коды в порядке точек раскладки (группы по углу в порядке первого появления) — таблицей массивов.
            codes = payload.code_table()
            angles = codes.hue_count()
            self._log_dataset(angles, total_codes)
            tuned_params = self._tune_layout(total_codes)
            expected_pair_radius = _expected_pair_radius(total_codes, tuned_params)
            layout_path = _layout_json_path(self.count, expected_pair_radius)
            payload.add_metadata({settings.PLANNED_PAIR_RADIUS_KEY: expected_pair_radius})
            # Файл больше LAYOUT_CACHE_MAX_FILE_BYTES как кеш не читается: до раскладки пишется только заголовок.
            payload.save_base(layout_path, max_points_bytes=settings.LAYOUT_CACHE_MAX_FILE_BYTES)
            if not self._reuse_cache:
                self._checkpoint_store.reset(layout_path, target_samples=self._activation_samples)
            encoding_checkpoint = self._checkpoint_store.load(layout_path)
            encoding_stages = dict(encoding_checkpoint.get("stages", {}))
            encoding_stages["encoding"] = True
            self._checkpoint_store.save(
                layout_path,
                stages=encoding_stages,
                target_samples=self._activation_samples,
                embeddings=self._checkpoint_store.records(encoding_checkpoint),
            )
            layout = _run_layout(
                codes,
                tuned_params,
                count=self.count,
                total_codes=total_codes,
                colorizer=self.colorizer,
                expected_code_length=self.encoder.code_length,
                use_cache=self._reuse_cache,
            )
            save_json = True
        checkpoint = self._checkpoint_store.load(layout_path)
        stages = dict(checkpoint.get("stages", {}))
        stages["encoding"] = True
        stages["layout"] = True
        checkpoint_embeddings = self._checkpoint_store.records(checkpoint)
        self._checkpoint_store.save(
            layout_path,
            stages=stages,
            target_samples=self._activation_samples,
            embeddings=checkpoint_embeddings,
        )
        if self._layout_only:
            # Эмбеддинги чекпоинта остаются как были: обычный запуск возьмёт эту раскладку из кеша
            # и продолжит с активации.
            LOGGER.event(
                "activation.skip",
                section=OUTSIDE_ARTICLE,
                data={
                    "path": layout_path,
                    "reason": "layout_only",
                    "embeddings": len(checkpoint_embeddings),
                    "note": "--layout-only: активация и эмбеддинги не считаются",
                },
            )
            self._export_stage(
                layout,
                tuned_params,
                layout_path=layout_path,
                save_json=save_json,
                stages=stages,
                embeddings=checkpoint_embeddings,
            )
            return layout
        try:
            cached_embeddings = self._embedding_store.load(layout_path)
        except FileNotFoundError:
            cached_embeddings = []
        resume_embeddings = (
            checkpoint_embeddings
            if len(checkpoint_embeddings) >= len(cached_embeddings)
            else cached_embeddings
        )
        if len(resume_embeddings) >= self._activation_samples:
            embeddings = resume_embeddings[: self._activation_samples]
            LOGGER.event(
                "activation.embedding.skip",
                section=OUTSIDE_ARTICLE,
                data={
                    "path": layout_path,
                    "embeddings": len(embeddings),
                    "reason": "target_present",
                },
            )
        else:
            embeddings = self._run_activation(
                layout,
                dataset,
                layout_path=layout_path,
                stages=stages,
                initial_embeddings=resume_embeddings,
            )
            self._save_embeddings(layout_path, embeddings)
        stages["activation"] = True
        self._checkpoint_store.save(
            layout_path,
            stages=stages,
            target_samples=self._activation_samples,
            embeddings=embeddings,
        )
        self._export_stage(
            layout,
            tuned_params,
            layout_path=layout_path,
            save_json=save_json,
            stages=stages,
            embeddings=embeddings,
        )
        return layout
