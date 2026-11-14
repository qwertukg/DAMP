"""Этап 2 для команды layout: кодер стимула (угол, X, Y) — общий код (§2.2.3, §4.2.1, §4.4),
коды цифр набора и их кеш на диске."""

from __future__ import annotations
import concurrent.futures
import hashlib
import os
import time
from dataclasses import dataclass
from typing import ClassVar, Sequence

import numpy as np

from damp import settings
from damp.article_refs import OUTSIDE_ARTICLE
from damp.encoding.bitarray import BitArray
from damp.encoding.damp_encoder import ClosedDimension, Encoder, EncoderCodeColorizer, OpenedDimension, geometric_detector_layers
from damp.layout.code_table import LayoutCodeTable
from damp.layout.payload import LayoutPayloadBuilder
from damp.logging import LOGGER
from damp.stimuli.MnistSobelAngleMap import MnistSobelAngleMap
from damp.stimuli.mnist import _build_extractor, _stimulus_metadata


def _build_encoder() -> Encoder:
    """Кодер MNIST-стимула (угол, X, Y): общий код или конкатенация сегментов (§2.2.3, §4.2.1, §4.4)."""
    angle_layers = geometric_detector_layers(
        layers_count=settings.MNIST_ANGLE_LAYERS,
        widest_detectors=settings.MNIST_ANGLE_WIDEST_DETECTORS,
        narrowest_detectors=settings.MNIST_ANGLE_NARROWEST_DETECTORS,
        overlap=settings.MNIST_ANGLE_OVERLAP,
    )
    position_layers = geometric_detector_layers(
        layers_count=settings.MNIST_POSITION_LAYERS,
        widest_detectors=settings.MNIST_POSITION_WIDEST_DETECTORS,
        narrowest_detectors=settings.MNIST_POSITION_NARROWEST_DETECTORS,
        overlap=settings.MNIST_POSITION_OVERLAP,
    )
    angle_length = settings.ENCODER_SHARED_CODE_LENGTH if settings.ENCODER_SHARED_CODE else settings.MNIST_ANGLE_CODE_LENGTH
    position_length = settings.ENCODER_SHARED_CODE_LENGTH if settings.ENCODER_SHARED_CODE else settings.MNIST_POSITION_CODE_LENGTH
    encoder = Encoder(
        ClosedDimension(
            "Angle",
            (settings.MNIST_ANGLE_MIN, settings.MNIST_ANGLE_MAX),
            angle_layers,
            code_length=angle_length,
        ),
        OpenedDimension(
            "X",
            (settings.MNIST_POSITION_MIN, settings.MNIST_POSITION_MAX),
            position_layers,
            code_length=position_length,
            edge_detectors=settings.MNIST_POSITION_EDGE_DETECTORS,
        ),
        OpenedDimension(
            "Y",
            (settings.MNIST_POSITION_MIN, settings.MNIST_POSITION_MAX),
            position_layers,
            code_length=position_length,
            edge_detectors=settings.MNIST_POSITION_EDGE_DETECTORS,
        ),
        random_seed=settings.ENCODER_RANDOM_SEED,
        log_every=settings.ENCODER_LOG_EVERY,
        layer_phase=settings.ENCODER_LAYER_PHASE,
        shared_code=settings.ENCODER_SHARED_CODE,
    )
    # §4.5, §5.13.1: у каждой координаты сходство кодов должно убывать с расстоянием
    # на всём диапазоне, иначе раскладка не воспроизведёт её топологию.
    for dimension in encoder.dimensions:
        step = None if dimension.closed else settings.MNIST_POSITION_STEP
        # Encoder.long_range_order сам логирует профиль и предупреждение (§4.5, §5.13.1).
        encoder.long_range_order(dimension.title, step=step)
    return encoder


def _build_colorizer() -> EncoderCodeColorizer:
    """Псевдоцвет всего кода: младший бит красный, старший фиолетовый (§5.14)."""
    return EncoderCodeColorizer(
        min_hue=settings.LAYOUT_PSEUDO_COLOR_MIN_HUE,
        max_hue=settings.LAYOUT_PSEUDO_COLOR_MAX_HUE,
    )


@dataclass(frozen=True)
class DigitCodes:
    """Одна цифра набора: метка, углы её углопозиций и их коды (§4.4, §4.6).

    Коды хранятся матрицей бит, упакованной по 8 в байт (k × ⌈L/8⌉): объект BitArray на каждый код для
    больших наборов не помещается в память. codes строит объекты по требованию (активация, декодер).
    """

    label: int
    angles: tuple[float, ...]
    packed: "np.ndarray"
    code_length: int

    @property
    def count(self) -> int:
        """Число углопозиций цифры."""
        return len(self.angles)

    @property
    def codes(self) -> tuple[BitArray, ...]:
        bits = np.unpackbits(self.packed, axis=1, count=self.code_length)
        return tuple(
            BitArray.from_bits(bits[row].tobytes(), np.flatnonzero(bits[row]).tolist()) for row in range(bits.shape[0])
        )


class DigitEncodingWorker:
    """Кодирование цифр в рабочем процессе: свой кодер и своё извлечение углопозиций (вне статьи).

    Результат — те же углопозиции и коды, что в основном процессе (тот же код, те же константы); журнал
    рабочего процесса выключен, сводку ведёт основной процесс.
    """

    _encoder: ClassVar[Encoder | None] = None
    _extractor: ClassVar[MnistSobelAngleMap | None] = None

    @classmethod
    def initialize(cls) -> None:
        LOGGER.mute()
        cls._encoder = _build_encoder()
        cls._extractor = _build_extractor()

    @classmethod
    def encode(cls, task: list[tuple["np.ndarray", int]]) -> list[tuple[int, tuple[float, ...], "np.ndarray"]]:
        """Цифры задания (изображение, метка) → (метка, углы углопозиций, упакованные коды)."""
        if cls._encoder is None or cls._extractor is None:
            cls.initialize()
        result = []
        for img, digit in task:
            item = StimulusCodeCache.encode_digit(cls._encoder, cls._extractor, img, digit, log=False)
            result.append((item.label, item.angles, item.packed))
        return result


class EncodedCodesFile:
    """Коды цифр набора на диске (вне статьи): повторный запуск с тем же кодером не кодирует набор заново.

    Файл хранит первые D цифр набора в его порядке: метки, число углопозиций каждой цифры, углы и упакованные
    коды — те же байты, что дало кодирование. Имя файла содержит отпечаток всего, от чего зависят углопозиции
    (§4.4) и коды (§4.6): подпись кодера, значения настроек MNIST_* и ENCODER_*, тексты модулей
    кодирования и извлечения стимулов, часть набора и отбор по метке. Изменилось что-то из этого — имя другое,
    файл не находится, набор кодируется заново.
    """

    # Модули, от текста которых зависят углопозиции и коды.
    SOURCES = (
        os.path.join("damp", "stimuli", "MnistSobelAngleMap.py"),
        os.path.join("damp", "encoding", "bitarray.py"),
        os.path.join("damp", "encoding", "damp_encoder.py"),
        os.path.join("damp", "layout", "code_table.py"),
    )
    CONSTANT_PREFIXES = ("MNIST_", "ENCODER_")

    def __init__(self, directory: str, *, encoder: Encoder, split: str, label: int | None) -> None:
        self._code_length = int(encoder.code_length)
        self._fingerprint = self._compute_fingerprint(encoder, split, label)
        suffix = "" if label is None else f"-label{int(label)}"
        self._path = os.path.join(directory, f"codes-{split}{suffix}-{self._fingerprint[:16]}.npz")

    @classmethod
    def _compute_fingerprint(cls, encoder: Encoder, split: str, label: int | None) -> str:
        digest = hashlib.sha256()
        digest.update(f"{encoder.signature}|{encoder.code_length}|{split}|{label}".encode("utf-8"))
        constants = vars(settings)
        for name in sorted(constants):
            if name.startswith(cls.CONSTANT_PREFIXES):
                digest.update(f"{name}={constants[name]!r};".encode("utf-8"))
        # Пути SOURCES — от папки системного слоя (system/): system/damp/encoding/mnist_codes.py → system.
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for source in cls.SOURCES:
            with open(os.path.join(root, source), "rb") as fp:
                digest.update(fp.read())
        return digest.hexdigest()

    @property
    def path(self) -> str:
        return self._path

    def load(self, digits: int) -> "list[DigitCodes] | None":
        """Первые digits цифр набора из файла; None — файла нет, он от другого кодера или цифр в нём меньше."""
        if not os.path.exists(self._path):
            return None
        started = time.perf_counter()
        try:
            with np.load(self._path, allow_pickle=False) as data:
                if str(data["fingerprint"]) != self._fingerprint or int(data["code_length"]) != self._code_length:
                    return None
                counts = data["counts"]
                if counts.size < digits:
                    return None
                counts = counts[:digits].astype(np.int64)
                total = int(counts.sum())
                labels = data["labels"][:digits].tolist()
                angles = data["angles"][:total]
                packed = data["packed"][:total]
        except Exception as exc:  # noqa: BLE001
            LOGGER.event(
                "mnist.encoding.cache.error",
                section=OUTSIDE_ARTICLE,
                data={"path": self._path, "error": str(exc), "stage": "read"},
            )
            return None
        stops = np.cumsum(counts).tolist()
        items: list[DigitCodes] = []
        start = 0
        for label, stop in zip(labels, stops):
            items.append(
                DigitCodes(
                    label=int(label),
                    angles=tuple(angles[start:stop].tolist()),
                    packed=packed[start:stop],
                    code_length=self._code_length,
                )
            )
            start = stop
        LOGGER.event(
            "mnist.encoding.cache.load",
            section=OUTSIDE_ARTICLE,
            data={
                "path": self._path,
                "digits": digits,
                "codes": total,
                "seconds": time.perf_counter() - started,
                "note": "углопозиции (§4.4) и коды (§4.6) цифр взяты из файла: кодер, константы MNIST_* и ENCODER_* "
                "и тексты модулей кодирования — те же, что при записи файла",
            },
        )
        return items

    def save(self, items: "Sequence[DigitCodes]") -> None:
        """Записать цифры набора (в его порядке), если в файле их меньше."""
        digits = len(items)
        try:
            if os.path.exists(self._path):
                with np.load(self._path, allow_pickle=False) as data:
                    if str(data["fingerprint"]) == self._fingerprint and int(data["counts"].size) >= digits:
                        return
            started = time.perf_counter()
            counts = np.asarray([item.count for item in items], dtype=np.int32)
            total = int(counts.sum())
            packed = np.zeros((total, -(-self._code_length // 8)), dtype=np.uint8)
            angles = np.zeros(total, dtype=np.float64)
            start = 0
            for item in items:
                stop = start + item.count
                if item.count:
                    packed[start:stop] = item.packed
                    angles[start:stop] = item.angles
                start = stop
            directory = os.path.dirname(self._path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            temporary = self._path + ".tmp.npz"
            np.savez(
                temporary,
                fingerprint=np.asarray(self._fingerprint),
                code_length=np.asarray(self._code_length),
                labels=np.asarray([item.label for item in items], dtype=np.int16),
                counts=counts,
                angles=angles,
                packed=packed,
            )
            os.replace(temporary, self._path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.event(
                "mnist.encoding.cache.error",
                section=OUTSIDE_ARTICLE,
                data={"path": self._path, "error": str(exc), "stage": "write"},
            )
            return
        LOGGER.event(
            "mnist.encoding.cache.save",
            section=OUTSIDE_ARTICLE,
            data={"path": self._path, "digits": digits, "codes": total, "seconds": time.perf_counter() - started},
        )


class StimulusCodeCache:
    """Углопозиции и коды цифр считаются один раз: раскладка, калибровка и база эмбеддингов берут их отсюда.

    Цифры идут в порядке набора; все углопозиции цифры кодируются одним батчем (Encoder.encode_matrix).
    Цифры без углопозиций остаются в списке пустыми: раскладка учитывает их в счётчике цифр, а база
    эмбеддингов пропускает — как и раньше, когда каждый потребитель читал набор заново (вне статьи).
    Большой набор (от ENCODING_PARALLEL_MIN_DIGITS недостающих цифр) кодируется рабочими процессами.
    """

    def __init__(
        self,
        dataset,
        encoder: Encoder,
        extractor: MnistSobelAngleMap,
        *,
        label: int | None,
        store: "EncodedCodesFile | None" = None,
    ) -> None:
        self._iterator = iter(dataset)
        self._encoder = encoder
        self._extractor = extractor
        self._label = label
        self._items: list[DigitCodes] = []
        self._exhausted = False
        # Файл кодов набора на диске (большие наборы); _skip — сколько цифр набора взято из файла и ещё
        # не пропущено в самом наборе (набор читается дальше, только если понадобятся следующие цифры).
        self._store = store
        self._skip = 0

    @staticmethod
    def encode_digit(
        encoder: Encoder, extractor: MnistSobelAngleMap, img: "np.ndarray", digit: int, *, log: bool = True
    ) -> DigitCodes:
        """Углопозиции цифры (§4.4) и их коды (§4.6) одним батчем; log — записи журнала кодера с картинкой цифры."""
        measurements = extractor.extract(img, digit)[int(digit)]
        packed = np.zeros((0, -(-encoder.code_length // 8)), dtype=np.uint8)
        if measurements:
            bits = encoder.encode_matrix(
                [(float(a), float(x), float(y)) for a, x, y in measurements],
                log_image=img if log else None,
                log_label=int(digit) if log else None,
                log_measurements=measurements if log else None,
            )
            packed = LayoutCodeTable.pack_bits(bits)
        return DigitCodes(
            label=int(digit),
            angles=tuple(float(a) for a, _, _ in measurements),
            packed=packed,
            code_length=encoder.code_length,
        )

    def _next_image(self) -> tuple["np.ndarray", int] | None:
        """Следующая цифра набора с нужной меткой: изображение и метка; None — набор кончился."""
        for img_tensor, digit in self._iterator:
            if self._label is not None and int(digit) != self._label:
                continue
            if self._skip:
                # Эта цифра уже взята из файла кодов.
                self._skip -= 1
                continue
            return img_tensor.squeeze(0).numpy(), int(digit)
        self._exhausted = True
        return None

    def _load_next(self) -> bool:
        image = self._next_image()
        if image is None:
            return False
        self._items.append(self.encode_digit(self._encoder, self._extractor, image[0], image[1]))
        return True

    @staticmethod
    def _worker_count() -> int:
        if settings.ENCODING_WORKERS is not None:
            return max(1, int(settings.ENCODING_WORKERS))
        return max(1, min(settings.ENCODING_WORKERS_MAX, (os.cpu_count() or 1) - 1))

    def _load_parallel(self, count: int) -> None:
        """Следующие count цифр набора — рабочими процессами, в порядке набора."""
        workers = self._worker_count()
        if workers <= 1:
            return
        started = time.perf_counter()
        tasks: list[list[tuple["np.ndarray", int]]] = []
        task: list[tuple["np.ndarray", int]] = []
        digits = 0
        while digits < count:
            image = self._next_image()
            if image is None:
                break
            task.append(image)
            digits += 1
            if len(task) >= settings.ENCODING_CHUNK_DIGITS:
                tasks.append(task)
                task = []
        if task:
            tasks.append(task)
        if not tasks:
            return
        LOGGER.event(
            "mnist.encoding.parallel",
            section=OUTSIDE_ARTICLE,
            data={
                "digits": digits,
                "workers": workers,
                "chunk_digits": settings.ENCODING_CHUNK_DIGITS,
                "note": "углопозиции (§4.4) и коды (§4.6) этих цифр считают рабочие процессы тем же кодером; "
                "записи журнала кодера по ним не выводятся",
            },
        )
        code_length = self._encoder.code_length
        codes = 0
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)), initializer=DigitEncodingWorker.initialize
        ) as executor:
            for encoded in executor.map(DigitEncodingWorker.encode, tasks):
                for label, angles, packed in encoded:
                    self._items.append(DigitCodes(label=label, angles=angles, packed=packed, code_length=code_length))
                    codes += len(angles)
        LOGGER.event(
            "mnist.encoding.parallel.done",
            section=OUTSIDE_ARTICLE,
            data={"digits": digits, "codes": codes, "seconds": time.perf_counter() - started},
        )

    def take(self, count: int) -> list[DigitCodes]:
        """Первые count цифр набора (включая цифры без углопозиций)."""
        store = self._store if count >= settings.ENCODING_CODES_CACHE_MIN_DIGITS else None
        if store is not None and not self._items and not self._exhausted:
            cached = store.load(count)
            if cached is not None:
                self._items = cached
                self._skip = len(cached)
        missing = count - len(self._items)
        encoded = False
        if missing >= settings.ENCODING_PARALLEL_MIN_DIGITS and not self._exhausted:
            self._load_parallel(missing)
            encoded = True
        while len(self._items) < count and not self._exhausted:
            self._load_next()
            encoded = True
        if store is not None and encoded:
            store.save(self._items[:count])
        return self._items[:count]

    def take_nonempty(self, count: int) -> list[DigitCodes]:
        """Первые count цифр, у которых есть хотя бы одна углопозиция."""
        while sum(1 for item in self._items if item.count) < count and not self._exhausted:
            self._load_next()
        result = [item for item in self._items if item.count]
        return result[:count]


def _collect_codes(
    cache: StimulusCodeCache,
    count: int,
    encoder: Encoder,
    extractor: MnistSobelAngleMap,
) -> tuple[LayoutPayloadBuilder, int]:
    payload = LayoutPayloadBuilder(
        similarity=None,
        lambda_threshold=None,
        eta=None,
        metadata={settings.ENCODER_SIGNATURE_KEY: encoder.signature, **_stimulus_metadata(extractor)},
    )
    total_codes = 0
    for item in cache.take(count):
        # Коды цифры уходят в таблицу массивами, без объекта на каждый код.
        payload.add_packed(label=item.label, hues=item.angles, packed=item.packed, code_length=item.code_length)
        total_codes += item.count
    return payload, total_codes


def _collect_samples(cache: StimulusCodeCache, sample_count: int) -> list[tuple[int, list[BitArray]]]:
    return [(item.label, list(item.codes)) for item in cache.take_nonempty(sample_count)]
