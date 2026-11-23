"""Этап 2: код окна внимания — «что и где» одним кодом (продолжение статьи, П.4–П.7; детекторы места — §4.3)."""

from __future__ import annotations

from damp import settings
from damp.encoding.keyhole import KeyholePieces
from damp.encoding.damp_encoder import Encoder, PlaneDimension, plane_detector_layers
from damp.encoding.elements import LadderElements
from damp.encoding.position_binding import PositionBindingEncoder, KeyholeCodes


class KeyholeCoder:
    """Коды окон уровня: биты места через OR и код содержания под маской своего места (П.4–П.7).

    Детекторы позиции — круги на плоскости окна, одни и те же на всех уровнях (§4.3); длина сегмента места — своя у
    уровня (LADDER_BINDING_PLACE_CODE_LENGTH), у каждого уровня свои случайные маски (зерно — номер уровня).
    """

    def __init__(self) -> None:
        self._places: dict[int, Encoder] = {}
        self._fixed: dict[tuple, PositionBindingEncoder] = {}  # кодеры с заданной длиной содержания: один на уровень, а не на окно

    def _signature(self, number: int) -> tuple:
        """Всё, из чего складывается кодер уровня number (настройки читаются здесь прямо, чтобы они входили в ключ кеша шагов)."""
        return (
            int(number), repr(settings.LADDER_BINDING_PLACE_CODE_LENGTH), repr(settings.LADDER_PLACE_EXTENT),
            repr(settings.LADDER_BINDING_PLACE_OVERLAP), repr(settings.LADDER_BINDING_PLACE_LAYERS),
            repr(settings.LADDER_BINDING_PLACE_WIDEST), repr(settings.LADDER_BINDING_PLACE_NARROWEST),
            repr(settings.ENCODER_LAYER_PHASE), repr(settings.LADDER_RANDOM_SEED), repr(settings.LADDER_BINDING_ENUMERATE_FROM),
            repr(settings.LADDER_BINDING_POOLING), repr(settings.LADDER_BINDING_MASK), repr(settings.LADDER_BINDING_MASK_BITS),
            repr(settings.LADDER_BINDING_TARGET_DENSITY), repr(settings.LADDER_BINDING_SATURATION),
            repr(settings.LADDER_BINDING_LENGTH_MAX), repr(settings.ACTIVATION_USE_GPU), repr(settings.LADDER_BINDING_CONCEPTS),
            repr(self.place_bits(number)), repr(self.content_layers(number)),
        )

    def fixed_encoder(self, number: int, length: int) -> PositionBindingEncoder:
        """Кодер окон уровня number с уже подобранной длиной содержания length. Он ничего не копит между вызовами
        (длина задана, маски и приоритеты зависят только от номера уровня и длины), поэтому один кодер служит всем
        окнам и картинкам уровня — прежде новый кодер строился на каждую картинку (1.6 с на проход порогов и проверочных)."""
        key = (*self._signature(number), int(length))
        encoder = self._fixed.get(key)
        if encoder is None:
            encoder = self.encoder(number)
            encoder.fix_content_length(int(length))
            self._fixed[key] = encoder
        return encoder

    @staticmethod
    def place_length(number: int) -> int:
        """Длина сегмента места окна уровня number: LADDER_BINDING_PLACE_CODE_LENGTH — число на все уровни либо список
        по уровням 0, 1, 2, … (последнее значение — на всех уровнях выше). Биты места всех элементов окна входят в код
        через OR (П.5), поэтому чем больше элементов в окне, тем длиннее сегмент при той же плотности (§2)."""
        value = settings.LADDER_BINDING_PLACE_CODE_LENGTH
        if isinstance(value, (list, tuple)):
            return int(value[min(max(int(number), 0), len(value) - 1)])
        return int(value)

    def place(self, number: int) -> Encoder:
        """Кодер места окон уровня number — как кодер угла (§4.2, §4.3, §4.5): круги на плоскости окна слоями от
        широких к узким, число кругов по стороне растёт геометрически от LADDER_BINDING_PLACE_WIDEST до
        LADDER_BINDING_PLACE_NARROWEST, у каждого слоя своя цветовая полоса бит (§3.2, §4.2.1). Слои и перекрытие
        подобраны под плотность кода угла (автор 2026-11-19), длина — под плотность кода окна уровня (§2)."""
        length = self.place_length(number)
        if length not in self._places:
            extent = float(settings.LADDER_PLACE_EXTENT)
            overlap = float(settings.LADDER_BINDING_PLACE_OVERLAP)
            # Радиус круга слоя из n кругов по стороне — 1 / (2 · n · (1 − перекрытие)) стороны (plane_detector_layers).
            radius = lambda count: 1.0 / (2.0 * float(count) * (1.0 - overlap))  # noqa: E731
            self._places[length] = Encoder(
                PlaneDimension(
                    "Place",
                    ((-extent, extent), (-extent, extent)),
                    plane_detector_layers(
                        layers_count=settings.LADDER_BINDING_PLACE_LAYERS,
                        widest_radius=radius(settings.LADDER_BINDING_PLACE_WIDEST),
                        narrowest_radius=radius(settings.LADDER_BINDING_PLACE_NARROWEST),
                        overlap=overlap,
                    ),
                    code_length=length,
                    layer_bands=True,
                ),
                random_seed=settings.LADDER_RANDOM_SEED,
                layer_phase=settings.ENCODER_LAYER_PHASE,
            )
        return self._places[length]

    @staticmethod
    def pooling(number: int) -> str:
        """Пулинг окон уровня number: перечисляющий с уровня LADDER_BINDING_ENUMERATE_FROM (П.5, задача #17), ниже —
        LADDER_BINDING_POOLING."""
        return "enumerate" if 0 < int(settings.LADDER_BINDING_ENUMERATE_FROM) <= int(number) else settings.LADDER_BINDING_POOLING

    @staticmethod
    def pooling_tie(number: int) -> float:
        """Допуск при выборе элемента круга места (план v8.3): с уровня LADDER_BINDING_POOLING_TIE_FROM (0 — нет) в круг идёт
        и второй элемент области, если его сила не меньше (1 − LADDER_BINDING_POOLING_TIE) от лучшего; 0.0 — выкл."""
        return float(settings.LADDER_BINDING_POOLING_TIE) if 0 < int(settings.LADDER_BINDING_POOLING_TIE_FROM) <= int(number) else 0.0

    @classmethod
    def level_settings(cls, number: int) -> dict[str, object]:
        """Настройки кода окна, заданные по уровням или действующие с уровня, — их значение на уровне number: длина
        сегмента места, бит места на круг, слои содержания, пулинг. Кодер (place, encoder) берёт их отсюда же, и ровно
        они идут в ключ кеша кодов окон уровня (С3): правка значения одного уровня не сбрасывает коды остальных."""
        return {
            "LADDER_BINDING_PLACE_CODE_LENGTH": cls.place_length(number),
            "LADDER_BINDING_PLACE_BITS": cls.place_bits(number),
            "LADDER_BINDING_CONTENT_LAYERS": cls.content_layers(number),
            "LADDER_BINDING_ENUMERATE_FROM": None,
            "LADDER_BINDING_POOLING": cls.pooling(number),
            "LADDER_BINDING_POOLING_TIE_FROM": None,
            "LADDER_BINDING_POOLING_TIE": cls.pooling_tie(number),
        }

    def encoder(self, number: int) -> PositionBindingEncoder:
        """Кодер окон уровня number (П.5)."""
        return PositionBindingEncoder(
            self.place(number),
            # Перечисляющий пулинг с уровня LADDER_BINDING_ENUMERATE_FROM (П.5, задача #17); 0 — LADDER_BINDING_POOLING.
            pooling=self.pooling(number),
            mask=settings.LADDER_BINDING_MASK,
            mask_bits=settings.LADDER_BINDING_MASK_BITS,
            target_density=settings.LADDER_BINDING_TARGET_DENSITY,
            saturation=settings.LADDER_BINDING_SATURATION,
            max_length=settings.LADDER_BINDING_LENGTH_MAX,
            seed=settings.LADDER_RANDOM_SEED + number,
            use_gpu=settings.ACTIVATION_USE_GPU,
            name=f"level{number}",
            concepts=settings.LADDER_BINDING_CONCEPTS,  # П.7: окно объединяет около 5–7 понятий
            place_bits=self.place_bits(number),
            content_layers=self.content_layers(number),
            pooling_tie=self.pooling_tie(number),
        )

    @staticmethod
    def content_layers(number: int) -> int | None:
        """Сколько широких слоёв кругов места кладут содержание у окон уровня number (Б4 стратегии v3): LADDER_BINDING_CONTENT_LAYERS —
        число на все уровни либо список по уровням 0, 1, 2, … (последнее — на всех выше); 0 — все слои."""
        # Прямое обращение settings.ИМЯ нужно кешу шагов: по нему настройка входит в ключ шага «коды окон».
        value = settings.LADDER_BINDING_CONTENT_LAYERS if hasattr(settings, "LADDER_BINDING_CONTENT_LAYERS") else 0
        if isinstance(value, (list, tuple)):
            value = value[min(max(int(number), 0), len(value) - 1)]
        return None if int(value) <= 0 else int(value)

    @staticmethod
    def place_bits(number: int) -> int:
        """Бит места на круг у окон уровня number: LADDER_BINDING_PLACE_BITS — число на все уровни либо список по уровням
        0, 1, 2, … (последнее — на всех выше). Чем больше, тем весомее место в коде окна против содержания."""
        # Прямое обращение settings.ИМЯ нужно кешу шагов: по нему настройка входит в ключ шага «коды окон».
        value = settings.LADDER_BINDING_PLACE_BITS if hasattr(settings, "LADDER_BINDING_PLACE_BITS") else 1
        if isinstance(value, (list, tuple)):
            return int(value[min(max(int(number), 0), len(value) - 1)])
        return int(value)

    def encode(
        self, elements: LadderElements, pieces: KeyholePieces, *, number: int, unit: float, taught: int,
        content_length: int | None = None,
    ) -> KeyholeCodes:
        """Код каждого окна: его элементы со своими местами в координатах окна (П.4, П.5).

        unit — шаг мест элементов в долях радиуса окна; taught — окон на обучающих картинках (они идут первыми,
        по ним подбирается длина кода, §6.6); content_length — длина сегмента содержания, уже подобранная по
        обучающим (обученный уровень, новая картинка): тогда она не подбирается. Результат — из кеша шагов, если
        такой вход уже был.
        """
        from damp.cache import CACHE

        what_index = elements.what_index[pieces.element]
        strength = elements.power()[pieces.element]
        calibration = max(1, min(int(taught), int(settings.LADDER_CALIBRATION_SETS)))
        fixed = 0 if content_length is None else int(content_length)

        def compute() -> dict[str, object]:
            if fixed:
                encoder = self.fixed_encoder(number, fixed)
            else:
                encoder = self.encoder(number)
            codes = encoder.encode(
                elements.what_packed,
                elements.what_length,
                what_index,
                pieces.offsets,
                pieces.offset,
                pieces.piece,
                pieces.count,
                unit=float(unit),
                strength=strength,
                calibration=calibration,
            )
            return {
                "packed": codes.packed, "code_length": int(codes.code_length),
                "place_length": int(codes.place_length), "content_length": int(codes.content_length),
            }

        inputs = {
            "what": elements.what_packed, "what_length": elements.what_length, "what_index": what_index,
            "offsets": pieces.offsets, "offset": pieces.offset, "piece": pieces.piece, "keyholes": pieces.count,
            "unit": float(unit), "strength": strength, "calibration": calibration, "number": int(number),
            "content_length": fixed,
        }
        found = CACHE.run("coding.keyholes", inputs, compute, modules=("damp.encoding.keyhole_code",),
                          level_settings=self.level_settings(number))
        return KeyholeCodes(
            packed=found["packed"], code_length=int(found["code_length"]),
            place_length=int(found["place_length"]), content_length=int(found["content_length"]),
        )
