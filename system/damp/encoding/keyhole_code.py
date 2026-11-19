"""Этап 2: код окна внимания — «что и где» одним кодом (продолжение статьи, П.4–П.7; детекторы места — §4.3)."""

from __future__ import annotations

from damp import settings
from damp.encoding.keyhole import KeyholePieces
from damp.encoding.damp_encoder import Encoder, PlaneDimension, plane_detector_layers
from damp.encoding.elements import LadderElements
from damp.encoding.position_binding import PositionBindingEncoder, KeyholeCodes


class KeyholeCoder:
    """Коды окон уровня: биты места через OR и код содержания под маской своего места (П.4–П.7).

    Детекторы позиции — круги на плоскости окна одного на все уровни кодера места (§4.3); у каждого уровня
    свои случайные маски (зерно — номер уровня).
    """

    def __init__(self) -> None:
        self._place: Encoder | None = None

    @property
    def place(self) -> Encoder:
        """Кодер места — как кодер угла (§4.2, §4.3, §4.5): круги на плоскости окна слоями от широких к узким, число
        кругов по стороне растёт геометрически от LADDER_BINDING_PLACE_WIDEST до LADDER_BINDING_PLACE_NARROWEST, у
        каждого слоя своя цветовая полоса бит (§3.2, §4.2.1). Слои, перекрытие и длина кода подобраны под плотность
        кода угла (автор 2026-11-19)."""
        if self._place is None:
            extent = float(settings.LADDER_PLACE_EXTENT)
            overlap = float(settings.LADDER_BINDING_PLACE_OVERLAP)
            # Радиус круга слоя из n кругов по стороне — 1 / (2 · n · (1 − перекрытие)) стороны (plane_detector_layers).
            radius = lambda count: 1.0 / (2.0 * float(count) * (1.0 - overlap))  # noqa: E731
            self._place = Encoder(
                PlaneDimension(
                    "Place",
                    ((-extent, extent), (-extent, extent)),
                    plane_detector_layers(
                        layers_count=settings.LADDER_BINDING_PLACE_LAYERS,
                        widest_radius=radius(settings.LADDER_BINDING_PLACE_WIDEST),
                        narrowest_radius=radius(settings.LADDER_BINDING_PLACE_NARROWEST),
                        overlap=overlap,
                    ),
                    code_length=settings.LADDER_BINDING_PLACE_CODE_LENGTH,
                    layer_bands=True,
                ),
                random_seed=settings.LADDER_RANDOM_SEED,
                layer_phase=settings.ENCODER_LAYER_PHASE,
            )
        return self._place

    def encoder(self, number: int) -> PositionBindingEncoder:
        """Кодер окон уровня number (П.5)."""
        return PositionBindingEncoder(
            self.place,
            pooling=settings.LADDER_BINDING_POOLING,
            mask=settings.LADDER_BINDING_MASK,
            mask_bits=settings.LADDER_BINDING_MASK_BITS,
            target_density=settings.LADDER_BINDING_TARGET_DENSITY,
            saturation=settings.LADDER_BINDING_SATURATION,
            max_length=settings.LADDER_BINDING_LENGTH_MAX,
            seed=settings.LADDER_RANDOM_SEED + number,
            use_gpu=settings.ACTIVATION_USE_GPU,
            name=f"level{number}",
            concepts=settings.LADDER_BINDING_CONCEPTS,  # П.7: окно объединяет около 5–7 понятий
        )

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
            encoder = self.encoder(number)
            if fixed:
                encoder.fix_content_length(fixed)
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
        found = CACHE.run("coding.keyholes", inputs, compute, modules=("damp.encoding.keyhole_code",))
        return KeyholeCodes(
            packed=found["packed"], code_length=int(found["code_length"]),
            place_length=int(found["place_length"]), content_length=int(found["content_length"]),
        )
