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
        """Кодер места: круги на плоскости окна, от широких к узким (§4.3)."""
        if self._place is None:
            extent = float(settings.LADDER_PLACE_EXTENT)
            self._place = Encoder(
                PlaneDimension(
                    "Place",
                    ((-extent, extent), (-extent, extent)),
                    plane_detector_layers(
                        layers_count=settings.LADDER_BINDING_PLACE_LAYERS,
                        widest_radius=settings.LADDER_BINDING_PLACE_RADIUS_MAX / (2.0 * extent),
                        narrowest_radius=settings.LADDER_BINDING_PLACE_RADIUS_MIN / (2.0 * extent),
                        overlap=settings.LADDER_BINDING_PLACE_OVERLAP,
                    ),
                    code_length=settings.LADDER_BINDING_PLACE_CODE_LENGTH,
                    layer_bands=False,
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
        )

    def encode(self, elements: LadderElements, pieces: KeyholePieces, *, number: int, unit: float, taught: int) -> KeyholeCodes:
        """Код каждого окна: его элементы со своими местами в координатах окна (П.4, П.5).

        unit — шаг мест элементов в долях радиуса окна; taught — окон на обучающих картинках (они идут первыми,
        по ним подбирается длина кода, §6.6). Результат — из кеша шагов, если такой вход уже был.
        """
        from damp.cache import CACHE

        what_index = elements.what_index[pieces.element]
        strength = elements.power()[pieces.element]
        calibration = max(1, min(int(taught), int(settings.LADDER_CALIBRATION_SETS)))

        def compute() -> dict[str, object]:
            codes = self.encoder(number).encode(
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
        }
        found = CACHE.run("coding.keyholes", inputs, compute, modules=("damp.encoding.keyhole_code",))
        return KeyholeCodes(
            packed=found["packed"], code_length=int(found["code_length"]),
            place_length=int(found["place_length"]), content_length=int(found["content_length"]),
        )
