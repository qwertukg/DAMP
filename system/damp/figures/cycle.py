"""Рисунки этапов пайплайна по одному прогону цикла (данные — LadderRecorder).

Этапы — как в README: 1 стимулы, 2 первичное кодирование, 3 раскладка, 4 детекторы и активации, 5 эмбеддинги,
6 цикл, 7 распознавание. Пример — цифра-образец (автор проекта всегда берёт 5): окно в середине цифры на каждом
уровне. Всё, что рисунок показывает о счёте (зажжённые точки, сработавшие детекторы), пересчитывается здесь по
правилам статьи и сверяется с записанным в прогоне; расхождение пишется в лог.
"""
from __future__ import annotations

import math
import os
import shutil
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

from damp.article_refs import (
    CODE_SPACE_ACTIVATION,
    CONTINUATION_FIELD_EXPANSION,
    CONTINUATION_LOCAL_DETECTOR,
    DETECTOR_LEVELS,
    DETECTOR_RADIUS,
    FUZZY_SEARCH,
    LAID_OUT_STRUCTURE,
    MORPHOLOGICAL_EMBEDDINGS,
    OUTSIDE_ARTICLE,
    STIMULUS_DETECTION,
)
from damp import settings
from damp.figures.recorder import LadderRecorder
from damp.figures.style import DigitView, FigureStyle, Sheet, bits_raster, crop_cells, plural
from damp.cycle.ladder_trace import LadderLevelTrace
from damp.logging import LOGGER


class CycleFigures:
    """Рисунки этапов 1–7 по записанному прогону цикла."""

    def __init__(
        self,
        recorder: LadderRecorder,
        style: FigureStyle,
        *,
        side_cells: float,
        panel: int,
        strongest: int,
        digit: int,
        digit_index: "int | None" = None,
        test_digits: int,
    ) -> None:
        if recorder.run is None or not recorder.levels:
            raise ValueError("recorder holds no run")
        self.recorder = recorder
        self.run = recorder.run
        self.style = style
        self.side_cells = float(side_cells)
        self.panel = int(panel)
        self.strongest = int(strongest)
        self.digit = int(digit)
        self.test_digits = int(test_digits)
        self.starts = np.searchsorted(self.run.stick_picture, np.arange(self.run.labels.shape[0] + 1))
        train_first, train_last = self.run.ranges["train"]
        # Пример — пятёрка прогона (её выбирает MnistLadderRunner один раз, в каждом прогоне другая); digit_index —
        # запасной путь для прогонов без примера.
        if self.run.example is not None:
            self.picture = int(self.run.example)
        else:
            found = np.flatnonzero(self.run.labels[train_first:train_last] == self.digit) + train_first
            self.picture = int(found[min(int(digit_index or 0), found.shape[0] - 1)]) if found.size else train_first
        # Рисунки идут по одной цифре через все уровни. Если у пятёрки прогона на каком-то уровне нет ни одного окна
        # (окно на всю картинку выключено — LADDER_KEYHOLE_WHOLE_PICTURE), берётся первая обучающая пятёрка, у которой
        # окна есть на всех уровнях; замена пишется в журнал.
        if not self._has_keyholes(self.picture):
            chosen = next(
                (int(p) for p in np.flatnonzero(self.run.labels[train_first:train_last] == self.digit) + train_first
                 if self._has_keyholes(int(p))),
                None,
            )
            if chosen is None:
                raise ValueError(f"нет обучающей цифры {self.digit} с окнами на всех уровнях")
            LOGGER.event(
                "figures.example.replaced",
                section=OUTSIDE_ARTICLE,
                data={
                    "picture": self.picture, "replacement": chosen,
                    "levels_without_keyholes": [lv.number for lv in recorder.levels if not np.any(lv.keyhole_picture == self.picture)],
                    "note": "у пятёрки прогона нет окон на одном из уровней (окно на всю картинку выключено) — рисунки идут по "
                    "другой обучающей пятёрке, у которой окна есть на всех уровнях",
                },
            )
            self.picture = chosen
        # Окно-пример каждого уровня: ближайшее к середине цифры, дальше — к окну прошлого уровня.
        y, x, _ = self.sticks(self.picture)
        anchor = (float(y.mean()), float(x.mean()))
        self.example: dict[int, int] = {}
        for level in self.recorder.levels:
            mine = np.flatnonzero(level.keyhole_picture == self.picture)
            centre = level.keyhole_centre[mine].astype(np.float64)
            if level.number == 0:
                # Уровень 0: у всех окон цифры один центр — центр цифры; пример — окно штриха, ближайшего к середине.
                where = np.stack([level.element_y[mine], level.element_x[mine]], axis=1).astype(np.float64)
            else:
                where = centre
            chosen = int(mine[int(np.argmin(np.hypot(where[:, 0] - anchor[0], where[:, 1] - anchor[1])))])
            self.example[level.number] = chosen
            anchor = (float(level.keyhole_centre[chosen, 0]), float(level.keyhole_centre[chosen, 1]))
        self._panels: dict[int, dict] = {}

    # ------------------------------------------------------------------ общие части
    def _has_keyholes(self, picture: int) -> bool:
        """Есть ли у картинки окна на каждом записанном уровне."""
        return all(np.any(level.keyhole_picture == picture) for level in self.recorder.levels)

    def sticks(self, picture: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        first, last = int(self.starts[picture]), int(self.starts[picture + 1])
        return (
            self.run.stick_y[first:last].astype(np.float64),
            self.run.stick_x[first:last].astype(np.float64),
            self.run.stick_angle[first:last].astype(np.float64),
        )

    def view(self, size: int, picture: int, *, scale: float | None = None, centre=None, backdrop: float = 0.35) -> DigitView:
        scale = size / 26.0 if scale is None else scale
        return DigitView(self.style, size, self.run.images[picture], self.side_cells, scale, centre=centre, backdrop=backdrop)

    @staticmethod
    def unpack(packed: np.ndarray, length: int) -> np.ndarray:
        return np.unpackbits(np.ascontiguousarray(packed), axis=-1, count=int(length))

    def members(self, level: LadderLevelTrace, keyhole: int) -> np.ndarray:
        """Вхождения окна: номера строк в entry_*."""
        return np.flatnonzero(level.entry_keyhole == keyhole)

    # ------------------------------------------------------------------ этап 1
    def stimuli(self, path_digit: str, path_digits: str) -> None:
        """Цифра-образец, точки, из которых собраны штрихи, и сами штрихи; цифры 0…9 из одних штрихов."""
        from damp import settings
        from damp.figures.style import hue_colour, stick_hue

        style, size = self.style, self.panel
        picture = self.picture
        y, x, angle = self.sticks(picture)
        plain = Image.fromarray(self.run.images[picture]).resize((size, size), Image.BILINEAR).convert("RGB")
        lines = self.view(size, picture, backdrop=0.25)
        tiles = [("Картинка MNIST 28×28", "линии ищутся на картинке, увеличенной в 8 раз (224×224)", plain)]
        detectors = bool(settings.MNIST_STROKE_DETECTORS)
        midline = detectors and bool(settings.MNIST_STROKE_MIDLINE)
        points = 0
        if detectors:
            # Точки, над которыми детекторы собирают штрихи (§6.4 на плоскости картинки): середина линии или край.
            from damp.stimuli.mnist import _build_extractor

            found = _build_extractor().stroke_points(self.run.images[picture].astype(np.float32) / np.float32(255.0))
            dots = self.view(size, picture, backdrop=0.25)
            for py, px, pa in zip(found["point_y"], found["point_x"], found["point_angle"]):
                dots.dot(float(py), float(px), 1.6, hue_colour(stick_hue(float(pa))))
            points = int(found["point_y"].shape[0])
            tiles.append((
                f"{points} {plural(points, 'точка', 'точки', 'точек')} " + ("середины линии" if midline else "края"),
                ("гребень расстояния до края: одна середина вместо двух краёв; цвет — направление поперёк линии"
                 if midline else "пиксели контура; цвет — направление перепада яркости"),
                dots.image,
            ))
        else:
            for k in range(int(math.ceil(self.side_cells)) + 1):
                edge = k - 0.5
                lines.draw.line((lines.px(edge), lines.py(-0.5), lines.px(edge), lines.py(self.side_cells - 0.5)), fill=(30, 30, 29))
                lines.draw.line((lines.px(-0.5), lines.py(edge), lines.px(self.side_cells - 0.5), lines.py(edge)), fill=(30, 30, 29))
        for row in range(y.shape[0]):
            lines.stick(y[row], x[row], angle[row])
        tiles.append((
            f"{y.shape[0]} {plural(y.shape[0], 'штрих', 'штриха', 'штрихов')}",
            (f"штрих — детектор над точками: соседние точки с похожим направлением, центр — место штриха, центры не ближе "
             f"{int(settings.MNIST_STROKE_STEP)} px (§6.4 на картинке); "
             + ("отрезок вдоль линии, цвет — направление 0…180°" if midline else "стрелка вдоль края, цвет — направление перепада"))
            if detectors else "сетка 24×24 окна; окно на контуре даёт стрелку вдоль края, цвет — направление перепада яркости",
            lines.image,
        ))
        sheet = Sheet(style, len(tiles) * size + (len(tiles) + 1) * 20, f"Этап 1. Цифра {self.digit} и её штрихи")
        bottom = sheet.grid(tiles, columns=len(tiles))
        sheet.save(path_digit, bottom, note="стимулы — штрихи", data={"sticks": int(y.shape[0]), "points": points})
        tile = 240
        train_first, train_last = self.run.ranges["train"]
        tiles = []
        for digit in range(10):
            found = np.flatnonzero(self.run.labels[train_first:train_last] == digit)
            if not found.size:
                continue
            index = int(found[0]) + train_first
            dy, dx, da = self.sticks(index)
            view = self.view(tile, index, backdrop=0.0)
            for row in range(dy.shape[0]):
                view.stick(dy[row], dx[row], da[row])
            tiles.append((f"{digit}", f"{dy.shape[0]} {plural(dy.shape[0], 'штрих', 'штриха', 'штрихов')}", view.image))
        sheet = Sheet(style, 5 * tile + 6 * 20, "Цифры 0…9 из одних штрихов")
        bottom = sheet.grid(tiles, columns=5)
        sheet.save(path_digits, bottom, note="форма цифры читается по одним штрихам")

    # ------------------------------------------------------------------ этап 2
    @property
    def first(self) -> int:
        """Первый уровень прогона: 0 — с уровнем 0 (LADDER_LEVEL0), иначе 1. На нём в окно входит код угла штриха."""
        return min(self.example)

    def primary_coding(self, path: str, views: Callable[[object], object]) -> None:
        """Этап 2: первичные детекторы угла и места устроены одинаково (§4.2, §4.3, §4.5) — слои от широких к узким,
        значение в каждом слое накрывают около трёх детекторов, по случайному биту на детектор в полосе своего слоя
        (§4.2.1). Разница одна: угол — замкнутая координата (§4.4), место — относительно центра окна (на уровне 0 —
        центра цифры, в долях расстояния до самого дальнего штриха). Рисунок — две одинаковые строки и код окна (П.5)."""
        number = self.first
        level = self.recorder.level_trace(number)
        keyhole = self.example[number]
        if number == 0:
            # На уровне 0 в окне один штрих, место — от центра цифры: для рисунка берётся штрих, заметно удалённый от
            # центра (около 0.65 R), чтобы смещение было видно.
            own = np.flatnonzero(level.keyhole_picture == self.picture)
            entry = np.asarray([np.flatnonzero(level.entry_keyhole == k)[0] for k in own])
            shift = np.hypot(level.entry_offset[entry, 0], level.entry_offset[entry, 1])
            keyhole = int(own[int(np.argmin(np.abs(shift - 0.65)))])
        rows = self.members(level, keyhole)
        elements = level.entry_element[rows]
        points = level.entry_offset[rows].astype(np.float64) / float(level.unit)
        angles = self.run.stick_angle[elements].astype(np.float64)
        angle = float(angles[0]) if number == 0 else self.example_angle()
        angle_view = views(self.run.angle_encoder)
        place_view = views(level.place_encoder)
        angle_dim = self.run.angle_encoder.dimensions[0]
        place_dim = level.place_encoder.dimensions[0]
        angle_detectors = list(self.run.angle_encoder.detectors)
        circles = list(level.place_encoder.detectors)
        layer_count = len(angle_dim.detector_layers)
        place_layer_count = len(place_dim.detector_layers)
        angle_by_layer = [sum(1 for w in angle_detectors if w.layer_index == k and w.is_active(angle)) for k in range(layer_count)]
        place_by_layer = [
            sum(1 for w in circles if w.layer_index == k and any(w.is_active(p) for p in points)) for k in range(place_layer_count)
        ]
        angle_code, angle_stats = angle_view.code((angle,))
        bits = self.unpack(level.keyhole_codes[keyhole], level.code_length)
        place_bits = np.flatnonzero(bits[: level.place_length])
        place_code, place_stats = place_view.code(None, bits=place_bits)
        # Сверка: биты места окна — OR битов кругов, в которые попали его элементы (§2.2.1, П.5).
        expected: set[int] = set()
        for point in points:
            expected.update(level.place_encoder.dimension_bits(0, point))
        content = bits[level.place_length :]
        saturated = content.mean() >= float(settings.LADDER_BINDING_SATURATION) - 1e-9
        # Окно на цифре: центр и радиус (у уровня 0 — центр цифры и расстояние до самого дальнего штриха).
        wy, wx = (float(v) for v in level.keyhole_centre[keyhole])
        reach = float(level.keyhole_extent[keyhole])
        span = float(level.unit) * reach  # клеток картинки на единицу плоскости кругов места
        stroke = (float(level.element_y[elements[0]]), float(level.element_x[elements[0]]))
        big, mini = 300, 112
        ring = angle_view.ring(0, angle).resize((big, big), Image.LANCZOS)
        origin_view = self.place_origin(big, (wy, wx), reach, stroke, float(angles[0]), place_view, points, span)
        angle_strip = self.layer_strip(
            mini,
            [self.angle_layer(angle_view, angle_dim, k, angle, mini) for k in range(layer_count)],
            [f"слой {k + 1}: {sum(1 for w in angle_detectors if w.layer_index == k)}|горят {angle_by_layer[k]}" for k in range(layer_count)],
        )
        place_strip = self.layer_strip(
            mini,
            [self.place_layer(place_view, k, points, angles, (wy, wx), span, reach, mini) for k in range(place_layer_count)],
            [f"слой {k + 1}: {self.per_side(circles, k)}×{self.per_side(circles, k)}|горят {place_by_layer[k]}"
             for k in range(place_layer_count)],
        )
        origin = "центра цифры" if number == 0 else "центра окна"
        width = big + angle_strip.width + angle_code.width + 2 * 20 + 2 * 20
        sheet = Sheet(self.style, width, "Этап 2. Угол и место кодируются одинаково: слоями, по биту на детектор")
        top = sheet.grid(
            [
                (f"Угол {angle:.0f}° — замкнутый", f"замкнутая координата: круг {angle_dim.period:.0f}° (у штриха середины линии "
                 f"нет стрелки); все {layer_count} слоёв кольцами: внутри широкие, снаружи узкие", ring),
                (f"Угол: {layer_count} слоёв, {len(angle_detectors)} детекторов", f"в каждом слое угол накрывают 2–4 детектора; "
                 f"сработавшие — цветом своего бита, остальные серые; всего горят {sum(angle_by_layer)}", angle_strip),
                (f"Код угла: {angle_stats['length']} бит", f"по случайному биту на сработавший детектор, полоса бит — слой "
                 f"(широкие красные, узкие фиолетовые); единиц {angle_stats['ones']} ({angle_stats['density']:.1%})", angle_code),
            ],
            columns=3,
        )
        middle = sheet.grid(
            [
                (f"Место — от {origin}", f"смещение штриха от {origin} (белая линия) в долях R = {reach:.1f} клетки (серый "
                 f"круг); все {sum(place_by_layer)} сработавших кругов {place_layer_count} слоёв — цветом своего бита", origin_view),
                (f"Место: {place_layer_count} слоёв, {len(circles)} кругов", f"в каждом слое место накрывают 3–5 кругов; "
                 f"сработавшие — цветом своего бита, остальные серые; всего горят {sum(place_by_layer)}", place_strip),
                (f"Код места: {place_stats['length']} бит", f"так же: по случайному биту на сработавший круг, полоса бит — слой; "
                 f"единиц {place_stats['ones']} ({place_stats['density']:.1%})", place_code),
            ],
            columns=3,
            top=top,
        )
        content_picture = bits_raster(content, 128, 4, self.style.blue, self.style.off)
        bottom = sheet.grid(
            [
                (
                    f"Код окна: место {place_stats['length']} + содержание {content.shape[0]} бит",
                    f"место — код места (OR); содержание — код угла штриха под маской каждого из {sum(place_by_layer)} сработавших "
                    f"кругов (П.5): единиц {int(content.sum())} ({content.mean():.0%})"
                    + (f" — упёрлось в предел {float(settings.LADDER_BINDING_SATURATION):.0%}, лишние биты сброшены слиянием "
                       "цветов (§3.3)" if saturated else ""),
                    content_picture,
                ),
            ],
            columns=1,
            top=middle,
        )
        sheet.save(
            path,
            bottom,
            note="первичное кодирование: угол (§4.4) и место (§4.3) одинаково, слоями; код окна (П.5)",
            section=CONTINUATION_LOCAL_DETECTOR,
            data={
                "level": number,
                "place_circles": len(circles),
                "place_fired": int(sum(place_by_layer)),
                "place_by_layer": place_by_layer,
                "angle_by_layer": angle_by_layer,
                "content_saturated": bool(saturated),
                "angle": angle,
                "angle_ones": angle_stats["ones"],
                "place_ones": place_stats["ones"],
                "content_ones": int(content.sum()),
                "place_bits_match": sorted(expected) == [int(b) for b in place_bits],
            },
        )

    @staticmethod
    def per_side(circles: list, layer: int) -> int:
        """Кругов по стороне у слоя layer."""
        return int(round(math.sqrt(sum(1 for w in circles if w.layer_index == layer))))

    def layer_strip(self, side: int, pictures: list, labels: list) -> Image.Image:
        """Слои в ряд по пять, под каждым — подпись."""
        per_row, gap, label = 5, 10, 34
        rows = -(-len(pictures) // per_row)
        width = per_row * side + (per_row - 1) * gap
        strip = Image.new("RGB", (width, rows * (side + label) + (rows - 1) * gap), self.style.background)
        draw = ImageDraw.Draw(strip)
        font = self.style.font(12)
        for index, (picture, text) in enumerate(zip(pictures, labels)):
            row, column = divmod(index, per_row)
            left, top = column * (side + gap), row * (side + label + gap)
            strip.paste(picture, (left, top))
            for line, part in enumerate(text.split("|")):  # подпись в две строки: «|» — перенос
                draw.text((left, top + side + 2 + 15 * line), part, fill=self.style.second, font=font)
        return strip

    def angle_layer(self, view, dim, layer: int, angle: float, side: int) -> Image.Image:  # noqa: ANN001
        """Один слой детекторов угла: детектор — дуга своего поля на круге, сработавшие — цветом бита, остальные серые."""
        picture = Image.new("RGB", (side, side), self.style.background)
        draw = ImageDraw.Draw(picture)
        centre, outer = side / 2.0, side * 0.46
        windows = sorted((w for w in view._windows(0) if w.layer_index == layer), key=lambda w: w.center)
        lanes = view.layer_lanes(0, layer)
        thick = max(2.0, side * 0.14 / lanes)
        for position, window in enumerate(windows):
            active = window.is_active(float(angle))
            radius = outer - (position % lanes) * thick
            box = (centre - radius, centre - radius, centre + radius, centre + radius)
            fill = view.bit_colour(window.bit) if active else (78, 78, 74)
            width = 3 if active else 1
            if 2.0 * window.half_width >= dim.period:
                draw.ellipse(box, outline=fill, width=width)
                continue
            middle = view._degrees(dim, window.center)
            arc = 360.0 * window.half_width / dim.period
            draw.arc(box, -(middle + arc), -(middle - arc), fill=fill, width=width)
        ray = math.radians(view._degrees(dim, float(angle)))
        draw.line((centre, centre, centre + outer * math.cos(ray), centre - outer * math.sin(ray)), fill=self.style.ink, width=1)
        return picture

    def place_layer(self, view, layer: int, points: np.ndarray, angles: np.ndarray, centre: tuple, span: float,  # noqa: ANN001
                    reach: float, side: int) -> Image.Image:
        """Один слой кругов места прямо на цифре: круги стоят вокруг центра окна (у уровня 0 — центра цифры);
        сработавшие — цветом бита, остальные серые."""
        digit = self.view(side, self.picture, scale=side / (2.0 * 1.55 * reach), centre=centre, backdrop=0.75)
        draw = ImageDraw.Draw(digit.image, "RGBA")
        cy, cx = centre
        windows = [w for w in view._windows(0) if w.layer_index == layer]
        active = [any(w.is_active(p) for p in points) for w in windows]
        for window, on in sorted(zip(windows, active), key=lambda item: item[1]):
            y, x = cy + window.center_y * span, cx + window.center_x * span
            r = window.radius * span * digit.scale
            box = (digit.px(x) - r, digit.py(y) - r, digit.px(x) + r, digit.py(y) + r)
            if on:
                colour = view.bit_colour(window.bit)
                draw.ellipse(box, fill=colour + (28,), outline=colour + (255,), width=1)
            else:
                draw.ellipse(box, outline=(90, 90, 86, 150), width=1)
        for (dy, dx), a in zip(points, angles):
            digit.stick(cy + dy * span, cx + dx * span, float(a))
        digit.dot(cy, cx, 2, self.style.ink)
        return digit.image

    def place_origin(self, side: int, centre: tuple, reach: float, stroke: tuple, angle: float, view, points: np.ndarray,  # noqa: ANN001
                     span: float) -> Image.Image:
        """Место как смещение, все слои сразу (как кольца угла): цифра, её центр (центр окна), круг радиуса R, линия от
        центра к штриху и все сработавшие круги места всех слоёв — цветом своего бита, широкие снизу, узкие сверху."""
        y, x, a = self.sticks(self.picture)
        digit = self.view(side, self.picture, scale=side / (2.0 * 1.6 * reach), centre=centre, backdrop=0.85)
        for row in range(y.shape[0]):
            digit.stick(y[row], x[row], a[row], dim=0.5)
        cy, cx = centre
        draw = ImageDraw.Draw(digit.image, "RGBA")
        fired = [w for w in view._windows(0) if any(w.is_active(p) for p in points)]
        for window in sorted(fired, key=lambda w: (w.layer_index, w.bit)):
            colour = view.bit_colour(window.bit)
            r = window.radius * span * digit.scale
            px, py = digit.px(cx + window.center_x * span), digit.py(cy + window.center_y * span)
            draw.ellipse((px - r, py - r, px + r, py + r), outline=colour + (230,), width=1)
        digit.circle(cy, cx, reach, (150, 150, 145), width=1)
        digit.draw.line((digit.px(cx), digit.py(cy), digit.px(stroke[1]), digit.py(stroke[0])), fill=self.style.ink, width=1)
        digit.stick(stroke[0], stroke[1], angle)
        digit.circle(stroke[0], stroke[1], 0.8, self.style.orange, width=2)
        digit.dot(cy, cx, 4, self.style.ink)
        return digit.image

    def example_angle(self) -> float:
        """Угол штриха в центре окна-примера первого уровня прогона (с уровнем 0 — штрих его окна)."""
        level = self.recorder.level_trace(self.first)
        rows = self.members(level, self.example[self.first])
        offsets = level.entry_offset[rows].astype(np.float64)
        centre = int(np.argmin(np.hypot(offsets[:, 0], offsets[:, 1])))
        return float(self.run.stick_angle[level.entry_element[rows][centre]])

    @staticmethod
    def local_energy(cells: np.ndarray, activation: np.ndarray) -> np.ndarray:
        """Активация после энергии активации (ACTIVATION_LOCAL_ENERGY_THRESHOLD; расширение статьи, §5.7 на активации):
        точка гаснет, если Σ a_v / d(j, v) по соседям в радиусе ACTIVATION_POINT_ENERGY_RADIUS меньше доли μ_a от
        наибольшей по карте. Тот же счёт, что HierarchyDetection._local, на процессоре."""
        from damp import settings

        threshold = float(settings.ACTIVATION_LOCAL_ENERGY_THRESHOLD or 0.0)
        if threshold <= 0.0 or cells.shape[0] == 0:
            return activation
        from scipy.spatial import cKDTree

        pairs = cKDTree(cells).query_pairs(r=float(settings.ACTIVATION_POINT_ENERGY_RADIUS), output_type="ndarray")
        first = np.concatenate([pairs[:, 0], pairs[:, 1]]).astype(np.int64)
        second = np.concatenate([pairs[:, 1], pairs[:, 0]]).astype(np.int64)
        weight = 1.0 / np.hypot(cells[first, 0] - cells[second, 0], cells[first, 1] - cells[second, 1])
        energy = np.zeros(cells.shape[0], dtype=np.float64)
        np.add.at(energy, first, weight * activation[second])
        top = float(energy.max())
        relative = energy / top if top > 0.0 else np.zeros_like(energy)
        return np.where(relative >= threshold, activation, 0.0)

    @staticmethod
    def detail_places(level: LadderLevelTrace, keyholes: np.ndarray, keyhole: int) -> tuple[np.ndarray, tuple[float, float]]:
        """Где стоят детали окон keyholes и деталь окна keyhole. Деталь уровня встаёт в центр своего окна; у уровня 0 с
        местом от центра цифры (окно держит один штрих, центр окна — центр цифры) деталь — сам штрих, на его месте."""
        if level.number == 0 and level.entry_keyhole.shape[0] == level.keyhole_picture.shape[0]:
            places = np.stack([level.element_y, level.element_x], axis=1).astype(np.float64)
            first = level.entry_element[np.argsort(level.entry_keyhole, kind="stable")]  # штрих каждого окна
            return places[first[keyholes]], (float(places[first[keyhole], 0]), float(places[first[keyhole], 1]))
        centre = level.keyhole_centre.astype(np.float64)
        return centre[keyholes], (float(centre[keyhole, 0]), float(centre[keyhole, 1]))

    # ------------------------------------------------------------------ этапы 3–6: панели окна
    def keyhole_panels(self, number: int) -> dict:
        """Панели окна-примера уровня: окно на цифре, код окна, зажжённые точки памяти, детекторы, деталь."""
        if number in self._panels:
            return self._panels[number]
        style, size = self.style, self.panel
        level = self.recorder.level_trace(number)
        keyhole = self.example[number]
        picture = self.picture
        wy, wx = (float(v) for v in level.keyhole_centre[keyhole])
        radius = float(level.keyhole_extent[keyhole])
        rows = self.members(level, keyhole)
        local = level.entry_element[rows]
        # Окно на цифре: вид вмещает и цифру, и круг окна.
        scale = min(size / 26.0, (size - 8) / (2.0 * radius + 1.0))
        view = self.view(size, picture, scale=scale, centre=None if scale >= size / 26.0 else (wy, wx))
        y, x, angle = self.sticks(picture)
        seen = np.hypot(y - wy, x - wx) <= radius
        if number <= 1:  # элементы уровней 0 и 1 — штрихи (у уровня 1 с кодами уровня 0)
            inside = np.zeros(y.shape[0], dtype=bool)
            inside[local - int(self.starts[picture])] = True
            for row in range(y.shape[0]):
                view.stick(y[row], x[row], angle[row], dim=1.0 if inside[row] else 0.3)
            if number == 0:
                # Окно уровня 0 держит один штрих: кольцо показывает, какой.
                for row in np.flatnonzero(inside):
                    view.circle(y[row], x[row], 1.2, style.orange, width=3)
        else:
            for row in range(y.shape[0]):
                view.stick(y[row], x[row], angle[row], dim=0.75 if seen[row] else 0.22)
            own = np.flatnonzero(level.element_picture == picture)
            taken = np.isin(own, local)
            for row in own[~taken]:
                view.dot(level.element_y[row], level.element_x[row], max(1.5, scale * 0.16), (70, 78, 90))
            for row in own[taken]:
                view.dot(level.element_y[row], level.element_x[row], max(2.5, scale * 0.28), (120, 200, 255))
        view.circle(wy, wx, radius, style.green, width=3)
        view.dot(wy, wx, 4, None, outline=style.ink)
        # Код окна.
        bits = self.unpack(level.keyhole_codes[keyhole], level.code_length)
        place, content = bits[: level.place_length], bits[level.place_length :]
        cell = 6
        code = Image.new("RGB", (size, size), style.background)
        place_picture = bits_raster(place, 64, cell, style.orange, style.off)
        content_picture = bits_raster(content, 64, cell, style.blue, style.off)
        left = (size - 64 * cell) // 2
        code.paste(place_picture, (left, 0))
        code.paste(content_picture.crop((0, 0, 64 * cell, size - place_picture.height - 10)), (left, place_picture.height + 10))
        shown = (size - place_picture.height - 10) // cell * 64
        # Память уровня: зажжённые точки (§6.1) и сработавшие детекторы (§6.5) — пересчёт по правилам статьи.
        memory = self.unpack(level.map_codes, level.code_length).astype(np.float32)
        code_bits = bits.astype(np.float32)
        similarity = (memory @ code_bits) / np.sqrt(np.maximum(memory.sum(axis=1) * code_bits.sum(), 1.0))
        lit = (similarity >= level.activation_lambda) | (similarity >= similarity.max())
        activation = np.where(lit, similarity, 0.0)
        cells, detectors = level.map_cells.astype(np.float64), level.detectors
        # Энергия активации (расширение статьи, как в прогоне — HierarchyDetection._local): Ê_a(j) = Σ a_v / d(j, v) по
        # соседям в радиусе r_e, делённая на наибольшую по карте; точки с Ê_a < μ_a гаснут и в детекторы не идут.
        lit_all = int(lit.sum())
        activation = self.local_energy(cells, activation)
        lit = activation > 0.0
        weights = np.where(level.energies >= level.energy_threshold, level.energies, 0.0)
        inside = np.hypot(cells[:, None, 0] - detectors[None, :, 0], cells[:, None, 1] - detectors[None, :, 1]) <= detectors[None, :, 2]
        strength = ((activation * weights) @ inside) / detectors[:, 4]
        fired = np.flatnonzero(strength >= level.detector_threshold)
        mine = level.fired_keyhole == keyhole
        recorded = np.sort(level.fired_detector[mine])
        order = np.argsort(-level.fired_level[mine], kind="stable")
        strongest = level.fired_detector[mine][order][: self.strongest]
        same = recorded.shape == fired.shape and bool(np.all(recorded == fired))
        height, width = level.map_size
        unit = max(1, size // max(height, width))
        shift_x, shift_y = (size - width * unit) // 2, (size - height * unit) // 2
        low, high = float(level.activation_lambda), float(similarity.max())

        def plane(with_lit: bool, with_detectors: bool) -> Image.Image:
            picture_ = Image.new("RGB", (size, size), style.background)
            draw = ImageDraw.Draw(picture_)
            for index in range(cells.shape[0]):
                cy, cx = cells[index]
                colour = (46, 46, 44)
                if with_lit and lit[index]:
                    share = 0.0 if high <= low else min(1.0, max(0.0, (similarity[index] - low) / (high - low)))
                    colour = tuple(int(a + (b - a) * share) for a, b in zip((40, 96, 170), (170, 215, 255)))
                left_, upper = shift_x + int(cx) * unit, shift_y + int(cy) * unit
                draw.rectangle((left_, upper, left_ + unit - 2, upper + unit - 2), fill=colour)
            if with_detectors:
                best = set(int(v) for v in strongest)

                def ring(index: int, colour, thickness: int) -> None:
                    cy, cx, r = detectors[index, 0], detectors[index, 1], detectors[index, 2]
                    draw.ellipse(
                        (shift_x + (cx - r + 0.5) * unit, shift_y + (cy - r + 0.5) * unit, shift_x + (cx + r + 0.5) * unit, shift_y + (cy + r + 0.5) * unit),
                        outline=colour,
                        width=thickness,
                    )

                for index in recorded:
                    if int(index) not in best:
                        ring(int(index), (96, 52, 34), 1)
                for index in strongest:
                    ring(int(index), style.orange, 2)
            return picture_

        lit_picture = plane(True, False)
        fired_picture = plane(True, True)
        # Деталь — эмбеддинг окна (§6.5): код и место — центр окна.
        detail = self.unpack(level.detail_codes[keyhole], level.detail_length)
        band = bits_raster(detail, 44, 9, style.aqua, style.off)
        detail_view_size = size - band.height - 14
        details = self.view(detail_view_size, picture, scale=detail_view_size / 26.0, backdrop=0.3)
        for row in range(y.shape[0]):
            details.stick(y[row], x[row], angle[row], dim=0.25)
        own_keyholes = np.flatnonzero(level.keyhole_picture == picture)
        spots, spot = self.detail_places(level, own_keyholes, keyhole)
        for sy, sx in spots:
            details.dot(sy, sx, 3.0, style.aqua)
        details.dot(spot[0], spot[1], 6.5, style.aqua, outline=style.ink)
        box = Image.new("RGB", (size, size), style.background)
        box.paste(band, ((size - band.width) // 2, 0))
        box.paste(details.image, ((size - details.size) // 2, band.height + 14))
        stats = dict(
            number=number,
            radius=radius,
            members=int(rows.shape[0]),
            seen=int(seen.sum()),
            sticks=int(y.shape[0]),
            place_bits=int(place.sum()),
            content_bits=int(content.sum()),
            code_length=int(bits.shape[0]),
            content_length=int(content.shape[0]),
            place_length=int(place.shape[0]),
            shown=int(min(shown, content.shape[0])),
            points=int(cells.shape[0]),
            lit=int(lit.sum()),
            lit_all=lit_all,
            low=low,
            fired=int(recorded.shape[0]),
            detectors=int(detectors.shape[0]),
            strongest=int(strongest.shape[0]),
            detail_length=int(detail.shape[0]),
            detail_ones=int(detail.sum()),
            detail_strength=float(level.detail_strength[keyhole]),
            details=int(own_keyholes.shape[0]),
            same=same,
        )
        LOGGER.event(
            "figures.keyhole",
            section=STIMULUS_DETECTION,
            data={
                **stats,
                "note": "окно-пример уровня: код окна зажёг точки памяти (§6.1), по ним сработали детекторы (§6.5); "
                "пересчёт совпал с прогоном — same",
            },
        )
        self._panels[number] = dict(keyhole=view.image, code=code, lit=lit_picture, fired=fired_picture, detail=box, stats=stats)
        return self._panels[number]

    def step(self, number: int, path: str) -> None:
        """Шаг уровня на цифре-образце: окно, код окна, память уровня, деталь."""
        panels = self.keyhole_panels(number)
        s = panels["stats"]
        if number == 0:
            what = "штрих; центр окна — центр цифры, радиус — до самого дальнего штриха"
        elif number == 1:
            what = "штрихи" + (" с кодами уровня 0" if self.first == 0 else "")
        else:
            what = f"детали уровня {number - 1} (голубые точки)"
        last = number == max(self.example)
        sheet = Sheet(self.style, 2 * self.panel + 3 * 20, f"Уровень {number}: шаг на цифре {self.digit}")
        bottom = sheet.grid(
            [
                (
                    "1. Окно на цифре",
                    f"радиус поля {s['radius']:.1f} клетки; в окне {s['members']} {plural(s['members'], 'элемент', 'элемента', 'элементов')} — {what}"
                    + ("" if number <= 1 else f"; на картинке окно видит {s['seen']} из {s['sticks']} штрихов"),
                    panels["keyhole"],
                ),
                (
                    f"2. Код окна — {s['code_length']} бит",
                    f"место (оранжевые): {s['place_length']} бит, единиц {s['place_bits']}; содержание (синие): {s['content_length']} "
                    f"бит, единиц {s['content_bits']}" + ("" if s["shown"] >= s["content_length"] else f"; показаны первые {s['shown']}"),
                    panels["code"],
                ),
                (
                    "3. Память уровня",
                    f"карта из {s['points']} кодов окон обучающих цифр; код окна зажёг {s['lit_all']} "
                    f"{plural(s['lit_all'], 'точку', 'точки', 'точек')} (сходство не ниже {s['low']:.2f}), в детекторы идут "
                    f"{s['lit']} с соседями (синие; энергия активации); сработало {s['fired']} из "
                    f"{s['detectors']} детекторов, {s['strongest']} сильнейших — яркие круги",
                    panels["fired"],
                ),
                (
                    "4. Деталь",
                    f"код детали {s['detail_length']} бит, единиц {s['detail_ones']} — по биту на сильнейший детектор; уровень "
                    f"активации {s['detail_strength']:.1f}; "
                    + ("деталь — сам штрих с выученным кодом (обведён)" if number == 0 else "деталь стоит в центре окна (обведена)")
                    + f", всего на цифре {s['details']} {plural(s['details'], 'деталь', 'детали', 'деталей')} — "
                    + ("по ним голосуется класс" if last else "элементы следующего уровня"),
                    panels["detail"],
                ),
            ],
            columns=2,
        )
        sheet.save(path, bottom, note="шаг уровня цикла (§6.6)", section=STIMULUS_DETECTION, data={"level": number, "same": s["same"]})

    def layout_crop(self, glyph_path: str, path: str, *, columns: int, rows: int, number: int = 1) -> None:
        """Этап 3: вырезка из середины карты памяти уровня number с рисунками окон в ячейках."""
        level = self.recorder.level_trace(number)
        image = Image.open(glyph_path).convert("RGB")
        tile = image.width // int(level.map_size[1])
        top_skip = image.height - int(level.map_size[0]) * tile
        crop = crop_cells(image, columns * tile, rows * tile, tile=tile, top_skip=top_skip)
        if number == 0:
            title = "Этап 3. Раскладка: память уровня 0 — карта штрихов"
            note = ("в ячейке — цифра точки памяти, середина ячейки — центр цифры; ярко — штрих этой точки, тускло — "
                    "остальные; цвет рамки — цифра; рядом легли штрихи с похожим наклоном и местом в цифре")
        else:
            title = f"Этап 3. Раскладка: память уровня {number}"
            note = ("в ячейке — что видит окно этой точки памяти (ярко) на фоне соседних штрихов (тускло); "
                    "цвет рамки — цифра; рядом легли похожие обрывки контура")
        sheet = Sheet(self.style, crop.width + 40, title)
        bottom = sheet.grid([(f"Вырезка из середины карты ({level.map_cells.shape[0]} точек)", note, crop)], columns=1)
        sheet.save(path, bottom, note=f"раскладка кодов окон уровня {number}", section=LAID_OUT_STRUCTURE)

    # ------------------------------------------------------------------ этап 4
    def hierarchy(self, path: str) -> None:
        """Иерархия детекторов над картой памяти первого уровня, по панели на слой (§6.4, §7.1.10)."""
        level = self.recorder.level_trace(1)
        cells, detectors, layer_index = level.map_cells, level.detectors, level.layer_index
        height, width = level.map_size
        unit, gap, margin = 3, 14, 20
        side = max(height, width) * unit
        layers = len(level.layer_thresholds)
        sheet = Sheet(self.style, layers * side + (layers - 1) * gap + 2 * margin, f"Этап 4. Детекторы над картой памяти первого уровня: {detectors.shape[0]} в {layers} слоях")
        panels = []
        counts = []
        for layer in range(layers):
            plane = Image.new("RGB", (side, side), self.style.background)
            draw = ImageDraw.Draw(plane)
            for cy, cx in cells:
                draw.rectangle((int(cx) * unit, int(cy) * unit, int(cx) * unit + unit - 2, int(cy) * unit + unit - 2), fill=(58, 58, 55))
            chosen = np.flatnonzero(layer_index == layer)
            for index in chosen:
                cy, cx, r = detectors[index, 0], detectors[index, 1], detectors[index, 2]
                draw.ellipse(((cx - r + 0.5) * unit, (cy - r + 0.5) * unit, (cx + r + 0.5) * unit, (cy + r + 0.5) * unit), outline=self.style.orange, width=1)
            radius = detectors[chosen, 2]
            counts.append(int(chosen.shape[0]))
            panels.append(
                (
                    f"слой {layer + 1}: λ = {level.layer_thresholds[layer]:.2f}",
                    f"{chosen.shape[0]} детекторов, радиус {np.median(radius):.1f} (до {radius.max():.0f})",
                    plane,
                )
            )
        bottom = sheet.grid(panels, columns=layers, gap=gap)
        sheet.save(path, bottom, note="иерархия детекторов статьи над картой уровня", section=DETECTOR_LEVELS, data={"per_layer": counts})

    def activation(self, path: str) -> None:
        """Этап 4: окно-пример первого уровня зажигает точки памяти, по ним срабатывают детекторы."""
        panels = self.keyhole_panels(1)
        s = panels["stats"]
        sheet = Sheet(self.style, 3 * self.panel + 4 * 20, "Этап 4. Активация: код окна → точки памяти → детекторы")
        bottom = sheet.grid(
            [
                ("Окно на цифре", f"окно уровня 1, в нём {s['members']} штрихов; его код — стимул (этап 2)", panels["keyhole"]),
                (
                    "Зажглись точки памяти",
                    f"{s['lit_all']} из {s['points']}: сходство кода окна с кодом точки не ниже λ_a = {s['low']:.2f} (§6.1); "
                    f"синие — {s['lit']}, у которых зажжены и соседи (энергия активации ≥ 0.5): только они идут в детекторы; ярче — похожее",
                    panels["lit"],
                ),
                (
                    "Сработали детекторы",
                    f"{s['fired']} из {s['detectors']}: энергия зажжённых точек поля не меньше половины собственной (§6.5); "
                    f"яркие — {s['strongest']} сильнейших",
                    panels["fired"],
                ),
            ],
            columns=3,
        )
        sheet.save(path, bottom, note="активация карты и срабатывание детекторов", section=CODE_SPACE_ACTIVATION, data={"lit": s["lit"], "fired": s["fired"], "same": s["same"]})

    # ------------------------------------------------------------------ этап 5
    def embedding(self, path: str) -> None:
        """Этап 5: эмбеддинг окна — деталь: код из сильнейших детекторов, место — центр окна."""
        panels = self.keyhole_panels(1)
        s = panels["stats"]
        sheet = Sheet(self.style, 2 * self.panel + 3 * 20, "Этап 5. Эмбеддинг окна — деталь первого уровня")
        bottom = sheet.grid(
            [
                ("Сильнейшие детекторы", f"{s['strongest']} ярких кругов из {s['fired']} сработавших", panels["fired"]),
                (
                    "Деталь: код и место",
                    f"код {s['detail_length']} бит, единиц {s['detail_ones']} — по случайному биту на детектор (§7.1.12); "
                    f"место — центр окна; на цифре {s['details']} {plural(s['details'], 'деталь', 'детали', 'деталей')}",
                    panels["detail"],
                ),
            ],
            columns=2,
        )
        sheet.save(path, bottom, note="эмбеддинг окна — деталь", section=MORPHOLOGICAL_EMBEDDINGS, data={"ones": s["detail_ones"], "length": s["detail_length"]})

    # ------------------------------------------------------------------ этап 6
    def keyholes_growth(self, fields_path: str, path: str) -> None:
        """Окна уровней вокруг одного места цифры-образца — рисунок, который пишет сам прогон."""
        shutil.copyfile(fields_path, path)
        LOGGER.event("figure.save", section=CONTINUATION_FIELD_EXPANSION, data={"path": path, "source": fields_path, "note": "окна уровней растут"})

    def level_maps(self, glyph_paths: dict[int, str], path: str, *, columns: int, rows: int) -> None:
        """Вырезки из середины карт памяти всех уровней."""
        panels = []
        about = {0: "штрихи и их место в цифре", 1: "обрывки контура", 2: "части цифр", 3: "цифры почти целиком"}
        for level in self.recorder.levels:
            image = Image.open(glyph_paths[level.number]).convert("RGB")
            tile = image.width // int(level.map_size[1])
            top_skip = image.height - int(level.map_size[0]) * tile
            crop = crop_cells(image, columns * tile, rows * tile, tile=tile, top_skip=top_skip)
            note = ("в окне один штрих (ярко); середина ячейки — центр цифры, тускло — остальные штрихи" if level.number == 0
                    else f"окно видит в среднем {level.seen:.0%} штрихов цифры")
            panels.append((f"Уровень {level.number}: {about.get(level.number, '')}", note, crop))
        width = max(panel.width for _, _, panel in panels) + 40
        sheet = Sheet(self.style, width, "Карты памяти уровней (вырезки из середины)")
        bottom = sheet.grid(panels, columns=1)
        sheet.save(path, bottom, note="карты памяти уровней", section=LAID_OUT_STRUCTURE)

    def loop_thumbnails(self, glyph_paths: dict[int, str], side: int) -> list[tuple[int, float, Image.Image]]:
        """Миниатюры карт уровней для обзорной полосы: номер, что видит окно, картинка."""
        result = []
        for level in self.recorder.levels:
            image = Image.open(glyph_paths[level.number]).convert("RGB")
            tile = image.width // int(level.map_size[1])
            top_skip = image.height - int(level.map_size[0]) * tile
            crop = crop_cells(image, 4 * tile, 4 * tile, tile=tile, top_skip=top_skip)
            result.append((level.number, level.seen, crop.resize((side, side), Image.LANCZOS)))
        return result

    # ------------------------------------------------------------------ этап 7
    def class_clusters(self, path: str) -> list[float]:
        """Собираются ли цифры в пятна на картах уровней: доля соседей точки той же цифры."""
        style = self.style
        cell, gap, left, top = 2, 14, 20, 64
        shares_by_level = []
        rows = []
        for level in self.recorder.levels:
            height, width = level.map_size
            grid = np.full((height, width), -1, dtype=np.int64)
            grid[level.map_cells[:, 0], level.map_cells[:, 1]] = level.map_labels
            padded = np.full((height + 2, width + 2), -1, dtype=np.int64)
            padded[1:-1, 1:-1] = grid
            total = np.zeros(grid.shape, dtype=np.int64)
            same = np.zeros(grid.shape, dtype=np.int64)
            # Соседи по горизонтали и вертикали — та же мера, что у прогона (LabelMapQuality).
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                other = padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
                total += (other >= 0) & (grid >= 0)
                same += (other == grid) & (grid >= 0)
            overall = float(same.sum()) / max(1, int(total.sum()))
            shares = []
            for digit in range(10):
                mine = grid == digit
                shares.append(float(same[mine].sum()) / max(1, int(total[mine].sum())))
            counts = np.bincount(level.map_labels, minlength=10).astype(np.float64)
            chance = float(((counts / counts.sum()) ** 2).sum())
            shares_by_level.append(overall)
            rows.append((level.number, grid, shares, overall, chance))
        side = max(grid.shape[0] for _, grid, _, _, _ in rows) * cell
        width = left + 10 * side + 9 * gap + 20
        height = top + len(rows) * (side + 70) + 20
        canvas = Image.new("RGB", (width, height), style.background)
        draw = ImageDraw.Draw(canvas)
        draw.text((left, 16), "Собираются ли цифры в пятна на картах памяти уровней", fill=style.ink, font=style.font(26, bold=True))
        for row, (number, grid, shares, overall, chance) in enumerate(rows):
            upper = top + row * (side + 70)
            draw.text((left, upper), f"Уровень {number}", fill=style.ink, font=style.font(20, bold=True))
            draw.text((left + 110, upper + 2), f"соседей той же цифры (по горизонтали и вертикали) {overall:.0%}, при случайной расстановке {chance:.0%}", fill=style.second, font=style.font(17))
            ys, xs = np.nonzero(grid >= 0)
            for digit in range(10):
                x0, y0 = left + digit * (side + gap), upper + 30
                draw.rectangle((x0, y0, x0 + side - 1, y0 + side - 1), fill=style.surface)
                for yy, xx in zip(ys, xs):
                    colour = style.blue if grid[yy, xx] == digit else (56, 56, 53)
                    draw.rectangle((x0 + xx * cell, y0 + yy * cell, x0 + xx * cell + cell - 1, y0 + yy * cell + cell - 1), fill=colour)
                draw.text((x0, y0 + side + 5), f"цифра {digit}", fill=style.ink, font=style.font(15, bold=True))
                draw.text((x0 + 70, y0 + side + 6), f"{shares[digit]:.0%}", fill=style.second, font=style.font(15))
        canvas.save(path, optimize=True)
        LOGGER.event("figure.save", section=OUTSIDE_ARTICLE, data={"path": path, "same_label": shares_by_level, "note": "доля соседей той же цифры по уровням"})
        return shares_by_level

    def top_map(self, glyph_path: str, path: str, *, tile: int, columns: int, rows: int) -> None:
        image = Image.open(glyph_path).convert("RGB")
        crop = crop_cells(image, columns * tile, rows * tile, tile=tile, top_skip=image.height - image.width)
        sheet = Sheet(self.style, crop.width + 40, "Карта верхних деталей (вырезка из середины)")
        bottom = sheet.grid([("", "в ячейке — что видит деталь верхнего уровня; цвет рамки — цифра", crop)], columns=1)
        sheet.save(path, bottom, note="карта деталей верхнего уровня", section=LAID_OUT_STRUCTURE)

    def recognition(self, path: str) -> dict:
        """Первые проверочные цифры-образцы: голоса верхних деталей и ближайшая обучающая по отклику уровней."""
        style = self.style
        top = self.recorder.top_details
        if top is None:
            raise ValueError("no top details recorded")
        labels, images = self.run.labels, self.run.images
        train_first, train_last = self.run.ranges["train"]
        test_first, test_last = self.run.ranges["test"]
        train_rows = np.flatnonzero((top.picture >= train_first) & (top.picture < train_last))
        train_bits = self.unpack(top.codes[train_rows], top.code_length).astype(np.float32)
        train_ones = train_bits.sum(axis=1)
        train_labels = labels[top.picture[train_rows]]

        def votes_of(picture: int) -> tuple[np.ndarray, np.ndarray]:
            rows = np.flatnonzero(top.picture == picture)
            bits = self.unpack(top.codes[rows], top.code_length).astype(np.float32)
            similarity = (bits @ train_bits.T) / np.sqrt(np.maximum(bits.sum(axis=1)[:, None] * train_ones[None, :], 1.0))
            return rows, train_labels[similarity.argmax(axis=1)]

        right = 0
        for picture in range(test_first, test_last):
            _, votes = votes_of(picture)
            if votes.size and int(np.bincount(votes, minlength=10).argmax()) == int(labels[picture]):
                right += 1
        accuracy = right / max(1, test_last - test_first)
        chosen = [int(p) for p in np.flatnonzero(labels[test_first:test_last] == self.digit)[: self.test_digits] + test_first]
        cell, bars_width, thumb, gap, margin = 190, 300, 86, 18, 20
        columns = [margin, margin + cell + gap, margin + 2 * (cell + gap), margin + 2 * (cell + gap) + bars_width + gap]
        levels = self.recorder.levels
        width = columns[3] + len(levels) * (thumb + 10) + margin
        sheet = Sheet(style, width, f"Этап 7. Узнавание: первые проверочные цифры {self.digit}")
        heads = ["Проверочная цифра", "Верхние детали и их голоса", "Голоса по цифрам", "Ближайшая обучающая по отклику уровней"]
        widths = [cell, cell, bars_width, len(levels) * (thumb + 10)]
        for left, head, head_width in zip(columns, heads, widths):
            sheet.text(left, sheet.top, head, head_width)
        top_edge = sheet.top + 48
        answers = []
        for picture in chosen:
            rows, votes = votes_of(picture)
            counts = np.bincount(votes, minlength=10)
            answer = int(counts.argmax())
            answers.append(answer)
            sheet.paste(self.view(cell, picture, backdrop=1.0).image, columns[0], top_edge)
            voted = self.view(cell, picture, backdrop=0.28)
            for index, vote in zip(rows, votes):
                voted.dot(top.y[index], top.x[index], 3.2, style.blue if int(vote) == self.digit else style.orange)
            sheet.paste(voted.image, columns[1], top_edge)
            base, peak = top_edge + cell - 26, max(1, int(counts.max()))
            step = bars_width / 10.0
            sheet.draw.line((columns[2], base + 1, columns[2] + bars_width, base + 1), fill=(56, 56, 53), width=1)
            for value in range(10):
                bar = int(round((cell - 60) * counts[value] / peak))
                left = columns[2] + value * step + 5
                colour = style.blue if value == self.digit else style.orange
                if bar > 0:
                    sheet.draw.rounded_rectangle((left, base - bar, left + step - 10, base), radius=3, fill=colour)
                    sheet.draw.rectangle((left, base - min(bar, 3), left + step - 10, base), fill=colour)
                sheet.draw.text((left + (step - 10) / 2 - 4, base + 5), str(value), fill=style.muted, font=style.font(15))
                if value in (answer, self.digit) and counts[value] > 0:
                    text = str(int(counts[value]))
                    sheet.draw.text((left + (step - 10) / 2 - sheet.draw.textlength(text, font=style.font(14)) / 2, base - bar - 18), text, fill=style.ink, font=style.font(14))
            verdict = f"ответ: {answer} — " + ("верно" if answer == self.digit else "ошибка") + f", голосов {int(counts[answer])} из {int(counts.sum())}"
            sheet.draw.text((columns[2], top_edge + cell + 4), verdict, fill=style.ink, font=style.font(16))
            for number, level in enumerate(levels):
                response = level.response
                nearest = int(np.argmax(response[train_first:train_last] @ response[picture])) + train_first
                left = columns[3] + number * (thumb + 10)
                tile = Image.fromarray(images[nearest]).resize((thumb, thumb), Image.BILINEAR).convert("RGB")
                sheet.paste(tile, left, top_edge + 8)
                good = int(labels[nearest]) == self.digit
                sheet.draw.rectangle((left - 1, top_edge + 7, left + thumb, top_edge + 8 + thumb), outline=style.blue if good else style.orange, width=2)
                sheet.draw.text((left, top_edge + thumb + 14), f"ур. {level.number} → {int(labels[nearest])}", fill=style.second, font=style.font(15))
            top_edge += cell + 44
        legend = top_edge + 2
        sheet.draw.ellipse((margin, legend + 4, margin + 12, legend + 16), fill=style.blue)
        sheet.draw.text((margin + 20, legend), f"деталь голосует за {self.digit}", fill=style.second, font=style.font(16))
        sheet.draw.ellipse((margin + 220, legend + 4, margin + 232, legend + 16), fill=style.orange)
        sheet.draw.text((margin + 240, legend), "за другую цифру; рамка обучающей цифры — тем же цветом", fill=style.second, font=style.font(16))
        sheet.save(path, legend + 22, note="узнавание голосованием верхних деталей", section=FUZZY_SEARCH, data={"accuracy_vote": accuracy, "answers": answers})
        return {"accuracy": accuracy, "answers": answers, "pictures": chosen}

    def radius_note(self) -> dict:
        """Радиусы окна-примера по уровням — для подписей (§6.4.3)."""
        data = {number: float(self.recorder.level_trace(number).keyhole_extent[keyhole]) for number, keyhole in self.example.items()}
        LOGGER.event("figures.example_radius", section=DETECTOR_RADIUS, data={"radius": data})
        return data
