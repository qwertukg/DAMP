# -*- coding: utf-8 -*-
"""Полосы-нарезки для README (вне статьи: показ).

Собирает кадры пошаговых схем и карты прогонов стенда в полосы `docs/readme/*.png`:
кадры по порядку шага, номер в кружке, стрелки между кадрами; подписи кадров — в README под полосой.

    python tools/readme_strips.py

Откуда кадры (папка data/ в git не входит — полосы пересобираются только там, где она есть):
    data/maps/atomic/panels/        пошаговая схема одного прохода, четвёрка MNIST train № 2 (разбор 2026-11-21, 250 цифр);
    data/maps/level-diagrams/panels/ тот же проход на уровнях 0–4 и распознавание;
    data/maps/tune2-<прогон>.png    карты уровней прогона стенда (1568 и 1569 — весь MNIST train, 60 000 картинок).
Счёта нет: только чтение готовых картинок.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ATOMIC = ROOT / "data/maps/atomic/panels"
LEVELS = ROOT / "data/maps/level-diagrams/panels"
MAPS = ROOT / "data/maps"
OUT = ROOT / "docs/readme"

BACK = (13, 14, 18)        # фон кадров схем
CARD = (27, 29, 38)
EDGE = (52, 56, 72)
INK = (236, 238, 242)
MUTED = (150, 155, 170)
GREEN = (62, 207, 142)     # стрелки и номера — как на схемах блоков
ORANGE = (255, 122, 48)
# Кадры, у которых сверху остаток обрезанной подписи: сколько строк срезать.
CROP_TOP = {"rec-memory.png": 34, "rec-reader.png": 34}


def save(sheet: Image.Image, target: Path) -> Path:
    """PNG с палитрой 256 цветов: полосы в 3–5 раз легче, на глаз — то же."""
    sheet.quantize(colors=256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).save(target, optimize=True)
    return target


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for name in (("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit(path: Path, width: int, height: int) -> Image.Image:
    """Кадр, вписанный в width × height с сохранением пропорций, на фоне схем."""
    picture = Image.open(path).convert("RGB")
    if path.name in CROP_TOP:  # у кадра обрезана подпись сверху — срезать её остаток
        picture = picture.crop((0, CROP_TOP[path.name], picture.width, picture.height))
    picture.thumbnail((width, height), Image.LANCZOS)
    canvas = Image.new("RGB", (width, height), BACK)
    canvas.paste(picture, ((width - picture.width) // 2, (height - picture.height) // 2))
    return canvas


def badge(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, radius: int = 22) -> None:
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=GREEN)
    draw.text((x, y + 1), text, fill=(10, 12, 16), font=font(int(radius * 1.15), bold=True), anchor="mm")


def arrow(draw: ImageDraw.ImageDraw, x0: int, x1: int, y: int, colour=GREEN, width: int = 6) -> None:
    draw.line((x0, y, x1 - 12, y), fill=colour, width=width)
    draw.polygon([(x1, y), (x1 - 16, y - 12), (x1 - 16, y + 12)], fill=colour)


def strip(name: str, frames: list[Path], *, size=(300, 252), titles: list[str] | None = None, gap: int = 44) -> Path:
    """Полоса: карточки с кадрами слева направо, номер в кружке, стрелки между карточками."""
    width, height = size
    pad, head = 12, 50
    card_w, card_h = width + 2 * pad, height + 2 * pad + head
    total = len(frames) * card_w + (len(frames) - 1) * gap + 2 * 16
    sheet = Image.new("RGB", (total, card_h + 2 * 16), BACK)
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(frames):
        x = 16 + index * (card_w + gap)
        draw.rounded_rectangle((x, 16, x + card_w, 16 + card_h), radius=14, fill=CARD, outline=EDGE, width=2)
        if titles:
            draw.text((x + pad + 56, 16 + pad + head // 2 - 4), titles[index], fill=INK, font=font(25, bold=True), anchor="lm")
        sheet.paste(fit(path, width, height), (x + pad, 16 + pad + head))
        badge(draw, x + pad + 22, 16 + pad + head // 2 - 4, str(index + 1), radius=20)
        if index:
            arrow(draw, x - gap + 6, x - 6, 16 + card_h // 2 + head // 2)
    target = OUT / f"{name}.png"
    return save(sheet, target)


def pipeline() -> Path:
    """Обзор: по кадру на этап 1–7 и петля цикла от этапа 6 к этапу 2."""
    frames = [ATOMIC / "2-08-sticks.png", ATOMIC / "5-05-window.png", ATOMIC / "6-final.png", ATOMIC / "8-04-fired.png",
              ATOMIC / "8-06-detail.png", ATOMIC / "10-L3-window.png", LEVELS / "rec-votes.png"]
    titles = ["Стимулы", "Код окна", "Раскладка", "Детекторы", "Эмбеддинг", "Цикл", "Ответ"]
    width, height, pad, head, gap = 230, 200, 10, 48, 40
    card_w, card_h = width + 2 * pad, height + 2 * pad + head
    total = 7 * card_w + 6 * gap + 32
    loop = 92
    sheet = Image.new("RGB", (total, card_h + 32 + loop), BACK)
    draw = ImageDraw.Draw(sheet)
    centres = []
    for index, path in enumerate(frames):
        x = 16 + index * (card_w + gap)
        centres.append(x + card_w // 2)
        cycle = 1 <= index <= 5
        draw.rounded_rectangle((x, 16, x + card_w, 16 + card_h), radius=14, fill=CARD,
                               outline=GREEN if cycle else EDGE, width=3 if cycle else 2)
        draw.text((x + 58, 16 + pad + head // 2), titles[index], fill=INK, font=font(26, bold=True), anchor="lm")
        sheet.paste(fit(path, width, height), (x + pad, 16 + pad + head))
        badge(draw, x + pad + 24, 16 + pad + 24, str(index + 1), radius=20)
        if index:
            arrow(draw, x - gap + 4, x - 4, 16 + card_h // 2 + head // 2)
    # петля цикла: из-под этапа 6 назад к этапу 2
    bottom, low = 16 + card_h, 16 + card_h + loop - 34
    draw.line((centres[5], bottom + 4, centres[5], low), fill=GREEN, width=6)
    draw.line((centres[5], low, centres[1], low), fill=GREEN, width=6)
    draw.line((centres[1], low, centres[1], bottom + 18), fill=GREEN, width=6)
    draw.polygon([(centres[1], bottom + 4), (centres[1] - 13, bottom + 20), (centres[1] + 13, bottom + 20)], fill=GREEN)
    label = "детали уровня — снова на этап 2: уровни 0 → 1 → 2 → 3 → 4"
    middle = (centres[1] + centres[5]) // 2
    box = draw.textbbox((middle, low), label, font=font(26, bold=True), anchor="mm")
    draw.rectangle((box[0] - 14, box[1] - 6, box[2] + 14, box[3] + 6), fill=BACK)
    draw.text((middle, low), label, fill=GREEN, font=font(26, bold=True), anchor="mm")
    target = OUT / "pipeline.png"
    return save(sheet, target)


def level_grid() -> Path:
    """Один и тот же шаг на уровнях 0–4: строки — уровни, столбцы — части шага."""
    columns = [("2-window", "окно"), ("3-window-code", "код окна"), ("4-map", "карта уровня"), ("6-lit", "зажглось"),
               ("7-fired", "сработало"), ("10-details", "детали")]
    rows = ["уровень 0", "уровень 1", "уровень 2", "уровень 3", "уровень 4"]
    cell, gap, label, head = 230, 10, 150, 56
    sheet = Image.new("RGB", (label + len(columns) * (cell + gap) + 16, head + len(rows) * (cell + gap) + 16), BACK)
    draw = ImageDraw.Draw(sheet)
    for column, (_, title) in enumerate(columns):
        x = label + column * (cell + gap) + cell // 2
        draw.text((x, head // 2 + 4), title, fill=INK, font=font(28, bold=True), anchor="mm")
    for row, title in enumerate(rows):
        y = head + row * (cell + gap)
        draw.text((label - 18, y + cell // 2), title.replace("уровень ", "ур. "), fill=GREEN, font=font(34, bold=True), anchor="rm")
        for column, (suffix, _) in enumerate(columns):
            x = label + column * (cell + gap)
            sheet.paste(fit(LEVELS / f"L{row}-{suffix}.png", cell, cell), (x, y))
            draw.rectangle((x, y, x + cell - 1, y + cell - 1), outline=EDGE, width=2)
    target = OUT / "s6-levels.png"
    return save(sheet, target)


def run_maps(run: int, name: str, labels: list[tuple[str, str]], levels=range(5)) -> Path:
    """Карты уровней прогона стенда, вырезанные из его общей картинки, с крупными подписями и легендой цифр."""
    source = Image.open(MAPS / f"tune2-{run}.png").convert("RGB")
    tiles = [source.crop((20 + 320 * level, 76, 320 + 320 * level, 376)) for level in levels]
    legend = source.crop((20, 386, 880, 408))
    tile, gap, head = 300, 26, 92
    width = len(tiles) * tile + (len(tiles) - 1) * gap + 40
    sheet = Image.new("RGB", (width, head + tile + 20 + round(legend.height * (width - 40) / legend.width) + 24), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    for index, (picture, (title, note)) in enumerate(zip(tiles, labels)):
        x = 20 + index * (tile + gap)
        draw.text((x, 30), title, fill=INK, font=font(30, bold=True), anchor="lm")
        draw.text((x, 68), note, fill=MUTED, font=font(22), anchor="lm")
        sheet.paste(picture, (x, head))
    scale = (width - 40) / legend.width
    legend = legend.resize((width - 40, round(legend.height * scale)), Image.LANCZOS)
    sheet.paste(legend, (20, head + tile + 16))
    target = OUT / f"{name}.png"
    return save(sheet, target)


def seeds_top() -> Path:
    """Верхние карты двух зёрен на 60 000 картинках: двойка одним пятном и двумя."""
    tiles, titles = [], [("зерно 10 (прогон 1569)", "двойка — одно пятно, М1 0.92"), ("зерно 3 (прогон 1568)", "двойка — два пятна, М1 0.50")]
    for run in (1569, 1568):
        tiles.append(Image.open(MAPS / f"tune2-{run}.png").convert("RGB").crop((1300, 76, 1600, 376)))
    tile, gap, head = 300, 40, 92
    sheet = Image.new("RGB", (2 * tile + gap + 40, head + tile + 24), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    for index, (picture, (title, note)) in enumerate(zip(tiles, titles)):
        x = 20 + index * (tile + gap)
        draw.text((x, 30), title, fill=INK, font=font(26, bold=True), anchor="lm")
        draw.text((x, 68), note, fill=MUTED, font=font(22), anchor="lm")
        sheet.paste(picture, (x, head))
    target = OUT / "s7-two-seeds.png"
    return save(sheet, target)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    a = lambda *names: [ATOMIC / f"{n}.png" for n in names]  # noqa: E731
    l = lambda *names: [LEVELS / f"{n}.png" for n in names]  # noqa: E731
    made = [
        pipeline(),
        run_maps(1569, "maps-60000", [("уровень 0", "штрих · цвет — угол"), ("уровень 1", "обрывок контура"),
                                      ("уровень 2", "часть цифры"), ("уровень 3", "почти вся цифра"),
                                      ("уровень 4", "цифра целиком")]),
        strip("s1-midline", a("1-01-raw", "1-02-work", "1-04-distance", "1-07-direction", "1-13-bridges", "1-14-midline"), titles=['MNIST 28×28', '×8 → 224×224', 'до края', 'поперёк линии', 'мостики', 'середина линии']),
        strip("s1-strokes", a("2-01-activation", "2-02-cluster", "2-03-field", "2-05-centre", "2-07-insert", "2-08-sticks"), titles=['соседи точки', 'кластер', 'поле детектора', 'центр и угол', 'без повторов', 'штрихи']),
        strip("s2-angle", a("3-01-stroke", "3-02-layer0", "3-05-layer3", "3-08-layer6", "3-11-layer9", "3-12-angle-code"), titles=['штрих', 'слой 1: 2 дуги', 'слой 4: 9 дуг', 'слой 7: 40 дуг', 'слой 10: 180', 'код угла']),
        strip("s2-place", a("4-01-offset", "4-03-place0", "4-06-place3", "4-09-place6", "4-12-place9", "4-13-place-code"), titles=['место в цифре', 'слой 1: 2×2', 'слой 4: 5×5', 'слой 7: 8×8', 'слой 10: 16×16', 'код места']),
        strip("s2-keyhole", a("5-01-place-bits", "5-02-mask", "5-03-content", "5-04-merge", "5-05-window"), titles=['биты места', 'маска круга', 'содержание', 'слияние цветов', 'код окна']),
        strip("s3-layout", a("6-01-dictionary", "6-02-start", "6-05-long16", "6-05-long256", "6-05-long2048", "6-final"), titles=['слить копии', 'случайный старт', '16 шагов', '256 шагов', '2048 шагов', 'карта уровня']),
        strip("s4-detectors", a("7-01-energy", "7-03-activation", "7-04-dbscan", "7-07-layer0", "7-09-layer2", "7-11-layer4"), titles=['энергия точек', 'активация', 'кластер DBSCAN', 'слой 1', 'слой 3', 'слой 5']),
        strip("s4-firing", a("8-01-similarity", "8-02-lit", "8-03-detector", "8-04-fired"), titles=['сходство с картой', 'зажглось', 'уровень E(d)', 'сработало']),
        strip("s5-detail", a("8-05-sigma", "8-06-detail", "8-07-details"), titles=['σ сильнейших', 'код детали', 'детали штрихов']),
        strip("s6-keyholes", a("9-01-inhibition", "9-02-window", "10-L2-window", "10-L3-window", "10-L4-window"), titles=['пятна', 'окно ур. 1', 'окно ур. 2', 'окно ур. 3', 'окно ур. 4']),
        level_grid(),
        strip("s7-levels", l("rec-2-sticks", "rec-L0-lit", "rec-L1-lit", "rec-L2-lit", "rec-L3-lit", "rec-L4-lit"), size=(260, 260), titles=['новая картинка', 'ур. 0 зажёгся', 'ур. 1', 'ур. 2', 'ур. 3', 'ур. 4']),
        strip("s7-answer", l("rec-votes", "rec-memory", "rec-reader"), size=(300, 300), titles=['голоса уровней', 'память', 'читатель']),
        seeds_top(),
    ]
    for path in made:
        with Image.open(path) as picture:
            print(f"{path.relative_to(ROOT)}  {picture.width}×{picture.height}  {path.stat().st_size // 1024} КБ")


if __name__ == "__main__":
    main()
