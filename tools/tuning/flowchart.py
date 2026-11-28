# -*- coding: utf-8 -*-
"""Блок-схема DAMP по атомарным шагам — классический вид: шаг — прямоугольник, условие — ромб (да — вниз, нет — вбок),
петля «для каждого» — ромб «есть ещё…?» с телом и стрелкой назад слева, выход — справа; цикл уровней — тело, затем
ромб с возвратом через кружок-переход. Картинки — клетки атомарной схемы (tools/tuning/atomic_diagram.py → flow.json).

    python tools/tuning/flowchart.py [папка=data/maps/atomic]

Холст квадратный: колонки слева направо, в колонке — сверху вниз; переход между колонками — кружок с номером.
Автор 2026-11-21: «вот как я вижу диаграмму, расположи так по шагам», «атомарно».
"""
import json
import math
import os
import sys

from PIL import Image, ImageDraw, ImageFont

FOLDER = sys.argv[1] if len(sys.argv) > 1 else "data/maps/atomic"
ITEMS = {}
for item in json.load(open(os.path.join(FOLDER, "flow.json"), encoding="utf-8")):
    if item[0] in ("cell", "if"):
        ITEMS[item[1].split(" ", 1)[0]] = {"title": item[1].split(" ", 1)[1], "path": item[2], "note": item[3]}

BG, BOX, EDGE, TEXT, MUTED = (13, 14, 18), (36, 39, 50), (110, 115, 132), (240, 240, 245), (170, 172, 182)
YELLOW, ORANGE, GREEN, GREY = (245, 205, 70), (255, 150, 60), (70, 205, 140), (130, 134, 148)
BW, PIC_W, PIC_H, GAL_H = 760, 700, 400, 560  # блок, картинка, галерея
DW_, DH = 600, 230  # ромб
SIDE_W = 300
GAP = 56
CW = 1500  # ширина колонки
FONT = {}


def F(size, bold=False):
    key = (size, bold)
    if key not in FONT:
        FONT[key] = ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)
    return FONT[key]


PROBE = ImageDraw.Draw(Image.new("RGB", (8, 8)))


def wrap(text, font, width, limit=None):
    out, line = [], ""
    for word in str(text).split(" "):
        cand = (line + " " + word).strip()
        if PROBE.textlength(cand, font=font) <= width:
            line = cand
        else:
            if line:
                out.append(line)
            line = word
    if line:
        out.append(line)
    return out[:limit] if limit else out


# ---------------------------------------------------------------- узлы
class Node:
    def height(self) -> int:
        raise NotImplementedError

    def draw(self, d, sheet, xc, y, depth) -> None:
        raise NotImplementedError


class Start(Node):
    def __init__(self, text):
        self.text = text

    def height(self):
        return 90

    def draw(self, d, sheet, xc, y, depth):
        w = 620
        d.rounded_rectangle((xc - w / 2, y, xc + w / 2, y + 90), radius=45, fill=(30, 60, 45), outline=GREEN, width=3)
        lines = wrap(self.text, F(30, True), w - 60, 2)
        ty = y + 45 - 19 * len(lines)
        for line in lines:
            d.text((xc - PROBE.textlength(line, font=F(30, True)) / 2, ty), line, font=F(30, True), fill=TEXT)
            ty += 38


class Section(Node):
    def __init__(self, text, colour):
        self.text, self.colour = text, colour

    def height(self):
        return 70

    def draw(self, d, sheet, xc, y, depth):
        d.rounded_rectangle((xc - BW / 2, y, xc + BW / 2, y + 70), radius=14, fill=tuple(int(v * 0.35) for v in self.colour),
                            outline=self.colour, width=3)
        line = wrap(self.text, F(28, True), BW - 30, 1)[0]
        d.text((xc - PROBE.textlength(line, font=F(28, True)) / 2, y + 18), line, font=F(28, True), fill=TEXT)


def paste_fit(sheet, path, box):
    x0, y0, x1, y1 = box
    if not path or not os.path.exists(path):
        return
    im = Image.open(path).convert("RGB")
    k = min((x1 - x0) / im.width, (y1 - y0) / im.height)
    im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS)
    sheet.paste(im, (int(x0 + (x1 - x0 - im.width) / 2), int(y0 + (y1 - y0 - im.height) / 2)))


def gallery(sheet, d, paths, box, labels=None):
    """Картинки всех проходов петли сеткой; подпись — номер прохода."""
    x0, y0, x1, y1 = box
    n = len(paths)
    cols = min(5, n)
    rows = math.ceil(n / cols)
    cw, ch = (x1 - x0) / cols, (y1 - y0) / rows
    for i, path in enumerate(paths):
        r, c = divmod(i, cols)
        bx, by = x0 + c * cw, y0 + r * ch
        paste_fit(sheet, path, (bx + 3, by + 22, bx + cw - 3, by + ch - 2))
        if labels:
            d.text((bx + 6, by), labels[i], font=F(20, True), fill=MUTED)


class Process(Node):
    """Шаг: заголовок, картинка (или галерея проходов), пояснение."""

    def __init__(self, key=None, title=None, note=None, paths=None, labels=None, colour=None):
        item = ITEMS.get(key, {}) if key else {}
        self.title = title or (f"{key} {item.get('title', '')}" if key else "")
        self.note = item.get("note", "") if note is None else note
        self.paths = paths if paths is not None else ([item["path"]] if item.get("path") else [])
        self.labels = labels
        self.colour = colour or EDGE

    def lines(self):
        return wrap(self.title, F(30, True), BW - 30, 2), wrap(self.note, F(24), BW - 30, 3)

    def height(self):
        title, note = self.lines()
        pic = 0 if not self.paths else (GAL_H if len(self.paths) > 1 else PIC_H)
        return 16 + 38 * len(title) + 8 + pic + 8 + 30 * len(note) + 14

    def draw(self, d, sheet, xc, y, depth):
        h = self.height()
        x0 = xc - BW / 2
        d.rounded_rectangle((x0, y, x0 + BW, y + h), radius=16, fill=BOX, outline=self.colour, width=2)
        title, note = self.lines()
        ty = y + 16
        for line in title:
            d.text((x0 + 15, ty), line, font=F(30, True), fill=TEXT)
            ty += 38
        ty += 8
        if self.paths:
            pic = GAL_H if len(self.paths) > 1 else PIC_H
            box = (xc - PIC_W / 2, ty, xc + PIC_W / 2, ty + pic)
            if len(self.paths) > 1:
                gallery(sheet, d, self.paths, box, self.labels)
            else:
                paste_fit(sheet, self.paths[0], box)
            ty += pic
        ty += 8
        for line in note:
            d.text((x0 + 15, ty), line, font=F(24), fill=MUTED)
            ty += 30


class Small(Node):
    """Короткий шаг без картинки: «следующий …», «k = k + 1»."""

    def __init__(self, text):
        self.text = text

    def height(self):
        return 80

    def draw(self, d, sheet, xc, y, depth):
        w = 520
        d.rectangle((xc - w / 2, y, xc + w / 2, y + 80), fill=BOX, outline=ORANGE, width=2)
        line = wrap(self.text, F(28, True), w - 30, 1)[0]
        d.text((xc - PROBE.textlength(line, font=F(28, True)) / 2, y + 24), line, font=F(28, True), fill=TEXT)


def diamond(d, xc, y, text, colour=YELLOW):
    pts = [(xc, y), (xc + DW_ / 2, y + DH / 2), (xc, y + DH), (xc - DW_ / 2, y + DH / 2)]
    d.polygon(pts, fill=(56, 46, 16), outline=colour, width=3)
    lines = wrap(text, F(27, True), DW_ * 0.58, 4)
    ty = y + DH / 2 - 17 * len(lines)
    for line in lines:
        d.text((xc - PROBE.textlength(line, font=F(27, True)) / 2, ty), line, font=F(27, True), fill=(255, 240, 200))
        ty += 34


def arrow(d, points, colour=(205, 207, 215), width=5, head=20):
    d.line(points, fill=colour, width=width, joint="curve")
    (x0, y0), (x1, y1) = points[-2], points[-1]
    a = math.atan2(y1 - y0, x1 - x0)
    d.polygon([(x1, y1), (x1 - head * math.cos(a) + head * 0.55 * math.sin(a), y1 - head * math.sin(a) - head * 0.55 * math.cos(a)),
               (x1 - head * math.cos(a) - head * 0.55 * math.sin(a), y1 - head * math.sin(a) + head * 0.55 * math.cos(a))], fill=colour)


def label(d, xy, text, colour):
    d.text(xy, text, font=F(26, True), fill=colour)


class Decision(Node):
    """Условие: ромб с вопросом; вниз (да) — блок с картинкой того, что прошло; вбок (нет) — исход ветки.
    rejoin — боковая ветка возвращается в линию (действие, а не отброс); connector — вбок к кружку-переходу."""

    def __init__(self, key=None, question=None, side="", down="да", result=None, keys=None, labels=None, rejoin=False,
                 connector=None, note=None):
        keys = keys or ([key] if key else [])
        item = ITEMS.get(keys[0], {}) if keys else {}
        self.question = question or item.get("title", "")
        self.side, self.down, self.rejoin, self.connector = side, down, rejoin, connector
        paths = [ITEMS[k]["path"] for k in keys if k in ITEMS]
        number = keys[0] if len(keys) == 1 else f"{keys[0]}…{keys[-1]}"
        self.block = Process(title=f"{number} {result or down}", note=item.get("note", "") if note is None else note,
                             paths=paths, labels=labels, colour=YELLOW)

    def height(self):
        return DH + GAP + self.block.height()

    def draw(self, d, sheet, xc, y, depth):
        diamond(d, xc, y, self.question)
        arrow(d, [(xc, y + DH), (xc, y + DH + GAP - 4)])
        label(d, (xc + 14, y + DH + 6), self.down.split(" ")[0], YELLOW)
        self.block.draw(d, sheet, xc, y + DH + GAP, depth)
        if not self.side:
            return
        sx = xc + DW_ / 2 + 70
        sy = y + DH / 2
        word, _, rest = self.side.partition(" → ")
        if self.connector:
            arrow(d, [(xc + DW_ / 2, sy), (sx + 40, sy)], colour=GREEN, width=6)
            d.ellipse((sx + 40, sy - 40, sx + 120, sy + 40), outline=GREEN, width=5, fill=(20, 50, 35))
            d.text((sx + 80 - PROBE.textlength(self.connector, font=F(34, True)) / 2, sy - 20), self.connector, font=F(34, True), fill=GREEN)
            for i, line in enumerate(wrap(rest, F(24, True), 300, 3)):
                d.text((sx + 135, sy - 40 + 30 * i), line, font=F(24, True), fill=GREEN)
            label(d, (xc + DW_ / 2 + 8, sy - 38), word, YELLOW)
            return
        arrow(d, [(xc + DW_ / 2, sy), (sx, sy)])
        label(d, (xc + DW_ / 2 + 8, sy - 38), word, YELLOW)
        lines = wrap(rest, F(24, True), SIDE_W - 24, 4)
        bh = 22 + 30 * len(lines)
        box = (sx, sy - bh / 2, sx + SIDE_W, sy + bh / 2)
        if self.rejoin:
            d.rectangle(box, fill=BOX, outline=ORANGE, width=2)
        else:
            d.rounded_rectangle(box, radius=int(bh / 2), fill=(30, 31, 38), outline=GREY, width=2)
        ty = sy - bh / 2 + 11
        for line in lines:
            d.text((sx + 12, ty), line, font=F(24, True), fill=TEXT if self.rejoin else MUTED)
            ty += 30
        if self.rejoin:
            bottom = y + self.height() + GAP / 2
            arrow(d, [(sx + SIDE_W / 2, sy + bh / 2), (sx + SIDE_W / 2, bottom), (xc + 6, bottom)], colour=ORANGE, width=4, head=16)


class Loop(Node):
    """Петля «для каждого»: ромб «есть ещё…?» → да: тело → «следующий» → назад к ромбу (слева); нет — выход справа."""

    def __init__(self, question, body, step):
        self.question, self.body, self.step = question, body, Small(step)

    def height(self):
        return 24 + DH + GAP + sum(node.height() + GAP for node in self.body) + self.step.height() + GAP + 30

    def draw(self, d, sheet, xc, y, depth):
        top = y + 24
        d.line([(xc, y - 2), (xc, top)], fill=(205, 207, 215), width=5)
        diamond(d, xc, top, self.question, colour=ORANGE)
        yy = top + DH
        arrow(d, [(xc, yy), (xc, yy + GAP - 4)])
        label(d, (xc + 14, yy + 6), "да", ORANGE)
        yy += GAP
        for node in self.body:
            node.draw(d, sheet, xc, yy, depth + 1)
            yy += node.height()
            arrow(d, [(xc, yy), (xc, yy + GAP - 4)])
            yy += GAP
        self.step.draw(d, sheet, xc, yy, depth + 1)
        yy += self.step.height()
        lane = xc - BW / 2 - 60 - 48 * (3 - depth)  # внешние петли — дальше от линии: стрелки не пересекаются
        d.line([(xc, yy), (xc, yy + 20), (lane, yy + 20), (lane, y + 8)], fill=ORANGE, width=5, joint="curve")
        arrow(d, [(lane, y + 8), (xc - 6, y + 8)], colour=ORANGE, width=5, head=18)
        exit_lane = xc + BW / 2 + 300 + 45 * (3 - depth)
        end = y + self.height() - 4
        d.line([(xc + DW_ / 2, top + DH / 2), (exit_lane, top + DH / 2), (exit_lane, end - 18)], fill=GREY, width=5, joint="curve")
        arrow(d, [(exit_lane, end - 18), (xc + 6, end - 18)], colour=GREY, width=5, head=18)
        label(d, (xc + DW_ / 2 + 10, top + DH / 2 - 38), "нет", GREY)


class Entry(Node):
    """Кружок-переход, в который возвращается цикл уровней (до шага «Код окна»)."""

    def __init__(self, mark, text):
        self.mark, self.text = mark, text

    def height(self):
        return 100

    def draw(self, d, sheet, xc, y, depth):
        cx = xc - BW / 2 - 40
        d.ellipse((cx - 40, y + 10, cx + 40, y + 90), outline=GREEN, width=5, fill=(20, 50, 35))
        d.text((cx - PROBE.textlength(self.mark, font=F(34, True)) / 2, y + 30), self.mark, font=F(34, True), fill=GREEN)
        arrow(d, [(cx + 40, y + 50), (xc - 6, y + 50)], colour=GREEN, width=6)
        for i, line in enumerate(wrap(self.text, F(24, True), 520, 2)):
            d.text((xc + 20, y + 6 + 30 * i), line, font=F(24, True), fill=GREEN)


# ---------------------------------------------------------------- схема
def P(key, **kw):
    return Process(key, **kw)


def D(key, side, **kw):
    return Decision(key, side=side, **kw)


L_ANGLE = [f"3.{k}" for k in range(2, 12)]
L_PLACE = [f"4.{k}" for k in range(3, 13)]
LAYER_LABELS = [f"слой {k}" for k in range(10)]
LEVELS = sorted({key.split(".")[1][:-1] for key in ITEMS if key.startswith("10.")}, key=int)
VOTES = [key for key in ITEMS if key.startswith("11.") and "Уровень" in ITEMS[key]["title"]]

SCHEME = [
    Start("Начало: обучающая картинка MNIST"),
    Section("1. Картинка → середина линии", (90, 170, 255)),
    P("1.1"), P("1.2"),
    D("1.3", "нет → фон", result="да: чернила"),
    P("1.4"), P("1.5"), P("1.6"), P("1.7"),
    D("1.8", "нет → не середина линии", result="да: вершина гребня"),
    D("1.9", "нет → слабая точка (пойдёт в мостики)", result="да: сильная"),
    D("1.10", "нет → пятно, не линия", result="да: линия"),
    D("1.11", "нет → кончик линии", result="да: точка гребня"),
    P("1.12"),
    D("1.13", "нет → разрыв остаётся", result="да: мостик по слабым точкам"),
    P("1.14"),
    Section("2. Точки → штрихи (детекторы над точками, §6.4)", (90, 170, 255)),
    Loop("есть ещё точка — центр детектора c?", [
        P("2.1"), P("2.2"), P("2.3"),
        D("2.4", "нет → не кандидат", result="да: кандидат"),
        P("2.5"),
    ], "следующая точка"),
    P("2.6"),
    Loop("есть ещё кандидат (по порядку)?", [
        D("2.7", "да → пропустить", down="нет", result="нет: взять штрих"),
    ], "следующий кандидат"),
    P("2.8"),
    Section("3. Код угла штриха (§4.2, §4.4)", (255, 160, 70)),
    P("3.1"),
    Loop("есть ещё слой детекторов угла k (широкий → узкий)?", [
        Loop("есть ещё детектор слоя k?", [
            Decision(keys=L_ANGLE, question="угол внутри дуги детектора?", side="нет → детектор молчит",
                     result="да: бит детектора = 1 (все 10 слоёв)", labels=LAYER_LABELS,
                     note="у слоя k — свои детекторы-дуги; ширина дуги падает от 180° до 7°; бит — случайный в полосе слоя"),
        ], "следующий детектор"),
    ], "k = k + 1"),
    P("3.12"),
    Section("4. Место штриха — от центра цифры (§4.3, П.4)", (255, 160, 70)),
    P("4.1"), P("4.2"),
    Loop("есть ещё слой кругов места k (широкий → узкий)?", [
        Loop("есть ещё круг слоя k?", [
            Decision(keys=L_PLACE, question="точка внутри круга?", side="нет → круг молчит",
                     result="да: бит круга = 1, ×39 копий (все 10 слоёв)", labels=LAYER_LABELS,
                     note="у слоя k — n×n кругов; радиус падает от 2.1 до 0.26 радиуса цифры"),
        ], "следующий круг"),
    ], "k = k + 1"),
    P("4.13"),
    Section("5–9. Виток уровня: код окна → карта → детекторы → детали", (70, 205, 140)),
    Entry("У", "сюда возвращается цикл уровней (из 9.4): детали уровня — элементы следующего"),
    P("5.1"),
    Loop("есть ещё сработавший круг места d?", [P("5.2")], "следующий круг"),
    P("5.3"),
    D("5.4", "да → слить цвета: первыми уходят узкие слои", down="нет", result="код содержания", rejoin=True),
    P("5.5"),
    D("6.1", "да → слить с образцом", down="нет", result="новый образец карты", rejoin=True),
    P("6.2"),
    Loop("дальняя фаза: радиус пар > наименьшего?", [
        P("6.3"),
        D("6.4", "нет → оставить", result="да: обменять"),
        D("6.9", "да → радиус меньше, λ выше", down="нет", result="доля обменов по шагам", rejoin=True),
    ], "следующий шаг"),
    P(None, title="6.5…6.8 Карта по ходу дальней фазы", note="после 1, 16, 256 и 2048 шагов: от случайной к пинвилам",
      paths=[ITEMS[k]["path"] for k in ("6.5", "6.6", "6.7", "6.8") if k in ITEMS], labels=["1", "16", "256", "2048"]),
    P("6.10"), P("6.11"),
    P("7.1"),
    Loop("есть ещё слой детекторов k = 0…4?", [
        Loop("меньше 400 попыток подряд без нового детектора?", [
            D("7.2", "нет → другой центр", result="да: центр"),
            P("7.3"), P("7.4"), P("7.5"),
            D("7.6", "нет → попытка не засчитана", result="да: новый детектор слоя"),
        ], "следующая попытка"),
    ], "k = k + 1"),
    P(None, title="7.7…7.11 Детекторы по слоям", note="слои 0…4: от крупных областей карты к мелким кластерам",
      paths=[ITEMS[k]["path"] for k in ("7.7", "7.8", "7.9", "7.10", "7.11") if k in ITEMS], labels=[f"слой {k}" for k in range(5)]),
    Loop("есть ещё окно уровня?", [
        P("8.1"),
        D("8.2", "нет → точка не горит", result="да: точка зажглась"),
        Loop("есть ещё детектор d?", [
            P("8.3"),
            D("8.4", "нет → детектор молчит", result="да: сработал"),
        ], "следующий детектор"),
        D("8.5", "нет → бит не берётся", result="да: бит в код детали"),
        P("8.6"),
    ], "следующее окно"),
    P("8.7"),
    Loop("есть ещё деталь?", [D("9.1", "да → погаснуть", down="нет", result="нет: деталь идёт выше")], "следующая деталь"),
    Loop("есть ещё центр окна следующего уровня?", [
        D("9.2", "нет → растить радиус", result="да: окно", rejoin=True),
        D("9.3", "да → слить с образцом", down="нет", result="окна следующего уровня", rejoin=True),
    ], "следующий центр"),
    D("9.4", "нет → следующий уровень: назад к «Код окна»", result="да: обучение закончено", connector="У"),
    Section("10. Что дал каждый виток (данные уровней)", (70, 205, 140)),
    P(None, title="10.a Окна уровней", paths=[ITEMS[f"10.{L}a"]["path"] for L in LEVELS], labels=[f"ур. {L}" for L in LEVELS],
      note="круг — окно уровня в середине четвёрки; от части цифры к целой"),
    P(None, title="10.b Карты уровней зажглись", paths=[ITEMS[f"10.{L}b"]["path"] for L in LEVELS], labels=[f"ур. {L}" for L in LEVELS],
      note="цвет — цифра; белые — точки, зажжённые окном; круги — сработавшие детекторы"),
    P(None, title="10.c Коды деталей уровней", paths=[ITEMS[f"10.{L}c"]["path"] for L in LEVELS], labels=[f"ур. {L}" for L in LEVELS],
      note="96 бит, около 10 единиц: σ сильнейших детекторов"),
    Section("11. Распознавание: та же четвёрка как новая картинка (§7.1.13)", (255, 205, 70)),
    Loop("есть ещё уровень?", [
        Decision(keys=VOTES, question="доля голосов уровня ≥ порога знания?", side="нет → уровень не знает",
                 result="да: уровень знает (голоса уровней)", labels=[ITEMS[k]["title"].split(":")[0] for k in VOTES],
                 note="у уровня — доли голосов его зажжённых точек за цифры"),
    ], "следующий уровень"),
    D(VOTES and f"11.{len(VOTES) + 1}" or "11.5", "нет → по частям / не знаю", result="да: ответ памяти целиком"),
    P(f"11.{len(VOTES) + 2}"),
    Start("Конец: ответ"),
]


# ---------------------------------------------------------------- колонки и холст
def layout(height_limit):
    """Колонки: узлы схемы сверху вниз, перенос — между узлами верхнего уровня."""
    cols, col, used = [], [], 0
    for node in SCHEME:
        h = node.height() + GAP
        if col and used + h > height_limit:
            cols.append(col)
            col, used = [], 0
        col.append(node)
        used += h
    cols.append(col)
    return cols


def main() -> None:
    total = sum(node.height() + GAP for node in SCHEME)
    tallest = max(node.height() + GAP for node in SCHEME)
    best = None
    for limit in range(max(tallest, 4000), total + 1, 200):
        cols = layout(limit)
        width = len(cols) * CW + 400
        height = max(sum(n.height() + GAP for n in c) for c in cols) + 700
        score = abs(width - height)
        if best is None or score < best[0]:
            best = (score, limit, cols, width, height)
    _, limit, cols, width, height = best
    S = max(width, height)
    sheet = Image.new("RGB", (S, S), BG)
    d = ImageDraw.Draw(sheet)
    d.text((200, 70), "DAMP по шагам: блок-схема на одной четвёрке (атомарно)", font=F(110, True), fill=(255, 255, 255))
    legend = ("прямоугольник — шаг с картинкой настоящих данных · ромб — условие: да — вниз, нет — вбок · оранжевый ромб — петля «есть ещё…?»: "
              "тело, «следующий», стрелка назад слева, выход справа · зелёный кружок — переход: цикл уровней (У), колонки (1, 2, …) · "
              "читать сверху вниз, колонки слева направо")
    y = 230
    for line in wrap(legend, F(44), S - 400):
        d.text((200, y), line, font=F(44), fill=MUTED)
        y += 58
    top = y + 120
    ox = 200 + (S - width) // 2
    for c, col in enumerate(cols):
        xc = ox + c * CW + CW / 2 - 60
        yy = top
        if c > 0:
            d.ellipse((xc - 40, yy - 110, xc + 40, yy - 30), outline=GREEN, width=5, fill=(20, 50, 35))
            d.text((xc - PROBE.textlength(str(c + 1), font=F(34, True)) / 2, yy - 90), str(c + 1), font=F(34, True), fill=GREEN)
            arrow(d, [(xc, yy - 30), (xc, yy - 4)], colour=GREEN)
        for i, node in enumerate(col):
            node.draw(d, sheet, xc, yy, 0)
            yy += node.height()
            if i + 1 < len(col):
                arrow(d, [(xc, yy), (xc, yy + GAP - 4)])
            yy += GAP
        if c + 1 < len(cols):
            yy -= GAP
            arrow(d, [(xc, yy), (xc, yy + 30)], colour=GREEN)
            d.ellipse((xc - 40, yy + 30, xc + 40, yy + 110), outline=GREEN, width=5, fill=(20, 50, 35))
            d.text((xc - PROBE.textlength(str(c + 2), font=F(34, True)) / 2, yy + 50), str(c + 2), font=F(34, True), fill=GREEN)
    out = os.path.join(FOLDER, "flowchart-four.png")
    sheet.save(out, optimize=True)
    sheet.resize((S // 4, S // 4), Image.LANCZOS).save(os.path.join(FOLDER, "flowchart-four-preview.png"))
    print(out, sheet.size, "колонок", len(cols), "предел колонки", limit)


if __name__ == "__main__":
    main()
