# -*- coding: utf-8 -*-
"""Схема витка цикла на каждый уровень: слева — статичная часть (алгоритм витка с петлями и условиями, одна и та же для
всех уровней), справа — часть уровня (картинки подшагов и числа этого уровня на четвёрке), напротив своего шага.

    python tools/tuning/level_cycle.py [папка=data/maps/level-diagrams] [журнал стенда tune2_log.jsonl] [прогон=648]

Берёт готовые картинки подшагов (level_diagram.py → panels/, summary.json) и мерки прогона из журнала стенда.
Автор 2026-11-21: «хочу диаграммы (с циклами и условиями) для каждого уровня, в которых одна часть статична, а вторая
зависит от уровня».
"""
import json
import math
import os
import sys

from PIL import Image, ImageDraw, ImageFont

FOLDER = sys.argv[1] if len(sys.argv) > 1 else "data/maps/level-diagrams"
LOG = sys.argv[2] if len(sys.argv) > 2 else None
RUN = int(sys.argv[3]) if len(sys.argv) > 3 else 648
BG, CELL, EDGE = (13, 14, 18), (36, 39, 50), (95, 100, 115)
GREEN, ORANGE, YELLOW, GREY, BLUE = (70, 205, 140), (240, 140, 60), (245, 205, 70), (215, 217, 225), (90, 160, 255)


def F(size, bold=False):
    return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)


F_T, F_SUB, F_HEAD, F_BOX, F_DYN, F_SMALL = F(110, True), F(44), F(60, True), F(38), F(38), F(32)
W, LEFT_W, RIGHT_X = 7200, 2900, 3350
ROW_H, GAP, TOP = 680, 110, 560

# Шаги витка: (вид, текст). Вид: box — действие; if — условие (ромб, справа — ветка «нет/да»); loop — действие с петлёй
# внутри (мини-схема с оранжевой стрелкой назад).
STEPS = [
    ("box", "1. Элементы уровня: детали прошлого уровня той же картинки (на уровне 0 — штрихи)"),
    ("if", "2. Кластер ≥ N похожих соседей?|да → окно = кластер; нет → окно на всю картинку"),
    ("box", "3. Код окна = место всех элементов (OR) + содержание: код элемента под XOR-маской своего места; пулинг «сильнейший»"),
    ("if", "4. Похожие окна (косинус ≥ 0.8)?|да → слить к одному образцу"),
    ("loop", "5. Первая подача — раскладка|пара точек (первая по −ln Ê)~◇ энергия после обмена выше?~обменять|петля: дальняя фаза, пока доля обменов > порога; колокол сжимается; потом ближняя фаза"),
    ("loop", "6. Детекторы над картой (5 слоёв)|центр с энергией ≥ μ_e~кластер DBSCAN → радиус~◇ центр не перекрыт? → в слой|петля: пока 400 попыток подряд без нового детектора; затем следующий слой"),
    ("if", "7. Вторая подача: сходство кода окна с точкой ≥ λ_a и энергия активации ≥ 0.5?|да → точка зажглась"),
    ("box", "8. Детекторы срабатывают: E(d) = Σ a·e / e_d, нормировка по обучающим окнам"),
    ("if", "9. Детектор среди σ = 10 сильнейших?|да → его случайный бит в код детали"),
    ("box", "10. Код детали (эмбеддинг окна) ≈ σ / плотность бит"),
    ("if", "11. Рядом похожая деталь той же картинки (≥ 0.8)?|да → остаётся сильнейшая (пятна)"),
    ("if", "12. Окно уровня видит ≥ 90 % цифры? (с 4 уровнями — ≥ 99 %)|нет → детали — элементы окон уровня k + 1 (шаг 1); да → распознавание"),
]


def wrap(d, s, fnt, width):
    out, line = [], ""
    for word in s.split(" "):
        probe = (line + " " + word).strip()
        if d.textlength(probe, font=fnt) <= width:
            line = probe
        else:
            out.append(line)
            line = word
    out.append(line)
    return out


def text(d, x, y, s, fnt, width, fill=GREY, gap=1.25, center=False):
    for line in wrap(d, s, fnt, width):
        dx = (width - d.textlength(line, font=fnt)) / 2 if center else 0
        d.text((x + dx, y), line, font=fnt, fill=fill)
        y += int(fnt.size * gap)
    return y


def arrow(d, pts, colour=GREY, width=8, head=30):
    d.line(pts, fill=colour, width=width, joint="curve")
    (x1, y1), (x2, y2) = pts[-2], pts[-1]
    a = math.atan2(y2 - y1, x2 - x1)
    d.polygon([(x2, y2), (x2 - head * math.cos(a) + head * 0.6 * math.sin(a), y2 - head * math.sin(a) - head * 0.6 * math.cos(a)),
               (x2 - head * math.cos(a) - head * 0.6 * math.sin(a), y2 - head * math.sin(a) + head * 0.6 * math.cos(a))], fill=colour)


def row_y(i):
    return TOP + i * (ROW_H + GAP)


def draw_static(d):
    """Статичная часть: одна и та же на всех уровнях."""
    x0, w = 160, LEFT_W - 160
    d.text((x0, TOP - 110), "Алгоритм витка — одинаков на всех уровнях", font=F_HEAD, fill=GREEN)
    centres = []
    for i, (kind, s) in enumerate(STEPS):
        y = row_y(i)
        if kind == "box":
            d.rounded_rectangle((x0, y, x0 + w, y + ROW_H), radius=28, fill=CELL, outline=EDGE, width=2)
            lines = wrap(d, s, F_BOX, w - 60)
            yy = y + ROW_H / 2 - len(lines) * 24
            text(d, x0 + 30, yy, s, F_BOX, w - 60, fill=(240, 240, 245), center=True)
        elif kind == "if":
            question, branch = s.split("|")
            cx, cy, dw, dh = x0 + w * 0.42, y + ROW_H / 2, w * 0.8, ROW_H * 0.92
            d.polygon([(cx, cy - dh / 2), (cx + dw / 2, cy), (cx, cy + dh / 2), (cx - dw / 2, cy)], fill=(56, 46, 16), outline=YELLOW, width=3)
            lines = wrap(d, question, F_BOX, dw * 0.58)
            yy = cy - len(lines) * 24
            for line in lines:
                d.text((cx - d.textlength(line, font=F_BOX) / 2, yy), line, font=F_BOX, fill=(255, 240, 200))
                yy += 48
            text(d, cx + dw / 2 + 20, y + 30, branch, F_SMALL, x0 + w - (cx + dw / 2 + 20), fill=YELLOW)
        else:
            title, chain, loop = s.split("|")
            d.rounded_rectangle((x0, y, x0 + w, y + ROW_H), radius=28, fill=CELL, outline=EDGE, width=2)
            d.text((x0 + 30, y + 20), title, font=F(42, True), fill=(240, 240, 245))
            items = chain.split("~")
            bw = (w - 60 - 50 * (len(items) - 1)) / len(items)
            by, bh = y + 110, ROW_H - 290
            for k, it in enumerate(items):
                bx = x0 + 30 + k * (bw + 50)
                if it.startswith("◇"):
                    cx, cy = bx + bw / 2, by + bh / 2
                    d.polygon([(cx, by), (bx + bw, cy), (cx, by + bh), (bx, cy)], fill=(56, 46, 16), outline=YELLOW, width=3)
                    lines = wrap(d, it[1:].strip(), F_SMALL, bw * 0.62)
                    yy = cy - len(lines) * 20
                    for line in lines:
                        d.text((cx - d.textlength(line, font=F_SMALL) / 2, yy), line, font=F_SMALL, fill=(255, 240, 200))
                        yy += 40
                else:
                    d.rounded_rectangle((bx, by, bx + bw, by + bh), radius=18, fill=(48, 52, 66), outline=EDGE, width=2)
                    lines = wrap(d, it, F_SMALL, bw - 30)
                    yy = by + bh / 2 - len(lines) * 20
                    for line in lines:
                        d.text((bx + bw / 2 - d.textlength(line, font=F_SMALL) / 2, yy), line, font=F_SMALL, fill=(235, 235, 240))
                        yy += 40
                if k:
                    arrow(d, [(bx - 46, by + bh / 2), (bx - 4, by + bh / 2)], width=6, head=22)
            first, last = x0 + 30 + bw / 2, x0 + 30 + (len(items) - 1) * (bw + 50) + bw / 2
            ly = by + bh + 50
            arrow(d, [(last, by + bh + 4), (last, ly), (first, ly), (first, by + bh + 8)], ORANGE, 6, 22)
            text(d, x0 + 30, ly + 14, loop, F_SMALL, w - 60, fill=ORANGE)
        centres.append(y + ROW_H / 2)
        if i:
            arrow(d, [(x0 + w * 0.42, row_y(i) - GAP + 6), (x0 + w * 0.42, row_y(i) - 6)], GREY, 8, 30)
    # Петля цикла: из шага 12 («нет») назад к шагу 1 — по левому краю.
    y12, y1 = centres[-1], centres[0]
    arrow(d, [(x0 + w * 0.42 - (LEFT_W - 160) * 0.4, y12), (60, y12), (60, y1), (x0 - 6, y1)], GREEN, 12, 40)
    d.text((70, (y1 + y12) / 2), "нет:\nследующий\nуровень", font=F_SMALL, fill=GREEN)
    return centres


def main() -> None:
    summary = json.load(open(os.path.join(FOLDER, "summary.json"), encoding="utf-8"))
    entry = None
    if LOG and os.path.exists(LOG):
        for line in open(LOG, encoding="utf-8"):
            e = json.loads(line)
            if e["run"] == RUN:
                entry = e
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    H = TOP + len(STEPS) * (ROW_H + GAP) + 120
    neighbours = {1: 10, 2: 3}
    shares = {1: 1.15, 2: 0.5}
    for item in summary:
        L = item["L"]
        f = item["files"]
        sheet = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(sheet)
        centres = draw_static(d)
        honest = (entry or {}).get("honest", [])
        m1 = (entry or {}).get("m1_main", [])
        shapes = (entry or {}).get("detector_shapes", [])
        zero = (entry or {}).get("level0") or {}
        if L == 0:
            level_honest, level_m1, shape = zero.get("honest"), zero.get("m1_main"), None
        else:
            level_honest = honest[L - 1] if L - 1 < len(honest) else None
            level_m1 = m1[L - 1] if L - 1 < len(m1) else None
            shape = shapes[L - 1] if L - 1 < len(shapes) else None
        d.text((160, 60), f"Уровень {L}: виток цикла — алгоритм и данные уровня (четвёрка MNIST train №{item['mnist']})", font=F_T, fill=(255, 255, 255))
        text(d, 160, 220, "слева — алгоритм витка, одинаковый на всех уровнях (петли — оранжевые, условия — ромбы); справа — что этот шаг "
             f"дал на уровне {L} для этой четвёрки (лучшая опора: перекрытие угла 0.85, код окна до 8192, 4 уровня, зерно 0, прогон {RUN})",
             F_SUB, W - 320, fill=(185, 188, 198))
        d.text((RIGHT_X, TOP - 110), f"Уровень {L} — данные", font=F_HEAD, fill=BLUE)
        n = neighbours.get(L, 14)
        share = shares.get(L, 2.0)
        dyn = [
            (f[0], "штрихи цифры" if L <= 1 else f"детали уровня {L - 1}: {item['elements']} на этой картинке"),
            (f[1], "окно — один штрих, место от центра цифры" if L == 0 else
             f"N = {n} соседей; нижний предел радиуса — {share} поля детали · окно: радиус {item['radius']:.1f} клетки, в окне {item['members']} элементов, "
             f"видит ~{100 * item['sees']:.0f} % штрихов цифры"),
            (f[2], f"код окна {item['window_bits']} бит, единиц {item['window_ones']}: место {item['place']} бит (оранж.), содержание — остальное (синие)"),
            (None, f"после слияния на карте {item['points']} точек"),
            (f[3], f"карта уровня {L}: {item['points']} точек; " + (f"честная сходимость {level_honest:.3f}" if level_honest is not None else "")
             + (f", М1 {level_m1:.3f}" if level_m1 is not None else "") + ("; цвет — угол штриха" if L == 0 else "; цвет — цифра")),
            (f[4], f"{item['detectors']} детекторов в {item['layers']} слоях" + (f": по слоям {shape['count']}, медиана радиуса {shape['radius']}" if shape else "")),
            (f[5], f"зажглось {item['lit']} из {item['points']} точек"),
            (f[6], f"сработало {item['fired']} детекторов; красные — σ сильнейших, толщина — E(d)"),
            (f[7], f"в код взято {item['kept']} из {item['fired']} сработавших"),
            (f[8], f"код детали {item['code']} бит, единиц {item['ones']}"),
            (f[9], f"деталей уровня {L} на картинке: {item['details']}; жёлтая — деталь этого окна"),
            (None, f"окно видит ~{100 * item['sees']:.0f} % штрихов → " + ("дальше уровень " + str(L + 1) if L < len(summary) - 1 else "обучение закончено → распознавание")),
        ]
        for i, (path, note) in enumerate(dyn):
            y = row_y(i)
            x = RIGHT_X
            d.rounded_rectangle((x, y, W - 140, y + ROW_H), radius=28, fill=(22, 26, 36), outline=(60, 90, 140), width=2)
            tx = x + 40
            if path:
                im = Image.open(path).convert("RGB")
                k = min((ROW_H - 40) / im.height, 1800 / im.width)
                im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS)
                sheet.paste(im, (x + 20, int(y + (ROW_H - im.height) / 2)))
                tx = x + 40 + im.width + 20
            text(d, tx, y + 40, note, F_DYN, W - 180 - tx, fill=(225, 230, 240))
            # Связь шага алгоритма с данными уровня — пунктир.
            for s in range(LEFT_W + 20, RIGHT_X - 20, 40):
                d.line([(s, centres[i]), (s + 20, centres[i])], fill=(70, 110, 170), width=4)
        out = os.path.join(FOLDER, f"level{L}-cycle.png")
        sheet.save(out, optimize=True)
        print(out, sheet.size)


if __name__ == "__main__":
    main()
