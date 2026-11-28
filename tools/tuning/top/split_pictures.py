# -*- coding: utf-8 -*-
"""Какие картинки цифры отрываются от её главного пятна на большой верхней карте (к tools/tuning/top/extra_pictures.py).

    python tools/tuning/top/split_pictures.py <зерно=6> [новых=3500] [раскладка=0] [цифры=2,5]

Раскладывает верх на кодах data/top-stand/extra-seedN.npz настройками опоры и для каждой цифры из списка рисует лист:
картинки главного пятна (случайная выборка) и картинки, попавшие в другие пятна, — с подписью, рядом с какой цифрой они
лежат. Числа: сколько точек цифры вне главного пятна, среднее сходство кодов «оторвавшихся» со своим главным пятном и
с пятном соседа. Лист — data/top-stand/split-seedN-digitD.png.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "client"), os.path.join(ROOT, "system"), os.path.dirname(os.path.abspath(__file__))]
os.environ.setdefault("DAMP_LOG_CONSOLE", "heartbeat")
os.environ.setdefault("RERUN", "off")
os.environ.setdefault("DAMP_RERUN_VIEWER", "off")
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

import extra_pictures as E  # noqa: E402

EIGHT = np.ones((3, 3), bool)


def main() -> None:
    from PIL import Image, ImageDraw, ImageFont
    from torchvision.datasets import MNIST

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    extra = int(sys.argv[2]) if len(sys.argv) > 2 else 3500
    s = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    wanted = [int(v) for v in (sys.argv[4] if len(sys.argv) > 4 else "2,5").split(",")]
    z = np.load(os.path.join(E.FOLDER, f"extra-seed{seed}.npz"))
    bits, labels, _, new, level = E.load_codes(seed)
    mnist_index = z["mnist"]
    old_rows, new_rows = np.flatnonzero(~new), np.flatnonzero(new)
    chosen = np.concatenate([old_rows, new_rows[:extra]])
    cells, got, order, size, _ = E.lay(bits[chosen], labels[chosen], level, {**E.base_settings(), **E.LIGHT}, s)
    rows = chosen[order]  # строка кода каждой точки раскладки
    grid = np.full(size, -1, dtype=np.int64)
    grid[cells[:, 0], cells[:, 1]] = got
    empty = grid < 0
    codes = bits[rows].astype(np.float32)
    norm = np.sqrt(codes.sum(axis=1))
    images = MNIST(root="./data", train=True, download=True).data.numpy()
    try:
        big, small = ImageFont.truetype("arialbd.ttf", 18), ImageFont.truetype("arial.ttf", 14)
    except OSError:
        big = small = ImageFont.load_default()
    rng = np.random.default_rng(0)
    for digit in wanted:
        mask = grid == digit
        parts, _ = ndimage.label(mask | (empty & ndimage.binary_dilation(mask, structure=EIGHT)), structure=EIGHT)
        of_digit = np.flatnonzero(got == digit)
        part_of = parts[cells[of_digit, 0], cells[of_digit, 1]]
        sizes = np.bincount(part_of)
        main = int(sizes.argmax())
        inside, outside = of_digit[part_of == main], of_digit[part_of != main]
        # Сосед «оторвавшейся» точки — самая частая чужая цифра в квадрате 5 × 5 вокруг неё.
        near = []
        for point in outside:
            y, x = cells[point]
            window = grid[max(0, y - 2): y + 3, max(0, x - 2): x + 3]
            others = window[(window >= 0) & (window != digit)]
            near.append(int(np.bincount(others, minlength=10).argmax()) if others.size else -1)
        near = np.asarray(near, dtype=np.int64)
        own = codes[inside].sum(axis=0)
        to_own = (codes[outside] @ own) / (norm[outside] * np.sqrt((codes[inside] ** 2).sum()) + 1e-9) if outside.size else np.zeros(0)
        text = f"цифра {digit}: точек {of_digit.size}, в главном пятне {inside.size} ({inside.size / of_digit.size:.3f}), вне его {outside.size}"
        if outside.size:
            pairs = np.bincount(near[near >= 0], minlength=10)
            text += "; соседи оторвавшихся: " + ", ".join(f"«{d}» — {int(pairs[d])}" for d in np.argsort(-pairs)[:3] if pairs[d])
            # Сходство кода с ближайшим по коду своим из главного пятна и ближайшим чужим на карте в целом.
            sim = (codes[outside] @ codes.T) / (norm[outside][:, None] * norm[None, :] + 1e-9)
            best_own = sim[:, inside].max(axis=1)
            foreign = np.flatnonzero(got != digit)
            best_foreign = sim[:, foreign].max(axis=1)
            text += (f"; у оторвавшихся ближайший по коду свой из главного пятна — сходство {best_own.mean():.3f}, ближайший чужой — "
                     f"{best_foreign.mean():.3f}; ближе чужой у {int((best_foreign > best_own).sum())} из {outside.size}")
        print(text, flush=True)
        cell, per_row = 34, 30
        shown_in = rng.permutation(inside)[: per_row * 3]
        shown_out = outside[np.argsort(near, kind="stable")][: per_row * 4]
        rows_in, rows_out = -(-shown_in.size // per_row), max(1, -(-shown_out.size // per_row))
        sheet = Image.new("RGB", (20 + per_row * cell, 60 + rows_in * cell + 46 + rows_out * (cell + 14) + 10), "black")
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 8), f"Цифра {digit} на верхней карте из {chosen.size} точек (зерно {seed}): в главном пятне {inside.size} из {of_digit.size}",
                  fill="white", font=big)
        draw.text((10, 36), "картинки главного пятна (случайные):", fill=(200, 200, 200), font=small)
        for n, point in enumerate(shown_in):
            picture = Image.fromarray(images[int(mnist_index[rows[point]])]).resize((cell - 4, cell - 4))
            sheet.paste(picture, (10 + (n % per_row) * cell, 60 + (n // per_row) * cell))
        top = 60 + rows_in * cell + 12
        draw.text((10, top), f"картинки вне главного пятна ({outside.size}); под каждой — цифра-сосед на карте:", fill=(200, 200, 200), font=small)
        for n, point in enumerate(shown_out):
            picture = Image.fromarray(images[int(mnist_index[rows[point]])]).resize((cell - 4, cell - 4))
            x, y = 10 + (n % per_row) * cell, top + 26 + (n // per_row) * (cell + 14)
            sheet.paste(picture, (x, y))
            neighbour = int(near[np.argsort(near, kind="stable")][n])
            draw.text((x + 10, y + cell - 4), "?" if neighbour < 0 else str(neighbour), fill=(255, 200, 80), font=small)
        path = os.path.join(E.FOLDER, f"split-seed{seed}-digit{digit}.png")
        sheet.save(path)
        print("лист:", path, flush=True)


if __name__ == "__main__":
    main()
