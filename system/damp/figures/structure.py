"""Картинка устройства проекта: клиентский и системный слои, пакеты системного слоя по этапам пайплайна (вне статьи).

Рисуется по настоящим папкам: модули пакетов берутся с диска, поэтому картинка не расходится с кодом.
"""
from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw

from damp import __version__
from damp.article_refs import OUTSIDE_ARTICLE
from damp.figures.style import FigureStyle
from damp.logging import LOGGER

# Пакет системного слоя → этап пайплайна и что в нём.
PACKAGES = (
    ("stimuli", "1", "Стимулы: углопозиции MNIST"),
    ("encoding", "2", "Первичное кодирование, код окна"),
    ("layout", "3", "Раскладка"),
    ("activation", "4–5", "Детекторы, активации, эмбеддинги"),
    ("cycle", "6", "Цикл уровней"),
    ("decode", "7", "Распознавание"),
    ("figures", "—", "Рисунки README"),
)


class StructureFigure:
    """Две колонки: клиентский слой (чем управляют) и системный (что под капотом)."""

    def __init__(self, style: FigureStyle, root: str) -> None:
        self.style = style
        self.root = Path(root)

    def _modules(self, package: str) -> list[str]:
        folder = self.root / "system" / "damp" / package
        return sorted(name[:-3] for name in os.listdir(folder) if name.endswith(".py") and name != "__init__.py")

    def _client_items(self) -> list[tuple[str, str, bool]]:
        """Карточки клиентского слоя: имя, пояснение, выделить ли (фреймворк damp_api — вход в систему)."""
        bridge = "_system.py"
        api = sorted(
            name for name in os.listdir(self.root / "client" / "damp_api") if name.endswith(".py") and name not in ("__init__.py", bridge)
        )
        return [
            ("config.toml", "все настройки: важные сверху,\nтехника внизу", False),
            ("run.py", "команды поверх damp_api: ladder, bench,\ndashboard, figures, cache, docs, …", False),
            ("dashboard/", "живой дашборд: рисуем цифру кистью —\nкусочки и кластеры всех уровней", False),
            ("apidocs.py", "HTML-справка по damp_api\nиз docstring'ов (pdoc) → docs/api/", False),
            (
                "damp_api/",
                "фреймворк по этапам пайплайна:\nDamp().stimuli · coding · layout ·\ndetectors · embeddings · cycle ·\n"
                "recognition · figures\n" + " · ".join(api) + f"\n{bridge} — единственный мост в system/",
                True,
            ),
            ("examples/", "картинка для predict", False),
        ]

    def save(self, path: str) -> str:
        style = self.style
        columns = 4
        width, margin, gap, inner = 1560, 24, 36, 12
        client_width = 400
        system_left = margin + client_width + gap
        system_width = width - system_left - margin
        card_width = (system_width - 2 * 16 - (columns - 1) * inner) // columns
        rows = [(package, stage, about, self._modules(package)) for package, stage, about in PACKAGES]
        grid = [rows[start : start + columns] for start in range(0, len(rows), columns)]
        heights = [70 + 19 * max(len(modules) for _, _, _, modules in row) for row in grid]
        root_modules = sorted(name[:-3] for name in os.listdir(self.root / "system" / "damp") if name.endswith(".py") and name != "__init__.py")
        items = self._client_items()
        item_heights = [52 + 21 * (note.count("\n") + 1) for _, note, _ in items]
        aside = ["docs/ — статья, продолжение, сверка, рисунки", "tests/ — инварианты статьи, быстрые пути", "data/, logs/ — создают прогоны (в .gitignore)", "pyproject.toml, CHANGELOG.md — версия"]
        client_height = 80 + sum(h + 14 for h in item_heights) + 8 + 24 + 22 * len(aside) + 16
        box_height = max(76 + sum(heights) + inner * (len(grid) - 1) + 60, client_height)
        top = margin + 52
        height = top + box_height + margin
        image = Image.new("RGB", (width, height), style.background)
        draw = ImageDraw.Draw(image)
        draw.text((margin, margin - 4), f"Устройство проекта DAMP {__version__}", fill=style.ink, font=style.font(26, bold=True))
        # Клиентский слой.
        draw.rounded_rectangle((margin, top, margin + client_width, top + box_height), radius=10, outline=style.green, width=2)
        draw.text((margin + 18, top + 14), "client/ — клиентский слой", fill=style.green, font=style.font(20, bold=True))
        draw.text((margin + 18, top + 42), "чем клиент, оператор и админ управляют", fill=style.second, font=style.font(15))
        y = top + 80
        bridge_y = y
        for (name, note, framework), item_height in zip(items, item_heights):
            box = (margin + 18, y, margin + client_width - 18, y + item_height)
            draw.rounded_rectangle(box, radius=8, fill=style.surface, outline=style.green if framework else None, width=2)
            draw.text((margin + 34, y + 12), name, fill=style.green if framework else style.ink, font=style.font(18, bold=True))
            draw.multiline_text((margin + 34, y + 40), note, fill=style.second, font=style.font(15), spacing=6)
            if framework:
                bridge_y = y + item_height // 2
            y += item_height + 14
        y += 8
        draw.text((margin + 18, y), "рядом", fill=style.muted, font=style.font(15, bold=True))
        for line, text in enumerate(aside):
            draw.text((margin + 18, y + 24 + line * 22), text, fill=style.second, font=style.font(15))
        # Единственный путь в систему: damp_api/_system.py → system/damp.
        start, end = margin + client_width - 18, system_left
        draw.line((start, bridge_y, end - 2, bridge_y), fill=style.green, width=3)
        draw.polygon([(end, bridge_y), (end - 12, bridge_y - 7), (end - 12, bridge_y + 7)], fill=style.green)
        # Системный слой.
        draw.rounded_rectangle((system_left, top, system_left + system_width, top + box_height), radius=10, outline=style.blue, width=2)
        draw.text((system_left + 18, top + 14), "system/damp/ — системный слой", fill=style.blue, font=style.font(20, bold=True))
        draw.text((system_left + 18, top + 42), "под капотом: пакеты по этапам пайплайна; вход — только через client/damp_api", fill=style.second, font=style.font(15))
        upper = top + 76
        for row, row_height in zip(grid, heights):
            for column, (package, stage, about, modules) in enumerate(row):
                left = system_left + 16 + column * (card_width + inner)
                draw.rounded_rectangle((left, upper, left + card_width, upper + row_height), radius=8, fill=style.surface)
                badge = f"этап {stage}" if stage != "—" else "рисунки"
                draw.text((left + 14, upper + 10), f"{package}/", fill=style.ink, font=style.font(18, bold=True))
                draw.text((left + card_width - 14 - draw.textlength(badge, font=style.font(14)), upper + 13), badge, fill=style.orange, font=style.font(14))
                draw.text((left + 14, upper + 36), about, fill=style.second, font=style.font(14))
                for line, module in enumerate(modules):
                    draw.text((left + 22, upper + 62 + line * 19), module + ".py", fill=style.muted, font=style.font(14))
            upper += row_height + inner
        draw.text(
            (system_left + 18, upper + 10),
            "в корне damp/: " + ", ".join(name + ".py" for name in root_modules) + " — settings.py читает client/config.toml",
            fill=style.second,
            font=style.font(15),
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        image.save(path, optimize=True)
        LOGGER.event(
            "figure.save",
            section=OUTSIDE_ARTICLE,
            data={"path": path, "packages": [package for package, _, _, _ in rows], "version": __version__,
                  "note": "устройство проекта: клиентский и системный слои"},
        )
        return path
