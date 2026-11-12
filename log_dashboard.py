"""Дашборд инсайтов из логов: один файл HTML без внешних ссылок.

Дашборд показывает то, что нашёл конвейер log_insights.py: сводку, главные выводы, карту кодового
пространства (§5.14, рис. 18) с семействами событий (§8.2), ленту сигналов по файлам, инциденты, новые
строки и числа, связи событий и сведения о том, что было подано на вход. Сам он ничего не вычисляет
по статье: только раскладывает готовые результаты по странице.

Все размеры и цвета — константы LOG_DASHBOARD_* из main.py.
"""
from __future__ import annotations

import base64
from collections import Counter
import html
import io
import json
import os
import re
from typing import TYPE_CHECKING, Sequence

import numpy as np

from damp.article_refs import INTERPRETABILITY
from damp.logging import LOGGER
from main import (
    LOG_DASHBOARD_COLORS,
    LOG_DASHBOARD_FAMILIES,
    LOG_DASHBOARD_MAP_PIXELS,
    LOG_DASHBOARD_RARE_SHARE,
    LOG_DASHBOARD_ROWS,
    LOG_DASHBOARD_SOURCES,
    LOG_DASHBOARD_TEXT_CHARS,
)

if TYPE_CHECKING:
    from log_insights import EventTypes, LogCodeSpace, LogCorpus


def _count(value: float) -> str:
    """Целое число с пробелами между тысячами."""
    return f"{int(value):,}".replace(",", " ")


def _share(part: float, whole: float) -> str:
    """Доля в процентах; очень малая доля не округляется до нуля."""
    if whole <= 0 or part <= 0:
        return "0%"
    value = 100.0 * part / whole
    if value < 0.01:
        return "<0,01%"
    text = f"{value:.2f}" if value < 1 or 99 < value < 100 else f"{value:.1f}"
    return text.rstrip("0").rstrip(".").replace(".", ",") + "%"


class LogInsightsDashboard:
    """Страница с инсайтами: сводка, карта семейств, лента сигналов, инциденты, новизна, связи, вход.

    Цвет точки карты — группа её семейства: обычные события, проблемные (в большинстве строк семейства
    есть слова ошибок) и редкие (строк семейства меньше доли LOG_DASHBOARD_RARE_SHARE всех строк).
    """

    STATUSES = (
        ("usual", "Обычные события"),
        ("problem", "Проблемные: в строках слова ошибок"),
        ("rare", "Редкие события"),
    )
    _ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

    def __init__(self, corpus: "LogCorpus", space: "LogCodeSpace", types: "EventTypes", insights: dict) -> None:
        self.corpus = corpus
        self.space = space
        self.types = types
        self.insights = insights
        self._lines = len(corpus.lines)
        # Общее начало путей всех файлов на странице не показывается.
        folders = sorted({source.rsplit("/", 1)[0] + "/" if "/" in source else "" for source in corpus.sources})
        prefix = os.path.commonprefix(folders)
        self._prefix = prefix[: prefix.rfind("/") + 1]
        self._pile = insights.get("input", {}).get("mode") == "pile"
        self._point_family = [
            int(types.family_of[space.point_owner(point)]) for point in range(len(space.points))
        ]
        self._family_points = Counter(self._point_family)
        self._family_status = [self._status(family) for family in types.families]

    # ------------------------------------------------------------------ данные
    def _status(self, family) -> str:
        problem = int(sum(int(self.corpus.count[m]) for m in family.members if self.corpus.problem[m]))
        if 2 * problem >= family.lines:
            return "problem"
        return "rare" if family.lines <= LOG_DASHBOARD_RARE_SHARE * self._lines else "usual"

    def _readable(self, text: str, limit: int = LOG_DASHBOARD_TEXT_CHARS) -> str:
        """Строка лога для показа: без цветовых кодов; у записи JSON — её самое словесное поле."""
        text = self._ANSI.sub("", text).strip()
        if text.startswith("{"):
            try:
                record = json.loads(text)
            except ValueError:
                record = None
            if isinstance(record, dict):
                strings = [value for value in record.values() if isinstance(value, str) and value.strip()]
                if strings:
                    text = max(strings, key=lambda value: (len(value.split()), len(value)))
        text = " ".join(text.split())
        return text if len(text) <= limit else text[: limit - 1] + "…"

    def _family_text(self, family) -> str:
        top = max(family.members, key=lambda member: int(self.corpus.count[member]))
        return self._readable(self.corpus.example[top].text)

    def data(self) -> dict:
        """Данные карты для страницы: клетки точек с семейством и энергией, сведения о семействах."""
        layout = self.space.layout
        cells = layout.cells_array()
        energies = self.space.energies
        return {
            "map": {
                "width": int(layout.width),
                "height": int(layout.height),
                "cell": max(3, LOG_DASHBOARD_MAP_PIXELS // max(int(layout.width), int(layout.height))),
                "points": [
                    [int(x), int(y), self._point_family[point], int(round(100 * float(energies[point])))]
                    for point, (y, x) in enumerate(cells)
                ],
            },
            "families": [
                {
                    "id": family.index,
                    "status": self._family_status[family.index],
                    "text": self._family_text(family),
                    "lines": int(family.lines),
                    "kinds": len(family.members),
                    "points": int(self._family_points[family.index]),
                }
                for family in self.types.families
            ],
            "statuses": {key: label for key, label in self.STATUSES},
        }

    def _energy_image(self) -> str:
        """Карта «цвет — код, яркость — энергия» (рис. 18) как встроенная картинка."""
        from PIL import Image

        layout = self.space.layout
        image = Image.fromarray(np.asarray(layout.render_energy_image(self.space.energies.tolist(), log=False), dtype=np.uint8))
        scale = max(1, LOG_DASHBOARD_MAP_PIXELS // max(image.size))
        image = image.resize((image.size[0] * scale, image.size[1] * scale), Image.Resampling.NEAREST)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    # ------------------------------------------------------------------ части страницы
    @staticmethod
    def _table(header: Sequence[tuple[str, str]], rows: Sequence[Sequence[str]], *, row_attrs: Sequence[str] = ()) -> str:
        """Таблица: заголовок — пары (название, класс ячейки); ячейки строк — готовый HTML."""
        head = "".join(f'<th class="{cls}">{html.escape(name)}</th>' for name, cls in header)
        body = []
        for index, row in enumerate(rows):
            attrs = row_attrs[index] if index < len(row_attrs) else ""
            cells = "".join(f'<td class="{cls}">{cell}</td>' for cell, (_, cls) in zip(row, header))
            body.append(f"<tr {attrs}>{cells}</tr>")
        return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'

    def _text_cell(self, text: str) -> str:
        return html.escape(self._readable(text))

    def _path(self, text: str) -> str:
        """Путь или место в файле без общего начала путей."""
        return text[len(self._prefix):] if self._prefix and text.startswith(self._prefix) else text

    def _where(self) -> str:
        return "в конце файлов" if self._pile else "в целевых логах"

    def _tiles(self) -> str:
        summary = self.insights["summary"]
        problem_lines = int(self.corpus.count[self.corpus.problem].sum())
        new_lines = self.insights["new_lines"]
        tiles = [
            (_count(summary["lines"]), "строк со словами"),
            (_count(summary["stimuli"]), "видов строк"),
            (f'{_count(summary["families"])} / {_count(summary["types"])}', "семейств / типов событий"),
            (_count(len(new_lines)), f"новых видов строк {self._where()}"),
            (_count(len(self.insights["incidents"])), "инцидентов"),
            (_share(problem_lines, self._lines), "строк со словами ошибок"),
        ]
        return '<div class="tiles">' + "".join(
            f'<div class="tile"><b>{html.escape(value)}</b><span>{html.escape(label)}</span></div>' for value, label in tiles
        ) + "</div>"

    def _highlights(self) -> str:
        """Главные выводы простыми предложениями."""
        corpus, insights = self.corpus, self.insights
        items: list[str] = []
        counts = np.sort(corpus.count)[::-1]
        need = int(np.searchsorted(np.cumsum(counts), 0.9 * counts.sum()) + 1)
        items.append(
            f"90% строк покрывают самые частые виды строк: <b>{_count(need)}</b> из {_count(len(counts))}."
        )
        largest = max(self.types.families, key=lambda family: family.lines)
        items.append(
            f"Самое крупное семейство событий — {_share(largest.lines, self._lines)} строк: "
            f"<code>{html.escape(self._family_text(largest)[:120])}</code>"
        )
        problem = np.flatnonzero(corpus.problem)
        if problem.size:
            worst = int(problem[np.argmax(corpus.count[problem])])
            items.append(
                f"Строк со словами ошибок: <b>{_count(corpus.count[problem].sum())}</b> "
                f"({_share(corpus.count[problem].sum(), self._lines)}), {_count(problem.size)} видов. Чаще всего "
                f"({_count(corpus.count[worst])} раз): <code>{html.escape(self._readable(corpus.example[worst].text, 140))}</code>"
            )
        else:
            items.append("Строк со словами ошибок нет.")
        new_lines = insights["new_lines"]
        if new_lines:
            bad = sum(1 for item in new_lines if item["problem"])
            items.append(
                f"Новых видов строк {self._where()}: <b>{_count(len(new_lines))}</b> "
                f"({_count(sum(item['count'] for item in new_lines))} строк), из них со словами ошибок: {_count(bad)}."
            )
        else:
            items.append(f"Новых видов строк {self._where()} нет: всё уже встречалось раньше.")
        incidents = insights["incidents"]
        if incidents:
            top = max(incidents, key=lambda item: item["signal_lines"])
            items.append(
                f"Инцидентов: <b>{_count(len(incidents))}</b>. Самый насыщенный — <code>{html.escape(self._path(top['span']))}</code>: "
                f"{_count(top['signal_lines'])} строк-сигналов из {_count(top['lines'])}."
            )
        else:
            items.append("Инцидентов не найдено.")
        if insights["new_numbers"]:
            items.append(f"Чисел вне привычного диапазона: <b>{_count(len(insights['new_numbers']))}</b>.")
        similarity = insights["run_similarity"]
        matrix = np.asarray(similarity["matrix"], dtype=np.float64)
        if matrix.shape[0] > 2:
            mean = (matrix.sum(axis=1) - np.diag(matrix)) / (matrix.shape[0] - 1)
            odd = int(np.argmin(mean))
            if mean[odd] < 0.6 and mean[odd] < np.median(mean) - 0.2:
                items.append(
                    f"Файл <code>{html.escape(self._path(similarity['runs'][odd]))}</code> не похож на остальные: "
                    f"среднее сходство {mean[odd]:.2f} при обычном {np.median(mean):.2f}."
                )
        budget = insights.get("input", {}).get("space", {})
        if budget and budget["stimuli_inside"] < budget["stimuli"]:
            items.append(
                f"На карту вошли {_count(budget['stimuli_inside'])} самых частых видов строк из {_count(budget['stimuli'])} "
                f"({_share(budget['lines_inside_share'], 1.0)} строк): больше точек раскладка на процессоре не тянет. "
                "Остальные виды сверяются с картой детекторами."
            )
        files = insights.get("input", {}).get("files", [])
        without_words = sum(item["without_words"] for item in files)
        skipped = [item for item in files if item["lines"] == 0]
        if without_words or skipped:
            items.append(
                f"Мусор на входе: {_count(without_words)} строк без единого слова пропущено"
                + (f". Файлов без единой строки: {_count(len(skipped))}." if skipped else ".")
            )
        return '<ul class="main">' + "".join(f"<li>{item}</li>" for item in items) + "</ul>"

    def _map_section(self, data: dict) -> str:
        families = sorted(data["families"], key=lambda family: -family["lines"])
        shown = families[:LOG_DASHBOARD_FAMILIES]
        labels = dict(self.STATUSES)
        rows = [
            [
                f'<span class="sw" style="background:var(--{family["status"]})"></span>{family["id"]}',
                _count(family["lines"]),
                _share(family["lines"], self._lines),
                _count(family["kinds"]),
                _count(family["points"]),
                html.escape(labels[family["status"]].split(":")[0].split()[0].lower()),
                html.escape(family["text"]),
            ]
            for family in shown
        ]
        table = self._table(
            [("№", "n"), ("строк", "n"), ("доля", "n"), ("видов", "n"), ("точек", "n"), ("группа", ""),
             ("самая частая строка семейства", "t")],
            rows,
            row_attrs=[f'class="fam" data-family="{family["id"]}"' for family in shown],
        )
        rest = families[LOG_DASHBOARD_FAMILIES:]
        note = (
            f'<p class="note">Показаны {len(shown)} крупнейших семейств; ещё {_count(len(rest))} семейств — '
            f'{_count(sum(family["lines"] for family in rest))} строк.</p>' if rest else ""
        )
        legend = "".join(
            f'<span><span class="sw" style="background:var(--{key})"></span>{html.escape(label)} — точек: '
            f'{_count(sum(f["points"] for f in families if f["status"] == key))}, строк: '
            f'{_count(sum(f["lines"] for f in families if f["status"] == key))}</span>'
            for key, label in self.STATUSES
        )
        summary = self.insights["summary"]
        return (
            "<h2>Карта событий</h2>"
            f'<p class="note">{_count(summary["points"])} точек (коды строк и их фрагментов) на поле '
            f'{summary["grid"][0]}×{summary["grid"][1]}. Похожие строки лежат рядом; площадь достаётся разнообразию, '
            "а не частоте. Наведите на клетку — увидите строку; нажмите на клетку или на строку таблицы — "
            "подсветится семейство.</p>"
            '<div class="maps">'
            '<figure><figcaption>Цвет — группа событий</figcaption><canvas id="map"></canvas></figure>'
            '<figure><figcaption>Цвет — код точки, яркость — насколько похожи соседи</figcaption>'
            f'<img class="energy" alt="Карта энергии точек" src="{self._energy_image()}"></figure>'
            "</div>"
            f'<div class="legend">{legend}</div>'
            "<h2>Семейства событий</h2>" + table + note
        )

    def _timeline_section(self) -> str:
        timeline = self.insights.get("timeline", [])
        if not timeline:
            return ""
        ranked = sorted(timeline, key=lambda row: -sum(cell["signals"] for cell in row["cells"]))
        shown = set(id(row) for row in ranked[:LOG_DASHBOARD_SOURCES])
        strips = []
        for row in timeline:
            if id(row) not in shown:
                continue
            cells = []
            for index, cell in enumerate(row["cells"]):
                share = cell["signals"] / cell["lines"] if cell["lines"] else 0.0
                colour = "var(--empty)" if not cell["signals"] else f"var(--ramp-{min(6, int(share * 7))})"
                tip = (
                    f'{self._path(row["source"])}\nсо строки {cell["line"]}, строк: {cell["lines"]}\n'
                    f'сигналов: {cell["signals"]}, со словами ошибок: {cell["problems"]}'
                )
                cells.append(
                    f'<rect x="{index}" y="0" width="0.92" height="1" style="fill:{colour}" data-tip="{html.escape(tip, quote=True)}"></rect>'
                )
            signals = sum(cell["signals"] for cell in row["cells"])
            strips.append(
                f'<div class="strip"><span class="name" title="{html.escape(row["source"], quote=True)}">'
                f'<bdi>{html.escape(self._path(row["source"]))}</bdi></span>'
                f'<svg viewBox="0 0 {len(row["cells"])} 1" preserveAspectRatio="none">{"".join(cells)}</svg>'
                f'<span class="num">{_count(signals)} / {_count(row["lines"])}</span></div>'
            )
        note = (
            f'<p class="note">Показаны {len(strips)} файлов с наибольшим числом сигналов из {len(timeline)}.</p>'
            if len(timeline) > len(strips) else ""
        )
        meaning = (
            "новая строка, строка редкого типа либо редкая строка со словами ошибок" if self._pile
            else "новая строка либо редкая строка со словами ошибок"
        )
        return (
            "<h2>Лента сигналов</h2>"
            f'<p class="note">Каждая полоса — файл слева направо по порядку строк. Чем темнее отрезок, тем больше '
            f"в нём доля строк-сигналов; серый — сигналов нет. Сигнал — {meaning}. Справа — сигналов / строк.</p>"
            + "".join(strips) + note
        )

    def _incidents_section(self) -> str:
        incidents = sorted(self.insights["incidents"], key=lambda item: -item["signal_lines"])
        if not incidents:
            return "<h2>Инциденты</h2><p class=\"note\">Скоплений строк-сигналов не найдено.</p>"
        rows = []
        for item in incidents[:LOG_DASHBOARD_ROWS]:
            signals = "<br>".join(
                f'{_count(signal["count"])}× {self._text_cell(signal["text"])}' for signal in item["signals"]
            )
            before = "<br>".join(self._text_cell(text) for text in item["before"]) or "—"
            rows.append([
                html.escape(self._path(item["span"])), _count(item["lines"]), _count(item["signal_lines"]),
                _count(item["signal_kinds"]), html.escape(self._path(item["first_signal"])), signals, before,
            ])
        note = (
            f'<p class="note">Показаны {len(rows)} самых насыщенных из {_count(len(incidents))}.</p>'
            if len(incidents) > len(rows) else ""
        )
        return (
            "<h2>Инциденты</h2>"
            '<p class="note">Инцидент — идущие подряд окна строк, в которых есть сигналы.</p>'
            + self._table(
                [("где", "t"), ("строк", "n"), ("сигналов", "n"), ("видов", "n"), ("первый сигнал", "t"),
                 ("строки-сигналы", "t"), ("что было перед", "t")],
                rows,
            ) + note
        )

    def _novelty_section(self) -> str:
        new_lines = self.insights["new_lines"]
        if not new_lines:
            return ""
        rows = [
            [
                html.escape(item["status"]) + (" · ошибка" if item["problem"] else ""),
                _count(item["count"]), html.escape(self._path(item["where"])), self._text_cell(item["text"]),
                self._text_cell(item["nearest_known"]) if item["nearest_known"] else "—",
            ]
            for item in new_lines[:LOG_DASHBOARD_ROWS]
        ]
        note = (
            f'<p class="note">Показаны {len(rows)} из {_count(len(new_lines))}: сначала со словами ошибок, '
            "потом наименее знакомые.</p>" if len(new_lines) > len(rows) else ""
        )
        return (
            f"<h2>Новые строки {self._where()}</h2>"
            '<p class="note">«Неизвестна» — на строку не откликнулся ни один детектор; «по частям» — строка '
            "собрана из знакомых кусков.</p>"
            + self._table(
                [("статус", ""), ("раз", "n"), ("где", "t"), ("строка", "t"), ("ближайшая известная", "t")], rows
            ) + note
        )

    def _numbers_section(self) -> str:
        numbers = self.insights["new_numbers"]
        if not numbers:
            return ""
        rows = [
            [
                html.escape(str(item["slot"])), f'{item["value"]:g}',
                f'{item["reference_min"]:g} … {item["reference_max"]:g}', _count(item["reference_values"]),
                html.escape(self._path(item["where"])), html.escape(str(item["type"])),
            ]
            for item in numbers[:LOG_DASHBOARD_ROWS]
        ]
        return (
            "<h2>Числа вне привычного диапазона</h2>"
            + self._table(
                [("что за число", ""), ("значение", "n"), ("раньше было", "n"), ("значений раньше", "n"),
                 ("где", "t"), ("тип события", "t")],
                rows,
            )
        )

    def _rules_section(self) -> str:
        rules = self.insights["rules"]
        if not rules:
            return ""
        rows = [
            [html.escape(item["after"]), html.escape(item["then"]), _count(item["support"]),
             f'{item["confidence"]:.2f}', f'{item["lift"]:.1f}']
            for item in rules[:LOG_DASHBOARD_ROWS]
        ]
        return (
            "<h2>Что за чем следует</h2>"
            '<p class="note">Уверенность — доля случаев, когда за первым событием шло второе; подъём — во сколько '
            "раз это чаще случайного.</p>"
            + self._table(
                [("после события", "t"), ("идёт событие", "t"), ("раз", "n"), ("уверенность", "n"), ("подъём", "n")],
                rows,
            )
        )

    def _input_section(self) -> str:
        source = self.insights.get("input", {})
        files = source.get("files", [])
        if not files:
            return ""
        split = source.get("pile_split", {})
        ranked = sorted(files, key=lambda item: -item["lines"])
        rows = [
            [
                html.escape(self._path(item["source"])), _count(item["lines"]), _count(item["without_words"]),
                _count(item["cut_lines"]),
                html.escape(
                    "; ".join(part for part in (
                        item["note"],
                        f'конец проверяется со строки {split[item["source"]]}' if item["source"] in split else "",
                    ) if part) or "—"
                ),
            ]
            for item in ranked[:LOG_DASHBOARD_SOURCES]
        ]
        note = (
            f'<p class="note">Показаны {len(rows)} самых больших файлов из {_count(len(files))}.</p>'
            if len(files) > len(rows) else ""
        )
        how = (
            f"Файлы не делились на эталон и цель: последние {_share(source.get('pile_target_share', 0), 1.0)} строк "
            "каждого файла проверяются по всему остальному."
            if self._pile else "Целевые логи проверяются по эталонным."
        )
        return (
            "<h2>Что было на входе</h2>"
            f'<p class="note">{how} Строка длиннее {_count(source.get("line_chars_max", 0))} знаков обрезается; '
            f'после {_count(source.get("lines_max", 0))} строк чтение останавливается.</p>'
            + self._table(
                [("файл", "t"), ("строк взято", "n"), ("без слов", "n"), ("обрезано", "n"), ("пометка", "")], rows
            ) + note
        )

    # ------------------------------------------------------------------ страница
    @staticmethod
    def _css_vars(theme: str) -> str:
        colours = LOG_DASHBOARD_COLORS[theme]
        parts = [f"--{name}:{value}" for name, value in colours.items() if isinstance(value, str)]
        parts += [f"--ramp-{index}:{value}" for index, value in enumerate(colours["ramp"])]
        return ";".join(parts)

    def html(self) -> str:
        data = self.data()
        summary = self.insights["summary"]
        files = self.insights.get("input", {}).get("files", [])
        body = (
            "<h1>Инсайты из логов</h1>"
            f'<p class="sub">Файлов: {_count(len(files) or len(self.corpus.sources))}. '
            f'{_count(summary["lines"])} строк → {_count(summary["stimuli"])} видов строк → '
            f'{_count(summary["points"])} точек на поле {summary["grid"][0]}×{summary["grid"][1]}.</p>'
            + self._tiles()
            + "<h2>Главное</h2>" + self._highlights()
            + self._map_section(data)
            + self._timeline_section()
            + self._incidents_section()
            + self._novelty_section()
            + self._numbers_section()
            + self._rules_section()
            + self._input_section()
        )
        payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
        return (
            _PAGE.replace("%%LIGHT%%", self._css_vars("light"))
            .replace("%%DARK%%", self._css_vars("dark"))
            .replace("%%BODY%%", body)
            .replace("%%DATA%%", payload)
        )

    def save(self, path: str) -> None:
        page = self.html()
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(page)
        LOGGER.event(
            "logs.dashboard.save",
            section=INTERPRETABILITY,
            data={"path": path, "bytes": len(page.encode("utf-8")), "families": len(self.types.families)},
        )


_PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Инсайты из логов</title>
<style>
:root{color-scheme:light dark;%%LIGHT%%}
@media (prefers-color-scheme: dark){:root{%%DARK%%}}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--text);font:14px/1.45 "Segoe UI",system-ui,sans-serif}
main{max-width:1500px;margin:0 auto;padding:24px 16px 60px}
h1{font-size:26px;margin:0 0 4px}
h2{font-size:18px;margin:36px 0 10px}
code{font-family:Consolas,monospace;font-size:12px;color:var(--text2);word-break:break-word}
.sub,.note{color:var(--text2)}
.note{font-size:13px;margin:6px 0 10px;max-width:1100px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:18px 0}
.tile{background:var(--panel);border-radius:8px;padding:12px 14px}
.tile b{display:block;font-size:26px;line-height:1.2;font-variant-numeric:tabular-nums}
.tile span{color:var(--text2);font-size:13px}
ul.main{margin:0;padding-left:20px;max-width:1100px}
ul.main li{margin:5px 0}
.maps{display:flex;flex-wrap:wrap;gap:24px;align-items:flex-start}
.maps figure{margin:0;max-width:100%}
figcaption{font-weight:600;margin-bottom:8px}
canvas,img.energy{display:block;max-width:100%;height:auto;border-radius:4px}
canvas{cursor:pointer}
.legend{display:flex;flex-wrap:wrap;gap:6px 22px;margin:12px 0}
.sw{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}
th{text-align:left;color:var(--text2);font-weight:400;border-bottom:1px solid var(--line);padding:6px 8px;white-space:nowrap}
td{border-bottom:1px solid var(--line);padding:6px 8px;vertical-align:top}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
td.t{font-family:Consolas,monospace;font-size:12px;color:var(--text2);word-break:break-word;min-width:160px}
tr.fam{cursor:pointer}
tr.fam:hover,tr.fam.on{background:var(--panel)}
.strip{display:flex;align-items:center;gap:10px;margin:3px 0}
.strip .name{flex:0 0 300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12px;color:var(--text2);direction:rtl;text-align:left}
.strip svg{flex:1 1 auto;height:16px;min-width:200px}
.strip .num{flex:0 0 130px;font-size:12px;color:var(--text2);text-align:right;font-variant-numeric:tabular-nums}
#tip{position:fixed;pointer-events:none;background:var(--text);color:var(--surface);padding:6px 8px;border-radius:4px;font-size:12px;max-width:520px;display:none;z-index:9;white-space:pre-wrap;word-break:break-word}
@media (max-width:700px){.strip .name{flex-basis:120px}.strip .num{display:none}}
</style></head><body><main>
%%BODY%%
</main><div id="tip"></div>
<script id="data" type="application/json">%%DATA%%</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const css = name => getComputedStyle(document.documentElement).getPropertyValue('--' + name).trim();
const tip = document.getElementById('tip');
function showTip(text, event) {
  tip.textContent = text;
  tip.style.display = 'block';
  tip.style.left = Math.max(4, Math.min(event.clientX + 14, innerWidth - tip.offsetWidth - 8)) + 'px';
  tip.style.top = Math.max(4, Math.min(event.clientY + 14, innerHeight - tip.offsetHeight - 8)) + 'px';
}
const canvas = document.getElementById('map');
const ctx = canvas.getContext('2d');
const W = D.map.width, H = D.map.height, cell = D.map.cell;
const grid = new Int32Array(W * H).fill(-1);
D.map.points.forEach((point, index) => { grid[point[1] * W + point[0]] = index; });
const families = new Map(D.families.map(family => [family.id, family]));
let picked = -1;
function draw() {
  const ratio = window.devicePixelRatio || 1;
  canvas.width = W * cell * ratio;
  canvas.height = H * cell * ratio;
  canvas.style.width = W * cell + 'px';
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.fillStyle = css('surface');
  ctx.fillRect(0, 0, W * cell, H * cell);
  const gap = cell >= 6 ? 1 : 0;
  const colours = {usual: css('usual'), problem: css('problem'), rare: css('rare'), empty: css('empty')};
  const ink = css('text');
  ctx.font = '600 ' + Math.max(9, Math.floor(cell * 0.42)) + 'px "Segoe UI",sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const index = grid[y * W + x];
    const family = index < 0 ? null : families.get(D.map.points[index][2]);
    ctx.globalAlpha = family && picked >= 0 && family.id !== picked ? 0.15 : 1;
    ctx.fillStyle = family ? colours[family.status] : colours.empty;
    ctx.fillRect(x * cell + gap, y * cell + gap, cell - 2 * gap, cell - 2 * gap);
    if (family && cell >= 18) { ctx.fillStyle = ink; ctx.fillText(family.id, x * cell + cell / 2, y * cell + cell / 2 + 1); }
  }
  ctx.globalAlpha = 1;
  document.querySelectorAll('tr.fam').forEach(row => row.classList.toggle('on', Number(row.dataset.family) === picked));
}
function pointAt(event) {
  const box = canvas.getBoundingClientRect();
  const x = Math.floor((event.clientX - box.left) / box.width * W), y = Math.floor((event.clientY - box.top) / box.height * H);
  return x < 0 || y < 0 || x >= W || y >= H ? -1 : grid[y * W + x];
}
canvas.addEventListener('mousemove', event => {
  const index = pointAt(event);
  if (index < 0) { tip.style.display = 'none'; return; }
  const point = D.map.points[index], family = families.get(point[2]);
  showTip('Семейство № ' + family.id + ' · ' + D.statuses[family.status] + '\\nстрок: ' + family.lines + ', видов строк: ' + family.kinds +
    ', точек: ' + family.points + '\\nсходство с соседями: ' + (point[3] / 100).toFixed(2) + '\\n' + family.text, event);
});
canvas.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
canvas.addEventListener('click', event => {
  const index = pointAt(event);
  const family = index < 0 ? -1 : D.map.points[index][2];
  picked = family === picked ? -1 : family;
  draw();
});
document.querySelectorAll('tr.fam').forEach(row => row.addEventListener('click', () => {
  const family = Number(row.dataset.family);
  picked = family === picked ? -1 : family;
  draw();
}));
document.addEventListener('mousemove', event => {
  if (event.target === canvas) return;
  const element = event.target.closest ? event.target.closest('[data-tip]') : null;
  if (element) showTip(element.dataset.tip, event); else tip.style.display = 'none';
});
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
draw();
</script></body></html>
"""
