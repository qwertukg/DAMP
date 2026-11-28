"""Сборщик HTML-документации клиентского фреймворка `damp_api` из docstring'ов (pdoc).

Берёт только клиентский слой: пакет `damp_api` и его модули `damp`, `stages`, `parts`, `reports`.
Системный слой `system/damp` и мост `_system.py` внутренние и в справку не попадают.

    python client/apidocs.py            # собрать HTML в docs/api/ (открыть docs/api/index.html)
    python client/apidocs.py --serve    # живой сервер с перечитыванием при правке кода
    python client/apidocs.py --open     # собрать и открыть в браузере

Нужен pdoc: `python -m pip install pdoc` или `pip install -e .[docs]`.
"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

CLIENT = Path(__file__).resolve().parent
ROOT = CLIENT.parent


class ApiDocs:
    """Сборщик справки по API: HTML из docstring'ов формата Google (`Args`, `Returns`, `Raises`, `Example`).

    Args:
        output: папка для HTML; по умолчанию `docs/api/` (в `.gitignore`).
        package: имя документируемого пакета.
    """

    PACKAGE = "damp_api"
    MODULES = ("damp", "stages", "parts", "reports", "cache")
    OUTPUT = ROOT / "docs" / "api"
    LOGO_TITLE = "DAMP: клиентский фреймворк"

    def __init__(self, output: str | Path | None = None, package: str = PACKAGE) -> None:
        self.output = Path(output) if output else self.OUTPUT
        self.package = package

    @property
    def modules(self) -> list[str]:
        """Что отдать pdoc: пакет и его модули (без `_system`)."""
        return [self.package, *(f"{self.package}.{name}" for name in self.MODULES)]

    @property
    def index(self) -> Path:
        """Главная страница справки."""
        return self.output / f"{self.package}.html"

    def build(self) -> Path:
        """Собрать HTML в папку `output`.

        Returns:
            Путь к главной странице.

        Raises:
            SystemExit: pdoc не установлен.
        """
        pdoc = self._pdoc()
        self._prepare(pdoc)
        pdoc.pdoc(*self.modules, output_directory=self.output)
        (self.output / "index.html").write_text(
            f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0; url={self.package}.html">'
            f'<a href="{self.package}.html">{self.LOGO_TITLE}</a>',
            encoding="utf-8",
        )
        return self.index

    def serve(self, port: int = 8080) -> None:
        """Живой сервер: страницы собираются при запросе, правка кода видна после перезагрузки страницы.

        Args:
            port: порт на localhost.
        """
        pdoc = self._pdoc()
        self._prepare(pdoc)
        from pdoc import web

        server = web.DocServer(("localhost", port), self.modules)
        print(f"Справка: http://localhost:{port}  (Ctrl+C — стоп)")
        with server:
            server.serve_forever()

    def _prepare(self, pdoc) -> None:  # noqa: ANN001
        """Путь к пакету и настройки pdoc: Google-стиль docstring'ов, примеры кода с подсветкой."""
        if str(CLIENT) not in sys.path:
            sys.path.insert(0, str(CLIENT))
        pdoc.render.configure(docformat="google", math=False, show_source=True, search=True, footer_text="DAMP, клиентский фреймворк damp_api")

    @staticmethod
    def _pdoc():  # noqa: ANN205
        """Импорт pdoc — здесь, а не наверху модуля: без pdoc остальной код работает."""
        try:
            import pdoc
            import pdoc.render
        except ImportError:
            raise SystemExit("Нужен pdoc: python -m pip install pdoc  (или pip install -e .[docs])") from None
        return pdoc


class ReadmeApiLinks:
    """Ссылки справочника README «Функции API по этапам» на строки кода: имя функции → `client/damp_api/<файл>.py#Lnn`.

    Код сдвинулся — ссылки ставятся заново по нынешним номерам строк (старые снимаются). Тест
    `tests.test_client_api.ReadmeApiLinksTest` падает, если ссылка ушла мимо своей функции.
    """

    SECTION = "### Функции API по этапам"
    STAGE_CLASSES = {
        "stimuli": "Stimuli", "coding": "Coding", "layout": "Layout", "detectors": "Detectors",
        "embeddings": "Embeddings", "cycle": "Cycle", "recognition": "Recognition", "figures": "Figures",
    }
    PARTS = ("Digits", "Elements", "Keyholes", "KeyholeCodes", "Memory", "MapDetectors", "Firing", "Level", "TrainedLadder")

    def __init__(self, readme: Path = ROOT / "README.md", api: Path = CLIENT / "damp_api") -> None:
        self.readme, self.api = Path(readme), Path(api)
        self.where = {name: self._lines(self.api / name) for name in ("stages.py", "parts.py", "damp.py")}

    @staticmethod
    def _lines(path: Path) -> dict[str, int]:
        """«Класс.метод» и «Класс» → номер строки определения."""
        import ast

        found: dict[str, int] = {}
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.ClassDef):
                found[node.name] = node.lineno
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        found[f"{node.name}.{item.name}"] = item.lineno
        return found

    def _link(self, text: str, file: str, key: str) -> str:
        return f"[`{text}`](client/damp_api/{file}#L{self.where[file][key]})"

    def refresh(self) -> tuple[int, list[str]]:
        """Поставить ссылки заново; вернуть (число ссылок, функции, которых нет в коде)."""
        import re

        raw = self.readme.read_bytes().decode("utf-8")
        crlf = "\r\n" in raw
        text = raw.replace("\r\n", "\n")
        start = text.index(self.SECTION)
        end = text.index("[↑ к содержанию]", start)
        section = re.sub(r"\[(`[^`]+`)\]\(client/damp_api/\w+\.py#L\d+\)", r"\1", text[start:end])
        current, missed, out = None, [], []
        for line in section.split("\n"):
            heading = re.match(r"^\*\*(?:Этап \d\. [^*]+|Рисунки)\*\* — `damp\.(\w+)`", line)
            if heading:
                current = self.STAGE_CLASSES[heading.group(1)]
            if line.startswith("**Сам `Damp`**"):
                line = re.sub(r"`([^`]+)`", self._damp_span, line)
            elif line.startswith(("| 🧩", "| ▶️", "**Рисунки**")):
                line = self._stage_line(line, current, missed)
            elif re.match(rf"^\| `({'|'.join(self.PARTS)})` \|", line):
                line = self._part_line(line)
            out.append(line)
        text = text[:start] + "\n".join(out) + text[end:]
        self.readme.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
        return "\n".join(out).count("](client/damp_api/"), missed

    def _damp_span(self, match) -> str:  # noqa: ANN001
        span = match.group(1)
        keys = (("Damp(", "Damp.__init__"), ("with_settings(", "Damp.with_settings"), ("from_snapshot(", "Damp.from_snapshot"),
                ("settings[", "Settings.__getitem__"))
        for prefix, key in keys:
            if span.startswith(prefix):
                return self._link(span, "damp.py", key)
        if span in ("stages", "version"):
            return self._link(span, "damp.py", f"Damp.{span}")
        return match.group(0)

    def _stage_line(self, line: str, current: str | None, missed: list[str]) -> str:
        import re

        def span(match) -> str:  # noqa: ANN001
            name = re.match(r"(detectors\.)?(\w+)\(", match.group(1))
            if not name:
                return match.group(0)
            if name.group(1):
                return self._link(match.group(1), "parts.py", f"MapDetectors.{name.group(2)}")
            key = f"{current}.{name.group(2)}"
            if key not in self.where["stages.py"]:
                missed.append(key)
                return match.group(0)
            return self._link(match.group(1), "stages.py", key)

        if line.startswith("|"):
            cells = line.split(" | ")
            cells[1] = re.sub(r"`([^`]+)`", span, cells[1])
            return " | ".join(cells)
        return re.sub(r"`((?:readme|structure)\(\))`", span, line)

    def _part_line(self, line: str) -> str:
        import re

        cells = line.split(" | ")
        part = re.match(r"^\| `(\w+)`", line).group(1)
        cells[0] = "| " + self._link(part, "parts.py", part)

        def span(match) -> str:  # noqa: ANN001
            name = re.match(r"(\w+)\(", match.group(1))
            if name and f"{part}.{name.group(1)}" in self.where["parts.py"]:
                return self._link(match.group(1), "parts.py", f"{part}.{name.group(1)}")
            return match.group(0)

        cells[2] = re.sub(r"`([^`]+)`", span, cells[2])
        return " | ".join(cells)


def main(argv: list[str] | None = None) -> int:
    """Командная строка: сборка, живой сервер, открытие в браузере, ссылки README на код."""
    parser = argparse.ArgumentParser(description="HTML-справка по фреймворку damp_api из docstring'ов")
    parser.add_argument("--output", help="папка HTML (по умолчанию docs/api)")
    parser.add_argument("--serve", action="store_true", help="живой сервер вместо сборки")
    parser.add_argument("--port", type=int, default=8080, help="порт сервера (по умолчанию 8080)")
    parser.add_argument("--open", action="store_true", help="открыть собранную справку в браузере")
    parser.add_argument("--links", action="store_true", help="обновить ссылки README «Функции API по этапам» на строки кода")
    args = parser.parse_args(argv)

    if args.links:
        count, missed = ReadmeApiLinks().refresh()
        print(f"Ссылок на код в README: {count}" + (f"; нет в коде: {', '.join(missed)}" if missed else ""))
        return 1 if missed else 0
    docs = ApiDocs(args.output)
    if args.serve:
        docs.serve(args.port)
        return 0
    index = docs.build()
    print(f"Справка собрана: {index}")
    if args.open:
        webbrowser.open(index.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
