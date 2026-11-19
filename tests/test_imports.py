"""Каждый модуль системного слоя и команды клиента импортируются без ошибок."""
import importlib
import os
import pkgutil
import unittest
from pathlib import Path

os.environ.setdefault("RERUN", "off")

import damp  # noqa: E402


class ImportsTest(unittest.TestCase):
    def test_every_system_module_imports(self) -> None:
        names = [info.name for info in pkgutil.walk_packages(damp.__path__, prefix="damp.")]
        self.assertGreater(len(names), 30)
        for name in names:
            with self.subTest(module=name):
                importlib.import_module(name)

    def test_every_import_statement_resolves(self) -> None:
        """Импорты на любой глубине (и внутри функций) указывают на существующие модули."""
        import ast

        root = Path(__file__).resolve().parents[1]
        files = list((root / "system").rglob("*.py")) + list((root / "client").rglob("*.py")) + list((root / "tests").rglob("*.py"))
        for path in files:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module]
                elif isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                else:
                    continue
                for name in names:
                    with self.subTest(file=path.name, module=name):
                        self.assertIsNotNone(importlib.util.find_spec(name), f"{path}: нет модуля {name}")

    def test_client_commands_build(self) -> None:
        spec = importlib.util.spec_from_file_location("client_run", Path(__file__).resolve().parents[1] / "client" / "run.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        parser = module.Commands().parser
        for command in ("ladder", "layout", "evaluate", "predict", "figures", "pinwheel", "detectors", "sketch", "contour", "dashboard"):
            with self.subTest(command=command):
                self.assertIn(command, parser._subparsers._group_actions[0].choices)


if __name__ == "__main__":
    unittest.main()
