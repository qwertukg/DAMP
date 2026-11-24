"""Кеш шагов: готовый результат шага берётся с диска, если его вход, настройки, код и окружение те же (вне статьи).

Каждый шаг — запись сама по себе, независимо от других: ключ — хеш содержимого входа шага, значений настроек,
которые может читать код шага, самого кода (по дереву разбора: пояснения и комментарии не в счёт) и окружения
(версии библиотек, устройство). Настройки, действующие с уровня, шаг уровня может отдать в ключ их действующим
значением на своём уровне (level_settings у run). Пересчитанный шаг,
давший тот же результат, не трогает следующие: у них тот же вход («ранний обрыв» систем сборки). Массивы
хранятся один раз по хешу своего содержимого и делятся между записями.

    data/cache/steps/<шаг>/<ключ>.json   — что вошло в ключ, где лежат выходы, сколько секунд стоил счёт
    data/cache/blobs/<xx>/<хеш>.npy      — массивы выходов (без pickle)

Запись атомарная (временный файл → переименование). Сверка: доля CACHE_VERIFY_SHARE попаданий пересчитывается и
сравнивается с кешем бит в бит; расхождение — предупреждение и свежий результат. Предел размера — CACHE_MAX_BYTES:
сверх него удаляются давно не нужные записи. Выключение — CACHE_ENABLED = false либо DAMP_CACHE=off; на время —
CACHE.bypassed() (одна картинка дашборда: шаги на каждый штрих в кеш не пишутся).
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from damp import settings
from damp.article_refs import OUTSIDE_ARTICLE
from damp.logging import LOGGER

FORMAT = 2  # версия формата записей: меняется — старые записи не находятся (2 — ключ кода по дереву разбора, С3)
SYSTEM_ROOT = Path(__file__).resolve().parents[1]  # папка system/: в ней пакет damp


def _digest(*parts: bytes) -> str:
    hasher = hashlib.blake2b(digest_size=20)
    for part in parts:
        hasher.update(len(part).to_bytes(8, "little"))
        hasher.update(part)
    return hasher.hexdigest()


def fingerprint(value: object) -> str:
    """Хеш значения входа: массив — по типу, форме и байтам; остальное — по repr (числа, строки, кортежи)."""
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return _digest(b"array", str(array.dtype).encode(), str(array.shape).encode(), array.tobytes())
    if isinstance(value, (list, tuple)) and any(isinstance(item, np.ndarray) for item in value):
        return _digest(b"seq", *(fingerprint(item).encode() for item in value))
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    return _digest(b"plain", repr(value).encode())


class CodeFingerprint:
    """Код шага и настройки, которые он может читать: модуль шага и все системные модули, которые он импортирует.

    Код хешируется по дереву разбора (ast), а не текстом: строки документации, комментарии, пустые строки и перенос
    строк в ключ не входят — правка пояснений не сбрасывает кеш (С3, docs/SPEEDUP.md). Имена настроек — из узлов кода
    settings.ИМЯ (и getattr/hasattr(settings, "ИМЯ")), а не из текста: имя, упомянутое только в пояснении, в ключ не
    идёт. Импорты под `if TYPE_CHECKING:` в замыкание не входят (они не исполняются); относительные импорты («from
    .x import y») входят. Шаг может исключить модули, которые на его результат не влияют (ignore у StepCache.run, с
    пояснением у шага). Служебные модули (журнал, настройки, окружение, кеш) в код шага не входят: на счёт они не
    влияют; значения настроек входят в ключ сами.
    """

    INFRASTRUCTURE = frozenset({"damp", "damp.logging", "damp.settings", "damp.environment", "damp.article_refs", "damp.cache"})
    # Настройки, которые не меняют результат шага: журнал, кеш, Rerun, рисунки.
    IGNORED = ("LOG_", "CACHE_", "RERUN_", "FIGURE_", "README_FIGURES_")
    SETTING = re.compile(r"^[A-Z][A-Z0-9_]*$")

    def __init__(self) -> None:
        self._known: dict[tuple, tuple[str, tuple[str, ...]]] = {}
        self._modules: dict[str, tuple[str, frozenset[str], frozenset[str]]] = {}

    @staticmethod
    def _file(module: str) -> Path | None:
        relative = Path(*module.split("."))
        for candidate in (SYSTEM_ROOT / relative.with_suffix(".py"), SYSTEM_ROOT / relative / "__init__.py"):
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _type_checking(test: ast.expr) -> bool:
        """Условие `TYPE_CHECKING` / `typing.TYPE_CHECKING`: такие импорты нужны только проверке типов."""
        return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
            isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
        )

    def _imports(self, module: str, tree: ast.Module, path: Path) -> set[str]:
        """Системные модули, которые импортирует module (в любом месте кода, кроме веток `if TYPE_CHECKING:`)."""
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        found: set[str] = set()

        def visit(node: ast.AST) -> None:
            if isinstance(node, ast.If) and self._type_checking(node.test):
                for child in node.orelse:
                    visit(child)
                return
            names: list[str] = []
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    parts = package.split(".")
                    target = ".".join(parts[: len(parts) - (node.level - 1)] + ([node.module] if node.module else []))
                else:
                    target = node.module or ""
                if target.split(".")[0] == "damp":
                    names.append(target)
                    names.extend(f"{target}.{alias.name}" for alias in node.names)  # «from damp.x import модуль»
            elif isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names if alias.name.split(".")[0] == "damp")
            found.update(name for name in names if self._file(name) is not None)
            for child in ast.iter_child_nodes(node):
                visit(child)

        visit(tree)
        return found

    @classmethod
    def _settings(cls, tree: ast.Module) -> set[str]:
        """Имена настроек, которые читает код: settings.ИМЯ (и псевдонимы `from damp import settings as x`) и
        getattr/hasattr(settings, "ИМЯ")."""
        aliases = {"settings"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "damp":
                aliases.update(alias.asname for alias in node.names if alias.name == "settings" and alias.asname)
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases:
                if cls.SETTING.match(node.attr):
                    found.add(node.attr)
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("getattr", "hasattr")
                  and len(node.args) >= 2 and isinstance(node.args[0], ast.Name) and node.args[0].id in aliases
                  and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str)
                  and cls.SETTING.match(node.args[1].value)):
                found.add(node.args[1].value)
        return found

    @staticmethod
    def _code(tree: ast.Module) -> str:
        """Дерево кода без строк документации и строк-пояснений (выражение-строка ничего не исполняет) и без мест
        в файле (строки, столбцы): комментарии и разметка в него не попадают вовсе."""
        for node in ast.walk(tree):
            for name in ("body", "orelse", "finalbody"):
                body = getattr(node, name, None)
                if isinstance(body, list) and body and all(isinstance(item, ast.stmt) for item in body):
                    kept = [item for item in body
                            if not (isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant)
                                    and isinstance(item.value.value, str))]
                    setattr(node, name, kept or ([ast.Pass()] if name == "body" else []))
        return ast.dump(tree, annotate_fields=True, include_attributes=False)

    def _module(self, module: str) -> tuple[str, frozenset[str], frozenset[str]]:
        """(хеш кода модуля, его системные импорты, имена настроек) — по одному разбору на модуль за процесс."""
        if module not in self._modules:
            path = self._file(module)
            if path is None:
                raise ValueError(f"нет модуля {module} в {SYSTEM_ROOT}")
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imports, names = frozenset(self._imports(module, tree, path)), frozenset(self._settings(tree))
            self._modules[module] = (_digest(module.encode(), self._code(tree).encode()), imports, names)
        return self._modules[module]

    def closure(self, modules: tuple[str, ...], ignore: tuple[str, ...] = ()) -> list[str]:
        """Модули кода шага: modules и все системные модули, которые они импортируют, кроме ignore (и их импортов,
        если до них нет другого пути) и служебных."""
        seen: set[str] = set()
        queue = list(modules)
        while queue:
            module = queue.pop()
            if module in seen or module in ignore:
                continue
            seen.add(module)
            queue.extend(sorted(self._module(module)[1] - seen))
        return [name for name in sorted(seen) if name not in self.INFRASTRUCTURE]

    def of(self, modules: tuple[str, ...], ignore: tuple[str, ...] = ()) -> tuple[str, tuple[str, ...]]:
        """(хеш кода шага, имена настроек, которые он может читать)."""
        known = (tuple(modules), tuple(sorted(ignore)))
        if known in self._known:
            return self._known[known]
        code = self.closure(tuple(modules), tuple(ignore))
        digest = _digest(*(self._module(name)[0].encode() for name in code))
        names = sorted({name for module in code for name in self._module(module)[2] if not name.startswith(self.IGNORED)})
        self._known[known] = (digest, tuple(names))
        return self._known[known]


def environment() -> dict[str, str]:
    """Окружение, от которого может зависеть счёт: версии библиотек и устройство."""
    found = {"python": f"{sys.version_info.major}.{sys.version_info.minor}", "numpy": np.__version__}
    try:
        import torch

        found["torch"] = torch.__version__
        found["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        found["device"] = "cpu"
    try:
        import numba

        found["numba"] = numba.__version__
    except ImportError:
        pass
    return found


class StepCache:
    """Кеш результатов шагов на диске: запись на каждый шаг, ключ — от содержимого его входа."""

    def __init__(self) -> None:
        self._code = CodeFingerprint()
        self._environment: dict[str, str] | None = None
        self._size: int | None = None  # байт под кешем в папке _size_root (считается один раз, дальше — приростом)
        self._size_root: Path | None = None
        self._bypass = 0  # > 0 — шаги считаются заново и в кеш не пишутся (bypassed)

    # ------------------------------------------------------------------ где и включён ли
    @property
    def enabled(self) -> bool:
        return bool(settings.CACHE_ENABLED) and os.environ.get("DAMP_CACHE", "on").lower() != "off"

    @contextmanager
    def bypassed(self):
        """Внутри блока каждый шаг считается заново и в кеш не пишется: так идёт одна картинка дашборда — записи на
        каждый штрих только засоряли бы кеш, а счёт шага тот же, что при промахе."""
        self._bypass += 1
        try:
            yield
        finally:
            self._bypass -= 1

    @property
    def root(self) -> Path:
        return Path(settings.CACHE_DIR)

    def _entry_path(self, step: str, key: str) -> Path:
        return self.root / "steps" / step / f"{key}.json"

    def _blob_path(self, digest: str) -> Path:
        return self.root / "blobs" / digest[:2] / f"{digest}.npy"

    # ------------------------------------------------------------------ ключ
    def key(
        self,
        step: str,
        inputs: Mapping[str, object],
        modules: tuple[str, ...],
        *,
        ignore: tuple[str, ...] = (),
        level_settings: Mapping[str, object] | None = None,
    ) -> tuple[str, dict[str, object]]:
        """Ключ записи шага и то, из чего он сложен (для файла записи).

        ignore — модули, которые на результат шага не влияют (в замыкание кода не входят); level_settings —
        настройки, действующие с уровня: имя → действующее значение на уровне шага (в ключ идёт оно, а не значение
        настройки целиком; задаёт шаг уровня рядом с кодом, который эти настройки читает)."""
        if self._environment is None:
            self._environment = environment()
        code, names = self._code.of(tuple(modules), tuple(ignore))
        level = dict(level_settings or {})
        values = {name: repr(getattr(settings, name)) for name in names if hasattr(settings, name) and name not in level}
        parts = {
            "format": FORMAT,
            "step": step,
            "inputs": {name: fingerprint(value) for name, value in sorted(inputs.items())},
            "settings": values,
            "code": code,
            "environment": self._environment,
        }
        if level:
            parts["level_settings"] = {name: repr(value) for name, value in sorted(level.items())}
        return _digest(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()), parts

    # ------------------------------------------------------------------ шаг
    def run(
        self,
        step: str,
        inputs: Mapping[str, object],
        compute: Callable[[], Mapping[str, object]],
        *,
        modules: tuple[str, ...],
        ignore: tuple[str, ...] = (),
        level_settings: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Выходы шага: из кеша, если такая запись есть, иначе compute() — и запись в кеш.

        inputs — всё, от чего зависит результат, кроме настроек и кода (их кеш берёт сам); выходы — массивы numpy и
        простые значения JSON (числа, строки, списки). modules — модули с кодом шага; ignore — модули из их
        импортов, которые на результат не влияют; level_settings — действующие значения настроек «с уровня» (key).
        """
        if not self.enabled or self._bypass:
            return dict(compute())
        key, parts = self.key(step, inputs, modules, ignore=ignore, level_settings=level_settings)
        entry_path = self._entry_path(step, key)
        cached = self._read(entry_path)
        if cached is not None:
            entry, outputs = cached
            if self._checked(key):
                fresh = dict(compute())
                same = self.same(outputs, fresh)
                LOGGER.event(
                    "cache.verify" if same else "cache.mismatch.warning",
                    section=OUTSIDE_ARTICLE,
                    data={"step": step, "key": key[:16], "same": same,
                          "note": "сверка: шаг посчитан заново и сравнён с кешем бит в бит"},
                )
                if not same:
                    self._write(entry_path, parts, fresh, float(entry.get("seconds", 0.0)))
                    return fresh
            os.utime(entry_path)  # давность использования — для вытеснения старых записей
            LOGGER.event(
                "cache.hit",
                section=OUTSIDE_ARTICLE,
                data={"step": step, "key": key[:16], "saved_seconds": float(entry.get("seconds", 0.0))},
            )
            return outputs
        start = time.perf_counter()
        outputs = dict(compute())
        seconds = time.perf_counter() - start
        size = self._write(entry_path, parts, outputs, seconds)
        LOGGER.event(
            "cache.miss",
            section=OUTSIDE_ARTICLE,
            data={"step": step, "key": key[:16], "seconds": seconds, "bytes": size},
        )
        self._evict()
        return outputs

    def _checked(self, key: str) -> bool:
        """Попадает ли запись в сверяемую долю CACHE_VERIFY_SHARE (выбор по ключу — одинаков от прогона к прогону)."""
        share = float(settings.CACHE_VERIFY_SHARE)
        return share > 0.0 and int(key[:8], 16) / float(0xFFFFFFFF) < share

    @staticmethod
    def same(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
        """Совпадают ли выходы бит в бит: массивы — тип, форма и байты; значения — после записи в JSON."""
        if set(left) != set(right):
            return False
        for name, value in left.items():
            other = right[name]
            if isinstance(value, np.ndarray) or isinstance(other, np.ndarray):
                a, b = np.asarray(value), np.asarray(other)
                if a.dtype != b.dtype or a.shape != b.shape or np.ascontiguousarray(a).tobytes() != np.ascontiguousarray(b).tobytes():
                    return False
            elif json.dumps(_plain(value), sort_keys=True) != json.dumps(_plain(other), sort_keys=True):
                return False
        return True

    # ------------------------------------------------------------------ диск
    def _read(self, path: Path) -> tuple[dict, dict[str, object]] | None:
        if not path.exists():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            outputs: dict[str, object] = {}
            for name, item in entry["outputs"].items():
                if "blob" in item:
                    outputs[name] = np.load(self._blob_path(item["blob"]), allow_pickle=False)
                else:
                    outputs[name] = item["value"]
            return entry, outputs
        except (OSError, ValueError, KeyError) as error:
            LOGGER.event(
                "cache.read.warning",
                section=OUTSIDE_ARTICLE,
                data={"path": str(path), "error": repr(error), "note": "запись не читается — шаг считается заново"},
            )
            return None

    @staticmethod
    def _place(temporary: Path, target: Path) -> bool:
        """Подменить target готовым файлом temporary. Два процесса могут посчитать один и тот же шаг одновременно (пары
        прогонов подбора): на Windows подмена файла, который держит другой процесс, падает с отказом в доступе. Тогда
        тот же шаг уже записан соседом — блоб назван по содержимому, запись шага — по ключу входа, — и своя копия не
        нужна. False — запись оставлена соседу."""
        for attempt in range(5):
            try:
                os.replace(temporary, target)
                return True
            except PermissionError:
                if target.exists():
                    temporary.unlink(missing_ok=True)
                    return False
                time.sleep(0.2 * (attempt + 1))
        os.replace(temporary, target)
        return True

    def _write(self, path: Path, parts: dict[str, object], outputs: Mapping[str, object], seconds: float) -> int:
        stored: dict[str, object] = {}
        size = 0
        for name, value in outputs.items():
            if isinstance(value, np.ndarray):
                array = np.ascontiguousarray(value)
                digest = fingerprint(array)
                blob = self._blob_path(digest)
                if not blob.exists():
                    blob.parent.mkdir(parents=True, exist_ok=True)
                    temporary = blob.with_name(f"{blob.stem}.{os.getpid()}.tmp.npy")
                    np.save(temporary, array, allow_pickle=False)
                    if self._place(temporary, blob):
                        size += blob.stat().st_size
                stored[name] = {"blob": digest, "dtype": str(array.dtype), "shape": list(array.shape)}
            else:
                stored[name] = {"value": _plain(value)}
        entry = {**parts, "outputs": stored, "seconds": seconds, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
        self._place(temporary, path)
        size += path.stat().st_size
        if self._size is not None and self._size_root == self.root:
            self._size += size
        self._counter_add(size)
        return size

    # ------------------------------------------------------------------ размер
    def entries(self) -> list[tuple[str, Path, int, float]]:
        """Записи кеша: (шаг, файл записи, байты выходов, давность использования — mtime)."""
        found = []
        steps = self.root / "steps"
        if not steps.exists():
            return found
        for path in steps.glob("*/*.json"):
            try:
                entry = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            try:
                size = path.stat().st_size + sum(
                    self._blob_path(item["blob"]).stat().st_size
                    for item in entry.get("outputs", {}).values()
                    if "blob" in item and self._blob_path(item["blob"]).exists()
                )
                found.append((path.parent.name, path, size, path.stat().st_mtime))
            except OSError:
                continue  # запись удалил другой процесс (два прогона на одном кеше вытесняют одновременно)
        return found

    def size(self) -> int:
        """Байт на диске под кешем."""
        if not self.root.exists():
            return 0
        total = 0
        for path in self.root.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue  # файл удалил другой процесс
        return total

    # Счётчик размера на диске (size.json в корне кеша): общий для всех процессов, чтобы первый промах процесса не обходил
    # десятки тысяч файлов (8 с). Приблизителен: растёт на размер записанных записей и массивов, после вытеснения и
    # пересчёта обходом выставляется точно. Нет файла, он порван или старше COUNTER_MAX_AGE_SECONDS — размер считается
    # обходом и счётчик пишется заново. Результат шагов от счётчика не зависит (он решает только, когда вытеснять).
    COUNTER_NAME = "size.json"
    COUNTER_MAX_AGE_SECONDS = 24 * 3600
    COUNTER_LOCK_SECONDS = 30.0  # замок старше — оставлен упавшим процессом, снимается

    def _counter_path(self) -> Path:
        return self.root / self.COUNTER_NAME

    def _counter_locked(self, action: Callable[[], object]) -> object | None:
        """Выполнить action под замком (файл создаётся атомарно). None — замок не взят: счётчик не обновляется, это
        безопасно (он приблизителен и пересчитывается обходом)."""
        lock = self.root / (self.COUNTER_NAME + ".lock")
        deadline = time.time() + 5.0
        while True:
            try:
                handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(handle)
                break
            except FileExistsError:
                try:
                    if time.time() - lock.stat().st_mtime > self.COUNTER_LOCK_SECONDS:
                        lock.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                if time.time() > deadline:
                    return None
                time.sleep(0.01)
            except OSError:
                return None
        try:
            return action()
        finally:
            try:
                lock.unlink(missing_ok=True)
            except OSError:
                pass

    def _counter_read(self) -> tuple[int, float] | None:
        """(байт, время последнего точного пересчёта) или None, если счётчика нет / он порван / устарел."""
        try:
            data = json.loads(self._counter_path().read_text(encoding="utf-8"))
            total, walked = int(data["bytes"]), float(data["walked"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if total < 0 or time.time() - walked > self.COUNTER_MAX_AGE_SECONDS:
            return None
        return total, walked

    def _counter_store(self, total: int, walked: float) -> None:
        path = self._counter_path()
        temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps({"bytes": int(total), "walked": walked}), encoding="utf-8")
            os.replace(temporary, path)
        except OSError:
            temporary.unlink(missing_ok=True)

    def _counter_add(self, delta: int) -> None:
        """Прибавить delta к счётчику (если он есть и цел): под замком, чтобы два процесса пары не затирали друг друга."""
        if delta == 0:
            return

        def add() -> None:
            current = self._counter_read()
            if current is not None:
                self._counter_store(max(0, current[0] + delta), current[1])

        self._counter_locked(add)

    def _counter_set(self, total: int) -> None:
        """Выставить счётчик точным размером (после обхода)."""
        self._counter_locked(lambda: self._counter_store(total, time.time()))

    def _counter_size(self) -> int:
        """Размер кеша: из счётчика, а нет его — обходом (и счётчик пишется)."""
        current = self._counter_read()
        if current is not None:
            return current[0]
        total = self.size()
        self._counter_set(total)
        return total

    # До какой доли предела вытеснять. Было 0.9: у предела 20 ГиБ вытеснение (обход кеша, 150–230 с) шло раз в 2 ГиБ записей —
    # на стенде с добавочными картинками каждые 2–3 прогона, 8–10 % времени стенда (2026-11-24). 0.75 — в 2.5 раза реже.
    EVICT_TO = 0.75

    def _evict(self) -> None:
        """Сверх CACHE_MAX_BYTES — удалить давно не нужные записи до доли EVICT_TO предела и массивы без ссылок."""
        limit = int(settings.CACHE_MAX_BYTES)
        self._size, self._size_root = self._counter_size(), self.root
        if self._size <= limit:
            return
        self._size = self.size()  # счётчик приблизителен: перед вытеснением — точный размер
        self._counter_set(self._size)
        if self._size <= limit:
            return
        # Старые записи удаляются за один проход: размер записи (с её массивами, entries()) вычитается из счётчика, а
        # массивы без ссылок убираются после прохода (collect) — диск пересчитывается один раз. Прежде после каждой
        # удалённой записи кеш обходился заново, а её массивы ещё лежали на диске: размер не падал, и проход стирал
        # записи одну за другой по секунде на каждую (2026-11-20: подбор стоял часами в _evict). Массивы, общие с
        # оставшимися записями, не освобождаются, поэтому проход повторяется, пока размер не опустится до 90 % предела.
        removed = freed = 0
        target = int(limit * self.EVICT_TO)
        for _ in range(8):
            rows = sorted(self.entries(), key=lambda row: row[3])
            if not rows:
                break
            estimate = self._size
            for _, path, size, _ in rows:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    # 2026-11-21 (прогон 641): два прогона пары вытесняли одновременно, и файл, занятый другим процессом,
                    # не удалился — PermissionError ронял прогон. Такую запись пропускаем: её удалят в следующий раз.
                    continue
                removed += 1
                estimate -= size
                if estimate <= target:
                    break
            freed += self.collect()
            self._size = self.size()
            self._counter_set(self._size)
            if self._size <= target:
                break
        LOGGER.event(
            "cache.evict",
            section=OUTSIDE_ARTICLE,
            data={"entries": removed, "blobs": freed, "bytes": self._size, "limit": limit},
        )

    # Массив моложе стольких секунд не удаляется, даже если записи на него ещё нет: его мог только что записать другой
    # процесс (два прогона на одном кеше), а его запись ляжет следом.
    COLLECT_GRACE_SECONDS = 600

    def collect(self) -> int:
        """Удалить массивы, на которые не ссылается ни одна запись; вернуть, сколько удалено."""
        used: set[str] = set()
        for _, path, _, _ in self.entries():
            try:
                entry = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            used.update(item["blob"] for item in entry.get("outputs", {}).values() if "blob" in item)
        freed = 0
        blobs = self.root / "blobs"
        if blobs.exists():
            now = time.time()
            for blob in blobs.glob("*/*.npy"):
                if blob.stem not in used:
                    try:
                        # Свежий массив мог только что записать другой процесс, а его запись ещё не легла: не трогать.
                        if now - blob.stat().st_mtime < self.COLLECT_GRACE_SECONDS:
                            continue
                        blob.unlink(missing_ok=True)
                    except OSError:
                        continue
                    freed += 1
        return freed

    def clear(self, step: str | None = None) -> int:
        """Удалить записи шага step (None — все) и ставшие ненужными массивы; вернуть число удалённых записей."""
        removed = 0
        for name, path, _, _ in self.entries():
            if step is None or name == step:
                path.unlink(missing_ok=True)
                removed += 1
        if step is None:
            for path in (self.root / "pictures").glob("*") if (self.root / "pictures").exists() else ():
                path.unlink(missing_ok=True)
        self.collect()
        self._size = None
        self._counter_locked(lambda: self._counter_path().unlink(missing_ok=True))
        LOGGER.event("cache.clear", section=OUTSIDE_ARTICLE, data={"step": step or "все", "entries": removed})
        return removed

    def summary(self) -> list[dict[str, object]]:
        """По шагу: записей, байт, давность последнего использования (секунды назад)."""
        now = time.time()
        rows: dict[str, dict[str, object]] = {}
        for step, _, size, used in self.entries():
            row = rows.setdefault(step, {"step": step, "entries": 0, "bytes": 0, "last_used_seconds": None})
            row["entries"] = int(row["entries"]) + 1
            row["bytes"] = int(row["bytes"]) + size
            ago = now - used
            row["last_used_seconds"] = ago if row["last_used_seconds"] is None else min(float(row["last_used_seconds"]), ago)
        pictures = self.root / "pictures"
        if pictures.exists():
            stores = list(pictures.glob("*.npz"))
            if stores:
                rows["stimuli.sticks"] = {
                    "step": "stimuli.sticks",
                    "entries": len(stores),
                    "bytes": sum(path.stat().st_size for path in stores),
                    "last_used_seconds": now - max(path.stat().st_mtime for path in stores),
                }
        return sorted(rows.values(), key=lambda row: str(row["step"]))


def _plain(value: object) -> object:
    """Значение выхода → значение JSON: кортежи — списки, числа numpy — числа Python."""
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


CACHE = StepCache()
