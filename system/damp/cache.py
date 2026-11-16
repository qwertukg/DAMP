"""Кеш шагов: готовый результат шага берётся с диска, если его вход, настройки, код и окружение те же (вне статьи).

Каждый шаг — запись сама по себе, независимо от других: ключ — хеш содержимого входа шага, значений настроек,
которые может читать код шага, текста этого кода и окружения (версии библиотек, устройство). Пересчитанный шаг,
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

FORMAT = 1  # версия формата записей: меняется — старые записи не находятся
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

    Имена настроек собираются из текста кода (settings.ИМЯ) — с запасом: лишнее имя даёт лишний пересчёт, но не
    неверный результат. Служебные модули (журнал, настройки, окружение, кеш) в код шага не входят: на счёт они не
    влияют; значения настроек входят в ключ сами.
    """

    INFRASTRUCTURE = frozenset({"damp", "damp.logging", "damp.settings", "damp.environment", "damp.article_refs", "damp.cache"})
    # Настройки, которые не меняют результат шага: журнал, кеш, Rerun, рисунки.
    IGNORED = ("LOG_", "CACHE_", "RERUN_", "FIGURE_", "README_FIGURES_")
    SETTING = re.compile(r"\bsettings\.([A-Z][A-Z0-9_]*)")

    def __init__(self) -> None:
        self._known: dict[tuple[str, ...], tuple[str, tuple[str, ...]]] = {}

    @staticmethod
    def _file(module: str) -> Path | None:
        relative = Path(*module.split("."))
        for candidate in (SYSTEM_ROOT / relative.with_suffix(".py"), SYSTEM_ROOT / relative / "__init__.py"):
            if candidate.exists():
                return candidate
        return None

    def _imports(self, module: str, text: str) -> set[str]:
        found: set[str] = set()
        for node in ast.walk(ast.parse(text)):
            names: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and node.module.split(".")[0] == "damp":
                names.append(node.module)
                names.extend(f"{node.module}.{alias.name}" for alias in node.names)  # «from damp.x import модуль»
            elif isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names if alias.name.split(".")[0] == "damp")
            found.update(name for name in names if self._file(name) is not None)
        return found

    def of(self, modules: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
        """(хеш текста кода шага, имена настроек, которые он может читать)."""
        if modules in self._known:
            return self._known[modules]
        texts: dict[str, str] = {}
        queue = list(modules)
        while queue:
            module = queue.pop()
            if module in texts:
                continue
            path = self._file(module)
            if path is None:
                raise ValueError(f"нет модуля {module} в {SYSTEM_ROOT}")
            texts[module] = path.read_text(encoding="utf-8")
            queue.extend(sorted(self._imports(module, texts[module]) - set(texts)))
        code = [name for name in sorted(texts) if name not in self.INFRASTRUCTURE]
        digest = _digest(*(f"{name}\n{texts[name]}".encode() for name in code))
        names = sorted(
            {name for module in code for name in self.SETTING.findall(texts[module]) if not name.startswith(self.IGNORED)}
        )
        self._known[modules] = (digest, tuple(names))
        return self._known[modules]


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
    def key(self, step: str, inputs: Mapping[str, object], modules: tuple[str, ...]) -> tuple[str, dict[str, object]]:
        """Ключ записи шага и то, из чего он сложен (для файла записи)."""
        if self._environment is None:
            self._environment = environment()
        code, names = self._code.of(tuple(modules))
        values = {name: repr(getattr(settings, name)) for name in names if hasattr(settings, name)}
        parts = {
            "format": FORMAT,
            "step": step,
            "inputs": {name: fingerprint(value) for name, value in sorted(inputs.items())},
            "settings": values,
            "code": code,
            "environment": self._environment,
        }
        return _digest(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()), parts

    # ------------------------------------------------------------------ шаг
    def run(
        self,
        step: str,
        inputs: Mapping[str, object],
        compute: Callable[[], Mapping[str, object]],
        *,
        modules: tuple[str, ...],
    ) -> dict[str, object]:
        """Выходы шага: из кеша, если такая запись есть, иначе compute() — и запись в кеш.

        inputs — всё, от чего зависит результат, кроме настроек и кода (их кеш берёт сам); выходы — массивы numpy и
        простые значения JSON (числа, строки, списки). modules — модули с кодом шага.
        """
        if not self.enabled or self._bypass:
            return dict(compute())
        key, parts = self.key(step, inputs, modules)
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
                    os.replace(temporary, blob)
                    size += blob.stat().st_size
                stored[name] = {"blob": digest, "dtype": str(array.dtype), "shape": list(array.shape)}
            else:
                stored[name] = {"value": _plain(value)}
        entry = {**parts, "outputs": stored, "seconds": seconds, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(temporary, path)
        size += path.stat().st_size
        if self._size is not None and self._size_root == self.root:
            self._size += size
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
            size = path.stat().st_size + sum(
                self._blob_path(item["blob"]).stat().st_size
                for item in entry.get("outputs", {}).values()
                if "blob" in item and self._blob_path(item["blob"]).exists()
            )
            found.append((path.parent.name, path, size, path.stat().st_mtime))
        return found

    def size(self) -> int:
        """Байт на диске под кешем."""
        if not self.root.exists():
            return 0
        return sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file())

    def _evict(self) -> None:
        """Сверх CACHE_MAX_BYTES — удалить давно не нужные записи до 90 % предела и массивы без ссылок."""
        limit = int(settings.CACHE_MAX_BYTES)
        if self._size is None or self._size_root != self.root:
            self._size, self._size_root = self.size(), self.root
        if self._size <= limit:
            return
        removed = 0
        for step, path, _, _ in sorted(self.entries(), key=lambda row: row[3]):
            path.unlink(missing_ok=True)
            removed += 1
            self._size = None
            if self.size() <= int(limit * 0.9):
                break
        freed = self.collect()
        self._size = self.size()
        LOGGER.event(
            "cache.evict",
            section=OUTSIDE_ARTICLE,
            data={"entries": removed, "blobs": freed, "bytes": self._size, "limit": limit},
        )

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
            for blob in blobs.glob("*/*.npy"):
                if blob.stem not in used:
                    blob.unlink(missing_ok=True)
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
