"""Снимок графа объектов системного слоя на диске — без pickle (вне статьи: техника хранения).

Граф (например, обученный цикл) пишется двумя файлами: структура — JSON (`graph.json`), массивы — один npz без pickle
(`arrays.npz`). Восстанавливаются только классы пакета `damp` (по имени модуля и класса), контейнеры Python, массивы
numpy и тензоры torch, генераторы случайных чисел; любой другой объект — ошибка с путём к нему в графе. Общие ссылки и
циклы сохраняются: объект, на который ссылаются дважды, после загрузки тоже один.

Объект восстанавливается без своего __init__: `cls.__new__(cls)` и его атрибуты. Поэтому снимок годится только для
того же кода, что его записал (версию и коммит сверяет тот, кто пишет снимок, — например TrainedLadder.save).
Атрибуты-кеши, которые объект строит сам по запросу (память видеокарты), пишутся пустыми — `transient`.
"""

from __future__ import annotations

import base64
import enum
import importlib
import json
import random
import sys
import types
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

GRAPH_FILE = "graph.json"
ARRAYS_FILE = "arrays.npz"
FORMAT = 1


class SnapshotError(ValueError):
    """Граф нельзя записать или прочитать: в нём чужой объект, другой формат или нет файла."""


class GraphSnapshot:
    """Запись и чтение графа объектов: структура — JSON, массивы — npz (allow_pickle=False при чтении).

    transient — имена атрибутов по имени класса, которые пишутся как None (кеши, которые объект строит сам);
    rebuild — по имени класса: что вызвать над объектом после чтения, чтобы заново построить пропущенное (то, что
    строит его __init__ из уже записанного, — копии тех же данных в другом виде).
    Числовые списки и множества не короче INLINE и байты длиннее BYTES_INLINE уходят в npz, чтобы JSON не разрастался.
    """

    INLINE = 64
    BYTES_INLINE = 4096
    SAME_BYTES = 1 << 20  # массивы от 1 МБ с одинаковым содержимым пишутся один раз (копии кодов карты)
    PACKAGE = "damp"
    MODULES = frozenset({"torch", "numpy"})  # модули, ссылку на которые объект может держать атрибутом

    def __init__(self, transient: Mapping[str, frozenset[str] | set[str]] | None = None,
                 rebuild: Mapping[str, Callable[[object], None]] | None = None) -> None:
        self.transient = {name: frozenset(attributes) for name, attributes in (transient or {}).items()}
        self.rebuild = dict(rebuild or {})

    # ------------------------------------------------------------------ запись
    def save(self, root: object, folder: str | Path) -> dict[str, int]:
        """Записать граф с корнем root в папку folder; вернуть счёт: объектов, массивов, байт массивов."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self._ids: dict[int, int] = {}
        self._alive: list[object] = []  # всё, что запомнено по id, живо до конца записи: иначе id освободится и повторится
        self._arrays: dict[str, np.ndarray] = {}
        self._digests: dict[tuple, str] = {}
        self._objects = 0
        limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(limit, 20000))
        try:
            graph = {"format": FORMAT, "root": self._encode(root, "root")}
        finally:
            sys.setrecursionlimit(limit)
        with open(folder / GRAPH_FILE, "w", encoding="utf-8") as stream:
            json.dump(graph, stream, ensure_ascii=False, separators=(",", ":"))
        np.savez(folder / ARRAYS_FILE, **self._arrays)
        counts = {
            "objects": self._objects,
            "arrays": len(self._arrays),
            "array_bytes": int(sum(int(array.nbytes) for array in self._arrays.values())),
        }
        del self._ids, self._alive, self._arrays, self._digests
        return counts

    def _array(self, array: np.ndarray) -> str:
        """Ключ массива в npz. Большие одинаковые массивы (разные объекты с одним содержимым) — один ключ: при чтении
        каждый получает свою копию, как и было."""
        array = np.ascontiguousarray(array)
        digest = None
        if array.nbytes >= self.SAME_BYTES:
            import hashlib

            digest = (array.dtype.str, array.shape, hashlib.blake2b(memoryview(array).cast("B"), digest_size=32).hexdigest())
            if digest in self._digests:
                return self._digests[digest]
        key = f"a{len(self._arrays)}"
        self._arrays[key] = array
        if digest is not None:
            self._digests[digest] = key
        return key

    def _seen(self, value: object) -> "dict | None":
        number = self._ids.get(id(value))
        return None if number is None else {"$ref": number}

    def _remember(self, value: object) -> int:
        number = len(self._ids)
        self._ids[id(value)] = number
        self._alive.append(value)
        return number

    @staticmethod
    def _numbers(values: "list | tuple | set | frozenset") -> "tuple[np.ndarray, str] | None":
        """Список чисел одного типа как массив: целые в int64 либо float; иначе None (идёт в JSON)."""
        items = list(values)
        if all(type(item) is int for item in items):
            if -(2**63) <= min(items) and max(items) < 2**63:
                return np.asarray(items, dtype=np.int64), "int"
            return None
        if all(type(item) is float for item in items):
            return np.asarray(items, dtype=np.float64), "float"
        return None

    def _encode(self, value: object, path: str) -> object:  # noqa: C901, PLR0911, PLR0912 — разбор по типам
        if value is None or isinstance(value, (bool, str)):
            return value
        if isinstance(value, np.generic):
            if value.dtype == object:
                raise SnapshotError(f"{path}: скаляр numpy с типом object")
            return {"$np": value.dtype.str, "v": value.item()}
        if isinstance(value, enum.Enum):
            return {"$enum": self._class_name(type(value), path), "name": value.name}
        if type(value) is int or type(value) is float:
            return value
        seen = self._seen(value)
        if seen is not None:
            return seen
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                raise SnapshotError(f"{path}: массив numpy с типом object")
            return {"$nd": self._array(value), "$id": self._remember(value)}
        if isinstance(value, (bytes, bytearray)):
            kind = "bytearray" if isinstance(value, bytearray) else "bytes"
            node: dict = {"$bytes": kind}
            if isinstance(value, bytearray):
                node["$id"] = self._remember(value)
            if len(value) > self.BYTES_INLINE:
                node["key"] = self._array(np.frombuffer(bytes(value), dtype=np.uint8))
            else:
                node["b64"] = base64.b64encode(bytes(value)).decode("ascii")
            return node
        if isinstance(value, (list, tuple, set, frozenset)):
            kind = {list: "list", tuple: "tuple", set: "set", frozenset: "fset"}.get(type(value))
            if kind is None:
                raise SnapshotError(f"{path}: контейнер {type(value).__module__}.{type(value).__qualname__}")
            node = {}
            if kind in ("list", "set"):
                node["$id"] = self._remember(value)
            if len(value) >= self.INLINE:
                numbers = self._numbers(value)
                if numbers is not None:
                    node.update({"$nums": self._array(numbers[0]), "kind": kind, "type": numbers[1]})
                    return node
            items = list(value)
            if kind in ("set", "fset"):
                items = sorted(items, key=repr)  # порядок в файле не важен; repr — чтобы запись была повторяемой
            node["$" + kind] = [self._encode(item, f"{path}[{index}]") for index, item in enumerate(items)]
            return node
        if isinstance(value, dict):
            if type(value) is not dict:
                raise SnapshotError(f"{path}: словарь {type(value).__module__}.{type(value).__qualname__}")
            number = self._remember(value)
            return {
                "$dict": [[self._encode(key, f"{path}<key>"), self._encode(item, f"{path}[{key!r}]")] for key, item in value.items()],
                "$id": number,
            }
        if isinstance(value, types.ModuleType):
            if value.__name__ not in self.MODULES:
                raise SnapshotError(f"{path}: ссылка на модуль {value.__name__}")
            return {"$module": value.__name__}
        if isinstance(value, random.Random):
            number = self._remember(value)
            version, internal, gauss = value.getstate()
            return {"$random": [version, list(internal), gauss], "$id": number}
        if isinstance(value, np.random.Generator):
            number = self._remember(value)
            return {"$npgen": self._encode(value.bit_generator.state, f"{path}.state"), "$id": number}
        if isinstance(value, np.random.RandomState):
            number = self._remember(value)
            return {"$rstate": self._encode(value.get_state(legacy=False), f"{path}.state"), "$id": number}
        torch = sys.modules.get("torch")
        if torch is not None:
            if isinstance(value, torch.Tensor):
                number = self._remember(value)
                data = value.detach().cpu()
                if data.dtype == torch.bfloat16:
                    raise SnapshotError(f"{path}: тензор bfloat16")
                return {"$tensor": self._array(data.numpy()), "device": str(value.device), "$id": number}
            if isinstance(value, torch.device):
                return {"$device": str(value)}
        if isinstance(value, (type, types.FunctionType, types.MethodType, types.BuiltinFunctionType)):
            raise SnapshotError(f"{path}: ссылка на код ({value!r}) — такой объект не пишется")
        return self._object(value, path)

    def _class_name(self, cls: type, path: str) -> str:
        module = cls.__module__
        if module != self.PACKAGE and not module.startswith(self.PACKAGE + "."):
            raise SnapshotError(f"{path}: объект {module}.{cls.__qualname__} — не из пакета {self.PACKAGE}")
        if "<locals>" in cls.__qualname__:
            raise SnapshotError(f"{path}: класс {cls.__qualname__} объявлен внутри функции")
        return f"{module}:{cls.__qualname__}"

    def _object(self, value: object, path: str) -> dict:
        cls = type(value)
        name = self._class_name(cls, path)
        number = self._remember(value)
        self._objects += 1
        skip = frozenset().union(*(self.transient.get(klass.__name__, frozenset()) for klass in cls.__mro__))
        node: dict = {"$obj": name, "$id": number}
        state = getattr(value, "__dict__", None)
        if state is not None:
            if not isinstance(state, dict):
                raise SnapshotError(f"{path}: у {name} __dict__ не словарь")
            node["dict"] = {key: (None if key in skip else self._encode(item, f"{path}.{key}")) for key, item in state.items()}
        slots = [slot for klass in cls.__mro__ for slot in getattr(klass, "__slots__", ()) if slot not in ("__dict__", "__weakref__")]
        if slots:
            node["slots"] = {
                slot: (None if slot in skip else self._encode(getattr(value, slot), f"{path}.{slot}"))
                for slot in slots if hasattr(value, slot)
            }
        if state is None and not slots:
            raise SnapshotError(f"{path}: у {name} нет состояния для записи")
        return node

    # ------------------------------------------------------------------ чтение
    def load(self, folder: str | Path, *, device: str | None = None) -> object:
        """Прочитать граф из папки folder. device — куда класть тензоры: None — туда, где они были (без видеокарты — на
        процессор)."""
        folder = Path(folder)
        graph_path, arrays_path = folder / GRAPH_FILE, folder / ARRAYS_FILE
        if not graph_path.exists() or not arrays_path.exists():
            raise SnapshotError(f"{folder}: нет {GRAPH_FILE} или {ARRAYS_FILE}")
        with open(graph_path, encoding="utf-8") as stream:
            graph = json.load(stream)
        if graph.get("format") != FORMAT:
            raise SnapshotError(f"{folder}: формат снимка {graph.get('format')}, этот код читает {FORMAT}")
        self._device = device
        self._made: dict[int, object] = {}
        limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(limit, 20000))
        try:
            with np.load(arrays_path, allow_pickle=False) as arrays:
                self._npz = arrays
                root = self._decode(graph["root"])
        finally:
            sys.setrecursionlimit(limit)
            self._npz = None
        del self._made
        return root

    def _keep(self, node: dict, value: object) -> object:
        if "$id" in node:
            self._made[int(node["$id"])] = value
        return value

    def _tensor_device(self, saved: str) -> str:
        import torch

        wanted = self._device or saved
        if wanted.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return wanted

    def _decode(self, node: object) -> object:  # noqa: C901, PLR0911, PLR0912 — разбор по видам узлов
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return self._made[int(node["$ref"])]
        if "$np" in node:
            return np.dtype(node["$np"]).type(node["v"])
        if "$nd" in node:
            return self._keep(node, np.array(self._npz[node["$nd"]]))
        if "$nums" in node:
            values = self._npz[node["$nums"]].tolist()
            built = {"list": list, "tuple": tuple, "set": set, "fset": frozenset}[node["kind"]](values)
            return self._keep(node, built)
        if "$bytes" in node:
            raw = bytes(np.asarray(self._npz[node["key"]], dtype=np.uint8)) if "key" in node else base64.b64decode(node["b64"])
            return self._keep(node, bytearray(raw) if node["$bytes"] == "bytearray" else raw)
        if "$list" in node:
            built: list = []
            self._keep(node, built)
            built.extend(self._decode(item) for item in node["$list"])
            return built
        if "$tuple" in node:
            return tuple(self._decode(item) for item in node["$tuple"])
        if "$set" in node:
            return self._keep(node, {self._decode(item) for item in node["$set"]})
        if "$fset" in node:
            return frozenset(self._decode(item) for item in node["$fset"])
        if "$dict" in node:
            built_dict: dict = {}
            self._keep(node, built_dict)
            for key, item in node["$dict"]:
                built_dict[self._decode(key)] = self._decode(item)
            return built_dict
        if "$module" in node:
            if node["$module"] not in self.MODULES:
                raise SnapshotError(f"модуль {node['$module']} не из разрешённых")
            return importlib.import_module(node["$module"])
        if "$random" in node:
            version, internal, gauss = node["$random"]
            generator = random.Random()
            generator.setstate((version, tuple(internal), gauss))
            return self._keep(node, generator)
        if "$npgen" in node:
            state = self._decode(node["$npgen"])
            bit = getattr(np.random, str(state["bit_generator"]))()
            bit.state = state
            return self._keep(node, np.random.Generator(bit))
        if "$rstate" in node:
            legacy = np.random.RandomState()
            legacy.set_state(self._decode(node["$rstate"]))
            return self._keep(node, legacy)
        if "$tensor" in node:
            import torch

            tensor = torch.from_numpy(np.array(self._npz[node["$tensor"]])).to(self._tensor_device(node["device"]))
            return self._keep(node, tensor)
        if "$device" in node:
            import torch

            return torch.device(self._tensor_device(node["$device"]))
        if "$enum" in node:
            return self._class(node["$enum"])[node["name"]]
        if "$obj" in node:
            cls = self._class(node["$obj"])
            value = cls.__new__(cls)
            self._keep(node, value)
            if "dict" in node:
                value.__dict__.update({key: self._decode(item) for key, item in node["dict"].items()})
            for slot, item in node.get("slots", {}).items():
                object.__setattr__(value, slot, self._decode(item))
            for klass in cls.__mro__:
                if klass.__name__ in self.rebuild:
                    self.rebuild[klass.__name__](value)
                    break
            return value
        raise SnapshotError(f"неизвестный узел снимка: {sorted(node)[:4]}")

    def _class(self, name: str) -> type:
        module, _, qualname = name.partition(":")
        if module != self.PACKAGE and not module.startswith(self.PACKAGE + "."):
            raise SnapshotError(f"класс {name} не из пакета {self.PACKAGE}")
        found: object = importlib.import_module(module)
        for part in qualname.split("."):
            found = getattr(found, part)
        if not isinstance(found, type):
            raise SnapshotError(f"{name} — не класс")
        return found
