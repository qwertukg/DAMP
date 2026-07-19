from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import os
from typing import Dict, List, Mapping, Sequence

from damp.article_refs import LAID_OUT_STRUCTURE
from damp.encoding.bitarray import BitArray
from damp.logging import LOGGER


@dataclass(frozen=True)
class LayoutEncodedCode:
    code: BitArray
    label: str


@dataclass(frozen=True)
class LayoutCodeRecord:
    index: int
    label: str
    hue: float
    code: BitArray

    @property
    def ones(self) -> int:
        return self.code.count()

    def as_payload(self, *, y: int | None, x: int | None) -> Dict[str, object]:
        return {
            "index": self.index,
            "y": y,
            "x": x,
            "label": self.label,
            "value": self.code.to01(),
            "hue": self.hue,
            "ones": self.ones,
        }


class LayoutPayloadBuilder:
    def __init__(
        self,
        *,
        similarity: str | None,
        lambda_threshold: float | None,
        eta: float | None,
    ) -> None:
        self._records: List[LayoutCodeRecord] = []
        self._similarity = similarity
        self._lambda_threshold = lambda_threshold
        self._eta = eta

    @property
    def records(self) -> Sequence[LayoutCodeRecord]:
        return tuple(self._records)

    @classmethod
    def load_base(cls, path: str) -> "LayoutPayloadBuilder":
        """Восстанавливает коды незавершённой раскладки без перекодирования (§5.14)."""
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        payload = cls(
            similarity=data.get("similarity"),
            lambda_threshold=data.get("lambda_threshold"),
            eta=data.get("eta"),
        )
        items = sorted(data.get("layout", []), key=lambda item: int(item["index"]))
        for expected_index, item in enumerate(items):
            if int(item["index"]) != expected_index:
                raise ValueError("layout payload indices must be contiguous")
            value = str(item["value"])
            if not value:
                raise ValueError("layout payload code must not be empty")
            code = BitArray(len(value))
            for index, bit in enumerate(value):
                if bit == "1":
                    code.set(index)
                elif bit != "0":
                    raise ValueError("layout payload code must be binary")
            payload.add_code(
                label=str(item.get("label", "")),
                hue=float(item["hue"]),
                code=code,
            )
        LOGGER.event(
            "layout.payload.load",
            section=LAID_OUT_STRUCTURE,
            data={
                "path": path,
                "points": len(payload._records),
                "code_length": len(payload._records[0].code) if payload._records else 0,
            },
        )
        return payload

    def add_code(self, *, label: int | str, hue: float, code: BitArray) -> LayoutCodeRecord:
        record = LayoutCodeRecord(
            index=len(self._records),
            label=str(label),
            hue=float(hue),
            code=code,
        )
        self._records.append(record)
        return record

    def codes_by_hue(self) -> Dict[float, List[LayoutEncodedCode]]:
        grouped: Dict[float, List[LayoutEncodedCode]] = defaultdict(list)
        for record in self._records:
            grouped[record.hue].append(LayoutEncodedCode(code=record.code, label=record.label))
        return grouped

    def base_payload(self) -> Mapping[str, object]:
        return {
            "width": None,
            "height": None,
            "points": len(self._records),
            "similarity": self._similarity,
            "lambda_threshold": self._lambda_threshold,
            "eta": self._eta,
            "pair_radius": None,
            "layout": [record.as_payload(y=None, x=None) for record in self._records],
        }

    def save_base(self, path: str) -> None:
        if not path:
            raise ValueError("path must be provided for layout payload export")
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = self.base_payload()
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=True, indent=2)
        LOGGER.event(
            "layout.payload.base",
            section=LAID_OUT_STRUCTURE,
            data={
                "path": path,
                "points": len(self._records),
                "similarity": self._similarity,
                "lambda_threshold": self._lambda_threshold,
                "eta": self._eta,
            },
        )
