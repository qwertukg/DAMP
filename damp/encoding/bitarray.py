from __future__ import annotations


class BitArray:
    def __init__(self, length: int, fill: int = 0) -> None:
        if length <= 0:
            raise ValueError("bit array length must be positive")
        if fill not in (0, 1):
            raise ValueError("fill must be 0 or 1")
        self._bits = bytearray([fill] * length)

    def set(self, index: int, value: int = 1) -> None:
        self._bits[index] = 1 if value else 0

    def count(self) -> int:
        return sum(self._bits)

    def common(self, other: "BitArray") -> int:
        if len(self) != len(other):
            raise ValueError("BitArray sizes must match")
        return sum(a & b for a, b in zip(self._bits, other._bits))

    def to01(self) -> str:
        return "".join("1" if bit else "0" for bit in self._bits)

    def __len__(self) -> int:
        return len(self._bits)

    def __getitem__(self, index: int) -> int:
        return self._bits[index]

    def __iter__(self):
        return iter(self._bits)

    def __repr__(self) -> str:
        return f"BitArray(len={len(self._bits)}, ones={self.count()})"
