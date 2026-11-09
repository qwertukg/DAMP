from __future__ import annotations

from typing import Iterable


class BitArray:
    """Разреженный битовый вектор фиксированной длины (§2)."""

    def __init__(self, length: int, fill: int = 0) -> None:
        if length <= 0:
            raise ValueError("bit array length must be positive")
        if fill not in (0, 1):
            raise ValueError("fill must be 0 or 1")
        self._bits = bytearray([fill] * length)
        self._active: set[int] = set(range(length)) if fill else set()

    @classmethod
    def from_indices(cls, length: int, indices: Iterable[int]) -> "BitArray":
        result = cls(length)
        for index in indices:
            result.set(int(index), 1)
        return result

    @classmethod
    def from_bits(cls, bits: bytes | bytearray, active: Iterable[int]) -> "BitArray":
        """Вектор из готового буфера 0/1 и списка его единичных индексов (без побитовой установки).

        Буфер и индексы должны согласовываться: это быстрый путь для кодера, который уже посчитал
        биты векторно; проверка не делается ради скорости.
        """
        if len(bits) <= 0:
            raise ValueError("bit array length must be positive")
        result = cls.__new__(cls)
        result._bits = bytearray(bits)
        result._active = {int(index) for index in active}
        return result

    def set(self, index: int, value: int = 1) -> None:
        resolved = 1 if value else 0
        self._bits[index] = resolved
        if resolved:
            self._active.add(index)
        else:
            self._active.discard(index)

    def count(self) -> int:
        return len(self._active)

    def active_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self._active))

    def common(self, other: "BitArray") -> int:
        if len(self) != len(other):
            raise ValueError("BitArray sizes must match")
        smaller, larger = (
            (self._active, other._active)
            if len(self._active) <= len(other._active)
            else (other._active, self._active)
        )
        return sum(1 for index in smaller if index in larger)

    def to01(self) -> str:
        return "".join("1" if bit else "0" for bit in self._bits)

    def __len__(self) -> int:
        return len(self._bits)

    def __getitem__(self, index: int) -> int:
        return self._bits[index]

    def __iter__(self):
        return iter(self._bits)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BitArray):
            return NotImplemented
        return self._bits == other._bits

    def __hash__(self) -> int:
        return hash(bytes(self._bits))

    def __repr__(self) -> str:
        return f"BitArray(len={len(self._bits)}, ones={self.count()})"
