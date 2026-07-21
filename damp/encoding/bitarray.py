from __future__ import annotations


class BitArray:
    def __init__(self, length: int, fill: int = 0) -> None:
        if length <= 0:
            raise ValueError("bit array length must be positive")
        if fill not in (0, 1):
            raise ValueError("fill must be 0 or 1")
        self._bits = bytearray([fill] * length)
        self._active = set(range(length)) if fill else set()

    @classmethod
    def from_indices(cls, length: int, indices) -> "BitArray":
        result = cls(length)
        active = {int(index) for index in indices}
        for index in active:
            result._bits[index] = 1
        result._active = active
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

    def active_indices(self):
        return iter(self._active)

    def common(self, other: "BitArray") -> int:
        if len(self) != len(other):
            raise ValueError("BitArray sizes must match")
        smaller, larger = (
            (self._active, other._active)
            if len(self._active) <= len(other._active)
            else (other._active, self._active)
        )
        return sum(index in larger for index in smaller)

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
