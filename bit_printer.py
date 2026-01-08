from typing import Iterable


def print_bits(bits: Iterable[int]) -> None:
    print("".join("1" if bit else "0" for bit in bits))
