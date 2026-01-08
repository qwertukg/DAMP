# -*- coding: utf-8 -*-
from __future__ import annotations

import colorsys
import math
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

Code = Sequence[float]
Label = Union[str, int]
SimilarityFn = Callable[[Optional[Code], Optional[Code]], float]


def _coerce_code(code: Code) -> Tuple[float, ...]:
    return tuple(float(v) for v in code)


def _is_zero_code(code: Code) -> bool:
    return all(value == 0.0 for value in code)


def _is_flat_numeric_list(values: Sequence, code_length: Optional[int]) -> bool:
    if code_length is None or not values:
        return False
    first = values[0]
    return isinstance(first, (int, float, bool))


def _chunk_codes(flat: Sequence, code_length: int) -> List[Tuple[float, ...]]:
    if code_length <= 0:
        raise ValueError("code_length must be positive")
    if len(flat) % code_length != 0:
        raise ValueError("flat codes length must be divisible by code_length")
    return [_coerce_code(flat[i : i + code_length]) for i in range(0, len(flat), code_length)]


def _jaccard_similarity(a: Optional[Code], b: Optional[Code]) -> float:
    if a is None or b is None:
        return 0.0
    intersection = 0.0
    union = 0.0
    for av, bv in zip(a, b):
        if av or bv:
            union += 1.0
            if av and bv:
                intersection += 1.0
    return intersection / union if union else 0.0


def _cosine_similarity(a: Optional[Code], b: Optional[Code]) -> float:
    if a is None or b is None:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for av, bv in zip(a, b):
        dot += av * bv
        norm_a += av * av
        norm_b += bv * bv
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


class Layout:
    """Discrete DAMP layout for codes on a 2D grid with long- and short-range phases."""

    def __init__(
        self,
        codes: Union[
            Sequence[Sequence[float]],
            Sequence[float],
            Dict[Label, Sequence[Sequence[float]]],
            Dict[Label, Sequence[float]],
        ],
        *,
        grid_shape: Optional[Tuple[int, int]] = None,
        code_length: Optional[int] = None,
        empty_margin: float = 0.15,
        labels: Optional[Sequence[Label]] = None,
        similarity: Union[str, SimilarityFn] = "jaccard",
        lambda_start: float = 0.5,
        lambda_end: Optional[float] = None,
        eta: float = 12.0,
        rng_seed: Optional[int] = None,
        rr_app_id: str = "damp_layout",
    ) -> None:
        if empty_margin < 0.0:
            raise ValueError("empty_margin must be non-negative")
        self._rng = random.Random(rng_seed)
        self.codes, self.labels, empty_slots = self._normalize_codes(
            codes=codes,
            code_length=code_length,
            labels=labels,
        )
        if not self.codes:
            raise ValueError("codes must be non-empty")
        if grid_shape is None:
            empty_slots += int(round(len(self.codes) * empty_margin))
        self._empty_slots = empty_slots
        self.rows, self.cols = self._choose_grid_shape(grid_shape, len(self.codes) + empty_slots)
        self.grid: List[List[Optional[int]]] = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        self.positions: List[Tuple[int, int]] = [(-1, -1) for _ in range(len(self.codes))]
        self._place_codes()
        self.similarity = self._resolve_similarity(similarity)
        self.lambda_start = float(lambda_start)
        self.lambda_end = float(lambda_end) if lambda_end is not None else float(lambda_start)
        self.eta = float(eta)
        self._rr_app_id = str(rr_app_id)
        self._rr_initialized = False
        self._label_colors = self._build_label_colors(self.labels)

    def _normalize_codes(
        self,
        *,
        codes: Union[
            Sequence[Sequence[float]],
            Sequence[float],
            Dict[Label, Sequence[Sequence[float]]],
            Dict[Label, Sequence[float]],
        ],
        code_length: Optional[int],
        labels: Optional[Sequence[Label]],
    ) -> Tuple[List[Tuple[float, ...]], List[Optional[Label]], int]:
        result_codes: List[Tuple[float, ...]] = []
        result_labels: List[Optional[Label]] = []
        empty_slots = 0

        def add_code(code: Code, label: Optional[Label]) -> None:
            nonlocal empty_slots
            coerced = _coerce_code(code)
            if _is_zero_code(coerced):
                empty_slots += 1
                return
            result_codes.append(coerced)
            result_labels.append(label)

        if isinstance(codes, dict):
            for label, payload in codes.items():
                if isinstance(payload, Sequence) and _is_flat_numeric_list(payload, code_length):
                    expanded = _chunk_codes(payload, int(code_length))
                else:
                    expanded = payload  # type: ignore[assignment]
                for code in expanded:  # type: ignore[assignment]
                    add_code(code, label)
        else:
            if isinstance(codes, Sequence) and _is_flat_numeric_list(codes, code_length):
                expanded = _chunk_codes(codes, int(code_length))
            else:
                expanded = list(codes)  # type: ignore[arg-type]
            if labels is not None:
                if len(labels) != len(expanded):
                    raise ValueError("labels length must match number of codes")
                for code, label in zip(expanded, labels):
                    add_code(code, label)
            else:
                for code in expanded:
                    add_code(code, None)

        return result_codes, result_labels, empty_slots

    def _choose_grid_shape(self, grid_shape: Optional[Tuple[int, int]], count: int) -> Tuple[int, int]:
        if grid_shape is not None:
            rows, cols = grid_shape
            if rows <= 0 or cols <= 0:
                raise ValueError("grid_shape must contain positive values")
            if rows * cols < count:
                raise ValueError("grid_shape is too small for the number of codes")
            return int(rows), int(cols)
        side = int(math.ceil(math.sqrt(count)))
        return side, side

    def _place_codes(self) -> None:
        coords = [(y, x) for y in range(self.rows) for x in range(self.cols)]
        self._rng.shuffle(coords)
        for idx, (y, x) in enumerate(coords[: len(self.codes)]):
            self.grid[y][x] = idx
            self.positions[idx] = (y, x)

    def _resolve_similarity(self, similarity: Union[str, SimilarityFn]) -> SimilarityFn:
        if callable(similarity):
            return similarity
        name = str(similarity).lower()
        if name == "jaccard":
            return _jaccard_similarity
        if name == "cosine":
            return _cosine_similarity
        raise ValueError(f"unknown similarity: {similarity}")

    def _build_label_colors(self, labels: Sequence[Optional[Label]]) -> Dict[Optional[Label], Tuple[int, int, int]]:
        colors: Dict[Optional[Label], Tuple[int, int, int]] = {}
        label_values: Dict[Label, float] = {}
        for label in {label for label in labels if label is not None}:
            try:
                label_values[label] = float(label)
            except (TypeError, ValueError):
                continue

        if label_values:
            min_value = min(label_values.values())
            max_value = max(label_values.values())
            span = max(max_value - min_value, 1.0)
            for label, value in sorted(label_values.items(), key=lambda item: item[1]):
                hue = (value - min_value) / span
                r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.9)
                colors[label] = (int(r * 255), int(g * 255), int(b * 255))

        if None not in colors:
            colors[None] = (200, 200, 200)
        return colors

    def _pair_offsets(self, radius: int) -> List[Tuple[int, int]]:
        offsets: List[Tuple[int, int]] = []
        r_sq = radius * radius
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy == 0 and dx == 0:
                    continue
                if dy * dy + dx * dx <= r_sq:
                    offsets.append((dy, dx))
        return offsets

    def _energy_offsets(self, radius: int) -> List[Tuple[int, int, float]]:
        offsets: List[Tuple[int, int, float]] = []
        r_sq = radius * radius
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy == 0 and dx == 0:
                    continue
                dist_sq = dy * dy + dx * dx
                if dist_sq <= r_sq:
                    offsets.append((dy, dx, 1.0 / math.sqrt(dist_sq)))
        return offsets

    def _pair_energy_long(
        self,
        y1: int,
        x1: int,
        y2: int,
        x2: int,
        lam: float,
    ) -> Tuple[float, float]:
        idx1 = self.grid[y1][x1]
        idx2 = self.grid[y2][x2]
        code1 = self.codes[idx1] if idx1 is not None else None
        code2 = self.codes[idx2] if idx2 is not None else None
        similarity = self.similarity
        positions = self.positions
        codes = self.codes
        eta = self.eta
        exp = math.exp
        phi_c = 0.0
        phi_s = 0.0
        for idx in range(len(codes)):
            if idx == idx1 or idx == idx2:
                continue
            base1 = 0.0
            base2 = 0.0
            if code1 is not None:
                base1 = similarity(codes[idx], code1)
            if code2 is not None:
                base2 = similarity(codes[idx], code2)
            if base1 == 0.0 and base2 == 0.0:
                continue
            y, x = positions[idx]
            d1 = math.sqrt((y1 - y) * (y1 - y) + (x1 - x) * (x1 - x))
            d2 = math.sqrt((y2 - y) * (y2 - y) + (x2 - x) * (x2 - x))
            if base1 != 0.0:
                s1 = base1 / (1.0 + exp(-eta * (base1 - lam)))
            else:
                s1 = 0.0
            if base2 != 0.0:
                s2 = base2 / (1.0 + exp(-eta * (base2 - lam)))
            else:
                s2 = 0.0
            phi_c += s1 * d1 + s2 * d2
            phi_s += s2 * d1 + s1 * d2
        return phi_c, phi_s

    def _pair_energy_short(
        self,
        y1: int,
        x1: int,
        y2: int,
        x2: int,
        lam: float,
        radius: float,
    ) -> Tuple[float, float]:
        idx1 = self.grid[y1][x1]
        idx2 = self.grid[y2][x2]
        code1 = self.codes[idx1] if idx1 is not None else None
        code2 = self.codes[idx2] if idx2 is not None else None
        cy = (y1 + y2) / 2.0
        cx = (x1 + x2) / 2.0
        r_sq = radius * radius
        y_min = max(int(math.floor(cy - radius)), 0)
        y_max = min(int(math.ceil(cy + radius)), self.rows - 1)
        x_min = max(int(math.floor(cx - radius)), 0)
        x_max = min(int(math.ceil(cx + radius)), self.cols - 1)
        similarity = self.similarity
        codes = self.codes
        eta = self.eta
        exp = math.exp
        phi_c = 0.0
        phi_s = 0.0
        for y in range(y_min, y_max + 1):
            dy1 = y1 - y
            dy2 = y2 - y
            dyc = y - cy
            for x in range(x_min, x_max + 1):
                if (y == y1 and x == x1) or (y == y2 and x == x2):
                    continue
                dcy = dyc
                dcx = x - cx
                if dcy * dcy + dcx * dcx > r_sq:
                    continue
                idx = self.grid[y][x]
                if idx is None:
                    continue
                base1 = 0.0
                base2 = 0.0
                if code1 is not None:
                    base1 = similarity(codes[idx], code1)
                if code2 is not None:
                    base2 = similarity(codes[idx], code2)
                if base1 == 0.0 and base2 == 0.0:
                    continue
                if base1 != 0.0:
                    s1 = base1 / (1.0 + exp(-eta * (base1 - lam)))
                else:
                    s1 = 0.0
                if base2 != 0.0:
                    s2 = base2 / (1.0 + exp(-eta * (base2 - lam)))
                else:
                    s2 = 0.0
                d1 = dy1 * dy1 + (x1 - x) * (x1 - x)
                d2 = dy2 * dy2 + (x2 - x) * (x2 - x)
                inv_d1 = 1.0 / math.sqrt(d1)
                inv_d2 = 1.0 / math.sqrt(d2)
                phi_c += s1 * inv_d1 + s2 * inv_d2
                phi_s += s2 * inv_d1 + s1 * inv_d2
        return phi_c, phi_s

    def _swap(self, y1: int, x1: int, y2: int, x2: int) -> None:
        idx1 = self.grid[y1][x1]
        idx2 = self.grid[y2][x2]
        self.grid[y1][x1], self.grid[y2][x2] = idx2, idx1
        if idx1 is not None:
            self.positions[idx1] = (y2, x2)
        if idx2 is not None:
            self.positions[idx2] = (y1, x1)

    def _select_pair(
        self,
        *,
        radius: Optional[int],
        max_attempts: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        for _ in range(max_attempts):
            first_idx = self._rng.randrange(len(self.codes))
            y1, x1 = self.positions[first_idx]
            if radius is None or radius >= max(self.rows, self.cols):
                y2 = self._rng.randrange(self.rows)
                x2 = self._rng.randrange(self.cols)
                if y1 == y2 and x1 == x2:
                    continue
            else:
                offsets = self._pair_offsets(radius)
                if not offsets:
                    continue
                found = False
                for _ in range(max_attempts):
                    dy, dx = offsets[self._rng.randrange(len(offsets))]
                    y2 = y1 + dy
                    x2 = x1 + dx
                    if 0 <= y2 < self.rows and 0 <= x2 < self.cols:
                        found = True
                        break
                if not found:
                    continue
            idx1 = self.grid[y1][x1]
            idx2 = self.grid[y2][x2]
            if idx1 is None and idx2 is None:
                continue
            return y1, x1, y2, x2
        return None

    def _point_energy(self, y: int, x: int, radius: int, lam: float) -> float:
        idx = self.grid[y][x]
        if idx is None:
            return 0.0
        code = self.codes[idx]
        similarity = self.similarity
        codes = self.codes
        eta = self.eta
        exp = math.exp
        offsets = self._energy_offsets(radius)
        total = 0.0
        for dy, dx, inv_dist in offsets:
            ny = y + dy
            nx = x + dx
            if ny < 0 or ny >= self.rows or nx < 0 or nx >= self.cols:
                continue
            n_idx = self.grid[ny][nx]
            if n_idx is None:
                continue
            base = 0.0
            base = similarity(code, codes[n_idx])
            if base == 0.0:
                continue
            total += (base / (1.0 + exp(-eta * (base - lam)))) * inv_dist
        return total

    def energy_matrix(self, radius: int, lam: float) -> List[List[float]]:
        energies = [[0.0 for _ in range(self.cols)] for _ in range(self.rows)]
        max_energy = 0.0
        for y, x in self.positions:
            value = self._point_energy(y, x, radius, lam)
            energies[y][x] = value
            if value > max_energy:
                max_energy = value
        if max_energy > 0.0:
            for y in range(self.rows):
                for x in range(self.cols):
                    energies[y][x] = energies[y][x] / max_energy
        return energies

    def _init_rerun(self, spawn: bool) -> None:
        import rerun as rr

        rr.init(self._rr_app_id, spawn=spawn)
        self._rr_initialized = True

    def _log_rerun(self, step: int, lam: float, *, energy_radius: Optional[int]) -> None:
        import rerun as rr

        if not self._rr_initialized:
            self._init_rerun(spawn=True)

        rr.set_time("step", sequence=step)
        points: List[Tuple[float, float]] = []
        colors: List[Tuple[int, int, int]] = []
        labels: List[str] = []
        for (y, x), label, code in zip(self.positions, self.labels, self.codes):
            if _is_zero_code(code):
                continue
            points.append((float(x), float(y)))
            colors.append(self._label_colors[label])
            labels.append("" if label is None else str(label))
        rr.log("layout/points", rr.Points2D(points, colors=colors, labels=labels, radii=0.5))
        if energy_radius is not None and energy_radius > 0:
            energy = self.energy_matrix(energy_radius, lam)
            rr.log("layout/energy", rr.Image(energy))

    def _step_long_range(
        self,
        *,
        pairs: int,
        pair_radius: Optional[int],
        lam: float,
    ) -> int:
        swaps = 0
        for _ in range(pairs):
            selected = self._select_pair(
                radius=pair_radius,
                max_attempts=64,
            )
            if selected is None:
                continue
            y1, x1, y2, x2 = selected
            phi_c, phi_s = self._pair_energy_long(y1, x1, y2, x2, lam)
            if phi_s < phi_c:
                self._swap(y1, x1, y2, x2)
                swaps += 1
        return swaps

    def _step_short_range(
        self,
        *,
        pairs: int,
        pair_radius: Optional[int],
        lam: float,
        local_radius: int,
    ) -> int:
        swaps = 0
        for _ in range(pairs):
            selected = self._select_pair(
                radius=pair_radius,
                max_attempts=64,
            )
            if selected is None:
                continue
            y1, x1, y2, x2 = selected
            pair_dist = math.sqrt((y1 - y2) * (y1 - y2) + (x1 - x2) * (x1 - x2))
            radius = max(local_radius, int(math.ceil(pair_dist)))
            phi_c, phi_s = self._pair_energy_short(y1, x1, y2, x2, lam, radius)
            if phi_s > phi_c:
                self._swap(y1, x1, y2, x2)
                swaps += 1
        return swaps

    def layout(
        self,
        *,
        long_steps: int = 200,
        short_steps: int = 100,
        pairs_per_step: int = 64,
        long_pair_radius: Optional[int] = None,
        short_pair_radius: Optional[int] = None,
        short_local_radius: int = 6,
        min_swaps: Optional[int] = None,
        visualize: bool = True,
        visualize_every: int = 1,
        energy_radius: Optional[int] = 5,
    ) -> None:
        """Run long-range then short-range swaps; set visualize=True to stream to rerun."""
        total_steps = max(long_steps, 0) + max(short_steps, 0)
        if total_steps <= 0:
            return

        step_index = 0
        for phase, steps in (("long", long_steps), ("short", short_steps)):
            for phase_step in range(steps):
                if total_steps > 1:
                    t = step_index / float(total_steps - 1)
                else:
                    t = 1.0
                lam = self.lambda_start + (self.lambda_end - self.lambda_start) * t
                if phase == "long":
                    pair_radius = long_pair_radius
                    swaps = self._step_long_range(
                        pairs=pairs_per_step,
                        pair_radius=pair_radius,
                        lam=lam,
                    )
                else:
                    pair_radius = short_pair_radius
                    swaps = self._step_short_range(
                        pairs=pairs_per_step,
                        pair_radius=pair_radius,
                        lam=lam,
                        local_radius=short_local_radius,
                    )
                if visualize and (step_index % max(visualize_every, 1) == 0):
                    self._log_rerun(step_index, lam, energy_radius=energy_radius)
                step_index += 1
                if min_swaps is not None and swaps < min_swaps:
                    break

    def get_layout(self) -> List[List[Optional[Code]]]:
        output: List[List[Optional[Code]]] = []
        for y in range(self.rows):
            row: List[Optional[Code]] = []
            for x in range(self.cols):
                idx = self.grid[y][x]
                row.append(self.codes[idx] if idx is not None else None)
            output.append(row)
        return output
