from __future__ import annotations

from dataclasses import dataclass
import atexit
import json
import math
import random
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

from damp.encoding.MnistSobelAngleMap import MnistSobelAngleMap
from damp.encoding.damp_encoder import Encoder
from damp.layout.damp_layout import Layout, BitArray as LayoutBitArray

LOG_ENABLED = True
LOG_EVERY = 50
LOG_DECODE_DETAILS = False


@dataclass(frozen=True)
class CodeVector:
    bits: int
    ones: int
    length: int


@dataclass
class _RunningStats:
    count: int = 0
    total: float = 0.0
    min: float | None = None
    max: float | None = None

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        if self.min is None or value < self.min:
            self.min = value
        if self.max is None or value > self.max:
            self.max = value

    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0


@dataclass(frozen=True)
class ActivationPoint:
    y: int
    x: int
    activation: float
    energy: float


@dataclass
class Detector:
    center_y: float
    center_x: float
    radius: float
    lambda_d: float
    n: int
    energy: float
    bit_index: int
    layer_index: int


@dataclass
class DetectorLayer:
    lambda_d: float
    detectors: list[Detector]


@dataclass
class DetectorHierarchy:
    layers: list[DetectorLayer]
    code_length: int


@dataclass
class CodeSpace:
    grid: list[list[CodeVector | None]]
    height: int
    width: int
    code_length: int
    energy: list[list[float]] | None = None
    energy_radius: int | None = None
    energy_lambda: float | None = None
    similarity: str = "cosine"
    eta: float | None = None


@dataclass(frozen=True)
class MemoryEntry:
    code: CodeVector
    label: int


@dataclass
class DetectorBuildParams:
    lambda_levels: Sequence[float]
    activation_radius: int
    energy_radius: int
    detector_code_length: int
    cluster_eps: float = 2.5
    cluster_min_points: int = 3
    energy_threshold_mu: float = 0.0
    energy_lambda: float | None = None
    max_attempts: int = 2000
    max_detectors_per_layer: int | None = None
    min_radius: float = 1.0
    patience: int = 200
    similarity: str = "cosine"
    eta: float | None = None
    seed: int = 0


@dataclass
class EmbedParams:
    lambda_activation: float
    mu_e: float
    mu_d: float
    sigma: int
    similarity: str = "cosine"
    eta: float | None = None
    merge_order: str = "high"


@dataclass
class LayoutConfig:
    layout_kwargs: Mapping[str, object]
    run_schedule: Sequence[Mapping[str, object]] = ()


@dataclass
class HierarchyConfig:
    encoder: Encoder
    extractor: MnistSobelAngleMap
    v0: CodeSpace
    build_l1: DetectorBuildParams
    build_l2: DetectorBuildParams
    build_l3: DetectorBuildParams
    embed_l1: EmbedParams
    embed_l2: EmbedParams
    embed_l3: EmbedParams
    layout_l2: LayoutConfig
    layout_l3: LayoutConfig


@dataclass
class HierarchyModel:
    v0: CodeSpace
    v1: CodeSpace
    v2: CodeSpace
    d1: DetectorHierarchy
    d2: DetectorHierarchy
    d3: DetectorHierarchy
    memory: list[MemoryEntry]
    encoder: Encoder
    extractor: MnistSobelAngleMap
    embed_l1: EmbedParams
    embed_l2: EmbedParams
    embed_l3: EmbedParams


def _log(group: str, message: str) -> None:
    if LOG_ENABLED:
        print(f"[{group}] {message}")


_DECODE_STATS: dict[str, object] | None = None
_DECODE_REGISTERED = False


def _get_decode_stats() -> dict[str, object]:
    global _DECODE_STATS, _DECODE_REGISTERED
    if _DECODE_STATS is None:
        _DECODE_STATS = {
            "total": 0,
            "correct": 0,
            "topk_correct": 0,
            "top_k": None,
            "none_pred": 0,
            "top1_stats": _RunningStats(),
            "label_counts": defaultdict(int),
            "pred_counts": defaultdict(int),
            "confusion": defaultdict(lambda: defaultdict(int)),
        }
    if not _DECODE_REGISTERED:
        atexit.register(_log_decode_summary)
        _DECODE_REGISTERED = True
    return _DECODE_STATS


def _log_decode_summary() -> None:
    stats = _DECODE_STATS
    if not stats:
        return
    total = int(stats["total"])
    if total <= 0:
        return
    correct = int(stats["correct"])
    topk_correct = int(stats["topk_correct"])
    none_pred = int(stats["none_pred"])
    top_k = stats["top_k"]
    accuracy = correct / total if total else 0.0
    topk_accuracy = None
    if isinstance(top_k, int) and top_k > 0:
        topk_accuracy = topk_correct / total if total else 0.0

    label_counts = stats["label_counts"]
    pred_counts = stats["pred_counts"]
    confusion = stats["confusion"]
    labels = sorted(label_counts.keys())
    per_label_accuracy: dict[int, float] = {}
    confusion_matrix: list[list[int]] = []
    for label in labels:
        row = []
        correct_label = int(confusion[label].get(label, 0))
        total_label = int(label_counts[label])
        per_label_accuracy[label] = (correct_label / total_label) if total_label else 0.0
        for pred_label in labels:
            row.append(int(confusion[label].get(pred_label, 0)))
        confusion_matrix.append(row)

    top1_stats = stats["top1_stats"]
    topk_label = "varied"
    if isinstance(top_k, int) and top_k > 0:
        topk_label = str(top_k)
    topk_part = f"topk={topk_label}"
    if topk_accuracy is not None:
        topk_part = f"{topk_part} topk_accuracy={topk_accuracy:.4f}"
    _log(
        "decode",
        f"summary total={total} accuracy={accuracy:.4f} "
        f"none_pred={none_pred} {topk_part}",
    )
    if isinstance(top1_stats, _RunningStats) and top1_stats.count:
        _log(
            "decode",
            "top1_score "
            f"avg={top1_stats.mean():.4f} "
            f"min={0.0 if top1_stats.min is None else top1_stats.min:.4f} "
            f"max={0.0 if top1_stats.max is None else top1_stats.max:.4f}",
        )
    _log("decode", f"labels={json.dumps(labels, ensure_ascii=True)}")
    _log(
        "decode",
        f"per_label_accuracy={json.dumps(per_label_accuracy, ensure_ascii=True)}",
    )
    _log(
        "decode",
        f"pred_counts={json.dumps(dict(pred_counts), ensure_ascii=True)}",
    )
    _log(
        "decode",
        f"confusion={json.dumps(confusion_matrix, ensure_ascii=True)}",
    )


def _detector_count(hierarchy: DetectorHierarchy) -> int:
    return sum(len(layer.detectors) for layer in hierarchy.layers)


def _code_from_any(code: object, expected_length: int | None = None) -> CodeVector:
    if isinstance(code, CodeVector):
        if expected_length is not None and code.length != expected_length:
            raise ValueError("code length mismatch")
        return code
    if isinstance(code, int):
        if expected_length is None:
            raise ValueError("expected_length is required for int codes")
        return CodeVector(code, code.bit_count(), expected_length)

    data = None
    if hasattr(code, "_bits"):
        data = getattr(code, "_bits")
    elif hasattr(code, "_data"):
        data = getattr(code, "_data")
    elif hasattr(code, "to01"):
        data = code.to01()
    else:
        data = code

    length = len(data)
    if expected_length is not None and length != expected_length:
        raise ValueError("code length mismatch")

    bits = 0
    ones = 0
    if isinstance(data, str):
        for idx, ch in enumerate(data):
            if ch == "1":
                bits |= 1 << idx
                ones += 1
    else:
        for idx, bit in enumerate(data):
            if bit:
                bits |= 1 << idx
                ones += 1
    return CodeVector(bits, ones, length)


def _similarity(a: CodeVector, b: CodeVector, mode: str) -> float:
    if a.ones == 0 or b.ones == 0:
        return 0.0
    common = (a.bits & b.bits).bit_count()
    if common == 0:
        return 0.0
    if mode == "cosine":
        denom = math.sqrt(a.ones * b.ones)
        return 0.0 if denom == 0 else common / denom
    union = a.ones + b.ones - common
    return 0.0 if union == 0 else common / union


def _sim_lambda(
    a: CodeVector,
    b: CodeVector,
    lambda_threshold: float,
    similarity: str,
    eta: float | None,
) -> float:
    sim = _similarity(a, b, similarity)
    if sim <= 0.0:
        return 0.0
    if eta is None:
        return sim if sim >= lambda_threshold else 0.0
    return sim * (1.0 / (1.0 + math.exp(-eta * (sim - lambda_threshold))))


def _activation_map(
    stimuli: Sequence[CodeVector],
    space: CodeSpace,
    lambda_threshold: float,
    similarity: str,
    eta: float | None,
) -> list[list[float]]:
    activation = [[0.0 for _ in range(space.width)] for _ in range(space.height)]
    if not stimuli:
        return activation
    for y in range(space.height):
        row = space.grid[y]
        out_row = activation[y]
        for x in range(space.width):
            code = row[x]
            if code is None:
                continue
            best = 0.0
            for stim in stimuli:
                sim = _sim_lambda(stim, code, lambda_threshold, similarity, eta)
                if sim > best:
                    best = sim
            out_row[x] = best
    return activation


def _local_activation(
    center_y: int,
    center_x: int,
    center_code: CodeVector,
    space: CodeSpace,
    radius: float,
    lambda_threshold: float,
    similarity: str,
    eta: float | None,
) -> list[ActivationPoint]:
    if space.energy is None:
        raise ValueError("space energy map is not available")
    points: list[ActivationPoint] = []
    r_int = int(math.ceil(radius))
    y_min = max(0, center_y - r_int)
    y_max = min(space.height - 1, center_y + r_int)
    x_min = max(0, center_x - r_int)
    x_max = min(space.width - 1, center_x + r_int)
    radius_sq = radius * radius
    for y in range(y_min, y_max + 1):
        row = space.grid[y]
        for x in range(x_min, x_max + 1):
            if (y - center_y) ** 2 + (x - center_x) ** 2 > radius_sq:
                continue
            code = row[x]
            if code is None:
                continue
            sim = _sim_lambda(center_code, code, lambda_threshold, similarity, eta)
            if sim <= 0.0:
                continue
            points.append(ActivationPoint(y=y, x=x, activation=sim, energy=space.energy[y][x]))
    return points


def _dbscan(points: Sequence[ActivationPoint], eps: float, min_points: int) -> list[list[ActivationPoint]]:
    if not points:
        return []
    eps_sq = eps * eps
    visited = [False] * len(points)
    assigned = [-1] * len(points)
    clusters: list[list[ActivationPoint]] = []

    def region_query(index: int) -> list[int]:
        target = points[index]
        neighbors: list[int] = []
        for j, point in enumerate(points):
            dy = point.y - target.y
            dx = point.x - target.x
            if dy * dy + dx * dx <= eps_sq:
                neighbors.append(j)
        return neighbors

    for i in range(len(points)):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = region_query(i)
        if len(neighbors) < min_points:
            continue
        cluster_index = len(clusters)
        clusters.append([])
        seeds = neighbors[:]
        while seeds:
            j = seeds.pop()
            if not visited[j]:
                visited[j] = True
                j_neighbors = region_query(j)
                if len(j_neighbors) >= min_points:
                    for n in j_neighbors:
                        if n not in seeds:
                            seeds.append(n)
            if assigned[j] == -1:
                assigned[j] = cluster_index
                clusters[cluster_index].append(points[j])
        if assigned[i] == -1:
            assigned[i] = cluster_index
            clusters[cluster_index].append(points[i])
    return clusters


def _weighted_centroid(points: Sequence[ActivationPoint]) -> tuple[float, float] | None:
    weight_sum = 0.0
    sum_y = 0.0
    sum_x = 0.0
    for point in points:
        weight = point.activation * point.energy
        weight_sum += weight
        sum_y += point.y * weight
        sum_x += point.x * weight
    if weight_sum <= 0.0:
        return None
    return sum_y / weight_sum, sum_x / weight_sum


def _optimal_radius(
    points: Sequence[ActivationPoint],
    center_y: float,
    center_x: float,
    min_radius: float,
) -> float:
    distances = []
    for point in points:
        dist = math.hypot(point.y - center_y, point.x - center_x)
        distances.append(dist)
    distances.sort()
    best_f = -1.0
    best_r = min_radius
    count = 0
    for dist in distances:
        if dist <= 0.0:
            continue
        count += 1
        f = count / (math.pi * dist * dist)
        if f > best_f:
            best_f = f
            best_r = dist
    return max(best_r, min_radius)


def _nearest_point(points: Sequence[ActivationPoint], center_y: float, center_x: float) -> ActivationPoint:
    best = points[0]
    best_dist = math.hypot(best.y - center_y, best.x - center_x)
    for point in points[1:]:
        dist = math.hypot(point.y - center_y, point.x - center_x)
        if dist < best_dist:
            best = point
            best_dist = dist
    return best


def _build_energy_map(
    space: CodeSpace,
    radius: int,
    lambda_threshold: float,
    similarity: str,
    eta: float | None,
) -> list[list[float]]:
    energies = [[0.0 for _ in range(space.width)] for _ in range(space.height)]
    if radius <= 0:
        return energies
    radius_sq = radius * radius
    emax = 0.0
    for y in range(space.height):
        for x in range(space.width):
            center = space.grid[y][x]
            if center is None:
                continue
            energy = 0.0
            y_min = max(0, y - radius)
            y_max = min(space.height - 1, y + radius)
            x_min = max(0, x - radius)
            x_max = min(space.width - 1, x + radius)
            for yy in range(y_min, y_max + 1):
                row = space.grid[yy]
                for xx in range(x_min, x_max + 1):
                    if yy == y and xx == x:
                        continue
                    dy = yy - y
                    dx = xx - x
                    dist_sq = dy * dy + dx * dx
                    if dist_sq > radius_sq:
                        continue
                    neighbor = row[xx]
                    if neighbor is None:
                        continue
                    sim = _sim_lambda(center, neighbor, lambda_threshold, similarity, eta)
                    if sim <= 0.0:
                        continue
                    energy += sim / math.sqrt(dist_sq)
            energies[y][x] = energy
            if energy > emax:
                emax = energy
    if emax <= 0.0:
        _log("detectors", "energy_map max=0.0 (empty or weak similarities)")
        return energies
    for y in range(space.height):
        for x in range(space.width):
            if energies[y][x] > 0.0:
                energies[y][x] /= emax
    _log(
        "detectors",
        f"energy_map built radius={radius} lambda={lambda_threshold:.3f} max={emax:.4f}",
    )
    return energies


def _detector_from_cluster(
    cluster: Sequence[ActivationPoint],
    space: CodeSpace,
    lambda_d: float,
    layer_index: int,
    params: DetectorBuildParams,
    rng: random.Random,
) -> Detector | None:
    centroid = _weighted_centroid(cluster)
    if centroid is None:
        return None
    center_y, center_x = centroid
    radius = _optimal_radius(cluster, center_y, center_x, params.min_radius)
    nearest = _nearest_point(cluster, center_y, center_x)
    center_code = space.grid[nearest.y][nearest.x]
    if center_code is None:
        return None

    if space.energy is None:
        raise ValueError("space energy map is not available")
    n_d = 0
    e_d = 0.0
    r_int = int(math.ceil(radius))
    y_min = max(0, nearest.y - r_int)
    y_max = min(space.height - 1, nearest.y + r_int)
    x_min = max(0, nearest.x - r_int)
    x_max = min(space.width - 1, nearest.x + r_int)
    radius_sq = radius * radius
    for y in range(y_min, y_max + 1):
        row = space.grid[y]
        for x in range(x_min, x_max + 1):
            if (y - center_y) ** 2 + (x - center_x) ** 2 > radius_sq:
                continue
            code = row[x]
            if code is None:
                continue
            energy = space.energy[y][x]
            if energy < params.energy_threshold_mu:
                continue
            sim = _sim_lambda(center_code, code, lambda_d, params.similarity, params.eta)
            if sim <= 0.0:
                continue
            n_d += 1
            e_d += sim * energy
    if n_d == 0 or e_d <= 0.0:
        return None
    bit_index = rng.randrange(params.detector_code_length)
    return Detector(
        center_y=center_y,
        center_x=center_x,
        radius=radius,
        lambda_d=lambda_d,
        n=n_d,
        energy=e_d,
        bit_index=bit_index,
        layer_index=layer_index,
    )


def _fill_factor(detector: Detector) -> float:
    radius = max(detector.radius, 1e-6)
    return detector.n / radius


def _insert_detector(detector: Detector, layer: list[Detector]) -> bool:
    overlaps: list[Detector] = []
    for existing in layer:
        dist = math.hypot(detector.center_y - existing.center_y, detector.center_x - existing.center_x)
        if dist <= max(detector.radius, existing.radius):
            overlaps.append(existing)
    if not overlaps:
        layer.append(detector)
        return True
    candidate_fill = _fill_factor(detector)
    for existing in overlaps:
        if candidate_fill <= _fill_factor(existing):
            return False
    for existing in overlaps:
        layer.remove(existing)
    layer.append(detector)
    return True


def build_detectors(space: CodeSpace, params: DetectorBuildParams) -> DetectorHierarchy:
    if not params.lambda_levels:
        raise ValueError("lambda_levels must be non-empty")
    energy_lambda = params.energy_lambda
    if energy_lambda is None:
        energy_lambda = min(params.lambda_levels)

    recompute_energy = (
        space.energy is None
        or space.energy_radius != params.energy_radius
        or space.energy_lambda != energy_lambda
        or space.similarity != params.similarity
        or space.eta != params.eta
    )
    _log(
        "detectors",
        "params "
        f"lambda_levels={list(params.lambda_levels)} "
        f"activation_radius={params.activation_radius} "
        f"energy_radius={params.energy_radius} "
        f"code_len={params.detector_code_length} "
        f"cluster_eps={params.cluster_eps} "
        f"cluster_min_points={params.cluster_min_points} "
        f"energy_threshold_mu={params.energy_threshold_mu} "
        f"energy_lambda={energy_lambda} "
        f"max_attempts={params.max_attempts} "
        f"max_detectors_per_layer={params.max_detectors_per_layer} "
        f"min_radius={params.min_radius} "
        f"patience={params.patience} "
        f"similarity={params.similarity} "
        f"eta={params.eta} "
        f"seed={params.seed}"
    )
    _log(
        "detectors",
        "space "
        f"{space.height}x{space.width} code_len={space.code_length} "
        f"layers={len(params.lambda_levels)}"
    )
    if recompute_energy:
        space.energy = _build_energy_map(
            space,
            radius=params.energy_radius,
            lambda_threshold=energy_lambda,
            similarity=params.similarity,
            eta=params.eta,
        )
        space.energy_radius = params.energy_radius
        space.energy_lambda = energy_lambda
        space.similarity = params.similarity
        space.eta = params.eta
    else:
        _log(
            "detectors",
            "reuse energy map "
            f"radius={space.energy_radius} lambda={space.energy_lambda}"
        )

    coords = [
        (y, x)
        for y in range(space.height)
        for x in range(space.width)
        if space.grid[y][x] is not None
    ]
    if not coords:
        raise ValueError("space contains no points")
    space_total = space.height * space.width
    density = (len(coords) / space_total) if space_total else 0.0
    _log(
        "detectors",
        f"space density={density:.3f} filled={len(coords)}/{space_total}",
    )
    if space.energy is not None:
        energy_min = None
        energy_max = None
        energy_sum = 0.0
        energy_count = 0
        above_mu = 0
        for y, x in coords:
            energy = float(space.energy[y][x])
            if energy_min is None or energy < energy_min:
                energy_min = energy
            if energy_max is None or energy > energy_max:
                energy_max = energy
            energy_sum += energy
            energy_count += 1
            if energy >= params.energy_threshold_mu:
                above_mu += 1
        energy_avg = (energy_sum / energy_count) if energy_count else 0.0
        above_mu_frac = (above_mu / energy_count) if energy_count else 0.0
        _log(
            "detectors",
            "energy_map stats "
            f"min={0.0 if energy_min is None else energy_min:.4f} "
            f"max={0.0 if energy_max is None else energy_max:.4f} "
            f"avg={energy_avg:.4f} "
            f"above_mu={above_mu}/{energy_count} "
            f"above_mu_frac={above_mu_frac:.3f}",
        )

    rng = random.Random(params.seed)
    layers: list[DetectorLayer] = []
    for layer_index, lambda_d in enumerate(params.lambda_levels):
        _log(
            "detectors",
            "layer "
            f"{layer_index + 1}/{len(params.lambda_levels)} lambda={lambda_d:.3f}"
        )
        detectors: list[Detector] = []
        attempts = 0
        no_insert = 0
        progress_every = max(50, params.max_attempts // 10)
        while attempts < params.max_attempts:
            if params.max_detectors_per_layer is not None and len(detectors) >= params.max_detectors_per_layer:
                break
            center_y, center_x = rng.choice(coords)
            center_code = space.grid[center_y][center_x]
            if center_code is None:
                attempts += 1
                continue
            activated = _local_activation(
                center_y,
                center_x,
                center_code,
                space,
                params.activation_radius,
                lambda_d,
                params.similarity,
                params.eta,
            )
            clusters = _dbscan(activated, params.cluster_eps, params.cluster_min_points)
            inserted = False
            for cluster in clusters:
                detector = _detector_from_cluster(cluster, space, lambda_d, layer_index, params, rng)
                if detector is None:
                    continue
                if _insert_detector(detector, detectors):
                    inserted = True
            if inserted:
                no_insert = 0
            else:
                no_insert += 1
            if no_insert >= params.patience:
                break
            attempts += 1
            if attempts % progress_every == 0:
                _log(
                    "detectors",
                    "layer "
                    f"{layer_index + 1} attempts={attempts} detectors={len(detectors)}"
                )
        _log(
            "detectors",
            "layer "
            f"{layer_index + 1} done detectors={len(detectors)} attempts={attempts}"
        )
        if detectors:
            radius_stats = _RunningStats()
            energy_stats = _RunningStats()
            n_stats = _RunningStats()
            fill_stats = _RunningStats()
            for detector in detectors:
                radius_stats.add(float(detector.radius))
                energy_stats.add(float(detector.energy))
                n_stats.add(float(detector.n))
                fill_stats.add(float(_fill_factor(detector)))
            _log(
                "detectors",
                "layer "
                f"{layer_index + 1} stats "
                f"radius_avg={radius_stats.mean():.2f} "
                f"radius_min={0.0 if radius_stats.min is None else radius_stats.min:.2f} "
                f"radius_max={0.0 if radius_stats.max is None else radius_stats.max:.2f} "
                f"energy_avg={energy_stats.mean():.2f} "
                f"energy_min={0.0 if energy_stats.min is None else energy_stats.min:.2f} "
                f"energy_max={0.0 if energy_stats.max is None else energy_stats.max:.2f} "
                f"n_avg={n_stats.mean():.2f} "
                f"n_min={0.0 if n_stats.min is None else n_stats.min:.0f} "
                f"n_max={0.0 if n_stats.max is None else n_stats.max:.0f} "
                f"fill_avg={fill_stats.mean():.2f}",
            )
        layers.append(DetectorLayer(lambda_d=lambda_d, detectors=detectors))
    hierarchy = DetectorHierarchy(layers=layers, code_length=params.detector_code_length)
    _log("detectors", f"done total_detectors={_detector_count(hierarchy)}")
    return hierarchy


def _color_merge(
    active: Sequence[Detector],
    code_length: int,
    sigma: int,
    merge_order: str,
) -> CodeVector:
    if sigma <= 0:
        return CodeVector(0, 0, code_length)
    priorities: dict[int, float] = {}
    for detector in active:
        bit = detector.bit_index
        priority = detector.lambda_d
        if bit in priorities:
            if merge_order == "high":
                priorities[bit] = max(priorities[bit], priority)
            else:
                priorities[bit] = min(priorities[bit], priority)
        else:
            priorities[bit] = priority
    reverse = merge_order == "high"
    ranked = sorted(priorities.items(), key=lambda item: item[1], reverse=reverse)
    if len(ranked) > sigma:
        ranked = ranked[:sigma]
    bits = 0
    for bit, _ in ranked:
        bits |= 1 << bit
    ones = len(ranked)
    return CodeVector(bits, ones, code_length)


def _init_embed_stats(layer_count: int) -> dict[str, object]:
    return {
        "samples": 0,
        "active_total": 0,
        "active_min": None,
        "active_max": None,
        "active_by_layer": [0 for _ in range(layer_count)],
    }


def _update_embed_stats(
    stats: dict[str, object],
    active_total: int,
    active_by_layer: list[int],
) -> None:
    stats["samples"] = int(stats["samples"]) + 1
    stats["active_total"] = int(stats["active_total"]) + active_total
    active_min = stats["active_min"]
    active_max = stats["active_max"]
    if active_min is None or active_total < int(active_min):
        stats["active_min"] = active_total
    if active_max is None or active_total > int(active_max):
        stats["active_max"] = active_total
    layer_totals = stats["active_by_layer"]
    for idx, count in enumerate(active_by_layer):
        layer_totals[idx] += count


def _log_embed_activity(
    level: str,
    stats: dict[str, object],
    detectors: DetectorHierarchy,
) -> None:
    samples = int(stats["samples"])
    if samples <= 0:
        return
    active_total = int(stats["active_total"])
    avg_active = active_total / samples if samples else 0.0
    active_min = stats["active_min"]
    active_max = stats["active_max"]
    layer_totals = stats["active_by_layer"]
    per_layer = []
    for idx, layer in enumerate(detectors.layers):
        layer_count = len(layer.detectors)
        avg_layer = layer_totals[idx] / samples if samples else 0.0
        frac_layer = (avg_layer / layer_count) if layer_count else 0.0
        per_layer.append(
            {
                "layer": idx + 1,
                "avg_active": round(avg_layer, 2),
                "frac": round(frac_layer, 3),
            }
        )
    _log(
        "embed",
        f"{level} active_detectors "
        f"avg={avg_active:.2f} "
        f"min={0 if active_min is None else active_min} "
        f"max={0 if active_max is None else active_max} "
        f"per_layer={json.dumps(per_layer, ensure_ascii=True)}",
    )


def embed_stimulus(
    stimuli: Sequence[object],
    space: CodeSpace,
    detectors: DetectorHierarchy,
    params: EmbedParams,
    *,
    stats: dict[str, object] | None = None,
) -> CodeVector:
    if not detectors.layers:
        return CodeVector(0, 0, detectors.code_length)
    stimuli_codes = [_code_from_any(code, space.code_length) for code in stimuli]
    activation = _activation_map(
        stimuli_codes,
        space,
        params.lambda_activation,
        params.similarity,
        params.eta,
    )
    if space.energy is None:
        raise ValueError("space energy map is not available")

    active_detectors: list[Detector] = []
    active_by_layer = [0 for _ in range(len(detectors.layers))] if stats is not None else None
    for layer in detectors.layers:
        for detector in layer.detectors:
            if detector.energy <= 0.0:
                continue
            r_int = int(math.ceil(detector.radius))
            y_min = max(0, int(math.floor(detector.center_y)) - r_int)
            y_max = min(space.height - 1, int(math.floor(detector.center_y)) + r_int)
            x_min = max(0, int(math.floor(detector.center_x)) - r_int)
            x_max = min(space.width - 1, int(math.floor(detector.center_x)) + r_int)
            radius_sq = detector.radius * detector.radius
            energy_sum = 0.0
            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    dy = y - detector.center_y
                    dx = x - detector.center_x
                    if dy * dy + dx * dx > radius_sq:
                        continue
                    if space.energy[y][x] < params.mu_e:
                        continue
                    a = activation[y][x]
                    if a <= 0.0:
                        continue
                    energy_sum += a * space.energy[y][x]
            level = energy_sum / detector.energy
            if level >= params.mu_d:
                active_detectors.append(detector)
                if active_by_layer is not None:
                    active_by_layer[detector.layer_index] += 1
    if stats is not None and active_by_layer is not None:
        _update_embed_stats(stats, len(active_detectors), active_by_layer)
    return _color_merge(active_detectors, detectors.code_length, params.sigma, params.merge_order)


def _codevector_to_layout(code: CodeVector) -> LayoutBitArray:
    out = LayoutBitArray(code.length)
    bits = code.bits
    while bits:
        lsb = bits & -bits
        idx = lsb.bit_length() - 1
        out.set(idx, 1)
        bits ^= lsb
    return out


def build_layout_from_embeddings(
    embeddings_by_label: Mapping[int, Sequence[CodeVector]],
    config: LayoutConfig,
) -> Layout:
    codes: dict[float, list[LayoutBitArray]] = {}
    total_embeddings = 0
    ones_stats = _RunningStats()
    zero_codes = 0
    label_counts: dict[int, int] = {}
    for label, entries in embeddings_by_label.items():
        bucket: list[LayoutBitArray] = []
        label_counts[int(label)] = len(entries)
        for code in entries:
            bucket.append(_codevector_to_layout(code))
            ones_stats.add(float(code.ones))
            if code.ones == 0:
                zero_codes += 1
        codes[float(label)] = bucket
        total_embeddings += len(entries)
    zero_frac = (zero_codes / total_embeddings) if total_embeddings else 0.0
    _log(
        "layout",
        f"build labels={len(embeddings_by_label)} total={total_embeddings} "
        f"ones_avg={ones_stats.mean():.2f} "
        f"ones_min={0 if ones_stats.min is None else ones_stats.min:.0f} "
        f"ones_max={0 if ones_stats.max is None else ones_stats.max:.0f} "
        f"zero_frac={zero_frac:.3f} "
        f"per_label_counts={json.dumps(dict(sorted(label_counts.items())), ensure_ascii=True)}",
    )
    layout = Layout(codes, **config.layout_kwargs)
    for run in config.run_schedule:
        layout.run(**run)
    _log("layout", f"done grid={layout.height}x{layout.width}")
    return layout


def space_from_layout(layout: Layout) -> CodeSpace:
    if not layout._points:
        raise ValueError("layout is empty")
    code_length = len(layout._points[0].code)
    grid: list[list[CodeVector | None]] = [
        [None for _ in range(layout.width)] for _ in range(layout.height)
    ]
    for y in range(layout.height):
        for x in range(layout.width):
            idx = layout.grid[y][x]
            if idx is None:
                continue
            code = layout._points[idx].code
            grid[y][x] = _code_from_any(code, code_length)
    return CodeSpace(
        grid=grid,
        height=layout.height,
        width=layout.width,
        code_length=code_length,
    )


def encode_image(
    image: object,
    encoder: Encoder,
    extractor: MnistSobelAngleMap,
    *,
    label: int | None = None,
) -> list[CodeVector]:
    if hasattr(image, "numpy"):
        img = image.numpy()
    else:
        img = image
    if hasattr(img, "squeeze"):
        img = img.squeeze()
    extract_label = 0 if label is None else int(label)
    data = extractor.extract(img, extract_label)
    values = next(iter(data.values()), [])
    stimuli: list[CodeVector] = []
    for angle, x, y in values:
        _, code = encoder.encode(float(angle), float(x), float(y))
        stimuli.append(_code_from_any(code, encoder.code_length))
    return stimuli


def train_hierarchy(train_images: Iterable[tuple[object, int]], config: HierarchyConfig) -> HierarchyModel:
    total = None
    if hasattr(train_images, "__len__"):
        total = len(train_images)  # type: ignore[arg-type]
    _log("levels", f"train start samples={total if total is not None else 'unknown'}")
    d1 = build_detectors(config.v0, config.build_l1)
    _log("levels", f"D1 detectors={_detector_count(d1)}")
    _log(
        "embed",
        "L1 params "
        f"lambda_activation={config.embed_l1.lambda_activation} "
        f"mu_e={config.embed_l1.mu_e} "
        f"mu_d={config.embed_l1.mu_d} "
        f"sigma={config.embed_l1.sigma} "
        f"similarity={config.embed_l1.similarity} "
        f"eta={config.embed_l1.eta} "
        f"merge_order={config.embed_l1.merge_order}",
    )

    c1_by_label: dict[int, list[CodeVector]] = defaultdict(list)
    c1_ordered: list[tuple[CodeVector, int]] = []
    c1_ones = _RunningStats()
    c1_zero = 0
    c1_sigma_hits = 0
    stimuli_stats = _RunningStats()
    stimuli_zero = 0
    stimuli_per_label_count: dict[int, int] = defaultdict(int)
    stimuli_per_label_sum: dict[int, int] = defaultdict(int)
    stimuli_per_label_zero: dict[int, int] = defaultdict(int)
    stimulus_ones = _RunningStats()
    embed_stats_l1 = _init_embed_stats(len(d1.layers))
    for idx, (image, label) in enumerate(train_images, start=1):
        label_int = int(label)
        stimuli = encode_image(image, config.encoder, config.extractor, label=label_int)
        stimuli_count = len(stimuli)
        stimuli_stats.add(float(stimuli_count))
        stimuli_per_label_count[label_int] += 1
        stimuli_per_label_sum[label_int] += stimuli_count
        if stimuli_count == 0:
            stimuli_zero += 1
            stimuli_per_label_zero[label_int] += 1
        for stim_code in stimuli:
            stimulus_ones.add(float(stim_code.ones))
        c1 = embed_stimulus(stimuli, config.v0, d1, config.embed_l1, stats=embed_stats_l1)
        c1_by_label[label_int].append(c1)
        c1_ordered.append((c1, label_int))
        c1_ones.add(float(c1.ones))
        if c1.ones == 0:
            c1_zero += 1
        if c1.ones >= config.embed_l1.sigma:
            c1_sigma_hits += 1
        if idx % LOG_EVERY == 0:
            suffix = f"/{total}" if total is not None else ""
            avg_ones = c1_ones.mean()
            avg_stimuli = stimuli_stats.mean()
            _log(
                "embed",
                f"L1 embeddings {idx}{suffix} avg_ones={avg_ones:.1f} "
                f"avg_stimuli={avg_stimuli:.1f}",
            )
    if c1_ordered:
        avg_ones = c1_ones.mean()
        avg_stimuli = stimuli_stats.mean()
        zero_frac = c1_zero / len(c1_ordered)
        sigma_frac = c1_sigma_hits / len(c1_ordered)
        _log(
            "embed",
            "L1 done "
            f"count={len(c1_ordered)} avg_ones={avg_ones:.1f} "
            f"min_ones={0 if c1_ones.min is None else c1_ones.min:.0f} "
            f"max_ones={0 if c1_ones.max is None else c1_ones.max:.0f} "
            f"zero_frac={zero_frac:.3f} "
            f"sigma_frac={sigma_frac:.3f} "
            f"avg_stimuli={avg_stimuli:.1f} "
            f"min_stimuli={0 if stimuli_stats.min is None else stimuli_stats.min:.0f} "
            f"max_stimuli={0 if stimuli_stats.max is None else stimuli_stats.max:.0f}",
        )
        stim_zero_frac = stimuli_zero / len(c1_ordered) if c1_ordered else 0.0
        per_label_avg = {
            label: (stimuli_per_label_sum[label] / stimuli_per_label_count[label])
            if stimuli_per_label_count[label]
            else 0.0
            for label in sorted(stimuli_per_label_count.keys())
        }
        per_label_zero = {
            label: (stimuli_per_label_zero[label] / stimuli_per_label_count[label])
            if stimuli_per_label_count[label]
            else 0.0
            for label in sorted(stimuli_per_label_count.keys())
        }
        _log(
            "measure",
            "stimuli_per_image "
            f"avg={stimuli_stats.mean():.2f} "
            f"min={0 if stimuli_stats.min is None else stimuli_stats.min:.0f} "
            f"max={0 if stimuli_stats.max is None else stimuli_stats.max:.0f} "
            f"zero_frac={stim_zero_frac:.3f} "
            f"per_label_count={json.dumps(dict(sorted(stimuli_per_label_count.items())), ensure_ascii=True)} "
            f"per_label_avg={json.dumps(per_label_avg, ensure_ascii=True)} "
            f"per_label_zero_frac={json.dumps(per_label_zero, ensure_ascii=True)}",
        )
        _log(
            "encode",
            "stimulus_code_ones "
            f"avg={stimulus_ones.mean():.2f} "
            f"min={0 if stimulus_ones.min is None else stimulus_ones.min:.0f} "
            f"max={0 if stimulus_ones.max is None else stimulus_ones.max:.0f} "
            f"count={stimulus_ones.count}",
        )
        _log_embed_activity("L1", embed_stats_l1, d1)

    layout1 = build_layout_from_embeddings(c1_by_label, config.layout_l2)
    v1 = space_from_layout(layout1)
    d2 = build_detectors(v1, config.build_l2)
    _log("levels", f"D2 detectors={_detector_count(d2)}")
    _log(
        "embed",
        "L2 params "
        f"lambda_activation={config.embed_l2.lambda_activation} "
        f"mu_e={config.embed_l2.mu_e} "
        f"mu_d={config.embed_l2.mu_d} "
        f"sigma={config.embed_l2.sigma} "
        f"similarity={config.embed_l2.similarity} "
        f"eta={config.embed_l2.eta} "
        f"merge_order={config.embed_l2.merge_order}",
    )

    c2_by_label: dict[int, list[CodeVector]] = defaultdict(list)
    c2_ordered: list[tuple[CodeVector, int]] = []
    c2_ones = _RunningStats()
    c2_zero = 0
    c2_sigma_hits = 0
    embed_stats_l2 = _init_embed_stats(len(d2.layers))
    for idx, (c1, label) in enumerate(c1_ordered, start=1):
        c2 = embed_stimulus([c1], v1, d2, config.embed_l2, stats=embed_stats_l2)
        c2_by_label[int(label)].append(c2)
        c2_ordered.append((c2, int(label)))
        c2_ones.add(float(c2.ones))
        if c2.ones == 0:
            c2_zero += 1
        if c2.ones >= config.embed_l2.sigma:
            c2_sigma_hits += 1
        if idx % LOG_EVERY == 0:
            avg = c2_ones.mean()
            _log("embed", f"L2 embeddings {idx}/{len(c1_ordered)} avg_ones={avg:.1f}")
    if c2_ordered:
        avg = c2_ones.mean()
        zero_frac = c2_zero / len(c2_ordered)
        sigma_frac = c2_sigma_hits / len(c2_ordered)
        _log(
            "embed",
            "L2 done "
            f"count={len(c2_ordered)} avg_ones={avg:.1f} "
            f"min_ones={0 if c2_ones.min is None else c2_ones.min:.0f} "
            f"max_ones={0 if c2_ones.max is None else c2_ones.max:.0f} "
            f"zero_frac={zero_frac:.3f} "
            f"sigma_frac={sigma_frac:.3f}",
        )
        _log_embed_activity("L2", embed_stats_l2, d2)

    layout2 = build_layout_from_embeddings(c2_by_label, config.layout_l3)
    v2 = space_from_layout(layout2)
    d3 = build_detectors(v2, config.build_l3)
    _log("levels", f"D3 detectors={_detector_count(d3)}")
    _log(
        "embed",
        "L3 params "
        f"lambda_activation={config.embed_l3.lambda_activation} "
        f"mu_e={config.embed_l3.mu_e} "
        f"mu_d={config.embed_l3.mu_d} "
        f"sigma={config.embed_l3.sigma} "
        f"similarity={config.embed_l3.similarity} "
        f"eta={config.embed_l3.eta} "
        f"merge_order={config.embed_l3.merge_order}",
    )

    memory: list[MemoryEntry] = []
    c3_ones = _RunningStats()
    c3_zero = 0
    c3_sigma_hits = 0
    embed_stats_l3 = _init_embed_stats(len(d3.layers))
    for idx, (c2, label) in enumerate(c2_ordered, start=1):
        c3 = embed_stimulus([c2], v2, d3, config.embed_l3, stats=embed_stats_l3)
        memory.append(MemoryEntry(code=c3, label=label))
        c3_ones.add(float(c3.ones))
        if c3.ones == 0:
            c3_zero += 1
        if c3.ones >= config.embed_l3.sigma:
            c3_sigma_hits += 1
        if idx % LOG_EVERY == 0:
            avg = c3_ones.mean()
            _log("embed", f"L3 embeddings {idx}/{len(c2_ordered)} avg_ones={avg:.1f}")
    if memory:
        avg = c3_ones.mean()
        zero_frac = c3_zero / len(memory)
        sigma_frac = c3_sigma_hits / len(memory)
        _log(
            "embed",
            "L3 done "
            f"count={len(memory)} avg_ones={avg:.1f} "
            f"min_ones={0 if c3_ones.min is None else c3_ones.min:.0f} "
            f"max_ones={0 if c3_ones.max is None else c3_ones.max:.0f} "
            f"zero_frac={zero_frac:.3f} "
            f"sigma_frac={sigma_frac:.3f}",
        )
        _log_embed_activity("L3", embed_stats_l3, d3)
    _log("levels", f"memory size={len(memory)}")
    if memory:
        mem_label_counts: dict[int, int] = defaultdict(int)
        for entry in memory:
            mem_label_counts[int(entry.label)] += 1
        _log(
            "levels",
            f"memory labels={json.dumps(dict(sorted(mem_label_counts.items())), ensure_ascii=True)}",
        )

    return HierarchyModel(
        v0=config.v0,
        v1=v1,
        v2=v2,
        d1=d1,
        d2=d2,
        d3=d3,
        memory=memory,
        encoder=config.encoder,
        extractor=config.extractor,
        embed_l1=config.embed_l1,
        embed_l2=config.embed_l2,
        embed_l3=config.embed_l3,
    )


def _decode_memory(
    code: CodeVector,
    memory: Sequence[MemoryEntry],
    top_k: int,
    similarity: str,
) -> tuple[int | None, list[tuple[int, float]]]:
    if not memory:
        return None, []
    scored: list[tuple[int, float]] = []
    for entry in memory:
        score = _similarity(code, entry.code, similarity)
        scored.append((entry.label, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    top = scored[: max(1, top_k)]
    votes: dict[int, float] = {}
    for label, score in top:
        votes[label] = votes.get(label, 0.0) + score
    predicted = max(votes.items(), key=lambda item: item[1])[0] if votes else None
    return predicted, top


def infer(
    image: object,
    model: HierarchyModel,
    *,
    top_k: int = 5,
    similarity: str = "cosine",
    label: int | None = None,
) -> tuple[int | None, list[tuple[int, float]]]:
    stimuli = encode_image(image, model.encoder, model.extractor)
    if LOG_DECODE_DETAILS:
        _log("decode", f"stimuli={len(stimuli)}")
    c1 = embed_stimulus(stimuli, model.v0, model.d1, model.embed_l1)
    c2 = embed_stimulus([c1], model.v1, model.d2, model.embed_l2)
    c3 = embed_stimulus([c2], model.v2, model.d3, model.embed_l3)
    if LOG_DECODE_DETAILS:
        _log("decode", f"C1 ones={c1.ones} C2 ones={c2.ones} C3 ones={c3.ones}")
    predicted, top = _decode_memory(c3, model.memory, top_k, similarity)
    if LOG_DECODE_DETAILS and top:
        _log("decode", f"top1 label={top[0][0]} score={top[0][1]:.4f}")
    if LOG_DECODE_DETAILS:
        _log("decode", f"predicted={predicted}")
    if label is not None:
        stats = _get_decode_stats()
        stats["total"] = int(stats["total"]) + 1
        label_int = int(label)
        stats["label_counts"][label_int] += 1
        if predicted is None:
            stats["none_pred"] = int(stats["none_pred"]) + 1
        else:
            pred_int = int(predicted)
            stats["pred_counts"][pred_int] += 1
            stats["confusion"][label_int][pred_int] += 1
            if pred_int == label_int:
                stats["correct"] = int(stats["correct"]) + 1
        if top:
            stats["top1_stats"].add(float(top[0][1]))
            if any(lbl == label_int for lbl, _score in top):
                stats["topk_correct"] = int(stats["topk_correct"]) + 1
        if stats["top_k"] is None:
            stats["top_k"] = int(top_k)
        elif isinstance(stats["top_k"], int) and stats["top_k"] != int(top_k):
            stats["top_k"] = 0
    return predicted, top
