from __future__ import annotations

from dataclasses import dataclass
import math
import random
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

from encoding.MnistSobelAngleMap import MnistSobelAngleMap
from encoding.damp_encoder import Encoder
from layout.damp_layout import Layout, BitArray as LayoutBitArray

LOG_ENABLED = True
LOG_EVERY = 50


@dataclass(frozen=True)
class CodeVector:
    bits: int
    ones: int
    length: int


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


def _log(message: str) -> None:
    if LOG_ENABLED:
        print(message)


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
        _log("[energy] max=0.0 (empty or weak similarities)")
        return energies
    for y in range(space.height):
        for x in range(space.width):
            if energies[y][x] > 0.0:
                energies[y][x] /= emax
    _log(f"[energy] built radius={radius} lambda={lambda_threshold:.3f} max={emax:.4f}")
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
        "[build_detectors] space="
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
            "[build_detectors] reuse energy map "
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

    rng = random.Random(params.seed)
    layers: list[DetectorLayer] = []
    for layer_index, lambda_d in enumerate(params.lambda_levels):
        _log(
            "[build_detectors] layer "
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
                    "[build_detectors] layer "
                    f"{layer_index + 1} attempts={attempts} detectors={len(detectors)}"
                )
        _log(
            "[build_detectors] layer "
            f"{layer_index + 1} done detectors={len(detectors)} attempts={attempts}"
        )
        layers.append(DetectorLayer(lambda_d=lambda_d, detectors=detectors))
    hierarchy = DetectorHierarchy(layers=layers, code_length=params.detector_code_length)
    _log(f"[build_detectors] done total_detectors={_detector_count(hierarchy)}")
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


def embed_stimulus(
    stimuli: Sequence[object],
    space: CodeSpace,
    detectors: DetectorHierarchy,
    params: EmbedParams,
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
    for label, entries in embeddings_by_label.items():
        bucket: list[LayoutBitArray] = []
        for code in entries:
            bucket.append(_codevector_to_layout(code))
        codes[float(label)] = bucket
        total_embeddings += len(entries)
    _log(f"[layout] build labels={len(embeddings_by_label)} total={total_embeddings}")
    layout = Layout(codes, **config.layout_kwargs)
    for run in config.run_schedule:
        layout.run(**run)
    _log(f"[layout] done grid={layout.height}x{layout.width}")
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


def encode_image(image: object, encoder: Encoder, extractor: MnistSobelAngleMap) -> list[CodeVector]:
    if hasattr(image, "numpy"):
        img = image.numpy()
    else:
        img = image
    if hasattr(img, "squeeze"):
        img = img.squeeze()
    data = extractor.extract(img, 0)
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
    _log(f"[train] start samples={total if total is not None else 'unknown'}")
    d1 = build_detectors(config.v0, config.build_l1)
    _log(f"[train] D1 detectors={_detector_count(d1)}")

    c1_by_label: dict[int, list[CodeVector]] = defaultdict(list)
    c1_ordered: list[tuple[CodeVector, int]] = []
    ones_sum = 0
    for idx, (image, label) in enumerate(train_images, start=1):
        stimuli = encode_image(image, config.encoder, config.extractor)
        c1 = embed_stimulus(stimuli, config.v0, d1, config.embed_l1)
        c1_by_label[int(label)].append(c1)
        c1_ordered.append((c1, int(label)))
        ones_sum += c1.ones
        if idx % LOG_EVERY == 0:
            suffix = f"/{total}" if total is not None else ""
            avg = ones_sum / idx if idx else 0.0
            _log(f"[train] L1 embeddings {idx}{suffix} avg_ones={avg:.1f}")
    if c1_ordered:
        avg = ones_sum / len(c1_ordered)
        _log(f"[train] L1 done count={len(c1_ordered)} avg_ones={avg:.1f}")

    layout1 = build_layout_from_embeddings(c1_by_label, config.layout_l2)
    v1 = space_from_layout(layout1)
    d2 = build_detectors(v1, config.build_l2)
    _log(f"[train] D2 detectors={_detector_count(d2)}")

    c2_by_label: dict[int, list[CodeVector]] = defaultdict(list)
    c2_ordered: list[tuple[CodeVector, int]] = []
    ones_sum = 0
    for idx, (c1, label) in enumerate(c1_ordered, start=1):
        c2 = embed_stimulus([c1], v1, d2, config.embed_l2)
        c2_by_label[int(label)].append(c2)
        c2_ordered.append((c2, int(label)))
        ones_sum += c2.ones
        if idx % LOG_EVERY == 0:
            avg = ones_sum / idx if idx else 0.0
            _log(f"[train] L2 embeddings {idx}/{len(c1_ordered)} avg_ones={avg:.1f}")
    if c2_ordered:
        avg = ones_sum / len(c2_ordered)
        _log(f"[train] L2 done count={len(c2_ordered)} avg_ones={avg:.1f}")

    layout2 = build_layout_from_embeddings(c2_by_label, config.layout_l3)
    v2 = space_from_layout(layout2)
    d3 = build_detectors(v2, config.build_l3)
    _log(f"[train] D3 detectors={_detector_count(d3)}")

    memory: list[MemoryEntry] = []
    ones_sum = 0
    for idx, (c2, label) in enumerate(c2_ordered, start=1):
        c3 = embed_stimulus([c2], v2, d3, config.embed_l3)
        memory.append(MemoryEntry(code=c3, label=label))
        ones_sum += c3.ones
        if idx % LOG_EVERY == 0:
            avg = ones_sum / idx if idx else 0.0
            _log(f"[train] L3 embeddings {idx}/{len(c2_ordered)} avg_ones={avg:.1f}")
    if memory:
        avg = ones_sum / len(memory)
        _log(f"[train] L3 done count={len(memory)} avg_ones={avg:.1f}")
    _log(f"[train] memory size={len(memory)}")

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
) -> tuple[int | None, list[tuple[int, float]]]:
    stimuli = encode_image(image, model.encoder, model.extractor)
    _log(f"[infer] stimuli={len(stimuli)}")
    c1 = embed_stimulus(stimuli, model.v0, model.d1, model.embed_l1)
    c2 = embed_stimulus([c1], model.v1, model.d2, model.embed_l2)
    c3 = embed_stimulus([c2], model.v2, model.d3, model.embed_l3)
    _log(f"[infer] C1 ones={c1.ones} C2 ones={c2.ones} C3 ones={c3.ones}")
    predicted, top = _decode_memory(c3, model.memory, top_k, similarity)
    if top:
        _log(f"[infer] top1 label={top[0][0]} score={top[0][1]:.4f}")
    _log(f"[infer] predicted={predicted}")
    return predicted, top
