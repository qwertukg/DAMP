# damp_opp_pinwheels_180.py
# Python 3.10+
#
# pip install rerun-sdk numpy
# python damp_opp_pinwheels_180.py
#
# В Rerun смотри:
# - maps/angle_all          — итоговая карта ориентации (должно быть ~5 pinwheel центров)
# - maps/pos_0..4           — слои по позиции (каждый слой формирует "свой" pinwheel)
# - maps/angle_pos_brightness — общая карта, где яркость = позиция (помогает увидеть кластеры)

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import rerun as rr


# ---------------------------- параметры ----------------------------

@dataclass
class Config:
    # Домен: 5 позиций × 180 ориентаций (как в OPP: orientation, не direction)
    n_pos: int = 5
    n_theta: int = 720  # θ ∈ [0,180)

    # Решетка "коры": 30*30 = 900 = 5*180
    H: int = 60
    W: int = 60

    # Разряженные коды (как в DAMP): позиции + угол
    k_theta: int = 1024
    w_theta: int = 14
    k_pos: int = 128
    w_pos: int = 26

    # DAMP tau: sim_lambda = s * sigmoid(eta*(s-lambda))
    eta: float = 22.0
    lam_start: float = 0.05
    lam_mid: float = 0.25
    lam_end: float = 0.40

    # OPP-подобная связь внутри позиции: c(d)=c0 + cp*exp(-d^2/(2*sigma^2))
    c0: float = 0.18      # baseline (постоянная компонента) — "нужно всё рядом"
    cp: float = 0.82      # пик по близким ориентациям
    sigma_deg: float = 18.0  # ширина "ориентационного" пика (градусы, по модулю 180)

    # Межпозиционная связь (чтобы 5 позиций не перемешались в кашу)
    # 0.0 -> позиции независимы (могут смешиваться из-за равнодушия).
    # небольшое значение помогает собрать 5 отдельных "островов".
    cross_pos: float = 0.02

    # DAMP итерации
    pairs_per_iter: int = 220

    iters_long: int = 1400
    pair_radius_long: int = 10
    sample_long: int = 260  # подвыборка точек при оценке long-range энергии

    iters_short: int = 800
    pair_radius_short: int = 4
    neigh_radius_short: int = 8  # R вокруг середины пары (short-range polishing)

    # визуализация
    vis_every: int = 10
    sleep_sec: float = 0.0

    seed: int = 1


# ---------------------------- sparse bit codes ----------------------------

def _rot_window_bits(k: int, center: int, width: int) -> int:
    if k <= 0 or width <= 0:
        return 0
    width = min(width, k)
    half = width // 2
    bits = 0
    for off in range(-half, -half + width):
        idx = (center + off) % k
        bits |= (1 << idx)
    return bits


def encode_theta_sparse(theta_deg: int, cfg: Config) -> int:
    # theta in [0,180)
    t = theta_deg % cfg.n_theta
    center = int(t * cfg.k_theta / cfg.n_theta)
    return _rot_window_bits(cfg.k_theta, center, cfg.w_theta)


def encode_pos_sparse(pos: int, cfg: Config) -> int:
    # делаем циклическим (кольцо), чтобы соседние позиции имели небольшой overlap
    p = pos % cfg.n_pos
    center = int(p * cfg.k_pos / cfg.n_pos)
    return _rot_window_bits(cfg.k_pos, center, cfg.w_pos)


def encode_pos_theta(pos: int, theta_deg: int, cfg: Config) -> int:
    # конкатенация: [pos_bits] + [theta_bits]
    pos_bits = encode_pos_sparse(pos, cfg)
    th_bits = encode_theta_sparse(theta_deg, cfg)
    return pos_bits | (th_bits << cfg.k_pos)


# ---------------------------- similarity: модуль 180 + OPP-like c(d) ----------------------------

def sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


def ang_dist(a: int, b: int, cfg: Config) -> float:
    """d = min(|a-b|, n_theta-|a-b|) for theta index in [0,n_theta)."""
    d = abs((a % cfg.n_theta) - (b % cfg.n_theta))
    return float(min(d, cfg.n_theta - d))


def opp_connection(d_idx: float, cfg: Config) -> float:
    s = sigma_idx(cfg)
    s2 = s * s
    return cfg.c0 + cfg.cp * math.exp(-(d_idx * d_idx) / (2.0 * s2))


def sim_base(item_i: int, item_j: int, item_pos: np.ndarray, item_theta: np.ndarray, cfg: Config) -> float:
    """
    "Похожесть" по канону OPP внутри позиции:
      - если pos одинаковый: baseline+gaussian по углу (mod 180)
      - если pos разный: маленькая константа (чтобы позиции могли "разойтись" в 5 островов)
    """
    pi = int(item_pos[item_i])
    pj = int(item_pos[item_j])
    if pi == pj:
        d = ang_dist(int(item_theta[item_i]), int(item_theta[item_j]), cfg)
        return opp_connection(d, cfg)
    else:
        return cfg.cross_pos
    

def sigma_idx(cfg: Config) -> float:
    # сколько индексов соответствует 1 градусу
    # 180 градусов / n_theta индексов => 1 градус = n_theta/180 индексов
    return cfg.sigma_deg * (cfg.n_theta / 180.0)


def sim_lambda(s: float, lam: float, eta: float) -> float:
    # DAMP: τ(s)=s*σ(η(s-λ))
    return s * sigmoid(eta * (s - lam))


# ---------------------------- тор (periodic distances) ----------------------------

def idx_to_xy(idx: int, W: int) -> Tuple[int, int]:
    return idx // W, idx % W


def xy_to_idx(y: int, x: int, H: int, W: int) -> int:
    return (y % H) * W + (x % W)


def torus_d2(y1: int, x1: int, y2: int, x2: int, H: int, W: int) -> int:
    dy = abs(y1 - y2)
    dx = abs(x1 - x2)
    dy = min(dy, H - dy)
    dx = min(dx, W - dx)
    return dy * dy + dx * dx


def random_pair_near(idx1: int, radius: int, cfg: Config, rng: random.Random) -> int:
    y1, x1 = idx_to_xy(idx1, cfg.W)
    for _ in range(40):
        dy = rng.randint(-radius, radius)
        dx = rng.randint(-radius, radius)
        if dy == 0 and dx == 0:
            continue
        if dy * dy + dx * dx <= radius * radius:
            return xy_to_idx(y1 + dy, x1 + dx, cfg.H, cfg.W)
    return rng.randrange(cfg.H * cfg.W)


def circle_neighbourhood(mid_y: float, mid_x: float, r: int, cfg: Config) -> List[int]:
    out: List[int] = []
    r2 = r * r
    y0 = int(math.floor(mid_y)) - r
    y1 = int(math.floor(mid_y)) + r
    x0 = int(math.floor(mid_x)) - r
    x1 = int(math.floor(mid_x)) + r
    for yy in range(y0, y1 + 1):
        for xx in range(x0, x1 + 1):
            y = yy % cfg.H
            x = xx % cfg.W
            dy = abs(y - mid_y)
            dx = abs(x - mid_x)
            dy = min(dy, cfg.H - dy)
            dx = min(dx, cfg.W - dx)
            if dy * dy + dx * dx <= r2:
                out.append(xy_to_idx(y, x, cfg.H, cfg.W))
    return out


# ---------------------------- DAMP delta энергии пары (phi_s - phi_c) ----------------------------

def pair_delta_long(
    grid_items_flat: np.ndarray,  # [N] -> item_id
    idx1: int,
    idx2: int,
    sample_indices: List[int],
    item_pos: np.ndarray,
    item_theta: np.ndarray,
    lam: float,
    cfg: Config,
) -> float:
    """
    Long-range: глобальный порядок.
    delta = Σ ( (s2 - s1) * (d1 - d2) )
    где s1 = simλ(v, a), s2 = simλ(v, b)
    """
    y1, x1 = idx_to_xy(idx1, cfg.W)
    y2, x2 = idx_to_xy(idx2, cfg.W)

    a = int(grid_items_flat[idx1])
    b = int(grid_items_flat[idx2])

    delta = 0.0
    for j in sample_indices:
        if j == idx1 or j == idx2:
            continue
        v = int(grid_items_flat[j])
        y, x = idx_to_xy(j, cfg.W)

        d1 = torus_d2(y1, x1, y, x, cfg.H, cfg.W)
        d2 = torus_d2(y2, x2, y, x, cfg.H, cfg.W)

        s1 = sim_lambda(sim_base(v, a, item_pos, item_theta, cfg), lam, cfg.eta)
        s2 = sim_lambda(sim_base(v, b, item_pos, item_theta, cfg), lam, cfg.eta)

        delta += (s2 - s1) * (d1 - d2)

    return delta


def pair_delta_short(
    grid_items_flat: np.ndarray,
    idx1: int,
    idx2: int,
    neigh_indices: List[int],
    item_pos: np.ndarray,
    item_theta: np.ndarray,
    lam: float,
    cfg: Config,
) -> float:
    """
    Short-range: локальная полировка (DAMP: swap если phi_s > phi_c => delta>0).
    """
    y1, x1 = idx_to_xy(idx1, cfg.W)
    y2, x2 = idx_to_xy(idx2, cfg.W)

    a = int(grid_items_flat[idx1])
    b = int(grid_items_flat[idx2])

    delta = 0.0
    for j in neigh_indices:
        if j == idx1 or j == idx2:
            continue
        v = int(grid_items_flat[j])
        y, x = idx_to_xy(j, cfg.W)

        d1 = torus_d2(y1, x1, y, x, cfg.H, cfg.W)
        d2 = torus_d2(y2, x2, y, x, cfg.H, cfg.W)

        s1 = sim_lambda(sim_base(v, a, item_pos, item_theta, cfg), lam, cfg.eta)
        s2 = sim_lambda(sim_base(v, b, item_pos, item_theta, cfg), lam, cfg.eta)

        delta += (s2 - s1) * (d1 - d2)

    return delta


# ---------------------------- визуализация ----------------------------

def hsv_to_rgb_u8(h: np.ndarray) -> np.ndarray:
    h6 = (h * 6.0).astype(np.float32)
    i = np.floor(h6).astype(np.int32) % 6
    f = h6 - np.floor(h6)

    q = 1.0 - f
    t = f

    r = np.zeros_like(h, dtype=np.float32)
    g = np.zeros_like(h, dtype=np.float32)
    b = np.zeros_like(h, dtype=np.float32)

    m = (i == 0)
    r[m], g[m], b[m] = 1.0, t[m], 0.0
    m = (i == 1)
    r[m], g[m], b[m] = q[m], 1.0, 0.0
    m = (i == 2)
    r[m], g[m], b[m] = 0.0, 1.0, t[m]
    m = (i == 3)
    r[m], g[m], b[m] = 0.0, q[m], 1.0
    m = (i == 4)
    r[m], g[m], b[m] = t[m], 0.0, 1.0
    m = (i == 5)
    r[m], g[m], b[m] = 1.0, 0.0, q[m]

    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def build_maps(grid_items: np.ndarray, item_theta: np.ndarray, item_pos: np.ndarray, cfg: Config):
    flat = grid_items.reshape(-1)
    theta_grid = item_theta[flat].reshape(cfg.H, cfg.W)  # [H,W]
    pos_grid = item_pos[flat].reshape(cfg.H, cfg.W)

    # hue по θ∈[0,180)
    hue = (theta_grid.astype(np.float32) / float(cfg.n_theta)) % 1.0
    rgb_theta = hsv_to_rgb_u8(hue)

    # яркость как позиция (чтобы видеть 5 "островов")
    brightness = (pos_grid.astype(np.float32) + 1.0) / float(cfg.n_pos)
    rgb_mix = np.clip(rgb_theta.astype(np.float32) * brightness[..., None], 0, 255).astype(np.uint8)

    layers = []
    for p in range(cfg.n_pos):
        mask = (pos_grid == p)
        img = np.zeros_like(rgb_theta)
        img[mask] = rgb_theta[mask]
        layers.append(img)

    return rgb_theta, rgb_mix, layers


# ---------------------------- main ----------------------------

def main():
    cfg = Config()
    rng = random.Random(cfg.seed)
    np.random.seed(cfg.seed)

    assert cfg.H * cfg.W == cfg.n_pos * cfg.n_theta, "H*W должно быть равно 5*180"

    N = cfg.H * cfg.W

    # items: id -> (pos, theta) + sparse code (для канона DAMP)
    item_pos = np.zeros(N, dtype=np.int32)
    item_theta = np.zeros(N, dtype=np.int32)
    item_code: List[int] = [0] * N  # не обязателен для sim, но сохраняем как "кодирование разряженными битами"
    item_pop: List[int] = [0] * N

    k = 0
    for p in range(cfg.n_pos):
        for t in range(cfg.n_theta):
            item_pos[k] = p
            item_theta[k] = t
            c = encode_pos_theta(p, t, cfg)
            item_code[k] = c
            item_pop[k] = c.bit_count()
            k += 1

    # старт — случайная перестановка
    perm = np.arange(N, dtype=np.int32)
    rng.shuffle(perm.tolist())
    grid_items = perm.reshape(cfg.H, cfg.W).copy()

    rr.init("damp_opp_pinwheels_180", spawn=True)

    rgb_theta, rgb_mix, layers = build_maps(grid_items, item_theta, item_pos, cfg)
    rr.log("maps/angle_all", rr.Image(rgb_theta))
    rr.log("maps/angle_pos_brightness", rr.Image(rgb_mix))
    for p, img in enumerate(layers):
        rr.log(f"maps/pos_{p}", rr.Image(img))

    rr.log(
        "meta/info",
        rr.TextDocument(
            f"""
# DAMP×OPP demo (θ in [0,180))
- items: {cfg.n_pos} positions × {cfg.n_theta} orientations = {N}
- grid: {cfg.H}×{cfg.W} (toroidal)
- sparse codes: pos {cfg.k_pos}b(w={cfg.w_pos}) + theta {cfg.k_theta}b(w={cfg.w_theta})

## Similarity
- d = min(|Δθ|, 180-|Δθ|)
- c(d)=c0 + cp*exp(-d^2/(2*sigma^2)) within same position
  c0={cfg.c0}, cp={cfg.cp}, sigma={cfg.sigma_deg}°
- cross_pos={cfg.cross_pos}
- tau(s)=s*sigmoid(eta*(s-lambda)), eta={cfg.eta}
"""
        ),
    )

    total_iters = cfg.iters_long + cfg.iters_short
    t0 = time.time()

    for it in range(total_iters):
        # совместимость: старые SDK имели set_time_sequence, новые — set_time(sequence=...)
        if hasattr(rr, "set_time_sequence"):
            rr.set_time_sequence("iter", it)
        else:
            rr.set_time("iter", sequence=it)

        if it < cfg.iters_long:
            # lambda schedule
            u = it / max(1, cfg.iters_long - 1)
            lam = cfg.lam_start + (cfg.lam_mid - cfg.lam_start) * u
            phase = "long"
            pair_r = cfg.pair_radius_long
        else:
            u = (it - cfg.iters_long) / max(1, cfg.iters_short - 1)
            lam = cfg.lam_mid + (cfg.lam_end - cfg.lam_mid) * u
            phase = "short"
            pair_r = cfg.pair_radius_short

        grid_flat = grid_items.reshape(-1)

        swaps = 0
        for _ in range(cfg.pairs_per_iter):
            idx1 = rng.randrange(N)
            idx2 = random_pair_near(idx1, pair_r, cfg, rng)
            if idx1 == idx2:
                continue

            if phase == "long":
                sample = [rng.randrange(N) for _ in range(cfg.sample_long)]
                delta = pair_delta_long(grid_flat, idx1, idx2, sample, item_pos, item_theta, lam, cfg)
                do_swap = (delta < 0.0)  # минимизируем
            else:
                y1, x1 = idx_to_xy(idx1, cfg.W)
                y2, x2 = idx_to_xy(idx2, cfg.W)

                dy = y2 - y1
                dx = x2 - x1
                if dy > cfg.H / 2:
                    dy -= cfg.H
                elif dy < -cfg.H / 2:
                    dy += cfg.H
                if dx > cfg.W / 2:
                    dx -= cfg.W
                elif dx < -cfg.W / 2:
                    dx += cfg.W

                mid_y = (y1 + (y1 + dy)) / 2.0
                mid_x = (x1 + (x1 + dx)) / 2.0
                neigh = circle_neighbourhood(mid_y, mid_x, cfg.neigh_radius_short, cfg)

                delta = pair_delta_short(grid_flat, idx1, idx2, neigh, item_pos, item_theta, lam, cfg)
                do_swap = (delta > 0.0)  # максимизируем локально

            if do_swap:
                y1, x1 = idx_to_xy(idx1, cfg.W)
                y2, x2 = idx_to_xy(idx2, cfg.W)
                grid_items[y1, x1], grid_items[y2, x2] = grid_items[y2, x2], grid_items[y1, x1]
                swaps += 1

        rr.log("metrics/lambda", rr.Scalars(lam))
        rr.log("metrics/swaps", rr.Scalars(swaps))
        rr.log("metrics/phase", rr.TextLog(phase))
        rr.log("metrics/elapsed_sec", rr.Scalars(time.time() - t0))

        if it % cfg.vis_every == 0:
            rgb_theta, rgb_mix, layers = build_maps(grid_items, item_theta, item_pos, cfg)
            rr.log("maps/angle_all", rr.Image(rgb_theta))
            rr.log("maps/angle_pos_brightness", rr.Image(rgb_mix))
            for p, img in enumerate(layers):
                rr.log(f"maps/pos_{p}", rr.Image(img))

        if cfg.sleep_sec > 0:
            time.sleep(cfg.sleep_sec)

    # финал
    rgb_theta, rgb_mix, layers = build_maps(grid_items, item_theta, item_pos, cfg)
    rr.log("final/angle_all", rr.Image(rgb_theta))
    rr.log("final/angle_pos_brightness", rr.Image(rgb_mix))
    for p, img in enumerate(layers):
        rr.log(f"final/pos_{p}", rr.Image(img))

    rr.log("meta/done", rr.TextLog("done"))


if __name__ == "__main__":
    main()