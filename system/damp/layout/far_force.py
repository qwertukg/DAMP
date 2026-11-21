"""Смешанное ядро дальней фазы (Р2, задача #18; вне статьи по форме, по статье по смыслу).

Гауссово ядро сглаживания w(d) = exp(−d² / 2σ²) почти не тянет друг к другу далёкие куски одного класса: на зазоре 8–18
клеток его сила в 20–100 раз слабее ядра статьи φ = Σ s·d (§5.5.1: «penalises strongly correlated points that are
distant»). Смешанное ядро добавляет слабый член статьи, не убывающий с расстоянием: w(d) = exp(−d² / 2σ²) − α·d / r,
r — радиус окрестности. α = LAYOUT_LONG_KERNEL_FAR_ALPHA (0 — чистое гауссово ядро, как раньше). Читается при каждом
вызове, поэтому переопределение настроек верхних карт (LADDER_LAYOUT_UPPER_CHANGES) включает его только на них.
"""
from __future__ import annotations


def far_alpha() -> float:
    from damp import settings

    # Имя — буквально «settings.ИМЯ»: по нему кеш шагов находит настройку в коде (CodeFingerprint).
    return float(settings.LAYOUT_LONG_KERNEL_FAR_ALPHA or 0.0)


def gauss_weight() -> str:
    """Имя веса для скомпилированных ядер: «gauss» или «gauss+α» (α входит в ключ кеша ядра)."""
    alpha = far_alpha()
    return "gauss" if alpha <= 0.0 else f"gauss+{alpha!r}"
