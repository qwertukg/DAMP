"""Этап 1: штрихи детекторами над точками края — правила §6.4 статьи, перенесённые с карты кодов на картинку.

Замысел автора проекта (2026-11-16): место штриха задаёт не окно жёсткой сетки, привязанной к углу картинки, а
детектор, найденный по самим точкам края, — как детекторы и окна цикла. Штрихи тогда стоят в координатах самой цифры:
сдвиг цифры сдвигает их все вместе, сетка не режет наклонный край «лестницей». Угол читается Собелем уже в найденном
месте. Статья строит детекторы над разложенным кодовым пространством; перенос на плоскость картинки — доработка.

Точка — пиксель контура; её «код» — направление градиента рядом с ней. Для каждой точки-центра c̃ (§6.4):

    1. активация A(c̃, r_a): точки не дальше r_a от c̃, чьё направление похоже на направление c̃ (§6.4.1). У встречного
       края линии направление обратное, поэтому он в детектор не входит при любой толщине линии;
    2. кластер — точки активации, связанные с c̃ цепочкой соседних точек (§6.4.1; DBSCAN с соседством в пиксель);
    3. центр — среднее координат точек кластера в поле детектора с весом «активация × энергия» (§6.4.2): это место
       штриха, дробное;
    4. поле детектора — круг радиусом в шаг штрихов. Правило наибольшего заполнения (§6.4.3) у тонкой линии края
       всегда выбирает наименьший радиус, поэтому радиус задан настройкой MNIST_STROKE_STEP. Эллипс по главным
       компонентам (§6.4.4) пробовали 2026-11-17 и по слову автора убрали;
    5. вставка: центры не попадают в поля друг друга; сначала плотные кандидаты, потом редкие (§6.4.5).

Угол штриха — направление суммы градиентов по точкам кластера на куске края длиннее шага (поля соседних детекторов
перекрываются): чем длиннее кусок, тем точнее угол. Сила штриха — энергия детектора e_d (§6.3).

Точки без стрелки (axial): на вход идут не точки края, а точки середины линии (StrokeMidline, замысел автора
2026-11-17: «вместо двух краёв — одна середина»). У середины направление поперёк линии не имеет «внутри» и «снаружи»:
направления θ и θ + 180° — одно и то же. Правила те же, отличий три: похожесть направлений — модуль косинуса; угол
штриха — по сумме удвоенных углов, 0…180°; два штриха могут стоять в одном месте, если их направления разные
(перекрестие линий).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from damp.article_refs import DETECTOR_CONSTRUCTION
from damp.logging import LOGGER


@dataclass(frozen=True)
class Stroke:
    """Штрих края: угол (радианы, ось Y вверх), место и поле в пикселях рабочей картинки, сила и число точек поля.

    half_length — на сколько поле детектора тянется вдоль края от центра: расстояние до самой дальней точки поля
    (как r(p) в §6.4.3); нужна, чтобы рисовать штрих по длине его поля.
    """

    angle: float
    x: float
    y: float
    strength: float
    points: int
    half_length: float


@dataclass(frozen=True)
class StrokeDetectors:
    """Детекторы над точками края картинки (§6.4.1–§6.4.5 на плоскости картинки): по штриху на детектор.

    Все длины — в пикселях рабочей картинки (той, на которой считается Собель).

    Args:
        step: шаг штрихов — радиус поля детектора r_d и наименьшее расстояние между центрами (§6.4.5).
        similarity: наименьший косинус угла между направлением точки и направлением центра — «похожее направление».
        angle_span: полудлина куска края, по которому читается угол, в шагах; не меньше 1.
        min_points: наименьшее число точек края в поле детектора (n_d, §6.3).
        dense_share: кандидат плотный, если точек в его поле не меньше этой доли от медианы; плотные вставляются первыми.
        smoothing: полуширина окна, по которому складывается градиент вокруг точки края.
        grad_threshold: точка края со сложенным градиентом не больше этого направления не имеет и не участвует.
        axial: направления точек без стрелки (точки середины линии): θ и θ + 180° — одно направление.
    """

    step: float
    similarity: float
    angle_span: float
    min_points: int
    dense_share: float
    smoothing: int
    grad_threshold: float
    axial: bool = False

    # Соседние точки контура — пиксели, касающиеся стороной или углом: расстояние между центрами не больше √2.
    NEIGHBOUR = 1.5
    # Допуск сравнения расстояния между центрами с шагом. На ровном крае вдоль осей центры кандидатов стоят ровно в
    # шаге друг от друга, и без допуска исход решала бы ошибка округления (она своя у каждого места картинки):
    # сдвинутая цифра получала бы другую расстановку. С допуском центры ровно в шаге друг другу не мешают.
    ROUNDING = 1e-6

    def __post_init__(self) -> None:
        if self.step <= 0:
            raise ValueError("step must be positive")
        if not -1.0 <= self.similarity <= 1.0:
            raise ValueError("similarity is a cosine: must be in [-1, 1]")
        if self.angle_span < 1.0:
            raise ValueError("angle_span must be at least 1 step")
        if self.min_points < 1 or self.smoothing < 0:
            raise ValueError("min_points must be positive, smoothing non-negative")
        if not 0.0 < self.dense_share <= 1.0:
            raise ValueError("dense_share must be in (0, 1]")

    @property
    def signature(self) -> str:
        """Подпись способа для кеша штрихов и раскладки."""
        return (
            f"detectors(step={self.step:g},sim={self.similarity:g},span={self.angle_span:g},"
            f"min={self.min_points},dense={self.dense_share:g},smooth={self.smoothing}{',axial' if self.axial else ''})"
        )

    # ------------------------------------------------------------------ точки края
    def _edge_points(
        self, contour: "np.ndarray", gx: "np.ndarray", gy: "np.ndarray"
    ) -> tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:
        """Точки края: место центра пикселя (y, x) и градиент, сложенный по окну вокруг точки.

        Отклик Собеля на резком крае лежит в двух рядах пикселей, по одному с каждой стороны от ступеньки; в одном
        пикселе направление грубое. Сумма по окну собирает весь отклик вокруг точки — её направление плавное.
        """
        from scipy import ndimage

        size = 2 * int(self.smoothing) + 1
        area = float(size * size)
        sum_x = ndimage.uniform_filter(np.asarray(gx, dtype=np.float64), size=size, mode="constant") * area
        sum_y = ndimage.uniform_filter(np.asarray(gy, dtype=np.float64), size=size, mode="constant") * area
        rows, cols = np.nonzero(contour)
        vx, vy = sum_x[rows, cols], sum_y[rows, cols]
        alive = np.hypot(vx, vy) > float(self.grad_threshold)
        return rows[alive] + 0.5, cols[alive] + 0.5, vx[alive], vy[alive]

    # ------------------------------------------------------------------ кластеры
    def _clusters(self, distance: "np.ndarray", cosine: "np.ndarray") -> "np.ndarray":
        """Кластер каждой точки-центра (§6.4.1): member[c, j] — точка j в кластере центра c.

        Активация — точки не дальше куска края, по которому читается угол, с направлением, похожим на направление
        центра. Кластер растёт от центра по соседним точкам, не выходя из активации: один шаг роста сразу для всех
        центров — произведение разрежённых матриц.
        """
        from scipy import sparse

        count = distance.shape[0]
        reach = float(self.step) * float(self.angle_span)
        allowed = sparse.csr_matrix((distance <= reach) & (cosine >= float(self.similarity)))
        near = distance <= self.NEIGHBOUR
        np.fill_diagonal(near, False)
        neighbours = sparse.csr_matrix(near.astype(np.float32))
        member = sparse.identity(count, dtype=np.float32, format="csr")
        size = count
        while True:
            grown = (member + (member @ neighbours).multiply(allowed)).astype(bool).astype(np.float32)
            if grown.nnz == size:
                break
            member, size = grown.tocsr(), grown.nnz
        return member.toarray().astype(bool)

    # ------------------------------------------------------------------ штрихи картинки
    def strokes(self, contour: "np.ndarray", gx: "np.ndarray", gy: "np.ndarray") -> list[Stroke]:
        """Штрихи картинки: contour — пиксели контура, gx, gy — отклик Собеля на той же картинке (ось Y вверх)."""
        return self.strokes_of(*self._edge_points(contour, gx, gy))

    def strokes_of(self, y: "np.ndarray", x: "np.ndarray", vx: "np.ndarray", vy: "np.ndarray") -> list[Stroke]:
        """Штрихи по готовым точкам: место центра пикселя (y, x) и вектор направления точки (vx, vy), ось Y вверх.

        Точки края — направление перепада яркости; точки середины линии (axial) — направление поперёк линии, без
        стрелки. Длина вектора — сила точки.
        """
        return self.strokes_with_fields(y, x, vx, vy)[0]

    def strokes_with_fields(
        self, y: "np.ndarray", x: "np.ndarray", vx: "np.ndarray", vy: "np.ndarray"
    ) -> tuple[list[Stroke], list["np.ndarray"]]:
        """Штрихи (как strokes_of) и точки поля каждого: номера входных точек, из которых собран штрих (§6.4.3: точки
        кластера не дальше шага от точки-центра). По ним центр штриха — их взвешенное среднее (§6.4.2); для уровня 0
        по общему подходу они — элементы окна-штриха (LADDER_LEVEL0_PLACE = "stroke")."""
        y, x = np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)
        vx, vy = np.asarray(vx, dtype=np.float64), np.asarray(vy, dtype=np.float64)
        count = int(y.shape[0])
        if count == 0:
            return [], []
        magnitude = np.hypot(vx, vy)
        ux, uy = vx / magnitude, vy / magnitude
        distance = np.hypot(y[:, None] - y[None, :], x[:, None] - x[None, :])
        cosine = ux[:, None] * ux[None, :] + uy[:, None] * uy[None, :]
        if self.axial:
            cosine = np.abs(cosine)  # без стрелки: θ и θ + 180° — одно направление
        member = self._clusters(distance, cosine)
        # Поле детектора — точки кластера не дальше шага от точки-центра (r_d = шаг).
        field = member & (distance <= float(self.step))
        points = field.sum(axis=1)
        # §6.4.2: вес точки — активация (сходство направления с центром) × энергия (сила её градиента).
        energy = magnitude / float(magnitude.max())
        weight = np.where(field, cosine * energy[None, :], 0.0)
        total = weight.sum(axis=1)
        valid = (points >= int(self.min_points)) & (total > 0.0)
        if not bool(valid.any()):
            return [], []
        safe = np.where(total > 0.0, total, 1.0)
        centre_y, centre_x = (weight @ y) / safe, (weight @ x) / safe
        # Угол — по куску края длиннее шага: сумма градиентов точек кластера (весь кластер лежит в пределах куска).
        span = member.astype(np.float64)
        if self.axial:
            # Векторы без стрелки складываются удвоенными углами (иначе встречные гасили бы друг друга); угол — 0…180°.
            double_x, double_y = (vx * vx - vy * vy) / magnitude, 2.0 * vx * vy / magnitude
            angle = np.mod(0.5 * np.arctan2(span @ double_y, span @ double_x), np.pi)
        else:
            angle = np.arctan2(span @ vy, span @ vx)
        chosen = self._insert(y, x, centre_y, centre_x, points, valid, angle)
        found = []
        fields = []
        for i in chosen:
            # Длина поля вдоль края: расстояние от центра до самой дальней точки поля (r(p), §6.4.3).
            inside = field[i]
            fields.append(np.flatnonzero(inside))
            half_length = float(np.hypot(y[inside] - centre_y[i], x[inside] - centre_x[i]).max())
            found.append(
                Stroke(
                    angle=float(angle[i]), x=float(centre_x[i]), y=float(centre_y[i]), strength=float(total[i]),
                    points=int(points[i]), half_length=half_length,
                )
            )
        LOGGER.event(
            "strokes.detectors",
            section=DETECTOR_CONSTRUCTION,
            data={
                "edge_points": count,
                "candidates": int(valid.sum()),
                "strokes": len(found),
                "points_in_field_median": float(np.median(points[valid])),
                "half_length_mean": float(np.mean([stroke.half_length for stroke in found])) if found else 0.0,
                "step": float(self.step),
                "similarity": float(self.similarity),
                "angle_span_steps": float(self.angle_span),
                "axial": bool(self.axial),
                "note": "штрихи — детекторы над точками края по §6.4, перенесённому на картинку (доработка): в детектор "
                "идут точки с похожим направлением (один край), место — взвешенный центр (§6.4.2), центры не ближе "
                "шага (§6.4.5), угол — по куску края длиннее шага",
            },
        )
        return found, fields

    def _insert(
        self, y: "np.ndarray", x: "np.ndarray", centre_y: "np.ndarray", centre_x: "np.ndarray", points: "np.ndarray",
        valid: "np.ndarray", angle: "np.ndarray",
    ) -> list[int]:
        """Вставка детекторов (§6.4.5): центры не попадают в поля друг друга; номера принятых кандидатов.

        У точек без стрелки (середина линии) мешают друг другу только детекторы похожего направления: на перекрестии
        двух линий в одном месте стоят два штриха — по одному на линию.

        Сначала плотные кандидаты — середины ровных кусков края, потом редкие — концы и изгибы, от плотных к редким.
        Равные по плотности идут по порядку мест (сверху вниз, слева направо): на ровном крае центры ложатся
        вплотную, через шаг. Порядок не зависит от того, где цифра стоит на картинке.
        """
        candidates = np.flatnonzero(valid)
        dense = points[candidates] >= float(self.dense_share) * float(np.median(points[candidates]))
        rank = np.where(dense, 0, 1)
        sparse_points = np.where(dense, 0, -points[candidates])
        order = candidates[np.lexsort((x[candidates], y[candidates], sparse_points, rank))]
        step = float(self.step) - self.ROUNDING
        taken = np.zeros(order.shape[0], dtype=np.int64)
        count = 0
        for index in order.tolist():
            if count:
                held = taken[:count]
                near = np.hypot(centre_y[held] - centre_y[index], centre_x[held] - centre_x[index]) < step
                if self.axial:
                    near &= np.abs(np.cos(angle[held] - angle[index])) >= float(self.similarity)
                if bool(np.any(near)):
                    continue
            taken[count] = index
            count += 1
        return taken[:count].tolist()
