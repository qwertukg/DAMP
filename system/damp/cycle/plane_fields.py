"""Поля зрения по кластерам: детекторы над плоскостью картинки, построенные по §6.4.

Статья строит детекторы над разложенным кодовым пространством: их рецептивные поля не задаются заранее, а
находятся по кластерам похожих соседних точек (§6.4.1–§6.4.5). Элементы картинки уже лежат на плоскости — на
своих местах, — поэтому тот же алгоритм применяется к самой картинке: поле зрения — кластер похожих соседних
элементов, его центр и радиус получаются из данных, а не из решётки.

Для каждой точки-центра c̃ (§6.4):

    1. активация A_λd(c̃, r_a): точки не дальше r_a от c̃ со сходством с кодом c̃ не ниже λ_d (§6.4.1);
    2. кластер DBSCAN, содержащий c̃ (§6.4.1);
    3. центр — среднее координат точек кластера с весом «активация × энергия» (§6.4.2);
    4. радиус — максимум f(p) = |{q : r(q) ≤ r(p)}| / (π r(p)²) среди радиусов, вмещающих не меньше min_points
       точек кластера (§6.4.3);
    5. вставка: центры детекторов не попадают в поля друг друга, при перекрытии остаётся более плотный (§6.4.5).

Отличие от §6.4: центры перебираются не случайно, а все по разу, и кандидаты вставляются от плотных к редким.
Случайный перебор с заменой менее плотных сходится к тому же набору: самый плотный кандидат вытесняет всех, кого
перекрывает, и сам уже не вытесняется. Перебор всех точек делает результат детерминированным и не зависящим от
места картинки в кадре.

Плотная расстановка (dense) — по продолжению статьи (docs/DAMP_continuation.md, П.2, П.3): поле зрения — окно
внимания миниколонки, окно стоит на месте своей миниколонки и смещается плавно, поля соседних миниколонок сильно
перекрываются. Поэтому окно получает каждый элемент: центр — сам элемент (шаг 3 не нужен, место выхода задаёт
геометрия), радиус — по шагу 4 вокруг этого центра, а вытеснения перекрывающихся полей (шаг 5) нет. Если кластер
элемента найден на нескольких слоях, остаётся более плотное поле — тем же правилом n_d / r_d, что у вставки.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from damp.article_refs import CONTINUATION_MINICOLUMNS, DETECTOR_CONSTRUCTION, DETECTOR_INSERTION, OUTSIDE_ARTICLE
from damp.encoding.keyhole import KeyholePieces
from damp.logging import LOGGER


@dataclass(frozen=True)
class PlaneFields:
    """Поля зрения набора картинок: куски (элементы каждого поля от его центра в долях радиуса) и размеры полей."""

    pieces: KeyholePieces
    radius: "np.ndarray"  # (F,) радиус поля в единицах плоскости
    points: "np.ndarray"  # (F,) число точек поля с энергией не ниже порога (n_d, §6.3)
    covered: float  # доля элементов, попавших хотя бы в одно поле

    @property
    def count(self) -> int:
        return self.pieces.count


class PlaneFieldBuilder:
    """Поля зрения над плоскостью элементов по §6.4: кластер похожих соседей, центр, радиус, вставка."""

    # Смещение элемента от центра поля — в долях радиуса поля, с этим шагом (число различных мест конечно).
    OFFSET_STEP = 0.125

    def __init__(
        self,
        *,
        lambda_thresholds: "float | tuple[float, ...] | list[float]",
        activation_radius: float,
        eps: float,
        min_samples: int,
        min_points: int,
        radius_min: float,
        energy_radius: float,
        energy_threshold: float,
        dense: bool = False,
        name: str = "fields",
    ) -> None:
        # True — окно на каждом элементе, без вытеснения перекрывающихся полей (продолжение статьи, П.2).
        self._dense = bool(dense)
        thresholds = (lambda_thresholds,) if isinstance(lambda_thresholds, (int, float)) else tuple(lambda_thresholds)
        if not thresholds or any(not 0.0 <= float(value) <= 1.0 for value in thresholds):
            raise ValueError("lambda thresholds must be in [0, 1]")
        if activation_radius <= 0 or eps <= 0 or radius_min <= 0 or energy_radius <= 0:
            raise ValueError("radii must be positive")
        if min_samples < 1 or min_points < 1:
            raise ValueError("min_samples and min_points must be positive")
        # Слои иерархии (§6.4, §7.1.10): у каждого слоя свой порог λ_d; слои заполняются независимо.
        self._thresholds = tuple(sorted({float(value) for value in thresholds}))
        self._lambda = self._thresholds[0]
        self._activation_radius = float(activation_radius)
        self._eps = float(eps)
        self._min_samples = int(min_samples)
        self._min_points = int(min_points)
        self._radius_min = float(radius_min)
        self._energy_radius = float(energy_radius)
        self._energy_threshold = float(energy_threshold)
        self._name = str(name)

    # ------------------------------------------------------------------ одна картинка
    def _clusters(self, disc: "np.ndarray", near: "np.ndarray") -> "np.ndarray":
        """Кластеры DBSCAN всех точек-центров разом (§6.4.1): member[…, c, j] — точка j в кластере центра c.

        disc[…, c, j] — точка j активирована центром c (у каждого центра свой набор активированных точек), near —
        матрица соседства точек (расстояние не больше eps). Впереди у disc могут стоять оси пакета (слои порогов):
        кластеры считаются для всех слоёв сразу. Для каждого центра это тот же кластер, что дал бы обход от него:
        ядро — точка, у которой среди активированных не меньше min_samples соседей (считая её саму); кластер
        растёт от ядер к их соседям; центр-не-ядро начинает с первого соседнего ядра, а без такого — шум (пустая
        строка). Обход заменён умножением матриц: один шаг роста сразу для всех центров.
        """
        count = disc.shape[-1]
        seeds = np.arange(count)
        linked = near.astype(np.float32)
        core = ((disc.astype(np.float32) @ linked.T) >= self._min_samples) & disc
        around = near & core
        start = np.where(core[..., seeds, seeds], seeds, np.where(around.any(axis=-1), around.argmax(axis=-1), -1))
        valid = (start >= 0) & disc[..., seeds, seeds]
        where = np.nonzero(valid)
        member = np.zeros_like(disc)
        member[where + (start[valid],)] = True
        while True:
            grown = member | ((((member & core).astype(np.float32) @ linked) > 0) & disc)
            if np.array_equal(grown, member):
                break
            member = grown
        member[where + (where[-1],)] = True
        return member

    def picture_fields(self, positions: "np.ndarray", bits: "np.ndarray") -> list[tuple[float, float, float, int]]:
        """Поля одной картинки по всем слоям: (центр y, центр x, радиус, n_d); positions — (n, 2), bits — (n, L)."""
        positions = np.asarray(positions, dtype=np.float64)
        bits = np.asarray(bits, dtype=np.float32)
        if positions.shape[0] < self._min_points:
            return []
        ones = bits.sum(axis=1)
        similarity = (bits @ bits.T) / np.sqrt(np.maximum(ones[:, None] * ones[None, :], 1.0))
        distance = np.hypot(positions[:, None, 0] - positions[None, :, 0], positions[:, None, 1] - positions[None, :, 1])
        if self._dense:
            # Окно на каждом элементе: из полей, найденных для него на разных слоях, остаётся более плотное.
            best: dict[tuple[float, float], tuple[float, float, float, int]] = {}
            for fields in self._dense_fields(positions, ones, similarity, distance):
                for y, x, radius, points in fields:
                    held = best.get((y, x))
                    if held is None or points / radius > held[3] / held[2]:
                        best[(y, x)] = (y, x, radius, points)
            return [best[key] for key in sorted(best)]
        chosen: list[tuple[float, float, float, int]] = []
        seen: set[tuple[float, float, float]] = set()
        for threshold in self._thresholds:
            for field in self._layer_fields(positions, ones, similarity, distance, threshold):
                # Одно и то же поле, найденное на разных слоях, остаётся одно.
                key = (round(field[0], 6), round(field[1], 6), round(field[2], 6))
                if key not in seen:
                    seen.add(key)
                    chosen.append(field)
        return chosen

    def _activated(
        self, ones: "np.ndarray", similarity: "np.ndarray", distance: "np.ndarray", thresholds: "np.ndarray"
    ) -> tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:
        """Активация, энергия точек, живые точки и активированные диски для слоёв порогов разом (§6.4.1, §5.7).

        thresholds — пороги λ_d слоёв тем же типом, что сходства (сравнение идёт в точности сходств). Первая ось
        ответов — слой.
        """
        count = ones.shape[0]
        diagonal = np.arange(count)
        # sim_λ с жёстким порогом (§5.4, η = ∞); точка сама с собой — активация 1.
        activation = np.where(similarity[None] >= thresholds[:, None, None], similarity[None], 0.0).astype(np.float64)
        activation[:, diagonal, diagonal] = np.where(ones > 0, 1.0, 0.0)
        # Энергия точки (§5.7): Σ sim_λ / d по окрестности r_e, нормированная на максимум по картинке. В статье
        # точки стоят в ячейках, и d не меньше шага; здесь элементы могут лежать ближе шага друг к другу, поэтому
        # d берётся не меньше наименьшего радиуса поля — иначе две почти совпавшие точки забирают всю энергию.
        around = (distance <= self._energy_radius) & ~np.eye(count, dtype=bool)
        energy = np.where(around[None], activation / np.maximum(distance, self._radius_min)[None], 0.0).sum(axis=2)
        top = energy.max(axis=1)
        energy = np.where(top[:, None] > 0, energy / np.where(top > 0, top, 1.0)[:, None], 0.0)
        alive = energy >= self._energy_threshold
        # §6.4.1: активация кодом центра в радиусе r_a; точки с энергией ниже порога не участвуют.
        disc = (distance <= self._activation_radius)[None] & (activation > 0) & alive[:, None, :] & alive[:, :, None]
        return activation, energy, alive, disc

    def _layer_fields(
        self,
        positions: "np.ndarray",
        ones: "np.ndarray",
        similarity: "np.ndarray",
        distance: "np.ndarray",
        threshold: float,
    ) -> list[tuple[float, float, float, int]]:
        """Поля одного слоя иерархии (порог λ_d = threshold) на одной картинке, со вставкой §6.4.5."""
        layers = np.asarray([threshold], dtype=similarity.dtype)
        activation, energy, alive, disc = (item[0] for item in self._activated(ones, similarity, distance, layers))
        # Кластеры всех центров считаются разом.
        members = self._clusters(disc, distance <= self._eps)
        candidates: dict[tuple[float, float, float], tuple[float, int]] = {}
        for centre in np.flatnonzero(alive):
            cluster = np.flatnonzero(members[centre])
            if cluster.size < self._min_points:
                continue
            # §6.4.2: вес точки — произведение активации и энергии.
            weight = activation[centre, cluster] * energy[cluster]
            total = float(weight.sum())
            if total <= 0.0:
                continue
            middle = (positions[cluster] * weight[:, None]).sum(axis=0) / total
            # §6.4.3: радиус по максимуму f(p) = |{q: r(q) ≤ r(p)}| / (π r(p)²).
            reach = np.sort(np.hypot(positions[cluster, 0] - middle[0], positions[cluster, 1] - middle[1]))
            inside = np.arange(1, reach.size + 1, dtype=np.float64)
            radii = np.maximum(self._radius_min, reach)
            allowed = (radii <= self._activation_radius) & (inside >= self._min_points)
            if not allowed.any():
                continue
            fill = np.where(allowed, inside / (math.pi * radii * radii), -np.inf)
            radius = float(radii[int(np.argmax(fill))])
            # §6.3: n_d — точки поля с энергией не ниже порога.
            field = np.hypot(positions[:, 0] - middle[0], positions[:, 1] - middle[1]) <= radius
            points = int(np.count_nonzero(field & alive))
            if points < self._min_points:
                continue
            key = (round(float(middle[0]), 6), round(float(middle[1]), 6), round(radius, 6))
            candidates.setdefault(key, (points / radius, points))
        # §6.4.5: центры не попадают в поля друг друга; при перекрытии остаётся более плотный (n_d / r_d).
        ordered = sorted(candidates.items(), key=lambda item: -item[1][0])
        chosen: list[tuple[float, float, float, int]] = []
        for (y, x, radius), (_, points) in ordered:
            if any(math.hypot(y - other[0], x - other[1]) <= max(radius, other[2]) for other in chosen):
                continue
            chosen.append((y, x, radius, points))
        return chosen

    def _dense_fields(
        self, positions: "np.ndarray", ones: "np.ndarray", similarity: "np.ndarray", distance: "np.ndarray"
    ) -> list[list[tuple[float, float, float, int]]]:
        """Окна внимания картинки по каждому слою порогов: окно на каждом элементе (П.2), радиус — по кластеру (§6.4.3).

        Центр окна — сам элемент (П.2, П.3: окно стоит на месте своей миниколонки), вытеснения перекрывающихся
        полей (§6.4.5) нет. Все слои и все окна считаются одним проходом: расстояния от элемента до точек его
        кластера по возрастанию, заполнение f(p) = |{q: r(q) ≤ r(p)}| / (π r(p)²) и его максимум среди радиусов,
        вмещающих не меньше min_points точек кластера.
        """
        count = positions.shape[0]
        layers = np.asarray(self._thresholds, dtype=similarity.dtype)
        _, _, alive, disc = self._activated(ones, similarity, distance, layers)
        members = self._clusters(disc, distance <= self._eps)
        reach = np.sort(np.where(members, distance[None], np.inf), axis=2)
        inside = np.arange(1, count + 1, dtype=np.float64)[None, None, :]
        radii = np.maximum(self._radius_min, reach)
        allowed = np.isfinite(reach) & (radii <= self._activation_radius) & (inside >= self._min_points)
        fill = np.where(allowed, inside / (math.pi * radii * radii), -np.inf)
        found = allowed.any(axis=2)
        radius = np.where(found, np.take_along_axis(radii, fill.argmax(axis=2)[:, :, None], axis=2)[:, :, 0], 0.0)
        # §6.3: n_d — точки поля с энергией не ниже порога.
        points = ((distance[None] <= radius[:, :, None]) & alive[:, None, :]).sum(axis=2)
        valid = alive & found & (members.sum(axis=2) >= self._min_points) & (points >= self._min_points)
        return [
            [
                (
                    round(float(positions[row, 0]), 6),
                    round(float(positions[row, 1]), 6),
                    round(float(radius[layer, row]), 6),
                    int(points[layer, row]),
                )
                for row in np.flatnonzero(valid[layer])
            ]
            for layer in range(layers.shape[0])
        ]

    # ------------------------------------------------------------------ набор картинок
    def build(
        self,
        picture: "np.ndarray",
        y: "np.ndarray",
        x: "np.ndarray",
        packed: "np.ndarray",
        index: "np.ndarray",
        code_length: int,
    ) -> PlaneFields:
        """Поля всех картинок: picture — номер картинки элемента (по возрастанию), y, x — его место, packed и
        index — различные коды элементов и номер кода у каждого элемента."""
        picture = np.asarray(picture, dtype=np.int64)
        position = np.stack([np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)], axis=1)
        index = np.asarray(index, dtype=np.int64)
        if picture.ndim != 1 or position.shape[0] != picture.shape[0] or index.shape != picture.shape:
            raise ValueError("picture, y, x and index must be one-dimensional arrays of the same length")
        if picture.size and np.any(np.diff(picture) < 0):
            raise ValueError("elements must be ordered by picture")
        table = np.unpackbits(np.ascontiguousarray(packed, dtype=np.uint8), axis=1, count=int(code_length))
        pictures, starts = np.unique(picture, return_index=True)
        bounds = np.append(starts, picture.shape[0])
        field_picture: list[int] = []
        centres: list[tuple[float, float]] = []
        radii: list[float] = []
        points: list[int] = []
        owner: list["np.ndarray"] = []
        members: list["np.ndarray"] = []
        shifts: list["np.ndarray"] = []
        covered = np.zeros(picture.shape[0], dtype=bool)
        for number, first, last in zip(pictures, bounds[:-1], bounds[1:]):
            own = position[first:last]
            for centre_y, centre_x, radius, inside in self.picture_fields(own, table[index[first:last]]):
                reach = np.hypot(own[:, 0] - centre_y, own[:, 1] - centre_x)
                rows = np.flatnonzero(reach <= radius)
                owner.append(np.full(rows.size, len(radii), dtype=np.int64))
                members.append(rows + first)
                # Место элемента — смещение от центра поля в долях его радиуса: поле любого размера кодируется
                # одними и теми же детекторами места.
                shifts.append((own[rows] - (centre_y, centre_x)) / radius)
                covered[rows + first] = True
                field_picture.append(int(number))
                centres.append((centre_y, centre_x))
                radii.append(radius)
                points.append(inside)
        if radii:
            shift = np.round(np.concatenate(shifts) / self.OFFSET_STEP) * self.OFFSET_STEP
            offsets, offset = np.unique(shift, axis=0, return_inverse=True)
            pieces = KeyholePieces(
                offsets=offsets.astype(np.float64),
                picture=np.asarray(field_picture, dtype=np.int64),
                centre=np.asarray(centres, dtype=np.float64),
                piece=np.concatenate(owner),
                element=np.concatenate(members),
                offset=offset.reshape(-1).astype(np.int64),
            )
        else:
            pieces = KeyholePieces(
                offsets=np.zeros((0, 2), dtype=np.float64),
                picture=np.zeros(0, dtype=np.int64),
                centre=np.zeros((0, 2), dtype=np.float64),
                piece=np.zeros(0, dtype=np.int64),
                element=np.zeros(0, dtype=np.int64),
                offset=np.zeros(0, dtype=np.int64),
            )
        fields = PlaneFields(
            pieces=pieces,
            radius=np.asarray(radii, dtype=np.float64),
            points=np.asarray(points, dtype=np.int64),
            covered=float(covered.mean()) if covered.size else 0.0,
        )
        sizes = np.bincount(pieces.piece, minlength=pieces.count) if pieces.count else np.zeros(0, dtype=np.int64)
        LOGGER.event(
            "plane_fields.build",
            section=DETECTOR_CONSTRUCTION,
            data={
                "name": self._name,
                "pictures": int(pictures.shape[0]),
                "elements": int(picture.shape[0]),
                "fields": fields.count,
                "fields_per_picture": float(fields.count) / max(1, int(pictures.shape[0])),
                "elements_per_field": float(sizes.mean()) if sizes.size else 0.0,
                "radius_mean": float(fields.radius.mean()) if fields.count else 0.0,
                "radius_min": float(fields.radius.min()) if fields.count else 0.0,
                "radius_max": float(fields.radius.max()) if fields.count else 0.0,
                "covered_elements": fields.covered,
                "layers": len(self._thresholds),
                "lambda_levels": list(self._thresholds),
                "activation_radius": self._activation_radius,
                "dbscan_eps": self._eps,
                "dbscan_min_samples": self._min_samples,
                "min_points": self._min_points,
                "note": "поля зрения — детекторы над плоскостью картинки по §6.4: кластер похожих соседних элементов "
                "(§6.4.1), взвешенный центр (§6.4.2), радиус по максимуму заполнения (§6.4.3)",
            },
        )
        if self._dense:
            LOGGER.event(
                "plane_fields.keyholes",
                section=CONTINUATION_MINICOLUMNS,
                data={
                    "name": self._name,
                    "keyholes": fields.count,
                    "keyholes_per_element": float(fields.count) / max(1, int(picture.shape[0])),
                    "points_mean": float(fields.points.mean()) if fields.count else 0.0,
                    "note": "окна внимания миниколонок (П.2): окно на каждом элементе, центр — сам элемент, радиус — "
                    "по кластеру его похожих соседей (§6.4.3); поля соседних миниколонок сильно перекрываются, "
                    "вытеснения §6.4.5 нет",
                },
            )
            return fields
        LOGGER.event(
            "plane_fields.insertion",
            section=DETECTOR_INSERTION,
            data={
                "name": self._name,
                "fields": fields.count,
                "points_mean": float(fields.points.mean()) if fields.count else 0.0,
                "note": "центры полей не попадают в поля друг друга; при перекрытии остаётся более плотное (§6.4.5); "
                "центры перебраны все, кандидаты вставлены от плотных к редким — устойчивое состояние процесса §6.4",
            },
        )
        return fields


class NeighbourScale:
    """Масштаб плоскости элементов (вне статьи): расстояние до ближайшего соседа и сходство соседних кодов.

    Радиусы построения полей (окрестность DBSCAN, радиус активации, наименьший радиус поля) задаются в шагах
    между соседними элементами, а порог сходства — квантилем сходства соседних элементов: на каждом уровне
    элементы стоят со своим шагом и их коды сходны по-своему.
    """

    def __init__(self, *, eps_factor: float, sample: int = 400) -> None:
        self._eps_factor = float(eps_factor)
        self._sample = int(sample)

    def measure(
        self,
        picture: "np.ndarray",
        y: "np.ndarray",
        x: "np.ndarray",
        packed: "np.ndarray",
        index: "np.ndarray",
        code_length: int,
        quantiles: "tuple[float, ...] | list[float]",
        *,
        name: str,
    ) -> tuple[float, list[float]]:
        """Медиана расстояния до ближайшего соседа и квантили сходства кодов соседних элементов (по части картинок)."""
        picture = np.asarray(picture, dtype=np.int64)
        position = np.stack([np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)], axis=1)
        table = np.unpackbits(np.ascontiguousarray(packed, dtype=np.uint8), axis=1, count=int(code_length)).astype(np.float32)
        pictures, starts = np.unique(picture, return_index=True)
        bounds = np.append(starts, picture.shape[0])
        nearest: list["np.ndarray"] = []
        spans: list[tuple["np.ndarray", "np.ndarray"]] = []
        for first, last in list(zip(bounds[:-1], bounds[1:]))[: self._sample]:
            if last - first < 2:
                continue
            own = position[first:last]
            distance = np.hypot(own[:, None, 0] - own[None, :, 0], own[:, None, 1] - own[None, :, 1])
            np.fill_diagonal(distance, np.inf)
            nearest.append(distance.min(axis=1))
            bits = table[np.asarray(index[first:last], dtype=np.int64)]
            ones = bits.sum(axis=1)
            similarity = (bits @ bits.T) / np.sqrt(np.maximum(ones[:, None] * ones[None, :], 1.0))
            spans.append((distance, similarity))
        if not nearest:
            return 1.0, [1.0 for _ in quantiles]
        step = float(np.median(np.concatenate(nearest)))
        step = step if step > 0 else 1.0
        close = np.concatenate([similarity[distance <= self._eps_factor * step] for distance, similarity in spans])
        levels = [float(np.quantile(close, float(q))) if close.size else 1.0 for q in quantiles]
        LOGGER.event(
            "plane_fields.scale",
            section=OUTSIDE_ARTICLE,
            data={
                "name": name,
                "pictures": len(nearest),
                "neighbour_step": step,
                "neighbour_pairs": int(close.size),
                "similarity_quantiles": [float(q) for q in quantiles],
                "similarity_levels": levels,
                "similarity_mean": float(close.mean()) if close.size else 0.0,
                "note": "шаг между соседними элементами и сходство их кодов: в них задаются радиусы и пороги "
                "слоёв построения полей зрения",
            },
        )
        return step, levels
