"""Этап 1: точки середины линии — след ручки на бумаге вместо двух краёв (вне статьи; доработка).

Замысел автора проекта (2026-11-17): «ходить не вдоль края штриха, а в точке касания ручки бумаги. У нас будет
вместо двух краёв — одна середина… мозгу не важна толщина штриха, ему важны его изгибы, перекрестия и прерывы».
Всё остальное — как было: над точками строятся те же детекторы штрихов (StrokeDetectors, §6.4 на плоскости
картинки), только на вход идут точки середины линии, а не точки двух её краёв.

В статье и её продолжении этого нет. Подсказка из устройства зрения: простые клетки первичной зрительной коры
бывают двух видов — на край и на полосу (Хьюбел и Визель); середина линии — отклик клеток-«полос».

Середина линии — гребень расстояния до края линии: точка линии, равноудалённая от двух её краёв.

    1. чернила — пиксели рабочей картинки не темнее level (те же, по которым ищется контур);
    2. у каждого пикселя чернил — расстояние до ближайшего пикселя фона. Поперёк линии оно растёт от края к середине
       и падает к другому краю «шатром» с одним и тем же наклоном при любой толщине линии: сила гребня от толщины
       не зависит;
    3. вторые производные расстояния (гессиан, сглаживание σ) дают направление поперёк линии и силу гребня;
    4. точка — на гребне, если расстояние в ней не меньше, чем в соседних точках поперёк линии (у линии чётной
       ширины две равные точки посередине — берётся одна: знак направления поперёк линии один на всю линию);
    5. линия, а не пятно: вдоль линии расстояние почти не гнётся (у конца линии и в пятне оно гнётся во все стороны);
    6. не кончик: вдоль линии расстояние не падает (за концом линии гребень тянулся бы «хвостом» к её краю).

Направление точки — поперёк линии, без стрелки: у середины нет «внутри» и «снаружи», как у края.

Разрывы. Условия 5 и 6 рвут середину на перекрестиях и стыках линий: там расстояние до края — не гребень, а бугор.
Гребень находит середину точно, но связности не обещает — известное свойство (Latecki и др., skeleton strength maps);
куски соединяют заново, как линии у Стегера и края у Кэнни (два порога) и как в скелетах по вершинам расстояния
(maxima tracking):

    7. слабые точки — точки гребня (условие 4), отброшенные условиями 5, 6 или порогом силы, но не слабее weak_share;
    8. мостик — кратчайший путь по слабым точкам между двумя кусками середины, не длиннее bridge. Слабая точка идёт в
       середину, только если лежит на мостике: «веер» у кончика линии примыкает к одному куску и не возвращается.
       Три и больше кусков у одного стыка соединяются мостиками через общие слабые точки в его середине;
    9. общий центр стыка. Где линия упирается в линию сбоку, слабых точек между кусками может не быть. В стыке
       расстояние до края имеет вершину — центр вписанного в стык круга. Если рядом с вершиной (в двух с половиной
       радиусах этого круга) кончаются куски, ещё не соединённые между собой, каждый соединяется с вершиной отрезком: два
       куска, три или четыре сходятся в общем центре;
    10. разрыв внутри петли. У восьмёрки, нуля, шестёрки и девятки оба конца разрыва принадлежат одному куску:
       петля замкнута с другой стороны. Поэтому соединяются не только разные куски, но и конец линии с другой её
       частью, до которой по самой линии далеко (путь по линии длиннее разрыва в LOOP раз): по слабым точкам, как в
       п. 8, а где их нет — отрезком по чернилам, не длиннее стыка (п. 9): к середине другой линии — если отрезок
       продолжает ход своей (не вбок и не назад), к другому концу — всегда (у острого угла концы сходятся к вершине
       под углом друг к другу);
    11. направление точек мостика — поперёк самого мостика: через середину перекрестия идут мостики двух направлений,
       и детекторы штрихов ставят там два штриха — по одному на линию.

Что пробовали и отвергли (2026-11-17, стенд подбора: сходимость классов на последнем уровне, 100 цифр):
    гребень яркости сглаженной картинки — у жирной линии профиль яркости плоский, середина терялась;
    середины пар встречных точек края (идея автора: «просто брать середину — там же всё векторно») — линия ровнее, но
    у стыка линий напротив края нет встречного края на всю толщину перекладины, стыки и перекрестия вырезаются:
    0.358 против 0.429 у гребня расстояния, хуже на всех трёх уровнях.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from damp.article_refs import OUTSIDE_ARTICLE
from damp.logging import LOGGER


@dataclass(frozen=True)
class StrokeMidline:
    """Точки середины линии на рабочей картинке — гребень расстояния до края линии (вне статьи; доработка).

    Все длины — в пикселях рабочей картинки.

    Args:
        level: чернила — пиксели не темнее этого (как у контура, MNIST_CONTOUR_LEVEL).
        sigma: сглаживание расстояния перед вторыми производными.
        strength_share: точка на гребне, если сила гребня не меньше этой доли от наибольшей на картинке.
        blob: «линия, а не пятно» — изгиб расстояния вдоль линии не больше этой доли изгиба поперёк.
        tip: «не кончик» — наклон расстояния вдоль линии не больше этой доли от (изгиб поперёк × σ).
        depth: середина — не ближе этого к краю линии: линия тоньше двух таких расстояний середины не даёт.
        weak_share: слабая точка гребня — сила не меньше этой доли от наибольшей (меньше strength_share).
        bridge: наибольшая длина мостика по слабым точкам между двумя кусками середины; 0 — разрывы не соединяются.
    """

    level: float
    sigma: float
    strength_share: float
    blob: float
    tip: float
    depth: float = 1.5
    weak_share: float = 0.0
    bridge: float = 0.0

    # Куски кончаются у стыка не дальше стольких радиусов вписанного в стык круга от его центра: условие «не пятно»
    # обрывает гребень там, где начинается бугор стыка, — примерно в толщине линии от центра.
    JUNCTION = 2.5
    # Наклон, прибавляемый к расстоянию на пиксель: выбирает один из двух равных пикселей у вершины (условие 4).
    TIE = 1e-3
    # Соседние пиксели — касающиеся стороной или углом; RING — они же по кругу.
    AROUND = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    RING = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
    # Разрыв внутри петли: конец линии и другая её часть разорваны, если путь между ними по самой линии длиннее
    # разрыва во столько раз (плюс запас в пикселях) — линия обходит петлю, а не идёт рядом.
    LOOP = 3.0
    LOOP_SLACK = 6
    # Отрезок от конца линии продолжает её ход: косинус угла между ходом линии у конца и отрезком — не меньше этого.
    TURN = 0.3
    # Ход линии у конца — от точек, отстоящих от конца на столько шагов вдоль линии, к самому концу.
    TANGENT = 6

    def __post_init__(self) -> None:
        if self.sigma <= 0 or self.depth < 0:
            raise ValueError("sigma must be positive, depth non-negative")
        if not 0.0 < self.strength_share <= 1.0:
            raise ValueError("strength_share must be in (0, 1]")
        if self.blob <= 0 or self.tip <= 0:
            raise ValueError("blob and tip must be positive")
        if not 0.0 <= self.weak_share <= self.strength_share or self.bridge < 0:
            raise ValueError("weak_share must be in [0, strength_share], bridge non-negative")

    @property
    def signature(self) -> str:
        """Подпись способа для кеша штрихов и раскладки."""
        return (
            f"midline(level={self.level:g},sigma={self.sigma:g},strength={self.strength_share:g},blob={self.blob:g},"
            f"tip={self.tip:g},depth={self.depth:g},weak={self.weak_share:g},bridge={self.bridge:g})"
        )

    def points(self, work: "np.ndarray") -> tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:
        """Точки середины линии: место центра пикселя (y, x) и вектор поперёк линии (vx, vy), ось Y вверх.

        Длина вектора — сила гребня; знак вектора произволен — направление без стрелки.
        """
        from scipy import ndimage

        ink = np.asarray(work) >= float(self.level)
        empty = (np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))
        if not bool(ink.any()):
            return empty
        distance = ndimage.distance_transform_edt(ink)
        sigma = float(self.sigma)
        smooth = ndimage.gaussian_filter(distance, sigma)
        dxx = ndimage.gaussian_filter(distance, sigma, order=(0, 2))
        dyy = ndimage.gaussian_filter(distance, sigma, order=(2, 0))
        dxy = ndimage.gaussian_filter(distance, sigma, order=(1, 1))
        mean, root = (dxx + dyy) / 2.0, np.sqrt(((dxx - dyy) / 2.0) ** 2 + dxy**2)
        across, along = mean - root, mean + root  # изгиб поперёк линии (самый отрицательный) и вдоль неё
        # Направление поперёк линии — собственный вектор гессиана для across; из двух записей — более устойчивая.
        ax, ay = dxy, across - dxx
        bx, by = across - dyy, dxy
        second = (bx * bx + by * by) > (ax * ax + ay * ay)
        nx, ny = np.where(second, bx, ax), np.where(second, by, ay)
        norm = np.hypot(nx, ny)
        norm[norm == 0.0] = 1.0
        nx, ny = nx / norm, ny / norm
        # Ничья. У линии чётной ширины вершина расстояния лежит между двумя пикселями с равным расстоянием, и
        # условие 4 оставляло из них то оба, то ни одного — середина двоилась или рвалась: знак направления у
        # собственного вектора произволен, а соседние точки поперёк линии берутся между пикселями, с погрешностью.
        # Поэтому знак направления — один на всю линию (большая составляющая положительна), а к расстоянию
        # прибавлен наклон TIE на пиксель: из двух равных пикселей он выбирает один и тот же, а настоящую вершину
        # не сдвигает — соседи поперёк линии отличаются от неё на десятые доли пикселя.
        flip = np.where(np.abs(nx) >= np.abs(ny), nx < 0.0, ny < 0.0)
        nx, ny = np.where(flip, -nx, nx), np.where(flip, -ny, ny)
        rows, cols = np.mgrid[0 : distance.shape[0], 0 : distance.shape[1]].astype(np.float64)
        tilted = smooth + self.TIE * (rows + 0.618 * cols)
        ahead = ndimage.map_coordinates(tilted, [rows + ny, cols + nx], order=1, mode="nearest")
        behind = ndimage.map_coordinates(tilted, [rows - ny, cols - nx], order=1, mode="nearest")
        strength = np.maximum(-across, 0.0)
        top = float(strength.max())
        if top <= 0.0:
            return empty
        slope_x = ndimage.gaussian_filter(distance, sigma, order=(0, 1))
        slope_y = ndimage.gaussian_filter(distance, sigma, order=(1, 0))
        slope_along = np.abs(slope_x * (-ny) + slope_y * nx)
        crest = (tilted >= ahead) & (tilted > behind) & (smooth >= float(self.depth))
        ridge = (
            crest
            & (strength >= float(self.strength_share) * top)
            & (np.abs(along) <= float(self.blob) * strength)
            & (slope_along <= float(self.tip) * strength * sigma)
        )
        found_rows, found_cols = np.nonzero(ridge)
        power = strength[found_rows, found_cols]
        vector_x, vector_y = nx[found_rows, found_cols] * power, ny[found_rows, found_cols] * power
        bridges = 0
        if self.bridge > 0 and found_rows.shape[0]:
            # Мостики через разрывы: точки мостика — с направлением поперёк мостика и средней силой точек гребня.
            weak = crest & ~ridge & (strength >= float(self.weak_share) * top)
            paths = self._bridges(ridge, weak, smooth, ink)
            bridges = len(paths)
            typical = float(np.median(power))
            for path in paths:
                tangent = path[np.minimum(np.arange(path.shape[0]) + 2, path.shape[0] - 1)] - path[np.maximum(np.arange(path.shape[0]) - 2, 0)]
                length = np.hypot(tangent[:, 0], tangent[:, 1])
                # Концы пути — точки самих кусков середины (или повтор вершины) — в середину не идут; точка, у которой
                # путь стоит на месте, направления не имеет и тоже не идёт.
                inner = np.arange(1, path.shape[0] - 1)
                inner = inner[length[inner] > 0.0]
                found_rows = np.concatenate([found_rows, path[inner, 0]])
                found_cols = np.concatenate([found_cols, path[inner, 1]])
                # Поперёк пути (столбцы, строки): касательная (t_c, t_r) → нормаль (−t_r, t_c).
                vector_x = np.concatenate([vector_x, -tangent[inner, 0] / length[inner] * typical])
                vector_y = np.concatenate([vector_y, tangent[inner, 1] / length[inner] * typical])
        LOGGER.event(
            "strokes.midline",
            section=OUTSIDE_ARTICLE,
            data={
                "points": int(found_rows.shape[0]),
                "bridges": bridges,
                "ink_pixels": int(ink.sum()),
                "half_width_max": float(distance.max()),
                "method": self.signature,
                "note": "точки середины линии — гребень расстояния до края линии (доработка: след ручки вместо двух "
                "краёв; в зрении — простые клетки-«полосы»); направление — поперёк линии, без стрелки",
            },
        )
        # Ось Y — вверх, как у отклика Собеля: строки картинки идут вниз.
        return found_rows + 0.5, found_cols + 0.5, vector_x, -vector_y

    def _bridges(
        self, ridge: "np.ndarray", weak: "np.ndarray", smooth: "np.ndarray", ink: "np.ndarray"
    ) -> list["np.ndarray"]:
        """Мостики через разрывы середины, строки — (строка, столбец) пикселей пути.

        Сначала — пути по слабым точкам гребня. От всех кусков разом расходится волна по слабым точкам (поиск в
        ширину): каждая слабая точка узнаёт свой ближайший кусок и точку, от которой до неё дошла волна. Где
        встречаются волны двух кусков, ещё не соединённых, — мостик: путь от одного куска до другого через точку
        встречи. Где встречаются волны одного куска, мостик строится, если одна из них вышла из конца линии, а по
        самой линии между началами волн далеко: это разрыв внутри петли. Длиннее bridge мостик не строится.
        Затем — общий центр стыка: вершина расстояния до края (smooth), рядом с которой кончаются куски, ещё не
        соединённые между собой; каждый соединяется с вершиной отрезком по чернилам (ink).
        Наконец — концы линии, оставшиеся рядом с другой её частью, до которой по линии далеко: отрезок по чернилам,
        продолжающий ход линии.

        Первый и последний пиксель пути в середину не идут — это точки самой линии (у отрезка к вершине вершина
        повторена): по ним у крайних точек мостика считается направление.
        """
        from collections import deque

        from scipy import ndimage

        piece, pieces = ndimage.label(ridge, structure=np.ones((3, 3), dtype=bool))
        if pieces == 0:
            return []
        height, width = ridge.shape
        limit = int(self.bridge)
        line = ridge.copy()  # середина вместе с уже построенными мостиками
        tip = self._ends(ridge)
        owner = np.zeros((height, width), dtype=np.int32)  # кусок, чья волна дошла до слабой точки первой
        reach = np.zeros((height, width), dtype=np.int32)  # сколько шагов от куска
        back = np.full((height, width, 2), -1, dtype=np.int32)  # откуда пришла волна; у первой точки — точка куска
        origin = np.full((height, width), -1, dtype=np.int64)  # начало волны: точка куска (строка × ширина + столбец)
        group = list(range(pieces + 1))  # какие куски уже соединены
        tried: set[tuple[int, int]] = set()  # пары начал волн одного куска, уже проверенные на разрыв петли

        def joined(index: int) -> int:
            while group[index] != index:
                group[index] = group[group[index]]
                index = group[index]
            return index

        def trail(row: int, col: int) -> list[tuple[int, int]]:
            """Путь от слабой точки назад до точки её куска."""
            path = [(row, col)]
            while not ridge[row, col]:
                row, col = int(back[row, col, 0]), int(back[row, col, 1])
                path.append((row, col))
            return path

        def along(row: int, col: int, depth: int, target: tuple[int, int] | None = None) -> dict[tuple[int, int], int]:
            """Шаги вдоль линии от точки до других её точек, не дальше depth; с target — только до встречи с ним."""
            seen = {(row, col): 0}
            wave: deque = deque([(row, col)])
            while wave:
                at_row, at_col = wave.popleft()
                steps = seen[(at_row, at_col)]
                if steps >= depth:
                    continue
                for d_row, d_col in self.AROUND:
                    near = (at_row + d_row, at_col + d_col)
                    if 0 <= near[0] < height and 0 <= near[1] < width and line[near] and near not in seen:
                        seen[near] = steps + 1
                        if near == target:
                            return seen
                        wave.append(near)
            return seen

        def apart(first: int, second: int, gap: int) -> bool:
            """Разрыв петли: одна из двух точек линии — её конец, и по самой линии между ними далеко."""
            if first == second or not (tip.flat[first] or tip.flat[second]):
                return False
            pair = (min(first, second), max(first, second))
            if pair in tried:
                return False
            tried.add(pair)
            target = divmod(second, width)
            return target not in along(*divmod(first, width), int(self.LOOP * gap) + self.LOOP_SLACK, target)

        def add(path: "np.ndarray") -> None:
            paths.append(path)
            line[path[:, 0], path[:, 1]] = True

        queue: deque = deque()
        paths: list["np.ndarray"] = []
        rows, cols = np.nonzero(weak)
        for row, col in zip(rows.tolist(), cols.tolist()):
            for d_row, d_col in self.AROUND:
                near_row, near_col = row + d_row, col + d_col
                if not (0 <= near_row < height and 0 <= near_col < width) or not ridge[near_row, near_col]:
                    continue
                mine = int(piece[near_row, near_col])
                if owner[row, col] == 0:
                    owner[row, col], reach[row, col] = mine, 1
                    back[row, col] = (near_row, near_col)
                    origin[row, col] = near_row * width + near_col
                    queue.append((row, col))
                elif joined(int(owner[row, col])) != joined(mine) or apart(
                    int(origin[row, col]), near_row * width + near_col, 2
                ):
                    # Слабая точка касается двух кусков сразу (или двух сторон разрыва петли): мостик в один пиксель.
                    group[joined(mine)] = joined(int(owner[row, col]))
                    add(np.asarray([tuple(back[row, col]), (row, col), (near_row, near_col)], dtype=np.int64))
        while queue:
            row, col = queue.popleft()
            mine, steps = int(owner[row, col]), int(reach[row, col])
            if steps >= limit:
                continue
            for d_row, d_col in self.AROUND:
                near_row, near_col = row + d_row, col + d_col
                if not (0 <= near_row < height and 0 <= near_col < width) or not weak[near_row, near_col]:
                    continue
                if owner[near_row, near_col] == 0:
                    owner[near_row, near_col], reach[near_row, near_col] = mine, steps + 1
                    back[near_row, near_col] = (row, col)
                    origin[near_row, near_col] = origin[row, col]
                    queue.append((near_row, near_col))
                    continue
                gap = steps + int(reach[near_row, near_col])
                if gap > limit:
                    continue
                if joined(int(owner[near_row, near_col])) != joined(mine) or apart(
                    int(origin[row, col]), int(origin[near_row, near_col]), gap
                ):
                    group[joined(mine)] = joined(int(owner[near_row, near_col]))
                    add(np.asarray(trail(row, col)[::-1] + trail(near_row, near_col), dtype=np.int64))
        # Общий центр стыка: вершины расстояния до края, от высоких к низким.
        ridge_rows, ridge_cols = np.nonzero(ridge)
        ridge_piece = piece[ridge_rows, ridge_cols]
        peak = (smooth == ndimage.maximum_filter(smooth, size=7)) & ink & (smooth >= float(self.depth))
        peak_rows, peak_cols = np.nonzero(peak)
        for index in np.argsort(-smooth[peak_rows, peak_cols], kind="stable").tolist():
            row, col = int(peak_rows[index]), int(peak_cols[index])
            away = np.hypot(ridge_rows - row, ridge_cols - col)
            close = np.flatnonzero(away <= min(self.JUNCTION * float(smooth[row, col]) + 2.0, float(limit)))
            if close.size == 0:
                continue
            # У каждой группы уже соединённых кусков — её ближайшая к вершине точка.
            nearest: dict[int, int] = {}
            for point in close[np.argsort(away[close], kind="stable")].tolist():
                nearest.setdefault(joined(int(ridge_piece[point])), point)
            if len(nearest) < 2:
                continue
            links = []
            for point in nearest.values():
                if away[point] < 1.0:
                    links.append((point, None))  # кусок кончается в самой вершине: отрезок не нужен
                    continue
                steps = max(2, int(np.ceil(away[point])) + 1)
                line_rows = np.rint(np.linspace(ridge_rows[point], row, steps)).astype(np.int64)
                line_cols = np.rint(np.linspace(ridge_cols[point], col, steps)).astype(np.int64)
                if float(ink[line_rows, line_cols].mean()) >= 0.9:
                    links.append((point, np.stack([line_rows, line_cols], axis=1)))
            if len(links) < 2:
                continue
            first = joined(int(ridge_piece[links[0][0]]))
            for point, segment in links:
                group[joined(int(ridge_piece[point]))] = first
                if segment is not None:
                    add(np.concatenate([segment, segment[-1:]]))  # вершина повторена: она идёт в середину
        # Концы линии, оставшиеся рядом с другой её частью: по линии до неё далеко, а отрезок к ней идёт по чернилам
        # и продолжает ход линии. Дотягиваемся не дальше стыка — тех же JUNCTION полутолщин линии у конца.
        loose = self._ends(line)
        end_rows, end_cols = np.nonzero(loose)
        for row, col in zip(end_rows.tolist(), end_cols.tolist()):
            span = min(self.JUNCTION * float(smooth[row, col]) + 2.0, float(limit))
            seen = along(row, col, int(self.LOOP * span) + self.LOOP_SLACK)
            if sum(1 for steps in seen.values() if steps == 1) > 2:
                continue  # уже не конец: сюда пришёл мостик
            walked = min(self.TANGENT, max(seen.values()))
            heading = None
            if walked >= 2:
                behind = np.asarray([point for point, steps in seen.items() if steps == walked], dtype=np.float64)
                heading = np.asarray([row, col], dtype=np.float64) - behind.mean(axis=0)
                heading = heading / max(float(np.hypot(heading[0], heading[1])), 1e-9)
            line_rows, line_cols = np.nonzero(line)
            away = np.hypot(line_rows - row, line_cols - col)
            for index in np.argsort(away, kind="stable").tolist():
                gap = float(away[index])
                if gap > span:
                    break
                target = (int(line_rows[index]), int(line_cols[index]))
                if max(abs(target[0] - row), abs(target[1] - col)) < 2:
                    continue
                steps = seen.get(target)
                if steps is not None and steps <= self.LOOP * gap + self.LOOP_SLACK:
                    continue  # эта часть линии — рядом и по самой линии
                # К середине другой линии отрезок идёт, только продолжая ход своей (не вбок и не назад). Два конца
                # соединяются и без этого: у острого угла они сходятся к вершине под углом друг к другу.
                onward = ((target[0] - row) * heading[0] + (target[1] - col) * heading[1]) / gap if heading is not None else 1.0
                if onward < self.TURN and not loose[target]:
                    continue
                count = max(2, int(np.ceil(gap)) + 1)
                segment = np.stack(
                    [
                        np.rint(np.linspace(row, target[0], count)).astype(np.int64),
                        np.rint(np.linspace(col, target[1], count)).astype(np.int64),
                    ],
                    axis=1,
                )
                if float(ink[segment[:, 0], segment[:, 1]].mean()) < 0.9:
                    continue
                add(segment)
                break
        return paths

    @classmethod
    def _ends(cls, line: "np.ndarray") -> "np.ndarray":
        """Концы линии: точки, у которых соседей не больше двух и стоят они подряд — по одну сторону от точки."""
        height, width = line.shape
        padded = np.pad(line, 1)
        ring = [padded[1 + d_row : 1 + d_row + height, 1 + d_col : 1 + d_col + width] for d_row, d_col in cls.RING]
        count = sum(side.astype(np.int8) for side in ring)
        turns = sum((~ring[index] & ring[(index + 1) % 8]).astype(np.int8) for index in range(8))
        return line & (count <= 2) & (turns <= 1)
