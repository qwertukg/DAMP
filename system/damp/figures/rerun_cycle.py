"""Цикл в Rerun — «живой README»: вкладка на каждый этап 1–7 и обзор; шкала «cycle» — уровень (вне статьи: показ).

Вкладки заполняются по ходу счёта. Часть прохода уровня (окна, коды, карта, детекторы, срабатывания, детали,
узнавание) рисуется, как только готова (LadderObserver.part). Внутри долгих шагов картинка обновляется не чаще раза
в RERUN_LIVE_INTERVAL секунд: карта раскладывается — точки разъезжаются, слои детекторов заполняются один за другим
(подписки LOGGER.watch). Какой этап идёт — видно в сводке прогона на вкладке «Обзор»: таблица этапов уровня
(✓ готово, ▶ идёт, ждёт). При повторе из кеша шагов части приходят те же, только быстрее; ходов раскладки и слоёв
тогда нет — шаги не считаются.

Весь путь показан на одной картинке — цифре LADDER_FIGURE_DIGIT (5), той же, что на рисунках README: её окна, код
окна-примера, его место на карте, сработавшие детекторы, детали, окна уровней. Ещё по одной проверочной цифре каждого
класса (RERUN_EXAMPLE_DIGITS).

Схема данных устойчива — на неё опирается будущий свой UI (читать .rrd: rerun.dataframe):

    шкала cycle: 0 — стимулы и уровень 0 (первичная раскладка штрихов, LADDER_LEVEL0), 1…N — уровни цикла;
                 верхний уровень несёт и распознавание
    /pipeline                         цвета и подписи цифр 0–9 (AnnotationContext)
    /pipeline/overview/...            сводка прогона (пока он идёт — что считается сейчас), секунды шагов
    /pipeline/s1_stimuli/...          этап 1: цифра-пример и её штрихи, цифры 0–9, штрихов на картинку
    /pipeline/s2_coding/...           этап 2: детекторы угла и места, окна пятёрки, окно-пример и его код «место | содержание»
    /pipeline/s3_layout/...           этап 3: карта памяти уровня (по ходу раскладки, затем итог; цвет — цифра), окно-пример
                                      на карте (map/five), энергия точек, карта окон, качество Ē по ходу раскладки (live/quality)
    /pipeline/s4_detectors/...        этап 4: детекторы по слоям (заполняются по ходу); окно зажигает точки, срабатывают детекторы
    /pipeline/s5_embeddings/...       этап 5: код детали окна-примера, детали картинки, уровни срабатываний
    /pipeline/s6_cycle/...            этап 6: окна уровней вокруг одного места, отметки начала и конца цикла
    /pipeline/s7_recognition/...      этап 7: точность по уровням (сразу после уровня), карта верхних деталей, голоса, ошибки 10×10
    /pipeline/<этап>/stats/<мерка>    числа уровня (Scalars на шкале cycle)
    /pipeline/<этап>/about            что показывает вкладка и числа уровня (Markdown)
"""
from __future__ import annotations

import colorsys
import math
import time
from types import SimpleNamespace
from typing import Callable

import numpy as np

from damp import settings
from damp.article_refs import OUTSIDE_ARTICLE
from damp.cycle.ladder_trace import LEVEL_PARTS, LadderLevelTrace, LadderObserver, LadderRunTrace, LadderTopTrace
from damp.logging import LOGGER

ROOT = "pipeline"
OVERVIEW, STIMULI, CODING, LAYOUT = f"{ROOT}/overview", f"{ROOT}/s1_stimuli", f"{ROOT}/s2_coding", f"{ROOT}/s3_layout"
DETECTORS, EMBEDDINGS, CYCLE, RECOGNITION = f"{ROOT}/s4_detectors", f"{ROOT}/s5_embeddings", f"{ROOT}/s6_cycle", f"{ROOT}/s7_recognition"
TIMELINE = LOGGER.CYCLE_TIMELINE

# Цвета (те же, что у рисунков README): место — оранжевый, содержание — синий, деталь — бирюзовый.
PLACE, CONTENT, DETAIL, OFF = (247, 150, 70), (90, 160, 250), (70, 210, 200), (44, 46, 52)
KEYHOLE, LIT_LOW, LIT_HIGH, FIRED, STRONG = (90, 220, 120), (40, 96, 170), (170, 215, 255), (110, 62, 40), (247, 150, 70)
KEYHOLE_FAINT, FIVE = (50, 120, 70), (255, 255, 255)
LEVEL_COLOURS = ((90, 220, 120), (247, 150, 70), (190, 120, 250), (90, 160, 250), (240, 90, 90), (70, 210, 200))
LEVEL0_COLOUR = (200, 200, 200)  # окно уровня 0: вокруг центра цифры, в нём один штрих


def hue_colour(hue: float, value: float = 1.0) -> tuple[int, int, int]:
    red, green, blue = colorsys.hsv_to_rgb((float(hue) % 360.0) / 360.0, 0.85, value)
    return int(red * 255), int(green * 255), int(blue * 255)


def layer_colour(layer: int, layers: int) -> tuple[int, int, int]:
    """Цвет слоя детекторов: от широкого (слой 0) к узкому — по кругу оттенков, как на рисунках README."""
    return hue_colour(270.0 * layer / max(1, layers - 1), 0.95)


def circle(cy: float, cx: float, radius: float, points: int = 48) -> np.ndarray:
    """Окружность ломаной (x, y)."""
    turn = np.linspace(0.0, 2.0 * math.pi, points + 1)
    return np.stack([cx + radius * np.cos(turn), cy + radius * np.sin(turn)], axis=1)


def bits_image(bits: np.ndarray, columns: int, on: tuple[int, int, int], cell: int = 7) -> np.ndarray:
    """Биты кода растром: клетка на бит, зазор в пиксель; единица — цветом on."""
    bits = np.asarray(bits, dtype=bool)
    rows = max(1, -(-bits.shape[0] // columns))
    grid = np.zeros(rows * columns, dtype=np.int8) - 1
    grid[: bits.shape[0]] = bits
    grid = grid.reshape(rows, columns)
    image = np.full((rows * cell, columns * cell, 3), 18, dtype=np.uint8)
    for value, colour in ((0, OFF), (1, on)):
        ys, xs = np.nonzero(grid == value)
        for dy in range(cell - 1):
            for dx in range(cell - 1):
                image[ys * cell + dy, xs * cell + dx] = colour
    return image


def _num(value: object, spec: str = "") -> str:
    """Число для таблицы «Что видно»: ещё не посчитано — прочерк."""
    if value is None:
        return "—"
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def _share(values: dict | None) -> str:
    """Точность по k: «k=1: 61.2%, k=5: 60.4%»."""
    return ", ".join(f"k={k}: {v:.1%}" for k, v in (values or {}).items()) or "—"


class RerunSink:
    """Запись в Rerun: путь, данные и уровень цикла (None — статично, видно на любом уровне).

    live — кадр по ходу счёта (карта раскладывается, слой детекторов заполняется): при повторе из кеша его нет.
    """

    def log(self, path: str, data: object, *, level: int | None = None, live: bool = False) -> None:
        import rerun as rr

        rr.reset_time()
        if level is None:
            rr.log(path, data, static=True)
        else:
            rr.set_time(TIMELINE, sequence=int(level))
            rr.log(path, data)
        rr.reset_time()


class CycleRerunView(LadderObserver):
    """Прогон цикла в Rerun по этапам 1–7 (как README): части уровня по ходу счёта, весь путь — на цифре 5."""

    # Куда рисуется карта, пока она раскладывается: память уровня (этап 3) и карта верхних деталей (этап 7).
    LIVE_TARGETS = {3: f"{LAYOUT}/map", 7: f"{RECOGNITION}/top_map"}

    def __init__(self, sink: RerunSink | None = None, *, glyphs: Callable[[str], str | None] | None = None) -> None:
        self.sink = sink if sink is not None else RerunSink()
        self.glyphs = glyphs  # имя карты ("level2", "classes") → файл её рисунка окон, если есть
        self._stage = (0, 0)  # (уровень, этап), который идёт сейчас
        self._tabs, self._tab_clock = False, float("-inf")  # открывать вкладку идущего этапа (follow(tabs=True))
        self._started = time.monotonic()
        self._clock: dict[str, float] = {}
        self._reset()

    def _reset(self) -> None:
        self.levels: dict[int, dict] = {}  # числа уровня для «Что видно» и сводки
        self.top_numbers: dict = {}
        self._parts: dict[int, SimpleNamespace] = {}  # части уровня, пришедшие по ходу (имена полей LadderLevelTrace)
        self._shown: set[tuple[int, str]] = set()
        self._example_bits: dict[int, np.ndarray] = {}  # код окна-примера уровня
        self._similarity: dict[int, np.ndarray] = {}  # сходство точек карты уровня с окном-примером (§6.1)
        self._quality: list[tuple[float, float]] = []  # Ē (§5.8) по шагам раскладки, которая идёт сейчас
        self._quality_of: tuple[int, int] | None = None
        self._labels: tuple[tuple[int, int] | None, np.ndarray | None] = (None, None)
        self._live_level: int | None = None
        self._level0 = False  # начался ли уровень 0 (LADDER_LEVEL0): на шкале cycle он стоит на отметке 0

    # ------------------------------------------------------------------ общее
    def _log(self, path: str, data: object, level: int | None = None, *, live: bool = False) -> None:
        self.sink.log(path, data, level=level, live=live)

    def _scalar(self, path: str, value: float, level: int, name: str) -> None:
        import rerun as rr

        self._log(path, rr.SeriesLines(names=name, widths=2.5))
        self._log(path, rr.SeriesPoints(names=name, marker_sizes=6.0))
        self._log(path, rr.Scalars(float(value)), level)

    def _text(self, path: str, markdown: str, level: int | None = None) -> None:
        import rerun as rr

        self._log(path, rr.TextDocument(markdown, media_type=rr.MediaType.MARKDOWN), level)

    def _px(self, cells: np.ndarray) -> np.ndarray:
        """Место на сетке углопозиций, в клетках → пиксели рабочей картинки (целое — центр клетки)."""
        return (np.asarray(cells, dtype=np.float64) + 0.5) * self.unit

    def _digit(self, path: str, picture: int, level: int | None = None, *, brightness: float = 0.55) -> None:
        import rerun as rr
        from PIL import Image

        side = int(round(self.run.images.shape[1] * self.upscale))
        image = Image.fromarray(self.run.images[picture]).resize((side, side), Image.BILINEAR)
        grey = (np.asarray(image, dtype=np.float64) * brightness).astype(np.uint8)
        self._log(path, rr.Image(np.repeat(grey[:, :, None], 3, axis=2)), level)

    def _sticks(self, path: str, picture: int, level: int | None = None, *, dim: np.ndarray | None = None) -> None:
        """Штрихи картинки отрезками вдоль линии; цвет — направление (как на рисунках README). Штрих края — стрелка,
        штрих середины линии — отрезок без наконечника, его цвет идёт по кругу за 180° (axial_sticks)."""
        import rerun as rr

        from damp.figures.style import axial_sticks, stick_hue

        first, last = int(self.starts[picture]), int(self.starts[picture + 1])
        y, x = self.run.stick_y[first:last], self.run.stick_x[first:last]
        angle = self.run.stick_angle[first:last].astype(np.float64)
        along = np.radians(angle + 90.0)
        half = 0.45 * self.unit
        ux, uy = np.cos(along) * half, -np.sin(along) * half
        centre = np.stack([self._px(x), self._px(y)], axis=1)
        shade = np.ones(angle.shape[0]) if dim is None else dim
        colours = np.asarray([hue_colour(stick_hue(a), v) for a, v in zip(angle, shade)], dtype=np.uint8)
        start = centre - np.stack([ux, uy], axis=1)
        radii = max(0.6, 0.06 * self.unit)
        if axial_sticks():
            strips = np.stack([start, centre + np.stack([ux, uy], axis=1)], axis=1)
            self._log(path, rr.LineStrips2D(strips, colors=colours, radii=radii), level)
            return
        self._log(path, rr.Arrows2D(origins=start, vectors=np.stack([2 * ux, 2 * uy], axis=1), colors=colours, radii=radii), level)

    # ------------------------------------------------------------------ ход счёта
    # Этапы уровня в порядке счёта (LEVEL_PARTS): окна (6) → коды (2) → карта (3) → детекторы (4) → детали (5) → точность (7).
    LEVEL_ORDER = (6, 2, 3, 4, 5, 7)
    STAGE_WORK = {
        1: "картинки → штрихи контура",
        2: "окно → код «что и где»",
        3: "коды окон ложатся на карту",
        4: "детекторы над картой, их срабатывания",
        5: "сработавшие детекторы → детали",
        6: "окна-скважины уровня на элементах",
        7: "точность уровня; наверху — голосование",
    }

    def follow(self, *, tabs: bool = False) -> None:
        """Показ по ходу счёта: подписки на ход раскладки и слоёв детекторов (кадры раз в RERUN_LIVE_INTERVAL);
        tabs — окно само открывает вкладку идущего этапа (та же раскладка панелей с другой открытой вкладкой)."""
        self._tabs, self._tab_clock = bool(tabs), float("-inf")
        LOGGER.watch("layout.step", self._layout_step)
        LOGGER.watch("activation.detectors.layer", self._detector_layer)

    def close(self) -> None:
        """Снять подписки."""
        LOGGER.unwatch("layout.step", self._layout_step)
        LOGGER.unwatch("activation.detectors.layer", self._detector_layer)

    def stage(self, number: int, stage: int) -> None:
        """Идёт этап stage уровня number: сводка прогона на вкладке «Обзор» показывает, какой этап идёт."""
        self._stage = (int(number), int(stage))
        if (number, stage) == (0, 1):
            self._started = time.monotonic()  # прогон начался
        if stage:
            self._text(f"{OVERVIEW}/summary", self._progress(int(number), int(stage)))
        if self._tabs:
            # Открыть вкладку идущего этапа (в конце прогона — «Обзор»), не чаще раза в RERUN_LIVE_INTERVAL секунд.
            now = time.monotonic()
            if stage == 0 or now - self._tab_clock >= float(settings.RERUN_LIVE_INTERVAL):
                self._tab_clock = now
                tab = CycleBlueprint.TABS[int(stage)]
                LOGGER.use_blueprint(lambda: CycleBlueprint.build(active=tab), cycle=True)

    @staticmethod
    def _minutes(seconds: float) -> str:
        minutes, rest = divmod(int(round(seconds)), 60)
        return f"{minutes} мин {rest:02d} с" if minutes else f"{rest} с"

    def _progress(self, number: int, stage: int) -> str:
        """Сводка, пока идёт прогон: какой этап идёт сейчас и что уже готово на уровне; готовые уровни — с числами."""
        tabs = CycleBlueprint.TABS
        level = number > 0 or self._level0  # уровень 0 (LADDER_LEVEL0) идёт тем же проходом, что уровни, без узнавания
        if not level:
            done, total = set(range(1, stage)), 2
        else:
            order = self.LEVEL_ORDER if number else self.LEVEL_ORDER[:-1]
            done = {1, *order[: order.index(stage)]} if stage in order else {1}
            total = len(order)
        rows = []
        for value in range(1, 8):
            if value == stage:
                rows.append(f"| **▶ {tabs[value]}** | **идёт** — {self.STAGE_WORK[value]} |")
            else:
                rows.append(f"| {tabs[value]} | {'✓ готово' if value in done else 'ждёт'} |")
        where = f"уровень {number}" if level else "до первого уровня"
        text = (
            f"## ▶ Идёт этап {tabs[stage]} · {where}\n\n{self.STAGE_WORK[stage].capitalize()}. Этап начат через "
            f"{self._minutes(time.monotonic() - self._started)} от начала прогона; картинки этапа — на его вкладке.\n\n"
            f"| Этап | {where.capitalize()} |\n|---|---|\n" + "\n".join(rows) +
            f"\n| **Итого** | готово {len(done - {1}) if level else len(done)} из {total} этапов {'уровня' if level else 'до уровней'} |\n"
        )
        finished = [n for n in sorted(self.levels) if "seconds" in self.levels[n]]
        if finished:
            lines, seconds = [], 0.0
            for n in finished:
                s = self.levels[n]
                spent = sum(s["seconds"].values())
                seconds += spent
                lines.append(f"| {n} | {_num(s.get('points'))} | {_num(s.get('detectors'))} | {_num((s.get('alone') or {}).get(1), '.1%')} | "
                             f"{spent:.1f} |")
            text += (
                "\n### Готовые уровни\n\n| уровень | точек карты | детекторов | по отклику, k=1 | секунд |\n|---|---|---|---|---|\n"
                + "\n".join(lines) + f"\n| **Итого** | готово уровней: {len(finished)} | | | {seconds:.1f} |\n"
            )
        return text + "\nПолная сводка с итогом появится после верхнего уровня."

    def _due(self, name: str) -> bool:
        """Пора рисовать кадр name: с прошлого прошло не меньше RERUN_LIVE_INTERVAL секунд (0 — каждый раз)."""
        now = time.monotonic()
        last = self._clock.get(name)
        if last is not None and now - last < float(settings.RERUN_LIVE_INTERVAL):
            return False
        self._clock[name] = now
        return True

    def _layout_step(self, layout, step: int, energy: float | None, mode: str) -> None:  # noqa: ANN001
        """Карта раскладывается (§5.5): точки по ходу счёта, цвет — цифра; качество Ē (§5.8) по шагам дальней фазы."""
        import rerun as rr

        number, stage = self._stage
        target = self.LIVE_TARGETS.get(stage)
        if target is None:
            return
        if self._quality_of != (number, stage):  # новая карта — новый график качества
            self._quality_of, self._quality = (number, stage), []
        if energy is not None:
            self._quality.append((float(step), float(energy)))
        if not self._due("layout"):
            return
        cells = layout.cells_array().astype(np.float64)
        key = (id(layout), int(cells.shape[0]))
        if self._labels[0] != key:
            self._labels = (key, np.asarray([int(label) for label in layout.code_table.labels()], dtype=np.uint16))
        self._log(target, rr.Points2D(np.stack([cells[:, 1], cells[:, 0]], axis=1), class_ids=self._labels[1], radii=0.45), number,
                  live=True)
        if stage == 3:
            self._log_quality(number)

    def _log_quality(self, number: int) -> None:
        """Качество Ē раскладки, которая идёт сейчас: столбец на каждую оценку (раз в LAYOUT_ENERGY_EVAL_EVERY шагов)."""
        import rerun as rr

        if self._quality and self._quality_of == (number, 3):
            values = np.asarray([value for _, value in self._quality], dtype=np.float64)
            self._log(f"{LAYOUT}/live/quality", rr.BarChart(values, abscissa=np.arange(1, values.shape[0] + 1, dtype=np.float64)),
                      number, live=True)

    @staticmethod
    def _layer_count(layer: int = 0) -> int:
        """Сколько слоёв детекторов стирать и рисовать: по настройкам, но не меньше слоя layer + 1."""
        return max(int(settings.LADDER_HIERARCHY_LAYERS), len(settings.LADDER_HIERARCHY_THRESHOLDS or ()), layer + 1)

    def _detector_layer(self, layer: int, selected, built, done: bool) -> None:  # noqa: ANN001
        """Слои детекторов заполняются (§6.4): детекторы текущего слоя по ходу, готовые слои остаются; цвет — слой."""
        import rerun as rr

        number, stage = self._stage
        if stage != 4:
            return
        layers = self._layer_count(layer)
        if self._live_level != number:
            # Первый кадр уровня: слои прошлого уровня убираются, иначе они стояли бы до конца построения.
            self._live_level = number
            self._log(f"{DETECTORS}/layers", rr.AnnotationContext([(k, f"слой {k}", layer_colour(k, layers)) for k in range(layers)]),
                      number, live=True)
            for k in range(layers):
                self._log(f"{DETECTORS}/layers/l{k}", rr.LineStrips2D([]), number, live=True)
        if not done and not self._due("detectors"):
            return
        strips = [circle(float(item.center[0]), float(item.center[1]), float(item.radius), 32) for item in selected]
        self._log(f"{DETECTORS}/layers/l{layer}", rr.LineStrips2D(strips, class_ids=np.full(len(strips), layer, dtype=np.uint16), radii=0.06),
                  number, live=True)

    # ------------------------------------------------------------------ этап 1 и общие подписи
    def start(self, trace: LadderRunTrace) -> None:
        import rerun as rr

        self.run = trace
        self._reset()
        self.starts = np.searchsorted(trace.stick_picture, np.arange(trace.labels.shape[0] + 1))
        self.upscale = float(settings.MNIST_SOBEL_UPSCALE) if settings.MNIST_SOBEL_UPSCALE_ENABLED else 1.0
        self.unit = float(settings.MNIST_PATCH_SIZE)  # пикселей рабочей картинки на клетку сетки
        train_first, train_last = trace.ranges["train"]
        # Пример — пятёрка прогона: её выбрал MnistLadderRunner (в каждом прогоне другая).
        self.main = int(trace.example) if trace.example is not None else train_first
        self.main_index = int(trace.example_index) if trace.example_index is not None else self.main
        group = trace.ranges.get("test", trace.ranges["train"])
        self.examples: dict[int, int] = {}
        for value in settings.RERUN_EXAMPLE_DIGITS:
            own = np.flatnonzero(trace.labels[group[0] : group[1]] == int(value)) + group[0]
            if own.size:
                self.examples[int(value)] = int(own[0])
        if trace.example is not None:
            # Пятёрка во всех панелях (d5, голоса, ответ) - одна и та же: пример прогона, а не первая в группе.
            self.examples[int(settings.LADDER_FIGURE_DIGIT)] = self.main
        # Цвета и подписи цифр — одни на весь прогон (как цвет класса на картах README).
        step = float(settings.LADDER_CLASS_HUE_STEP)
        self._log(ROOT, rr.AnnotationContext([(k, f"цифра {k}", hue_colour(k * step)) for k in range(10)]))
        # Этап 1: цифра-пример крупно и цифры 0–9.
        self._digit(f"{STIMULI}/main/image", self.main)
        self._sticks(f"{STIMULI}/main/sticks", self.main)
        for value, picture in self.examples.items():
            self._digit(f"{STIMULI}/digits/d{value}/image", picture)
            self._sticks(f"{STIMULI}/digits/d{value}/sticks", picture)
        per_picture = np.diff(self.starts)[train_first:train_last]
        counts = np.bincount(per_picture, minlength=int(per_picture.max()) + 1 if per_picture.size else 1)
        self._log(f"{STIMULI}/sticks_per_picture", rr.BarChart(counts, abscissa=np.arange(counts.shape[0], dtype=np.float64)))
        sticks = int(self.starts[self.main + 1] - self.starts[self.main])
        views = trace.ranges.get("views")
        if views is None:
            taught = f"{train_last - train_first}"
        else:
            # §7.1.4: каждая обучающая цифра идёт ещё и в сдвинутых видах (LADDER_TRAIN_VIEWS); виды — тоже обучающие.
            originals = views[0] - train_first
            taught = (f"{originals} цифр и их сдвинутые виды, {(views[1] - views[0]) // max(1, originals)} на цифру "
                      f"(§7.1.4): всего {train_last - train_first}")
        self._text(f"{STIMULI}/about", self._stimuli_about(sticks, taught, float(per_picture.mean()) if per_picture.size else 0.0))
        # Этап 2: детекторы угла штриха (статично: кодер один на прогон).
        self._detectors_picture(f"{CODING}/angle_detectors", trace.angle_encoder, (45.0,))
        # Этап 6: цифра-пример, на которую ложатся окна уровней.
        self._digit(f"{CYCLE}/keyholes/image", self.main, brightness=0.45)
        self._sticks(f"{CYCLE}/keyholes/sticks", self.main, dim=np.full(int(self.starts[self.main + 1] - self.starts[self.main]), 0.7))

    def _stimuli_about(self, sticks: int, taught: str, mean: float) -> str:
        """«Что видно» этапа 1: как из картинки получаются штрихи (по настройкам MNIST_STROKE_*)."""
        if settings.MNIST_STROKE_DETECTORS:
            source = "середины линии (гребень расстояния до края)" if settings.MNIST_STROKE_MIDLINE else "края линии"
            how = (
                f"Штрих — детектор над точками {source}: кластер соседних точек с похожим направлением, место — его "
                f"взвешенный центр, центры не ближе шага (§6.4 на плоскости картинки — доработка). "
                + ("Отрезок идёт вдоль линии, цвет — направление (0–180°: у середины линии нет «внутри» и «снаружи»)."
                   if settings.MNIST_STROKE_MIDLINE else "Стрелка идёт вдоль края, цвет — направление перепада яркости (0–360°).")
            )
            grid = f"| шаг штрихов | {int(settings.MNIST_STROKE_STEP)} px рабочей картинки 224×224; место — дробное, в клетках по {int(self.unit)} px |\n"
        else:
            how = "Штрих — углопозиция окна сетки 24×24. Стрелка идёт вдоль края, цвет — направление перепада яркости (0–360°)."
            grid = f"| сетка | 24×24 клетки, окно Собеля {int(self.unit)} px |\n"
        return (
            f"### Этап 1. Стимулы: штрихи MNIST\n\nКартинка распадается на штрихи: где линия и куда она идёт. {how}\n\n"
            f"| | |\n|---|---|\n| цифра-пример | пятёрка № {self.main_index} в MNIST (в каждом прогоне другая, среди пятёрок со штрихами; как на рисунках README этого прогона) |\n| штрихов у неё | {sticks} |\n"
            f"| обучающих картинок | {taught} |\n| штрихов на картинку, в среднем | {mean:.1f} |\n"
            + grid
            + f"| **Итого** | пятёрка № {self.main_index} — {sticks} штрихов; дальше весь путь показан на ней |\n\n"
            "Внизу — по одной проверочной цифре каждого класса. Источник: вне статьи (извлечение стимулов; штрихи "
            "детекторами — перенос §6.4 на картинку)."
        )

    def _detectors_picture(self, path: str, encoder, values) -> None:  # noqa: ANN001
        """Картинка детекторов кодера (кольца угла, круги места) — та же, что рисует README (§4.1–§4.4)."""
        import rerun as rr

        try:
            from damp.encoding.detector_view import DetectorLayerView
            from damp.figures.style import figure_style

            view = DetectorLayerView(encoder, style=figure_style(), min_hue=settings.LAYOUT_PSEUDO_COLOR_MIN_HUE,
                                     max_hue=settings.LAYOUT_PSEUDO_COLOR_MAX_HUE, inactive_alpha=settings.DETECTORS_VIEW_INACTIVE_ALPHA,
                                     panel=settings.DETECTORS_VIEW_PANEL, code_columns=32, code_cell=10)
            self._log(path, rr.Image(np.asarray(view.render(values), dtype=np.uint8)))
        except Exception as error:  # картинка — не главное: без неё вкладка работает
            LOGGER.event("rerun.cycle.detectors_picture.warning", section=OUTSIDE_ARTICLE, data={"path": path, "error": repr(error)})

    # ------------------------------------------------------------------ уровень: части по ходу счёта
    def begin(self, number: int) -> None:
        import rerun as rr

        if number == 0:
            self._level0 = True
        what = "штрихи идут на этап 2: окно на каждом штрихе, место — от центра цифры" if number == 0 else "элементы идут на этап 2"
        self._log(f"{CYCLE}/marks", rr.TextLog(f"↻ уровень {number}: начало цикла — {what}", level="INFO"), number)

    def part(self, number: int, name: str, values: dict) -> None:
        """Часть прохода уровня готова — её вкладка заполняется сразу (LEVEL_PARTS)."""
        trace = self._parts.setdefault(number, SimpleNamespace(number=number))
        trace.__dict__.update(values)
        self._show(number, name, trace)

    def _show(self, number: int, name: str, trace) -> None:  # noqa: ANN001 — LadderLevelTrace или части уровня
        getattr(self, f"_show_{name}")(number, trace)
        self._shown.add((number, name))

    def _main_keyhole(self, n: int, trace) -> int:  # noqa: ANN001
        """Окно-пример уровня: ближайшее к середине цифры, дальше — к окну прошлого уровня (как на рисунках README)."""
        mine = np.flatnonzero(trace.keyhole_picture == self.main)
        if not mine.size:
            return int(np.argmin(trace.keyhole_picture))
        previous = self.levels.get(n - 1, {}).get("centre")
        if previous is None:
            first, last = int(self.starts[self.main]), int(self.starts[self.main + 1])
            previous = (float(self.run.stick_y[first:last].mean()), float(self.run.stick_x[first:last].mean()))
        if n == 0:
            # Уровень 0: у всех окон цифры один центр — центр цифры; пример — окно штриха, ближайшего к середине.
            centre = np.stack([trace.element_y[mine], trace.element_x[mine]], axis=1).astype(np.float64)
            return int(mine[int(np.argmin(np.hypot(centre[:, 0] - previous[0], centre[:, 1] - previous[1])))])
        centre = trace.keyhole_centre[mine].astype(np.float64)
        return int(mine[int(np.argmin(np.hypot(centre[:, 0] - previous[0], centre[:, 1] - previous[1])))])

    def _show_keyholes(self, n: int, t) -> None:  # noqa: ANN001
        """Этапы 2 и 6: окна пятёрки на цифре, окно-пример и его элементы; круг окна уровня — на цифре этапа 6."""
        import rerun as rr

        s = self.levels.setdefault(n, {})
        keyhole = self._main_keyhole(n, t)
        wy, wx = (float(v) for v in t.keyhole_centre[keyhole])
        radius = float(t.keyhole_extent[keyhole])
        members = t.entry_element[t.entry_keyhole == keyhole]
        own = np.flatnonzero(t.keyhole_picture == self.main)
        s.update(centre=(wy, wx), radius=radius, keyhole=keyhole, members=int(members.shape[0]), five_keyholes=int(own.shape[0]))
        self._digit(f"{CODING}/keyhole/image", self.main, n, brightness=0.4)
        if n <= 1:  # элементы уровней 0 и 1 — штрихи (у уровня 1 с кодами уровня 0)
            first, last = int(self.starts[self.main]), int(self.starts[self.main + 1])
            dim = np.full(last - first, 0.3)
            inside = members - first
            dim[inside[(inside >= 0) & (inside < dim.shape[0])]] = 1.0
            self._sticks(f"{CODING}/keyhole/elements", self.main, n, dim=dim)
        else:
            mine = np.flatnonzero(t.element_picture == self.main)
            taken = np.isin(mine, members)
            positions = np.stack([self._px(t.element_x[mine]), self._px(t.element_y[mine])], axis=1)
            colours = np.where(taken[:, None], np.asarray([[120, 200, 255]]), np.asarray([[70, 78, 90]])).astype(np.uint8)
            self._log(f"{CODING}/keyhole/elements", rr.Points2D(positions, colors=colours, radii=np.where(taken, 0.3, 0.18) * self.unit), n)
        # Все окна пятёрки — тонкими кругами: окна стоят на каждом её элементе (П.2).
        centres, extents = t.keyhole_centre[own].astype(np.float64), np.asarray(t.keyhole_extent)[own]
        self._log(f"{CODING}/keyhole/all", rr.LineStrips2D(
            [circle(self._px(cy), self._px(cx), float(r) * self.unit, 32) for (cy, cx), r in zip(centres, extents)],
            colors=[KEYHOLE_FAINT], radii=0.04 * self.unit), n)
        self._log(f"{CODING}/keyhole/circle", rr.LineStrips2D([circle(self._px(wy), self._px(wx), radius * self.unit)], colors=[KEYHOLE],
                                                             radii=0.12 * self.unit, labels=[f"окно уровня {n}"]), n)
        # Этап 6: окно уровня на цифре — круги растут от уровня к уровню (П.3); окно уровня 0 — вокруг центра цифры.
        colour = LEVEL0_COLOUR if n == 0 else LEVEL_COLOURS[(n - 1) % len(LEVEL_COLOURS)]
        self._log(f"{CYCLE}/keyholes/level", rr.LineStrips2D([circle(self._px(wy), self._px(wx), radius * self.unit)], colors=[colour],
                                                             radii=0.14 * self.unit, labels=[f"{n}"]), n)
        self._log(f"{CYCLE}/keyholes/centre", rr.Points2D([[self._px(wx), self._px(wy)]], colors=[colour], radii=0.25 * self.unit), n)
        self._scalar(f"{CYCLE}/stats/seen", float(t.seen), n, "доля цифры в окне")
        s["seen"] = float(t.seen)

    def _show_codes(self, n: int, t) -> None:  # noqa: ANN001
        """Этап 2: код окна-примера «место | содержание» (П.4–П.7) и плотность кодов окон."""
        import rerun as rr

        s = self.levels.setdefault(n, {})
        bits = np.unpackbits(np.ascontiguousarray(t.keyhole_codes[s["keyhole"]]), count=int(t.code_length))
        self._example_bits[n] = bits
        place, content = bits[: t.place_length], bits[t.place_length :]
        self._log(f"{CODING}/code/place", rr.Image(bits_image(place, 64, PLACE)), n)
        self._log(f"{CODING}/code/content", rr.Image(bits_image(content, 64, CONTENT)), n)
        ones = np.unpackbits(np.ascontiguousarray(t.keyhole_codes), axis=1, count=int(t.code_length)).sum(axis=1)
        s.update(place_bits=int(place.sum()), content_bits=int(content.sum()), code_length=int(t.code_length),
                 place_length=int(t.place_length), content_length=int(t.content_length), ones_mean=float(ones.mean()))
        self._scalar(f"{CODING}/stats/ones", float(ones.mean()), n, "единиц в коде окна")
        self._scalar(f"{CODING}/stats/density", float(ones.mean()) / max(1, int(t.code_length)), n, "плотность кода окна")
        if n <= 1 and t.place_encoder is not None:
            self._detectors_picture(f"{CODING}/place_detectors", t.place_encoder, ((0.0, 0.0),))

    def _show_memory(self, n: int, t) -> None:  # noqa: ANN001
        """Этап 3: карта памяти уровня (цвет — цифра), окно-пример пятёрки на ней, энергия точек (§5, §5.7)."""
        import rerun as rr

        s = self.levels.setdefault(n, {})
        cells = t.map_cells.astype(np.float64)
        positions = np.stack([cells[:, 1], cells[:, 0]], axis=1)
        self._log(f"{LAYOUT}/map", rr.Points2D(positions, class_ids=t.map_labels.astype(np.uint16), radii=0.45), n)
        energy = np.asarray(t.energies, dtype=np.float64)
        share = (energy - energy.min()) / max(1e-12, float(np.ptp(energy)))
        energy_colours = (np.asarray(OFF) + share[:, None] * (np.asarray((250, 230, 120)) - np.asarray(OFF))).astype(np.uint8)
        self._log(f"{LAYOUT}/energy", rr.Points2D(positions, colors=energy_colours, radii=0.45), n)
        # Окно-пример пятёрки на карте: точка, чей код ближе всех к его коду (§6.1) — туда оно и «попадает».
        similarity = self._similarity_of(t, self._example_bits[n])
        self._similarity[n] = similarity
        best = int(np.argmax(similarity))
        y, x = float(cells[best, 0]), float(cells[best, 1])
        self._log(f"{LAYOUT}/map/five", rr.LineStrips2D([circle(y, x, 1.6, 24)], colors=[FIVE], radii=0.18,
                                                         labels=[f"окно пятёрки (сходство {similarity[best]:.2f})"]), n)
        s.update(points=int(cells.shape[0]), map_size=tuple(int(v) for v in t.map_size), five_similarity=float(similarity[best]),
                 five_label=int(t.map_labels[best]))
        self._scalar(f"{LAYOUT}/stats/points", float(cells.shape[0]), n, "точек на карте")
        self._log_quality(n)
        # Этап 4 начинается на этой карте: серый фон под слоями детекторов.
        self._log(f"{DETECTORS}/map", rr.Points2D(positions, colors=np.full((cells.shape[0], 3), 60, dtype=np.uint8), radii=0.42), n)
        # Слои детекторов прошлого уровня на новой карте не к месту: пока детекторы этого уровня строятся (минуты на
        # больших картах), на шкале уровня n виднелись бы круги уровня n−1 — другой карты, другого размера. Стираем их
        # сейчас, вместе с картой; слои уровня n появятся кадрами по ходу счёта либо готовыми (_show_detectors).
        for k in range(self._layer_count()):
            self._log(f"{DETECTORS}/layers/l{k}", rr.LineStrips2D([]), n)

    @staticmethod
    def _similarity_of(t, bits: np.ndarray) -> np.ndarray:  # noqa: ANN001
        """Косинусное сходство кода окна-примера с кодами точек карты (§6.1)."""
        memory = np.unpackbits(np.ascontiguousarray(t.map_codes), axis=1, count=int(t.code_length)).astype(np.float32)
        code = bits.astype(np.float32)
        return (memory @ code) / np.sqrt(np.maximum(memory.sum(axis=1) * code.sum(), 1.0))

    def _show_detectors(self, n: int, t) -> None:  # noqa: ANN001
        """Этап 4: детекторы по слоям порогов λ_d (§6.4) и точки, которые зажигает окно-пример (§6.1)."""
        import rerun as rr

        s = self.levels.setdefault(n, {})
        cells = t.map_cells.astype(np.float64)
        positions = np.stack([cells[:, 1], cells[:, 0]], axis=1)
        detectors, layers = t.detectors, np.asarray(t.layer_index)
        thresholds = tuple(t.layer_thresholds)
        self._log(f"{DETECTORS}/layers", rr.AnnotationContext(
            [(k, f"слой {k}: λ_d = {value:.3f}", layer_colour(k, len(thresholds))) for k, value in enumerate(thresholds)]
        ), n)
        for k in range(len(thresholds)):
            rows = np.flatnonzero(layers == k)
            strips = [circle(detectors[i, 0], detectors[i, 1], detectors[i, 2], 32) for i in rows]
            self._log(f"{DETECTORS}/layers/l{k}", rr.LineStrips2D(strips, class_ids=np.full(len(strips), k, dtype=np.uint16), radii=0.06), n)
        self._log(f"{DETECTORS}/per_layer", rr.BarChart(np.bincount(layers, minlength=len(thresholds)),
                                                        abscissa=np.arange(len(thresholds), dtype=np.float64)), n)
        similarity = self._similarity.get(n)
        if similarity is None:
            similarity = self._similarity_of(t, self._example_bits[n])
            self._similarity[n] = similarity
        from damp.figures.cycle import CycleFigures

        low, high = float(t.activation_lambda), float(similarity.max())
        lit = (similarity >= low) | (similarity >= high)  # самая похожая точка зажигается всегда (как в прогоне)
        # В детекторы идут только зажжённые точки с зажжёнными соседями — энергия активации (как в прогоне).
        kept = CycleFigures.local_energy(cells, np.where(lit, similarity, 0.0)) > 0.0
        level_share = np.clip((similarity[kept] - low) / max(1e-12, high - low), 0.0, 1.0)
        lit_colours = (np.asarray(LIT_LOW) + level_share[:, None] * (np.asarray(LIT_HIGH) - np.asarray(LIT_LOW))).astype(np.uint8)
        grey = np.full((cells.shape[0], 3), 60, dtype=np.uint8)
        grey[lit & ~kept] = 100  # зажглись, но погасли: соседи не горят
        self._log(f"{DETECTORS}/activation/map", rr.Points2D(positions, colors=grey, radii=0.42), n)
        self._log(f"{DETECTORS}/activation/lit", rr.Points2D(positions[kept], colors=lit_colours, radii=0.5), n)
        s.update(detectors=int(detectors.shape[0]), layers=[int(v) for v in np.bincount(layers, minlength=len(thresholds))],
                 thresholds=thresholds, activation_lambda=low, lit=int(lit.sum()), kept=int(kept.sum()))
        self._scalar(f"{DETECTORS}/stats/detectors", float(detectors.shape[0]), n, "детекторов")

    def _show_firing(self, n: int, t) -> None:  # noqa: ANN001
        """Этап 4: сработавшие на окно-пример детекторы (§6.5), сильнейшие — оранжевым; этап 5 — их уровни E(d)."""
        import rerun as rr

        s = self.levels.setdefault(n, {})
        detectors, keyhole = t.detectors, s["keyhole"]
        mine = t.fired_keyhole == keyhole
        fired, levels = t.fired_detector[mine], t.fired_level[mine]
        order = np.argsort(-levels, kind="stable")
        strongest = fired[order][: int(settings.RERUN_STRONGEST)]
        self._log(f"{DETECTORS}/activation/fired", rr.LineStrips2D([circle(detectors[i, 0], detectors[i, 1], detectors[i, 2], 32) for i in fired],
                                                                   colors=[FIRED], radii=0.05), n)
        self._log(f"{DETECTORS}/activation/strongest", rr.LineStrips2D([circle(detectors[i, 0], detectors[i, 1], detectors[i, 2], 32) for i in strongest],
                                                                       colors=[STRONG], radii=0.12), n)
        per_keyhole = np.bincount(t.fired_keyhole, minlength=int(t.keyhole_picture.shape[0]))
        s.update(fired=int(fired.shape[0]), strongest=int(strongest.shape[0]), fired_mean=float(per_keyhole.mean()),
                 silent=float(np.mean(per_keyhole == 0)))
        self._scalar(f"{DETECTORS}/stats/fired", float(per_keyhole.mean()), n, "сработало на окно")
        self._log(f"{EMBEDDINGS}/fired_levels", rr.BarChart(np.sort(levels)[::-1].astype(np.float64),
                                                             abscissa=np.arange(1, levels.shape[0] + 1, dtype=np.float64)), n)

    def _show_details(self, n: int, t) -> None:  # noqa: ANN001
        """Этап 5: код детали окна-примера (§6.5, §7.1.12) и детали пятёрки — центры её окон (П.1)."""
        import rerun as rr

        from damp.figures.cycle import CycleFigures

        s = self.levels.setdefault(n, {})
        keyhole = s["keyhole"]
        detail = np.unpackbits(np.ascontiguousarray(t.detail_codes[keyhole]), count=int(t.detail_length))
        self._log(f"{EMBEDDINGS}/code", rr.Image(bits_image(detail, 32, DETAIL, cell=12)), n)
        self._digit(f"{EMBEDDINGS}/details/image", self.main, n, brightness=0.3)
        own = np.flatnonzero(t.keyhole_picture == self.main)
        # Деталь встаёт в центр окна; у уровня 0 — на место своего штриха (центр окна там — центр цифры).
        centres, (wy, wx) = CycleFigures.detail_places(t, own, keyhole)
        self._log(f"{EMBEDDINGS}/details/all", rr.Points2D(np.stack([self._px(centres[:, 1]), self._px(centres[:, 0])], axis=1),
                                                            colors=[DETAIL], radii=0.16 * self.unit), n)
        self._log(f"{EMBEDDINGS}/details/main", rr.Points2D([[self._px(wx), self._px(wy)]], colors=[(255, 255, 255)], radii=0.4 * self.unit,
                                                             labels=[f"деталь уровня {n}"]), n)
        detail_bits = np.unpackbits(np.ascontiguousarray(t.detail_codes), axis=1, count=int(t.detail_length)).sum(axis=1)
        s.update(detail_length=int(t.detail_length), detail_ones=int(detail.sum()), detail_bits=float(detail_bits.mean()),
                 strength=float(t.detail_strength[keyhole]), strength_mean=float(np.mean(t.detail_strength)), details=int(own.shape[0]))
        self._scalar(f"{EMBEDDINGS}/stats/bits", float(detail_bits.mean()), n, "единиц в коде детали")
        self._scalar(f"{EMBEDDINGS}/stats/strength", float(np.mean(t.detail_strength)), n, "сила детали, средняя")

    def _show_recognition(self, n: int, t) -> None:  # noqa: ANN001
        """Этап 7: точность узнавания по уровню — сразу после уровня, не дожидаясь конца прогона."""
        s = self.levels.setdefault(n, {})
        s.update(alone=getattr(t, "alone", None), glued=getattr(t, "glued", None))
        self._accuracy(n, getattr(t, "alone", None), getattr(t, "peak", None), getattr(t, "glued", None))

    def _accuracy(self, n: int, alone: dict | None, peak: dict | None, glued: dict | None) -> None:
        for name, values in (("response", alone), ("peak", peak), ("glued", glued)):
            title = {"response": "по отклику", "peak": "по сильнейшему", "glued": "склейка уровней"}[name]
            for k, value in (values or {}).items():
                self._scalar(f"{RECOGNITION}/accuracy/{name}_k{k}", float(value), n, f"{title}, k={k}")

    def level(self, trace: LadderLevelTrace) -> None:
        """Итог прохода: части, которых ещё не было, затем то, что есть только в итоге (карта окон, секунды, отметка)."""
        import rerun as rr

        n = trace.number
        for name in LEVEL_PARTS:
            if (n, name) not in self._shown and name != "recognition":  # узнавание — в итогах прогона (finish)
                self._show(n, name, trace)
        s = self.levels.setdefault(n, {})
        s["seconds"] = dict(trace.seconds)
        glyph = self.glyphs(f"level{n}") if self.glyphs is not None else None
        if glyph:
            self._log(f"{LAYOUT}/glyphs", rr.EncodedImage(path=glyph), n)
        for name, seconds in trace.seconds.items():
            self._scalar(f"{OVERVIEW}/seconds/{name}", float(seconds), n, f"{name}, с")
        self._log(f"{CYCLE}/marks", rr.TextLog(
            "↺ уровень 0: у штрихов выученные коды (детали уровня 0) — штрихи идут в окна уровня 1" if n == 0 else
            f"↺ уровень {n}: детали готовы (радиус окна {float(np.mean(trace.keyhole_extent)):.1f} клетки, доля цифры {trace.seen:.0%}) — "
            "снова этап 2, если окно ещё не видит цифру целиком", level="INFO"), n)
        self._abouts(n, s)
        # Части уровня больше не нужны: числа — в self.levels, картинки — в Rerun.
        self._parts.pop(n, None)
        self._similarity.pop(n, None)
        self._example_bits.pop(n, None)

    # ------------------------------------------------------------------ верх: распознавание
    def top(self, trace: LadderTopTrace) -> None:
        import rerun as rr

        from damp.decode.nearest_codes import NearestCodeClassifier

        n = int(trace.level)
        labels = self.run.labels
        if trace.map_cells is not None:
            cells = trace.map_cells.astype(np.float64)
            self._log(f"{RECOGNITION}/top_map", rr.Points2D(np.stack([cells[:, 1], cells[:, 0]], axis=1),
                                                            class_ids=trace.map_labels.astype(np.uint16), radii=0.45), n)
        if "test" not in self.run.ranges:
            return
        first, last = self.run.ranges["test"]
        # Пустые детали (нулевой код) не голосуют и не входят в базу; картинка без голосов ответа не получает.
        filled = np.asarray(trace.codes).any(axis=1)
        train = np.flatnonzero((trace.picture < self.run.ranges["train"][1]) & filled)
        asked = np.flatnonzero((trace.picture >= first) & (trace.picture < last) & filled)
        votes = np.zeros((last - first, 10), dtype=np.int64)
        if train.size and asked.size:
            classifier = NearestCodeClassifier(trace.codes[train], labels[trace.picture[train]], int(trace.code_length),
                                               use_gpu=settings.ACTIVATION_USE_GPU)
            nearest = labels[trace.picture[train]][classifier.neighbours(trace.codes[asked], 1)][:, 0]
            np.add.at(votes, (trace.picture[asked] - first, nearest), 1)
        voted = votes.sum(axis=1) > 0
        answer = votes.argmax(axis=1)
        confusion = np.zeros((10, 10), dtype=np.int64)
        np.add.at(confusion, (labels[first:last][voted], answer[voted]), 1)
        self._log(f"{RECOGNITION}/confusion", rr.Tensor(confusion, dim_names=["цифра", "ответ"]), n)
        for value, picture in self.examples.items():
            if first <= picture < last:  # пример прогона вне группы подписан в тексте вкладки
                self._log(f"{RECOGNITION}/votes/d{value}", rr.BarChart(votes[picture - first].astype(np.float64),
                                                                     abscissa=np.arange(10, dtype=np.float64)), n)
        five = self.examples.get(int(settings.LADDER_FIGURE_DIGIT))
        self.top_numbers = dict(level=n, details=int(trace.picture.shape[0]), right=int(np.trace(confusion)), asked=int(last - first),
                                map_points=None if trace.map_cells is None else int(trace.map_cells.shape[0]),
                                empty_details=float(1.0 - filled.mean()) if filled.size else 0.0,
                                no_votes=float(1.0 - voted.mean()) if voted.size else 0.0,
                                five_answer=None if five is None or not first <= five < last or not voted[five - first]
                                else int(answer[five - first]),
                                five_outside=five is not None and not first <= five < last)

    # ------------------------------------------------------------------ итоги
    def finish(self, reports: list, accuracy: dict | None) -> None:
        import rerun as rr

        for report in reports:
            n = int(report.number)
            s = self.levels.setdefault(n, {})
            s.update(same_label=float(report.same_label), same_label_random=float(report.same_label_random),
                     field_radius=float(report.field_radius), picture_seen=float(report.picture_seen),
                     keyholes=float(report.fields_per_picture), elements=float(report.elements_per_field),
                     distinct=int(report.distinct_details), map_points=int(report.map_points), alone=report.alone, glued=report.glued,
                     energy_mean=float(report.energy_mean), code_resolution=float(report.code_resolution),
                     code_largest=int(report.code_largest), detail_empty=float(report.detail_empty),
                     detail_full=float(report.detail_full))
            self._scalar(f"{LAYOUT}/stats/same_label", float(report.same_label), n, "соседи одной цифры")
            self._scalar(f"{LAYOUT}/stats/same_label_random", float(report.same_label_random), n, "то же у случайной карты")
            self._scalar(f"{CYCLE}/stats/radius", float(report.field_radius), n, "радиус окна, клеток")
            self._scalar(f"{CYCLE}/stats/elements", float(report.elements_per_field), n, "элементов в окне")
            self._scalar(f"{CYCLE}/stats/keyholes", float(report.fields_per_picture), n, "окон на картинку")
            if (n, "recognition") not in self._shown:
                self._accuracy(n, report.alone, report.peak, report.glued)
            self._abouts(n, s, accuracy if report is reports[-1] else None)
        if accuracy and reports:
            top = int(reports[-1].number)
            for k, value in accuracy.items():
                self._scalar(f"{RECOGNITION}/accuracy/vote_k{k}", float(value), top, f"голосование деталей, k={k}")
            self._log(f"{CYCLE}/marks", rr.TextLog(f"↺ уровень {top}: конец цикла — окно видит цифру целиком; голосование {accuracy}",
                                                   level="INFO"), top)
        if self.glyphs is not None and reports:
            glyph = self.glyphs("classes")
            if glyph:
                self._log(f"{RECOGNITION}/top_glyphs", rr.EncodedImage(path=glyph), int(reports[-1].number))
        self._summary(reports, accuracy)

    def _abouts(self, n: int, s: dict, vote: dict | None = None) -> None:
        """Что видно на вкладках этапов 2–7 на уровне n — с числами уровня (Markdown); чего ещё нет — прочерк."""
        seconds = s.get("seconds", {})
        length, ones = s.get("code_length"), s.get("ones_mean")
        density = None if not length or ones is None else ones / length
        intro = (
            "Уровень 0 — первичная раскладка штрихов: окно стоит на каждом штрихе и держит его одного; центр окна — центр "
            "цифры (середина её штрихов), радиус — до самого дальнего штриха (доработка). Код окна — **место** штриха "
            "от центра цифры (биты кругов места, П.4) и **содержание** — код направления штриха под маской своего круга "
            "(П.5). Все окна пятёрки — один круг вокруг центра цифры; яркий штрих — штрих окна-примера."
            if n == 0 else
            "Окно-скважина стоит на каждом элементе (П.2): тонкие круги — все окна пятёрки, яркий — окно-пример. Его "
            "содержимое — один код «что и где» (П.4–П.7): **место** — биты сработавших кругов места (OR), **содержание** "
            "— коды элементов, переставленные маской своего круга."
        )
        self._text(f"{CODING}/about", (
            f"### Этап 2. Первичное кодирование — уровень {n}\n\n{intro}\n\n"
            f"| | |\n|---|---|\n| окон у пятёрки | {_num(s.get('five_keyholes'))} |\n| элементов в окне-примере | {_num(s.get('members'))} |\n"
            f"| код окна | {_num(length)} бит: место {_num(s.get('place_length'))} + содержание {_num(s.get('content_length'))} |\n"
            f"| единиц у примера | место {_num(s.get('place_bits'))}, содержание {_num(s.get('content_bits'))} |\n"
            f"| единиц на окно, в среднем | {_num(ones, '.0f')} |\n"
            f"| секунд: окна / коды | {_num(seconds.get('keyholes'), '.1f')} / {_num(seconds.get('codes'), '.1f')} |\n"
            f"| **Итого** | код окна разреженный: плотность {_num(density, '.1%')} |\n"
        ), n)
        size = s.get("map_size")
        same, random_same = s.get("same_label"), s.get("same_label_random")
        gain = None if not same or not random_same else same / random_same
        self._text(f"{LAYOUT}/about", (
            f"### Этап 3. Раскладка — уровень {n}\n\nКоды окон обучающих картинок ложатся на карту: похожие — рядом (§5). Пока "
            f"карта считается, точки разъезжаются на глазах, столбцы качества Ē (§5.8) растут. Точка — образец, цвет — цифра, на "
            f"которой он встретился первым; белое кольцо — точка, ближе всех к окну-примеру пятёрки. Энергия точки — насколько "
            f"она похожа на соседей (§5.7).\n\n"
            f"| | |\n|---|---|\n| точек | {_num(s.get('map_points', s.get('points')))} |\n"
            f"| сетка | {'—' if size is None else f'{size[1]}×{size[0]}'} |\n"
            f"| окно пятёрки на карте | у точки цифры {_num(s.get('five_label'))}, сходство {_num(s.get('five_similarity'), '.2f')} |\n"
            f"| соседи одной цифры | {_num(same, '.3f')} (у случайной карты {_num(random_same, '.3f')}) |\n"
            f"| секунд | {_num(seconds.get('memory'), '.1f')} |\n"
            f"| **Итого** | соседей одной цифры в {_num(gain, '.1f')} раза больше, чем у случайной карты |\n"
        ), n)
        self._text(f"{DETECTORS}/about", (
            f"### Этап 4. Детекторы и активации — уровень {n}\n\nНад кластерами похожих точек — круги детекторов, по слоям порогов "
            f"λ_d (§6.4, §7.1.10); пока они строятся, слои заполняются один за другим. Код окна-примера **зажигает** точки со "
            f"сходством не ниже λ_a (§6.1); в детекторы идут только те, вокруг которых тоже горит (энергия активации ≥ "
            f"{float(settings.ACTIVATION_LOCAL_ENERGY_THRESHOLD):.2f}, расширение статьи; погасшие — светло-серые). По ним "
            f"**срабатывают** детекторы (§6.5); оранжевые — сильнейшие.\n\n"
            f"| | |\n|---|---|\n| детекторов | {_num(s.get('detectors'))} (по слоям {_num(s.get('layers'))}) |\n"
            f"| λ_a | {_num(s.get('activation_lambda'), '.3f')} |\n| зажглось точек у примера | {_num(s.get('lit'))} из {_num(s.get('points'))}; "
            f"в детекторы идут {_num(s.get('kept'))} |\n"
            f"| сработало у примера | {_num(s.get('fired'))} |\n"
            f"| сработало на окно, в среднем | {_num(s.get('fired_mean'), '.1f')}; окон без срабатываний {_num(s.get('silent'), '.1%')} |\n"
            f"| секунд: детекторы / срабатывания | {_num(seconds.get('detectors'), '.1f')} / {_num(seconds.get('firing'), '.1f')} |\n"
            f"| **Итого** | у окна-примера сработало {_num(s.get('fired'))} детекторов из {_num(s.get('detectors'))} |\n"
        ), n)
        levels_note = ("уровни E(d) нормированы по обучающим окнам в [0, 1] (доработка вне статьи; §7.1.12 требует лишь [0, 1])" if str(settings.LADDER_LEVEL_NORMALISATION) == "train_max" else "уровни E(d) по формуле §6.5, сверху не ограничены")
        self._text(f"{EMBEDDINGS}/about", (
            f"### Этап 5. Эмбеддинги — уровень {n}\n\nСильнейшие сработавшие детекторы дают код детали (§6.5, §7.1.12); "
            f"деталь встаёт в центр окна, её сила — самый сильный детектор (П.1). Детали — элементы следующего уровня. Окно, на котором не сработал ни один детектор, даёт **пустую деталь**: в базу она не входит и не голосует.\n\n"
            f"| | |\n|---|---|\n| код детали | {_num(s.get('detail_length'))} бит, у примера {_num(s.get('detail_ones'))} единиц |\n"
            f"| единиц на деталь, в среднем | {_num(s.get('detail_bits'), '.1f')} |\n| разных деталей | {_num(s.get('distinct'))} |\n"
            f"| сила детали-примера | {_num(s.get('strength'), '.2f')} (в среднем {_num(s.get('strength_mean'), '.2f')}); {levels_note} |\n"
            f"| качество карты Ē (§5.8) | {_num(s.get('energy_mean'), '.3f')} |\n"
            f"| окон на один код детали (§5.13.2.5) | {_num(s.get('code_resolution'), '.2f')}; самый большой набор — {_num(s.get('code_largest'))} |\n"
            f"| пустых деталей / упёршихся в σ (§7.1.8.1) | {_num(s.get('detail_empty'), '.1%')} / {_num(s.get('detail_full'), '.1%')} |\n"
            f"| **Итого** | у пятёрки {_num(s.get('details'))} деталей — элементы уровня {n + 1} |\n"
        ), n)
        seen = s.get("picture_seen", s.get("seen"))
        verdict = "—" if seen is None else ("окно видит цифру целиком — цикл закончен" if seen >= float(settings.LADDER_TOP_COVERAGE)
                                            else f"детали идут на уровень {n + 1}")
        cycle_intro = (
            "Уровень 0: у каждого штриха теперь выученный код — его сильнейшие детекторы карты уровня 0 (§6.5, §7.1.12). "
            "С этим кодом штрих идёт в окна уровня 1; место, поле и сила штриха прежние. Серый круг — окно уровня 0: "
            "вокруг центра цифры, в нём один штрих; доля цифры в окне — один штрих из штрихов картинки."
            if n == 0 else
            "Детали уровня снова идут на этап 2; окно следующего уровня охватывает большую область (П.3). Круги — окна "
            "уровней вокруг одного места пятёрки; шкала cycle внизу добавляет уровни."
        )
        self._text(f"{CYCLE}/about", (
            f"### Этап 6. Цикл — уровень {n}\n\n{cycle_intro}\n\n"
            f"| | |\n|---|---|\n| радиус окна | {_num(s.get('field_radius', s.get('radius')), '.1f')} клетки |\n"
            f"| доля цифры в окне | {_num(seen, '.0%')} |\n"
            f"| окон на картинку | {_num(s.get('keyholes'), '.0f')} |\n| элементов в окне | {_num(s.get('elements'), '.1f')} |\n"
            f"| **Итого** | {verdict} |\n"
        ), n)
        five = self.top_numbers.get("five_answer") if vote else None
        self._text(f"{RECOGNITION}/about", (
            f"### Этап 7. Распознавание — уровень {n}\n\nОтклик картинки — сколько раз сработал каждый детектор на её окнах; ответ — "
            f"по самым похожим откликам обучающих (§2.2.5). Точность уровня появляется сразу после него. Наверху детали голосуют (пустые не голосуют; картинка без голосов или без отклика — неверная) "
            f"за цифру.\n\n"
            f"| | |\n|---|---|\n| по отклику уровня | {_share(s.get('alone'))} |\n| склейка уровней 1…{n} | {_share(s.get('glued'))} |\n"
            + ("" if not vote else "| голосование верхних деталей | " + ", ".join(f"k={k}: **{v:.1%}**" for k, v in vote.items()) + " |\n")
            + ("" if five is None else f"| проверочная пятёрка | ответ {five} |\n")
            + ("" if not self.top_numbers.get("five_outside") else
               f"| пятёрка прогона № {self.main_index} | не в проверочной группе (обучающая): ответ для неё не считается |\n")
            + f"| **Итого** | {_share(vote) if vote else _share(s.get('glued'))} |\n"
            + ("" if not vote or not self.top_numbers else f"\nВерно {self.top_numbers.get('right')} из {self.top_numbers.get('asked')} проверочных "
               "(k=1). Таблица ошибок: строка — цифра, столбец — ответ. Голоса — у проверочной цифры каждого класса.\n")
        ), n)

    def _summary(self, reports: list, accuracy: dict | None) -> None:
        rows = []
        total_seconds = 0.0
        first = self.levels.get(0, {}) if self._level0 else {}
        if "seconds" in first:
            # Уровень 0 узнавания не считает: его строка — окно, карта и детекторы.
            seconds = sum(first["seconds"].values())
            total_seconds += seconds
            rows.append(f"| 0 (штрихи) | {_num(first.get('radius'), '.1f')} | {_num(first.get('seen'), '.0%')} | {_num(first.get('points'))} | "
                        f"{_num(first.get('detectors'))} | — | — | {seconds:.1f} |")
        for report in reports:
            s = self.levels.get(int(report.number), {})
            seconds = sum(s.get("seconds", {}).values())
            total_seconds += seconds
            alone = (report.alone or {}).get(1)
            rows.append(
                f"| {report.number} | {report.field_radius:.1f} | {report.picture_seen:.0%} | {report.map_points} | {report.detectors} | "
                f"{report.same_label:.3f} | {'—' if alone is None else f'{alone:.1%}'} | {seconds:.1f} |"
            )
        vote = "—" if not accuracy else ", ".join(f"k={k}: **{v:.1%}**" for k, v in accuracy.items())
        top = self.top_numbers
        self._text(f"{OVERVIEW}/summary", (
            "### Прогон цикла\n\nЭтапы: 1 стимулы → 2 кодирование (↻ начало цикла) → 3 раскладка → 4 детекторы → 5 эмбеддинги → "
            "6 цикл (↺ конец) → 7 распознавание. Шкала **cycle** внизу — уровень: 0 — стимулы и уровень 0 (первичная "
            "раскладка штрихов), дальше уровни.\n\n"
            "| уровень | радиус окна | доля цифры | точек карты | детекторов | соседи одной цифры | по отклику, k=1 | секунд |\n"
            "|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) +
            f"\n| **Итого** | | | | | | голосование: {vote} | {total_seconds:.1f} |\n\n"
            f"Верхних деталей: {top.get('details', '—')}; верно {top.get('right', '—')} из {top.get('asked', '—')} проверочных. "
            f"Файл записи: `{LOGGER.recording_path or 'не пишется'}`."
        ))


class CycleBlueprint:
    """Раскладка окна Rerun для цикла: обзор и вкладка на каждый этап 1–7 (как README).

    У каждого вида в заголовке названы обе оси: у графиков X — уровень цикла (шкала cycle), Y — мерка с единицами; у
    столбцов X — что по оси (число штрихов, слой, цифра, шаг раскладки), Y — сколько; у картинок — в чём координаты.
    """

    TABS = ("Обзор", "1. Стимулы", "2. Кодирование ↻", "3. Раскладка", "4. Детекторы", "5. Эмбеддинги", "6. Цикл ↺", "7. Распознавание")
    LEVEL = "X: уровень"

    @classmethod
    def _chart(cls, path: str, title: str, y: str):  # noqa: ANN206
        import rerun.blueprint as rrb

        # Заголовок — что по Y (с единицами), затем ось X; так он помещается и в узкий вид.
        return rrb.TimeSeriesView(origin=path, name=f"{title}{y} · {cls.LEVEL}")

    @classmethod
    def _charts(cls, *charts: tuple[str, str, str], columns: int = 2):  # noqa: ANN206
        import rerun.blueprint as rrb

        return rrb.Grid(*(cls._chart(*chart) for chart in charts), grid_columns=columns)

    @classmethod
    def build(cls, active: str = "Обзор"):  # noqa: ANN206
        """Раскладка панелей; active — открытая вкладка (имя из TABS)."""
        import rerun.blueprint as rrb

        def text(origin: str, name: str = "Что видно"):  # noqa: ANN202
            return rrb.TextDocumentView(origin=origin, name=name)

        def spatial(origin: str, name: str, contents=None, **options):  # noqa: ANN001, ANN202
            return rrb.Spatial2DView(origin=origin, name=name, contents=contents or "$origin/**", background=(14, 15, 18), **options)

        def bars(origin: str, name: str):  # noqa: ANN202
            return rrb.BarChartView(origin=origin, name=name)

        picture = "X, Y — пиксели картинки (клетка сетки — 9 px)"
        cells = "X, Y — клетки карты"
        all_levels = rrb.VisibleTimeRange(TIMELINE, start=rrb.TimeRangeBoundary.infinite(), end=rrb.TimeRangeBoundary.cursor_relative())
        overview = rrb.Horizontal(
            rrb.Vertical(text(f"{OVERVIEW}/summary", "Сводка прогона"),
                         rrb.Horizontal(cls._chart(f"{RECOGNITION}/accuracy", "Точность", ", доля верных"),
                                        cls._chart(f"{OVERVIEW}/seconds", "Время шагов", ", с")), row_shares=[3, 2]),
            rrb.TextLogView(origin="/damp/log", name="Журнал: события по порядку, столбец cycle — уровень"),
            column_shares=[3, 2], name="Обзор",
        )
        stimuli = rrb.Horizontal(
            spatial(f"{STIMULI}/main", f"Цифра-пример и её штрихи · {picture}"),
            rrb.Vertical(
                rrb.Grid(*(spatial(f"{STIMULI}/digits/d{k}", f"{k}") for k in range(10)), grid_columns=5),
                rrb.Horizontal(bars(f"{STIMULI}/sticks_per_picture", "Картинок · X: штрихов на картинке"), text(f"{STIMULI}/about")),
                row_shares=[3, 2],
            ),
            column_shares=[1, 1], name="1. Стимулы",
        )
        coding = rrb.Horizontal(
            rrb.Vertical(spatial(f"{CODING}/keyhole", f"Окна пятёрки (тонкие) и окно-пример · {picture}"),
                         rrb.Tabs(spatial(f"{CODING}/angle_detectors", "Детекторы угла (§4.4) · кольцо — угол 0–360°"),
                                  spatial(f"{CODING}/place_detectors", "Детекторы места окна (§4.3, П.5) · X, Y — место в окне")), row_shares=[3, 2]),
            rrb.Vertical(spatial(f"{CODING}/code/place", "Код окна: место (OR) · клетка — бит, 64 в строке"),
                         spatial(f"{CODING}/code/content", "Код окна: содержание · клетка — бит, 64 в строке"), row_shares=[1, 4]),
            rrb.Vertical(text(f"{CODING}/about"), cls._charts((f"{CODING}/stats/ones", "Единиц в коде окна", ""),
                                                               (f"{CODING}/stats/density", "Плотность кода окна", ""), columns=1),
                         row_shares=[3, 2]),
            column_shares=[3, 3, 2], name="2. Кодирование ↻",
        )
        layout = rrb.Horizontal(
            rrb.Tabs(spatial(f"{LAYOUT}/map", f"Карта памяти: по ходу раскладки и итог; цвет — цифра, кольцо — окно пятёрки · {cells}"),
                     spatial(f"{LAYOUT}/energy", f"Энергия точек (§5.7): ярче — выше · {cells}"),
                     spatial(f"{LAYOUT}/glyphs", "Карта окон: в ячейке — что видит окно · X, Y — пиксели рисунка")),
            rrb.Vertical(text(f"{LAYOUT}/about"), bars(f"{LAYOUT}/live/quality", f"Качество Ē (§5.8) по ходу раскладки · X: оценка, "
                                                                                  f"раз в {settings.LAYOUT_ENERGY_EVAL_EVERY} шагов"),
                         cls._charts((f"{LAYOUT}/stats/same_label", "Соседи одной цифры", ", доля пар"),
                                     (f"{LAYOUT}/stats/points", "Точек карты", ""), columns=1), row_shares=[3, 2, 3]),
            column_shares=[3, 1], name="3. Раскладка",
        )
        detectors = rrb.Horizontal(
            spatial(f"{DETECTORS}", f"Детекторы по слоям (§6.4): цвет — слой, слои заполняются по ходу · {cells}",
                    contents=[f"{DETECTORS}/map", f"{DETECTORS}/layers/**"]),
            spatial(f"{DETECTORS}/activation", f"Окно пятёрки зажигает точки, срабатывают детекторы (§6.1, §6.5) · {cells}"),
            rrb.Vertical(text(f"{DETECTORS}/about"), bars(f"{DETECTORS}/per_layer", "Детекторов · X: слой (0 — широкий)"),
                         cls._charts((f"{DETECTORS}/stats/fired", "Сработало на окно", ", детекторов"),
                                     (f"{DETECTORS}/stats/detectors", "Детекторов на карте", ""), columns=1),
                         row_shares=[3, 2, 3]),
            column_shares=[3, 3, 2], name="4. Детекторы",
        )
        embeddings = rrb.Horizontal(
            spatial(f"{EMBEDDINGS}/details", f"Детали пятёрки: центры окон · {picture}"),
            rrb.Vertical(spatial(f"{EMBEDDINGS}/code", "Код детали-примера · клетка — бит, 32 в строке"),
                         bars(f"{EMBEDDINGS}/fired_levels", "Уровень активации E(d) · X: детекторы окна по силе")),
            rrb.Vertical(text(f"{EMBEDDINGS}/about"), cls._charts((f"{EMBEDDINGS}/stats/bits", "Единиц в коде детали", ""),
                                                                   (f"{EMBEDDINGS}/stats/strength", "Сила детали", ", E(d)"),
                                                                   columns=1), row_shares=[3, 2]),
            column_shares=[3, 2, 2], name="5. Эмбеддинги",
        )
        cycle = rrb.Horizontal(
            spatial(f"{CYCLE}/keyholes", f"Окна уровней вокруг одного места пятёрки (П.3): цвет и номер — уровень · {picture}",
                    time_ranges=all_levels),
            rrb.Vertical(text(f"{CYCLE}/about"), rrb.TextLogView(origin=f"{CYCLE}/marks", name="Начало и конец цикла · столбец cycle — уровень"),
                         cls._charts((f"{CYCLE}/stats/radius", "Радиус окна", ", клеток"), (f"{CYCLE}/stats/seen", "Доля цифры в окне", ""),
                                     (f"{CYCLE}/stats/elements", "Элементов в окне", ""), (f"{CYCLE}/stats/keyholes", "Окон на картинку", "")),
                         row_shares=[3, 2, 4]),
            column_shares=[3, 2], name="6. Цикл ↺",
        )
        recognition = rrb.Horizontal(
            rrb.Tabs(spatial(f"{RECOGNITION}/top_map", f"Карта верхних деталей: по ходу раскладки и итог; цвет — цифра · {cells}"),
                     spatial(f"{RECOGNITION}/top_glyphs", "Карта верхних деталей: что в точке · X, Y — пиксели рисунка")),
            rrb.Vertical(rrb.TensorView(origin=f"{RECOGNITION}/confusion", name="Ошибки (k=1) · строка — цифра, столбец — ответ, цвет — картинок"),
                         rrb.Grid(*(bars(f"{RECOGNITION}/votes/d{k}", f"Голосов у {k} · X: ответ") for k in range(10)),
                                  grid_columns=5)),
            rrb.Vertical(text(f"{RECOGNITION}/about"), cls._chart(f"{RECOGNITION}/accuracy", "Точность", ", доля верных"), row_shares=[2, 3]),
            column_shares=[3, 3, 2], name="7. Распознавание",
        )
        return rrb.Blueprint(
            rrb.Tabs(overview, stimuli, coding, layout, detectors, embeddings, cycle, recognition, active_tab=cls.TABS.index(active)),
            # Шкала cycle — активная (уровни), окно следует за новыми данными; внизу одной строкой — место видам.
            rrb.TimePanel(state="collapsed", timeline=TIMELINE, play_state="following"),
            rrb.SelectionPanel(state="collapsed"),
            rrb.BlueprintPanel(state="collapsed"),
        )
