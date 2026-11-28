"""Этап 3: память уровня — карта образцов кодов обучающих картинок тем же Layout, что первая раскладка (§5, §7.1.7)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from damp import settings
from damp.article_refs import OUTSIDE_ARTICLE
from damp.decode.map_quality import LabelMapQuality
from damp.layout.code_dictionary import CodeDictionary, LeaderDictionary, PairwiseSimilarity
from damp.layout.code_table import LayoutCodeTable
from damp.layout.damp_layout import Layout
from damp.layout.mnist_layout import (LayoutLambdaSchedule, LayoutParameterTuner, _layout_from_codes, _new_layout,
                                      _run_long_phase, _run_short_phase, _sparse_layout_options)
from damp.logging import LOGGER


@dataclass(frozen=True)
class LevelMemory:
    """Карта памяти уровня: раскладка образцов (§5), энергии её точек (§5.7) и откуда взят каждый образец.

    rows — номер входного кода у каждой точки карты; подпись точки — цифра, на которой её код встретился первым.
    energies — нормированные энергии Ê точек с радиусом ACTIVATION_POINT_ENERGY_RADIUS: по ним строятся детекторы
    и считается их срабатывание (§6.4, §6.5).
    """

    layout: Layout
    dictionary: CodeDictionary
    rows: "np.ndarray"
    quality: dict[str, float]
    name: str
    energies: list[float]

    @property
    def points(self) -> int:
        return self.dictionary.count

    def release(self) -> None:
        """Освободить память видеокарты под раскладкой."""
        self.layout.release_device()


class MemoryMap:
    """Карта памяти из кодов: словарь образцов (похожие коды сливаются, §7.1.7) и раскладка тем же Layout (§5).

    Раскладка — шаг кеша (system/damp/cache.py): тот же набор образцов с теми же подписями — карта, её энергии и
    порог λ берутся с диска, Layout восстанавливается по ячейкам точек.
    """

    MODULES = ("damp.layout.level_memory", "damp.layout.far_force", "damp.layout.rescue", "damp.layout.smooth_start")  # far_force: α смешанного ядра (Р2), rescue: v4.15 — в ключ кеша
    # Модули из импортов шага, которые на карту не влияют, — в ключ кеша не входят (С3, docs/SPEEDUP.md): раскраска точек
    # в псевдоцвета (§5.14, encoding.mnist_codes — она тянет кодер угла и разбор картинок) идёт только в рисунки карты
    # (Layout._colors_array: журнал, render_*), мерка качества карты (decode.map_quality) считается после шага и в его
    # выход не входит. Сверка: CACHE_VERIFY_SHARE пересчитывает долю попаданий и сравнивает бит в бит.
    IGNORED_MODULES = ("damp.encoding.mnist_codes", "damp.decode.map_quality")

    def __init__(self) -> None:
        self._colorizer = None

    @property
    def colorizer(self):  # noqa: ANN201
        """Раскраска точек карты в псевдоцвета (§5.14) — та же, что у первой раскладки."""
        if self._colorizer is None:
            from damp.encoding.mnist_codes import _build_colorizer

            self._colorizer = _build_colorizer()
        return self._colorizer

    @staticmethod
    def samples(
        codes: "np.ndarray",
        code_length: int,
        picture: "np.ndarray",
        train_pictures: int,
        *,
        merge_similarity: float,
        max_points: int,
        shuffle_seed: int | None,
    ) -> tuple[CodeDictionary, "np.ndarray"]:
        """Образцы карты из кодов обучающих картинок: (словарь, номера кодов обучающих картинок в порядке обхода).

        shuffle_seed — обход в случайном порядке, чтобы с пределом карты на неё попали коды всех картинок;
        None — по порядку.
        """
        train_rows = np.flatnonzero(np.asarray(picture) < int(train_pictures))
        if shuffle_seed is not None:
            train_rows = np.random.default_rng(int(shuffle_seed)).permutation(train_rows)
        dictionary = LeaderDictionary(
            similarity_threshold=float(merge_similarity),
            max_size=int(max_points),
            use_gpu=settings.ACTIVATION_USE_GPU,
        ).select(codes, int(code_length), order=train_rows)  # обход train_rows без копии всех кодов
        return dictionary, train_rows

    @staticmethod
    def lambda_schedule(bits: "np.ndarray", name: str) -> LayoutLambdaSchedule:
        """Пороги λ раскладки кодов — квантилями их попарных сходств (§7.1.7)."""
        quantiles = (settings.LADDER_LAYOUT_LAMBDA_QUANTILE, settings.LADDER_LAYOUT_LAMBDA_MAX_QUANTILE)
        start, maximum = PairwiseSimilarity(seed=settings.LADDER_RANDOM_SEED).quantiles(bits, quantiles, name=name)
        return LayoutLambdaSchedule.from_quantiles(start, maximum, start_quantile=quantiles[0], maximum_quantile=quantiles[1])

    @staticmethod
    def _table(bits: "np.ndarray", labels: "np.ndarray") -> LayoutCodeTable:
        """Коды карты, сгруппированные по цвету класса: тот же порядок точек при счёте и при восстановлении."""
        return LayoutCodeTable.from_bits(
            bits,
            hues=(labels * settings.LADDER_CLASS_HUE_STEP).tolist(),
            labels=[int(label) for label in labels],
        ).grouped_by_hue()

    @staticmethod
    def _tuner() -> LayoutParameterTuner:
        return LayoutParameterTuner(
            empty_ratio=settings.LAYOUT_EMPTY_RATIO,
            start_radius_factor=settings.LAYOUT_ADAPTIVE_RADIUS_START_FACTOR,
        )

    @staticmethod
    def level(name: str) -> int:
        """Номер уровня по имени карты («level3» → 3); у других карт (первичная, перекладки стенда) — 0."""
        return int(name[5:]) if name.startswith("level") and name[5:].isdigit() else 0

    @staticmethod
    def level_settings(level: int) -> dict[str, object]:
        """Настройки раскладки, действующие с уровня, — их действующее значение на уровне level (вне статьи).

        Р5 «лучшая из K»: LADDER_LAYOUT_BEST_OF раскладок с уровня LADDER_LAYOUT_BEST_OF_FROM (1 — одна раскладка);
        v4.15 «лучший ответ»: LADDER_LAYOUT_RESCUE_PASSES проходов с уровня LADDER_LAYOUT_RESCUE_FROM (0 — выключен; доля
        и σ — только при включённом). Счёт (lay_out, _rescue) берёт значения отсюда, и ровно они идут в ключ кеша карты
        уровня (С3): правка «с уровня 3» не сбрасывает карты уровней 1–2, у которых действующее значение то же."""
        tries = int(settings.LADDER_LAYOUT_BEST_OF) if 0 < int(settings.LADDER_LAYOUT_BEST_OF_FROM) <= int(level) else 1
        passes = int(settings.LADDER_LAYOUT_RESCUE_PASSES)
        rescue = passes > 0 and int(level) >= int(settings.LADDER_LAYOUT_RESCUE_FROM)
        coarse = int(settings.LADDER_LAYOUT_COARSE_NEIGHBOURS) if 0 < int(settings.LADDER_LAYOUT_COARSE_FIRST_FROM) <= int(level) else 0
        smooth = 0 < int(settings.LADDER_LAYOUT_SMOOTH_START_FROM) <= int(level)
        again = int(settings.LADDER_LAYOUT_RELAYOUT_PASSES)
        relayout = again > 0 and 0 < int(settings.LADDER_LAYOUT_RELAYOUT_FROM) <= int(level)
        return {
            "LADDER_LAYOUT_SMOOTH_START_FROM": None,
            "LADDER_LAYOUT_SMOOTH_START": bool(smooth),
            "LADDER_LAYOUT_SMOOTH_START_RADIUS": float(settings.LADDER_LAYOUT_SMOOTH_START_RADIUS) if smooth else None,
            "LADDER_LAYOUT_SMOOTH_START_PERPLEXITY": float(settings.LADDER_LAYOUT_SMOOTH_START_PERPLEXITY) if smooth else None,
            "LADDER_LAYOUT_COARSE_FIRST_FROM": None,
            "LADDER_LAYOUT_COARSE_NEIGHBOURS": coarse,
            "LADDER_LAYOUT_COARSE_RADIUS": float(settings.LADDER_LAYOUT_COARSE_RADIUS) if coarse > 0 else None,
            "LADDER_LAYOUT_RELAYOUT_PASSES": again if relayout else 0,
            "LADDER_LAYOUT_RELAYOUT_FROM": None,
            "LADDER_LAYOUT_RELAYOUT_QUANTILE": float(settings.LADDER_LAYOUT_RELAYOUT_QUANTILE) if relayout else None,
            "LADDER_LAYOUT_RELAYOUT_RADIUS": float(settings.LADDER_LAYOUT_RELAYOUT_RADIUS) if relayout else None,
            "LADDER_LAYOUT_BEST_OF": tries if tries > 1 else 1,
            "LADDER_LAYOUT_BEST_OF_FROM": None,
            "LADDER_LAYOUT_BEST_OF_MEASURE": str(settings.LADDER_LAYOUT_BEST_OF_MEASURE) if tries > 1 else None,
            "LADDER_LAYOUT_RESCUE_PASSES": passes if rescue else 0,
            "LADDER_LAYOUT_RESCUE_FROM": None,
            "LADDER_LAYOUT_RESCUE_SHARE": float(settings.LADDER_LAYOUT_RESCUE_SHARE) if rescue else None,
            "LADDER_LAYOUT_RESCUE_SIGMA": float(settings.LADDER_LAYOUT_RESCUE_SIGMA) if rescue else None,
        }

    def lay_out(self, bits: "np.ndarray", labels: "np.ndarray", name: str) -> Layout:
        """Раскладка кодов тем же Layout и с той же настройкой, что первая; цвет точки — класс цифры.

        Р5 (задача #20, вне статьи по способу, по статье по мере): с уровня LADDER_LAYOUT_BEST_OF_FROM карта раскладывается
        LADDER_LAYOUT_BEST_OF раз с разными зёрнами, остаётся раскладка с наименьшей меркой LADDER_LAYOUT_BEST_OF_MEASURE —
        без меток цифр: "far" — дальняя энергия статьи φ = Σ s·d по парам с s ≥ λ (§5.5.1: «penalises strongly correlated
        points that are distant»), "near" — среднее расстояние на карте до 10 ближайших по коду соседей (near_distance).
        Попытка — та же раскладка уровня, что без выбора (гладкий старт и две фазы — как настроено на уровне), с зерном
        LAYOUT_RANDOM_SEED + 1009·номер: попытка 0 — раскладка без выбора. До 0.13.5 попытки шли со случайного старта, и
        на картах с гладким стартом выбор терял его (стенд верха, 750 картинок: −0.043 М1u)."""
        table = self._table(bits, labels)
        params, schedule = self._tuner().tune(table.count), self.lambda_schedule(bits, name)
        level = self.level(name)
        tries = int(self.level_settings(level)["LADDER_LAYOUT_BEST_OF"])
        if tries <= 1:
            layout = self._smooth_start(table, params, schedule, bits, labels, name, level)
            return self._rescue(table, self._relayout(layout, params, schedule, bits, name, level), name, level)
        measure = str(self.level_settings(level)["LADDER_LAYOUT_BEST_OF_MEASURE"])
        if measure not in ("far", "near"):
            raise ValueError(f"LADDER_LAYOUT_BEST_OF_MEASURE: {measure!r} — ждём \"far\" или \"near\"")
        best, best_energy, energies = None, None, []
        for attempt in range(tries):
            with settings.overridden({"LAYOUT_RANDOM_SEED": int(settings.LAYOUT_RANDOM_SEED) + 1009 * attempt}):
                layout = self._smooth_start(table, params, schedule, bits, labels, name, level)
            if measure == "near":
                energy = self.near_distance(table.bits(), layout.cells_array())
            else:
                energy = self.far_energy(table.bits(), layout.cells_array(), float(layout.lambda_threshold))
            energies.append(round(energy, 4))
            if best_energy is None or energy < best_energy:
                if best is not None:
                    best.release_device()
                best, best_energy = layout, energy
            else:
                layout.release_device()
        LOGGER.event("layout.best_of", section=OUTSIDE_ARTICLE,
                     data={"name": name, "tries": tries, "measure": measure, "energies": energies,
                           "chosen": energies.index(round(best_energy, 4)),
                           "note": "лучшая из K раскладок по мерке без меток: far — дальняя энергия статьи φ = Σ s·d "
                                   "(§5.5.1), near — расстояние на карте до 10 ближайших по коду соседей"})
        return self._rescue(table, self._relayout(best, params, schedule, bits, name, level), name, level)

    @staticmethod
    def shared_neighbours(bits: "np.ndarray", neighbours: int) -> "np.ndarray":
        """Грубый код точки (план v19.12, без меток): набор её neighbours ближайших по косинусу кодов и она сама — строка
        N бит. Косинус таких кодов — доля общих соседей: сходство второго порядка, одного масштаба у всех точек."""
        x = np.asarray(bits, dtype=np.float32)
        x = x / np.maximum(np.sqrt(x.sum(axis=1, keepdims=True)), 1e-6)
        count = x.shape[0]
        k = max(1, min(int(neighbours), count - 1))
        code = np.zeros((count, count), dtype=np.uint8)
        for start in range(0, count, 1024):
            sim = x[start:start + 1024] @ x.T
            rows = np.arange(start, min(start + 1024, count))
            sim[np.arange(rows.shape[0]), rows] = -1.0
            near = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            code[rows[:, None], near] = 1
            code[rows, rows] = 1
        return code

    def _smooth_start(self, table: LayoutCodeTable, params, schedule: LayoutLambdaSchedule, bits: "np.ndarray",  # noqa: ANN001
                      labels: "np.ndarray", name: str, level: int) -> Layout:
        """Раскладка карты; с уровня LADDER_LAYOUT_SMOOTH_START_FROM (0 — нет) — с гладкого старта (план v19.15, вне
        статьи; system/damp/layout/smooth_start.py): точки ставятся не в случайные клетки, а по сходству кодов (без меток),
        затем обычная дальняя фаза с радиуса LADDER_LAYOUT_SMOOTH_START_RADIUS · ширина карты и ближняя фаза. Малый радиус
        наводит местный порядок и не рвёт собранные классы (радиус 0.2 уже теряет треть выигрыша). Выключено — как раньше
        (в том числе две фазы со сменой кода, _coarse_first). Действующие на уровне значения — level_settings."""
        plan = self.level_settings(level)
        if not plan["LADDER_LAYOUT_SMOOTH_START"]:
            return self._coarse_first(table, params, schedule, bits, labels, name, level)
        from damp.layout.smooth_start import LARGE_LIMIT, LIMIT, smooth_cells

        if table.count > LARGE_LIMIT:
            LOGGER.event("layout.smooth_start.skipped.warning", section=OUTSIDE_ARTICLE,
                         data={"name": name, "points": int(table.count), "limit": int(LARGE_LIMIT),
                               "note": "карта крупнее предела гладкого старта — раскладывается со случайного старта"})
            return self._coarse_first(table, params, schedule, bits, labels, name, level)
        layout = _new_layout(table, params, self.colorizer, lambda_threshold=schedule.start)
        cells = smooth_cells(np.asarray(table.bits(), dtype=np.uint8), int(layout.height), int(layout.width),
                             perplexity=float(plan["LADDER_LAYOUT_SMOOTH_START_PERPLEXITY"]),
                             seed=int(settings.LAYOUT_RANDOM_SEED), use_gpu=bool(settings.LAYOUT_USE_GPU))
        layout.place_cells(cells, log=False)
        radius = max(int(settings.LAYOUT_ADAPTIVE_RADIUS_MIN),
                     int(round(layout.width * float(plan["LADDER_LAYOUT_SMOOTH_START_RADIUS"]))))
        _run_long_phase(layout, params, step_offset=1, layout_path=None, start_radius=radius, lambda_max=schedule.maximum)
        steps = int(layout.last_steps)
        _run_short_phase(layout, params, step_offset=1 + steps, similarity_cutoff=schedule.short_cutoff, lambda_max=schedule.maximum)
        LOGGER.event("layout.smooth_start", section=OUTSIDE_ARTICLE,
                     data={"name": name, "points": int(cells.shape[0]), "radius": radius, "long_steps": steps,
                           "path": "blocked" if int(cells.shape[0]) > LIMIT else "dense",
                           "perplexity": float(plan["LADDER_LAYOUT_SMOOTH_START_PERPLEXITY"]),
                           "note": "гладкий старт раскладки (план v19.15, вне статьи): точки расставлены по сходству кодов без "
                                   "меток, затем дальняя фаза с малого радиуса и ближняя фаза"})
        return layout

    def _coarse_first(self, table: LayoutCodeTable, params, schedule: LayoutLambdaSchedule, bits: "np.ndarray",  # noqa: ANN001
                      labels: "np.ndarray", name: str, level: int) -> Layout:
        """Раскладка карты; с уровня LADDER_LAYOUT_COARSE_FIRST_FROM (0 — нет) — в две фазы со сменой кода (план v19.12):
        сначала со случайного старта раскладываются грубые коды (shared_neighbours, LADDER_LAYOUT_COARSE_NEIGHBOURS соседей),
        затем настоящие коды ставятся на те же клетки и доводятся дальней фазой с радиуса LADDER_LAYOUT_COARSE_RADIUS ·
        ширина карты и ближней фазой. Статья: «initially, codes with maximum overlap and low resolution were used; as the
        space was laid out, the overlap decreased» (§5.13.2.4); замена кода на месте и раскладка заново (§7.1.8.5). Стенд
        верха (те же коды ур. 5, 6 зёрен × 2 раскладки): +0.036…0.039 М1u, t 4.7–6.8; вторая фаза без смены кода — +0.001.
        Выключено — обычная раскладка, бит в бит прежняя. Действующие на уровне значения — level_settings."""
        plan = self.level_settings(level)
        neighbours = int(plan["LADDER_LAYOUT_COARSE_NEIGHBOURS"])
        if neighbours <= 0:
            return _layout_from_codes(table, params, self.colorizer, lambda_schedule=schedule)
        rough_bits = self.shared_neighbours(bits, neighbours)
        rough = _layout_from_codes(self._table(rough_bits, labels), params, self.colorizer,
                                   lambda_schedule=self.lambda_schedule(rough_bits, f"{name}.coarse"))
        cells = np.asarray(rough.cells_array(), dtype=np.int64)  # порядок точек тот же: обе таблицы сгруппированы одинаково
        rough.release_device()
        layout = _new_layout(table, params, self.colorizer, initial_cells=cells, lambda_threshold=schedule.start)
        radius = max(int(settings.LAYOUT_ADAPTIVE_RADIUS_MIN), int(round(layout.width * float(plan["LADDER_LAYOUT_COARSE_RADIUS"]))))
        _run_long_phase(layout, params, step_offset=1, layout_path=None, start_radius=radius, lambda_max=schedule.maximum)
        steps = int(layout.last_steps)
        _run_short_phase(layout, params, step_offset=1 + steps, similarity_cutoff=schedule.short_cutoff, lambda_max=schedule.maximum)
        LOGGER.event("layout.coarse_first", section=OUTSIDE_ARTICLE,
                     data={"name": name, "points": int(cells.shape[0]), "neighbours": neighbours, "radius": radius, "long_steps": steps,
                           "note": "раскладка в две фазы со сменой кода (план v19.12; §5.13.2.4, §7.1.8.5): грубые коды (общие "
                                   "соседи) со случайного старта, затем настоящие коды с тех же клеток и малого радиуса"})
        return layout

    @staticmethod
    def _relayout(layout: Layout, params, schedule: LayoutLambdaSchedule, bits: "np.ndarray", name: str,  # noqa: ANN001
                  level: int) -> Layout:
        """План v9.8 (§7.1.8.3 — повторная раскладка с «растворением»): с уровня LADDER_LAYOUT_RELAYOUT_FROM готовая карта
        раскладывается ещё LADDER_LAYOUT_RELAYOUT_PASSES раз: λ опускается до квантиля LADDER_LAYOUT_RELAYOUT_QUANTILE
        попарных сходств (ниже начального — пространство «растворяется», куски класса снова чувствуют друг друга),
        дальняя фаза идёт заново с радиуса LADDER_LAYOUT_RELAYOUT_RADIUS · ширина карты на готовых клетках, λ растёт
        ступенями, как в первой раскладке (§7.1.7), затем ближняя фаза. После — λ не ниже начального. 0 — выкл., карта
        бит в бит прежняя. Действующие на уровне значения — level_settings (они же в ключе кеша карты)."""
        plan = MemoryMap.level_settings(level)
        passes = int(plan["LADDER_LAYOUT_RELAYOUT_PASSES"])
        if passes <= 0:
            return layout
        (low,) = PairwiseSimilarity(seed=settings.LADDER_RANDOM_SEED).quantiles(
            bits, (float(plan["LADDER_LAYOUT_RELAYOUT_QUANTILE"]),), name=f"{name}.relayout")
        low = min(float(low), float(schedule.start))
        radius = max(int(settings.LAYOUT_ADAPTIVE_RADIUS_MIN), int(layout.width * float(plan["LADDER_LAYOUT_RELAYOUT_RADIUS"])))
        report = []
        for _ in range(passes):
            before = float(layout.lambda_threshold)
            layout.set_similarity_params(lambda_threshold=low)
            _run_long_phase(layout, params, step_offset=1, layout_path=None, start_radius=radius, lambda_max=schedule.maximum)
            long_steps = int(layout.last_steps)
            if float(layout.lambda_threshold) < float(schedule.start):
                layout.set_similarity_params(lambda_threshold=float(schedule.start))
            _run_short_phase(layout, params, step_offset=1 + long_steps, similarity_cutoff=schedule.short_cutoff,
                             lambda_max=schedule.maximum)
            report.append({"lambda_before": round(before, 4), "lambda_low": round(low, 4), "radius": radius,
                           "long_steps": long_steps, "lambda_after": round(float(layout.lambda_threshold), 4)})
        LOGGER.event("layout.relayout", section=OUTSIDE_ARTICLE,
                     data={"name": name, "passes": report,
                           "note": "повторная раскладка с растворением (план v9.8, §7.1.8.3): λ вниз до квантиля, дальняя "
                                   "фаза заново с части ширины карты на готовых клетках, λ растёт ступенями, ближняя фаза"})
        return layout

    @staticmethod
    def _rescue(table: LayoutCodeTable, layout: Layout, name: str, level: int) -> Layout:
        """План v4.15: с уровня LADDER_LAYOUT_RESCUE_FROM — LADDER_LAYOUT_RESCUE_PASSES проходов «лучший ответ»
        (system/damp/layout/rescue.py): заблудшие точки (доля LADDER_LAYOUT_RESCUE_SHARE с наибольшим разрывом между
        лучшей ячейкой и своей) меняются местами с жителем лучшей ячейки, если растёт энергия Σ (s − λ)·g(d), g — гаусс
        с σ = LADDER_LAYOUT_RESCUE_SIGMA (масштаб кластера). 0 проходов — выкл., раскладка бит в бит прежняя.
        Действующие на уровне значения — level_settings (они же в ключе кеша карты)."""
        plan = MemoryMap.level_settings(level)
        passes = int(plan["LADDER_LAYOUT_RESCUE_PASSES"])
        if passes <= 0:
            return layout
        from damp.layout.rescue import rescue_cells

        cells, report = rescue_cells(table.bits(), layout.cells_array(), int(layout.height), int(layout.width),
                                     float(layout.lambda_threshold), passes=passes,
                                     share=float(plan["LADDER_LAYOUT_RESCUE_SHARE"]),
                                     sigma=float(plan["LADDER_LAYOUT_RESCUE_SIGMA"]))
        layout.place_cells(cells, log=False)
        LOGGER.event("layout.rescue", section=OUTSIDE_ARTICLE,
                     data={"name": name, "points": int(cells.shape[0]), "passes": report,
                           "note": "проход «лучший ответ» (план v4.15): заблудшие точки — к месту с наибольшим полем "
                                   "Σ (s − λ)·g(d), обмен при росте энергии; без меток"})
        return layout

    @staticmethod
    def far_energy(bits: "np.ndarray", cells: "np.ndarray", threshold: float) -> float:
        """Дальняя энергия статьи на готовой карте: φ = Σ s·d по парам со сходством s ≥ λ (косинус кодов), делённая на
        число таких пар (§5.5.1). Меньше — похожие точки ближе друг к другу."""
        codes = np.asarray(bits, dtype=np.float32)
        norms = np.sqrt(np.maximum(codes.sum(axis=1), 1e-9))
        cells = np.asarray(cells, dtype=np.float32)
        total, pairs, step = 0.0, 0, 512
        for start in range(0, codes.shape[0], step):
            sim = (codes[start:start + step] @ codes.T) / (norms[start:start + step, None] * norms[None, :])
            dist = np.sqrt(((cells[start:start + step, None, :] - cells[None, :, :]) ** 2).sum(axis=2))
            keep = sim >= threshold
            keep[np.arange(keep.shape[0]), np.arange(start, start + keep.shape[0])] = False
            total += float((sim[keep] * dist[keep]).sum())
            pairs += int(keep.sum())
        return total / max(1, pairs)

    NEAR_NEIGHBOURS = 10

    @classmethod
    def near_distance(cls, bits: "np.ndarray", cells: "np.ndarray") -> float:
        """Мерка соседства на готовой карте (вне статьи по способу; по смыслу §5.1 — похожие коды рядом): среднее по точкам
        расстояние на карте до NEAR_NEIGHBOURS ближайших по коду соседей (косинус кодов; при равенстве — меньший номер).
        Меньше — ближайшие по коду точки лежат рядом. Замер на верхней карте в 750 точек (48 раскладок): корреляция с
        целостью классов −0.5, у дальней энергии φ — −0.16; выбор лучшей из 3 даёт +0.013…0.020 М1u."""
        codes = np.asarray(bits, dtype=np.float32)
        count = int(codes.shape[0])
        if count < 2:
            return 0.0
        norms = np.sqrt(np.maximum(codes.sum(axis=1), 1e-9))
        place = np.asarray(cells, dtype=np.float64)
        nearest = min(int(cls.NEAR_NEIGHBOURS), count - 1)
        total, step = 0.0, 512
        for start in range(0, count, step):
            sim = (codes[start:start + step] @ codes.T) / (norms[start:start + step, None] * norms[None, :])
            sim[np.arange(sim.shape[0]), np.arange(start, start + sim.shape[0])] = -np.inf
            order = np.argsort(-sim, axis=1, kind="stable")[:, :nearest]
            total += float(np.sqrt(((place[start:start + step, None, :] - place[order]) ** 2).sum(axis=2)).sum())
        return total / (count * nearest)

    def restore(self, bits: "np.ndarray", labels: "np.ndarray", cells: "np.ndarray", width: int, lambda_threshold: float,
                pair_radius: int | None) -> Layout:
        """Раскладка из кеша: те же коды в том же порядке, ячейки точек, порог λ и радиус пар (§5.1.2)."""
        table = self._table(bits, labels)
        params = self._tuner().tune(table.count)
        layout = Layout(
            table,
            grid_size=int(width),
            empty_ratio=settings.LAYOUT_EMPTY_RATIO,
            similarity=settings.LAYOUT_SIMILARITY,
            lambda_threshold=float(lambda_threshold),
            eta=settings.LAYOUT_ETA,
            seed=settings.LAYOUT_RANDOM_SEED,
            use_gpu=settings.LAYOUT_USE_GPU,
            long_subset_size=params.long_subset_size,
            long_subset_refresh=params.long_subset_refresh,
            colorizer=self.colorizer,
            long_distance=settings.LAYOUT_LONG_DISTANCE,
            short_distance=settings.LAYOUT_SHORT_DISTANCE,
            long_kernel=settings.LAYOUT_LONG_KERNEL,
            long_kernel_radius_min=settings.LAYOUT_LONG_KERNEL_RADIUS_MIN,
            long_kernel_width=settings.LAYOUT_LONG_KERNEL_WIDTH,
            component_period=settings.LAYOUT_COMPONENT_PERIOD,
            **_sparse_layout_options(),
        )
        layout.restore_positions([(int(y), int(x)) for y, x in np.asarray(cells).tolist()])
        layout.pair_radius = pair_radius
        return layout

    def remember(self, dictionary: CodeDictionary, rows: "np.ndarray", labels: "np.ndarray", name: str) -> LevelMemory:
        """Карта образцов словаря; labels — цифра каждой точки, rows — номер входного кода каждой точки."""
        from damp.cache import CACHE

        bits = dictionary.bits()
        labels = np.asarray(labels)
        computed: dict[str, Layout] = {}

        def compute() -> dict[str, object]:
            layout = self.lay_out(bits, labels, name)
            computed["layout"] = layout
            energies = layout.point_energies(settings.ACTIVATION_POINT_ENERGY_RADIUS)
            return {
                "cells": layout.cells_array(), "width": int(layout.width), "height": int(layout.height),
                "lambda": float(layout.lambda_threshold), "pair_radius": layout.pair_radius,
                "energies": np.asarray(energies, dtype=np.float64),
            }

        found = CACHE.run("layout.memory", {"codes": dictionary.packed, "code_length": dictionary.code_length, "labels": labels},
                          compute, modules=self.MODULES, ignore=self.IGNORED_MODULES,
                          level_settings=self.level_settings(self.level(name)))
        layout = computed.get("layout")
        if layout is None:
            layout = self.restore(bits, labels, found["cells"], int(found["width"]), float(found["lambda"]), found["pair_radius"])
            if (int(layout.width), int(layout.height)) != (int(found["width"]), int(found["height"])):
                raise RuntimeError(f"{name}: раскладка из кеша восстановилась с другой сеткой")
            LOGGER.event(
                "layout.memory.restore",
                section=OUTSIDE_ARTICLE,
                data={"name": name, "points": dictionary.count, "width": layout.width, "height": layout.height,
                      "note": "карта уровня восстановлена из кеша шагов по ячейкам точек; энергии точек — из кеша"},
            )
        energies = [float(value) for value in np.asarray(found["energies"]).tolist()]
        return LevelMemory(layout=layout, dictionary=dictionary, rows=np.asarray(rows), quality=LabelMapQuality().measure(layout),
                           name=name, energies=energies)
