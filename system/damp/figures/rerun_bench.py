"""Стенд честного сравнения в Rerun: таблица результатов, столбцы точности методов, ход обучения CNN (вне статьи).

Стенд (decode/bench.py) открывает свою запись «bench» и после каждого набора и каждого прогона DAMP открывает её
заново: окно показывает ход сравнения. Прогоны DAMP идут в своих записях цикла со всеми вкладками этапов.

Пути записи: `bench/table` (таблица Markdown), `bench/accuracy/validation` и `bench/accuracy/test` (столбцы точности
методов в порядке таблицы), `bench/cnn/<конфигурация>/<набор>/loss|validation_accuracy` (ход обучения по эпохам;
пишет ConvNetBaseline). Раскладка панелей ссылается на эти пути.
"""

from __future__ import annotations

from damp.article_refs import OUTSIDE_ARTICLE
from damp.logging import LOGGER

BENCH = "bench"
TABLE = f"{BENCH}/table"
ACCURACY = f"{BENCH}/accuracy"
CNN = f"{BENCH}/cnn"
EPOCH = "epoch"  # шкала хода обучения CNN


class BenchBlueprint:
    """Раскладка окна Rerun для стенда: таблица, столбцы точности по методам, ход обучения CNN, журнал.

    В заголовках видов названы оси: у столбцов X — метод (порядок как в таблице), Y — доля верных ответов; у графиков
    обучения X — эпоха, Y — потеря или доля верных ответов на validation.
    """

    @classmethod
    def build(cls):  # noqa: ANN206
        import rerun.blueprint as rrb

        table = rrb.TextDocumentView(origin=TABLE, name="Результаты стенда: среднее ± стандартное отклонение по наборам, 95 % интервал")
        bars = rrb.Horizontal(
            rrb.BarChartView(origin=f"{ACCURACY}/validation", name="Точность на validation (часть report) · X: метод (порядок таблицы), Y: доля верных"),
            rrb.BarChartView(origin=f"{ACCURACY}/test", name="Точность на test (выбранные настройки) · X: метод, Y: доля верных"),
        )
        training = rrb.Vertical(
            rrb.TimeSeriesView(
                origin=CNN, contents=[f"{CNN}/**/loss"], name="Ход обучения CNN: потеря · X: эпоха, Y: cross-entropy"
            ),
            rrb.TimeSeriesView(
                origin=CNN, contents=[f"{CNN}/**/validation_accuracy"],
                name="Ход обучения CNN: точность на select (по ней ранняя остановка; итог — по report) · X: эпоха, Y: доля верных",
            ),
            rrb.TextLogView(origin="/damp/log", name="Журнал: события стенда"),
            row_shares=[2, 2, 2],
        )
        return rrb.Blueprint(
            rrb.Horizontal(rrb.Vertical(table, bars, row_shares=[3, 2]), training, column_shares=[3, 2]),
            rrb.TimePanel(state="collapsed", timeline=EPOCH),
            rrb.SelectionPanel(state="collapsed"),
        )


class BenchTable:
    """Таблица результатов стенда в Markdown: строка на метод и режим аугментации, последняя — «Итого»."""

    TITLES = {"damp": "DAMP", "pixels": "kNN по пикселям", "cnn": "CNN"}

    @staticmethod
    def score(summary: dict | None) -> str:
        """«93.4 ± 1.2 [92.1; 94.7]» в процентах; нет данных — прочерк."""
        if summary is None:
            return "—"
        text = f"{100 * summary['mean']:.1f} ± {100 * summary['std']:.1f}"
        if summary["ci_low"] is not None:
            text += f" [{100 * summary['ci_low']:.1f}; {100 * summary['ci_high']:.1f}]"
        return text

    @staticmethod
    def points(value: float | None) -> str:
        return "—" if value is None else f"{100 * value:+.1f} п.п."

    @classmethod
    def order(cls, result: dict) -> list[dict]:
        """Строки методов в порядке таблицы (он же — порядок столбцов точности)."""
        return list(result["methods"])

    @classmethod
    def markdown(cls, result: dict, *, final: bool = False) -> str:
        protocol = result["protocol"]
        done, sets = result["sets_done"], protocol["sets"]
        head = (
            f"**Стенд честного сравнения** · версия {result['version']} · наборов {done} из {sets} "
            f"({'итог' if final else 'предварительно: настройки пока выбраны по готовым наборам'})\n\n"            f"**Что видно.** Строка на метод и режим аугментации: настройка, выбранная по select, точность на report (столбец validation) и на test, среднее ± отклонение по наборам и 95 % интервал. Последняя строка — разрыв DAMP до CNN с парным интервалом. Столбцы ниже — те же точности, справа — ход обучения CNN. Вне статьи.\n\n"
            f"Картинки — случайная выборка номеров MNIST (зерно {protocol.get('seed', '—')}), не сплошные блоки. "
            f"Обучающих в наборе {protocol['train']}, validation {protocol['validation']}, test "
            f"{protocol['test'] or 'не считается'}; настройки и остановка CNN — по select ({protocol['validation_select']} картинок "
            f"validation), итог — по report ({protocol['validation_report']}), в выборе не участвовавшей; "
            f"устойчивость — {protocol['pixel_pictures']} картинок validation, сдвинутых на {tuple(protocol['pixel_shift'])} px: "
            f"доля совпавших ответов.\n\n"
        )
        lines = [
            "| Метод | Аугментация | Выбранная настройка | validation, % | test, % | Ответы при сдвиге |",
            "|---|---|---|---|---|---|",
        ]
        for row in cls.order(result):
            title = cls.TITLES.get(row["method"], row["method"])
            if not row["supported"]:
                lines.append(f"| {title} | {row['augment']} | {row['note']} | — | — | — |")
                continue
            robust = row["robustness"]
            same = "—" if robust is None else f"{100 * robust['same_answers']:.1f} %"
            lines.append(
                f"| {title} | {row['augment']} | {row['setting'] or '—'} | {cls.score(row['validation'])} | "
                f"{cls.score(row['test'])} | {same} |"
            )
        gap = result["gap"]
        if gap is None:
            verdict = "для вывода нужны DAMP и CNN"
            validation = test = "—"
        else:
            same = gap["damp_minus_cnn_none"]
            best = gap["damp_minus_best_cnn"]
            validation = cls.points(None if same is None else same["validation"])
            test = cls.points(None if same is None else same["test"])
            ranges = gap.get("intervals") or {}
            spans = ranges.get("damp_minus_best_cnn") or {}
            span = spans.get("validation")
            tail = ""
            if span is not None and span["ci_low"] is not None:
                tail = f" [{100 * span['ci_low']:+.1f}; {100 * span['ci_high']:+.1f}]"
            verdict = (
                f"DAMP − CNN без аугментации; DAMP − лучшая CNN ({gap['best_cnn']}): validation "
                f"{cls.points(best['validation'])}{tail} (парный интервал по наборам), test {cls.points(best['test'])}"
            )
        lines.append(f"| **Итого** | | {verdict} | {validation} | {test} | |")
        return head + "\n".join(lines) + "\n"


class BenchRerunView:
    """Показ стенда в Rerun: открывает запись «bench» с раскладкой стенда и пишет таблицу и столбцы точности."""

    def open(self) -> None:
        """Новая запись «bench» и раскладка стенда (базовая раскладка окна: после цикла окно возвращается к ней)."""
        LOGGER.use_blueprint(BenchBlueprint.build)
        LOGGER.start_recording("bench")

    def show(self, result: dict, *, final: bool = False) -> None:
        """Таблица результатов и столбцы точности методов; запись статична, виден всегда последний итог."""
        rows = BenchTable.order(result)
        validation = [0.0 if row["validation"] is None else float(row["validation"]["mean"]) for row in rows]
        test = [0.0 if row["test"] is None else float(row["test"]["mean"]) for row in rows]
        LOGGER.event(
            "bench.table",
            section=OUTSIDE_ARTICLE,
            data={"sets_done": result["sets_done"], "rows": len(rows), "final": final,
                  "note": "таблица результатов стенда обновлена в Rerun"},
            visuals=[
                LOGGER.visual_text(TABLE, BenchTable.markdown(result, final=final), static=True),
                LOGGER.visual_bar_chart(f"{ACCURACY}/validation", validation, static=True),
                LOGGER.visual_bar_chart(f"{ACCURACY}/test", test, static=True),
            ],
        )
