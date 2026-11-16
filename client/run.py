"""Команды DAMP — интерфейс командной строки поверх фреймворка damp_api (к системе — только через него).

    python client/run.py ladder --count 100 --limit 500     # весь пайплайн: цикл уровней и распознавание
    python client/run.py --help                             # все команды

Настройки — client/config.toml (самые значимые сверху).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from damp_api import Damp  # noqa: E402


class Commands:
    """Разбор команды и вызов нужной части фреймворка."""

    def __init__(self, damp: Damp | None = None) -> None:
        self.damp = damp or Damp()
        self.parser = self._build_parser()

    # ------------------------------------------------------------------ разбор
    def _build_parser(self) -> argparse.ArgumentParser:
        s = self.damp.settings
        parser = argparse.ArgumentParser(
            prog="python client/run.py",
            description=f"DAMP {self.damp.version} на MNIST: этапы 1–7 по статье и её продолжению. Настройки — {s.path.name}.",
        )
        commands = parser.add_subparsers(dest="command", metavar="команда")

        ladder = commands.add_parser("ladder", help="весь пайплайн: цикл уровней (этапы 1–6) и распознавание (этап 7)")
        ladder.add_argument("--count", type=int, default=s["DECODER_DEFAULT_COUNT"], help="обучающих цифр")
        ladder.add_argument("--limit", type=int, default=s["LADDER_EVALUATE_LIMIT"], help="проверочных; 0 — без проверки")
        ladder.add_argument("--split", choices=["train", "test"], default="test")
        ladder.add_argument("--verify-cache", action="store_true", help="пересчитать шаги из кеша и сверить с ним бит в бит")

        layout = commands.add_parser("layout", help="один проход без цикла: раскладка, детекторы, база эмбеддингов (этапы 2–5)")
        layout.add_argument("--count", type=int, default=s["DECODER_DEFAULT_COUNT"])
        layout.add_argument("--label", type=int, default=None, help="только цифры одного класса")
        layout.add_argument("--activation-samples", type=int, default=None, help="число эмбеддингов; по умолчанию --count")
        layout.add_argument("--rebuild", action="store_true", help="не брать готовую раскладку и эмбеддинги из data/")
        layout.add_argument("--layout-only", action="store_true", help="только раскладка, без активации и эмбеддингов")

        evaluate = commands.add_parser("evaluate", help="точность по эмбеддингам первого уровня (этап 7)")
        evaluate.add_argument("--layout-path", type=str, default=None)
        evaluate.add_argument("--count", type=int, default=s["DECODER_DEFAULT_COUNT"])
        evaluate.add_argument("--limit", type=int, default=s["DECODER_DATASET_LIMIT"])
        evaluate.add_argument("--split", choices=["train", "test"], default="test")
        evaluate.add_argument("--temperature", type=float, default=None)

        predict = commands.add_parser("predict", help="распознать картинку или батч по эмбеддингам первого уровня")
        predict.add_argument("--layout-path", type=str, default=None)
        predict.add_argument("--count", type=int, default=s["DECODER_DEFAULT_COUNT"])
        predict.add_argument("--image", type=str, default=None, help="картинка 28×28, например client/examples/digit.png")
        predict.add_argument("--dataset", choices=["train", "test"], default=None)
        predict.add_argument("--limit", type=int, default=s["DECODER_DATASET_LIMIT"])
        predict.add_argument("--temperature", type=float, default=None)
        predict.add_argument("--calibrate", action="store_true", help="подобрать температуру softmax")

        bench = commands.add_parser(
            "bench", help="честное сравнение DAMP, kNN по пикселям и CNN: настройки по validation, наборы, интервалы"
        )
        bench.add_argument("--count", type=int, default=s["BENCH_TRAIN"], help="обучающих в наборе")
        bench.add_argument("--sets", type=int, default=s["BENCH_SETS"], help="обучающих наборов")
        bench.add_argument("--validation", type=int, default=s["BENCH_VALIDATION"], help="картинок отложенной выборки")
        bench.add_argument("--test", type=int, default=s["BENCH_TEST"], help="проверочных из t10k; 0 — не считать")
        bench.add_argument("--map-points", type=int, default=s["BENCH_MAP_POINTS"], help="предел точек карты уровня у DAMP")
        bench.add_argument("--methods", nargs="+", choices=["damp", "pixels", "cnn"], default=list(s["BENCH_METHODS"]))
        # «none» в файле настроек читается как None: по умолчанию — те же строки, что принимает --augment.
        augment = ["none" if mode is None else mode for mode in s["BENCH_AUGMENT"]]
        bench.add_argument("--augment", nargs="+", choices=["none", "shifts"], default=augment)

        dashboard = commands.add_parser(
            "dashboard", help="живой дашборд: рисуем цифру кистью — кусочки и кластеры всех уровней цикла на одном экране"
        )
        dashboard.add_argument("--count", type=int, default=s["DASHBOARD_TRAIN"], help="обучающих цифр цикла")
        dashboard.add_argument("--map-points", type=int, default=s["DASHBOARD_MAP_POINTS"], help="предел точек карты уровня")
        dashboard.add_argument("--port", type=int, default=s["DASHBOARD_PORT"])
        dashboard.add_argument("--no-browser", action="store_true", help="не открывать страницу в браузере")
        dashboard.add_argument("--snapshot", type=str, default=None,
                               help="только снимок страницы в файл (пример-пятёрка и след кусочка; нужен playwright)")

        figures = commands.add_parser("figures", help="все рисунки README в docs/")
        figures.add_argument("--rerun", action="store_true", help="только скриншоты вкладок цикла в Rerun (окна ненадолго откроются)")
        commands.add_parser("pinwheel", help="тест пинвила: 360 кодов угла на позицию в одной раскладке")

        detectors = commands.add_parser("detectors", help="картинка первичных детекторов кодера команды layout")
        detectors.add_argument("--angle", type=float, default=s["DETECTORS_VIEW_STIMULUS"][0])
        detectors.add_argument("--x", type=float, default=s["DETECTORS_VIEW_STIMULUS"][1])
        detectors.add_argument("--y", type=float, default=s["DETECTORS_VIEW_STIMULUS"][2])
        detectors.add_argument("--no-stimulus", action="store_true", help="все детекторы без выделения")
        detectors.add_argument("--output", type=str, default=s["DETECTORS_VIEW_OUTPUT_PATH"])

        sketch = commands.add_parser("sketch", help="одна цифра MNIST из линий её углопозиций (Rerun)")
        sketch.add_argument("digit", type=int, choices=range(10), help="цифра 0…9")
        sketch.add_argument("--seed", type=int, default=None, help="зерно выбора картинки; по умолчанию случайное")
        sketch.add_argument("--split", choices=["train", "test"], default=s["DIGIT_SKETCH_SPLIT"])
        sketch.add_argument("--patch-size", type=int, default=s["MNIST_PATCH_SIZE"], help="сторона окна Собеля")
        sketch.add_argument("--upscale", action=argparse.BooleanOptionalAction, default=s["MNIST_SOBEL_UPSCALE_ENABLED"])

        contour = commands.add_parser("contour", help="таблица: цифры 0…9 из углопозиций контура")
        contour.add_argument("--seed", type=int, default=s["DIGITS_CONTOUR_SEED"])
        contour.add_argument("--split", choices=["train", "test"], default=s["DIGIT_SKETCH_SPLIT"])
        contour.add_argument("--upscale", action=argparse.BooleanOptionalAction, default=s["MNIST_SOBEL_UPSCALE_ENABLED"])

        cache = commands.add_parser("cache", help="кеш шагов: сводка по шагам; --clear — очистить")
        cache.add_argument("--clear", nargs="?", const="*", default=None, metavar="ШАГ", help="очистить шаг (без имени — весь кеш)")

        docs = commands.add_parser("docs", help="HTML-справка по фреймворку damp_api в docs/api/ (нужен pdoc)")
        docs.add_argument("--serve", action="store_true", help="живой сервер вместо сборки")
        docs.add_argument("--open", action="store_true", help="открыть собранную справку в браузере")
        docs.add_argument("--links", action="store_true", help="обновить ссылки README на строки кода API")
        return parser

    # ------------------------------------------------------------------ запуск
    def run(self, argv: list[str] | None = None) -> None:
        args = self.parser.parse_args(argv)
        if not args.command:
            self.parser.print_help()
            return
        getattr(self, "_" + args.command.replace("-", "_"))(args)

    def _ladder(self, args: argparse.Namespace) -> None:
        damp = self.damp.with_settings(CACHE_VERIFY_SHARE=1.0) if args.verify_cache else self.damp
        report = damp.cycle.run(train=args.count, test=args.limit, split=args.split)
        for level in report.levels:
            alone = "" if level.response is None else f" alone@1={level.response[1]:.4f}"
            glued = "" if level.glued is None else f" glued@1={level.glued[1]:.4f}"
            print(
                f"ladder level {level.number}: field_radius={level.field_radius:.1f} "
                f"picture_seen={level.picture_seen:.2f} keyholes/picture={level.keyholes_per_picture:.1f} "
                f"elements/keyhole={level.elements_per_keyhole:.2f} map={level.map_points} "
                f"same_label={level.same_label:.3f}{alone}{glued}"
            )
        if report.vote is not None:
            vote = " ".join(f"vote@{k}={value:.4f}" for k, value in report.vote.items())
            print(f"ladder top: dataset={args.split} samples={args.limit} {vote}")

    def _dashboard(self, args: argparse.Namespace) -> None:
        import webbrowser

        from dashboard.server import DashboardServer

        server = DashboardServer(self.damp, count=args.count, map_points=args.map_points, port=args.port)
        print(f"dashboard: учу цикл на {args.count} обучающих цифрах (шаги — из кеша, если такой цикл уже был)…", flush=True)
        ladder = server.prepare()
        if args.snapshot:
            print(f"dashboard: снимок страницы — {server.snapshot(args.snapshot)}")
            return
        print(f"dashboard: уровней {ladder.levels}, обучающих картинок {ladder.train_pictures}; страница — {server.address} (Ctrl+C — стоп)",
              flush=True)
        if not args.no_browser:
            webbrowser.open(server.address)
        server.serve()

    def _bench(self, args: argparse.Namespace) -> None:
        report = self.damp.recognition.bench(
            train=args.count, sets=args.sets, validation=args.validation, test=args.test, map_points=args.map_points,
            methods=tuple(args.methods), augment=tuple(args.augment),
        )

        def cell(score) -> str:  # noqa: ANN001
            if score is None:
                return "—"
            text = f"{100 * score.mean:.1f} ± {100 * score.std:.1f}"
            return text if score.ci_low is None else f"{text} [{100 * score.ci_low:.1f}; {100 * score.ci_high:.1f}]"

        print(f"bench: train={report.train} sets={report.sets} validation={report.validation} test={report.test or 'нет'}")
        print(f"{'метод':<10} {'аугм.':<7} {'настройка':<34} {'validation, %':<24} {'test, %':<24} ответы при сдвиге")
        for row in report.methods:
            if not row.supported:
                print(f"{row.method:<10} {row.augment:<7} {row.note}")
                continue
            same = "—" if row.robustness is None else f"{100 * row.robustness.same_answers:.1f} %"
            print(f"{row.method:<10} {row.augment:<7} {row.setting or '—':<34} {cell(row.validation):<24} {cell(row.test):<24} {same}")
        gap = report.gap
        if gap is None:
            print("Итого: для вывода нужны DAMP и CNN")
        else:
            def points(value) -> str:  # noqa: ANN001
                return "—" if value is None else f"{100 * value:+.1f} п.п."

            best = gap["damp_minus_best_cnn"]
            same = gap["damp_minus_cnn_none"] or {}
            print(
                f"Итого: DAMP − CNN без аугментации: validation {points(same.get('validation'))}, test {points(same.get('test'))}; "
                f"DAMP − лучшая CNN ({gap['best_cnn']}): validation {points(best['validation'])}, test {points(best['test'])}"
            )
        print(f"bench: файл {report.path}")

    def _layout(self, args: argparse.Namespace) -> None:
        if args.layout_only:
            summary = self.damp.layout.build(args.count, label=args.label, rebuild=args.rebuild)
        else:
            summary = self.damp.embeddings.build(args.count, samples=args.activation_samples, label=args.label, rebuild=args.rebuild)
        print(f"layout: points={summary.points} grid={summary.width}x{summary.height}")

    def _evaluate(self, args: argparse.Namespace) -> None:
        result = self.damp.recognition.evaluate(
            args.count, split=args.split, limit=args.limit, temperature=args.temperature, layout_path=args.layout_path
        )
        print(f"dataset={result.split} samples={result.samples} accuracy={result.accuracy:.4f}")

    def _predict(self, args: argparse.Namespace) -> None:
        recognition = self.damp.recognition
        temperature = args.temperature
        if args.calibrate:
            temperature, best = recognition.calibrate(args.count, split=args.dataset or "test", limit=args.limit, layout_path=args.layout_path)
            print(f"calibration: temp={temperature:.3f} acc={best:.4f}")
        if args.image:
            answer = recognition.predict(args.image, args.count, temperature=temperature, layout_path=args.layout_path)
            print(f"image={args.image} pred={answer.digit} conf={answer.confidence:.4f} probs={list(answer.probabilities)}")
            return
        result = recognition.predict_dataset(
            args.count, split=args.dataset or "test", limit=args.limit, temperature=temperature, layout_path=args.layout_path
        )
        print(f"dataset={result.split} samples={result.samples} accuracy={result.accuracy:.4f}")

    def _figures(self, args: argparse.Namespace) -> None:
        if args.rerun:
            for path in self.damp.figures.rerun():
                print(f"figures: {path}")
            return
        print(f"figures: {self.damp.figures.readme()}")
        # Снимок живого дашборда для README (пример-пятёрка и след кусочка); без playwright — пропускается.
        import importlib.util

        if importlib.util.find_spec("playwright") is None:
            print("figures: снимок дашборда пропущен — нет playwright (pip install playwright)")
            return
        from dashboard.server import DashboardServer

        s = self.damp.settings
        server = DashboardServer(self.damp)
        server.prepare()
        print(f"figures: {server.snapshot(str(Path(s['README_FIGURES_DIR']) / 'dashboard.png'))}")

    def _pinwheel(self, args: argparse.Namespace) -> None:
        report = self.damp.layout.pinwheel()
        print(
            f"pinwheel: mean_neighbour_delta={report.mean_neighbour_delta:.2f} sharp={report.sharp_transition_ratio:.3f} "
            f"singularities={report.singularities} images={', '.join(report.images)}"
        )

    def _detectors(self, args: argparse.Namespace) -> None:
        angle = None if args.no_stimulus else args.angle
        print(f"detectors: {self.damp.coding.detectors_picture(angle, args.x, args.y, path=args.output)}")

    def _sketch(self, args: argparse.Namespace) -> None:
        self.damp.stimuli.sketch(args.digit, seed=args.seed, split=args.split, patch_size=args.patch_size, upscale=args.upscale)

    def _contour(self, args: argparse.Namespace) -> None:
        print(f"contour: {self.damp.stimuli.contour_table(seed=args.seed, split=args.split, upscale=args.upscale)}")

    def _cache(self, args: argparse.Namespace) -> None:
        cache = self.damp.cache
        if args.clear is not None:
            removed = cache.clear(None if args.clear == "*" else args.clear)
            print(f"cache: очищено записей {removed}")
        rows = cache.summary()
        print(f"cache: {cache.path} {'включён' if cache.enabled else 'выключен'}")
        print(f"{'шаг':<24} {'записей':>8} {'МБ':>9} {'взят, мин назад':>16}")
        for row in rows:
            print(f"{row.step:<24} {row.entries:>8} {row.bytes / 2**20:>9.1f} {row.last_used_seconds / 60:>16.1f}")
        print(f"{'Итого':<24} {sum(r.entries for r in rows):>8} {cache.size() / 2**20:>9.1f}")

    def _docs(self, args: argparse.Namespace) -> None:
        from apidocs import main

        flags = [flag for flag, on in (("--serve", args.serve), ("--open", args.open), ("--links", args.links)) if on]
        raise SystemExit(main(flags))


if __name__ == "__main__":
    Commands().run()
