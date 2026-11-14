"""Скриншоты вкладок цикла в Rerun для README (вне статьи: показ).

Прогон README (README_FIGURES_TRAIN / _TEST, карта до README_FIGURES_MAP_POINTS) идёт отдельным процессом, как обычный
прогон: в окно Rerun (его видно) и в файл .rrd. Потом запись открывается в Rerun без окна на экране (`rerun --headless`),
вкладки открываются по очереди (ViewerClient: та же раскладка с другой открытой вкладкой) и снимаются
(`ViewerClient.save_screenshot`).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from damp import settings
from damp.article_refs import OUTSIDE_ARTICLE
from damp.figures.rerun_cycle import CycleBlueprint
from damp.logging import LOGGER

SLUGS = ("overview", "stimuli", "coding", "layout", "detectors", "embeddings", "cycle", "recognition")
SYSTEM = Path(__file__).resolve().parents[2]


class RerunScreenshots:
    """Картинки docs/rerun-<номер>-<вкладка>.png: вкладки Обзор и этапы 1–7 окна Rerun на прогоне README."""

    def __init__(self, docs: str | None = None, *, size: str = "1600x1000") -> None:
        self.docs = Path(docs or settings.README_FIGURES_DIR)
        self.size = size
        self.work = Path(settings.README_FIGURES_WORK_DIR) / "rerun"

    def record(self) -> Path:
        """Прогон README отдельным процессом: в окно Rerun (его видно) и в файл .rrd — с раскладкой панелей цикла."""
        self.work.mkdir(parents=True, exist_ok=True)
        before = set(Path(settings.RERUN_RECORD_DIR).glob("*.rrd"))
        env = {**os.environ, "RERUN": "on"}
        env.pop("DAMP_RERUN_VIEWER", None)
        env.pop("DAMP_RERUN_BLUEPRINT", None)
        code = (
            "import sys; sys.path.insert(0, {system!r})\n"
            "from damp import settings\n"
            "from damp.cycle.ladder import MnistLadderRunner\n"
            "MnistLadderRunner(count=settings.README_FIGURES_TRAIN, evaluate_limit=settings.README_FIGURES_TEST, "
            "map_max_points=settings.README_FIGURES_MAP_POINTS, shifts=()).run()\n"
        ).format(system=str(SYSTEM))
        subprocess.run([sys.executable, "-c", code], env=env, check=True, stdout=subprocess.DEVNULL)
        made = sorted(set(Path(settings.RERUN_RECORD_DIR).glob("*.rrd")) - before, key=lambda path: path.stat().st_mtime)
        if not made:
            raise RuntimeError("прогон не записал .rrd: проверьте RERUN_RECORD")
        return made[-1]

    def run(self) -> list[str]:
        """Записать прогон и снять все вкладки; вернуть пути картинок."""
        recording = self.record()
        saved = self.shoot(recording)
        LOGGER.event(
            "figures.rerun.screenshots",
            section=OUTSIDE_ARTICLE,
            data={"recording": str(recording), "images": saved, "note": "вкладки цикла в Rerun на прогоне README"},
        )
        return saved

    def shoot(self, recording: Path) -> list[str]:
        """Одно окно Rerun без экрана с записью: вкладки Обзор и 1–7 открываются по очереди и снимаются в docs/."""
        import rerun.experimental as rre

        port = 9930
        env = {key: value for key, value in os.environ.items() if key != "RERUN"}
        process = subprocess.Popen(
            ["rerun", "--headless", "--port", str(port), "--window-size", self.size, str(recording)],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        saved = []
        try:
            delay = float(settings.RERUN_SHOT_DELAY)
            time.sleep(delay)  # запись и раскладка загружены
            client = rre.ViewerClient(f"rerun+http://127.0.0.1:{port}/proxy")
            for index, name in enumerate(CycleBlueprint.TABS):
                path = self.docs / f"rerun-{index}-{SLUGS[index]}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                self._open(client, name)
                time.sleep(max(1.0, delay / 4))  # вкладка перерисована
                client.save_screenshot(str(path))
                saved.append(str(path).replace("\\", "/"))
        finally:
            self._stop(process)
        return saved

    @staticmethod
    def _open(client, name: str) -> None:  # noqa: ANN001 — rerun.experimental.ViewerClient
        """Открыть вкладку name: раскладка окна целиком, у корневых вкладок другая открытая (set_blueprint)."""
        current = dict(client.get_blueprint().blueprint)
        for key, entry in current.items():
            box = entry.get("ContainerBlueprint") or {}
            contents = list(box.get("contents") or [])
            if box.get("container_kind") != "Tabs" or len(contents) != len(CycleBlueprint.TABS):
                continue
            for item in contents:
                child = current.get("/" + str(item).strip("/"), {}).get("ContainerBlueprint") or {}
                if child.get("display_name") == name:
                    current[key] = {**entry, "ContainerBlueprint": {**box, "active_tab": str(item).strip("/")}}
                    client.set_blueprint(current, of="recording")
                    return
        raise RuntimeError(f"в окне Rerun нет вкладки {name!r}: раскладка цикла не дошла до записи")

    @staticmethod
    def _stop(process: subprocess.Popen) -> None:
        """Закрыть Rerun вместе с дочерними процессами (команда `rerun` запускает сам просмотрщик отдельно)."""
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        else:
            process.kill()
        process.wait(timeout=30)
