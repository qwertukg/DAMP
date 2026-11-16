"""damp_api — фреймворк DAMP для клиентского программиста.

DAMP — разреженные битовые коды, их раскладка на двумерной карте и детекторы над картой; проверка — на цифрах
MNIST. Проход «код → раскладка → детекторы → эмбеддинг» повторяется по кругу: детали одного уровня становятся
стимулами следующего, пока окно не увидит цифру целиком. Источники: статья `docs/DAMP.pdf` и её продолжение
(разделы П.1–П.9); параграфы в справке указаны как `§6.5` и `П.5`. Подробности и результаты замеров — в
[README](../../README.md).

Все обращения к нейросети — только через этот пакет: UI (`client/run.py`), скрипты и ноутбуки не трогают
`system/damp` напрямую. Внутри фреймворка к системному слою ходит один мост, `_system.py`.

## Этапы пайплайна

У `Damp()` по полю на каждый этап.

| № | Поле `Damp` | Класс | Что происходит | Источник |
|---|---|---|---|---|
| 1 | `stimuli` | `damp_api.stages.Stimuli` | картинка → углопозиции: угол края и клетка сетки 24×24 | вне статьи |
| 2 | `coding` | `damp_api.stages.Coding` | угол и место → разреженный код; окно → код «что и где» (**начало цикла** ↻) | §2, §3.2, §4; П.4–П.7 |
| 3 | `layout` | `damp_api.stages.Layout` | коды → карта, похожие рядом | §5, §7.1.7 |
| 4 | `detectors` | `damp_api.stages.Detectors` | карта → иерархия детекторов; стимул → зажжённые точки → сработавшие детекторы | §6.1–§6.5, §7.1.10–§7.1.11 |
| 5 | `embeddings` | `damp_api.stages.Embeddings` | сработавшие детекторы → код детали | §6.5, §6.6, §7.1.12; П.1 |
| 6 | `cycle` | `damp_api.stages.Cycle` | детали → снова этап 2; окно растёт, пока не увидит цифру целиком (**конец цикла** ↺) | §6.2, §6.6; П.2, П.3, П.9 |
| 7 | `recognition` | `damp_api.stages.Recognition` | детали верхнего уровня голосуют за класс | §2.2.5; мерка вне статьи |

Рядом с ними — `figures` (рисунки README) и `settings` (настройки только для чтения).

## Кубики и готовые прогоны

У каждого этапа два вида вызовов:

- **кубики** — один шаг этапа. Результат — часть (`damp_api.parts`: `Digits`, `Elements`, `Keyholes`,
  `KeyholeCodes`, `Memory`, `MapDetectors`, `Firing`, `Level`, `TrainedLadder`) — идёт в следующий вызов; так
  клиент собирает свой цикл. `cycle.trained` — обученный цикл: по нему идёт любая новая картинка (дашборд);
- **готовые прогоны** — то же, собранное системой целиком: `cycle.run`, `layout.build`, `recognition.evaluate`
  и другие. Они возвращают простые записи из `damp_api.reports`.

Настройки лежат в `client/config.toml`; свои значения поверх файла — `Damp(changes={...})`.
Если уровень цикла не сложился, кубики поднимают `LevelFailed`.

## Свой цикл из кубиков

    import sys; sys.path.insert(0, "client")
    from damp_api import Damp

    damp = Damp()                                        # настройки — client/config.toml
    digits = damp.stimuli.digits(train=100, test=500)    # этап 1: цифры и их углопозиции
    elements = damp.coding.sticks(digits)                # этап 2: элементы уровня 0 — коды направлений
    while True:                                          # этап 6: цикл уровней
        keyholes = damp.cycle.keyholes(elements)           #   окна внимания уровня
        codes = damp.coding.keyholes(keyholes)             #   этап 2: код окна «что и где»
        memory = damp.layout.memory(codes)               #   этап 3: карта памяти уровня
        detectors = damp.detectors.build(memory)         #   этап 4: детекторы над картой
        firing = detectors.fire(codes)                   #   этап 4: что сработало на каждом окне
        elements = damp.embeddings.details(firing)       #   этап 5: детали — элементы следующего уровня
        if keyholes.picture_seen >= damp.settings["LADDER_TOP_COVERAGE"]:
            break
    damp.recognition.vote(elements)                      # этап 7: голосование верхних деталей

    damp.cycle.run(train=100, test=500, map_points=5000) # то же готовым прогоном, с мерками уровней
    Damp(changes={"LADDER_TOP_COVERAGE": 0.8})           # свои значения настроек поверх файла
"""
from .cache import Cache
from .damp import Damp, Settings
from .parts import Digits, DetectorLayer, Elements, Firing, Level, MapDetectors, Memory, KeyholeCodes, Keyholes, TrainedLadder
from .reports import (
    BenchLevel,
    BenchMethod,
    BenchReport,
    BenchRobustness,
    BenchScore,
    CacheStep,
    Code,
    CycleLevel,
    CycleReport,
    Evaluation,
    LayoutSummary,
    LevelFailed,
    LevelMap,
    Picture,
    PinwheelReport,
    Prediction,
    Sight,
    SightLevel,
    Stage,
    Stimulus,
    TrainingSticks,
)

__all__ = [
    "Damp",
    "Settings",
    # части цикла — результаты кубиков этапов
    "Digits",
    "Elements",
    "Keyholes",
    "KeyholeCodes",
    "Memory",
    "MapDetectors",
    "DetectorLayer",
    "Firing",
    "Level",
    "TrainedLadder",
    "LevelFailed",
    # простые результаты
    "Stage",
    "Stimulus",
    "Picture",
    "Code",
    "LayoutSummary",
    "PinwheelReport",
    "CycleLevel",
    "CycleReport",
    "Sight",
    "SightLevel",
    "LevelMap",
    "TrainingSticks",
    "Prediction",
    "Evaluation",
    # честное сравнение
    "BenchReport",
    "BenchMethod",
    "BenchScore",
    "BenchRobustness",
    "BenchLevel",
    # кеш шагов
    "Cache",
    "CacheStep",
]
