import colorsys
from collections import defaultdict
import json
from pathlib import Path

import rerun as rr
from torchvision.datasets import MNIST
from torchvision import transforms

from damp.decoding.damp_hierarchy import (
    CodeSpace,
    CodeVector,
    DetectorBuildParams,
    EmbedParams,
    HierarchyConfig,
    LayoutConfig,
    embed_inputs,
    encode_image,
    space_from_layout,
    train_hierarchy,
)
from damp.encoding.MnistSobelAngleMap import MnistSobelAngleMap
from damp.encoding.damp_encoder import ClosedDimension, Detectors, Encoder, OpenedDimension
from damp.layout.damp_layout import Layout
from damp.layout.visualize_layout import log_layout

TRAIN_COUNT = 90000
V0_TRAIN_COUNT: int | None = 800
TEST_COUNT = 200
V0_CACHE_PATH = Path("data/mnist_v0.json")
LOG_LAYOUT_EVERY_LONG = 1
LOG_LAYOUT_EVERY_SHORT = 1
LOG_LAYOUT_EVERY_HIER_LONG = 1
LOG_LAYOUT_EVERY_HIER_SHORT = 1
TOP_K = 7
PER_LABEL_TOP_K = 7
DECODE_SIMILARITY = "cosine"
EMBED_BACKEND = "auto"
LABEL_COUNT = 10
PATCH_SIZE = 4
PATCH_CENTER = PATCH_SIZE / 2.0
HIERARCHY_LEVELS = 1


def take_samples(dataset: MNIST, count: int) -> list[tuple[object, int]]:
    samples: list[tuple[object, int]] = []
    for img_tensor, label in dataset:
        samples.append((img_tensor, int(label)))
        if len(samples) == count:
            break
    return samples


def load_v0_cache(path: Path) -> CodeSpace | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    try:
        height = int(data["height"])
        width = int(data["width"])
        code_length = int(data["code_length"])
        grid_data = data["grid"]
    except (KeyError, TypeError, ValueError):
        return None
    grid: list[list[CodeVector | None]] = []
    for row in grid_data:
        row_codes: list[CodeVector | None] = []
        for cell in row:
            if cell is None:
                row_codes.append(None)
            else:
                bits, ones = cell
                row_codes.append(
                    CodeVector(
                        int(bits),  # битовая маска активных признаков, влияет на сходство кодов
                        int(ones),  # число единиц в коде, влияет на метрики похожести
                        code_length,  # длина кода, задает размер битовой маски
                    )
                )
        grid.append(row_codes)
    return CodeSpace(
        grid=grid,  # сетка кодов v0, определяет пространственное размещение признаков
        height=height,  # высота сетки, влияет на размер пространства
        width=width,  # ширина сетки, влияет на размер пространства
        code_length=code_length,  # длина кода, определяет размер вектора признаков
    )


def save_v0_cache(v0: CodeSpace, path: Path) -> None:
    payload = {
        "height": v0.height,
        "width": v0.width,
        "code_length": v0.code_length,
        "grid": [
            [None if cell is None else [cell.bits, cell.ones] for cell in row]
            for row in v0.grid
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")))


def _angle_to_rgb(angle_deg: float) -> tuple[int, int, int]:
    hue = (angle_deg % 360.0) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def _label_counts(samples: list[tuple[object, int]]) -> list[int]:
    counts = [0 for _ in range(LABEL_COUNT)]
    for _img, label in samples:
        idx = int(label)
        if 0 <= idx < LABEL_COUNT:
            counts[idx] += 1
    return counts


def _log_measurements(path: str, values: list[tuple[float, int, int]]) -> None:
    if not values:
        rr.log(path, rr.Points2D([]))
        return
    positions = []
    colors = []
    for angle, x, y in values:
        positions.append((x * PATCH_SIZE + PATCH_CENTER, y * PATCH_SIZE + PATCH_CENTER))
        colors.append(_angle_to_rgb(float(angle)))
    rr.log(path, rr.Points2D(positions, colors=colors, radii=1.5))


def _log_dataset_stats(train_digits: list[tuple[object, int]], test_digits: list[tuple[object, int]]) -> None:
    rr.log(
        "data/stats",
        rr.AnyValues(
            train_samples=len(train_digits),
            test_samples=len(test_digits),
            top_k=TOP_K,
        ),
    )
    train_counts = _label_counts(train_digits)
    test_counts = _label_counts(test_digits)
    rr.log("data/train_labels", rr.BarChart(train_counts, abscissa=list(range(LABEL_COUNT))))
    rr.log("data/test_labels", rr.BarChart(test_counts, abscissa=list(range(LABEL_COUNT))))


def _log_model_stats(model: object) -> None:
    detectors = getattr(model, "detectors", [])
    for level_index, hierarchy in enumerate(detectors, start=1):
        layer_counts = [len(layer.detectors) for layer in hierarchy.layers]
        rr.log(
            f"model/detectors/L{level_index}",
            rr.BarChart(layer_counts, abscissa=list(range(1, len(layer_counts) + 1))),
        )
        rr.log(
            f"model/detectors/L{level_index}/summary",
            rr.AnyValues(total=sum(layer_counts), code_length=hierarchy.code_length),
        )
    spaces = getattr(model, "spaces", [])
    for level_index, space in enumerate(spaces):
        rr.log(
            f"model/spaces/L{level_index}",
            rr.AnyValues(height=space.height, width=space.width, code_length=space.code_length),
        )
    memory = getattr(model, "memory", [])
    rr.log("model/memory", rr.AnyValues(size=len(memory)))


def _similarity(a: CodeVector, b: CodeVector, mode: str) -> float:
    if a.ones == 0 or b.ones == 0:
        return 0.0
    common = (a.bits & b.bits).bit_count()
    if common == 0:
        return 0.0
    if mode == "cosine":
        denom = (a.ones * b.ones) ** 0.5
        return 0.0 if denom == 0 else common / denom
    union = a.ones + b.ones - common
    return 0.0 if union == 0 else common / union


def _group_memory_by_label(memory: list[object]) -> dict[int, list[CodeVector]]:
    grouped: dict[int, list[CodeVector]] = defaultdict(list)
    for entry in memory:
        label = int(getattr(entry, "label"))
        code = getattr(entry, "code")
        grouped[label].append(code)
    return grouped


def _infer_by_label_topk(
    image: object,
    model: object,
    memory_by_label: dict[int, list[CodeVector]],
    *,
    top_k: int,
    per_label_k: int,
    similarity: str,
) -> tuple[int | None, list[tuple[int, float]]]:
    inputs = encode_image(image, model.encoder, model.extractor)
    code = embed_inputs(inputs, model.spaces[0], model.detectors[0], model.embed[0])
    for level_index in range(1, len(model.detectors)):
        code = embed_inputs(
            [code],
            model.spaces[level_index],
            model.detectors[level_index],
            model.embed[level_index],
        )

    scored: list[tuple[int, float]] = []
    per_label_scores: dict[int, list[float]] = {}
    for label, codes in memory_by_label.items():
        label_scores: list[float] = []
        for mem_code in codes:
            score = _similarity(code, mem_code, similarity)
            label_scores.append(score)
            scored.append((label, score))
        per_label_scores[label] = label_scores

    scored.sort(key=lambda item: item[1], reverse=True)
    top = scored[: max(1, top_k)]

    votes: dict[int, float] = {}
    for label, scores in per_label_scores.items():
        if not scores:
            votes[label] = 0.0
            continue
        scores.sort(reverse=True)
        take = min(per_label_k, len(scores)) if per_label_k > 0 else len(scores)
        top_scores = scores[:take]
        votes[label] = sum(top_scores) / len(top_scores)
    predicted = max(votes.items(), key=lambda item: item[1])[0] if votes else None
    return predicted, top


def main() -> None:


# ==============================================
# 0. Получение измерений из картинки (Measurements)
# ==============================================


    train_dataset = MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    test_dataset = MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
    extractor = MnistSobelAngleMap(
        angle_in_degrees=True,  # возвращать угол в градусах, влияет на шкалу значений угла
        grad_threshold=0.015,  # порог модуля градиента, влияет на отсеивание шумовых патчей
    )
    train_digits = take_samples(train_dataset, TRAIN_COUNT)
    if V0_TRAIN_COUNT is None:
        v0_train_digits = train_digits
    else:
        v0_train_digits = take_samples(train_dataset, V0_TRAIN_COUNT)
    test_digits = take_samples(test_dataset, TEST_COUNT)
    v0_cached = load_v0_cache(V0_CACHE_PATH)

    rr.init("mnist-hierarchy-test")
    rr.spawn()
    _log_dataset_stats(train_digits, test_digits)


# ==============================================
# 1. Кодирование (Encoding)
# ==============================================


    encoder = Encoder(
        ClosedDimension(
            "Angle",  # название измерения, используется в логах/отладке
            (
                0.0,  # нижняя граница диапазона углов, задает начало периода
                360.0,  # верхняя граница диапазона, задает полный круг
            ),
            [  # Detectors(count, overlap): count — детализация угла, overlap — перекрытие окон
                Detectors(360, 0.4),
                Detectors(180, 0.4),
                Detectors(90, 0.4),
                Detectors(45, 0.4),
                Detectors(30, 0.4),
                Detectors(10, 0.4),
                Detectors(5, 0.4),
            ],
        ),  # угловое измерение, формирует часть кода по направлению градиента
        OpenedDimension(
            "X",  # название измерения по X, используется в логах/отладке
            (
                0,  # минимальный индекс патча по X, задает нижнюю границу
                6,  # максимальный индекс патча по X, ограничивает допустимые значения
            ),
            [  # Detectors(count, overlap): count — дискретизация X, overlap — перекрытие окон
                Detectors(7, 0.4),
                Detectors(4, 0.4),
                Detectors(2, 0.4),
                Detectors(1, 0.4),
            ],
        ),  # позиционное измерение X, влияет на кодирование расположения
        OpenedDimension(
            "Y",  # название измерения по Y, используется в логах/отладке
            (
                0,  # минимальный индекс патча по Y, задает нижнюю границу
                6,  # максимальный индекс патча по Y, ограничивает допустимые значения
            ),
            [  # Detectors(count, overlap) для оси Y (аналогично X)
                Detectors(7, 0.4),
                Detectors(4, 0.4),
                Detectors(2, 0.4),
                Detectors(1, 0.4),
            ],
        ),  # позиционное измерение Y, влияет на кодирование расположения
    )
    codes = defaultdict(list)
    if v0_cached is None:
        for img_tensor, label in v0_train_digits:
            img = img_tensor.squeeze(0).numpy()
            measurements = extractor.extract(
                img,  # изображение 28x28, источник градиентов для извлечения углов
                label,  # метка, используется как ключ в результатах экстракции
            )
            values = measurements.get(label, [])
            for angle, x, y in values:
                _, code = encoder.encode(
                    float(angle),  # угол градиента, влияет на активацию угловых детекторов
                    float(x),  # координата патча по X, влияет на позиционные детекторы
                    float(y),  # координата патча по Y, влияет на позиционные детекторы
                )
                codes[float(angle)].append(code)


# ==============================================
# 2. Раскладка (Layout)
# ==============================================


    if v0_cached is None:
        layout = Layout(
            codes,  # коды по углам/меткам, определяют точки для размещения
            empty_ratio=0.15,  # доля пустых ячеек, ближе к рекомендации для ровной раскладки
            similarity="cosine",  # метрика сходства, влияет на энергию и перестановки
            lambda_threshold=0.06,  # порог сходства, влияет на силу притяжения/отталкивания
            eta=0.0,  # крутизна сигмоиды порога, 0.0 делает вклад почти равномерным
            seed=0,  # сид генератора, влияет на воспроизводимость раскладки
        )
        log_layout(
            layout,  # текущая раскладка, источник данных для визуализации
            path="v0/layout",  # путь в дереве логов Rerun
            step=0,  # номер шага для таймлайна визуализации
        )
        layout.run(
            steps=22000,  # число шагов оптимизации, влияет на качество и время
            pairs_per_step=1200,  # количество пар для обмена за шаг, влияет на интенсивность поиска
            pair_radius=layout.width // 2,  # радиус выбора пар, задает глобальность перестановок
            mode="long",  # режим дальних перестановок, влияет на глобальную структуру
            min_swap_ratio=0.001,  # минимум доли обменов, влияет на остановку при застое
            log_every=LOG_LAYOUT_EVERY_LONG,  # период логирования, влияет на частоту визуализаций
            log_path="v0/layout",  # путь для логов раскладки
            step_offset=1,  # смещение номера шага в логах, влияет на ось времени
            energy_radius=None,  # радиус расчета энергии, влияет на критерий ранней остановки
            energy_check_every=5,  # частота проверки энергии, влияет на чувствительность к стабилизации
            energy_delta=5e-4,  # порог изменения энергии, меньше -> сложнее остановить раньше
            energy_patience=4,  # число проверок без улучшений до остановки
        )
        layout.run(
            steps=9000,  # число шагов донастройки, влияет на локальную шлифовку
            pairs_per_step=500,  # количество пар для обмена, влияет на скорость улучшения
            pair_radius=7,  # радиус выбора пар, задает локальность перестановок
            mode="short",  # режим локальных перестановок, влияет на тонкую настройку
            local_radius=None,  # радиус локального поиска, влияет на область перестановок
            min_swap_ratio=0.001,
            log_every=LOG_LAYOUT_EVERY_SHORT,  # период логирования для короткого прогона
            log_path="v0/layout",
            step_offset=1 + layout.last_steps,  # смещение шагов, продолжает таймлайн после long-run
        )
        v0 = space_from_layout(
            layout,  # финальная раскладка, определяет сетку кодов пространства v0
        )
        save_v0_cache(v0, V0_CACHE_PATH)
    else:
        v0 = v0_cached


# ==============================================
# 3. Иерархия детекторов (Detectors)
# ==============================================


    build_l1 = DetectorBuildParams(
        lambda_levels=[
            0.45,  # порог похожести слоя, выше -> строже отбор активаций
            0.6,  # порог похожести слоя, влияет на плотность детекторов
            0.72,  # порог похожести слоя, делает детекторы более избирательными
            0.82,  # дополнительный коротковолновый слой для детализации
        ],
        activation_radius=6,  # радиус локальной активации, влияет на размер кластеров
        energy_radius=6,  # радиус energy map, влияет на оценку энергии вокруг точки
        detector_code_length=768,  # длина кода детекторов, влияет на размерность битового вектора
        cluster_eps=2.3,  # радиус DBSCAN, влияет на объединение точек в кластеры
        cluster_min_points=2,  # минимум точек в кластере, влияет на отсев шума
        energy_threshold_mu=0.06,  # порог энергии, ниже -> больше точек участвует
        energy_lambda=0.6,  # порог сходства для energy map, влияет на карту энергий
        max_attempts=800,  # максимум попыток подбора детекторов в слое
        max_detectors_per_layer=180,  # лимит числа детекторов в слое
        min_radius=1.0,  # минимальный радиус детектора, ограничивает область покрытия
        patience=200,  # сколько неудачных попыток терпеть до остановки слоя
        similarity="cosine",  # метрика сравнения кодов, влияет на вычисление сходства
        eta=None,  # отключает сигмоидальное сглаживание порога похожести
        seed=0,  # сид генератора, влияет на воспроизводимость выбора центров
    )
    build_l2 = DetectorBuildParams(  # параметры построения для уровня L2
        lambda_levels=[0.15, 0.22, 0.3, 0.38],
        activation_radius=7,
        energy_radius=7,
        detector_code_length=512,
        cluster_eps=2.5,
        cluster_min_points=2,
        energy_threshold_mu=0.02,
        energy_lambda=0.2,
        max_attempts=1200,
        max_detectors_per_layer=220,
        min_radius=1.0,
        patience=260,
        similarity="cosine",
        eta=None,
        seed=1,
    )
    build_l3 = DetectorBuildParams(
        lambda_levels=[0.12, 0.18, 0.25, 0.32],
        activation_radius=7,
        energy_radius=7,
        detector_code_length=512,
        cluster_eps=2.5,
        cluster_min_points=2,
        energy_threshold_mu=0.02,
        energy_lambda=0.18,
        max_attempts=1200,
        max_detectors_per_layer=220,
        min_radius=1.0,
        patience=260,
        similarity="cosine",
        eta=None,
        seed=2,
    )


# ==============================================
# 4. Детекция стимула и получение эмбеддинг-кода (Embedding)
# ==============================================


    embed_l1 = EmbedParams(
        lambda_activation=0.68,  # порог сходства в активации, выше -> меньше активных точек
        mu_e=0.07,  # порог энергии точки, влияет на отбор активаций по energy map
        mu_d=0.62,  # порог уровня детектора, выше -> более строгий отбор
        sigma=110,  # максимум активных бит в коде, снижает обрезание насыщенных кодов
        similarity="cosine",  # метрика сравнения кодов, влияет на активацию
        eta=None,  # отключает сигмоидальное сглаживание порога сходства
        merge_order="high",  # порядок слияния по lambda, влияет на выбор при коллизиях
    )
    embed_l2 = EmbedParams(  # параметры для L2
        lambda_activation=0.25,
        mu_e=0.02,
        mu_d=0.25,
        sigma=88,
        similarity="cosine",
        eta=None,
        merge_order="high",
    )
    embed_l3 = EmbedParams(
        lambda_activation=0.22,
        mu_e=0.02,
        mu_d=0.22,
        sigma=80,
        similarity="cosine",
        eta=None,
        merge_order="high",
    )


# ==============================================
# 5. 3 уровня детекторов (3 levels)
# ==============================================


    layout_l2 = LayoutConfig(  # параметры раскладки уровня v1
        layout_kwargs=dict(
            empty_ratio=0.2,
            similarity="cosine",
            lambda_threshold=0.05,
            eta=8.0,
            seed=0,
            precompute_similarity=False,
            use_gpu=True,
        ),
        run_schedule=(
            dict(
                steps=2000,
                pairs_per_step=600,
                pair_radius=13,
                mode="long",
                min_swap_ratio=0.0,
                energy_radius=7,
                energy_check_every=10,
                energy_delta=5e-4,
                energy_patience=6,
                log_every=LOG_LAYOUT_EVERY_HIER_LONG,
                log_path="v1/layout",
            ),
            dict(
                steps=600,
                pairs_per_step=300,
                pair_radius=5,
                mode="short",
                local_radius=5,
                min_swap_ratio=0.0,
                log_every=LOG_LAYOUT_EVERY_HIER_SHORT,
                log_path="v1/layout",
            ),
        ),
    )
    layout_l3 = LayoutConfig(  # параметры раскладки уровня v2
        layout_kwargs=dict(
            empty_ratio=0.2,
            similarity="cosine",
            lambda_threshold=0.05,
            eta=8.0,
            seed=1,
            precompute_similarity=False,
            use_gpu=True,
        ),
        run_schedule=(
            dict(
                steps=1800,
                pairs_per_step=520,
                pair_radius=13,
                mode="long",
                min_swap_ratio=0.0,
                energy_radius=6,
                energy_check_every=10,
                energy_delta=5e-4,
                energy_patience=6,
                log_every=LOG_LAYOUT_EVERY_HIER_LONG,
                log_path="v2/layout",
            ),
            dict(
                steps=520,
                pairs_per_step=260,
                pair_radius=4,
                mode="short",
                local_radius=4,
                min_swap_ratio=0.0,
                log_every=LOG_LAYOUT_EVERY_HIER_SHORT,
                log_path="v2/layout",
            ),
        ),
    )
    if HIERARCHY_LEVELS == 1:
        build = (build_l1,)
        embed = (embed_l1,)
        layout = ()
    elif HIERARCHY_LEVELS == 2:
        build = (build_l1, build_l2)
        embed = (embed_l1, embed_l2)
        layout = (layout_l2,)
    else:
        build = (build_l1, build_l2, build_l3)
        embed = (embed_l1, embed_l2, embed_l3)
        layout = (layout_l2, layout_l3)
    config = HierarchyConfig(
        encoder=encoder,  # кодировщик признаков, определяет структуру входных кодов
        extractor=extractor,  # экстрактор измерений, влияет на входные стимулы
        v0=v0,  # базовое пространство кодов, служит опорой для детекторов
        build=build,  # параметры построения детекторов по уровням
        embed=embed,  # параметры эмбеддинга по уровням
        layout=layout,  # параметры раскладки между уровнями
        embed_backend=EMBED_BACKEND,  # backend для ускорения эмбеддинга (auto/numpy/torch/python)
    )
    model = train_hierarchy(
        train_digits,  # обучающие изображения и метки, определяют память модели
        config,  # конфигурация иерархии, влияет на все этапы обучения
    )
    _log_model_stats(model)
    memory_by_label = _group_memory_by_label(model.memory)


# ==============================================
# 6. Декодирование в класс (Decoding)
# ==============================================


    correct = 0
    total = len(test_digits)
    label_counts: dict[int, int] = defaultdict(int)
    pred_counts: dict[int, int] = defaultdict(int)
    none_pred = 0
    confusion = [[0 for _ in range(LABEL_COUNT)] for _ in range(LABEL_COUNT)]
    top1_scores: list[float] = []
    for idx, (img_tensor, label) in enumerate(test_digits, start=1):
        label_idx = int(label)
        img = img_tensor.squeeze(0).numpy()
        measurements = extractor.extract(img, label_idx).get(label_idx, [])
        predicted, top = _infer_by_label_topk(
            img_tensor,  # изображение для распознавания, источник стимула
            model,  # обученная иерархия детекторов и память
            memory_by_label,  # память, сгруппированная по меткам
            top_k=TOP_K,  # размер списка top-k, влияет на ранжирование и статистику
            per_label_k=PER_LABEL_TOP_K,  # число лучших совпадений на метку
            similarity=DECODE_SIMILARITY,  # метрика сравнения кодов, влияет на выбор класса
        )
        label_counts[label_idx] += 1
        pred_is_none = predicted is None
        if pred_is_none:
            none_pred += 1
            pred_val = -1
        else:
            pred_val = int(predicted)
            pred_counts[pred_val] += 1
            if 0 <= label_idx < LABEL_COUNT and 0 <= pred_val < LABEL_COUNT:
                confusion[label_idx][pred_val] += 1
        is_correct = (not pred_is_none) and (pred_val == label_idx)
        if is_correct:
            correct += 1
        top_pairs = top[:TOP_K] if top else []
        top_labels = [int(top_label) for top_label, _ in top_pairs]
        top_scores = [float(top_score) for _top_label, top_score in top_pairs]
        top1_label = top_labels[0] if top_labels else -1
        top1_score = top_scores[0] if top_scores else 0.0
        top1_scores.append(top1_score)

        rr.set_time("sample", sequence=idx)
        rr.log("test/image", rr.Image(img))
        _log_measurements("test/image/measurements", measurements)
        rr.log(
            "test/prediction",
            rr.AnyValues(
                label=label_idx,
                predicted=pred_val,
                correct=is_correct,
                top1_label=top1_label,
                top1_score=top1_score,
                measurements=len(measurements),
                topk_labels=top_labels,
                topk_scores=top_scores,
            ),
        )
        if top_scores:
            rr.log("test/topk", rr.BarChart(top_scores, abscissa=top_labels))
        pred_text = "none" if pred_is_none else str(pred_val)
        message = (
            f"label={label_idx} pred={pred_text} correct={is_correct} "
            f"topk={list(zip(top_labels, top_scores))}"
        )
        level = rr.TextLogLevel.INFO if is_correct else rr.TextLogLevel.WARN
        rr.log("test/prediction/log", rr.TextLog(message, level=level))

    rr.reset_time()
    accuracy = (correct / total) if total else 0.0
    per_label_accuracy = []
    for label_idx in range(LABEL_COUNT):
        total_label = label_counts.get(label_idx, 0)
        correct_label = confusion[label_idx][label_idx] if total_label else 0
        acc = (correct_label / total_label) if total_label else 0.0
        per_label_accuracy.append(acc)
    pred_counts_full = [pred_counts.get(idx, 0) for idx in range(LABEL_COUNT)]
    label_counts_full = [label_counts.get(idx, 0) for idx in range(LABEL_COUNT)]
    rr.log(
        "metrics/summary",
        rr.AnyValues(
            accuracy=accuracy,
            correct=correct,
            total=total,
            none_pred=none_pred,
        ),
    )
    rr.log("metrics/confusion", rr.Tensor(confusion, dim_names=("true", "pred")))
    rr.log("metrics/per_label_accuracy", rr.BarChart(per_label_accuracy, abscissa=list(range(LABEL_COUNT))))
    rr.log("metrics/label_counts", rr.BarChart(label_counts_full, abscissa=list(range(LABEL_COUNT))))
    rr.log("metrics/pred_counts", rr.BarChart(pred_counts_full, abscissa=list(range(LABEL_COUNT))))
    if top1_scores:
        rr.log("metrics/top1_scores", rr.BarChart(top1_scores))
        rr.log(
            "metrics/top1_summary",
            rr.AnyValues(
                avg=sum(top1_scores) / len(top1_scores),
                min=min(top1_scores),
                max=max(top1_scores),
            ),
        )
    print(f"accuracy: {accuracy:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
