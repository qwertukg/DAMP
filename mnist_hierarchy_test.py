from collections import defaultdict
import json
from pathlib import Path

from torchvision.datasets import MNIST
from torchvision import transforms

from damp.decoding.damp_hierarchy import (
    CodeSpace,
    CodeVector,
    DetectorBuildParams,
    EmbedParams,
    HierarchyConfig,
    LayoutConfig,
    infer,
    space_from_layout,
    train_hierarchy,
)
from damp.encoding.MnistSobelAngleMap import MnistSobelAngleMap
from damp.encoding.damp_encoder import ClosedDimension, Detectors, Encoder, OpenedDimension
from damp.layout.damp_layout import Layout

TRAIN_COUNT = 3000
TEST_COUNT = 1000
V0_CACHE_PATH = Path("data/mnist_v0.json")


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
                row_codes.append(CodeVector(int(bits), int(ones), code_length))
        grid.append(row_codes)
    return CodeSpace(grid=grid, height=height, width=width, code_length=code_length)


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


def main() -> None:
    # 0. Получение измерений из картинки (Measurements)
    train_dataset = MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    test_dataset = MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
    extractor = MnistSobelAngleMap(angle_in_degrees=True, grad_threshold=0.03)
    train_digits = take_samples(train_dataset, TRAIN_COUNT)
    test_digits = take_samples(test_dataset, TEST_COUNT)
    v0_cached = load_v0_cache(V0_CACHE_PATH)

    # 1. Кодирование (Encoding)
    encoder = Encoder(
        ClosedDimension(
            "Angle",
            (0.0, 360.0),
            [
                Detectors(360, 0.4),
                Detectors(180, 0.4),
                Detectors(90, 0.4),
                Detectors(45, 0.4),
                Detectors(30, 0.4),
                Detectors(10, 0.4),
                Detectors(5, 0.4),
            ],
        ),
        OpenedDimension(
            "X",
            (0, 6),
            [
                Detectors(7, 0.4),
                Detectors(4, 0.4),
                Detectors(2, 0.4),
                Detectors(1, 0.4),
            ],
        ),
        OpenedDimension(
            "Y",
            (0, 6),
            [
                Detectors(7, 0.4),
                Detectors(4, 0.4),
                Detectors(2, 0.4),
                Detectors(1, 0.4),
            ],
        ),
    )
    codes = defaultdict(list)
    if v0_cached is None:
        for img_tensor, label in train_digits:
            img = img_tensor.squeeze(0).numpy()
            measurements = extractor.extract(img, label)
            values = measurements.get(label, [])
            for angle, x, y in values:
                _, code = encoder.encode(float(angle), float(x), float(y))
                codes[float(angle)].append(code)

    # 2. Раскладка (Layout)
    if v0_cached is None:
        layout = Layout(
            codes,
            empty_ratio=0.5,
            similarity="cosine",
            lambda_threshold=0.06,
            eta=0.0,
            seed=0,
        )
        layout.run(
            steps=22000,
            pairs_per_step=1200,
            pair_radius=layout.width // 2,
            mode="long",
            min_swap_ratio=0.001,
            log_every=None,
            step_offset=1,
            energy_radius=7,
            energy_check_every=5,
            energy_delta=5e-4,
            energy_patience=4,
        )
        layout.run(
            steps=900,
            pairs_per_step=500,
            pair_radius=7,
            mode="short",
            local_radius=7,
            min_swap_ratio=0.001,
            log_every=None,
            step_offset=1 + layout.last_steps,
        )
        v0 = space_from_layout(layout)
        save_v0_cache(v0, V0_CACHE_PATH)
    else:
        v0 = v0_cached

    # 3. Иерархия детекторов (Detectors)
    build_l1 = DetectorBuildParams(
        lambda_levels=[0.5, 0.6, 0.7],
        activation_radius=7,
        energy_radius=7,
        detector_code_length=256,
        cluster_eps=2.5,
        cluster_min_points=3,
        energy_threshold_mu=0.08,
        energy_lambda=0.6,
        max_attempts=800,
        max_detectors_per_layer=120,
        min_radius=1.0,
        patience=200,
        similarity="cosine",
        eta=None,
        seed=0,
    )
    build_l2 = DetectorBuildParams(
        lambda_levels=[0.45, 0.55, 0.7],
        activation_radius=6,
        energy_radius=6,
        detector_code_length=256,
        cluster_eps=2.5,
        cluster_min_points=2,
        energy_threshold_mu=0.05,
        energy_lambda=0.6,
        max_attempts=600,
        max_detectors_per_layer=120,
        min_radius=1.0,
        patience=200,
        similarity="cosine",
        eta=None,
        seed=1,
    )
    build_l3 = DetectorBuildParams(
        lambda_levels=[0.4, 0.55, 0.7],
        activation_radius=5,
        energy_radius=5,
        detector_code_length=256,
        cluster_eps=2.5,
        cluster_min_points=2,
        energy_threshold_mu=0.04,
        energy_lambda=0.6,
        max_attempts=600,
        max_detectors_per_layer=100,
        min_radius=1.0,
        patience=160,
        similarity="cosine",
        eta=None,
        seed=2,
    )

    # 4. Детекция стимула и получение эмбеддинг-кода (Embedding)
    embed_l1 = EmbedParams(
        lambda_activation=0.65,
        mu_e=0.09,
        mu_d=0.65,
        sigma=64,
        similarity="cosine",
        eta=None,
        merge_order="high",
    )
    embed_l2 = EmbedParams(
        lambda_activation=0.5,
        mu_e=0.05,
        mu_d=0.5,
        sigma=50,
        similarity="cosine",
        eta=None,
        merge_order="high",
    )
    embed_l3 = EmbedParams(
        lambda_activation=0.5,
        mu_e=0.05,
        mu_d=0.5,
        sigma=40,
        similarity="cosine",
        eta=None,
        merge_order="high",
    )

    # 5. 3 уровня детекторов (3 levels)
    layout_l2 = LayoutConfig(
        layout_kwargs=dict(
            empty_ratio=0.5,
            similarity="cosine",
            lambda_threshold=0.06,
            eta=0.0,
            seed=0,
            precompute_similarity=False,
            use_gpu=True,
        ),
        run_schedule=(
            dict(
                steps=800,
                pairs_per_step=400,
                pair_radius=7,
                mode="long",
                min_swap_ratio=0.001,
                log_every=None,
            ),
            dict(
                steps=300,
                pairs_per_step=200,
                pair_radius=5,
                mode="short",
                local_radius=5,
                min_swap_ratio=0.001,
                log_every=None,
            ),
        ),
    )
    layout_l3 = LayoutConfig(
        layout_kwargs=dict(
            empty_ratio=0.5,
            similarity="cosine",
            lambda_threshold=0.06,
            eta=0.0,
            seed=1,
            precompute_similarity=False,
            use_gpu=True,
        ),
        run_schedule=(
            dict(
                steps=600,
                pairs_per_step=300,
                pair_radius=5,
                mode="long",
                min_swap_ratio=0.001,
                log_every=None,
            ),
            dict(
                steps=250,
                pairs_per_step=160,
                pair_radius=4,
                mode="short",
                local_radius=4,
                min_swap_ratio=0.001,
                log_every=None,
            ),
        ),
    )
    config = HierarchyConfig(
        encoder=encoder,
        extractor=extractor,
        v0=v0,
        build_l1=build_l1,
        build_l2=build_l2,
        build_l3=build_l3,
        embed_l1=embed_l1,
        embed_l2=embed_l2,
        embed_l3=embed_l3,
        layout_l2=layout_l2,
        layout_l3=layout_l3,
    )
    model = train_hierarchy(train_digits, config)

    # 6. Декодирование в класс (Decoding)
    correct = 0
    total = len(test_digits)
    for img_tensor, label in test_digits:
        label_idx = int(label)
        predicted, top = infer(img_tensor, model, top_k=3, similarity="cosine", label=label_idx)
        if predicted == label_idx:
            correct += 1
    accuracy = (correct / total) if total else 0.0
    print(f"accuracy: {accuracy:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
