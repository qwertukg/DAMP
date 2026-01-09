from damp.encoding.damp_encoder import Encoder, ClosedDimension, OpenedDimension, Detectors
from damp.encoding.visualize_encoding import show, wait_for_close
import numpy as np
from damp.encoding.MnistSobelAngleMap import MnistSobelAngleMap
from torchvision.datasets import MNIST
from torchvision import transforms
from collections import defaultdict
import json
from pathlib import Path
from damp.layout.damp_layout import Layout
from damp.layout.visualize_layout import log_layout
import rerun as rr
import random
from damp.decoding.damp_hierarchy import (
    DetectorBuildParams,
    EmbedParams,
    HierarchyConfig,
    LayoutConfig,
    infer,
    space_from_layout,
    train_hierarchy,
)


def main() -> None:

    # 0. Подготавливаем MNIST и делитель картинок Собеля
    total_codes = 0
    codes = defaultdict(list)
    dataset = MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    extractor = MnistSobelAngleMap(angle_in_degrees=True, grad_threshold=0.05)
    
    # 1. Создаем Энкодер и слои первичные детекторы
    encoder = Encoder(
        # Angle
        ClosedDimension("Angle", (0.0, 360.0), [
            Detectors(360, 0.4),
            Detectors(180, 0.4),
            Detectors(90, 0.4),
            Detectors(45, 0.4),
            Detectors(30, 0.4),
            Detectors(10, 0.4),
            Detectors(5, 0.4),
        ]),
        # X
        OpenedDimension("X", (0, 6), [
            Detectors(7, 0.4),
            Detectors(4, 0.4),
            Detectors(2, 0.4),
            Detectors(1, 0.4),
        ]),
        # Y
        OpenedDimension("Y", (0, 6), [
            Detectors(7, 0.4),
            Detectors(4, 0.4),
            Detectors(2, 0.4),
            Detectors(1, 0.4),
        ]),
    )


    

    # количество картинок
    count = 100


    # достаем картинки из MNIST
    digits = []
    for img_tensor, label in dataset:
        digits.append((img_tensor, label))
        if len(digits) == count:
            break

    # кодируем картинки
    for i in range(len(digits)):
        img_tensor, label = digits[i]
        img = img_tensor.squeeze(0).numpy()

        digitValues = extractor.extract(img, label)

        print("label:", label)
        print("кол-во квадратов:", len(digitValues[label]))

        for (a, x, y) in digitValues[label]:
            print(f"Encoding: {label} -> \t{a}\t{x}\t{y}")

            # получаем разряженный битовый код
            values, code = encoder.encode(
                float(a), 
                float(x), 
                float(y)
            )

            print(f"Encoded to: {values} -> {code}")
            codes[a].append(code)
            total_codes += 1
            #show(encoder, values, code, img, int(label))
            
            
    print(f"{label}-{count}-{total_codes}")
    

    # 2. Раскладка
    rr.init("damp-layout")
    rr.spawn()
    layout = Layout(
        codes,
        empty_ratio=0.5,
        similarity="cosine",
        lambda_threshold=0.06,
        eta=0.0,
        seed=0,
    )
    log_layout(layout, step=0)
    step_offset = 1
    layout.run(
        steps=22000,
        pairs_per_step=1200,
        pair_radius=layout.width // 2,
        mode="long",
        min_swap_ratio=0.001,
        log_every=1,
        step_offset=step_offset,
        energy_radius=7,
        energy_check_every=5,
        energy_delta=5e-4,
        energy_patience=4,
    )
    step_offset += layout.last_steps
    layout.run(
        steps=900,
        pairs_per_step=500,
        pair_radius=7,
        mode="short",
        local_radius=7,
        min_swap_ratio=0.001,
        log_every=1,
        step_offset=step_offset,
    )

    # 3. Детекторы
    v0 = space_from_layout(layout)


    build_l1 = DetectorBuildParams(
        lambda_levels=[0.5, 0.6, 0.7],
        activation_radius=7,
        energy_radius=7,
        detector_code_length=256,
        cluster_eps=2.5,
        cluster_min_points=3,
        energy_threshold_mu=0.05,
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
        lambda_levels=[0.5, 0.65, 0.8],
        activation_radius=5,
        energy_radius=5,
        detector_code_length=256,
        cluster_eps=2.0,
        cluster_min_points=3,
        energy_threshold_mu=0.05,
        energy_lambda=0.6,
        max_attempts=600,
        max_detectors_per_layer=100,
        min_radius=1.0,
        patience=160,
        similarity="cosine",
        eta=None,
        seed=1,
    )
    build_l3 = DetectorBuildParams(
        lambda_levels=[0.55, 0.7, 0.85],
        activation_radius=4,
        energy_radius=4,
        detector_code_length=256,
        cluster_eps=2.0,
        cluster_min_points=3,
        energy_threshold_mu=0.05,
        energy_lambda=0.6,
        max_attempts=500,
        max_detectors_per_layer=80,
        min_radius=1.0,
        patience=120,
        similarity="cosine",
        eta=None,
        seed=2,
    )
    embed_l1 = EmbedParams(
        lambda_activation=0.55,
        mu_e=0.05,
        mu_d=0.5,
        sigma=50,
        similarity="cosine",
        eta=None,
        merge_order="high",
    )
    embed_l2 = EmbedParams(
        lambda_activation=0.55,
        mu_e=0.05,
        mu_d=0.5,
        sigma=40,
        similarity="cosine",
        eta=None,
        merge_order="high",
    )
    embed_l3 = EmbedParams(
        lambda_activation=0.6,
        mu_e=0.05,
        mu_d=0.5,
        sigma=30,
        similarity="cosine",
        eta=None,
        merge_order="high",
    )
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

    model = train_hierarchy(digits, config)

    max_test_digits = 100
    test_dataset = MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())
    test_digits = []
    for img_tensor, label in test_dataset:
        test_digits.append((img_tensor, label))
        if len(test_digits) == max_test_digits:
            break

    correct = 0
    total = len(test_digits)
    print(f"Total test images: {total}")
    for i in range(total):
        test_digit_image, test_label = test_digits[i]
        predicted, debug = infer(test_digit_image, model, top_k=3, similarity="cosine")
        if predicted == int(test_label):
            correct += 1
            print("CORRECT!!!")
        print(f"infer for {test_label}[{i}] -> {predicted}")
    accuracy = (correct / total) if total else 0.0
    print(f"total accuracy for {test_label}: {accuracy:.4f} ({correct}/{total})")


    wait_for_close()



if __name__ == "__main__":
    main()
