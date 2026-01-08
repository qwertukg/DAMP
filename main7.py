from damp_encoder import Encoder, ClosedDimension, OpenedDimension, Detectors
from visualize_detectors import show, wait_for_close
import numpy as np
from MnistSobelAngleMap import MnistSobelAngleMap
from torchvision.datasets import MNIST
from torchvision import transforms
from collections import defaultdict
import json
from pathlib import Path
from damp_layout import Layout
import rerun as rr


def main() -> None:
    
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

    total_codes = 0
    codes = defaultdict(list)
    dataset = MNIST(
        root="./data",  # локальный кэш для MNIST
        train=True,  # использовать обучающую выборку
        download=True,  # скачать, если нет на диске
        transform=transforms.ToTensor(),  # перевод в тензор [0,1]
    )
    extractor = MnistSobelAngleMap(
        angle_in_degrees=True,  # возвращать угол в градусах, а не в радианах
        grad_threshold=0.05,  # пропускать квадраты со слабым градиентом
    )

    value = 9  # цифра для выборки
    count = 100  # количество изображений этой цифры
    digits = []
    for img_tensor, label in dataset:
        if int(label) == value:
            digits.append((img_tensor, label))
            if len(digits) == count:
                break

    for i in range(len(digits)):
        img_tensor, label = digits[i]
        img = img_tensor.squeeze(0).numpy()

        digitValues = extractor.extract(img, label)

        print("label:", label)
        print("кол-во квадратов:", len(digitValues[label]))

        for (a, x, y) in digitValues[label]:
            print(f"Encoding: {label} -> \t{a}\t{x}\t{y}")
            values, code = encoder.encode(
                float(a), 
                float(x), 
                float(y)
            )
            print(f"Encoded to: {values} -> {code}")
            codes[a].append(code)
            total_codes += 1
            #show(encoder, values, code, img, int(label))
            
    print(f"{total_codes} codes saved to codes.json")
    


    layout = Layout(
        codes,
        grid_size=None,  # авто-размер сетки по числу точек
        similarity="jaccard",  # метрика сходства кодов
        lambda_threshold=0.06,  # порог сходства для притяжения
        eta=14.0,  # плавность пороговой функции сходства
        seed=0,  # сид ГПСЧ для начального размещения
    )

    rr.init("damp-layout")
    rr.spawn()
    layout.log_rerun(step=0)

    step_offset = 1
    layout.run(
        steps=22000,  # максимум итераций оптимизации
        pairs_per_step=1200,  # число кандидатных обменов на шаг
        pair_radius=layout.width // 2,  # радиус выборки для дальних пар
        mode="long",  # глобальный режим энергии
        min_swap_ratio=0.005,  # ранняя остановка при малом числе обменов
        log_every=1,  # логировать в rerun каждый N шагов
        step_offset=step_offset,  # смещение таймлайна для логов
        energy_radius=7,  # радиус соседства для энерго-проверок
        energy_check_every=5,  # частота проверки стабильности энергии
        energy_delta=5e-4,  # допуск изменения энергии
        energy_patience=4,  # число проверок до остановки
    )
    step_offset += layout.last_steps
    layout.set_similarity_params(
        lambda_threshold=0.16,  # повысить порог сходства для уточнения
        eta=14.0,  # оставить ту же плавность
    )
    layout.run(
        steps=900,  # итерации локального уточнения
        pairs_per_step=500,  # число кандидатных обменов на шаг
        pair_radius=7,  # радиус выборки для локальных пар
        mode="short",  # локальный режим энергии
        local_radius=7,  # размер локального окружения
        min_swap_ratio=0.005,  # ранняя остановка при малом числе обменов
        log_every=1,  # логировать в rerun каждый N шагов
        step_offset=step_offset,  # смещение таймлайна для логов
    )

    wait_for_close()



if __name__ == "__main__":
    main()
