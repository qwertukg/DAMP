from damp_encoder import Encoder, ClosedDimension, OpenedDimension, Detectors
from visualize_detectors import show, wait_for_close
import numpy as np
from MnistSobelAngleMap import MnistSobelAngleMap
from torchvision.datasets import MNIST
from torchvision import transforms
from collections import defaultdict
import json
from pathlib import Path
from layout import Layout


def main() -> None:
    
    encoder = Encoder(
        ClosedDimension("Angle", (0.0, 360.0), [
            Detectors(360, 0.4),
            Detectors(180, 0.4),
            Detectors(90, 0.4),
            Detectors(45, 0.4),
            Detectors(30, 0.4),
            Detectors(10, 0.4),
            Detectors(5, 0.4),
        ]),
    )

    total_codes = 0
    codes = defaultdict(list)
    empty_code = tuple(0.0 for _ in range(encoder.code_length))
    codes[None].extend([empty_code] * 100)

    for a in range(360):
        values, code = encoder.encode(float(a))
        print(f"Encoded to: {values} -> {code}")
        codes[a].append(code)
        total_codes += 1
        #show(encoder, values, code, None, int(a))
    
    print(f"{total_codes} codes saved to codes.json")


    layout = Layout(
        codes,
        lambda_start=0.3,  # начальный порог λ для simλ
        lambda_end=0.85,  # конечный порог λ к завершению раскладки
        rr_app_id="mnist_layout",  # имя сессии для визуализации rerun
    )

    layout.layout(
        long_steps=500,  # число шагов дальнего порядка
        short_steps=0,  # число шагов ближнего порядка
        pairs_per_step=64,  # количество тестируемых пар за шаг
        long_pair_radius=None,  # без ограничения радиуса для дальних пар
        short_pair_radius=6,  # радиус выбора второй точки для ближних пар
        short_local_radius=6,  # минимальный радиус окрестности для локальной энергии
        visualize=True,  # включить потоковую визуализацию
        visualize_every=1,  # логировать каждый шаг
        energy_radius=5,  # радиус для матрицы энергии в визуализации
    )

    wait_for_close()



if __name__ == "__main__":
    main()
