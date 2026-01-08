from damp_encoder import Encoder, ClosedDimension, OpenedDimension, Detectors
from visualize_detectors import show, wait_for_close

def main() -> None:
    encoder = Encoder(
        ClosedDimension((0.0, 179.0), [
            Detectors(180, 0.4),
            Detectors(90, 0.4),
            Detectors(45, 0.4),
        ]),
        OpenedDimension((0.0, 249.0), [
            Detectors(250, 0.4),
            Detectors(50, 0.4),
            Detectors(10, 0.4),
            Detectors(2, 0.4),
        ]),
        OpenedDimension((0.0, 249.0), [
            Detectors(250, 0.4),
            Detectors(50, 0.4),
            Detectors(10, 0.4),
            Detectors(2, 0.4),
        ]),
    )

    for angle in range(180):
        for x in range(249):
            for y in range(249):
                values, code = encoder.encode(float(angle), float(x), float(y))
                show(encoder, values, code)
    
    wait_for_close()



if __name__ == "__main__":
    main()
