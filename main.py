from damp_encoder import Encoder, ClosedDimension, OpenedDimension, Detectors
from visualize_detectors import show, wait_for_close

def main() -> None:
    
    encoder = Encoder(
        # Angle
        ClosedDimension((0.0, 360.0), [
            Detectors(360, 0.4),
            Detectors(180, 0.4),
            Detectors(90, 0.4),
            Detectors(45, 0.4),
            Detectors(30, 0.4),
            Detectors(10, 0.4),
            Detectors(5, 0.4),
        ]),
        # X
        OpenedDimension((0, 6), [
            Detectors(7, 0.4),
            Detectors(4, 0.4),
            Detectors(2, 0.4),
            Detectors(1, 0.4),
        ]),
        # Y
        OpenedDimension((0, 6), [
            Detectors(7, 0.4),
            Detectors(4, 0.4),
            Detectors(2, 0.4),
            Detectors(1, 0.4),
        ]),
    )

    for angle in range(360):
        for x in range(28):
            for y in range(28):
                values, code = encoder.encode(float(angle), float(x), float(y))
                print(f"{values} -> {code}")
                show(encoder, values, code)
    
    wait_for_close()



if __name__ == "__main__":
    main()
