from damp_encoder import Encoder, ClosedDimension, OpenedDimension, Detectors
from bit_printer import print_bits
from visualize_detectors import show, wait_for_close

def main() -> None:
    encoder = Encoder(
        ClosedDimension((0.0, 359.0), [
            Detectors(360, 0.4),
            Detectors(180, 0.4),
            Detectors(90, 0.4),
            Detectors(45, 0.4),
        ]),
        OpenedDimension((0.0, 500.0), [
            Detectors(500, 0.4),
            Detectors(250, 0.4),
            Detectors(50, 0.4),
            Detectors(10, 0.4),
            Detectors(2, 0.4),
        ]),
    )

    for angle in range(360):
        for x in range(500):
            values, code = encoder.encode(float(angle), float(x))
            show(encoder, values, code)
    
    wait_for_close()



if __name__ == "__main__":
    main()
