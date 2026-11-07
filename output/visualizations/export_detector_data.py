from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import main
from damp.logging import LOGGER


class DetectorDataExporter:
    def export(self) -> None:
        directory = Path(__file__).resolve().parent
        with patch.object(LOGGER, 'event'), patch.object(LOGGER, 'event_lazy'):
            encoder = main._build_encoder()
            layers = []
            for layer in encoder._spatial_encoder._joint_layer_indices:
                fields = layer.fields_by_dimension[0]
                layers.append(dict(count=len(fields), offset=layer.offset,
                                   centers=[f.center for f in fields], half=fields[0].half_width))
            angular = []
            for index, (count, overlap) in enumerate(main.MNIST_ORIENTATION_ANGLE_LAYERS):
                detectors = [dict(center=d.center, half=d.half_width, index=d.detector_index)
                             for d in encoder._angle_encoder.detectors if d.layer_index == index]
                angular.append(dict(count=count, overlap=overlap, detectors=detectors,
                                    width=detectors[0]['half'] * 2))
            samples = []
            for angle in range(int(main.MNIST_ORIENTATION_MIN), int(main.MNIST_ORIENTATION_MAX)):
                descriptor, code = encoder.encode_at(angle, center_index=encoder.center_index(1, 1))
                samples.append(dict(angle=angle, encoded=descriptor.encoded_angle,
                                    x=descriptor.position_x, y=descriptor.position_y,
                                    active=list(code.active_indices())))
            data = dict(layers=layers, angular_layers=angular, samples=samples,
                        spatial=encoder.spatial_code_length, angular=encoder.angular_code_length)
        serialized = json.dumps(data, separators=(',', ':'))
        (directory / 'detector-data.json').write_text(serialized, encoding='utf-8')
        path = directory / 'wide-detectors.html'
        fragment = path.read_text(encoding='utf-8')
        fragment, count = re.subn(
            r'(?<=<script type="application/json" id="wd-data">).*?(?=</script>)',
            lambda _: serialized, fragment, flags=re.DOTALL,
        )
        if count != 1:
            raise ValueError('expected one detector data block')
        path.write_text(fragment, encoding='utf-8')


if __name__ == '__main__':
    DetectorDataExporter().export()
