import unittest
import json
import tempfile
from pathlib import Path

import torch
from PIL import Image

from aerial_detection.data import AircraftDataset, detection_collate, xywh_to_xyxy


class DataTests(unittest.TestCase):
    def test_conversion(self):
        self.assertEqual(xywh_to_xyxy([20, 30, 40, 20], 100, 100), [20, 30, 60, 50])

    def test_reject_invalid_boxes(self):
        for box in ([0, 0, 0, 5], [-1, 0, 5, 5], [90, 0, 20, 5], [0, 0, float('nan'), 1]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                xywh_to_xyxy(box, 100, 100)

    def test_variable_box_counts(self):
        batch = [(torch.zeros(3, 10, 10), {"boxes": torch.zeros(n, 4)}) for n in (0, 3)]
        images, targets = detection_collate(batch)
        self.assertEqual(len(images), 2)
        self.assertEqual([len(t["boxes"]) for t in targets], [0, 3])

    def test_dataset_with_and_without_aircraft(self):
        # A negative image must still load with boxes shaped [0, 4].
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            Image.new('RGB', (20, 10), 'white').save(folder / 'image.png')
            samples = []
            for index, boxes in enumerate(([], [[2, 1, 8, 5]])):
                samples.append({'file': 'image.png', 'width': 20, 'height': 10,
                                'image_id': index, 'aircraft_boxes_xywh': boxes})
            manifest = folder / 'manifest.json'
            manifest.write_text(json.dumps({'samples': samples}))
            dataset = AircraftDataset(manifest)
            image, empty = dataset[0]
            self.assertEqual(image.shape, (3, 10, 20))
            self.assertEqual(image.max().item(), 1.0)
            self.assertEqual(empty['boxes'].shape, (0, 4))
            self.assertEqual(empty['labels'].dtype, torch.int64)
            _, positive = dataset[1]
            self.assertEqual(positive['boxes'].tolist(), [[2, 1, 10, 6]])
            self.assertEqual(positive['area'].tolist(), [40])
            self.assertEqual(positive['labels'].tolist(), [1])


if __name__ == "__main__":
    unittest.main()
