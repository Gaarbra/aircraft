import unittest

import torch

from aerial_detection.data import detection_collate, xywh_to_xyxy


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


if __name__ == "__main__":
    unittest.main()
