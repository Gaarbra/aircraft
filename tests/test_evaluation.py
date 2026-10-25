import unittest

import torch

from aerial_detection.evaluation import match_detections, summarize_counts


def boxes(values):
    return torch.tensor(values, dtype=torch.float32).reshape(-1, 4)


class MatchingTests(unittest.TestCase):
    def test_duplicate_counts_once(self):
        result = match_detections(boxes([[0, 0, 10, 10]] * 2), torch.tensor([0.6, 0.9]), boxes([[0, 0, 10, 10]]))
        self.assertEqual((result['tp'], result['fp'], result['fn']), (1, 1, 0))
        self.assertEqual(result['matches'][0]['prediction_index'], 1)

    def test_low_overlap_is_false_positive_and_miss(self):
        result = match_detections(boxes([[8, 8, 18, 18]]), torch.tensor([0.9]), boxes([[0, 0, 10, 10]]))
        self.assertEqual((result['tp'], result['fp'], result['fn']), (0, 1, 1))

    def test_empty_predictions(self):
        result = match_detections(boxes([]), torch.tensor([]), boxes([[0, 0, 10, 10]]))
        self.assertEqual((result['tp'], result['fp'], result['fn']), (0, 0, 1))
        self.assertIsNone(summarize_counts(0, 0, 1)['precision'])

    def test_empty_targets(self):
        result = match_detections(boxes([[0, 0, 10, 10]]), torch.tensor([0.9]), boxes([]))
        self.assertEqual((result['tp'], result['fp'], result['fn']), (0, 1, 0))

    def test_iou_threshold_inclusive(self):
        result = match_detections(boxes([[0, 0, 5, 10]]), torch.tensor([0.9]), boxes([[0, 0, 10, 10]]))
        self.assertEqual(result['tp'], 1)


if __name__ == '__main__':
    unittest.main()
