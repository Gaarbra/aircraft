"""Check that fine-tuning freezes features but leaves the new head trainable."""

import unittest

import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.misc import FrozenBatchNorm2d

from aerial_detection.model import load_trained_model, prepare_for_training


class ModelTests(unittest.TestCase):
    def test_freezing_and_checkpoint_reload(self):
        torch.set_num_threads(2)
        backbone = resnet_fpn_backbone(
            backbone_name="resnet50", weights=None, norm_layer=FrozenBatchNorm2d
        )
        model = prepare_for_training(FasterRCNN(backbone, num_classes=91))
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.roi_heads.box_predictor.parameters()))
        self.assertEqual(model.roi_heads.box_predictor.cls_score.out_features, 2)
        restored = load_trained_model({"model_state": model.state_dict()})
        self.assertFalse(restored.training)
        self.assertIsInstance(restored.backbone.body.bn1, FrozenBatchNorm2d)
        self.assertEqual(restored.backbone.body.bn1.eps, 0.0)
        self.assertTrue(torch.equal(model.roi_heads.box_predictor.cls_score.weight,
                                    restored.roi_heads.box_predictor.cls_score.weight))


if __name__ == "__main__":
    unittest.main()
