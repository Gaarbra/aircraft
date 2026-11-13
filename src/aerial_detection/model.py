"""Set up Faster R-CNN for background (0) and aircraft (1)."""

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d


def prepare_for_training(model):
    # 1. Replace the old 91-class output layer with a new two-class layer.
    # This new layer starts with random weights and must learn from our data.
    input_size = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(input_size, 2)

    # 2. Keep the pretrained image-feature extractor unchanged to save CPU work.
    # The region-proposal and detection layers will still learn.
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    return model


def load_trained_model(checkpoint):
    # Recreate the same model without downloading any pretrained weights.
    # Match the frozen normalization used by the COCO-pretrained backbone.
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50", weights=None, norm_layer=FrozenBatchNorm2d
    )
    model = FasterRCNN(backbone, num_classes=2)
    # COCO_V1 uses zero epsilon; this setting is not stored in state_dict.
    for layer in model.modules():
        if isinstance(layer, FrozenBatchNorm2d):
            layer.eps = 0.0
    model = prepare_for_training(model)
    model.load_state_dict(checkpoint["model_state"])
    return model.eval()
