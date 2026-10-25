"""Single-class, confidence-ordered matching at one IoU threshold (not mAP)."""

import torch
from torchvision.ops import box_iou


def match_detections(boxes, scores, targets, iou_threshold=0.5):
    """Match each prediction to the best unmatched target, at most once."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    if len(boxes) != len(scores):
        raise ValueError("Each prediction needs one score")
    overlaps = box_iou(boxes, targets)
    matched = set()
    matches = []
    for prediction_index in torch.argsort(scores, descending=True, stable=True).tolist():
        candidates = [i for i in range(len(targets)) if i not in matched]
        if not candidates:
            continue
        target_index = max(candidates, key=lambda i: overlaps[prediction_index, i].item())
        value = overlaps[prediction_index, target_index].item()
        if value >= iou_threshold:
            matched.add(target_index)
            matches.append({"prediction_index": prediction_index, "target_index": target_index, "iou": value})
    tp = len(matches)
    return {"tp": tp, "fp": len(boxes) - tp, "fn": len(targets) - tp, "matches": matches}


def summarize_counts(tp, fp, fn):
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None}
