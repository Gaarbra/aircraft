"""Read our small inspection manifest as a PyTorch detection dataset."""

import json
import math
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


def xywh_to_xyxy(box, width, height):
    """Convert COCO [left, top, width, height] to corner coordinates."""
    if len(box) != 4:
        raise ValueError("A box must contain four coordinates")
    x, y, w, h = map(float, box)
    if not all(math.isfinite(value) for value in (x, y, w, h)):
        raise ValueError("Box coordinates must be finite")
    if w <= 0 or h <= 0 or not (0 <= x < x + w <= width and 0 <= y < y + h <= height):
        raise ValueError(f"Invalid or out-of-bounds box: {box}")
    return [x, y, x + w, y + h]


class AircraftDataset(Dataset):
    def __init__(self, manifest):
        self.manifest_path = Path(manifest)
        self.document = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.samples = self.document["samples"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(self.manifest_path.parent / sample["file"]) as source:
            photo = source.convert("RGB")
        if photo.size != (sample["width"], sample["height"]):
            raise ValueError("Image dimensions do not match its annotations")
        image = pil_to_tensor(photo).float() / 255.0
        boxes = torch.tensor([
            xywh_to_xyxy(box, photo.width, photo.height)
            for box in sample["aircraft_boxes_xywh"]
        ], dtype=torch.float32).reshape(-1, 4)
        # Faster R-CNN reserves 0 for background, so our aircraft label is 1.
        target = {
            "boxes": boxes,
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "image_id": torch.tensor(sample["image_id"], dtype=torch.int64),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }
        return image, target


def detection_collate(batch):
    """Keep images/targets in lists: images can have different box counts."""
    images, targets = zip(*batch)
    return list(images), list(targets)
