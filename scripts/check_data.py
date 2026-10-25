"""Check saved labels and images, then save plain training/validation previews."""

import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from aerial_detection.data import AircraftDataset, detection_collate
from aerial_detection.visualization import save_boxes

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(4)
    seen_ids = set()
    seen_pixels = set()
    summary = {}

    # 1. Check every image, not just the images we preview.
    for split in ("train", "validation", "test"):
        dataset = AircraftDataset(ROOT / "data" / "aircraft" / split / "manifest.json")
        positive_count = negative_count = box_count = tiny_count = 0
        for index, sample in enumerate(dataset.samples):
            path = dataset.manifest_path.parent / sample["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != sample["sha256"]:
                raise ValueError(f"Image checksum changed: {path}")
            with Image.open(path) as source:
                photo = source.convert("RGB")
            pixel_hash = hashlib.sha256(str(photo.size).encode() + photo.tobytes()).hexdigest()
            if pixel_hash != sample["pixel_sha256"]:
                raise ValueError(f"Pixel checksum changed: {path}")
            if sample["image_id"] in seen_ids or pixel_hash in seen_pixels:
                raise ValueError("Duplicate image found within or across splits")
            if sample["image_id"] in (2003, 2007, 2033):
                raise ValueError("A demonstration image entered the new dataset")
            seen_ids.add(sample["image_id"])
            seen_pixels.add(pixel_hash)

            # 2. Use the same loader the future training loop will use.
            image, target = dataset[index]
            if not torch.isfinite(image).all() or not (0 <= image.min() <= image.max() <= 1):
                raise ValueError("Image pixels must be finite and scaled to [0, 1]")
            boxes = target["boxes"]
            positive_count += int(len(boxes) > 0)
            negative_count += int(len(boxes) == 0)
            box_count += len(boxes)
            sizes = boxes[:, 2:] - boxes[:, :2]
            tiny_count += int((sizes.min(dim=1).values < 8).sum())

            # 3. Save full-size boxes only. Keep test images out of tuning previews.
            if split != "test":
                output = ROOT / "outputs" / "data_check" / split
                output.mkdir(parents=True, exist_ok=True)
                save_boxes(image, boxes, output / f"{sample['image_id']}_labels.png")

        # 4. Confirm all images can pass through batches, including empty targets.
        loader = DataLoader(dataset, batch_size=2, collate_fn=detection_collate)
        batch_count = sum(1 for _ in loader)
        summary[split] = {"images": len(dataset), "positive_images": positive_count,
                          "negative_images": negative_count, "aircraft_boxes": box_count,
                          "boxes_with_side_under_8_pixels": tiny_count, "batches_at_size_2": batch_count}
        print(f"{split}: {len(dataset)} images, {box_count} aircraft boxes, {negative_count} negative images", flush=True)

    # 5. Save the counts so we can compare future datasets.
    report = {"checks": "image dimensions, checksums, valid boxes, pixels, duplicate IDs/pixels, batches",
              "limits": "does not certify label completeness or geographic separation; manual review still needed",
              "splits": summary}
    output = ROOT / "outputs" / "data_check"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Checks passed. Review the plain label images before training.")


if __name__ == "__main__":
    main()
