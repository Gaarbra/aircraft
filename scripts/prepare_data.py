"""Prepare a small learning dataset. Run this before training a model.

Run: .venv/Scripts/python.exe scripts/prepare_data.py
Uses the mirror's existing splits, not the official DIOR benchmark splits.
"""

import hashlib
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from PIL import Image

from aerial_detection.data import xywh_to_xyxy

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "aircraft"
SOURCE = "HichTala/dior"

# Start small so we can learn and test on a CPU.
# Each pair means: images with aircraft, images without aircraft labels.
SPLITS = {"train": (30, 10), "validation": (8, 2), "test": (8, 2)}


def download(url):
    with urlopen(url, timeout=60) as response:
        return response.read()


def get_rows(split, offset):
    # Ask for annotations first. Download photos only when we need them.
    query = urlencode({"dataset": SOURCE, "config": "default", "split": split,
                       "offset": offset, "length": 100})
    return json.loads(download("https://datasets-server.huggingface.co/rows?" + query))


def prepare_split(split, wanted_positive, wanted_negative, seen_ids, seen_pixels):
    folder = DATA / split
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"

    # Keep an existing selection unchanged on later runs.
    if manifest_path.exists():
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        positives = sum(bool(s["aircraft_boxes_xywh"]) for s in document["samples"])
        negatives = len(document["samples"]) - positives
        if (positives, negatives) != (wanted_positive, wanted_negative):
            raise ValueError("Cached split counts differ from the requested dataset")
        for sample in document["samples"]:
            content = (folder / sample["file"]).read_bytes()
            if hashlib.sha256(content).hexdigest() != sample["sha256"]:
                raise ValueError(f"Cached image changed: {sample['file']}")
            if sample["image_id"] in seen_ids or sample["pixel_sha256"] in seen_pixels:
                raise ValueError("Duplicate image across saved splits")
            seen_ids.add(sample["image_id"])
            seen_pixels.add(sample["pixel_sha256"])
        print(f"{split}: kept {len(document['samples'])} cached images", flush=True)
        return

    samples = []
    positive_count = negative_count = 0
    skipped = []

    # This is a bounded starter selection, not a representative benchmark.
    for offset in range(0, 1000, 100):
        page = get_rows(split, offset)
        features = next(f["type"] for f in page["features"] if f["name"] == "objects")
        names = features["category"]["feature"]["names"]
        aircraft_id = names.index("Airplane")

        for entry in page["rows"]:
            row = entry["row"]
            image_id = row["image_id"]
            if image_id in seen_ids:
                continue
            if entry.get("truncated_cells"):
                raise ValueError("Incomplete annotations returned by the server")
            objects = row["objects"]
            if len(objects["bbox"]) != len(objects["category"]):
                raise ValueError("The number of boxes and labels differs")
            boxes = [box for box, label in zip(objects["bbox"], objects["category"])
                     if label == aircraft_id]
            positive = bool(boxes)
            if positive and positive_count >= wanted_positive:
                continue
            if not positive and negative_count >= wanted_negative:
                continue

            # Check the boxes before saving an image. Never silently clip labels.
            try:
                for box in boxes:
                    xywh_to_xyxy(box, row["width"], row["height"])
            except ValueError as error:
                skipped.append({"image_id": image_id, "reason": str(error)})
                continue

            content = download(row["image"]["src"])
            with Image.open(BytesIO(content)) as source:
                photo = source.convert("RGB")
            if photo.size != (row["width"], row["height"]):
                raise ValueError(f"Image {image_id}: pixels and annotation dimensions differ")
            pixel_hash = hashlib.sha256(str(photo.size).encode() + photo.tobytes()).hexdigest()
            if pixel_hash in seen_pixels:
                skipped.append({"image_id": image_id, "reason": "duplicate pixels"})
                continue

            # Save the photo and its original annotations together.
            filename = f"{image_id}.jpg"
            (folder / filename).write_bytes(content)
            samples.append({"image_id": image_id, "row_index": entry["row_idx"],
                            "file": filename, "width": photo.width, "height": photo.height,
                            "sha256": hashlib.sha256(content).hexdigest(), "pixel_sha256": pixel_hash,
                            "aircraft_boxes_xywh": boxes, "original_objects": objects})
            seen_ids.add(image_id)
            seen_pixels.add(pixel_hash)
            positive_count += int(positive)
            negative_count += int(not positive)
            print(f"{split}: {len(samples)} images saved ({positive_count} positive, {negative_count} negative)", flush=True)

            if (positive_count, negative_count) == (wanted_positive, wanted_negative):
                break
        if (positive_count, negative_count) == (wanted_positive, wanted_negative):
            break

    if (positive_count, negative_count) != (wanted_positive, wanted_negative):
        raise RuntimeError(f"Not enough suitable {split} examples in first 1000 rows")

    document = {"source": SOURCE, "source_split": split, "source_box_format": "COCO xywh",
                "source_class_names": names, "source_aircraft_category": aircraft_id,
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "selection": "first eligible rows in mirror split, with fixed positive/negative quotas",
                "limits": "small learning subset; no geographic metadata or near-duplicate guarantee",
                "excluded_sample_ids": [2003, 2007, 2033], "skipped": skipped, "samples": samples}
    manifest_path.write_text(json.dumps(document, indent=2), encoding="utf-8")


def main():
    # 1. Exclude the three photos we already used for the demonstration.
    seen_ids = {2003, 2007, 2033}
    seen_pixels = set()
    sample_folder = ROOT / "data" / "sample"
    for image_id in seen_ids:
        with Image.open(sample_folder / f"{image_id}.jpg") as source:
            photo = source.convert("RGB")
        seen_pixels.add(hashlib.sha256(str(photo.size).encode() + photo.tobytes()).hexdigest())

    # 2. Keep training, validation, and test images in separate folders.
    for split, (positive, negative) in SPLITS.items():
        prepare_split(split, positive, negative, seen_ids, seen_pixels)

    # 3. Audit the saved data before we use it for training.
    print("\nDownload complete. Next run: python scripts/check_data.py")


if __name__ == "__main__":
    main()
