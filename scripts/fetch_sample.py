"""Download three aircraft images from the first 100 mirror training rows."""

import hashlib
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "HichTala/dior"


def read_url(url):
    with urlopen(url, timeout=60) as response:
        return response.read()


def main():
    destination = ROOT / "data" / "sample"
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all((destination / s["file"]).exists() and
               hashlib.sha256((destination / s["file"]).read_bytes()).hexdigest() == s["sha256"]
               for s in document["samples"]) and len(document["samples"]) == 3:
            print(f"Three cached images verified: {manifest_path}")
            return

    query = urlencode({"dataset": SOURCE, "config": "default", "split": "train", "offset": 0, "length": 100})
    response = json.loads(read_url("https://datasets-server.huggingface.co/rows?" + query))
    objects_feature = next(f["type"] for f in response["features"] if f["name"] == "objects")
    names = objects_feature["category"]["feature"]["names"]
    aircraft_id = next(i for i, name in enumerate(names) if name.lower() == "airplane")
    selected = [r for r in response["rows"] if aircraft_id in r["row"]["objects"]["category"]][:3]
    if len(selected) != 3:
        raise RuntimeError("Fewer than three aircraft rows found; no larger download attempted")
    destination.mkdir(parents=True, exist_ok=True)
    samples = []
    for entry in selected:
        if entry.get("truncated_cells"):
            raise ValueError("Dataset viewer returned truncated annotations")
        row = entry["row"]
        content = read_url(row["image"]["src"])
        with Image.open(BytesIO(content)) as photo:
            if photo.size != (row["width"], row["height"]):
                raise ValueError("Viewer image dimensions differ from annotation dimensions")
            photo.verify()
        filename = f"{row['image_id']}.jpg"
        (destination / filename).write_bytes(content)
        objects = row["objects"]
        if len(objects["bbox"]) != len(objects["category"]):
            raise ValueError("Mismatched boxes and labels")
        boxes = [box for box, label in zip(objects["bbox"], objects["category"]) if label == aircraft_id]
        samples.append({"image_id": row["image_id"], "row_index": entry["row_idx"],
                        "file": filename, "width": row["width"], "height": row["height"],
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "aircraft_boxes_xywh": boxes, "original_objects": objects})
        print(f"Downloaded {filename}: {len(boxes)} aircraft annotations")
    document = {"source": SOURCE, "source_card": f"https://huggingface.co/datasets/{SOURCE}",
                "source_split": "train", "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": "inspection only; selected positive examples, not an evaluation set",
                "source_box_format": "COCO xywh", "source_aircraft_category": aircraft_id,
                "source_class_names": names, "samples": samples}
    manifest_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(f"Saved provenance and annotations: {manifest_path}")


if __name__ == "__main__":
    main()
