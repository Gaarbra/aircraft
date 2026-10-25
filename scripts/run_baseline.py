"""Run pretrained COCO Faster R-CNN on the inspection sample, without training.

Run: .venv/Scripts/python.exe scripts/run_baseline.py
The first run downloads about 160 MB of model weights into data/model_cache.
These three selected images are an inspection sample, not a benchmark.
"""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import to_pil_image

from aerial_detection.data import AircraftDataset
from aerial_detection.evaluation import match_detections, summarize_counts

ROOT = Path(__file__).resolve().parents[1]


def draw_comparison(image, targets, prediction, image_id, metrics, threshold):
    canvas = Image.new("RGB", (1040, 585), "#f7f9fc")
    font = ImageFont.load_default(size=20)
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 10), f"Image {image_id}: {metrics['tp']} matched, {metrics['fp']} false positives, {metrics['fn']} missed", fill="#17243b", font=font)
    for offset, boxes, color, title, scores in [
        (10, targets, "#ffb000", f"Dataset labels: {len(targets)} aircraft", None),
        (530, prediction["boxes"], "#00d9ff", f"Model: {len(prediction['boxes'])} aircraft (score >= {threshold:g})", prediction["scores"]),
    ]:
        photo = to_pil_image(image)
        photo_draw = ImageDraw.Draw(photo)
        for index, box in enumerate(boxes.tolist()):
            photo_draw.rectangle(box, outline=color, width=3)
            if scores is not None:
                photo_draw.text((box[0], max(0, box[1] - 22)), f"{scores[index].item():.2f}", fill=color, font=font, stroke_width=1, stroke_fill="black")
        photo.thumbnail((500, 500))
        canvas.paste(photo, (offset, 75))
        draw.text((offset, 45), title, fill="#17243b", font=font)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.05 <= args.score_threshold <= 1 or not 0 < args.iou_threshold <= 1:
        parser.error("score threshold must be in [0.05, 1]; IoU threshold in (0, 1]")

    # Keep weights in ignored project data. Explicit weights keep the baseline stable.
    torch.hub.set_dir(str(ROOT / "data" / "model_cache"))
    torch.set_num_threads(4)
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    categories = weights.meta["categories"]
    airplane_id = categories.index("airplane")
    print(f"Loading pretrained Faster R-CNN on CPU; COCO airplane label = {airplane_id}", flush=True)
    model = fasterrcnn_resnet50_fpn(weights=weights).eval()
    dataset = AircraftDataset(ROOT / "data" / "sample" / "manifest.json")
    output_dir = ROOT / "outputs" / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    panels = []
    for index in range(len(dataset)):
        image, target = dataset[index]
        image_id = target["image_id"].item()
        print(f"Running image {image_id} ({index + 1}/{len(dataset)})...", flush=True)
        started = time.perf_counter()
        # Dataset already supplies float RGB pixels in [0, 1]. The model handles
        # its own resizing and normalization. Do not normalize a second time.
        with torch.inference_mode():
            raw = model([image])[0]
        seconds = time.perf_counter() - started
        # Our custom dataset label 1 means aircraft, but COCO label 1 is person.
        # Filter by the pretrained weights' category metadata, not our label IDs.
        keep = (raw["labels"] == airplane_id) & (raw["scores"] >= args.score_threshold)
        prediction = {"boxes": raw["boxes"][keep], "scores": raw["scores"][keep]}
        metrics = match_detections(prediction["boxes"], prediction["scores"], target["boxes"], args.iou_threshold)
        all_predictions = [{"label": categories[label], "score": score, "box_xyxy": box}
                           for label, score, box in zip(raw["labels"].tolist(), raw["scores"].tolist(), raw["boxes"].tolist())]
        record = {"image_id": image_id, "image_sha256": dataset.samples[index]["sha256"],
                  "target_count": len(target["boxes"]), "predicted_aircraft_count": len(prediction["boxes"]),
                  "inference_seconds": seconds, **metrics,
                  "target_boxes_xyxy": target["boxes"].tolist(),
                  "aircraft_boxes_xyxy": prediction["boxes"].tolist(), "aircraft_scores": prediction["scores"].tolist(),
                  "all_model_detections": all_predictions}
        results.append(record)
        panel = draw_comparison(image, target["boxes"], prediction, image_id, metrics, args.score_threshold)
        panel.save(output_dir / f"{image_id}_comparison.png")
        panels.append(panel)
        print(f"  {metrics['tp']} matched, {metrics['fp']} false positives, {metrics['fn']} missed; {seconds:.1f}s", flush=True)

    counts = summarize_counts(*(sum(r[key] for r in results) for key in ("tp", "fp", "fn")))
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "three selected positive images; inspection only, not held-out accuracy or mAP",
              "source": dataset.document["source"], "source_split": dataset.document["source_split"],
              "manifest_sha256": hashlib.sha256(dataset.manifest_path.read_bytes()).hexdigest(),
              "model": "fasterrcnn_resnet50_fpn", "weights": "COCO_V1", "weights_url": weights.url,
              "torch": torch.__version__, "torchvision": torchvision.__version__, "device": "cpu", "threads": 4,
              "score_threshold": args.score_threshold, "iou_threshold": args.iou_threshold,
              "model_score_floor": 0.05, "model_max_detections_per_image": 100,
              "matching": "descending confidence, best unmatched target at IoU threshold",
              "summary": counts, "images": results}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    sheet = Image.new("RGB", (1040, 585 * len(panels)))
    for index, panel in enumerate(panels):
        sheet.paste(panel, (0, index * 585))
    sheet.save(output_dir / "comparison_sheet.png")
    print(json.dumps(counts, indent=2))
    print(f"Saved predictions, comparisons, and settings: {output_dir}")
    print("No training performed. These sample counts are not a benchmark score.")


if __name__ == "__main__":
    main()
