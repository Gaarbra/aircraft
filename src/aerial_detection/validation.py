"""Measure predictions without changing the model's weights."""

import json

import torch

from aerial_detection.evaluation import match_detections, summarize_counts
from aerial_detection.visualization import save_boxes


def evaluate(model, dataset, aircraft_label, output_folder):
    output_folder.mkdir(parents=True, exist_ok=True)
    model.eval()
    results = []

    # Validation does not need gradients or optimizer updates.
    with torch.inference_mode():
        for index in range(len(dataset)):
            image, target = dataset[index]
            prediction = model([image])[0]
            keep = (prediction["labels"] == aircraft_label) & (prediction["scores"] >= 0.5)
            boxes = prediction["boxes"][keep]
            scores = prediction["scores"][keep]
            matches = match_detections(boxes, scores, target["boxes"], iou_threshold=0.5)
            image_id = target["image_id"].item()
            results.append({"image_id": image_id, **matches,
                            "boxes": boxes.tolist(), "scores": scores.tolist()})
            save_boxes(image, target["boxes"], output_folder / f"{image_id}_labels.png")
            save_boxes(image, boxes, output_folder / f"{image_id}_predictions.png")
            print(f"  validation {index + 1}/{len(dataset)}: {matches['tp']} matched, {matches['fp']} false alarms, {matches['fn']} missed", flush=True)

    counts = summarize_counts(*(sum(row[key] for row in results) for key in ("tp", "fp", "fn")))
    report = {"score_threshold": 0.5, "iou_threshold": 0.5,
              "scope": "small validation subset, not test results or mAP",
              "summary": counts, "images": results}
    (output_folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return counts
