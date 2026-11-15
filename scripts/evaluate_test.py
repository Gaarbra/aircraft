"""Step 5: compare the original model and my trained model on the test images.

1. Pick a confidence threshold for each model using the validation images.
2. Score both models once on the test images, which training never opened.

Run: .venv/Scripts/python.exe scripts/evaluate_test.py --checkpoint outputs/training/larger_first_run/last.pt
"""

import argparse
import json
from pathlib import Path

import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn

from aerial_detection.data import AircraftDataset
from aerial_detection.evaluation import match_detections, summarize_counts
from aerial_detection.model import load_trained_model

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def predict_all(model, dataset, aircraft_label):
    # Run the model once per image and keep every aircraft box with its score.
    results = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            image, target = dataset[index]
            prediction = model([image])[0]
            is_aircraft = prediction["labels"] == aircraft_label
            results.append((prediction["boxes"][is_aircraft], prediction["scores"][is_aircraft], target["boxes"]))
            print(f"  image {index + 1}/{len(dataset)}", flush=True)
    return results


def score(results, threshold):
    # Drop boxes below the threshold, then count matched, false alarms, missed.
    tp = fp = fn = 0
    for boxes, scores, targets in results:
        keep = scores >= threshold
        counts = match_detections(boxes[keep], scores[keep], targets, iou_threshold=0.5)
        tp, fp, fn = tp + counts["tp"], fp + counts["fp"], fn + counts["fn"]
    summary = summarize_counts(tp, fp, fn)
    # F1 balances precision and recall in one number (1.0 is perfect).
    summary["f1"] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.hub.set_dir(str(ROOT / "data" / "model_cache"))

    # 1. Load the validation and test images.
    folder = ROOT / "data" / "aircraft-larger"
    validation = AircraftDataset(folder / "validation" / "manifest.json")
    test = AircraftDataset(folder / "test" / "manifest.json")

    # 2. Load both models. The original model calls aircraft "airplane".
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    models = {
        "original": (fasterrcnn_resnet50_fpn(weights=weights).eval(), weights.meta["categories"].index("airplane")),
        "trained": (load_trained_model(torch.load(args.checkpoint, map_location="cpu")), 1),
    }

    report = {}
    for name, (model, label) in models.items():
        # 3. On validation, try each threshold and keep the one with the best F1.
        print(f"{name} model, validation images:")
        val_results = predict_all(model, validation, label)
        tries = {t: score(val_results, t) for t in THRESHOLDS}
        best = max(THRESHOLDS, key=lambda t: tries[t]["f1"] or 0)

        # 4. Score the test images once, at that threshold.
        print(f"{name} model, test images (threshold {best}):")
        report[name] = {"threshold": best, "validation": tries[best],
                        "test": score(predict_all(model, test, label), best)}

    out = ROOT / "docs" / "test-results.json"
    out.write_text(json.dumps({"iou_threshold": 0.5, "test_images": len(test), **report}, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
