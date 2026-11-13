"""A small CPU training run with a simple, visible training loop.

Run: .venv/Scripts/python.exe scripts/train.py
For another run: add --run-name second_run
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn

from aerial_detection.data import AircraftDataset, detection_collate
from aerial_detection.model import prepare_for_training
from aerial_detection.validation import evaluate

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--run-name", default="first_run")
    parser.add_argument("--size", choices=("starter", "larger"), default="larger")
    args = parser.parse_args()
    if args.epochs < 1 or not args.run_name.replace('_', '').replace('-', '').isalnum():
        parser.error("Use positive epochs and a run name made of letters, numbers, - or _")

    # 1. Use the same random seed and save this run in its own folder.
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.hub.set_dir(str(ROOT / "data" / "model_cache"))
    output = ROOT / "outputs" / "training" / args.run_name
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    # 2. Load training and validation data. Do not open the test set.
    data_folder = "aircraft" if args.size == "starter" else "aircraft-larger"
    train_data = AircraftDataset(ROOT / "data" / data_folder / "train" / "manifest.json")
    validation_data = AircraftDataset(ROOT / "data" / data_folder / "validation" / "manifest.json")
    if not len(train_data) or not len(validation_data):
        raise ValueError("Training and validation sets must both contain images")
    train_ids = {sample["image_id"] for sample in train_data.samples}
    validation_ids = {sample["image_id"] for sample in validation_data.samples}
    if train_ids & validation_ids:
        raise ValueError("Training and validation image IDs overlap")
    loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=detection_collate,
                        generator=torch.Generator().manual_seed(42))
    config = {"epochs": args.epochs, "batch_size": 1, "learning_rate": 0.001,
              "dataset_size": args.size, "training_images": len(train_data),
              "validation_images": len(validation_data),
              "seed": 42, "device": "cpu", "threads": 4, "frozen_backbone": True,
              "new_head": "randomly initialized, background=0 and aircraft=1",
              "pretrained_weights": "FasterRCNN_ResNet50_FPN_Weights.COCO_V1",
              "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
              "train_manifest_sha256": hashlib.sha256(train_data.manifest_path.read_bytes()).hexdigest(),
              "validation_manifest_sha256": hashlib.sha256(validation_data.manifest_path.read_bytes()).hexdigest()}
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # 3. Measure the original pretrained model on these same validation images.
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights)
    print("Checking the original model on validation images...", flush=True)
    before = evaluate(model, validation_data, weights.meta['categories'].index('airplane'), output / "before")

    # 4. Give the model a new aircraft output layer and create the optimizer.
    model = prepare_for_training(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=config["learning_rate"], momentum=0.9, weight_decay=0.0005)
    history = []

    # 5. Train: predict, calculate loss, calculate gradients, update weights.
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.backbone.eval()
        total_loss = 0.0
        for step, (images, targets) in enumerate(loader, start=1):
            optimizer.zero_grad()

            # During training, Faster R-CNN predicts AND calculates its losses.
            # These measure mistakes in classification, boxes, and proposals.
            loss_parts = model(images, targets)
            loss = sum(loss_parts.values())
            if not torch.isfinite(loss):
                raise RuntimeError("Training stopped because the loss is not finite")

            loss.backward()     # Calculate gradients.
            optimizer.step()    # Use the gradients to update the weights.

            total_loss += loss.item()
            row = {"epoch": epoch, "step": step, "loss": loss.item(),
                   "parts": {name: value.item() for name, value in loss_parts.items()}}
            with (output / "steps.jsonl").open('a', encoding='utf-8') as log:
                log.write(json.dumps(row) + '\n')
            print(f"epoch {epoch}/{args.epochs} | image {step}/{len(loader)} | loss {loss.item():.4f}", flush=True)

        # 6. Save the learned weights after each complete pass through the data.
        average_loss = total_loss / len(loader)
        history.append({"epoch": epoch, "average_training_loss": average_loss})
        torch.save({"model_state": model.state_dict(), "epoch": epoch, "config": config}, output / "last.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"Epoch {epoch} complete. Average training loss: {average_loss:.4f}", flush=True)

    # 7. Measure the trained model without changing its weights.
    print("Checking the trained model on validation images...", flush=True)
    after = evaluate(model, validation_data, aircraft_label=1, output_folder=output / "after")
    report = {"config": config, "training_history": history, "validation_before": before,
              "validation_after": after, "elapsed_seconds": time.perf_counter() - started,
              "note": "original COCO model versus a fine-tuned two-class model with a new head; tiny validation subset, not a test score"}
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved model and plain prediction images: {output}")


if __name__ == "__main__":
    main()
