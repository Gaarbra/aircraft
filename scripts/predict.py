"""Load saved weights and draw aircraft boxes on one image.

Run: python scripts/predict.py --checkpoint outputs/training/first_run/last.pt --image data/sample/2003.jpg
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from aerial_detection.model import load_trained_model
from aerial_detection.visualization import save_boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)

    # 1. Load the model we trained, without downloading weights.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = load_trained_model(checkpoint)

    # 2. Load one image and scale its pixels to [0, 1].
    with Image.open(args.image) as source:
        image = pil_to_tensor(source.convert("RGB")).float() / 255.0

    # 3. Predict without calculating gradients or updating weights.
    with torch.inference_mode():
        prediction = model([image])[0]
    keep = (prediction["labels"] == 1) & (prediction["scores"] >= 0.5)

    # 4. Save just the image and boxes.
    destination = args.checkpoint.parent / "predictions"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{args.image.stem}_predictions.png"
    save_boxes(image, prediction["boxes"][keep], path)
    print(f"Found {int(keep.sum())} aircraft. Saved: {path}")


if __name__ == "__main__":
    main()
