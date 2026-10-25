"""Draw dataset aircraft annotations and inspect a real detection batch."""

from pathlib import Path

from torch.utils.data import DataLoader

from aerial_detection.data import AircraftDataset, detection_collate
from aerial_detection.visualization import save_boxes

ROOT = Path(__file__).resolve().parents[1]


def main():
    dataset = AircraftDataset(ROOT / "data" / "sample" / "manifest.json")
    output_dir = ROOT / "outputs" / "inspection"
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(len(dataset)):
        image, target = dataset[index]
        image_id = target["image_id"].item()
        save_boxes(image, target["boxes"], output_dir / f"{image_id}_labels.png")
        print(f"Image {image_id}: image {tuple(image.shape)}, boxes {tuple(target['boxes'].shape)}, labels {target['labels'].tolist()}")
    loader = DataLoader(dataset, batch_size=2, collate_fn=detection_collate)
    for number, (images, targets) in enumerate(loader, start=1):
        print(f"Batch {number}: {len(images)} images, box counts {[len(t['boxes']) for t in targets]}")
    print(f"Saved annotated images: {output_dir}")


if __name__ == "__main__":
    main()
