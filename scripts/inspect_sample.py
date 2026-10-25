"""Draw dataset aircraft annotations and inspect a real detection batch."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from aerial_detection.data import AircraftDataset, detection_collate

ROOT = Path(__file__).resolve().parents[1]


def main():
    dataset = AircraftDataset(ROOT / "data" / "sample" / "manifest.json")
    output_dir = ROOT / "outputs" / "inspection"
    output_dir.mkdir(parents=True, exist_ok=True)
    panels = []
    for index in range(len(dataset)):
        image, target = dataset[index]
        photo = to_pil_image(image)
        draw = ImageDraw.Draw(photo)
        font = ImageFont.load_default(size=18)
        for box in target["boxes"].tolist():
            draw.rectangle(box, outline="#ffb000", width=3)
            if len(target["boxes"]) == 1:
                draw.text((box[0], max(0, box[1] - 22)), "aircraft (label)", fill="#ffb000", font=font, stroke_width=1, stroke_fill="black")
        image_id = target["image_id"].item()
        photo.save(output_dir / f"{image_id}_annotated.png")
        panel = Image.new("RGB", (420, 470), "#f7f9fc")
        photo.thumbnail((400, 400))
        panel.paste(photo, (10, 60))
        ImageDraw.Draw(panel).text((10, 12), f"Image {image_id} | {len(target['boxes'])} aircraft", fill="#17243b", font=font)
        panels.append(panel)
        print(f"Image {image_id}: image {tuple(image.shape)}, boxes {tuple(target['boxes'].shape)}, labels {target['labels'].tolist()}")
    sheet = Image.new("RGB", (420 * len(panels), 515), "#f7f9fc")
    for index, panel in enumerate(panels):
        sheet.paste(panel, (index * 420, 45))
    ImageDraw.Draw(sheet).text((10, 12), "DIOR sample: dataset annotations, NOT model predictions", fill="#17243b", font=ImageFont.load_default(size=22))
    sheet.save(output_dir / "contact_sheet.png")
    loader = DataLoader(dataset, batch_size=2, collate_fn=detection_collate)
    for number, (images, targets) in enumerate(loader, start=1):
        print(f"Batch {number}: {len(images)} images, box counts {[len(t['boxes']) for t in targets]}")
    print(f"Saved annotated images: {output_dir}")


if __name__ == "__main__":
    main()
