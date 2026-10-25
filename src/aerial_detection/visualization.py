"""Full-size images with boxes only: no resizing, panels, or text."""

from PIL import ImageDraw
from torchvision.transforms.functional import to_pil_image


def save_boxes(image, boxes, path):
    photo = to_pil_image(image)
    draw = ImageDraw.Draw(photo)
    for box in boxes.tolist():
        draw.rectangle(box, outline="#ffb000", width=2)
    photo.save(path)
