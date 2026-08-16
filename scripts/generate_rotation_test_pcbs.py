"""Generate two synthetic PCB OCR test images with 100 components.

Outputs:
  C:\\PCB_Zone_OCR\\pcb_rotation_100_no_zones.png
  C:\\PCB_Zone_OCR\\pcb_rotation_100_with_zones.png
  C:\\PCB_Zone_OCR\\pcb_rotation_100_with_zones.json
  C:\\PCB_Zone_OCR\\pcb_rotation_100_ground_truth.csv

The two PNGs use the same component placement.
The only difference is that the "with_zones" image has red zone outlines.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"C:\PCB_Zone_OCR")
NO_ZONE_PNG = ROOT / "pcb_rotation_100_no_zones.png"
WITH_ZONE_PNG = ROOT / "pcb_rotation_100_with_zones.png"
ZONE_JSON = ROOT / "pcb_rotation_100_with_zones.json"
GT_CSV = ROOT / "pcb_rotation_100_ground_truth.csv"

WIDTH = 1400
HEIGHT = 900
BOARD = (45, 45, 1355, 855)
RANDOM_SEED = 20260816


def load_font(size: int):
    """Load a Windows font, falling back to default if unavailable."""

    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\consola.ttf",
    ]
    for font_path in candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT_SMALL = load_font(17)
FONT_MID = load_font(22)
FONT_ZONE = load_font(28)


def boxes_overlap(a, b, pad=8) -> bool:
    """Return True if two axis-aligned boxes overlap."""

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 + pad < bx1 or bx2 + pad < ax1 or ay2 + pad < by1 or by2 + pad < ay1)


def inside_board(x, y, w, h) -> bool:
    """Check whether a component box is inside the board outline."""

    x1, y1, x2, y2 = BOARD
    return x1 + 22 <= x and y1 + 22 <= y and x + w <= x2 - 22 and y + h <= y2 - 22


def make_rotated_text(text: str, angle: int, font, fill=(0, 0, 0)) -> Image.Image:
    """Create a transparent image containing rotated text.

    angle follows Pillow convention: counter-clockwise.
    A 360-degree text is visually the same as 0-degree text,
    but the ground-truth CSV still records 360.
    """

    normalized_angle = angle % 360
    dummy = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    dummy_draw = ImageDraw.Draw(dummy)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0] + 8
    h = bbox[3] - bbox[1] + 8

    text_img = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_img)
    text_draw.text((4, 4), text, font=font, fill=fill + (255,))

    if normalized_angle == 0:
        return text_img
    return text_img.rotate(normalized_angle, expand=True)


def paste_center(base: Image.Image, overlay: Image.Image, cx: float, cy: float):
    """Paste overlay image centered at (cx, cy)."""

    x = int(cx - overlay.width / 2)
    y = int(cy - overlay.height / 2)
    base.paste(overlay, (x, y), overlay)


def build_components():
    """Create 100 fake components with text angles 0/90/270/360."""

    random.seed(RANDOM_SEED)
    components = []
    occupied = []

    prefixes = [
        ("R", "RES", (64, 24), 30),
        ("C", "CAP", (54, 26), 28),
        ("U", "IC", (92, 62), 10),
        ("J", "CONN", (105, 48), 8),
        ("D", "DIODE", (58, 26), 8),
        ("L", "IND", (60, 28), 6),
        ("Q", "MOS", (58, 46), 5),
        ("TP", "TEST_POINT", (34, 34), 5),
    ]
    rotations = [0, 90, 270, 360] * 25
    random.shuffle(rotations)

    clusters = [
        (115, 90, 360, 280),
        (515, 90, 370, 285),
        (920, 90, 330, 300),
        (120, 430, 385, 300),
        (535, 430, 380, 310),
        (945, 425, 315, 315),
    ]

    counters = {}

    def next_ref(prefix):
        counters[prefix] = counters.get(prefix, 0) + 1
        return f"{prefix}{counters[prefix]}"

    flat_plan = []
    for prefix, kind, size, count in prefixes:
        for _ in range(count):
            flat_plan.append((prefix, kind, size))
    random.shuffle(flat_plan)

    for idx, (prefix, kind, size) in enumerate(flat_plan[:100]):
        w, h = size
        if random.random() < 0.22:
            w, h = h, w

        ref = next_ref(prefix)
        text_angle = rotations[idx]

        placed = False
        for _ in range(600):
            if random.random() < 0.78:
                cx, cy, cw, ch = random.choice(clusters)
                x = random.randint(cx, cx + cw)
                y = random.randint(cy, cy + ch)
            else:
                bx1, by1, bx2, by2 = BOARD
                x = random.randint(bx1 + 35, bx2 - 120)
                y = random.randint(by1 + 35, by2 - 90)

            box = (x, y, x + w, y + h)
            if not inside_board(x, y, w, h):
                continue
            if any(boxes_overlap(box, old) for old in occupied):
                continue

            occupied.append(box)
            components.append(
                {
                    "designator": ref,
                    "kind": kind,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "cx": x + w / 2,
                    "cy": y + h / 2,
                    "text_angle": text_angle,
                }
            )
            placed = True
            break

        if not placed:
            raise RuntimeError(f"Failed to place {ref}")

    return components


def draw_component(draw: ImageDraw.ImageDraw, comp: dict):
    """Draw one simple no-copper component symbol."""

    x = comp["x"]
    y = comp["y"]
    w = comp["w"]
    h = comp["h"]
    kind = comp["kind"]

    box = (x, y, x + w, y + h)

    if kind == "IC":
        draw.rectangle(box, outline=(45, 45, 45), width=3, fill=(248, 248, 248))
        pin_count = max(4, min(10, w // 10))
        for i in range(pin_count):
            px = x + 8 + i * max(7, (w - 16) // pin_count)
            draw.rectangle((px, y - 4, px + 4, y), fill=(55, 55, 55))
            draw.rectangle((px, y + h, px + 4, y + h + 4), fill=(55, 55, 55))
    elif kind == "CONN":
        draw.rectangle(box, outline=(45, 45, 45), width=3, fill=(250, 250, 250))
        for i in range(6):
            px = x + 10 + i * max(9, (w - 20) // 6)
            draw.rectangle((px, y + h - 5, px + 5, y + h + 6), fill=(55, 55, 55))
    elif kind == "TEST_POINT":
        draw.ellipse(box, outline=(45, 45, 45), width=3, fill=(255, 255, 255))
    else:
        draw.rounded_rectangle(box, radius=10, outline=(45, 45, 45), width=2, fill=(255, 255, 255))
        if kind in {"RES", "CAP", "DIODE", "IND"}:
            draw.rectangle((x, y + h * 0.35, x + 7, y + h * 0.65), fill=(85, 85, 85))
            draw.rectangle((x + w - 7, y + h * 0.35, x + w, y + h * 0.65), fill=(85, 85, 85))


def zone_polygons():
    """Return two irregular test zones as LabelMe-style point lists."""

    zone_01 = [
        [80, 85],
        [650, 85],
        [655, 375],
        [590, 410],
        [575, 560],
        [420, 575],
        [250, 520],
        [80, 535],
    ]
    zone_02 = [
        [690, 80],
        [1325, 80],
        [1325, 815],
        [735, 815],
        [730, 610],
        [650, 560],
        [665, 360],
        [690, 330],
    ]
    return zone_01, zone_02


def draw_board(components, with_zones: bool) -> Image.Image:
    """Draw one PCB placement image."""

    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle(BOARD, radius=28, outline=(35, 35, 35), width=4, fill=(252, 252, 249))

    for hx, hy in [(90, 90), (1310, 90), (90, 810), (1310, 810)]:
        draw.ellipse((hx - 22, hy - 22, hx + 22, hy + 22), outline=(55, 55, 55), width=3)

    for x in range(200, 1300, 200):
        draw.line((x, BOARD[1] + 10, x, BOARD[3] - 10), fill=(235, 235, 232), width=1)
    for y in range(180, 820, 160):
        draw.line((BOARD[0] + 10, y, BOARD[2] - 10, y), fill=(235, 235, 232), width=1)

    for comp in components:
        draw_component(draw, comp)

    # Draw text after bodies so labels stay visible.
    for comp in components:
        font = FONT_MID if comp["kind"] in {"IC", "CONN"} else FONT_SMALL
        text_img = make_rotated_text(comp["designator"], comp["text_angle"], font)
        paste_center(image, text_img, comp["cx"], comp["cy"])

    if with_zones:
        z1, z2 = zone_polygons()
        draw.line([tuple(p) for p in z1 + [z1[0]]], fill=(225, 0, 0), width=5)
        draw.line([tuple(p) for p in z2 + [z2[0]]], fill=(225, 0, 0), width=5)
        draw.text((95, 95), "Zone_01", fill=(225, 0, 0), font=FONT_ZONE)
        draw.text((710, 95), "Zone_02", fill=(225, 0, 0), font=FONT_ZONE)

    return image


def save_labelme_json():
    """Save LabelMe JSON for the zoned test image."""

    z1, z2 = zone_polygons()
    data = {
        "version": "7.0.4",
        "flags": {},
        "shapes": [
            {
                "label": "Zone_01",
                "points": z1,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
                "mask": None,
            },
            {
                "label": "Zone_02",
                "points": z2,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
                "mask": None,
            },
        ],
        "imagePath": WITH_ZONE_PNG.name,
        "imageData": None,
        "imageHeight": HEIGHT,
        "imageWidth": WIDTH,
    }
    ZONE_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_ground_truth(components):
    """Save component placement truth for test checking."""

    with GT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["designator", "kind", "x", "y", "w", "h", "cx", "cy", "text_angle"],
        )
        writer.writeheader()
        writer.writerows(components)


def main():
    components = build_components()
    draw_board(components, with_zones=False).save(NO_ZONE_PNG)
    draw_board(components, with_zones=True).save(WITH_ZONE_PNG)
    save_labelme_json()
    save_ground_truth(components)

    print(f"components={len(components)}")
    print(NO_ZONE_PNG)
    print(WITH_ZONE_PNG)
    print(ZONE_JSON)
    print(GT_CSV)


if __name__ == "__main__":
    main()
