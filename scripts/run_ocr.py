"""Run OCR on PCB layout images.

這支程式負責：
1. 讀取 pcb*.png 圖檔
2. 呼叫 OCR 引擎讀出文字與文字框座標
3. 輸出 results/pcb*_ocr.json 與 results/pcb*_ocr.csv

後面的 classify_by_zone.py 會讀這個 OCR 結果，再判斷每個元件文字屬於哪個 Zone。
"""

# argparse：用來讀命令列參數，例如 --pattern "pcb2.png"
import argparse

# json：用來輸出 OCR 結果成 .json
import json

# re：regular expression，正規表示式。用來判斷文字是否像元件代號。
import re

# Path：比純字串更好用的檔案路徑工具。
from pathlib import Path

# Image：Pillow 的圖片物件，用來把整張 layout 旋轉後再 OCR。
# ImageDraw：用來依 LabelMe polygon 建立 zone mask。
from PIL import Image, ImageDraw


# 這個正規表示式用來判斷 OCR 文字是否像 PCB 元件代號。
# 例：R1、C10、U3、IC5、J2、TP7、D15、L1、Q4、S2、RV1、Y1、F1
COMPONENT_RE = re.compile(r"^(R|C|U|IC|J|TP|D|L|Q|S|SW|RV|Y|F)\d+[A-Z]?$", re.IGNORECASE)


def polygon_to_bbox(poly):
    """把 OCR 的四點 polygon 轉成 bbox：[x1, y1, x2, y2]。

    案例 1：水平文字框
        poly = [[100, 50], [160, 50], [160, 80], [100, 80]]
        xs = [100, 160, 160, 100]
        ys = [50, 50, 80, 80]
        return [100, 50, 160, 80]

    案例 2：歪斜文字框
        poly = [[98, 52], [162, 48], [165, 82], [100, 85]]
        return [98, 48, 165, 85]

    注意：
        bbox 包住的是「OCR 文字」，不是元件本體。
    """

    # 取出所有 x 座標。
    xs = [float(p[0]) for p in poly]

    # 取出所有 y 座標。
    ys = [float(p[1]) for p in poly]

    # bbox 左上角是 min(x), min(y)，右下角是 max(x), max(y)。
    return [min(xs), min(ys), max(xs), max(ys)]


def parse_rotations(rotation_text):
    """把命令列字串 '0,90,180,270' 轉成整數 list。

    回傳例：
        [0, 90, 180, 270]

    案例：
        parse_rotations("0") -> [0]
        parse_rotations("0,90") -> [0, 90]
        parse_rotations("0,90,90") -> [0, 90]
        parse_rotations("360") -> [0]

    為什麼只支援 0/90/180/270：
        PCB layout 最常見是水平、垂直、倒轉文字。
        任意角度，例如 45 度，座標轉換與 OCR 去重會複雜很多。
    """

    # rotations 用來收集合法角度。
    rotations = []

    # 用逗號切開字串。
    for part in rotation_text.split(","):
        # 去掉前後空白。
        part = part.strip()

        # 空字串跳過。
        if not part:
            continue

        # 轉成整數，並限制在 0/90/180/270。
        angle = int(part) % 360
        if angle not in (0, 90, 180, 270):
            raise ValueError("Only 0, 90, 180, 270 rotations are supported.")

        # 避免重複角度。
        if angle not in rotations:
            rotations.append(angle)

    # 如果使用者傳空字串，就至少跑原圖 0 度。
    return rotations or [0]


def make_image_for_ocr(image_path, angle, scale, temp_dir):
    """依照 angle / scale 產生給 OCR 使用的圖片路徑。

    angle = 0 且 scale = 1 時直接回傳原圖。
    angle = 90/180/270 或 scale > 1 時，建立一張暫存圖。

    這裡的 angle 使用 Pillow 規則：逆時針旋轉。
    scale 是圖片放大倍率，例如 2 代表寬高都變 2 倍。

    案例：
        image_path = C:\\PCB_Zone_OCR\\pcb2.png
        temp_dir   = C:\\PCB_Zone_OCR\\results\\_rotated_temp

        angle = 0, scale = 1：
            return C:\\PCB_Zone_OCR\\pcb2.png

        angle = 0, scale = 2：
            產生暫存圖：
            C:\\PCB_Zone_OCR\\results\\_rotated_temp\\pcb2_scale2x_rot0.png

        angle = 90, scale = 2：
            產生暫存圖：
            C:\\PCB_Zone_OCR\\results\\_rotated_temp\\pcb2_scale2x_rot90.png
            return 這個暫存圖路徑

    重點：
        原始 pcb2.png 不會被修改。
        Zone 的 pcb2.json 也不會被修改。
        放大/旋轉圖只是給 OCR 暫時看。
    """

    # 不旋轉也不放大時，不用建立暫存圖。
    if angle == 0 and scale == 1:
        return image_path

    # 開啟圖片。
    image = Image.open(image_path)

    # 如果 scale > 1，就先放大圖片，讓 OCR 比較容易看清楚小字。
    if scale != 1:
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # rotate(angle, expand=True)：旋轉整張圖，並自動調整畫布大小。
    ocr_image = image.rotate(angle, expand=True)

    # 暫存檔名，例如 pcb2_scale2x_rot90.png。
    scale_text = f"{scale:g}".replace(".", "p")
    temp_path = temp_dir / f"{image_path.stem}_scale{scale_text}x_rot{angle}.png"

    # 存出暫存圖。
    ocr_image.save(temp_path)

    # 回傳暫存圖路徑。
    return temp_path


def load_labelme_zones(labelme_json_path):
    """讀取 LabelMe JSON 裡 label 以 Zone 開頭的區域。

    回傳格式：
        [
            {"name": "Zone_01", "points": [[x1, y1], [x2, y2], ...]},
            {"name": "Zone_02", "points": [[x1, y1], [x2, y2], ...]},
        ]

    注意：
        這裡只讀 Zone 座標，不讀元件。
        LabelMe JSON 本身仍然不是元件清單。
    """

    # 如果沒有 LabelMe JSON，就回傳空 list。
    if not labelme_json_path.exists():
        return []

    # 讀 JSON。
    data = json.loads(labelme_json_path.read_text(encoding="utf-8"))

    # zones 收集所有 Zone。
    zones = []

    # 逐一讀 LabelMe shapes。
    for shape in data.get("shapes", []):
        # 只處理 Zone 開頭的 label。
        label = shape.get("label", "")
        if not label.lower().startswith("zone"):
            continue

        # 取出 points。
        points = shape.get("points", [])

        # rectangle 只有兩點，要轉成四點。
        if shape.get("shape_type") == "rectangle" and len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        # polygon 至少三點。
        if len(points) < 3:
            continue

        # 座標轉成 float。
        zone_points = [[float(x), float(y)] for x, y in points]
        zones.append({"name": label, "points": zone_points})

    # 回傳 Zone 清單。
    return zones


def safe_name(text):
    """把 Zone 名稱轉成適合當檔名的一小段文字。"""

    # 只保留英數、底線、減號，其餘變成底線。
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text)


def create_zone_crop_images(image_path, zones, crop_dir, padding_px, mask_outside):
    """依 LabelMe Zone 自動裁切圖片。

    這個函式會產生暫存 crop 圖，並回傳每張 crop 對原圖的 offset。

    為什麼要記 offset：
        OCR 在 crop 圖上得到的是 crop 座標。
        後面要把座標加回 offset，才會回到原始 PNG 座標。

    mask_outside=False：
        用 Zone 的外接矩形加 padding 裁切。
        優點是不容易切斷靠邊界文字。

    mask_outside=True：
        外接矩形內只保留 polygon 內部，polygon 外變白。
        優點是干擾少；缺點是跨邊界文字可能被切掉。
    """

    # 建立 crop 輸出資料夾。
    crop_dir.mkdir(parents=True, exist_ok=True)

    # 開啟原圖。
    original = Image.open(image_path).convert("RGB")
    original_width, original_height = original.size

    # crop_targets 收集每張暫存 crop。
    crop_targets = []

    # 逐一處理 Zone。
    for zone in zones:
        points = zone["points"]

        # 計算 Zone 外接矩形。
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        # 加 padding，但不可超出原圖範圍。
        left = max(0, int(min(xs) - padding_px))
        top = max(0, int(min(ys) - padding_px))
        right = min(original_width, int(max(xs) + padding_px))
        bottom = min(original_height, int(max(ys) + padding_px))

        # 避免空 crop。
        if right <= left or bottom <= top:
            continue

        # 先做矩形裁切。
        crop = original.crop((left, top, right, bottom))

        # 如果要遮掉 polygon 外部，就建立 mask。
        if mask_outside:
            mask = Image.new("L", crop.size, 0)
            draw = ImageDraw.Draw(mask)
            shifted_points = [(x - left, y - top) for x, y in points]
            draw.polygon(shifted_points, fill=255)

            # 白底，只把 polygon 內部貼上。
            white = Image.new("RGB", crop.size, "white")
            white.paste(crop, (0, 0), mask)
            crop = white

        # 暫存檔名。
        zone_name = safe_name(zone["name"])
        crop_path = crop_dir / f"{image_path.stem}_{zone_name}_crop.png"

        # 存出 crop 圖。
        crop.save(crop_path)

        # 記錄 crop 對原圖的偏移量。
        crop_targets.append(
            {
                "path": crop_path,
                "offset_x": left,
                "offset_y": top,
                "width": crop.width,
                "height": crop.height,
                "zone": zone["name"],
            }
        )

    return crop_targets


def add_offset_to_items(items, offset_x, offset_y, source_zone_crop=""):
    """把 crop 座標加回原始圖片座標。"""

    # converted 收集轉換後項目。
    converted = []

    # 逐筆加 offset。
    for item in items:
        new_item = dict(item)
        new_poly = [[float(x) + offset_x, float(y) + offset_y] for x, y in item["polygon"]]
        new_item["polygon"] = new_poly
        new_item["bbox"] = polygon_to_bbox(new_poly)
        if source_zone_crop:
            new_item["source_zone_crop"] = source_zone_crop
        converted.append(new_item)

    return converted


def rotate_point_back(x, y, angle, original_width, original_height):
    """把旋轉圖上的一個點轉回原圖座標。

    因為 OCR 是在旋轉後的圖上找到文字框，所以座標也在旋轉圖座標系。
    後面 Zone 分類需要原圖座標，因此要轉回來。

    angle 代表圖片 OCR 前被逆時針旋轉幾度。

    假設原圖大小：
        original_width = 1000
        original_height = 600

    案例 1：angle = 0
        旋轉圖座標 (100, 50)
        回原圖還是 (100, 50)

    案例 2：angle = 90
        原圖點 (200, 100)
        圖片逆時針旋轉 90 度後，會變成旋轉圖點 (100, 800)
        OCR 看到的是 (100, 800)
        rotate_point_back(100, 800, 90, 1000, 600)
        會回傳 [200, 100]

    案例 3：angle = 180
        原圖點 (200, 100)
        旋轉圖點會是 (800, 500)
        rotate_point_back(800, 500, 180, 1000, 600)
        會回傳 [200, 100]

    案例 4：angle = 270
        原圖點 (200, 100)
        旋轉圖點會是 (500, 200)
        rotate_point_back(500, 200, 270, 1000, 600)
        會回傳 [200, 100]
    """

    # 0 度：座標不變。
    if angle == 0:
        return [x, y]

    # 90 度逆時針：
    # 原圖 (x0, y0) -> 旋轉圖 (y0, original_width - x0)
    # 反推：
    # x0 = original_width - y
    # y0 = x
    if angle == 90:
        return [original_width - y, x]

    # 180 度：
    # 原圖 (x0, y0) -> 旋轉圖 (original_width - x0, original_height - y0)
    if angle == 180:
        return [original_width - x, original_height - y]

    # 270 度逆時針：
    # 原圖 (x0, y0) -> 旋轉圖 (original_height - y0, x0)
    # 反推：
    # x0 = y
    # y0 = original_height - x
    if angle == 270:
        return [y, original_height - x]

    # 理論上不會跑到這裡，因為前面已限制角度。
    raise ValueError(f"Unsupported rotation: {angle}")


def rotate_items_back(items, angle, original_width, original_height, scale=1):
    """把某個角度 OCR 結果裡的 polygon/bbox 全部轉回原圖座標。

    案例：
        OCR 在 pcb2_rot90.png 讀到 R10：

        item = {
            "text": "R10",
            "polygon": [[100, 800], [130, 800], [130, 820], [100, 820]],
            "bbox": [100, 800, 130, 820],
            "score": 0.91,
            "is_component_like": True,
        }

        rotate_items_back([item], 90, 1000, 600, scale=1)

        會把 polygon 每個點都轉回原始 pcb2.png 座標，
        並重新計算 bbox，另外加上：

        "source_rotation": 90

    source_rotation 的用途：
        你可以在 CSV 裡看到某個文字是由哪個角度 OCR 讀到的。
    """

    # converted 用來收集轉換後結果。
    converted = []

    # 逐筆處理 OCR 結果。
    for item in items:
        # OCR 看到的是「先放大、再旋轉」後的圖片。
        # 所以要先用放大後的寬高把旋轉座標轉回去，
        # 再除以 scale，回到原始圖片座標。
        scaled_width = original_width * scale
        scaled_height = original_height * scale
        new_poly = []
        for x, y in item["polygon"]:
            scaled_x, scaled_y = rotate_point_back(float(x), float(y), angle, scaled_width, scaled_height)
            new_poly.append([scaled_x / scale, scaled_y / scale])

        # 複製一份 item，避免直接修改原物件。
        new_item = dict(item)

        # 更新 polygon。
        new_item["polygon"] = new_poly

        # bbox 要根據轉回來的 polygon 重新計算。
        new_item["bbox"] = polygon_to_bbox(new_poly)

        # 多記錄來源角度，方便 debug。
        new_item["source_rotation"] = angle

        # 存到 converted。
        converted.append(new_item)

    # 回傳轉換後結果。
    return converted


def bbox_iou(a, b):
    """計算兩個 bbox 的 IoU，用來合併重複 OCR 結果。

    IoU = intersection area / union area
    數值越高，代表兩個框越重疊。

    案例 1：完全相同
        a = [100, 50, 160, 80]
        b = [100, 50, 160, 80]
        IoU = 1.0

    案例 2：完全沒重疊
        a = [100, 50, 160, 80]
        b = [300, 50, 360, 80]
        IoU = 0.0

    案例 3：部分重疊
        a = [100, 50, 160, 80]
        b = [140, 50, 200, 80]
        只有中間一段重疊，IoU 會介於 0 和 1 之間。
    """

    # 拆出兩個 bbox。
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    # 重疊區域的左上與右下。
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    # 如果沒有重疊，寬高至少為 0。
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    # intersection 面積。
    inter = iw * ih

    # 各自 bbox 面積。
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    # union 面積。
    union = area_a + area_b - inter

    # 避免除以 0。
    if union <= 0:
        return 0.0

    # 回傳 IoU。
    return inter / union


def merge_duplicate_items(items, iou_threshold=0.35):
    """合併多角度 OCR 造成的重複文字。

    同一個文字如果在多個旋轉角度都被讀到，可能會重複出現。
    這裡用：
    - 文字相同
    - bbox 重疊夠高
    來判斷重複，保留 score 較高的那筆。

    案例：
        多角度 OCR 可能得到兩筆 R10：

        item A:
            text = "R10"
            bbox = [100, 50, 160, 80]
            score = 0.92
            source_rotation = 0

        item B:
            text = "R10"
            bbox = [102, 51, 161, 81]
            score = 0.80
            source_rotation = 90

        因為文字相同，而且 bbox 很重疊，
        所以只保留分數較高的 item A。

    另一個案例：
        R10 和 R11 即使位置很近，也不會合併，
        因為 text 不同。
    """

    # 先依照分數高低排序，讓高分結果優先留下。
    sorted_items = sorted(items, key=lambda item: float(item.get("score") or 0.0), reverse=True)

    # merged 存最後結果。
    merged = []

    # 逐筆嘗試加入。
    for item in sorted_items:
        # duplicate 用來記錄是否跟已保留結果重複。
        duplicate = False

        # 跟已保留結果比較。
        for kept in merged:
            # 文字不同，就不是同一個元件文字。
            if item["text"] != kept["text"]:
                continue

            # 文字相同且 bbox 很重疊，就當成重複。
            if bbox_iou(item["bbox"], kept["bbox"]) >= iou_threshold:
                duplicate = True
                break

        # 如果不是重複，就保留。
        if not duplicate:
            merged.append(item)

    # 為了輸出穩定，依照 y 再 x 排序。
    merged.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], item["text"]))

    # 回傳合併後結果。
    return merged


def normalize_rapidocr_result(result):
    """把 RapidOCR 輸出格式轉成我們統一使用的 list[dict]。"""

    # items 用來收集每一筆 OCR 文字。
    items = []

    # 如果 OCR 沒有結果，直接回傳空 list。
    if not result:
        return items

    # 逐筆處理 RapidOCR 的輸出。每筆通常是 [polygon, text, score]。
    for row in result:
        # 正常 row 至少有 polygon、text、score 三個欄位。
        if len(row) < 3:
            continue

        # 拆出三個欄位。
        poly, text, score = row[0], row[1], row[2]

        # 清理 OCR 文字。
        clean = str(text).strip().replace(" ", "")

        # OCR 常把數字 0 看成英文字 O。
        # 如果文字看起來像元件代號，就把 O 改成 0。
        looks_like_component_text = re.match(r"^(R|C|U|IC|J|TP|D|L|Q|S|SW|RV|Y|F)[A-Z0-9]+$", clean, re.I)
        if looks_like_component_text:
            clean = clean.replace("O", "0")

        # 空字串跳過。
        if not clean:
            continue

        # 如果格式像元件代號，統一轉大寫，避免 c3 / C3 被當成不同文字。
        if COMPONENT_RE.match(clean):
            clean = clean.upper()

        # 把 polygon 座標轉成 float。
        poly_list = [[float(x), float(y)] for x, y in poly]

        # append 一筆標準化結果。
        items.append(
            {
                "text": clean,
                "score": float(score),
                "polygon": poly_list,
                "bbox": polygon_to_bbox(poly_list),
                "is_component_like": bool(COMPONENT_RE.match(clean)),
            }
        )

    # 回傳所有 OCR 結果。
    return items


def main():
    """程式進入點。"""

    # 建立命令列參數解析器。
    parser = argparse.ArgumentParser(description="Run OCR on PCB PNG files.")

    # --image-dir：圖片所在資料夾。
    parser.add_argument("--image-dir", default=r"C:\PCB_Zone_OCR")

    # --out-dir：OCR 結果輸出資料夾。
    parser.add_argument("--out-dir", default=r"C:\PCB_Zone_OCR\results")

    # --pattern：要處理哪些圖片。預設 pcb*.png 代表 pcb1.png、pcb2.png...
    parser.add_argument("--pattern", default="pcb*.png")

    # --rotations：同一張圖要用哪些旋轉角度跑 OCR。
    # 預設 0,90,180,270，適合 PCB 上水平/垂直/倒著的元件文字。
    # 如果想跑快一點，可以指定 --rotations "0"。
    parser.add_argument("--rotations", default="0,90,180,270")

    # --scale：OCR 前先把圖片放大幾倍。
    # scale=1 代表不放大；scale=2 代表寬高都放大 2 倍。
    # 放大圖只是暫時給 OCR 看，輸出的座標仍會換回原圖座標。
    parser.add_argument("--scale", type=float, default=1.0)

    # --keep-all-ocr：預設只輸出像元件代號的 OCR 文字。
    # 如果想 debug OCR 原始雜字，可以加這個參數保留全部 OCR 文字。
    parser.add_argument("--keep-all-ocr", action="store_true")

    # --zone-crop：先依 LabelMe Zone 裁切，再對每張 crop OCR。
    # 需要同名 JSON，例如 pcb1.png 對 pcb1.json。
    parser.add_argument("--zone-crop", action="store_true")

    # --zone-padding-px：Zone crop 外接矩形往外多留幾個 pixel，避免邊界文字被切掉。
    parser.add_argument("--zone-padding-px", type=int, default=30)

    # --zone-mask-outside：啟用後，crop 內 polygon 外部會變白。
    # 預設不開，因為跨邊界文字可能被 mask 切斷。
    parser.add_argument("--zone-mask-outside", action="store_true")

    # 完整指令範例：
    # python .\scripts\run_ocr.py --image-dir "C:\PCB_Zone_OCR" --out-dir "C:\PCB_Zone_OCR\results" --pattern "pcb2.png" --rotations "0,90,180,270" --scale 3 --zone-crop

    # 讀取命令列參數。
    args = parser.parse_args()

    # 把 image-dir 轉成 Path 物件。
    image_dir = Path(args.image_dir)

    # 把 out-dir 轉成 Path 物件。
    out_dir = Path(args.out_dir)

    # 建立輸出資料夾。
    out_dir.mkdir(parents=True, exist_ok=True)

    # 解析旋轉角度設定。
    rotations = parse_rotations(args.rotations)

    # 檢查放大倍率。太小沒有意義，太大會很慢且吃記憶體。
    if args.scale < 1:
        raise SystemExit("--scale must be >= 1")

    # 暫存旋轉圖片的資料夾。
    temp_dir = out_dir / "_rotated_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 暫存 Zone crop 圖片的資料夾。
    crop_dir = out_dir / "_zone_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    # 建立 RapidOCR 物件。
    from rapidocr_onnxruntime import RapidOCR

    ocr = RapidOCR()

    # glob 依照 pattern 找圖片；sorted 讓順序固定。
    images = sorted(image_dir.glob(args.pattern))

    # 如果沒有找到任何圖片，直接結束並印錯誤。
    if not images:
        raise SystemExit(f"No images matched: {image_dir / args.pattern}")

    # 逐張圖片跑 OCR。
    for image_path in images:
        # image_path.name 只取檔名，例如 pcb1.png。
        print(f"OCR: {image_path.name}")

        # 開啟圖片，只是為了知道原圖寬高。
        with Image.open(image_path) as original_image:
            original_width, original_height = original_image.size

        # all_items 收集這張圖所有角度的 OCR 結果。
        all_items = []

        # ocr_targets 是這張圖要丟給 OCR 的圖片清單。
        # 一般模式：只有原始整張圖。
        # Zone crop 模式：會變成 Zone_01 crop、Zone_02 crop...
        if args.zone_crop:
            labelme_json = image_dir / f"{image_path.stem}.json"
            zones = load_labelme_zones(labelme_json)

            if not zones:
                print(f"  zone_crop=ON but no Zone found in {labelme_json.name}; fallback to full image")
                ocr_targets = [
                    {
                        "path": image_path,
                        "offset_x": 0,
                        "offset_y": 0,
                        "width": original_width,
                        "height": original_height,
                        "zone": "",
                    }
                ]
            else:
                ocr_targets = create_zone_crop_images(
                    image_path,
                    zones,
                    crop_dir,
                    args.zone_padding_px,
                    args.zone_mask_outside,
                )
                print(f"  zone_crop=ON zones={len(ocr_targets)} padding={args.zone_padding_px}")
        else:
            ocr_targets = [
                {
                    "path": image_path,
                    "offset_x": 0,
                    "offset_y": 0,
                    "width": original_width,
                    "height": original_height,
                    "zone": "",
                }
            ]

        # 同一張圖片可能有多個 OCR target。
        for target in ocr_targets:
            target_path = target["path"]
            target_width = target["width"]
            target_height = target["height"]
            target_zone = target.get("zone", "")

            if target_zone:
                print(f"  target={target_zone} size={target_width}x{target_height}")

            # 同一個 target 跑多個角度。
            for angle in rotations:
                print(f"  rotation={angle}")

                # 取得 OCR 要讀的圖片路徑。
                # scale=1 且 angle=0 時是原圖；其他情況會建立暫存放大/旋轉圖。
                ocr_image_path = make_image_for_ocr(target_path, angle, args.scale, temp_dir)

                # RapidOCR 用 ocr(image_path)，會回傳 result 和 elapsed time。
                result, _ = ocr(str(ocr_image_path))

                # 轉成統一格式。
                angle_items = normalize_rapidocr_result(result)

                # OCR 圖座標轉回 target 座標。
                # 如果有 --scale 2，這裡會把座標除回 target 尺寸。
                angle_items = rotate_items_back(angle_items, angle, target_width, target_height, args.scale)

                # 如果 target 是 crop，座標要再加 offset 才會回到原始 PNG。
                angle_items = add_offset_to_items(
                    angle_items,
                    target["offset_x"],
                    target["offset_y"],
                    target_zone,
                )

                # 累加這個角度的結果。
                all_items.extend(angle_items)

                print(f"    texts={len(angle_items)} component_like={sum(i['is_component_like'] for i in angle_items)}")

        # 多角度可能讀到重複文字，這裡合併。
        items = merge_duplicate_items(all_items)

        # 預設移除不合規 OCR，只保留像 R1、C10、U3、TP7 這類元件代號。
        # 注意：這只能移除「格式不像元件代號」的文字。
        # 像 C287 這種格式合規但可能不是真實元件，沒有 BOM/Altium 清單就無法自動判定真假。
        if not args.keep_all_ocr:
            items = [item for item in items if item["is_component_like"]]

        # 輸出 JSON 路徑，例如 results/pcb1_ocr.json。
        out_json = out_dir / f"{image_path.stem}_ocr.json"

        # 輸出 CSV 路徑，例如 results/pcb1_ocr.csv。
        out_csv = out_dir / f"{image_path.stem}_ocr.csv"

        # 把 items 寫成 JSON。
        out_json.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

        # 另外輸出一份簡單 CSV，方便 Excel / Notepad 查看。
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            # 寫 CSV header。
            f.write("text,score,is_component_like,source_rotation,source_zone_crop,x1,y1,x2,y2\n")

            # 逐筆寫入 OCR 結果。
            for item in items:
                # 從 bbox 拆出四個座標。
                x1, y1, x2, y2 = item["bbox"]

                # f-string：Python 的字串格式化。:.1f 代表小數點 1 位。
                f.write(
                    f'{item["text"]},{item["score"]},{item["is_component_like"]},{item.get("source_rotation", 0)},'
                    f'{item.get("source_zone_crop", "")},'
                    f"{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}\n"
                )

        # 印出統計資訊。
        component_count = sum(i["is_component_like"] for i in items)
        print(f"  merged_texts={len(items)} component_like={component_count}")
        print(f"  saved: {out_json}")


# 如果這個檔案是被直接執行，就呼叫 main()。
if __name__ == "__main__":
    main()
