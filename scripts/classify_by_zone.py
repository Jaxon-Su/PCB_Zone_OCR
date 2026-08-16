"""Classify OCR component labels into LabelMe zones.

這支程式負責：
1. 讀取 LabelMe 產生的 pcb*.json
2. 讀取 run_ocr.py 產生的 results/pcb*_ocr.json
3. 計算每個 OCR 元件文字落在哪個 Zone
4. 輸出 results/pcb*_zone_components.csv

核心演算法：
文字框 polygon 跟 Zone polygon 重疊最多，就歸到那個 Zone。
如果靠近邊界、跨區、比例模糊，就標記 needs_review=true。
"""

# argparse：讀命令列參數，例如 --pattern "pcb2.png"
import argparse

# csv：輸出 CSV 檔案。
import csv

# json：讀 LabelMe JSON 與 OCR JSON。
import json

# Path：處理檔案路徑。
from pathlib import Path

# Image / ImageDraw：用來輸出人工確認用的 overlay 圖。
from PIL import Image, ImageDraw

# MultiPolygon / Polygon：Shapely 的多邊形物件，用來算 overlap、contains、distance。
from shapely.geometry import MultiPolygon, Polygon


def repair_polygon(poly):
    """嘗試修復無效 polygon。

    常見狀況：
        LabelMe 用 linestrip 畫區域時，
        起點/終點附近可能不小心交叉，造成 Self-intersection。

    Shapely 常用修復方式：
        poly.buffer(0)

    如果修復後變成 MultiPolygon，
    代表被切成多塊，這裡取面積最大的那一塊。
    """

    # 如果原本有效，直接回傳。
    if poly.is_valid and poly.area > 0:
        return poly

    # buffer(0) 是 Shapely 常見的幾何修復技巧。
    fixed = poly.buffer(0)

    # 如果修復後是一個有效 Polygon，就使用它。
    if isinstance(fixed, Polygon) and fixed.is_valid and fixed.area > 0:
        return fixed

    # 如果修復後是 MultiPolygon，取面積最大的 polygon。
    if isinstance(fixed, MultiPolygon):
        polygons = [p for p in fixed.geoms if p.is_valid and p.area > 0]
        if polygons:
            return max(polygons, key=lambda p: p.area)

    # 修不好就回傳 None。
    return None


def load_zones(labelme_json_path):
    """讀取 LabelMe JSON，取出 label 以 Zone 開頭的區域。

    支援：
    - rectangle：LabelMe 矩形工具
    - polygon：LabelMe 多邊形工具

    案例 1：LabelMe rectangle
        JSON 裡可能長這樣：

        {
            "label": "Zone_01",
            "shape_type": "rectangle",
            "points": [[10, 20], [300, 200]]
        }

        rectangle 只有兩個點，所以程式會轉成四點：

        [[10, 20], [300, 20], [300, 200], [10, 200]]

    案例 2：LabelMe polygon
        JSON 裡可能長這樣：

        {
            "label": "Zone_02",
            "shape_type": "polygon",
            "points": [[400, 20], [800, 50], [750, 300], [380, 260]]
        }

        polygon 已經是多點區域，所以直接拿來建立 Polygon。

    注意：
        label 必須以 Zone 開頭，例如 Zone_01、Zone_02、Zone_A。
        Example Zone A 這種如果沒有被 LabelMe 標成 label，就只是圖片文字，不算 Zone。
    """

    # Path(...).read_text()：讀取整個文字檔。
    # json.loads(...)：把 JSON 字串轉成 Python dict/list。
    data = json.loads(Path(labelme_json_path).read_text(encoding="utf-8"))

    # zones 用來收集所有 Zone。
    zones = []

    # LabelMe 的所有圖形都存在 data["shapes"]。
    # data.get("shapes", [])：如果沒有 shapes，就用空 list。
    for shape in data.get("shapes", []):
        # label 是你在 LabelMe 輸入的名字，例如 Zone_01。
        label = shape.get("label", "")

        # points 是 LabelMe 儲存的點座標。
        points = shape.get("points", [])

        # 只處理 label 以 Zone 開頭的區域。
        # 其他標記忽略。
        if not label.lower().startswith("zone"):
            continue

        # shape_type 可能是 rectangle 或 polygon。
        shape_type = shape.get("shape_type", "")

        # LabelMe rectangle 只存兩個點：左上與右下。
        if shape_type == "rectangle" and len(points) == 2:
            # 拆出兩個角。
            x1, y1 = points[0]
            x2, y2 = points[1]

            # 把矩形轉成四點 polygon。
            points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        # polygon 至少需要三個點才是一個面。
        elif len(points) < 3:
            continue

        # 用 Shapely 建立 polygon。
        poly = Polygon(points)

        # 嘗試修復無效 polygon。
        poly = repair_polygon(poly)

        # 修復後仍無效，就不能拿來分區。
        if poly is None:
            continue

        # 存下 Zone 名稱與 polygon。
        zones.append({"name": label, "polygon": poly})

    # 回傳所有 Zone。
    return zones


def classify_item(item, zones, near_boundary_px):
    """判斷一個 OCR 元件文字屬於哪個 Zone。

    item：一筆 OCR 結果，例如 R10 的文字框。
    zones：LabelMe 畫出的所有 Zone。
    near_boundary_px：距離邊界小於這個 pixel，就標記 needs_review。

    案例 1：文字完全在 Zone_01
        R10 的文字框面積 = 100
        跟 Zone_01 重疊面積 = 100
        跟 Zone_02 重疊面積 = 0

        overlap_ratio:
            Zone_01 = 1.00
            Zone_02 = 0.00

        結果：
            zone = Zone_01
            confidence = 1.000
            reason = bbox_mostly_inside
            needs_review = false

    案例 2：文字跨在 Zone_01 / Zone_02 分界上
        C77 的文字框面積 = 100
        跟 Zone_01 重疊面積 = 52
        跟 Zone_02 重疊面積 = 48

        overlap_ratio:
            Zone_01 = 0.52
            Zone_02 = 0.48

        因為兩邊很接近，結果會標成：
            reason = ambiguous_between_zones
            needs_review = true

    案例 3：文字中心太靠近邊界
        R5 大部分在 Zone_02，但中心點距離邊界只有 5 px。
        如果 near_boundary_px = 12，
        則：
            reason = near_boundary
            needs_review = true

    重點：
        這裡分類的是「OCR 文字框」落在哪個 Zone，
        不是保證「實際元件本體」落在哪個 Zone。
    """

    # OCR 結果裡的 polygon 是文字框四點。
    text_poly = Polygon(item["polygon"])

    # 如果 OCR polygon 無效，就改用 bbox 建立矩形 polygon。
    if not text_poly.is_valid or text_poly.area <= 0:
        # bbox 格式是 [x1, y1, x2, y2]。
        x1, y1, x2, y2 = item["bbox"]

        # 用 bbox 四個角建立 polygon。
        text_poly = Polygon([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

    # centroid 是文字框中心點。
    center = text_poly.centroid

    # overlaps 用來存「這個文字框跟每個 Zone 的關係」。
    overlaps = []

    # 逐一計算文字框跟每個 Zone 的 overlap。
    for zone in zones:
        # 取出 Zone polygon。
        zone_poly = zone["polygon"]

        # intersection 是文字框與 Zone 重疊的形狀。
        # .area 是重疊面積。
        overlap_area = text_poly.intersection(zone_poly).area

        # overlap_ratio = 重疊面積 / 文字框面積。
        # 例：0.95 代表文字框 95% 在這個 Zone 裡。
        overlap_ratio = overlap_area / text_poly.area if text_poly.area else 0.0

        # contains：中心點是否在 Zone 裡。
        # touches：中心點是否剛好碰到 Zone 邊界。
        center_inside = zone_poly.contains(center) or zone_poly.touches(center)

        # boundary_distance：文字中心點到 Zone 邊界的距離。
        boundary_distance = center.distance(zone_poly.boundary)

        # 把計算結果存起來。
        overlaps.append(
            {
                "zone": zone["name"],
                "overlap_ratio": overlap_ratio,
                "center_inside": center_inside,
                "boundary_distance": boundary_distance,
            }
        )

    # 依照 overlap_ratio 排序，最大重疊的 Zone 放第一個。
    # 如果 overlap_ratio 一樣，center_inside=True 的優先。
    overlaps.sort(key=lambda x: (x["overlap_ratio"], x["center_inside"]), reverse=True)

    # 如果完全沒有 Zone，回傳 NO_ZONE。
    if not overlaps:
        return "NO_ZONE", 0.0, "no_zone_defined", True

    # best 是最可能的 Zone。
    best = overlaps[0]

    # second 是第二可能的 Zone。若只有一個 Zone，就設為 None。
    second = overlaps[1] if len(overlaps) > 1 else None

    # 預設不需要人工確認。
    needs_review = False

    # reason 用來說明為什麼分到這個 Zone。
    reason = "bbox_overlap"

    # confidence 先用最大 overlap_ratio。
    confidence = best["overlap_ratio"]

    # 如果 80% 以上在同一區，通常很明確。
    if best["overlap_ratio"] >= 0.80:
        reason = "bbox_mostly_inside"

    # 如果 55%~80% 在同一區，能先分，但建議人工確認。
    elif best["overlap_ratio"] >= 0.55:
        reason = "bbox_partial_overlap"
        needs_review = True

    # 如果 overlap 不高，但中心點在 Zone 內，也先分，但要人工確認。
    elif best["center_inside"]:
        reason = "center_inside_only"
        confidence = 0.50
        needs_review = True

    # 如果連中心點也不在任何 Zone 內，就視為不明確。
    else:
        reason = "outside_all_zones"
        confidence = 0.0
        needs_review = True

    # 如果第一名與第二名 overlap 很接近，代表跨區或模糊。
    if second and abs(best["overlap_ratio"] - second["overlap_ratio"]) < 0.20:
        reason = "ambiguous_between_zones"
        needs_review = True

    # 如果文字中心點太靠近邊界，也要人工確認。
    if best["center_inside"] and best["boundary_distance"] <= near_boundary_px:
        reason = "near_boundary"
        needs_review = True

    # 回傳：Zone 名稱、信心值、原因、是否需要人工確認。
    return best["zone"], confidence, reason, needs_review


def write_rows_csv(path, rows):
    """把 list[dict] 寫成 CSV。"""

    # 欄位順序固定，方便 Excel / Notepad 查看。
    fieldnames = [
        "designator",
        "status",
        "zone",
        "confidence",
        "reason",
        "needs_review",
        "review_detail",
        "ocr_score",
        "x1",
        "y1",
        "x2",
        "y2",
    ]

    # 寫入 CSV。
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def draw_overlay(image_path, zones, rows, out_path):
    """輸出人工檢查用 overlay 圖。

    顏色規則：
        綠色：auto pass
        黃色：needs_review
        紅色：NO_ZONE 或疑似錯誤
    """

    # 如果原圖不存在，就不畫圖。
    if not image_path.exists():
        return

    # 轉成 RGB，避免 PNG palette / alpha 造成畫線問題。
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # 先畫 Zone 邊界。
    zone_colors = ["red", "green", "blue", "purple", "orange"]
    for index, zone in enumerate(zones):
        color = zone_colors[index % len(zone_colors)]
        points = [(float(x), float(y)) for x, y in zone["polygon"].exterior.coords]
        draw.line(points, fill=color, width=3)
        if points:
            draw.text(points[0], zone["name"], fill=color)

    # 再畫 OCR bbox。
    for row in rows:
        x1 = float(row["x1"])
        y1 = float(row["y1"])
        x2 = float(row["x2"])
        y2 = float(row["y2"])

        if row["zone"] == "NO_ZONE":
            color = "red"
        elif row["needs_review"] == "true":
            color = "orange"
        else:
            color = "lime"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, max(0, y1 - 12)), row["designator"], fill=color)

    image.save(out_path)


def process_one(image_stem, root_dir, results_dir, near_boundary_px, min_ocr_score, draw_check_image):
    """處理單一圖片，例如 pcb2.png。

    image_stem：不含副檔名的檔名，例如 pcb2。
    root_dir：圖片與 LabelMe JSON 所在資料夾。
    results_dir：OCR 結果與分類結果所在資料夾。

    案例：
        image_stem = "pcb2"
        root_dir = C:\\PCB_Zone_OCR
        results_dir = C:\\PCB_Zone_OCR\\results

        程式會找：
            C:\\PCB_Zone_OCR\\pcb2.json
            C:\\PCB_Zone_OCR\\results\\pcb2_ocr.json

        成功後會輸出：
            C:\\PCB_Zone_OCR\\results\\pcb2_zone_components.csv

    如果缺 pcb2.json：
        代表還沒用 LabelMe 畫 Zone 或沒保存。

    如果缺 pcb2_ocr.json：
        代表還沒跑 run_ocr.py。
    """

    # LabelMe JSON 路徑，例如 C:\PCB_Zone_OCR\pcb2.json。
    labelme_json = root_dir / f"{image_stem}.json"

    # OCR JSON 路徑，例如 C:\PCB_Zone_OCR\results\pcb2_ocr.json。
    ocr_json = results_dir / f"{image_stem}_ocr.json"

    # 如果沒有 LabelMe JSON，跳過。
    if not labelme_json.exists():
        print(f"SKIP {image_stem}: missing LabelMe zone file: {labelme_json.name}")
        return

    # 如果沒有 OCR JSON，跳過。
    if not ocr_json.exists():
        print(f"SKIP {image_stem}: missing OCR file: {ocr_json.name}")
        return

    # 讀取 Zone。
    zones = load_zones(labelme_json)

    # 如果沒有任何 Zone，跳過。
    if not zones:
        print(f"SKIP {image_stem}: no labels named Zone... in {labelme_json.name}")
        return

    # 讀取 OCR JSON。
    items = json.loads(ocr_json.read_text(encoding="utf-8"))

    # 只保留像元件代號的文字，例如 R1、C2、IC3。
    items = [item for item in items if item.get("is_component_like")]

    # 計算重複辨識。相同 designator 出現多次，一律進人工確認。
    designator_counts = {}
    for item in items:
        designator = str(item["text"]).strip().upper()
        designator_counts[designator] = designator_counts.get(designator, 0) + 1

    # rows 收集完整分類結果。
    rows = []

    # 逐筆分類 OCR 元件文字。
    for item in items:
        # 呼叫核心分類函式。
        zone, confidence, reason, needs_review = classify_item(item, zones, near_boundary_px)

        # designator 統一大寫，避免 r1 / R1 被當成不同元件。
        designator = str(item["text"]).strip().upper()

        # review_reasons 收集所有需要人工確認的原因。
        review_reasons = []

        # classify_item 已判斷邊界、跨區、低 overlap 等問題。
        if needs_review:
            review_reasons.append(reason)

        # OCR score 太低，人工確認。
        score = item.get("score")
        if score is not None and float(score) < min_ocr_score:
            review_reasons.append("low_ocr_score")

        # 同一顆元件被 OCR 抓到多次，人工確認。
        if designator_counts.get(designator, 0) > 1:
            review_reasons.append("duplicate_designator")

        # 只要 review_reasons 非空，就需要人工確認。
        final_needs_review = bool(review_reasons)

        # 取出 bbox 四個座標。
        x1, y1, x2, y2 = item["bbox"]

        # 收集成一列資料。
        rows.append(
            {
                "designator": designator,
                "status": "REVIEW" if final_needs_review else "AUTO_PASS",
                "zone": zone,
                "confidence": f"{confidence:.3f}",
                "reason": reason,
                "needs_review": str(final_needs_review).lower(),
                "review_detail": "|".join(review_reasons),
                "ocr_score": item.get("score"),
                "x1": f"{x1:.1f}",
                "y1": f"{y1:.1f}",
                "x2": f"{x2:.1f}",
                "y2": f"{y2:.1f}",
            }
        )

    # 輸出 CSV 路徑。
    out_csv = results_dir / f"{image_stem}_zone_components.csv"
    overlay_png = results_dir / f"{image_stem}_overlay_check.png"

    # 全部結果都放在同一份 CSV，方便 Excel 閱讀。
    # status=AUTO_PASS 代表可先採用；status=REVIEW 代表需要人工確認。
    write_rows_csv(out_csv, rows)

    # 統計高信任與需確認的數量，但不另外輸出多份 CSV。
    auto_rows = [row for row in rows if row["needs_review"] == "false"]
    review_rows = [row for row in rows if row["needs_review"] == "true"]

    # 視覺檢查圖。
    if draw_check_image:
        draw_overlay(root_dir / f"{image_stem}.png", zones, rows, overlay_png)

    # 印出完成訊息。
    print(
        f"OK {image_stem}: zones={len(zones)} components={len(items)} "
        f"auto={len(auto_rows)} review={len(review_rows)} -> {out_csv.name}"
    )


def main():
    """程式進入點。"""

    # 建立命令列參數解析器。
    parser = argparse.ArgumentParser(description="Classify OCR component labels into LabelMe zones.")

    # --root-dir：圖片與 LabelMe JSON 所在資料夾。
    parser.add_argument("--root-dir", default=r"C:\PCB_Zone_OCR")

    # --results-dir：OCR JSON 與分類 CSV 所在資料夾。
    parser.add_argument("--results-dir", default=r"C:\PCB_Zone_OCR\results")

    # --pattern：要處理哪些圖片。
    parser.add_argument("--pattern", default="pcb*.png")

    # --near-boundary-px：文字中心距離邊界多少 pixel 內要人工確認。
    parser.add_argument("--near-boundary-px", type=float, default=12.0)

    # --min-ocr-score：OCR 分數低於此值，就進 review_needed.csv。
    # RapidOCR 在 PCB 小字上分數常偏低，所以預設用 0.50。
    # 如果你想更保守，可以命令列指定 --min-ocr-score 0.65。
    parser.add_argument("--min-ocr-score", type=float, default=0.50)

    # --no-overlay：如果只想輸出 CSV，不想產生檢查圖，可以加這個參數。
    parser.add_argument("--no-overlay", action="store_true")

    # 讀取命令列參數。
    args = parser.parse_args()

    # root_dir 轉成 Path。
    root_dir = Path(args.root_dir)

    # results_dir 轉成 Path。
    results_dir = Path(args.results_dir)

    # 建立 results 資料夾。
    results_dir.mkdir(parents=True, exist_ok=True)

    # 依照 pattern 找圖片。
    for image_path in sorted(root_dir.glob(args.pattern)):
        # image_path.stem 是不含副檔名的檔名，例如 pcb2。
        process_one(
            image_path.stem,
            root_dir,
            results_dir,
            args.near_boundary_px,
            args.min_ocr_score,
            not args.no_overlay,
        )


# 如果這個檔案是被直接執行，就呼叫 main()。
if __name__ == "__main__":
    main()
