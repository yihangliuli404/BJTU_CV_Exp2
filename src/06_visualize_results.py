import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from config import RETRIEVAL_DIR, VIS_DIR
from common import ensure_dirs, safe_filename


def load_font(size=16):
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]

    for font_path in candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size)

    return ImageFont.load_default()


def make_thumb(image, size=(300, 220)):
    image = image.copy()
    image.thumbnail(size)

    canvas = Image.new("RGB", size, "white")
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))

    return canvas


def detect_text_regions_on_thumb(pil_image):
    """
    在缩略图上检测文字候选区域。
    这样画出来的框不会因为缩放而变得太细。
    """
    image_rgb = np.array(pil_image.convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    h, w = image_bgr.shape[:2]

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_x = cv2.convertScaleAbs(grad_x)

    _, binary = cv2.threshold(
        grad_x,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    closed = cv2.dilate(
        closed,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1
    )

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    image_area = h * w

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh

        # 缩略图上的阈值要更宽松
        if area < image_area * 0.00015:
            continue

        if area > image_area * 0.35:
            continue

        if bw < 8 or bh < 5:
            continue

        aspect = bw / max(bh, 1)

        if aspect < 0.8:
            continue

        if aspect > 40:
            continue

        boxes.append((x, y, bw, bh))

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    return boxes


def draw_text_boxes_on_thumb(pil_image):
    """
    在缩略图上画粗绿色框。
    """
    image = pil_image.copy()
    draw = ImageDraw.Draw(image)
    boxes = detect_text_regions_on_thumb(image)

    for i, (x, y, w, h) in enumerate(boxes, start=1):
        # 粗绿色框
        draw.rectangle(
            [x, y, x + w, y + h],
            outline=(0, 255, 0),
            width=4
        )

        # 红色编号
        draw.text(
            (x, max(0, y - 18)),
            f"T{i}",
            fill=(255, 0, 0)
        )

    return image, len(boxes)


def build_case_image(query_path, rows, out_path, topn=5):
    font = load_font(16)

    items = [("Query", Path(query_path), True, 0.0)]

    for _, r in rows.head(topn).iterrows():
        tag = f"Rank {int(r['rank'])}"
        img_path = Path(r["base_path"])
        is_relevant = bool(int(r["is_relevant"]))
        sim = float(r["similarity"])
        items.append((tag, img_path, is_relevant, sim))

    tile_w = 300
    tile_h = 265
    image_h = 220
    margin = 12

    canvas_w = len(items) * tile_w + (len(items) + 1) * margin
    canvas_h = tile_h + 20

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    x = margin

    for tag, img_path, is_relevant, sim in items:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (tile_w, image_h), "white")
            d = ImageDraw.Draw(img)
            d.text((20, 80), "Load failed", fill="black")

        thumb = make_thumb(img, size=(tile_w, image_h))
        thumb, box_count = draw_text_boxes_on_thumb(thumb)

        canvas.paste(thumb, (x, 38))

        if tag == "Query":
            title = f"Query | Text boxes: {box_count}"
        else:
            mark = "OK" if is_relevant else "NO"
            title = f"{tag} | {mark} | {sim:.3f} | T:{box_count}"

        draw.text((x, 10), title, font=font, fill="black")

        x += tile_w + margin

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", default=str(RETRIEVAL_DIR / "retrieval_top60.csv"))
    parser.add_argument("--topn", type=int, default=5)
    parser.add_argument("--cases-per-class", type=int, default=2)
    args = parser.parse_args()

    ensure_dirs()

    retrieval_path = Path(args.retrieval)

    if not retrieval_path.exists():
        print("没有找到 retrieval_top60.csv。请先运行：python src/02_retrieve.py")
        return

    df = pd.read_csv(retrieval_path)

    out_dir = VIS_DIR / "retrieval_text_cases"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0

    for label, label_df in tqdm(df.groupby("query_label"), desc="生成检索-文字检测可视化"):
        query_paths = list(label_df["query_path"].drop_duplicates())[:args.cases_per_class]

        for index, query_path in enumerate(query_paths, start=1):
            rows = label_df[label_df["query_path"] == query_path].sort_values("rank")
            out_name = f"{safe_filename(label)}_case{index}.jpg"
            out_path = out_dir / out_name

            build_case_image(
                query_path=query_path,
                rows=rows,
                out_path=out_path,
                topn=args.topn
            )

            total += 1

    print(f"已生成 {total} 组检索-检测可视化结果")
    print(f"保存目录: {out_dir}")
    print("说明：绿色框表示 OpenCV 检测到的候选文字区域，T 表示候选框数量。")


if __name__ == "__main__":
    main()