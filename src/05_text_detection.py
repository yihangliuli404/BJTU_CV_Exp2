import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from config import TEXT_DATA_DIR, VIS_DIR
from common import get_images, ensure_dirs


def detect_text_regions_opencv(image_bgr):
    """
    基于 OpenCV 的文字区域候选检测。
    方法：
    1. 灰度化
    2. 自适应对比度增强
    3. Sobel 提取水平梯度
    4. 二值化
    5. 形态学闭运算连接文字笔画
    6. 轮廓筛选
    """
    h, w = image_bgr.shape[:2]

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_x = cv2.convertScaleAbs(grad_x)

    _, binary = cv2.threshold(
        grad_x,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel_w = max(15, w // 80)
    kernel_h = max(3, h // 250)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))

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

        if area < image_area * 0.0003:
            continue

        if area > image_area * 0.25:
            continue

        if bw < 15 or bh < 8:
            continue

        aspect = bw / max(bh, 1)

        if aspect < 1.2:
            continue

        if aspect > 30:
            continue

        boxes.append((x, y, bw, bh))

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    return boxes


def draw_boxes(image_bgr, boxes):
    out = image_bgr.copy()

    for i, (x, y, w, h) in enumerate(boxes, start=1):
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"text-{i}",
            (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2
        )

    return out


def process_image(image_path, out_path):
    image_bgr = cv2.imdecode(
        np.fromfile(str(image_path), dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    if image_bgr is None:
        return 0

    boxes = detect_text_regions_opencv(image_bgr)
    out = draw_boxes(image_bgr, boxes)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    ext = out_path.suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".bmp"]:
        out_path = out_path.with_suffix(".jpg")

    success, encoded = cv2.imencode(out_path.suffix, out)

    if success:
        encoded.tofile(str(out_path))

    return len(boxes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(TEXT_DATA_DIR))
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    ensure_dirs()

    input_path = Path(args.input)

    if input_path.is_file():
        images = [input_path]
    else:
        images = get_images(input_path)

    if not images:
        print("没有找到图片。请先把文字检测数据放入 dataset/data。")
        return

    if args.limit > 0:
        images = images[:args.limit]

    out_dir = VIS_DIR / "text_detection_single"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_boxes = 0

    for img_path in tqdm(images, desc="OpenCV文字区域检测"):
        out_path = out_dir / img_path.name
        num = process_image(img_path, out_path)
        total_boxes += num
        print(f"{img_path.name}: 检测到 {num} 个候选文字区域")

    print(f"文字检测可视化结果保存到: {out_dir}")
    print(f"累计检测候选文字区域数量: {total_boxes}")


if __name__ == "__main__":
    main()