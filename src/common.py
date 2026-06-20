from pathlib import Path
import json
import re
import numpy as np

from config import IMAGE_EXTS, FEATURE_DIR, RETRIEVAL_DIR, FIGURE_DIR, VIS_DIR


def ensure_dirs():
    for folder in [FEATURE_DIR, RETRIEVAL_DIR, FIGURE_DIR, VIS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def get_images(folder: Path):
    if not folder.exists():
        return []

    images = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            images.append(p)

    return sorted(images)


def parse_label(path_or_name):
    """
    根据文件名前缀解析 landmark 类别。

    本数据集文件名常见形式：
    fhy-12k4jb1k421.jpg
    zx-xgfafd.jpg
    ty-1746580766456.jpg

    其中 '-' 前面的 fhy / zx / ty 才是类别前缀。
    """
    name = Path(path_or_name).name
    stem = Path(name).stem

    if "-" in stem:
        return stem.split("-")[0]

    if "_" in stem:
        return stem.split("_")[0]

    label = re.sub(r"\d+$", "", stem)
    if label == "":
        label = stem

    return label


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def l2_normalize(x: np.ndarray):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return x / norm


def safe_filename(name: str):
    name = str(name)
    name = re.sub(r'[\\/:*?"<>| ]+', "_", name)
    return name[:120]
