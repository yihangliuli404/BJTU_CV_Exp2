import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import FEATURE_DIR, RETRIEVAL_DIR, TOPK
from common import load_json, parse_label, ensure_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=TOPK)
    args = parser.parse_args()

    ensure_dirs()
    RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)

    base_features_path = FEATURE_DIR / "base_features.npy"
    query_features_path = FEATURE_DIR / "query_features.npy"
    base_paths_path = FEATURE_DIR / "base_paths.json"
    query_paths_path = FEATURE_DIR / "query_paths.json"

    if not base_features_path.exists() or not query_features_path.exists():
        print("没有找到特征文件。请先运行：python src/01_extract_features.py")
        return

    base_features = np.load(base_features_path)
    query_features = np.load(query_features_path)
    base_paths = load_json(base_paths_path)
    query_paths = load_json(query_paths_path)

    print(f"base_features: {base_features.shape}")
    print(f"query_features: {query_features.shape}")

    topk = min(args.topk, len(base_paths))
    rows = []

    for qi in tqdm(range(len(query_paths)), desc="执行图像检索"):
        query_path = query_paths[qi]
        query_label = parse_label(query_path)

        scores = query_features[qi] @ base_features.T
        top_indices = np.argsort(-scores)[:topk]

        for rank, bi in enumerate(top_indices, start=1):
            base_path = base_paths[bi]
            base_label = parse_label(base_path)

            rows.append({
                "query_path": query_path,
                "query_image": Path(query_path).name,
                "query_label": query_label,
                "rank": rank,
                "base_path": base_path,
                "base_image": Path(base_path).name,
                "base_label": base_label,
                "similarity": float(scores[bi]),
                "is_relevant": int(base_label == query_label)
            })

    df = pd.DataFrame(rows)
    out_path = RETRIEVAL_DIR / f"retrieval_top{topk}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n检索完成: {out_path}")
    print(df.head(10))


if __name__ == "__main__":
    main()
