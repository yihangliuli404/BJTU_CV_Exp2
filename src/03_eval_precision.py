import argparse
import pandas as pd

from config import RETRIEVAL_DIR, K_VALUES
from common import ensure_dirs


def precision_at_k(group, k):
    group = group.sort_values("rank").head(k)

    if len(group) == 0:
        return 0.0

    return float(group["is_relevant"].sum() / k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", default=str(RETRIEVAL_DIR / "retrieval_top60.csv"))
    args = parser.parse_args()

    ensure_dirs()

    retrieval_path = args.retrieval
    df = pd.read_csv(retrieval_path)

    query_rows = []

    for query_path, group in df.groupby("query_path"):
        group = group.sort_values("rank")

        row = {
            "query_path": query_path,
            "query_image": group["query_image"].iloc[0],
            "query_label": group["query_label"].iloc[0]
        }

        for k in K_VALUES:
            row[f"P@{k}"] = precision_at_k(group, k)

        query_rows.append(row)

    query_df = pd.DataFrame(query_rows)
    query_out = RETRIEVAL_DIR / "precision_by_query.csv"
    query_df.to_csv(query_out, index=False, encoding="utf-8-sig")

    class_rows = []

    for label, group in query_df.groupby("query_label"):
        row = {
            "landmark": label,
            "num_queries": len(group)
        }

        for k in K_VALUES:
            row[f"P@{k}"] = float(group[f"P@{k}"].mean())

        class_rows.append(row)

    class_df = pd.DataFrame(class_rows)
    class_out = RETRIEVAL_DIR / "precision_by_class.csv"
    class_df.to_csv(class_out, index=False, encoding="utf-8-sig")

    print("P@K 评价完成")
    print(f"按 query 保存: {query_out}")
    print(f"按 landmark 保存: {class_out}")
    print(class_df)


if __name__ == "__main__":
    main()
