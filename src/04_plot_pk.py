import pandas as pd
import matplotlib.pyplot as plt

from config import RETRIEVAL_DIR, FIGURE_DIR, K_VALUES
from common import ensure_dirs, safe_filename


def main():
    ensure_dirs()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    input_path = RETRIEVAL_DIR / "precision_by_class.csv"

    if not input_path.exists():
        print("没有找到 precision_by_class.csv。请先运行：python src/03_eval_precision.py")
        return

    df = pd.read_csv(input_path)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    for _, row in df.iterrows():
        landmark = row["landmark"]
        y_values = [row[f"P@{k}"] for k in K_VALUES]

        plt.figure(figsize=(6, 4))
        plt.plot(K_VALUES, y_values, marker="o")
        plt.ylim(0, 1.05)
        plt.xticks(K_VALUES)
        plt.xlabel("TopK")
        plt.ylabel("Precision")
        plt.title(f"Precision@K - {landmark}")
        plt.grid(True, linestyle="--", alpha=0.4)

        for x, y in zip(K_VALUES, y_values):
            plt.text(x, y + 0.03, f"{y:.2f}", ha="center")

        out_path = FIGURE_DIR / f"pk_{safe_filename(landmark)}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

    plt.figure(figsize=(9, 5))

    for _, row in df.iterrows():
        landmark = row["landmark"]
        y_values = [row[f"P@{k}"] for k in K_VALUES]
        plt.plot(K_VALUES, y_values, marker="o", label=str(landmark))

    plt.ylim(0, 1.05)
    plt.xticks(K_VALUES)
    plt.xlabel("TopK")
    plt.ylabel("Precision")
    plt.title("Precision@K Summary")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    summary_path = FIGURE_DIR / "pk_summary_all_landmarks.png"
    plt.savefig(summary_path, dpi=220)
    plt.close()

    print(f"P@K 曲线已保存到: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
