from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "dataset"
BASE_DIR = DATASET_DIR / "base"
QUERY_DIR = DATASET_DIR / "query"
TEXT_DATA_DIR = DATASET_DIR / "data"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEATURE_DIR = OUTPUT_DIR / "features"
RETRIEVAL_DIR = OUTPUT_DIR / "retrieval"
FIGURE_DIR = OUTPUT_DIR / "figures"
VIS_DIR = OUTPUT_DIR / "visualizations"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TOPK = 60
K_VALUES = [20, 40, 60]
