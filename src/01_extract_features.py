import argparse
import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm

from config import BASE_DIR, QUERY_DIR, FEATURE_DIR
from common import get_images, save_json, ensure_dirs, l2_normalize


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), "white")

        image = self.transform(image)
        return image, str(img_path)


def build_model(model_name, device):
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        transform = weights.transforms()
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        transform = weights.transforms()
    else:
        raise ValueError(f"不支持的模型: {model_name}")

    model.fc = nn.Identity()
    model.eval()
    model.to(device)

    return model, transform


@torch.no_grad()
def extract_features(image_paths, model, transform, device, batch_size):
    dataset = ImageDataset(image_paths, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    all_features = []
    all_paths = []

    for images, paths in tqdm(loader, desc="提取特征"):
        images = images.to(device)
        features = model(images)
        features = features.cpu().numpy().astype("float32")

        all_features.append(features)
        all_paths.extend(list(paths))

    if not all_features:
        return np.empty((0, 1), dtype="float32"), []

    features = np.vstack(all_features)
    features = l2_normalize(features).astype("float32")
    return features, all_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    ensure_dirs()
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"使用设备: {device}")
    print(f"使用模型: {args.model}")

    base_images = get_images(BASE_DIR)
    query_images = get_images(QUERY_DIR)

    print(f"base 图片数量: {len(base_images)}")
    print(f"query 图片数量: {len(query_images)}")

    if len(base_images) == 0 or len(query_images) == 0:
        print("base 或 query 为空。请先把数据集放入 dataset/base 和 dataset/query。")
        return

    model, transform = build_model(args.model, device)

    print("\n开始提取 base 特征")
    base_features, base_paths = extract_features(
        base_images,
        model,
        transform,
        device,
        args.batch_size
    )

    print("\n开始提取 query 特征")
    query_features, query_paths = extract_features(
        query_images,
        model,
        transform,
        device,
        args.batch_size
    )

    np.save(FEATURE_DIR / "base_features.npy", base_features)
    np.save(FEATURE_DIR / "query_features.npy", query_features)

    save_json(base_paths, FEATURE_DIR / "base_paths.json")
    save_json(query_paths, FEATURE_DIR / "query_paths.json")

    print("\n特征提取完成")
    print(f"base_features shape: {base_features.shape}")
    print(f"query_features shape: {query_features.shape}")
    print(f"保存目录: {FEATURE_DIR}")


if __name__ == "__main__":
    main()
