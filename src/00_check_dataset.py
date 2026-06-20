from pathlib import Path
from PIL import Image
from collections import Counter

from config import PROJECT_ROOT, BASE_DIR, QUERY_DIR, TEXT_DATA_DIR
from common import get_images, parse_label


def check_invalid_images(image_paths):
    invalid = []

    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            invalid.append((str(img_path), str(e)))

    return invalid


def show_folder_info(title, folder):
    images = get_images(folder)
    labels = [parse_label(p.name) for p in images]
    counter = Counter(labels)
    invalid = check_invalid_images(images)

    print("=" * 70)
    print(title)
    print(f"路径: {folder}")
    print(f"图片数量: {len(images)}")
    print(f"类别/前缀数量: {len(counter)}")
    print(f"损坏图片数量: {len(invalid)}")

    if counter:
        print("\n前20个文件名前缀统计:")
        for label, count in counter.most_common(20):
            print(f"{label}: {count}")

    if invalid:
        print("\n损坏图片示例:")
        for path, err in invalid[:10]:
            print(path, err)

    return images, counter, invalid


def main():
    print("BJTU_CV_Exp2 数据集检查")
    print(f"项目根目录: {PROJECT_ROOT}")

    base_images, base_counter, base_invalid = show_folder_info("base 图像检索数据库", BASE_DIR)
    query_images, query_counter, query_invalid = show_folder_info("query 查询图片", QUERY_DIR)
    data_images, data_counter, data_invalid = show_folder_info("data 文字检测数据", TEXT_DATA_DIR)

    print("=" * 70)
    print("总体检查结果")
    print(f"base 图片数量: {len(base_images)}")
    print(f"query 图片数量: {len(query_images)}")
    print(f"data 图片数量: {len(data_images)}")
    print(f"总损坏图片数量: {len(base_invalid) + len(query_invalid) + len(data_invalid)}")

    print("\n参考数量:")
    print("base 应接近 7728 张")
    print("query 应接近 135 张")
    print("data 应接近 1494 张")

    if len(base_images) == 0 or len(query_images) == 0:
        print("\n提示：当前 base 或 query 为空。等数据集下载并复制进去后，再重新运行本脚本。")


if __name__ == "__main__":
    main()
