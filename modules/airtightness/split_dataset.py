import os
import shutil
import random
from collections import defaultdict

random.seed(42)

BASE_DIR = "/root/autodl-tmp/ultralytics-main"
SRC1_ROOT = os.path.join(BASE_DIR, "dataset")
SRC2_ROOT = os.path.join(BASE_DIR, "total_dataset_train_val")
DST_ROOT = os.path.join(BASE_DIR, "dataset_split")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

VIDEO_PATTERNS = [
    r"^(\d{8})_",          # 20241101_frame_...
    r"^(example_\d+)",     # example_1_frame_...
    r"^(\d+)-frame",       # 001-frame... (total dataset)
]


def extract_video_id(filename):
    stem = os.path.splitext(filename)[0]
    for pat in VIDEO_PATTERNS:
        import re
        m = re.match(pat, stem)
        if m:
            return m.group(1)
    return f"_single_{stem}"


def collect_pairs(images_dir, labels_dir, ext=".jpg"):
    """收集有效的 image-label 配对（排除 _aug 增强副本）"""
    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        return []

    pairs = []
    for img_file in os.listdir(images_dir):
        if "_aug" in img_file:
            continue
        if not img_file.endswith(ext):
            continue
        img_stem = os.path.splitext(img_file)[0]
        label_file = img_stem + ".txt"
        label_path = os.path.join(labels_dir, label_file)
        img_path = os.path.join(images_dir, img_file)
        if os.path.exists(label_path):
            pairs.append((img_path, label_path))

    return pairs


def main():
    print("Step 1: 收集所有有效数据对（排除离线增强副本）")
    print("=" * 55)

    pairs1_train = collect_pairs(
        os.path.join(SRC1_ROOT, "train/images"),
        os.path.join(SRC1_ROOT, "train/labels"),
        ext=".jpg",
    )
    pairs1_val = collect_pairs(
        os.path.join(SRC1_ROOT, "val/images"),
        os.path.join(SRC1_ROOT, "val/labels"),
        ext=".jpg",
    )
    pairs2_train = collect_pairs(
        os.path.join(SRC2_ROOT, "images/train"),
        os.path.join(SRC2_ROOT, "labels/train"),
        ext=".png",
    )
    pairs2_val = collect_pairs(
        os.path.join(SRC2_ROOT, "images/val"),
        os.path.join(SRC2_ROOT, "labels/val"),
        ext=".png",
    )

    print(f"  dataset/ train:       {len(pairs1_train)} 对")
    print(f"  dataset/ val:         {len(pairs1_val)} 对")
    print(f"  total_dataset train:  {len(pairs2_train)} 对")
    print(f"  total_dataset val:    {len(pairs2_val)} 对")

    all_pairs = pairs1_train + pairs1_val + pairs2_train + pairs2_val
    print(f"\n  合并后总数据对: {len(all_pairs)}")

    print("\nStep 2: 按视频来源分组")
    print("=" * 55)

    video_groups = defaultdict(list)
    for img_path, label_path in all_pairs:
        vid = extract_video_id(os.path.basename(img_path))
        video_groups[vid].append((img_path, label_path))

    video_ids = sorted(video_groups.keys(), key=lambda v: len(video_groups[v]), reverse=True)
    total_frames = sum(len(v) for v in video_groups.values())

    print(f"  视频段数: {len(video_ids)}")
    print(f"  总帧数: {total_frames}")
    print(f"\n  前 10 个视频段:")
    for vid in video_ids[:10]:
        print(f"    {vid}: {len(video_groups[vid])} 帧")

    single_frame_count = sum(1 for vid in video_ids if vid.startswith("_single"))
    multi_frame_count = len(video_ids) - single_frame_count
    print(f"\n  多帧视频: {multi_frame_count} 段 | 单帧图像: {single_frame_count} 张")

    print("\nStep 3: 按 70/15/15 划分 train/val/test（贪心分配）")
    print("=" * 55)

    all_vids = list(video_ids)
    random.shuffle(all_vids)
    all_vids = sorted(all_vids, key=lambda v: len(video_groups[v]), reverse=True)

    splits = {"train": set(), "val": set(), "test": set()}
    counts = {"train": 0, "val": 0, "test": 0}
    targets = {
        "train": TRAIN_RATIO,
        "val": VAL_RATIO,
        "test": TEST_RATIO,
    }

    for vid in all_vids:
        n = len(video_groups[vid])
        best_split = min(["train", "val", "test"],
                         key=lambda s: (counts[s] + n) / max(total_frames, 1) - targets[s])
        splits[best_split].add(vid)
        counts[best_split] += n

    for split_name in ["train", "val", "test"]:
        pct = counts[split_name] / total_frames * 100
        print(f"  {split_name}: {counts[split_name]} 张 ({pct:.1f}%), {len(splits[split_name])} 个视频源")

    assert total == total_frames, f"Mismatch: {total} != {total_frames}"

    print("\nStep 4: 复制文件到新目录结构")
    print("=" * 55)

    if os.path.exists(DST_ROOT):
        ans = input(f"  {DST_ROOT} 已存在，覆盖? [y/N]: ")
        if ans.lower() != "y":
            print("  取消")
            return
        shutil.rmtree(DST_ROOT)

    for split_name in ["train", "val", "test"]:
        img_dir = os.path.join(DST_ROOT, split_name, "images")
        lbl_dir = os.path.join(DST_ROOT, split_name, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for vid in sorted(splits[split_name]):
            for img_src, lbl_src in video_groups[vid]:
                img_dst = os.path.join(img_dir, os.path.basename(img_src))
                lbl_dst = os.path.join(lbl_dir, os.path.basename(lbl_src))
                shutil.copy2(img_src, img_dst)
                shutil.copy2(lbl_src, lbl_dst)

        n_imgs = len(os.listdir(img_dir))
        n_lbls = len(os.listdir(lbl_dir))
        print(f"  {split_name}/images: {n_imgs} 张, {split_name}/labels: {n_lbls} 个")

    print(f"\n{'=' * 55}")
    print(f"完成! 新数据集目录: {DST_ROOT}")
    print(f"目录结构: {DST_ROOT}/{{train,val,test}}/{{images,labels}}/")
    print(f"\n论文应更新为:")
    print(f"  训练集: {sum(len(video_groups[vid]) for vid in splits['train'])} 张")
    print(f"  验证集: {sum(len(video_groups[vid]) for vid in splits['val'])} 张")
    print(f"  测试集: {sum(len(video_groups[vid]) for vid in splits['test'])} 张")
    print(f"  总计: {total_frames} 张")
    print(f"  划分比例: {TRAIN_RATIO:.0%} / {VAL_RATIO:.0%} / {TEST_RATIO:.0%}")
    print(f"  关键: 按视频源分组，同一视频的帧不会被拆分到不同集合")

    yaml_path = os.path.join(BASE_DIR, "bubble.yaml")
    if os.path.exists(yaml_path):
        print(f"\n  YAML 配置文件需更新 path 为: {DST_ROOT}")


if __name__ == "__main__":
    main()