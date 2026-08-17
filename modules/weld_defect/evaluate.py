import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ================= 类别映射 =================
JSON2YOLO = {
    "undercut": 0,
    "slaginclusion": 1,
    "porosity": 2,
    "crack": 3,
    "overlap": 4,
}

YOLO_CLASSES = ["undercut", "slaginclusion", "porosity", "crack", "overlap"]
NUM_CLASSES = len(YOLO_CLASSES)
BG_INDEX = NUM_CLASSES  # 背景/漏检列索引


def iou_xyxy(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter)

def yolo_to_xyxy(line, img_w, img_h):
    cls, cx, cy, w, h = map(float, line.split())
    cx *= img_w
    cy *= img_h
    w *= img_w
    h *= img_h
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return int(cls), [x1, y1, x2, y2]

# 主评估函数
def evaluate(json_dir, yolo_label_dir, cm_save_dir):
    cm_save_dir = Path(cm_save_dir)
    cm_save_dir.mkdir(parents=True, exist_ok=True)

    total_gt = 0
    detected_gt = 0

    total_det = 0
    wrong_det = 0

    # 混淆矩阵：行 = GT，列 = Pred + 背景列
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES+1), dtype=int)

    json_files = list(Path(json_dir).glob("*_result.json"))

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_w, img_h = data["image_size"]
        detections = data.get("detections", [])

        label_path = Path(yolo_label_dir) / (jf.stem.replace("_result", "") + ".txt")
        if not label_path.exists():
            continue

        # 读取 GT
        gt_boxes = []
        for line in open(label_path, "r"):
            cls, box = yolo_to_xyxy(line, img_w, img_h)
            gt_boxes.append((cls, box))

        # 检出率
        for gt_cls, gt_box in gt_boxes:
            total_gt += 1
            if any(iou_xyxy(gt_box, det["bbox_xyxy"]) > 0 for det in detections):
                detected_gt += 1

        # 误检率
        for det in detections:
            det_box = det["bbox_xyxy"]
            det_cls = JSON2YOLO.get(det["category"], -1)

            max_iou = 0.0
            matched_gt_cls = None

            for gt_cls, gt_box in gt_boxes:
                iou = iou_xyxy(det_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    matched_gt_cls = gt_cls

            total_det += 1
            if max_iou == 0 or det_cls != matched_gt_cls:
                wrong_det += 1

        #混淆矩阵（GT 视角，背景列表示漏检）
        for gt_cls, gt_box in gt_boxes:
            max_iou = 0.0
            matched_det_cls = BG_INDEX  # 默认漏检

            for det in detections:
                det_box = det["bbox_xyxy"]
                det_cls = JSON2YOLO.get(det["category"], -1)
                iou = iou_xyxy(gt_box, det_box)
                if iou > max_iou:
                    max_iou = iou
                    matched_det_cls = det_cls

            confusion_matrix[gt_cls, matched_det_cls] += 1

    detection_rate = detected_gt / total_gt if total_gt > 0 else 0
    false_rate = wrong_det / total_det if total_det > 0 else 0

    # 保存混淆矩阵图
    save_confusion_matrix(confusion_matrix, cm_save_dir / "confusion_matrix.png")

    return detection_rate*100, false_rate*100, confusion_matrix

# 混淆矩阵可视化
def save_confusion_matrix(cm, save_path):
    classes_with_bg = YOLO_CLASSES + ["BG"]
    plt.figure(figsize=(10, 6))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(classes_with_bg)), classes_with_bg, rotation=45)
    plt.yticks(range(len(YOLO_CLASSES)), YOLO_CLASSES)
    plt.xlabel("Predicted Class (JSON)")
    plt.ylabel("True Class (YOLO GT)")
    plt.title("Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# 主入口
if __name__ == "__main__":
    json_dir = r"F:\weldtoolv2.1\outputs"
    yolo_label_dir = r"D:\000\船级社项目\labels"
    cm_save_dir = r"outputs\metrics"

    det_rate, false_rate, cm = evaluate(json_dir, yolo_label_dir, cm_save_dir)

    print("\n 评估结果：")
    print(f"检出率：{det_rate:.4f}%")
    print(f"误检率：{false_rate:.4f}%")
    print("\n混淆矩阵已保存在：")
    print(cm_save_dir)
