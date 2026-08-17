import os
from ultralytics import YOLO

# ============================================================
# 消融实验开关：每个实验只开启一个
# ============================================================
EXPERIMENT = "combined"   # 可选: "baseline" | "gsconv" | "wiou" | "transfer" | "combined"

# ============================================================
# 通用训练参数
# ============================================================
common_args = dict(
    data="bubble.yaml",
    epochs=500,
    imgsz=640,
    batch=16,
    device=0,
    workers=8,
    optimizer="SGD",
    cos_lr=False,
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    degrees=15.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
)

# ============================================================
# 实验一：基线模型 (YOLO26x + CIoU)
# ============================================================
if EXPERIMENT == "baseline":
    model = YOLO("yolo26x.pt")

# ============================================================
# 实验二：GSConv 轻量化卷积 (Neck 下采样替换为 GSConv)
# ============================================================
elif EXPERIMENT == "gsconv":
    model = YOLO("ultralytics/cfg/models/26/yolo26-conv.yaml").load("yolo26x.pt")

# ============================================================
# 实验三：WIoU 损失函数 (替换 CIoU 为 WIoU v1)
# ============================================================
elif EXPERIMENT == "wiou":
    model = YOLO("yolo26x.pt")
    common_args["iou_type"] = "wiou"

# ============================================================
# 实验四：迁移学习 (冻结浅层 Backbone，微调 Neck+Head)
# ============================================================
elif EXPERIMENT == "transfer":
    model = YOLO("yolo26x.pt")
    common_args["freeze"] = 3

# ============================================================
# 实验五：三项改进联合 (GSConv + WIoU + 迁移学习)
# ============================================================
elif EXPERIMENT == "combined":
    model = YOLO("ultralytics/cfg/models/26/yolo26-conv.yaml").load("yolo26x.pt")
    common_args["iou_type"] = "wiou"
    common_args["freeze"] = 3

results = model.train(**common_args)