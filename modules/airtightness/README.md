# 模块：焊缝气密性泄漏检测

## 一、场景与目标

**应用阶段**：船舶建造期 · 管系 / 舱室密性试验

船舶交付前的密性试验（气密性试验）传统上依赖肥皂水 + 人工目视找漏点：主观、慢、可追溯性差、检测员需长时间凝视高风险工位。本模块面向工业级密性试验视频，实现气泡自动检测 + 时序泄漏判定，替代人工找漏点环节。

## 二、方法（两阶段架构）

### 阶段 1：气泡检测

- **主干**：YOLOv26
- **关键改进**：
  - **GSConv 轻量卷积**：参数量降 5%、推理 FPS 涨 40.7%，mAP 仅降 0.6%（面向边缘端部署的关键改进）
  - **WIoU 距离注意力损失**：mAP@0.5:0.95 提升至 0.856
  - **迁移学习**：小样本工业数据下达到与全参数训练相当的检测性能

### 阶段 2：泄漏判定

- **多目标跟踪**：BoT-SORT
- **空间聚类**：DBSCAN（对气泡空间轨迹聚类，判定是否为持续泄漏点）
- **时序平滑**：EMA（指数移动平均）
- **综合判定**：三权重（聚合度 / 持续性 / 一致性）加权评分 + 泄漏确认帧数阈值

## 三、数据

- **数据集**：42 段真实工业气密性试验视频
- **场景类型**：管系加压、舱室密性检查等典型工位

## 四、指标

| 指标 | 数值 |
|---|---|
| mAP@0.5 | **0.95** |
| mAP@0.5:0.95 | **0.848 ~ 0.856** |
| 泄漏判定召回率 | **92.86%（39/42 段视频）** |
| GSConv 参数量下降 | 5.0% |
| GSConv 推理 FPS 提升 | +40.7% |

## 五、快速上手

```bash
# 安装依赖
pip install ultralytics opencv-python numpy scikit-learn torch

# 准备待检测视频，命名为 your_video.mp4 放到本目录，然后：
python tracking.py

# 批处理多个视频：
python batch_detect.py

# 训练：
python train.py
```

结果视频输出至 `leak_detection_output.avi`。

## 六、目录

```
airtightness/
├── tracking.py             # 主推理：检测 + 跟踪 + 泄漏判定
├── leak_detector.py        # 泄漏判定核心（DBSCAN + EMA + 综合评分）
├── batch_detect.py         # 批量推理
├── train.py                # 训练入口
├── split_dataset.py        # 数据集划分
├── grid_search_weights.py  # 三权重网格搜索
├── stats_model.py          # 模型统计
├── configs/
│   ├── bubble.yaml         # 基础配置
│   ├── bubblegsconv.yaml   # GSConv 改进版配置
│   └── data.yaml           # 数据集配置
└── weights/
    └── best.pt             # 训练完成的检测模型
```
