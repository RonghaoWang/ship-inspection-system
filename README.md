# 面向船舶建造与运营的智能辅助检测系统

> 一套统一的视觉智能检测底座，覆盖船舶**建造期焊缝质量**（外观缺陷 + 气密性）与**运营期涂层保护**（损坏面积量算）三大关键节点。

## 一、系统定位

船舶从建造到运营的关键质量节点，长期依赖人工目视 / 接触式检测——效率低、可追溯性差、极端环境风险高。本系统以视觉智能技术为底座，串联建造与运营两个阶段的三类核心检测任务，实现精度、效率、可追溯性的全面提升。

## 二、系统架构

```
┌──────────── 应用场景层 ────────────────────────────────┐
│  建造期：焊缝外观复检、密性试验     运营期：涂层坞修检修 │
└───────────────────┬────────────────────────────────────┘
                    ↓
┌──────────── 数据采集层 ────────────────────────────────┐
│  工业相机（静态图） │ 视频流（气泡） │  RGB-D 相机（涂层）│
└───────────────────┬────────────────────────────────────┘
                    ↓
┌──────────── 图像预处理 ────────────────────────────────┐
│    CLAHE / LAB 归一化 / 深度校正 / letterbox            │
└───────────────────┬────────────────────────────────────┘
                    ↓
┌──────────── 视觉智能内核（两条主干） ──────────────────┐
│    YOLOv8 检测族                    SegFormer 分割族    │
│    ├─ 焊缝外观：WCM + SIoU          ├─ 涂层：LingBot-   │
│    └─ 气密性：GSConv + WIoU +          Depth 融合       │
│       BoT-SORT 跟踪                                    │
└───────────────────┬────────────────────────────────────┘
                    ↓
┌──────────── 后处理与判定 ──────────────────────────────┐
│  缺陷分类  │  DBSCAN 聚类 + EMA 平滑  │ RANSAC+SVD+凸包 │
└───────────────────┬────────────────────────────────────┘
                    ↓
┌──────────── 多后端推理引擎 ────────────────────────────┐
│      PyTorch (训练)  │  ONNX Runtime  │  RKNN (端侧)    │
└───────────────────┬────────────────────────────────────┘
                    ↓
┌────────────    输出  ────────────────────────────────┐
│  缺陷位置+分类 JSON │ 泄漏点定位 │ 损坏面积(m²) │      │
└──────────────────────────────────────────────────────┘
```

## 三、统一 GUI（向导式）

系统提供一个统一的 PySide6 图形界面，按船舶生命周期引导使用者进入对应检测任务：

```
首页
 ├─ 建造期检测
 │    ├─ 焊缝外观缺陷检测
 │    └─ 焊缝气密性泄漏检测
 └─ 运营期检测
      └─ 涂层损坏面积量算
```

启动：

```bash
python system_gui.py
```

首次运行会按需加载对应模块的模型权重，无需的检测任务不会占用资源。

## 四、模块索引

| 模块 | 应用阶段 | 主任务 | 主模型 | 目录 |
|---|---|---|---|---|
| **焊缝外观缺陷检测** | 建造期 | 静态图缺陷检测 | YOLOv8 + WCM 小波卷积 + SIoU | [modules/weld_defect/](modules/weld_defect/) |
| **焊缝气密性泄漏检测** | 建造期 | 视频泄漏识别 | YOLOv8 + GSConv + WIoU + BoT-SORT | [modules/airtightness/](modules/airtightness/) |
| **涂层损坏面积量算** | 运营期 | RGB-D 分割 + 面积计算 | SegFormer + LingBot-Depth | [modules/coating_damage/](modules/coating_damage/) |

## 五、关键指标

| 模块 | 核心指标 | 数据来源 |
|---|---|---|
| 焊缝外观 | mAP **72.5%**，FPS **67**（气孔 +1.9%、渣 +2.5%） | 真实船厂小组立焊缝图像 |
| 焊缝气密性 | mAP@0.5 = **0.95**，mAP@0.5:0.95 = **0.856**，泄漏召回 **92.86%** (39/42) | 42 段工业气密性试验视频 |
| 涂层损坏 | mIoU **65.36%**（超 SegFormer 基线 +8%），RK3588 部署精度损失仅 2.65% | 自建船体锈蚀 RGB-D 数据集 |

## 六、系统目录结构

```
ship-inspection-system/
├── system_gui.py                ← 统一 GUI 入口
├── README.md
├── gui_common/                    ← GUI 通用组件
│   ├── router.py                    页面路由 + 基类
│   ├── widgets.py                   卡片 / 按钮 / 图像显示工厂
│   └── styles.py                    浅色工程风样式表
├── modules/
│   ├── weld_defect/               ← 焊缝外观模块
│   │   ├── infer.py                 主推理脚本（CLI）
│   │   ├── demo.py                  快速演示
│   │   ├── weldseamDetector.py      焊缝定位类
│   │   ├── gui_adapter.py           GUI 适配层
│   │   ├── WTconv/                  小波卷积模块
│   │   ├── weights/                 模型权重
│   │   └── samples/                 测试样本
│   ├── airtightness/              ← 焊缝气密性模块
│   │   ├── tracking.py              CLI 入口
│   │   ├── leak_detector.py         泄漏判定核心（DBSCAN + EMA + 三权重）
│   │   ├── leak_pipeline.py         可复用处理管线（GUI/CLI 共用）
│   │   ├── gui_adapter.py           GUI 适配层
│   │   ├── configs/                 训练与数据配置
│   │   └── weights/                 检测模型权重
│   └── coating_damage/            ← 涂层损坏模块
│       ├── coating_widget.py        嵌入 GUI 的检测控件
│       ├── main_gui.py              原独立 GUI（保留）
│       ├── config.py                运行时配置
│       ├── requirements.txt         依赖清单
│       ├── src/                     算法与相机、可视化子模块
│       ├── mdm/                     深度估计骨干（DINOv2-RGBD）
│       └── models/                  SegFormer ONNX / RKNN 权重
└── assets/                        ← 架构图、演示图预留位
    ├── architecture/
    └── demo_images/
```

## 七、快速上手（GUI）
气密性模型权重下载

https://pan.quark.cn/s/f8e7f8d5e029

将best.pt下载后置于\modules\airtightness\weights文件夹之下

推荐从统一 GUI 使用：

```bash
# 1. 安装依赖
安装根目录下requirements.txt中的依赖包

# 2. 启动系统
python system_gui.py
```

各检测页额外依赖（首次进入对应页面时按需安装）：

| 检测页 | 关键依赖 |
|---|---|
| 焊缝外观 | `ultralytics opencv-python numpy` |
| 焊缝气密性 | `ultralytics scikit-learn torch` |
| 涂层损坏 | `pyorbbecsdk  torch onnxruntime` |

未安装依赖时对应检测页会显示友好提示，不影响其他模块使用。

## 八、快速上手（CLI，可选）

各模块保留独立命令行入口，可脱离 GUI 单独运行：

```bash
# 焊缝外观（批处理 guitest/ 目录下所有图）
python modules/weld_defect/infer.py

# 焊缝气密性（读取 your_video.mp4）
python modules/airtightness/tracking.py

# 涂层损坏（原独立 GUI）
python modules/coating_damage/main_gui.py
```

## 九、许可

系统代码与文档采用 MIT 许可，模型权重与数据集受各自协议约束。
