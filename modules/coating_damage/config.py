"""
全局配置常量与可调参数。

运行时可通过键盘交互动态修改的参数在 RuntimeConfig 类中管理。
"""
from __future__ import annotations
import cv2
from enum import Enum


# ─── 系统常量 ───────────────────────────────────────────────
ESC_KEY = 27
MIN_DEPTH = 20  # mm
MAX_DEPTH = 10000  # mm

# ─── 对齐 & 降噪 ───────────────────────────────────────────
ALIGN_MODE = "SW"  # 可选: "HW" / "SW"
HW_DENOISE = True  # 是否开启硬件降噪

# ─── 深度伪彩映射 ──────────────────────────────────────────
DEPTH_CMAP_MIN_MM = MIN_DEPTH
DEPTH_CMAP_MAX_MM = MAX_DEPTH
DEPTH_COLORMAP = cv2.COLORMAP_JET

# ─── 面积校准参数 ──────────────────────────────────────────
Kv = 182  # 深度变异修正权重（0=不修正）注意：这里的值在运行时会被覆盖
Kc = 0.00677  # 面积缩放校准系数


def set_calibration_params(kc: float, kv: float) -> None:
    """更新运行时校准参数。"""
    global Kc, Kv
    Kc = kc
    Kv = kv


def get_calibration_params() -> tuple[float, float]:
    """读取当前生效的校准参数。"""
    return Kc, Kv


class AreaMethod(str, Enum):
    """面积计算方法枚举。"""

    RANSAC = "RANSAC"
    DEPTH_CENTER = "DEPTH_CENTER"
    POINT_CLOUD = "POINT_CLOUD"
    AUTO = "AUTO"  # 新增自动切换方法（失败时回退深度重心法）


# ─── 输出文件 ──────────────────────────────────────────────
SAVE_PATH = "measured_areas_100.xlsx"

# ─── 流分辨率（SW_MODE 下使用） ───────────────────────────
SW_STREAM_WIDTH = 640
SW_STREAM_HEIGHT = 480
SW_STREAM_FPS = 30

# 深度学习模型相关
LOAD_MODEL = True
LOAD_REFINE_MODEL = False  # 是否加载深度精细化模型
REFINE_MODEL_PATH = "models/lingbot-depth-pretrain-vitl-14-v0.5.pt"
LOAD_SEGMENT_MODEL = None  # 是否加载分割模型 （可选：None / "ONNX" / "RKNN"）
SEGMENT_MODEL_PATH = "models/segformer_onnx_512x512.onnx"

# 区域分割方法
SEGMENT_METHOD = "OpenCV"  # 可选: "OpenCV" / "Model"

# ─── 运行时可调参数 ─────────────────────────────────────────

class RuntimeConfig:
    """
    运行时可通过按键动态切换的参数。
    使用类属性代替全局变量，方便在各模块间共享。
    """

    depth_refinement: bool = False  # 是否开启深度精细化（使用模型）
    use_fixed_depth_scale: bool = False  # 是否使用固定比例尺显示深度

    segment_image: bool = True  # 是否进行图像分割
    min_region_area: int = 500  # 连通域最小面积（像素）

    area_method: AreaMethod = AreaMethod.DEPTH_CENTER  # 面积计算方法
    using_sampling: bool = False  # 是否使用稀疏采样（仅 RANSAC 方法有效）

    save_in_xlsx: bool = False  # 是否保存结果到 Excel 文件
