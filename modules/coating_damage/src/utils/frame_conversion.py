"""
帧格式转换工具函数（源自上层 utils.py，保持兼容）。
"""

from typing import Union, Any, Optional

import cv2
import numpy as np
from pyorbbecsdk import VideoFrame, OBFormat


# ─── 颜色空间转换辅助 ─────────────────────────────────────


def _i420_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    u = frame[height : height + height // 4].reshape(height // 2, width // 2)
    v = frame[height + height // 4 :].reshape(height // 2, width // 2)
    yuv_image = cv2.merge([y, u, v])
    return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_I420)


def _nv21_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    uv = frame[height : height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV21)


def _nv12_to_bgr(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    y = frame[0:height, :]
    uv = frame[height : height + height // 2].reshape(height // 2, width)
    yuv_image = cv2.merge([y, uv])
    return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_NV12)


# ─── 公开接口 ─────────────────────────────────────────────


def frame_to_bgr_image(frame: VideoFrame) -> Union[Optional[np.ndarray], Any]:
    """将 Orbbec VideoFrame 转换为 BGR numpy 图像。"""
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())

    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif color_format == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif color_format == OBFormat.I420:
        return _i420_to_bgr(data, width, height)
    elif color_format == OBFormat.NV12:
        return _nv12_to_bgr(data, width, height)
    elif color_format == OBFormat.NV21:
        return _nv21_to_bgr(data, width, height)
    elif color_format == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    else:
        print(f"Unsupported color format: {color_format}")
        return None


def get_depth_data(
    frame: VideoFrame,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    """
    从深度 VideoFrame 提取有效深度数据（单位 mm，uint16），
    超出 [min_depth, max_depth] 的像素置 0。

    Returns: 深度图 (uint16, H×W)，单位 mm
    """
    width = frame.get_width()
    height = frame.get_height()
    scale = frame.get_depth_scale()

    depth_data = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape(
        (height, width)
    )
    depth_data = depth_data.astype(np.float32) * scale
    depth_data = np.where(
        (depth_data > min_depth) & (depth_data < max_depth), depth_data, 0
    ).astype(np.uint16)
    return depth_data
