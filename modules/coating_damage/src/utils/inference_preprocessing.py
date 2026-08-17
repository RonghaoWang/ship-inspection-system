"""
模型推理前处理

"""

import cv2
import numpy as np


def letterbox_resize(
    image: np.ndarray, target_size=(512, 512), pad_val=(114, 114, 114)
):
    """
    将输入图像调整为目标大小，保持宽高比，并在必要时添加边框填充。
    Args:
        image: 输入图像 (H×W×C)
        target_size: 目标尺寸 (width, height)
        pad_val: 填充颜色 (B, G, R)
    Returns:
        letterboxed: 调整后的图像 (target_height×target_width×C)
        new_size: 调整后图像的实际内容尺寸 (new_height, new_width)
        padding: 填充的像素数 (top, bottom, left, right)
    """

    src_h, src_w = image.shape[:2]
    dst_w, dst_h = target_size

    scale = min(dst_w / src_w, dst_h / src_h)
    new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = dst_w - new_w
    pad_h = dst_h - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    letterboxed = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_val,
    )
    return letterboxed, (new_h, new_w), (top, bottom, left, right)


def normalize(
    img: np.ndarray,
    mean: np.ndarray = np.array([123.675, 116.28, 103.53], dtype=np.float32),
    std: np.ndarray = np.array([58.395, 57.12, 57.375], dtype=np.float32),
):
    """
    对输入图像进行归一化处理。
    Args:
        img: 输入图像RGB (H×W×C)
        mean: 均值
        std: 标准差
    Returns:
        normalized: 归一化后的图像 (H×W×C)
    """
    return (img.astype(np.float32) - mean) / std
