"""
深度数据处理：伪彩映射。
"""

import cv2
import numpy as np

from config import DEPTH_COLORMAP


def depth_to_colormap_fixed_window(
    depth_mm: np.ndarray,
    vmin_mm: float,
    vmax_mm: float,
    colormap: int = DEPTH_COLORMAP,
    zero_black: bool = True,
) -> np.ndarray:
    """
    将毫米深度图做固定窗口的颜色映射。

    - depth==0 的像素保持黑色（若 zero_black=True）
    - 小于 vmin 按 vmin 显示，大于 vmax 按 vmax 显示

    Returns:
        BGR 伪彩图 (uint8, H×W×3)
    """
    depth_f = depth_mm.astype(np.float32)
    nonzero = depth_mm > 0
    vis8 = np.zeros_like(depth_mm, dtype=np.uint8)

    if vmax_mm <= vmin_mm:
        vmax_mm = vmin_mm + 1.0

    scale = 255.0 / (vmax_mm - vmin_mm)
    mapped = np.clip((depth_f - vmin_mm) * scale, 0, 255).astype(np.uint8)
    vis8[nonzero] = mapped[nonzero]

    color = cv2.applyColorMap(vis8, colormap)
    if zero_black:
        color[~nonzero] = (0, 0, 0)
    return color
