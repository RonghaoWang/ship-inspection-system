"""
将深度图反投影为 3D 点云。
"""
from __future__ import annotations
import numpy as np


def convert_depth_to_point_cloud(
    Z: np.ndarray, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    将 (H,W) 深度图根据相机内参反投影为 3D 点云。

    Args:
        Z: (H,W) 深度图（单位 mm，0 或 NaN 表示无效）
        K: 3×3 相机内参矩阵

    Returns:
        pts: (N,3)  有效 3D 点 (X, Y, Z)，单位 mm
        pix: (N,2)  对应像素坐标 (u, v)
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    valid = np.isfinite(Z) & (Z > 0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    v_valid, u_valid = np.nonzero(valid)
    u_valid = u_valid.astype(np.float64, copy=False)
    v_valid = v_valid.astype(np.float64, copy=False)
    z_valid = Z[valid].astype(np.float64, copy=False)

    X = (u_valid - cx) * z_valid / fx
    Y = (v_valid - cy) * z_valid / fy

    pts = np.stack([X, Y, z_valid], axis=1)
    pix = np.stack([u_valid, v_valid], axis=1)
    return pts, pix
