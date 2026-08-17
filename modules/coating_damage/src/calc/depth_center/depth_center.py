"""
计算深度图的加权重心。
"""
from __future__ import annotations
import numpy as np


def get_depth_center(
    depth_data: np.ndarray,
    mask: np.ndarray | None = None,
    ignore_zero: bool = True,
    method: str = "pixel",
) -> tuple[int, int, int]:
    """
    计算深度图的"加权重心"像素坐标 (x, y)，权重为深度值，
    并返回该位置的深度值。

    - 完全向量化，O(H*W)
    - 可选掩膜 mask（同尺寸，非零为有效）
    - 默认忽略深度为 0 的像素

    Returns:
        (x, y, depth_at_center)
        无有效权重时返回 (-1, -1, 0)
    """
    if depth_data.ndim != 2:
        raise ValueError("depth_data 必须是 2D 矩阵")

    h, w = depth_data.shape
    weights = depth_data.astype(np.float64, copy=True)

    if ignore_zero:
        weights[depth_data == 0] = 0.0

    if mask is not None:
        if mask.shape != depth_data.shape:
            raise ValueError("mask 尺寸与 depth_data 不一致")
        weights = weights * (mask != 0)

    row_sums = weights.sum(axis=1)
    col_sums = weights.sum(axis=0)
    w_sum = float(row_sums.sum())
    if w_sum <= 0:
        return -1, -1, 0

    ys = np.arange(h, dtype=np.float64)
    xs = np.arange(w, dtype=np.float64)
    x_c = float(np.dot(col_sums, xs) / w_sum)
    y_c = float(np.dot(row_sums, ys) / w_sum)

    x_pix = max(0, min(w - 1, int(round(x_c))))
    y_pix = max(0, min(h - 1, int(round(y_c))))

    depth_at_center = int(depth_data[y_pix, x_pix])
    return x_pix, y_pix, depth_at_center

if __name__ == "__main__":
    # 测试代码
    depth_test = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                       [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                       [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
                       [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                       [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
                       [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
                       [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
                       [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
                       [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
                       [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]], dtype=np.uint16)
    print("Depth Data:\n", depth_test)
    x, y, depth = get_depth_center(depth_test)
    print(f"Depth Center: ({y}, {x}), Depth: {depth}")
