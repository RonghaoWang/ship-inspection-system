"""
对深度图进行稀疏采样。
"""
from __future__ import annotations
import numpy as np


def array_sampling(
    array: np.ndarray,
    step: int,
    start_x: int = 0,
    start_y: int = 0,
    return_indices: bool = True,
) -> tuple[np.ndarray, list[tuple[int, int]] | None]:
    """
    对 2D 数组按指定步长做稀疏采样。

    Args:
        array: (H,W) 输入数组
        step:  采样步长（稀疏因子）
        start_x / start_y: 起始偏移
        return_indices: 是否返回采样点坐标列表

    Returns:
        sampled_array:   采样后的子数组
        sampled_indices: 采样点 (y, x) 坐标列表；当 return_indices=False 时为 None
    """
    array = np.asarray(array)
    height, width = array.shape

    x_idx = list(range(start_x, width, step))
    y_idx = list(range(start_y, height, step))

    sampled_array = array[np.ix_(y_idx, x_idx)]

    if return_indices:
        sampled_indices = [(y, x) for y in y_idx for x in x_idx]
    else:
        sampled_indices = None

    return sampled_array, sampled_indices


if __name__ == "__main__":
    # 测试示例
    test_array = np.arange(100).reshape(10, 10)
    print("原始数组：")
    print(test_array)

    sampled, indices = array_sampling(test_array, step=3, start_x=0, start_y=0)
    print("\n采样后的数组：")
    print(sampled)
    print("\n采样点坐标 (y, x)：")
    print(indices)
