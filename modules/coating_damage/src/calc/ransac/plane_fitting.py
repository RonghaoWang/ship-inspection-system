"""
RANSAC 平面拟合 + 平面基向量构建。
"""
from __future__ import annotations
import numpy as np


_RNG = np.random.default_rng()


def fit_plane_ransac(
    pts: np.ndarray,
    n_iters: int,
    dist_thresh: float,
    min_inliers_ratio: float = 0.5,
    return_inlier_points: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    RANSAC 平面拟合（随机 3 点定义平面，SVD 精化内点）。

    Args:
        pts: (N,3) 点云
        n_iters: 最大迭代次数
        dist_thresh: 内点距离门限（mm）
        min_inliers_ratio: 最小内点比例（预留，暂未使用）

    Returns:
        centroid: 平面质心 (3,)
        normal:   单位法向量 (3,)，朝向相机方向（Z<0）
        inlier_mask: (N,) bool 内点掩码
        inlier_pts: (M,3) 内点点云（当 return_inlier_points=True 时返回）
    """
    N = pts.shape[0]
    if N < 3:
        raise ValueError("点太少，至少需要 3 个点")

    best_inliers = None
    best_count = 0

    for _ in range(n_iters):
        idx = _RNG.choice(N, size=3, replace=False)
        a, b, c = pts[idx]
        v1, v2 = b - a, c - a
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-6:
            continue
        n = n / norm_n # 单位法向量

        dists = np.abs((pts - a) @ n)
        inliers = dists < dist_thresh
        cnt = int(np.sum(inliers))

        if cnt > best_count:
            best_count = cnt
            best_inliers = inliers
            if best_count > 0.99 * N:
                break

    if best_inliers is None:
        raise RuntimeError("RANSAC 未找到平面")

    # SVD 精化
    inlier_pts = pts[best_inliers]
    centroid = np.mean(inlier_pts, axis=0)
    _, _, vh = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
    normal = vh[-1, :]

    # 确保法向量朝向相机方向
    if normal[2] > 0:
        normal = -normal
    normal = normal / np.linalg.norm(normal)

    if return_inlier_points:
        return centroid, normal, best_inliers, inlier_pts
    return centroid, normal, best_inliers


def build_plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    由法向量构造平面上两个正交单位基向量 e1, e2。
    """
    n = normal / np.linalg.norm(normal)
    arbitrary = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(arbitrary, n)) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, arbitrary)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2
