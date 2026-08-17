import numpy as np
from .plane_fitting import build_plane_basis

def compute_area_on_plane(
    pts: np.ndarray,
    plane_point: np.ndarray,
    normal: np.ndarray,
    method: str = "convexhull",
) -> float:
    """
    将点云投影到拟合平面后计算面积（mm²）。

    Args:
        pts: (N,3)
        plane_point: 平面上一点
        normal: 平面单位法向量
        method: 'convexhull' 或 'delaunay'
    """
    e1, e2 = build_plane_basis(normal)
    rel = pts - plane_point
    xy = np.stack([rel @ e1, rel @ e2], axis=1)

    if method == "convexhull":
        from scipy.spatial import ConvexHull

        hull = ConvexHull(xy)
        hull_pts = xy[hull.vertices]
        x, y = hull_pts[:, 0], hull_pts[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        return float(area)

    elif method == "delaunay":
        from scipy.spatial import Delaunay

        tri = Delaunay(xy)
        tris = xy[tri.simplices]
        a, b, c = tris[:, 0, :], tris[:, 1, :], tris[:, 2, :]
        tri_areas = 0.5 * np.abs(
            a[:, 0] * (b[:, 1] - c[:, 1])
            + b[:, 0] * (c[:, 1] - a[:, 1])
            + c[:, 0] * (a[:, 1] - b[:, 1])
        )
        return float(np.sum(tri_areas))

    raise ValueError("method must be 'convexhull' or 'delaunay'")
