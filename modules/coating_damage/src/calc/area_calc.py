"""
面积计算：凸包法 / Delaunay 法 / 深度重心法 / 自动切换法。
"""
from __future__ import annotations
import math
import time
import numpy as np

import config
from src.calc.depth_center.sampling import array_sampling
from src.calc.depth_center.depth_center import get_depth_center
from src.calc.ransac.backproject import convert_depth_to_point_cloud
from src.calc.ransac.plane_fitting import fit_plane_ransac
from src.calc.ransac.compute import compute_area_on_plane


# 运行时耗时统计（单位：ms）
TIMING_STATS: dict[str, dict[str, float]] = {}


def _record_timing(name: str, elapsed_ms: float) -> None:
    stats = TIMING_STATS.setdefault(
        name,
        {
            "count": 0.0,
            "total_ms": 0.0,
            "avg_ms": 0.0,
            "last_ms": 0.0,
        },
    )
    stats["count"] += 1.0
    stats["total_ms"] += elapsed_ms
    stats["last_ms"] = elapsed_ms
    stats["avg_ms"] = stats["total_ms"] / stats["count"]


def get_timing_stats(reset: bool = False) -> dict[str, dict[str, float]]:
    """
    获取当前耗时统计。

    Args:
        reset: 是否在读取后清空统计。
    """
    snapshot = {k: dict(v) for k, v in TIMING_STATS.items()}
    if reset:
        TIMING_STATS.clear()
    return snapshot


# ───────────────────────────────────────────────────────────
# RANSAC 路径
# ───────────────────────────────────────────────────────────


def calc_ransac(
    Z: np.ndarray,
    K: np.ndarray,
    ransac_iters: int = 10,
    ransac_thresh: float = 10.0,
    area_method: str = "convexhull",
) -> tuple[float, float]:
    """
    RANSAC 路径：反投影 → 平面拟合 → 投影面积计算。
    """
    start = time.perf_counter()
    try:
        valid_count = int(np.count_nonzero(np.isfinite(Z) & (Z > 0)))
        if valid_count < 10:
            return 0.0, 0.0

        # 反投影得到点云
        pts, _ = convert_depth_to_point_cloud(Z, K)

        # 点数过少无法拟合平面，直接返回0面积
        if pts.shape[0] < 10:
            return 0.0, 0.0

        # RANSAC 平面拟合
        plane_pt, normal, _, inlier_pts = fit_plane_ransac(
            pts,
            n_iters=ransac_iters,
            dist_thresh=ransac_thresh,
            return_inlier_points=True,
        )

        # 计算平面内点的面积
        area_mm2 = compute_area_on_plane(
            inlier_pts, plane_pt, normal, method=area_method
        )

        # 转换为 cm²并修正
        area = area_mm2 / 100.0 * config.Kc

        # 视线夹角修正
        cos_theta = abs(float(normal[2]))
        theta_rad = math.acos(np.clip(cos_theta, -1.0, 1.0))

        area_fixed = area * (1 + 0.3 * theta_rad)

        return area, area_fixed
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _record_timing("calc_ransac", elapsed_ms)


# ───────────────────────────────────────────────────────────
# 使用open3d计算凸包面积路径
# ───────────────────────────────────────────────────────────


def calc_point_cloud(
    Z: np.ndarray,
    K: np.ndarray,
    voxel_size: float = 3.0,
    downsample_min_points: int = 1000,
) -> float:
    start = time.perf_counter()
    try:
        import open3d as o3d

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        H, W = Z.shape

        us, vs = np.meshgrid(np.arange(W), np.arange(H))
        valid = np.isfinite(Z) & (Z > 0)

        u_valid = us[valid].astype(np.float64)
        v_valid = vs[valid].astype(np.float64)
        z_valid = Z[valid].astype(np.float64)

        X = (u_valid - cx) * z_valid / fx
        Y = (v_valid - cy) * z_valid / fy

        points_3d = np.vstack((X, Y, z_valid)).T

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)

        # 先去离群点，减少凸包计算的干扰。
        clean_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        if len(clean_pcd.points) <= 3:
            return 0.0

        # 再体素下采样，降低凸包计算规模。
        pcd_for_hull = clean_pcd
        if voxel_size > 0 and len(clean_pcd.points) >= downsample_min_points:
            downsampled = clean_pcd.voxel_down_sample(voxel_size=voxel_size)
            # print(f"原始点数: {len(clean_pcd.points)}, 下采样后点数: {len(downsampled.points)}")
            if len(downsampled.points) > 3:
                pcd_for_hull = downsampled

        if len(pcd_for_hull.points) > 3:
            hull_mesh, _ = pcd_for_hull.compute_convex_hull()
            total_hull_area = (
                hull_mesh.get_surface_area() / 2.0
            )  # open3d 计算的面积是双面网格的总面积，除以2得到单面面积 mm2。

            area = total_hull_area / 100.0 * config.Kc  # 转换为 cm²并修正
            return area
        else:
            return 0.0
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _record_timing("calc_point_cloud", elapsed_ms)


# ───────────────────────────────────────────────────────────
# 深度重心路径
# ───────────────────────────────────────────────────────────


def calc_depth_center(
    region_depth_data: np.ndarray,
    K: np.ndarray,
    num_pixels: int,
    use_sampling: bool = False,
    sampling_step: int = 3,
) -> tuple[float, float, dict]:
    """
    深度重心法计算面积。

    Returns:
        area: 原始面积估计值
        area_fixed: 修正后的面积估计值
        info: 计算过程中的统计信息字典
    """
    start = time.perf_counter()
    try:
        valid = np.isfinite(region_depth_data) & (region_depth_data > 0)
        if not np.any(valid):
            info = {
                "depth_mean": 0.0,
                "depth_std": 0.0,
                "CV": 0.0,
                "CV2": 0.0,
                "center_x": -1,
                "center_y": -1,
            }
            return 0.0, 0.0, info

        depth_nozero = region_depth_data[valid]
        depth_mean = float(np.mean(depth_nozero, dtype=np.float64))  # 深度均值
        depth_std = float(np.std(depth_nozero, dtype=np.float64))  # 深度标准差

        CV = depth_std / depth_mean if depth_mean > 0 else 0.0
        CV2 = CV**2

        fx = K[0, 0]
        K_fix = 1 + config.Kv * CV2

        if use_sampling and sampling_step > 1:
            sampled_depth_data, _ = array_sampling(
                region_depth_data, sampling_step, return_indices=False
            )
            x_c, y_c, depth_centre = get_depth_center(
                sampled_depth_data, ignore_zero=True
            )
            # 坐标还原到原图（此处仅深度重心值有用）
            x_c *= sampling_step
            y_c *= sampling_step
        else:
            x_c, y_c, depth_centre = get_depth_center(
                region_depth_data, ignore_zero=True
            )

        area_mm2 = ((depth_centre / fx) ** 2) * num_pixels
        area = area_mm2 / 100.0 * config.Kc  # 转换为 cm²并修正
        area_fixed = K_fix * area  # 倾斜修正

        info = {
            "depth_mean": depth_mean,
            "depth_std": depth_std,
            "CV": CV,
            "CV2": CV2,
            "center_x": x_c,
            "center_y": y_c,
        }
        return area, area_fixed, info
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _record_timing("calc_depth_center", elapsed_ms)


def calc_auto(
    Z: np.ndarray,
    K: np.ndarray,
    num_pixels: int,
    use_sampling: bool = False,
    sampling_step: int = 3,
    switch_threshold: float = 0.20,
    cv2_weight: float = 100.0,
    center_diff_weight: float = 1.0,
    ransac_iters: int = 10,
    ransac_thresh: float = 10.0,
    ransac_area_method: str = "convexhull",
) -> tuple[float, float, dict]:
    """
    自动面积估计（简化版）：
    1) 先用输入深度图计算 CV2，若 cv2_weight * CV2 > switch_threshold，直接使用 RANSAC。
    2) 否则先用深度重心法，再叠加 center_depth_diff_ratio 判断是否切换到 RANSAC。

    Returns:
        area: 原始面积
        area_fixed: 修正面积
        info: 精简决策信息，仅包含
            - selected_method
            - should_switch
            - switch_score
            - ransac_fallback_reason
    """
    start = time.perf_counter()
    try:

        def _calc_depth_center_fast() -> tuple[float, float, float]:
            """
            轻量深度重心计算：
            仅用行/列加权和计算中心，避免构造整幅坐标网格。
            Returns: (area, area_fixed, center_depth_diff_ratio)
            """
            depth_for_center = Z
            step = 1
            if use_sampling and sampling_step > 1:
                depth_for_center = Z[::sampling_step, ::sampling_step]
                step = sampling_step

            weights = np.where(depth_for_center > 0, depth_for_center, 0).astype(
                np.float64, copy=False
            )
            row_sums = weights.sum(axis=1)
            col_sums = weights.sum(axis=0)
            weight_sum = float(row_sums.sum())

            if weight_sum <= 0.0:
                return 0.0, 0.0, 0.0

            ys = np.arange(depth_for_center.shape[0], dtype=np.float64)
            xs = np.arange(depth_for_center.shape[1], dtype=np.float64)

            y_c = float(np.dot(row_sums, ys) / weight_sum)
            x_c = float(np.dot(col_sums, xs) / weight_sum)

            y_pix = int(np.clip(np.round(y_c), 0, depth_for_center.shape[0] - 1))
            x_pix = int(np.clip(np.round(x_c), 0, depth_for_center.shape[1] - 1))
            depth_centre = float(depth_for_center[y_pix, x_pix])

            fx = float(K[0, 0])
            if fx <= 0.0 or depth_centre <= 0.0:
                return 0.0, 0.0, 0.0

            area_mm2 = ((depth_centre / fx) ** 2) * num_pixels
            area = area_mm2 / 100.0 * config.Kc
            K_fix = 1.0 + config.Kv * cv2_val
            area_fixed = K_fix * area

            center_depth_abs_diff = abs(depth_centre - depth_mean)
            center_depth_diff_ratio = (
                center_depth_abs_diff / depth_mean if depth_mean > 0.0 else 0.0
            )

            return float(area), float(area_fixed), float(center_depth_diff_ratio)

        def _calc_ransac_fast() -> tuple[float, float]:
            # 反投影得到点云
            pts, _ = convert_depth_to_point_cloud(Z, K)
            if pts.shape[0] < 10:
                return 0.0, 0.0

            plane_pt, normal, inlier_mask = fit_plane_ransac(
                pts, n_iters=ransac_iters, dist_thresh=ransac_thresh
            )
            inlier_pts = pts[inlier_mask]

            area_mm2 = compute_area_on_plane(
                inlier_pts, plane_pt, normal, method=ransac_area_method
            )
            area = area_mm2 / 100.0 * config.Kc

            # view_dir=[0,0,-1] 时，|dot(normal, view_dir)| = |normal[2]|
            cos_theta = abs(float(normal[2]))
            theta_rad = math.acos(np.clip(cos_theta, -1.0, 1.0))
            area_fixed = area * (1.0 + 0.3 * theta_rad)
            return float(area), float(area_fixed)

        valid = np.isfinite(Z) & (Z > 0)
        depth_nozero = Z[valid]

        if depth_nozero.size == 0:
            auto_info = {
                "selected_method": "DEPTH_CENTER",
                "should_switch": False,
                "switch_score": 0.0,
                "ransac_fallback_reason": "",
            }
            return 0.0, 0.0, auto_info

        # 统一做一次深度统计，供两阶段决策与重心修正复用。
        depth_mean = float(depth_nozero.mean(dtype=np.float64))
        depth_std = float(depth_nozero.std(dtype=np.float64))
        cv_val = depth_std / depth_mean if depth_mean > 0.0 else 0.0
        cv2_val = cv_val * cv_val
        cv2_score = cv2_weight * cv2_val

        fallback_reason = ""

        # 先验快速判断：若 CV2 分量已超阈值，直接切到 RANSAC。
        if cv2_score > switch_threshold:
            try:
                area, area_fixed = _calc_ransac_fast()
                auto_info = {
                    "selected_method": "RANSAC",
                    "should_switch": True,
                    "switch_score": float(cv2_score),
                    "ransac_fallback_reason": "",
                }
                return area, area_fixed, auto_info
            except Exception as exc:
                # 仅在 RANSAC 失败时才回退到深度重心。
                fallback_reason = str(exc)
                base_area, base_area_fixed, _ = _calc_depth_center_fast()
                auto_info = {
                    "selected_method": "DEPTH_CENTER_FALLBACK",
                    "should_switch": True,
                    "switch_score": float(cv2_score),
                    "ransac_fallback_reason": fallback_reason,
                }
                return base_area, base_area_fixed, auto_info

        # 未触发先验切换时，计算深度重心并做二阶段判定。
        base_area, base_area_fixed, center_diff_ratio = _calc_depth_center_fast()
        switch_score = cv2_score + center_diff_weight * center_diff_ratio
        should_switch = switch_score > switch_threshold

        selected_method = "DEPTH_CENTER"
        area = base_area
        area_fixed = base_area_fixed

        if should_switch:
            try:
                area, area_fixed = _calc_ransac_fast()
                selected_method = "RANSAC"
            except Exception as exc:
                selected_method = "DEPTH_CENTER_FALLBACK"
                fallback_reason = str(exc)

        auto_info = {
            "selected_method": selected_method,
            "should_switch": bool(should_switch),
            "switch_score": float(switch_score),
            "ransac_fallback_reason": fallback_reason,
        }
        return area, area_fixed, auto_info
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _record_timing("calc_auto", elapsed_ms)
