from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import (
    DEPTH_CMAP_MAX_MM,
    DEPTH_CMAP_MIN_MM,
    DEPTH_COLORMAP,
    MAX_DEPTH,
    MIN_DEPTH,
    AreaMethod,
    get_calibration_params,
    set_calibration_params,
)
from src.calc import calc_auto, calc_depth_center, calc_ransac
from src.processing import predict_cv, predict_ort, predict_rknn, split_mask
from src.utils import depth_to_colormap_fixed_window
from src.visualization import draw_annotation, draw_labeled_regions


CALIBRATION_JSON_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "calibration_data.json"
)


@dataclass(slots=True)
class FrameProcessingResult:
    aligned_image: np.ndarray
    seg_image: np.ndarray
    fps: float
    n_regions: int
    method_name: str
    seg_display: str
    raw_rgb: np.ndarray
    raw_depth: np.ndarray
    refined_depth: np.ndarray | None
    area_results: list[tuple[int, float]]


def _build_intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _default_intrinsics_matrix(width: int, height: int) -> np.ndarray:
    fx = float(max(width, height))
    fy = fx
    cx = float((width - 1) / 2.0)
    cy = float((height - 1) / 2.0)
    return _build_intrinsics_matrix(fx, fy, cx, cy)


def load_calibration_cache(
    json_path: Path = CALIBRATION_JSON_PATH,
) -> dict[str, dict[str, float]]:
    cache: dict[str, dict[str, float]] = {}
    try:
        with json_path.open("r", encoding="utf-8") as f:
            items = json.load(f)
    except Exception:
        return cache

    for item in items:
        resolution = str(item.get("resolution", "")).strip()
        intr = item.get("camera_intrinsics") or {}
        calib = item.get("calibration_parameters") or {}
        if not resolution:
            continue
        try:
            cache[resolution] = {
                "fx": float(intr["fx"]),
                "fy": float(intr["fy"]),
                "cx": float(intr["cx"]),
                "cy": float(intr["cy"]),
                "kc": float(calib["Kc"]),
                "kv": float(calib["Kv"]),
            }
        except Exception:
            continue
    return cache


def get_intrinsics_with_calibration(
    calibration_cache: dict[str, dict[str, float]],
    width: int,
    height: int,
    log_callback=None,
) -> np.ndarray:
    entry = calibration_cache.get(f"{width}*{height}")
    if entry is not None:
        K = _build_intrinsics_matrix(
            float(entry["fx"]),
            float(entry["fy"]),
            float(entry["cx"]),
            float(entry["cy"]),
        )
        kc = float(entry["kc"])
        kv = float(entry["kv"])
        set_calibration_params(kc, kv)
        message = (
            f"[Calibration] 使用分辨率标定参数: {width}*{height}, Kc={kc}, Kv={kv}"
        )
    else:
        K = _default_intrinsics_matrix(width, height)
        default_kc, default_kv = get_calibration_params()
        message = (
            f"[Calibration] 未找到分辨率标定参数({width}*{height}); "
            f"使用当前 Kc={default_kc}, Kv={default_kv}"
        )
    if log_callback is not None:
        log_callback(message)
    return K


def infer_segment_backend(seg_path: str) -> str:
    lower_path = seg_path.lower()
    if lower_path.endswith(".onnx"):
        return "ONNX"
    if lower_path.endswith(".rknn"):
        return "RKNN"
    raise ValueError("语义分割模型后缀不支持，仅支持 .onnx 或 .rknn")


def build_aligned_image(
    color_image: np.ndarray,
    depth_data: np.ndarray,
    show_depth: bool,
    depth_display_mode: str,
) -> np.ndarray:
    if not show_depth:
        return color_image.copy()

    if depth_display_mode == "固定比例尺显示":
        depth_vis = depth_to_colormap_fixed_window(
            depth_data,
            DEPTH_CMAP_MIN_MM,
            DEPTH_CMAP_MAX_MM,
            DEPTH_COLORMAP,
        )
    else:
        depth_vis = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(color_image, 0.5, depth_vis, 0.5, 0)


def normalize_depth_data(depth_data: np.ndarray) -> np.ndarray:
    depth_data = np.clip(depth_data, 0, 65535).astype(np.uint16)
    return np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)


def segment_and_measure(
    color_image: np.ndarray,
    depth_data: np.ndarray,
    K: np.ndarray,
    area_method: AreaMethod,
    min_region_area: int,
    segment_enabled: bool,
    use_semantic_segment: bool,
    segment_model,
    segment_backend: str | None,
    status_callback=None,
) -> tuple[np.ndarray, int, list[tuple[int, float]], bool]:
    blended = color_image.copy()
    n_regions = 0
    area_results: list[tuple[int, float]] = []
    used_semantic_model = False

    if not segment_enabled:
        return blended, n_regions, area_results, used_semantic_model

    pred_mask = None
    if use_semantic_segment:
        if segment_model is None:
            if status_callback is not None:
                status_callback("[Seg] 模型分割不可用，回退 Caddy")
            pred_mask = predict_cv(color_image)
        elif segment_backend == "ONNX":
            pred_mask = predict_ort(segment_model, color_image)
            used_semantic_model = True
        elif segment_backend == "RKNN":
            pred_mask = predict_rknn(segment_model, color_image)
            used_semantic_model = True
        else:
            if status_callback is not None:
                status_callback("[Seg] 分割后端未知，回退 Caddy")
            pred_mask = predict_cv(color_image)
    else:
        pred_mask = predict_cv(color_image)

    if pred_mask is None:
        return blended, n_regions, area_results, used_semantic_model

    _, labeled, n_regions = split_mask(pred_mask, min_region_area)
    blended = draw_labeled_regions(color_image, labeled, n_regions, pred_mask)

    for label in range(1, n_regions + 1):
        region_mask = labeled == label
        num_pixels = int(np.sum(region_mask))
        if num_pixels < 10:
            continue

        region_depth_data = np.where(region_mask, depth_data, 0)
        area = 0.0
        info = None

        if area_method == AreaMethod.RANSAC:
            area, _ = calc_ransac(region_depth_data, K)
        elif area_method == AreaMethod.DEPTH_CENTER:
            area, _, info = calc_depth_center(region_depth_data, K, num_pixels)
        elif area_method == AreaMethod.AUTO:
            area, _, info = calc_auto(region_depth_data, K, num_pixels)

        area_results.append((int(label), float(area)))
        text = f"{area:.2f}cm2"
        if (
            area_method == AreaMethod.DEPTH_CENTER
            and info is not None
            and "center_x" in info
            and "center_y" in info
        ):
            draw_annotation(
                blended,
                region_mask,
                text,
                cx=int(info["center_x"]),
                cy=int(info["center_y"]),
            )
        else:
            draw_annotation(blended, region_mask, text)

    return blended, int(n_regions), area_results, used_semantic_model
