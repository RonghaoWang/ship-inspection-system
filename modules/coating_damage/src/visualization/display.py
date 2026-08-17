"""
可视化：区域着色、面积标注叠加。
"""
from __future__ import annotations
import cv2
import numpy as np


PALETTE = [
    ["background", [127, 127, 127]],
    ["hard_rust", [0, 255, 0]],
    ["edge_rust", [0, 0, 255]],
    ["general_rust", [0, 255, 255]],
]


def draw_labeled_regions(
    color_image: np.ndarray,
    labeled: np.ndarray,
    n_regions: int,
    pred_mask: np.ndarray,
    alpha: float = 0.5,
    palette: list | None = None,
    contour_thickness: int = 2,
) -> np.ndarray:
    """
    先按语义类别为预测结果着色并与原图半透明叠加，再为每个连通域绘制
    与其类别一致的描边颜色。

    Args:
        color_image: 原始 BGR 图像，形状为 (H, W, 3)
        labeled: 连通域标签图，形状为 (H, W)，0 表示背景，1..n_regions 表示区域 ID
        n_regions: 连通域数量
        pred_mask: 语义分割类别图，形状为 (H, W)
        alpha: 原图权重，叠加图权重为 (1 - alpha)
        palette: 调色板，格式为 [["class_name", [B, G, R]], ...]
        contour_thickness: 连通域描边厚度

    Returns:
        blended: BGR 图像
    """
    if color_image.ndim != 3 or color_image.shape[2] != 3:
        raise ValueError("color_image must have shape (H, W, 3).")

    if labeled.ndim != 2:
        raise ValueError("labeled must have shape (H, W).")

    if pred_mask.ndim != 2:
        raise ValueError("pred_mask must have shape (H, W).")

    if color_image.shape[:2] != labeled.shape:
        raise ValueError("color_image and labeled must have the same height and width.")

    if color_image.shape[:2] != pred_mask.shape:
        raise ValueError(
            "color_image and pred_mask must have the same height and width."
        )

    if palette is None:
        palette = PALETTE

    color_overlay = np.zeros_like(color_image)
    unknown_mask = np.ones(pred_mask.shape, dtype=bool)
    for class_id, (_, bgr) in enumerate(palette):
        class_mask = pred_mask == class_id
        color_overlay[class_mask] = np.array(bgr, dtype=color_overlay.dtype)
        unknown_mask &= ~class_mask

    color_overlay[unknown_mask] = np.array([255, 255, 255], dtype=color_overlay.dtype)

    blended = cv2.addWeighted(color_image, alpha, color_overlay, 1 - alpha, 0)

    for label in range(1, n_regions + 1):
        region = (labeled == label).astype(np.uint8)
        if cv2.countNonZero(region) == 0:
            continue

        class_pixels = pred_mask[region > 0]
        if class_pixels.size == 0:
            continue

        class_id = int(np.bincount(class_pixels.astype(np.int32)).argmax())
        if 0 <= class_id < len(palette):
            contour_color = tuple(int(v) for v in palette[class_id][1])
        else:
            contour_color = (255, 255, 255)

        contours, _ = cv2.findContours(
            region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            blended, contours, -1, contour_color, contour_thickness, cv2.LINE_AA
        )

    return blended


def draw_segmentation_overlay(
    color_image: np.ndarray,
    pred_mask: np.ndarray,
    alpha: float = 0.5,
    palette: list | None = None,
) -> np.ndarray:
    """
    将语义分割预测结果按类别调色后，与原始彩色图半透明叠加。

    Args:
        color_image: 原始 BGR 图像，形状为 (H, W, 3)
        pred_mask: 语义分割类别图，形状为 (H, W)
        alpha: 原图权重，叠加图权重为 (1 - alpha)
        palette: 调色板，格式为 [["class_name", [B, G, R]], ...]

    Returns:
        blended: BGR 图像
    """
    if color_image.ndim != 3 or color_image.shape[2] != 3:
        raise ValueError("color_image must have shape (H, W, 3).")

    if pred_mask.ndim != 2:
        raise ValueError("pred_mask must have shape (H, W).")

    if color_image.shape[:2] != pred_mask.shape:
        raise ValueError(
            "color_image and pred_mask must have the same height and width."
        )

    if palette is None:
        palette = PALETTE

    overlay = np.zeros_like(color_image)
    unknown_mask = np.ones(pred_mask.shape, dtype=bool)
    for class_id, (_, bgr) in enumerate(palette):
        class_mask = pred_mask == class_id
        overlay[class_mask] = np.array(bgr, dtype=overlay.dtype)
        unknown_mask &= ~class_mask

    overlay[unknown_mask] = np.array([255, 255, 255], dtype=overlay.dtype)

    blended = cv2.addWeighted(color_image, alpha, overlay, 1 - alpha, 0)
    return blended


def draw_annotation(
    image: np.ndarray,
    region_mask: np.ndarray,
    text: str,
    font_scale: float = 0.6,
    cx: int | None = None,
    cy: int | None = None,
) -> None:
    """
    在区域中心标注面积文字（黑色描边 + 白字 + 黄色圆点），**原地修改图像**。
    """
    ys, xs = np.nonzero(region_mask)
    if xs.size == 0 or ys.size == 0:
        return

    if cx is None or cy is None or cx <= 0 or cy <= 0:
        cx = int(xs.mean())
        cy = int(ys.mean())
    h, w = image.shape[:2]
    cx = max(0, min(w - 1, cx))
    cy = max(0, min(h - 1, cy))

    # 描边 + 白字
    cv2.putText(
        image,
        text,
        (cx, cy),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        (cx, cy),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    # 中心圆点
    cv2.circle(image, (cx, cy), 3, (0, 255, 255), -1)


def draw_fps(
    image: np.ndarray,
    fps: float,
    pos: tuple = (10, 30),
    font_scale: float = 0.5,
) -> None:
    """在指定位置绘制 FPS（黑色描边 + 绿字），原地修改图像。"""
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        image,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
