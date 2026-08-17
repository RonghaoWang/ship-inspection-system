import numpy as np
import cv2


def split_mask(
    mask,
    min_region_area: int = 500,
):
    """
    将多类别分割掩码分离成不同的连通区域。
    Args:
        mask: 输入分割掩码（0 为背景，>0 为互斥类别）
        min_region_area: 最小区域面积（像素），低于此值的连通域被剔除
    Returns:
        foreground_mask: 二值前景掩膜 (H×W, uint8, 0/1)
        labeled: 全局连续标签图 (H×W, int32, 0=背景, 1..n=各区域)
        n: 前景区域总数
    """
    mask = np.asarray(mask)
    foreground_mask = np.zeros(mask.shape, dtype=np.uint8)
    labeled = np.zeros(mask.shape, dtype=np.int32)

    next_label = 1
    class_ids = np.unique(mask)
    class_ids = class_ids[class_ids > 0]

    for class_id in class_ids:
        class_mask = (mask == class_id).astype(np.uint8)

        if min_region_area and min_region_area > 0:
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                class_mask, connectivity=8
            )
            if n_labels > 1:
                keep = stats[:, cv2.CC_STAT_AREA] >= min_region_area
                keep[0] = False
                keep_labels = np.flatnonzero(keep)
                class_mask = np.isin(labels, keep_labels).astype(np.uint8)

        n_labels, class_labeled, _, _ = cv2.connectedComponentsWithStats(
            class_mask, connectivity=8
        )
        n_regions = n_labels - 1
        if n_regions <= 0:
            continue

        class_fg = class_labeled > 0
        foreground_mask[class_fg] = 1
        labeled[class_fg] = class_labeled[class_fg] + (next_label - 1)
        next_label += n_regions

    n = next_label - 1
    return foreground_mask, labeled, n
