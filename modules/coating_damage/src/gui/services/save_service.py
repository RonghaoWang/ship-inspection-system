from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..controller.dto import SaveService, ServiceResult


@dataclass(slots=True)
class SaveFrameSnapshot:
    raw_rgb: np.ndarray
    raw_depth: np.ndarray
    refined_depth: np.ndarray | None
    seg_image: np.ndarray | None
    area_results: list[tuple[int, float]]
    n_regions: int


class FrameSaveService:
    @staticmethod
    def _to_uint16_depth(depth_data: np.ndarray) -> np.ndarray:
        if depth_data.dtype == np.uint16:
            return depth_data
        return np.clip(depth_data, 0, 65535).astype(np.uint16)

    @classmethod
    def save_snapshot(
        cls,
        snapshot: SaveFrameSnapshot,
        save_root: Path | None = None,
    ) -> tuple[Path, list[str]]:
        if save_root is None:
            save_root = Path("output") / "saved_frames"
        save_root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        millis = int((time.time() % 1) * 1000)
        save_dir = save_root / f"{stamp}_{millis:03d}"
        save_dir.mkdir(parents=True, exist_ok=True)

        rgb_path = save_dir / "rgb_raw.png"
        depth_raw_path = save_dir / "depth_raw.png"
        cv2.imwrite(str(rgb_path), snapshot.raw_rgb)
        cv2.imwrite(str(depth_raw_path), cls._to_uint16_depth(snapshot.raw_depth))
        saved_items = [str(rgb_path), str(depth_raw_path)]

        if snapshot.refined_depth is not None:
            depth_refined_path = save_dir / "depth_refined.png"
            cv2.imwrite(
                str(depth_refined_path), cls._to_uint16_depth(snapshot.refined_depth)
            )
            saved_items.append(str(depth_refined_path))

        if snapshot.n_regions > 0 and snapshot.seg_image is not None:
            seg_path = save_dir / "segmentation_regions.png"
            cv2.imwrite(str(seg_path), snapshot.seg_image)
            saved_items.append(str(seg_path))

            csv_path = save_dir / "area_results.csv"
            with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["region_id", "area_cm2"])
                for region_id, area_cm2 in snapshot.area_results:
                    writer.writerow([region_id, f"{area_cm2:.6f}"])
            saved_items.append(str(csv_path))
        return save_dir, saved_items


class SnapshotSaveService(SaveService):
    def __init__(self) -> None:
        self._latest_frame: object | None = None

    def update_latest_frame(self, frame: object) -> None:
        self._latest_frame = frame

    def save_snapshot(self, payload: Any = None) -> ServiceResult:
        frame = payload if payload is not None else self._latest_frame
        if frame is None:
            return ServiceResult(False, "当前暂无可保存帧，请稍后再试")

        raw_rgb = getattr(frame, "raw_rgb", None)
        raw_depth = getattr(frame, "raw_depth", None)
        if raw_rgb is None or raw_depth is None:
            return ServiceResult(False, "帧数据不完整，无法保存")

        snapshot = SaveFrameSnapshot(
            raw_rgb=raw_rgb,
            raw_depth=raw_depth,
            refined_depth=getattr(frame, "refined_depth", None),
            seg_image=getattr(frame, "seg_image", None),
            area_results=list(getattr(frame, "area_results", [])),
            n_regions=int(getattr(frame, "n_regions", 0)),
        )
        save_dir, _ = FrameSaveService.save_snapshot(snapshot)
        return ServiceResult(True, f"已保存到目录: {save_dir}")
