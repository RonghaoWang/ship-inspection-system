"""焊缝外观检测 - GUI 适配层。

职责：
- 懒加载模型（首次调用时才加载权重，避免 GUI 启动阻塞）
- 包装 infer.run_pipeline，返回结构化结果给 GUI 显示
- 屏蔽依赖 import 错误（Ultralytics 缺失时给友好提示）

给 GUI 用的入口：
    adapter = WeldDefectAdapter()
    result = adapter.detect(image_path, overlap=0.2, conf=0.25)
    # result: {"detections": [...], "vis_image_path": "...", "json_path": "..."}
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODULE_DIR = Path(__file__).parent
WEIGHTS_DIR = MODULE_DIR / "weights"
DEFAULT_OUT_DIR = MODULE_DIR / "outputs"


@dataclass
class Detection:
    category: str
    bbox_xyxy: list[float]
    score: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Detection":
        return cls(
            category=str(d.get("category", "unknown")),
            bbox_xyxy=[float(v) for v in d.get("bbox_xyxy", [0, 0, 0, 0])],
            score=float(d.get("score", 0.0)),
        )


@dataclass
class DetectResult:
    detections: list[Detection]
    vis_image_path: str
    json_path: str
    elapsed_s: float


class WeldDefectAdapter:
    """焊缝外观检测适配器。线程不安全（不要跨线程调用同一实例的 detect）。"""

    AREA_WEIGHT = "weld_area.pt"
    DEFECT_WEIGHT = "liangdutongyi.pt"

    def __init__(self) -> None:
        self._area_model = None
        self._defect_model = None
        self._loaded = False

    def is_ready(self) -> bool:
        return self._loaded

    def check_environment(self) -> tuple[bool, str]:
        """启动前检查依赖与权重。"""
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            return False, "未安装 ultralytics。请执行：pip install ultralytics"
        area = WEIGHTS_DIR / self.AREA_WEIGHT
        defect = WEIGHTS_DIR / self.DEFECT_WEIGHT
        if not area.exists():
            return False, f"缺少焊缝区域权重：{area}"
        if not defect.exists():
            return False, f"缺少缺陷检测权重：{defect}"
        return True, "环境就绪"

    def load(self) -> None:
        """加载模型 + 预热。首次调用时耗时数秒。"""
        if self._loaded:
            return

        import numpy as np
        from ultralytics import YOLO

        # 切到模块目录以匹配 infer.py 里的相对路径预期
        old_cwd = os.getcwd()
        os.chdir(MODULE_DIR)
        try:
            self._area_model = YOLO(str(WEIGHTS_DIR / self.AREA_WEIGHT))
            self._defect_model = YOLO(str(WEIGHTS_DIR / self.DEFECT_WEIGHT))
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._area_model.predict(dummy, verbose=False)
            self._defect_model.predict(dummy, verbose=False)
        finally:
            os.chdir(old_cwd)

        self._loaded = True

    def detect(
        self,
        image_path: str,
        overlap: float = 0.2,
        conf: float = 0.25,
        out_dir: str | None = None,
    ) -> DetectResult:
        """对单张图像跑检测，返回结构化结果。"""
        import time

        if not self._loaded:
            self.load()

        # infer.run_pipeline 用相对路径写文件，切工作目录到模块内
        target_out = Path(out_dir) if out_dir else DEFAULT_OUT_DIR
        target_out.mkdir(parents=True, exist_ok=True)

        # 动态导入 infer（避免顶层 import 拖累环境检查）
        sys.path.insert(0, str(MODULE_DIR))
        try:
            import infer  # type: ignore
        finally:
            if str(MODULE_DIR) in sys.path:
                sys.path.remove(str(MODULE_DIR))

        old_cwd = os.getcwd()
        os.chdir(MODULE_DIR)
        t0 = time.time()
        try:
            infer.run_pipeline(
                image_path=image_path,
                area_model=self._area_model,
                defect_model=self._defect_model,
                out_dir=str(target_out),
                overrate=float(overlap),
                conf=float(conf),
                visualize=True,
                normalize_tiles=True,
                target_mean_L=150.0,
                clahe_clip=2.0,
                clahe_grid=8,
                gamma_only_if_dark=True,
                dark_mean_L_thresh=110.0,
            )
        finally:
            os.chdir(old_cwd)
        elapsed = time.time() - t0

        # 从 JSON 读回结果
        stem = Path(image_path).stem
        json_path = target_out / f"{stem}_result.json"
        vis_path = target_out / f"{stem}_vis.jpg"

        detections: list[Detection] = []
        if json_path.exists():
            with json_path.open(encoding="utf-8") as f:
                meta = json.load(f)
            for d in meta.get("detections", []):
                detections.append(Detection.from_dict(d))

        return DetectResult(
            detections=detections,
            vis_image_path=str(vis_path) if vis_path.exists() else "",
            json_path=str(json_path),
            elapsed_s=elapsed,
        )
