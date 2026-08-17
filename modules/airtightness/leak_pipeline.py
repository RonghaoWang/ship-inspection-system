"""气密性泄漏检测处理管线（可复用给 CLI 和 GUI）。

关键设计：
- process_video 是一个生成器，每帧 yield (frame_bgr, leak_result_dict)
- 调用方（CLI/GUI）决定用什么方式消费：写视频、显示、发信号
- 不做任何 cv2.imshow / cv2.waitKey / stdout 打印

模型加载在 LeakPipeline 里做一次，可以在 GUI 里跨视频复用。
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generator

import cv2
import numpy as np


MODULE_DIR = Path(__file__).parent
DEFAULT_WEIGHT = MODULE_DIR / "weights" / "best.pt"


def _patch_conv_layer() -> None:
    """兼容自定义 Conv 层（GSConv 等）——原 tracking.py 里的补丁，保持逻辑一致。"""
    import ultralytics.nn.modules.conv as conv_module

    def universal_forward(self, x):
        x = self.conv(x)
        if hasattr(self, "bn") and self.bn is not None:
            x = self.bn(x)
        return self.act(x)

    conv_module.Conv.forward = universal_forward
    conv_module.Conv.forward_fuse = universal_forward


@dataclass
class LeakDetectionParams:
    """GUI 可调参数 + 三权重（锁死默认）"""
    conf: float = 0.3
    iou: float = 0.5
    imgsz: int = 416
    confirm_frames: int = 3
    # 三权重锁死（作品定位：成熟模型的经验值）
    w_aggregation: float = 0.2
    w_persistence: float = 0.5
    w_consistency: float = 0.3
    # 其他 LeakDetector 参数
    window_size: int = 30
    cluster_eps: float = 180.0
    cluster_min_samples: int = 3
    min_track_age: int = 5
    leak_score_threshold: float = 0.5


@dataclass
class FrameResult:
    frame_idx: int
    frame_bgr: np.ndarray                # 已绘制标注的 BGR 图（可直接给 GUI 转 QPixmap）
    is_leaking: bool
    smoothed_score: float
    aggregation: float
    persistence: float
    consistency: float
    num_bubbles: int
    leak_source: tuple[int, int] | None  # None 表示无泄漏点


@dataclass
class VideoSummary:
    total_frames: int = 0
    leak_frames: int = 0
    max_score: float = 0.0
    final_verdict: bool = False          # 综合判定：是否泄漏
    leak_points: list[tuple[int, int]] = field(default_factory=list)  # 出现过的泄漏源点

    @property
    def leak_ratio(self) -> float:
        return self.leak_frames / self.total_frames if self.total_frames else 0.0


class LeakPipeline:
    """气密性检测管线：加载 YOLO + LeakDetector，跑视频。"""

    def __init__(self, weight_path: str | Path | None = None) -> None:
        self._weight_path = Path(weight_path) if weight_path else DEFAULT_WEIGHT
        self._model = None

    def check_environment(self) -> tuple[bool, str]:
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            return False, "未安装 ultralytics。请执行：pip install ultralytics"
        try:
            from sklearn.cluster import DBSCAN  # noqa: F401
        except ImportError:
            return False, "未安装 scikit-learn（DBSCAN 依赖）。请执行：pip install scikit-learn"
        if not self._weight_path.exists():
            return False, f"缺少检测权重：{self._weight_path}"
        return True, "环境就绪"

    def load(self) -> None:
        if self._model is not None:
            return
        _patch_conv_layer()
        from ultralytics import YOLO

        old_cwd = os.getcwd()
        os.chdir(MODULE_DIR)
        try:
            self._model = YOLO(str(self._weight_path))
        finally:
            os.chdir(old_cwd)

    def process_video(
        self,
        video_path: str | Path,
        params: LeakDetectionParams | None = None,
        should_stop: Callable[[], bool] | None = None,
        writer_output_path: str | Path | None = None,
    ) -> Generator[FrameResult, None, VideoSummary]:
        """逐帧生成 FrameResult；跑完返回 VideoSummary。

        Args:
            video_path: 输入视频路径
            params: 检测参数
            should_stop: 可选回调，返回 True 时提前中断（供 GUI 停止按钮用）
            writer_output_path: 若给出，则同步把带标注的画面写到该路径（.avi/.mp4）
        """
        if self._model is None:
            self.load()

        p = params or LeakDetectionParams()

        # 懒 import，避免顶层污染
        sys.path.insert(0, str(MODULE_DIR))
        try:
            from leak_detector import LeakDetector
        finally:
            if str(MODULE_DIR) in sys.path:
                sys.path.remove(str(MODULE_DIR))

        detector = LeakDetector(
            window_size=p.window_size,
            cluster_eps=p.cluster_eps,
            cluster_min_samples=p.cluster_min_samples,
            min_track_age=p.min_track_age,
            leak_score_threshold=p.leak_score_threshold,
            leak_confirm_frames=p.confirm_frames,
            w_aggregation=p.w_aggregation,
            w_persistence=p.w_persistence,
            w_consistency=p.w_consistency,
        )

        summary = VideoSummary()

        # 切工作目录，让 Ultralytics 找相对路径的 tracker 配置等
        old_cwd = os.getcwd()
        os.chdir(MODULE_DIR)
        writer = None
        writer_fps = 30  # 若无法从原视频取到 fps，用此兜底
        try:
            # 提前探视频 fps，用于 writer
            info = probe_video_info(video_path)
            if info.get("ok") and info.get("fps", 0) > 0:
                writer_fps = float(info["fps"])

            results_gen = self._model.track(
                source=str(video_path),
                conf=p.conf,
                iou=p.iou,
                persist=True,
                tracker="botsort.yaml",
                stream=True,
                show=False,
                save=False,
                imgsz=p.imgsz,
                verbose=False,
            )

            for frame_idx, result in enumerate(results_gen):
                if should_stop is not None and should_stop():
                    break

                orig_img = result.plot()
                boxes = result.boxes

                # 提取 track/conf/cls
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
                    track_ids = (
                        boxes.id.cpu().numpy()
                        if getattr(boxes, "is_track", False) and boxes.id is not None and hasattr(boxes.id, "cpu")
                        else None
                    )
                    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
                    classes = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
                else:
                    xyxy = np.zeros((0, 4))
                    track_ids = np.array([])
                    confs = np.array([])
                    classes = np.array([])

                if boxes is not None and len(boxes) > 0 and track_ids is not None:
                    leak_result = detector.update(xyxy, track_ids, confs, classes)
                else:
                    leak_result = detector.update(
                        np.zeros((0, 4)), np.array([]), np.array([]), np.array([])
                    )

                is_leaking = bool(leak_result.get("is_leaking", False))
                smoothed = float(leak_result.get("smoothed_score", 0.0))
                agg = float(leak_result.get("aggregation", 0.0))
                pers = float(leak_result.get("persistence", 0.0))
                cons = float(leak_result.get("consistency", 0.0))
                num_b = int(leak_result.get("num_bubbles", 0))
                src = leak_result.get("leak_source")
                leak_src: tuple[int, int] | None = None
                if src is not None:
                    leak_src = (int(src[0]), int(src[1]))

                # 绘制标注（与原 tracking.py 一致）
                if is_leaking:
                    cv2.putText(
                        orig_img, f"LEAK DETECTED! Score: {smoothed:.3f}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
                    )
                    cv2.putText(
                        orig_img, f"A={agg:.3f} P={pers:.3f} C={cons:.3f}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                    )
                    if leak_src is not None:
                        sx, sy = leak_src
                        cv2.drawMarker(orig_img, (sx, sy), (0, 0, 255),
                                       markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3)
                        cv2.circle(orig_img, (sx, sy), 12, (0, 0, 255), 2)
                        cv2.putText(orig_img, "LEAK SOURCE", (sx - 55, sy - 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                else:
                    color = (0, 255, 0) if num_b == 0 or track_ids is not None else (0, 165, 255)
                    label = f"No Leak | Score: {smoothed:.3f}" if num_b > 0 else f"No Bubbles | Score: {smoothed:.3f}"
                    cv2.putText(orig_img, label, (10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(orig_img, f"A={agg:.3f} P={pers:.3f} C={cons:.3f}",
                                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # 汇总
                summary.total_frames += 1
                if is_leaking:
                    summary.leak_frames += 1
                    summary.final_verdict = True
                    if leak_src is not None and leak_src not in summary.leak_points:
                        summary.leak_points.append(leak_src)
                if smoothed > summary.max_score:
                    summary.max_score = smoothed

                # 写视频（若开启）
                if writer_output_path is not None:
                    if writer is None:
                        h, w = orig_img.shape[:2]
                        out_path = str(writer_output_path)
                        # 根据扩展名选 fourcc
                        ext = Path(out_path).suffix.lower()
                        if ext == ".mp4":
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        else:
                            fourcc = cv2.VideoWriter_fourcc(*"XVID")
                        writer = cv2.VideoWriter(out_path, fourcc, writer_fps, (w, h))
                    writer.write(orig_img)

                yield FrameResult(
                    frame_idx=frame_idx,
                    frame_bgr=orig_img,
                    is_leaking=is_leaking,
                    smoothed_score=smoothed,
                    aggregation=agg,
                    persistence=pers,
                    consistency=cons,
                    num_bubbles=num_b,
                    leak_source=leak_src,
                )
        finally:
            if writer is not None:
                writer.release()
            os.chdir(old_cwd)

        return summary


def probe_video_info(video_path: str | Path) -> dict:
    """快速取视频元数据，供 GUI 显示。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"ok": False, "error": "无法打开视频"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return {
        "ok": True,
        "fps": fps,
        "frame_count": frame_count,
        "width": w,
        "height": h,
        "duration_s": duration,
    }
