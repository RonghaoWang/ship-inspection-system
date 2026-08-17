from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable

# import numpy as np
from pyorbbecsdk import AlignFilter, Context, OBStreamType, Pipeline

from config import ALIGN_MODE, HW_DENOISE, MAX_DEPTH, MIN_DEPTH, RuntimeConfig, AreaMethod
from src.camera import enable_hw_denoise, get_camera_parameters, get_stream_config
from src.processing import refine_depth_frame
from src.utils import frame_to_bgr_image, get_depth_data, model_loader

from ..controller.dto import RealtimeService, ServiceResult
from .pipeline_service import (
    FrameProcessingResult,
    build_aligned_image,
    get_intrinsics_with_calibration,
    infer_segment_backend,
    load_calibration_cache,
    segment_and_measure,
)

try:
    from PySide6.QtCore import QObject, QThread, Signal, Slot
except ImportError as exc:
    raise SystemExit("PySide6 未安装，请先执行: pip install PySide6") from exc


@dataclass
class ProcessingOptions:
    segment_enabled: bool = bool(RuntimeConfig.segment_image)
    show_depth: bool = True
    depth_display_mode: str = (
        "固定比例尺显示" if RuntimeConfig.use_fixed_depth_scale else "自适应比例尺显示"
    )
    depth_refinement: bool = bool(RuntimeConfig.depth_refinement)
    min_region_area: int = int(RuntimeConfig.min_region_area)
    area_method: AreaMethod = RuntimeConfig.area_method
    use_semantic_segment: bool = False
    load_refine_model: bool = False
    load_segment_model: bool = False
    refine_model_path: str = "models/lingbot-depth-pretrain-vitl-14-v0.5.pt"
    segment_model_path: str = "models/segformer_onnx_512x512.onnx"


class CameraWorker(QObject):
    frame_ready = Signal(object)
    status = Signal(str)
    error = Signal(str)
    finished = Signal()

    def __init__(
        self, calibration_cache: dict[str, dict[str, float]] | None = None
    ) -> None:
        super().__init__()
        self._calibration_cache = calibration_cache or {}
        self._running = False
        self._options = ProcessingOptions()
        self._lock = Lock()
        self._refine_model = None
        self._segment_model = None
        self._segment_backend: str | None = None
        self._segment_display_name = "Caddy"

    def stop(self) -> None:
        self._running = False

    def update_options(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._options.segment_enabled = bool(
                payload.get("segment_enabled", self._options.segment_enabled)
            )
            self._options.show_depth = bool(
                payload.get("show_depth", self._options.show_depth)
            )
            self._options.depth_display_mode = str(
                payload.get("depth_display_mode", self._options.depth_display_mode)
            )
            self._options.depth_refinement = bool(
                payload.get("refine_enabled", self._options.depth_refinement)
            )
            self._options.use_semantic_segment = bool(
                payload.get("use_semantic_segment", self._options.use_semantic_segment)
            )
            self._options.min_region_area = max(
                0, int(payload.get("min_region_area", self._options.min_region_area))
            )
            self._options.load_refine_model = bool(
                payload.get("load_refine_model", self._options.load_refine_model)
            )
            self._options.load_segment_model = bool(
                payload.get("load_segment_model", self._options.load_segment_model)
            )
            self._options.refine_model_path = str(
                payload.get("refine_model_path", self._options.refine_model_path)
            )
            self._options.segment_model_path = str(
                payload.get("segment_model_path", self._options.segment_model_path)
            )
            method_name = str(
                payload.get("area_method", self._options.area_method.value)
            )
            self._options.area_method = AreaMethod(method_name)

    def _get_options(self) -> ProcessingOptions:
        with self._lock:
            return ProcessingOptions(
                segment_enabled=self._options.segment_enabled,
                show_depth=self._options.show_depth,
                depth_display_mode=self._options.depth_display_mode,
                depth_refinement=self._options.depth_refinement,
                min_region_area=self._options.min_region_area,
                area_method=self._options.area_method,
                use_semantic_segment=self._options.use_semantic_segment,
                load_refine_model=self._options.load_refine_model,
                load_segment_model=self._options.load_segment_model,
                refine_model_path=self._options.refine_model_path,
                segment_model_path=self._options.segment_model_path,
            )

    def _load_models(self, opts: ProcessingOptions) -> None:
        self._refine_model = None
        self._segment_model = None
        self._segment_backend = None
        self._segment_display_name = "Caddy"
        if opts.load_refine_model:
            path = opts.refine_model_path.strip()
            if not path:
                raise ValueError("已启用深度优化模型，但模型路径为空")
            self.status.emit(f"[Model] 正在加载深度精细化模型: {path}")
            self._refine_model = model_loader(path, type="MDM")
            self.status.emit("[Model] 深度精细化模型加载成功")

        if opts.load_segment_model:
            seg_path = opts.segment_model_path.strip()
            if not seg_path:
                raise ValueError("已启用语义分割模型，但模型路径为空")
            backend = infer_segment_backend(seg_path)
            self._segment_backend = backend
            self._segment_display_name = Path(seg_path).name or "语义分割模型"
            self.status.emit(f"[Model] 正在加载分割模型({backend}): {seg_path}")
            self._segment_model = model_loader(seg_path, type=backend)
            self.status.emit(f"[Model] 分割模型({backend})加载成功")

    @Slot()
    def run(self) -> None:
        pipeline = None
        align_filter = None
        prev_time = time.time()
        fps_ema = None
        fps_alpha = 0.15
        self._running = True

        try:
            ctx = Context()
            if len(ctx.query_devices()) <= 0:
                raise RuntimeError("未检测到相机，请检查相机连接")

            pipeline = Pipeline()

            if HW_DENOISE:
                enable_hw_denoise(pipeline)
                self.status.emit("[Camera] 已启用硬件降噪")

            startup_opts = self._get_options()
            self._load_models(startup_opts)

            cfg_result = get_stream_config(pipeline)
            if cfg_result is None:
                raise RuntimeError("获取相机流配置失败")
            depth_profile, color_profile, config = cfg_result

            K = None
            try:
                depth_w = int(depth_profile.get_width())
                depth_h = int(depth_profile.get_height())
            except Exception:
                depth_w, depth_h = 0, 0
            if depth_w > 0 and depth_h > 0:
                K = get_intrinsics_with_calibration(
                    self._calibration_cache, depth_w, depth_h, self.status.emit
                )
            if K is None:
                K = get_camera_parameters(color_profile, depth_profile)

            pipeline.start(config)
            self.status.emit("[Camera] 启动相机流")
            if ALIGN_MODE == "SW":
                align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

            while self._running:
                frames = pipeline.wait_for_frames(1000)
                if frames is None:
                    continue
                if ALIGN_MODE == "HW":
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    if not color_frame or not depth_frame:
                        continue
                elif ALIGN_MODE == "SW":
                    frames = align_filter.process(frames)
                    if not frames:
                        continue
                    frames = frames.as_frame_set()
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    if not color_frame or not depth_frame:
                        continue
                else:
                    raise RuntimeError(f"不支持的 ALIGN_MODE: {ALIGN_MODE}")

                opts = self._get_options()
                raw_depth_data = get_depth_data(depth_frame, MIN_DEPTH, MAX_DEPTH)
                refined_depth_data = None
                if (
                    opts.load_refine_model
                    and opts.depth_refinement
                    and self._refine_model is not None
                ):
                    refined_depth_data = refine_depth_frame(
                        self._refine_model, color_frame, depth_frame, K
                    )
                    depth_data = refined_depth_data
                else:
                    depth_data = raw_depth_data

                color_image = frame_to_bgr_image(color_frame)
                if color_image is None:
                    continue

                aligned_image = build_aligned_image(
                    color_image, depth_data, opts.show_depth, opts.depth_display_mode
                )
                blended, n_regions, area_results, used_semantic_model = (
                    segment_and_measure(
                        color_image=color_image,
                        depth_data=depth_data,
                        K=K,
                        area_method=opts.area_method,
                        min_region_area=opts.min_region_area,
                        segment_enabled=opts.segment_enabled,
                        use_semantic_segment=opts.use_semantic_segment,
                        segment_model=self._segment_model,
                        segment_backend=self._segment_backend,
                        status_callback=self.status.emit,
                    )
                )
                seg_display = (
                    self._segment_display_name
                    if (
                        opts.use_semantic_segment
                        and opts.load_segment_model
                        and used_semantic_model
                    )
                    else "Caddy"
                )

                curr_time = time.time()
                inst_fps = 1.0 / max(curr_time - prev_time, 1e-6)
                prev_time = curr_time
                fps_ema = (
                    inst_fps
                    if fps_ema is None
                    else (fps_alpha * inst_fps + (1.0 - fps_alpha) * fps_ema)
                )

                self.frame_ready.emit(
                    FrameProcessingResult(
                        aligned_image=aligned_image,
                        seg_image=blended,
                        fps=float(fps_ema),
                        n_regions=int(n_regions),
                        method_name=opts.area_method.value,
                        seg_display=seg_display,
                        raw_rgb=color_image,
                        raw_depth=raw_depth_data,
                        refined_depth=refined_depth_data,
                        area_results=area_results,
                    )
                )
        except Exception as exc:
            if isinstance(exc, RuntimeError):
                self.error.emit(str(exc))
            else:
                detail = "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                )
                self.error.emit(detail)
        finally:
            try:
                if pipeline is not None:
                    pipeline.stop()
            except Exception:
                pass
            try:
                if self._segment_backend == "RKNN" and self._segment_model is not None:
                    self._segment_model.release()
            except Exception:
                pass
            self.status.emit("[Camera] 相机流已停止")
            self.finished.emit()


class RealtimeCameraService(RealtimeService):
    def __init__(self) -> None:
        self._on_frame: Callable[[object], None] | None = None
        self._on_status: Callable[[str], None] | None = None
        self._on_error: Callable[[str], None] | None = None
        self._thread: QThread | None = None
        self._worker: CameraWorker | None = None
        self._calibration_cache = load_calibration_cache()

    def set_callbacks(self, on_frame=None, on_status=None, on_error=None) -> None:
        self._on_frame = on_frame
        self._on_status = on_status
        self._on_error = on_error

    def start_realtime(self, payload: Any = None) -> ServiceResult:
        if self._thread is not None:
            return ServiceResult(False, "实时模式已在运行")

        self._thread = QThread()
        self._worker = CameraWorker(self._calibration_cache)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)

        if self._on_frame is not None:
            self._worker.frame_ready.connect(self._on_frame)
        if self._on_status is not None:
            self._worker.status.connect(self._on_status)
        if self._on_error is not None:
            self._worker.error.connect(self._on_error)

        if isinstance(payload, dict):
            self._worker.update_options(payload)
        self._thread.start()
        return ServiceResult(True, "")

    @Slot()
    def _on_thread_finished(self) -> None:
        self._worker = None
        self._thread = None

    def stop_realtime(self) -> ServiceResult:
        if self._worker is None or self._thread is None:
            return ServiceResult(True, "")
        self._worker.stop()
        self._thread.quit()
        self._thread.wait(2000)
        return ServiceResult(True, "")

    def update_runtime_options(self, payload: Any = None) -> ServiceResult:
        if self._worker is None:
            return ServiceResult(False, "实时模式未运行")
        if not isinstance(payload, dict):
            return ServiceResult(False, "参数格式错误")
        self._worker.update_options(payload)
        return ServiceResult(True, "")
