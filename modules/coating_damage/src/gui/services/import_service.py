from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from config import AreaMethod
from src.utils import model_loader

from ..controller.dto import OfflineService, ServiceResult
from .pipeline_service import (
    FrameProcessingResult,
    build_aligned_image,
    get_intrinsics_with_calibration,
    infer_segment_backend,
    load_calibration_cache,
    normalize_depth_data,
    segment_and_measure,
)


@dataclass(slots=True)
class _OfflineContext:
    rgb_path: Path
    color_image: np.ndarray
    depth_image: np.ndarray
    intrinsics: np.ndarray


class OfflineImportService(OfflineService):
    def __init__(self) -> None:
        self._on_frame: Callable[[object], None] | None = None
        self._on_status: Callable[[str], None] | None = None
        self._on_error: Callable[[str], None] | None = None
        self._context: _OfflineContext | None = None
        self._calibration_cache = load_calibration_cache()
        self._segment_model = None
        self._segment_backend: str | None = None
        self._segment_model_path = ""
        self._refine_model = None
        self._refine_model_path = ""
        self._refined_cache_key: tuple[str, str] | None = None
        self._refined_cache_data: np.ndarray | None = None

    def set_callbacks(self, on_frame=None, on_status=None, on_error=None) -> None:
        self._on_frame = on_frame
        self._on_status = on_status
        self._on_error = on_error

    def _status(self, message: str) -> None:
        if self._on_status is not None:
            self._on_status(message)

    def _error(self, message: str) -> None:
        if self._on_error is not None:
            self._on_error(message)

    @staticmethod
    def _find_depth_file(rgb_path: Path) -> Path | None:
        for name in ("depth_refined.png", "depth_raw.png"):
            candidate = rgb_path.parent / name
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _load_depth_file(
        path: Path, target_shape: tuple[int, int]
    ) -> np.ndarray | None:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            return None
        if depth.ndim == 3:
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        target_h, target_w = target_shape
        if depth.shape[:2] != (target_h, target_w):
            depth = cv2.resize(
                depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST
            )
        return depth

    def _prepare_segment_model(
        self, options: dict[str, Any]
    ) -> tuple[object, str | None, str]:
        segment_enabled = bool(options.get("segment_enabled", True))
        use_semantic_segment = bool(options.get("use_semantic_segment", False))
        load_segment_model = bool(options.get("load_segment_model", False))
        segment_model_path = str(options.get("segment_model_path", "")).strip()

        if not (segment_enabled and use_semantic_segment and load_segment_model):
            return None, None, "Caddy"
        if not segment_model_path:
            raise ValueError("请填写语义分割模型路径")

        segment_backend = infer_segment_backend(segment_model_path)
        seg_display_name = Path(segment_model_path).name or "语义分割模型"
        if (
            self._segment_model is None
            or self._segment_backend != segment_backend
            or self._segment_model_path != segment_model_path
        ):
            self._segment_model = None
            self._segment_backend = None
            self._segment_model_path = ""
            self._status(
                f"[Import] 正在加载分割模型({segment_backend}): {segment_model_path}"
            )
            self._segment_model = model_loader(segment_model_path, type=segment_backend)
            self._segment_backend = segment_backend
            self._segment_model_path = segment_model_path
        return self._segment_model, segment_backend, seg_display_name

    def _refine_depth_frame(self, model, color_image, depth_image, K) -> np.ndarray:
        """
        使用模型对深度图进行精细化处理。
        输入原始深度图和对应的彩色图，输出修正后的深度图。
        """

        # 1. 预处理输入数据（如归一化、调整尺寸等）
        # 2. 将数据输入模型进行推理
        # 3. 后处理模型输出（如反归一化、转换数据类型等）
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        height,width = depth_image.shape

        image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
        image = torch.from_numpy(image).to(device=device, dtype=torch.float32)
        image = (image / 255.0).permute(2, 0, 1).unsqueeze(0)

        depth_m = torch.from_numpy(depth_image.astype(np.float32) / 1000.0)
        depth_m = depth_m.to(device=device, dtype=torch.float32).unsqueeze(0)

        intrinsics = K.astype(np.float32).copy()
        intrinsics[0] /= float(width)  # Normalize fx and cx by width
        intrinsics[1] /= float(height)  # Normalize fy and cy by height
        intrinsics = torch.tensor(intrinsics, dtype=torch.float32, device=device)[None]

        # Run inference
        output = model.infer(
        image, depth_in=depth_m, intrinsics=intrinsics, enable_depth_mask=False
        )

        depth_pred = output["depth"]  # Refined depth map
        return depth_pred.squeeze().cpu().numpy() * 1000.0

    def _resolve_depth_data(
        self, context: _OfflineContext, options: dict[str, Any]
    ) -> np.ndarray:
        refine_enabled = bool(options.get("refine_enabled", False))
        load_refine_model = bool(options.get("load_refine_model", False))
        refine_model_path = str(options.get("refine_model_path", "")).strip()

        if not (refine_enabled and load_refine_model):
            return normalize_depth_data(context.depth_image)
        if not refine_model_path:
            raise ValueError("请填写深度优化模型路径")
        if self._refine_model is None or self._refine_model_path != refine_model_path:
            self._refine_model = None
            self._refine_model_path = ""
            self._refined_cache_key = None
            self._refined_cache_data = None
            self._status(f"[Import] 正在加载深度精细化模型: {refine_model_path}")
            self._refine_model = model_loader(refine_model_path, type="MDM")
            self._refine_model_path = refine_model_path

        cache_key = (str(context.rgb_path), refine_model_path)
        if self._refined_cache_data is None or self._refined_cache_key != cache_key:
            self._status("[Import] 正在执行深度优化模型推理")
            self._refined_cache_data = self._refine_depth_frame(
                self._refine_model,
                context.color_image,
                context.depth_image,
                context.intrinsics,
            )
            self._refined_cache_key = cache_key
        return normalize_depth_data(self._refined_cache_data)

    def _emit_frame(self, frame: FrameProcessingResult) -> None:
        if self._on_frame is not None:
            self._on_frame(frame)

    def _process_context(
        self, options: dict[str, Any], completion_message: str = "[Done] 离线图像处理完成"
    ) -> ServiceResult:
        if self._context is None:
            return ServiceResult(False, "离线上下文为空，请先导入")

        context = self._context
        area_method = AreaMethod(str(options.get("area_method", AreaMethod.AUTO.value)))
        min_region_area = int(options.get("min_region_area", 0))
        show_depth = bool(options.get("show_depth", True))
        depth_display_mode = str(options.get("depth_display_mode", "固定比例尺显示"))
        segment_enabled = bool(options.get("segment_enabled", True))
        use_semantic_segment = bool(options.get("use_semantic_segment", False))

        depth_data = self._resolve_depth_data(context, options)
        segment_model, segment_backend, seg_display = self._prepare_segment_model(
            options
        )

        aligned_image = build_aligned_image(
            context.color_image, depth_data, show_depth, depth_display_mode
        )
        blended, n_regions, area_results, used_semantic_model = segment_and_measure(
            color_image=context.color_image,
            depth_data=depth_data,
            K=context.intrinsics,
            area_method=area_method,
            min_region_area=min_region_area,
            segment_enabled=segment_enabled,
            use_semantic_segment=use_semantic_segment,
            segment_model=segment_model,
            segment_backend=segment_backend,
            status_callback=self._status,
        )
        if segment_enabled and use_semantic_segment and not used_semantic_model:
            seg_display = "Caddy"

        frame = FrameProcessingResult(
            aligned_image=aligned_image,
            seg_image=blended,
            fps=0.0,
            n_regions=int(n_regions),
            method_name=area_method.value,
            seg_display=seg_display,
            raw_rgb=context.color_image,
            raw_depth=depth_data,
            refined_depth=None,
            area_results=area_results,
        )
        self._emit_frame(frame)
        return ServiceResult(True, completion_message)

    def run_offline(self, payload: Any = None) -> ServiceResult:
        if not isinstance(payload, dict):
            return ServiceResult(False, "导入参数缺失")

        rgb_path_str = str(payload.get("rgb_path", "")).strip()
        options = dict(payload.get("options", {}))
        if not rgb_path_str:
            return ServiceResult(False, "未选择 RGB 图像")

        rgb_path = Path(rgb_path_str)
        color_image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if color_image is None:
            return ServiceResult(False, "RGB 图像读取失败")

        depth_path = self._find_depth_file(rgb_path)
        if depth_path is None:
            return ServiceResult(
                False, "同目录未找到深度图（depth_refined.png / depth_raw.png）"
            )
        depth_image = self._load_depth_file(depth_path, color_image.shape[:2])
        if depth_image is None:
            return ServiceResult(False, f"深度图读取失败: {depth_path}")

        intrinsics = get_intrinsics_with_calibration(
            self._calibration_cache,
            color_image.shape[1],
            color_image.shape[0],
            self._status,
        )
        self._context = _OfflineContext(
            rgb_path=rgb_path,
            color_image=color_image,
            depth_image=depth_image,
            intrinsics=intrinsics,
        )
        self._status(
            f"[Import] 导入完成: RGB={rgb_path}, DepthSource={depth_path.name}"
        )
        try:
            return self._process_context(options)
        except Exception as exc:
            self._error(str(exc))
            return ServiceResult(False, str(exc))

    def reprocess_offline(self, payload: Any = None) -> ServiceResult:
        options = dict(payload) if isinstance(payload, dict) else {}
        try:
            return self._process_context(options, completion_message="")
        except Exception as exc:
            self._error(str(exc))
            return ServiceResult(False, str(exc))

    def clear_context(self) -> None:
        self._context = None
        self._refined_cache_key = None
        self._refined_cache_data = None
