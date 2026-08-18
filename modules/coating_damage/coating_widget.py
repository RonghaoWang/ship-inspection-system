"""
涂层损坏面积量算 - 嵌入式检测控件。

从原 main_gui.py 的 MainWindow 改造而来：
- QMainWindow → QWidget（可嵌入向导页）
- closeEvent → shutdown()（父页面负责调用）
- 按钮名称对齐系统风格
"""

from __future__ import annotations

import sys
import time
import csv
import json
import traceback
from pathlib import Path
from dataclasses import dataclass
from threading import Lock

import cv2
import numpy as np

# 让"config"/"src"这些相对模块名可以被解析，
# 无论本文件是作为脚本运行，还是作为 modules.coating_damage.coating_widget 被导入。
_MODULE_DIR = str(Path(__file__).resolve().parent)
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from pyorbbecsdk import AlignFilter, Context, OBStreamType, Pipeline

from config import (
    ALIGN_MODE,
    DEPTH_CMAP_MAX_MM,
    DEPTH_CMAP_MIN_MM,
    DEPTH_COLORMAP,
    HW_DENOISE,
    MAX_DEPTH,
    MIN_DEPTH,
    RuntimeConfig,
    AreaMethod,
    get_calibration_params,
    set_calibration_params,
)
from src.calc import calc_auto, calc_depth_center, calc_ransac
from src.camera import enable_hw_denoise, get_camera_parameters, get_stream_config
from src.processing import (
    predict_cv,
    predict_ort,
    predict_rknn,
    refine_depth_frame,
    split_mask,
)
from src.utils import (
    depth_to_colormap_fixed_window,
    frame_to_bgr_image,
    get_depth_data,
    model_loader,
)
from src.visualization import draw_annotation, draw_labeled_regions

try:
    from PySide6.QtCore import QObject, QSize, QThread, Qt, QTimer, Signal, Slot
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QSlider,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    raise SystemExit("PySide6 未安装，请先执行: pip install PySide6") from exc


# GUI 本地默认值：不再依赖 config.py 中的模型开关与路径配置。
GUI_DEFAULT_LOAD_REFINE_MODEL = False
GUI_DEFAULT_LOAD_SEGMENT_MODEL = False
GUI_DEFAULT_USE_SEMANTIC_SEGMENT = False
_MODELS_DIR = Path(__file__).resolve().parent / "models"
GUI_DEFAULT_REFINE_MODEL_PATH = str(_MODELS_DIR / "lingbot-depth-pretrain-vitl-14-v0.5.pt")
GUI_DEFAULT_SEGMENT_MODEL_PATH = str(_MODELS_DIR / "segformer_onnx_512x512.onnx")
CALIBRATION_JSON_PATH = (
    Path(__file__).resolve().parent / "src" / "data" / "calibration_data.json"
)


def _build_intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _default_intrinsics_matrix(width: int, height: int) -> np.ndarray:
    fx = float(max(width, height))
    fy = fx
    cx = float((width - 1) / 2.0)
    cy = float((height - 1) / 2.0)
    return _build_intrinsics_matrix(fx, fy, cx, cy)


def load_calibration_cache(
    json_path: Path = CALIBRATION_JSON_PATH,
) -> dict[str, dict[str, float]]:
    """一次性加载标定 JSON 到内存缓存。"""
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


def lookup_calibration(
    calibration_cache: dict[str, dict[str, float]],
    width: int,
    height: int,
) -> tuple[np.ndarray, tuple[float, float] | None]:
    """按分辨率从缓存查找内参与标定参数。"""
    entry = calibration_cache.get(f"{width}*{height}")
    if entry is None:
        return _default_intrinsics_matrix(width, height), None
    intrinsics = _build_intrinsics_matrix(
        entry["fx"],
        entry["fy"],
        entry["cx"],
        entry["cy"],
    )
    return intrinsics, (entry["kc"], entry["kv"])


@dataclass
class ProcessingOptions:
    """线程安全共享的运行时选项。"""

    segment_image: bool = bool(RuntimeConfig.segment_image)
    show_depth: bool = True
    depth_display_mode: str = (
        "固定比例尺显示" if RuntimeConfig.use_fixed_depth_scale else "自适应比例尺显示"
    )
    depth_refinement: bool = bool(RuntimeConfig.depth_refinement)
    min_region_area: int = int(RuntimeConfig.min_region_area)
    area_method: AreaMethod = RuntimeConfig.area_method
    use_semantic_segment: bool = GUI_DEFAULT_USE_SEMANTIC_SEGMENT
    load_refine_model: bool = GUI_DEFAULT_LOAD_REFINE_MODEL
    load_segment_model: bool = GUI_DEFAULT_LOAD_SEGMENT_MODEL
    refine_model_path: str = GUI_DEFAULT_REFINE_MODEL_PATH
    segment_model_path: str = GUI_DEFAULT_SEGMENT_MODEL_PATH


def bgr_to_qpixmap(image_bgr: np.ndarray, target_size: QSize | None = None) -> QPixmap:
    """将 OpenCV BGR 图像转换为 QPixmap，可选先按目标尺寸缩放。"""
    if image_bgr is None or image_bgr.size == 0:
        return QPixmap()

    frame = image_bgr
    if (
        target_size is not None
        and target_size.width() > 0
        and target_size.height() > 0
        and (
            image_bgr.shape[1] != target_size.width()
            or image_bgr.shape[0] != target_size.height()
        )
    ):
        interp = (
            cv2.INTER_AREA
            if target_size.width() <= image_bgr.shape[1]
            and target_size.height() <= image_bgr.shape[0]
            else cv2.INTER_LINEAR
        )
        frame = cv2.resize(
            image_bgr,
            (target_size.width(), target_size.height()),
            interpolation=interp,
        )

    if not frame.flags.c_contiguous:
        frame = np.ascontiguousarray(frame)

    h, w, _ = frame.shape
    qimg = QImage(
        frame.data,
        w,
        h,
        int(frame.strides[0]),
        QImage.Format.Format_BGR888,
    )
    return QPixmap.fromImage(qimg)


def infer_segment_backend(seg_path: str) -> str:
    """根据模型文件后缀推断分割后端。"""
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
    """复用的深度叠加显示逻辑。"""
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
    """复用的分割+面积计算+标注逻辑。"""
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
                status_callback("[Seg] 模型分割不可用，回退 OpenCV")
            pred_mask = predict_cv(color_image)
        elif segment_backend == "ONNX":
            pred_mask = predict_ort(segment_model, color_image)
            used_semantic_model = True
        elif segment_backend == "RKNN":
            pred_mask = predict_rknn(segment_model, color_image)
            used_semantic_model = True
        else:
            if status_callback is not None:
                status_callback("[Seg] 分割后端未知，回退 OpenCV")
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

    return blended, n_regions, area_results, used_semantic_model


class ImportProcessingWorker(QObject):
    """导入模式后台处理线程：负责模型加载与单帧推理，避免阻塞 UI。"""

    result_ready = Signal(object)
    status = Signal(str)
    error = Signal(object)

    def __init__(
        self,
        calibration_cache: dict[str, dict[str, float]] | None = None,
    ) -> None:
        super().__init__()
        self._calibration_cache = calibration_cache or {}
        self._segment_model = None
        self._segment_backend: str | None = None
        self._segment_model_path: str = ""

        self._refine_model = None
        self._refine_model_path: str = ""
        self._refined_cache_source_id: int = -1
        self._refined_cache_model_path: str = ""
        self._refined_cache_data: np.ndarray | None = None

    def _release_segment_model(self) -> None:
        model = self._segment_model
        backend = self._segment_backend
        self._segment_model = None
        self._segment_backend = None
        self._segment_model_path = ""
        try:
            if backend == "RKNN" and model is not None:
                model.release()
        except Exception:
            pass

    def _release_refine_model(self) -> None:
        self._refine_model = None
        self._refine_model_path = ""
        self._refined_cache_source_id = -1
        self._refined_cache_model_path = ""
        self._refined_cache_data = None

    @Slot(object)
    def handle_control(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        action = str(payload.get("action", ""))
        if action == "clear_segment":
            self._release_segment_model()
        elif action == "clear_refine":
            self._release_refine_model()
        elif action == "clear_all":
            self._release_segment_model()
            self._release_refine_model()

    @Slot(object)
    def process(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return

        request_id = int(payload.get("request_id", -1))
        source_id = int(payload.get("source_id", -1))

        try:
            color_image = payload["color_image"]
            depth_raw = payload.get("depth_raw")
            depth_refined_file = payload.get("depth_refined_file")

            show_depth = bool(payload.get("show_depth", True))
            depth_display_mode = str(
                payload.get("depth_display_mode", "固定比例尺显示")
            )

            segment_enabled = bool(payload.get("segment_enabled", True))
            use_semantic_segment = bool(payload.get("use_semantic_segment", False))
            load_segment_model = bool(payload.get("load_segment_model", False))
            segment_model_path = str(payload.get("segment_model_path", "")).strip()

            refine_enabled = bool(payload.get("refine_enabled", False))
            load_refine_model = bool(payload.get("load_refine_model", False))
            refine_model_path = str(payload.get("refine_model_path", "")).strip()

            min_region_area = int(payload.get("min_region_area", 0))
            area_method = AreaMethod(
                str(payload.get("area_method", AreaMethod.DEPTH_CENTER.value))
            )

            h, w = color_image.shape[:2]
            K, calibration = lookup_calibration(self._calibration_cache, w, h)
            if calibration is not None:
                kc, kv = calibration
                set_calibration_params(kc, kv)
                self.status.emit(
                    f"[Import] 使用分辨率标定参数: {w}*{h}, Kc={kc}, Kv={kv}"
                )
            else:
                default_kc, default_kv = get_calibration_params()
                self.status.emit(
                    f"[Import] 未找到分辨率标定参数({w}*{h}); 使用当前 Kc={default_kc}, Kv={default_kv}"
                )

            depth_data = None
            if refine_enabled:
                if load_refine_model:
                    if not refine_model_path:
                        raise ValueError("请填写深度优化模型路径")
                    if (
                        self._refine_model is None
                        or self._refine_model_path != refine_model_path
                    ):
                        self._release_refine_model()
                        self.status.emit(
                            f"[Import] 正在加载深度精细化模型: {refine_model_path}"
                        )
                        self._refine_model = model_loader(refine_model_path, type="MDM")
                        self._refine_model_path = refine_model_path

                    if depth_raw is not None:
                        if (
                            self._refined_cache_data is None
                            or self._refined_cache_source_id != source_id
                            or self._refined_cache_model_path != self._refine_model_path
                        ):
                            self.status.emit("[Import] 正在执行深度优化模型推理")
                            import torch

                            device = torch.device(
                                "cuda" if torch.cuda.is_available() else "cpu"
                            )

                            color_image_rgb = cv2.cvtColor(
                                color_image,
                                cv2.COLOR_BGR2RGB,
                            )
                            image = torch.tensor(
                                color_image_rgb / 255.0,
                                dtype=torch.float32,
                                device=device,
                            ).permute(2, 0, 1)[None]

                            depth_m = depth_raw.astype(np.float32) / 1000.0
                            depth = torch.tensor(
                                depth_m,
                                dtype=torch.float32,
                                device=device,
                            )[None]

                            intrinsics = K.copy().astype(np.float32)
                            intrinsics[0] /= float(w)
                            intrinsics[1] /= float(h)
                            intrinsics = torch.tensor(
                                intrinsics,
                                dtype=torch.float32,
                                device=device,
                            )[None]

                            with torch.no_grad():
                                output = self._refine_model.infer(
                                    image,
                                    depth_in=depth,
                                    intrinsics=intrinsics,
                                    enable_depth_mask=False,
                                )

                            self._refined_cache_data = (
                                output["depth"].squeeze().detach().cpu().numpy()
                                * 1000.0
                            ).astype(np.float32)
                            self._refined_cache_source_id = source_id
                            self._refined_cache_model_path = self._refine_model_path
                        depth_data = self._refined_cache_data
                    elif depth_refined_file is not None:
                        self.status.emit(
                            "[Import] 原始深度缺失，使用 depth_refined.png"
                        )
                        depth_data = depth_refined_file
                elif depth_refined_file is not None:
                    depth_data = depth_refined_file
                elif depth_raw is not None:
                    depth_data = depth_raw
            else:
                if depth_raw is not None:
                    depth_data = depth_raw
                elif depth_refined_file is not None:
                    depth_data = depth_refined_file

            if depth_data is None:
                raise ValueError("导入深度图不可用")

            depth_data = np.clip(depth_data, 0, 65535).astype(np.uint16)
            depth_data = np.where(
                (depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0
            ).astype(np.uint16)

            segment_model = None
            segment_backend = None
            seg_display_name = "OpenCV"
            if segment_enabled and use_semantic_segment and load_segment_model:
                if not segment_model_path:
                    raise ValueError("请填写语义分割模型路径")
                segment_backend = infer_segment_backend(segment_model_path)
                seg_display_name = Path(segment_model_path).name or "语义分割模型"
                if (
                    self._segment_model is None
                    or self._segment_backend != segment_backend
                    or self._segment_model_path != segment_model_path
                ):
                    self._release_segment_model()
                    self.status.emit(
                        f"[Import] 正在加载分割模型({segment_backend}): {segment_model_path}"
                    )
                    self._segment_model = model_loader(
                        segment_model_path, type=segment_backend
                    )
                    self._segment_backend = segment_backend
                    self._segment_model_path = segment_model_path
                segment_model = self._segment_model

            aligned_image = build_aligned_image(
                color_image,
                depth_data,
                show_depth,
                depth_display_mode,
            )

            blended, n_regions, area_results, used_semantic_model = segment_and_measure(
                color_image=color_image,
                depth_data=depth_data,
                K=K,
                area_method=area_method,
                min_region_area=min_region_area,
                segment_enabled=segment_enabled,
                use_semantic_segment=use_semantic_segment,
                segment_model=segment_model,
                segment_backend=segment_backend,
                status_callback=self.status.emit,
            )

            if segment_enabled and use_semantic_segment and not used_semantic_model:
                seg_display_name = "OpenCV"

            self.result_ready.emit(
                {
                    "request_id": request_id,
                    "aligned_image": aligned_image,
                    "seg_image": blended,
                    "n_regions": int(n_regions),
                    "method_name": area_method.value,
                    "seg_display": seg_display_name,
                    "raw_rgb": color_image,
                    "raw_depth": depth_data,
                    "refined_depth": None,
                    "area_results": area_results,
                }
            )
        except Exception as exc:
            self.error.emit(
                {
                    "request_id": request_id,
                    "detail": str(exc),
                }
            )


class CameraWorker(QObject):
    """后台采集与处理线程。"""

    frame_ready = Signal(
        object,
        object,
        float,
        int,
        str,
        str,
        object,
        object,
        object,
        object,
    )
    status = Signal(str)
    error = Signal(str)
    finished = Signal()

    def __init__(
        self,
        calibration_cache: dict[str, dict[str, float]] | None = None,
    ) -> None:
        super().__init__()
        self._calibration_cache = calibration_cache or {}
        self._running = False
        self._options = ProcessingOptions()
        self._lock = Lock()

        self._refine_model = None
        self._segment_model = None
        self._segment_backend = None
        self._segment_display_name = "OpenCV"

    def stop(self) -> None:
        self._running = False

    def set_segment_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._options.segment_image = bool(enabled)

    def set_depth_display_mode(self, mode_name: str) -> None:
        with self._lock:
            self._options.depth_display_mode = str(mode_name)

    def set_show_depth(self, enabled: bool) -> None:
        with self._lock:
            self._options.show_depth = bool(enabled)

    def set_depth_refinement(self, enabled: bool) -> None:
        with self._lock:
            self._options.depth_refinement = bool(enabled)

    def set_load_refine_model(self, enabled: bool) -> None:
        with self._lock:
            self._options.load_refine_model = bool(enabled)

    def set_load_segment_model(self, enabled: bool) -> None:
        with self._lock:
            self._options.load_segment_model = bool(enabled)

    def set_refine_model_path(self, model_path: str) -> None:
        with self._lock:
            self._options.refine_model_path = str(model_path)

    def set_segment_model_path(self, model_path: str) -> None:
        with self._lock:
            self._options.segment_model_path = str(model_path)

    def set_min_region_area(self, area: int) -> None:
        with self._lock:
            self._options.min_region_area = max(0, int(area))

    def set_area_method(self, method_name: str) -> None:
        with self._lock:
            self._options.area_method = AreaMethod(method_name)

    def set_use_semantic_segment(self, enabled: bool) -> None:
        with self._lock:
            self._options.use_semantic_segment = bool(enabled)

    def _get_options_snapshot(self) -> ProcessingOptions:
        with self._lock:
            return ProcessingOptions(
                segment_image=self._options.segment_image,
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
        self._segment_display_name = "OpenCV"

        if opts.load_refine_model:
            refine_path = opts.refine_model_path.strip()
            if not refine_path:
                raise ValueError("已启用深度优化模型，但模型路径为空")
            self.status.emit(f"[Model] 正在加载深度精细化模型: {refine_path}")
            self._refine_model = model_loader(refine_path, type="MDM")
            self.status.emit("[Model] 深度精细化模型加载成功")
        else:
            self.status.emit("[Model] 深度优化模型未启用")

        if opts.load_segment_model:
            seg_path = opts.segment_model_path.strip()
            if not seg_path:
                raise ValueError("已启用语义分割模型，但模型路径为空")
            backend = infer_segment_backend(seg_path)
            self._segment_backend = backend
            self._segment_display_name = Path(seg_path).name or "语义分割模型"
            self.status.emit(f"[Model] 正在加载分割模型({backend}): {seg_path}")
            self._segment_model = model_loader(
                seg_path,
                type=backend,
            )
            self.status.emit(f"[Model] 分割模型({backend})加载成功")
        else:
            self.status.emit("[Model] 语义分割模型未启用，将仅支持 OpenCV 分割")

    @Slot()
    def run(self) -> None:
        pipeline = Pipeline()
        align_filter = None
        prev_time = time.time()
        fps_ema = None
        fps_alpha = 0.15
        self._running = True

        try:
            if HW_DENOISE:
                enable_hw_denoise(pipeline)
                self.status.emit("[Camera] 已启用硬件降噪")

            startup_opts = self._get_options_snapshot()
            self._load_models(startup_opts)

            cfg_result = get_stream_config(pipeline)
            if cfg_result is None:
                raise RuntimeError("无法获取相机流配置")
            depth_profile, color_profile, config = cfg_result

            resolution = None
            try:
                resolution = f"{depth_profile.get_width()}*{depth_profile.get_height()}"
            except Exception:
                resolution = None

            K = None
            if resolution:
                cached_K, calibration = lookup_calibration(
                    self._calibration_cache,
                    int(depth_profile.get_width()),
                    int(depth_profile.get_height()),
                )
                if calibration is not None:
                    kc, kv = calibration
                    set_calibration_params(kc, kv)
                    self.status.emit(
                        f"[Calibration] resolution={resolution}, Kc={kc}, Kv={kv}"
                    )
                    K = cached_K
                else:
                    default_kc, default_kv = get_calibration_params()
                    self.status.emit(
                        f"[Calibration] 未找到 resolution={resolution}; 使用默认 Kc={default_kc}, Kv={default_kv}"
                    )

            if K is None:
                K = get_camera_parameters(color_profile, depth_profile)
            pipeline.start(config)
            self.status.emit("[Camera] Pipeline started")

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

                opts = self._get_options_snapshot()

                seg_display = "OpenCV"
                if opts.use_semantic_segment and opts.load_segment_model:
                    if self._segment_model is not None:
                        seg_display = self._segment_display_name
                    else:
                        seg_display = "OpenCV"

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
                    color_image,
                    depth_data,
                    opts.show_depth,
                    opts.depth_display_mode,
                )

                blended, n_regions, area_results, _ = segment_and_measure(
                    color_image=color_image,
                    depth_data=depth_data,
                    K=K,
                    area_method=opts.area_method,
                    min_region_area=opts.min_region_area,
                    segment_enabled=opts.segment_image,
                    use_semantic_segment=opts.use_semantic_segment,
                    segment_model=self._segment_model,
                    segment_backend=self._segment_backend,
                    status_callback=self.status.emit,
                )

                curr_time = time.time()
                inst_fps = 1.0 / max(curr_time - prev_time, 1e-6)
                prev_time = curr_time

                if fps_ema is None:
                    fps_ema = inst_fps
                else:
                    fps_ema = fps_alpha * inst_fps + (1.0 - fps_alpha) * fps_ema
                fps = fps_ema

                self.frame_ready.emit(
                    aligned_image,
                    blended,
                    float(fps),
                    int(n_regions),
                    opts.area_method.value,
                    seg_display,
                    color_image,
                    raw_depth_data,
                    refined_depth_data,
                    area_results,
                )

        except Exception as exc:
            detail = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
            self.error.emit(detail)
        finally:
            try:
                pipeline.stop()
            except Exception:
                pass
            try:
                if self._segment_backend == "RKNN" and self._segment_model is not None:
                    self._segment_model.release()
            except Exception:
                pass
            self.status.emit("[Camera] Pipeline stopped")
            self.finished.emit()


class CoatingWidget(QWidget):
    """涂层损坏面积量算：图像显示 + 参数控制（可嵌入向导页）。"""

    import_process_requested = Signal(object)
    import_control_requested = Signal(object)

    IMPORT_REPROCESS_DEBOUNCE_MS = 180

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._calibration_cache = load_calibration_cache()

        self.worker_thread: QThread | None = None
        self.worker: CameraWorker | None = None

        self.import_thread = QThread(self)
        self.import_worker = ImportProcessingWorker(self._calibration_cache)
        self.import_worker.moveToThread(self.import_thread)
        self.import_process_requested.connect(self.import_worker.process)
        self.import_control_requested.connect(self.import_worker.handle_control)
        self.import_worker.result_ready.connect(self.on_import_process_result)
        self.import_worker.status.connect(self._append_log)
        self.import_worker.error.connect(self.on_import_process_error)
        self.import_thread.start()

        self._latest_raw_rgb: np.ndarray | None = None
        self._latest_raw_depth: np.ndarray | None = None
        self._latest_refined_depth: np.ndarray | None = None
        self._latest_seg_image: np.ndarray | None = None
        self._latest_area_results: list[tuple[int, float]] = []
        self._latest_n_regions: int = 0
        self._latest_area_method: str = ""
        self._current_source_name: str | None = None

        self._import_rgb_image: np.ndarray | None = None
        self._import_depth_raw: np.ndarray | None = None
        self._import_depth_refined: np.ndarray | None = None
        self._import_request_seq: int = 0
        self._latest_import_request_id: int = 0
        self._import_source_id: int = 0

        # 图像显示缓存：仅在控件尺寸或源图尺寸变化时重算缩放目标尺寸
        self._aligned_view_dirty = True
        self._seg_view_dirty = True
        self._aligned_label_size = QSize()
        self._seg_label_size = QSize()
        self._aligned_src_size: tuple[int, int] | None = None
        self._seg_src_size: tuple[int, int] | None = None
        self._aligned_target_size = QSize()
        self._seg_target_size = QSize()

        self._import_reprocess_timer = QTimer(self)
        self._import_reprocess_timer.setSingleShot(True)
        self._import_reprocess_timer.setInterval(self.IMPORT_REPROCESS_DEBOUNCE_MS)
        self._import_reprocess_timer.timeout.connect(self._on_import_reprocess_timeout)

        self._build_ui()

    def _build_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # 左侧图像区
        viewer_layout = QVBoxLayout()

        self.aligned_label = QLabel("Align Viewer")
        self.aligned_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.aligned_label.setMinimumSize(640, 360)
        self.aligned_label.setStyleSheet(
            "background:#111; color:#ddd; border:1px solid #333;"
        )

        self.segment_label = QLabel("Segmentation Regions")
        self.segment_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.segment_label.setMinimumSize(640, 360)
        self.segment_label.setStyleSheet(
            "background:#111; color:#ddd; border:1px solid #333;"
        )

        viewer_layout.addWidget(self.aligned_label, stretch=1)
        viewer_layout.addWidget(self.segment_label, stretch=1)

        # 右侧控制区
        panel_layout = QVBoxLayout()

        run_box = QGroupBox("运行控制")
        run_form = QFormLayout(run_box)
        self.btn_start = QPushButton("启动相机")
        self.btn_stop = QPushButton("停止")
        self.btn_save = QPushButton("保存当前帧")
        self.btn_import = QPushButton("离线导入 RGB-D")
        self.btn_export_json = QPushButton("导出 JSON 报告")
        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_import.setEnabled(True)
        self.btn_export_json.setEnabled(False)
        run_btn_row = QWidget()
        run_btn_layout = QHBoxLayout(run_btn_row)
        run_btn_layout.setContentsMargins(0, 0, 0, 0)
        run_btn_layout.addWidget(self.btn_start)
        run_btn_layout.addWidget(self.btn_stop)
        run_btn_layout.addWidget(self.btn_save)
        run_btn_layout.addWidget(self.btn_import)
        run_btn_layout.addWidget(self.btn_export_json)
        run_form.addRow(run_btn_row)

        model_group = QGroupBox("模型加载")
        model_form = QFormLayout(model_group)

        self.chk_load_refine_model = QCheckBox("加载深度优化模型")
        self.chk_load_refine_model.setChecked(GUI_DEFAULT_LOAD_REFINE_MODEL)
        model_form.addRow(self.chk_load_refine_model)

        refine_path_row = QWidget()
        refine_path_layout = QHBoxLayout(refine_path_row)
        refine_path_layout.setContentsMargins(0, 0, 0, 0)
        self.edt_refine_model_path = QLineEdit(GUI_DEFAULT_REFINE_MODEL_PATH)
        self.btn_browse_refine_model = QPushButton("浏览")
        refine_path_layout.addWidget(self.edt_refine_model_path, stretch=1)
        refine_path_layout.addWidget(self.btn_browse_refine_model)
        model_form.addRow("深度优化模型路径", refine_path_row)

        self.chk_load_segment_model = QCheckBox("加载语义分割模型")
        self.chk_load_segment_model.setChecked(GUI_DEFAULT_LOAD_SEGMENT_MODEL)
        model_form.addRow(self.chk_load_segment_model)

        segment_path_row = QWidget()
        segment_path_layout = QHBoxLayout(segment_path_row)
        segment_path_layout.setContentsMargins(0, 0, 0, 0)
        self.edt_segment_model_path = QLineEdit(GUI_DEFAULT_SEGMENT_MODEL_PATH)
        self.btn_browse_segment_model = QPushButton("浏览")
        segment_path_layout.addWidget(self.edt_segment_model_path, stretch=1)
        segment_path_layout.addWidget(self.btn_browse_segment_model)
        model_form.addRow("语义分割模型路径", segment_path_row)

        run_form.addRow(model_group)
        panel_layout.addWidget(run_box)

        option_box = QGroupBox("实时参数")
        option_form = QFormLayout(option_box)

        self.chk_show_depth = QCheckBox("显示深度")
        self.chk_show_depth.setChecked(True)
        option_form.addRow(self.chk_show_depth)

        self.chk_segment = QCheckBox("显示分割面积")
        self.chk_segment.setChecked(RuntimeConfig.segment_image)
        option_form.addRow(self.chk_segment)

        self.cmb_depth_display = QComboBox()
        self.cmb_depth_display.addItems(["固定比例尺显示", "自适应比例尺显示"])
        self.cmb_depth_display.setCurrentText(
            "固定比例尺显示"
            if RuntimeConfig.use_fixed_depth_scale
            else "自适应比例尺显示"
        )
        option_form.addRow("深度显示方式", self.cmb_depth_display)

        self.chk_refine = QCheckBox("启用深度优化")
        self.chk_refine.setChecked(RuntimeConfig.depth_refinement)
        if not GUI_DEFAULT_LOAD_REFINE_MODEL:
            self.chk_refine.setEnabled(False)
            self.chk_refine.setToolTip("当前配置未加载精细化模型")
        option_form.addRow(self.chk_refine)

        self.chk_use_semantic_segment = QCheckBox("启用语义分割模型")
        self.chk_use_semantic_segment.setChecked(GUI_DEFAULT_USE_SEMANTIC_SEGMENT)
        option_form.addRow(self.chk_use_semantic_segment)

        self.cmb_area_method = QComboBox()
        self.cmb_area_method.addItems(
            [
                AreaMethod.RANSAC.value,
                AreaMethod.DEPTH_CENTER.value,
                AreaMethod.AUTO.value,
            ]
        )
        default_area_method = RuntimeConfig.area_method.value
        if default_area_method not in {
            AreaMethod.RANSAC.value,
            AreaMethod.DEPTH_CENTER.value,
            AreaMethod.AUTO.value,
        }:
            default_area_method = AreaMethod.DEPTH_CENTER.value
        self.cmb_area_method.setCurrentText(default_area_method)
        option_form.addRow("面积计算算法", self.cmb_area_method)

        min_area_row = QWidget()
        min_area_layout = QHBoxLayout(min_area_row)
        min_area_layout.setContentsMargins(0, 0, 0, 0)

        self.slider_min_area = QSlider(Qt.Orientation.Horizontal)
        self.slider_min_area.setRange(0, 10000)
        self.slider_min_area.setSingleStep(100)
        self.slider_min_area.setPageStep(200)
        self.slider_min_area.setValue(int(RuntimeConfig.min_region_area))

        self.spin_min_area = QSpinBox()
        self.spin_min_area.setRange(0, 10000)
        self.spin_min_area.setSingleStep(100)
        self.spin_min_area.setValue(int(RuntimeConfig.min_region_area))

        min_area_layout.addWidget(self.slider_min_area, stretch=1)
        min_area_layout.addWidget(self.spin_min_area)
        option_form.addRow("连通域面积阈值", min_area_row)

        panel_layout.addWidget(option_box)

        status_box = QGroupBox("运行状态")
        status_form = QFormLayout(status_box)
        self.lbl_fps = QLabel("0.0")
        self.lbl_regions = QLabel("0")
        self.lbl_method = QLabel(RuntimeConfig.area_method.value)
        self.lbl_seg_method = QLabel("OpenCV")
        status_form.addRow("FPS", self.lbl_fps)
        status_form.addRow("区域数", self.lbl_regions)
        status_form.addRow("面积算法", self.lbl_method)
        status_form.addRow("分割方法", self.lbl_seg_method)
        panel_layout.addWidget(status_box)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("日志输出...")
        panel_layout.addWidget(self.log_output, stretch=1)

        main_layout.addLayout(viewer_layout, stretch=3)
        main_layout.addLayout(panel_layout, stretch=2)

        # 信号连接
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_save.clicked.connect(self.on_save)
        self.btn_import.clicked.connect(self.on_import)
        self.btn_export_json.clicked.connect(self.on_export_json)
        self.btn_browse_refine_model.clicked.connect(self.on_browse_refine_model)
        self.btn_browse_segment_model.clicked.connect(self.on_browse_segment_model)
        self.chk_load_refine_model.toggled.connect(self.on_load_refine_model_toggled)
        self.chk_load_segment_model.toggled.connect(self.on_load_segment_model_toggled)
        self.edt_refine_model_path.editingFinished.connect(
            self.on_refine_model_path_edit_finished
        )
        self.edt_segment_model_path.editingFinished.connect(
            self.on_segment_model_path_edit_finished
        )

        self.chk_segment.toggled.connect(self.on_segment_toggled)
        self.chk_show_depth.toggled.connect(self.on_show_depth_toggled)
        self.cmb_depth_display.currentTextChanged.connect(
            self.on_depth_display_mode_changed
        )
        self.chk_refine.toggled.connect(self.on_refine_toggled)
        self.chk_use_semantic_segment.toggled.connect(
            self.on_use_semantic_segment_toggled
        )
        self.cmb_area_method.currentTextChanged.connect(self.on_area_method_changed)

        self.slider_min_area.valueChanged.connect(self.spin_min_area.setValue)
        self.spin_min_area.valueChanged.connect(self.slider_min_area.setValue)
        self.spin_min_area.valueChanged.connect(self.on_min_area_changed)

        self._sync_model_control_states()

    def _set_model_controls_enabled(self, enabled: bool) -> None:
        self.chk_load_refine_model.setEnabled(enabled)
        self.edt_refine_model_path.setEnabled(enabled)
        self.btn_browse_refine_model.setEnabled(enabled)
        self.chk_load_segment_model.setEnabled(enabled)
        self.edt_segment_model_path.setEnabled(enabled)
        self.btn_browse_segment_model.setEnabled(enabled)

    def _sync_model_control_states(self) -> None:
        refine_on = self.chk_load_refine_model.isChecked()
        self.edt_refine_model_path.setEnabled(
            refine_on and self.chk_load_refine_model.isEnabled()
        )
        self.btn_browse_refine_model.setEnabled(
            refine_on and self.chk_load_refine_model.isEnabled()
        )

        segment_on = self.chk_load_segment_model.isChecked()
        self.edt_segment_model_path.setEnabled(
            segment_on and self.chk_load_segment_model.isEnabled()
        )
        self.btn_browse_segment_model.setEnabled(
            segment_on and self.chk_load_segment_model.isEnabled()
        )

        self.chk_refine.setEnabled(refine_on)
        if not refine_on:
            self.chk_refine.setChecked(False)

        self.chk_use_semantic_segment.setEnabled(segment_on)
        if not segment_on:
            self.chk_use_semantic_segment.setChecked(False)

        self.cmb_depth_display.setEnabled(self.chk_show_depth.isChecked())

    def _append_log(self, text: str) -> None:
        self.log_output.appendPlainText(text)
        sb = self.log_output.verticalScrollBar()
        sb.setValue(sb.maximum())

    @staticmethod
    def _to_uint16_depth(depth_data: np.ndarray) -> np.ndarray:
        """将深度图转为 16-bit 毫米图，便于无渲染保存。"""
        if depth_data.dtype == np.uint16:
            return depth_data
        return np.clip(depth_data, 0, 65535).astype(np.uint16)

    def _release_import_segment_model(self) -> None:
        self.import_control_requested.emit({"action": "clear_segment"})

    def _release_import_refine_model(self) -> None:
        self.import_control_requested.emit({"action": "clear_refine"})

    def _do_reprocess_imported_frame(self) -> None:
        """提交导入态重算任务到后台线程。"""
        if self.worker_thread is not None:
            return
        if self._import_rgb_image is None:
            return

        self._import_request_seq += 1
        request_id = self._import_request_seq
        self._latest_import_request_id = request_id

        self.import_process_requested.emit(
            {
                "request_id": request_id,
                "source_id": self._import_source_id,
                "color_image": self._import_rgb_image,
                "depth_raw": (
                    self._import_depth_raw
                    if self._import_depth_raw is not None
                    else None
                ),
                "depth_refined_file": (
                    self._import_depth_refined
                    if self._import_depth_refined is not None
                    else None
                ),
                "show_depth": self.chk_show_depth.isChecked(),
                "depth_display_mode": self.cmb_depth_display.currentText(),
                "segment_enabled": self.chk_segment.isChecked(),
                "use_semantic_segment": self.chk_use_semantic_segment.isChecked(),
                "load_segment_model": self.chk_load_segment_model.isChecked(),
                "segment_model_path": self.edt_segment_model_path.text().strip(),
                "refine_enabled": self.chk_refine.isChecked(),
                "load_refine_model": self.chk_load_refine_model.isChecked(),
                "refine_model_path": self.edt_refine_model_path.text().strip(),
                "min_region_area": self.spin_min_area.value(),
                "area_method": self.cmb_area_method.currentText(),
            }
        )

    @Slot()
    def _on_import_reprocess_timeout(self) -> None:
        self._do_reprocess_imported_frame()

    def _reprocess_imported_frame(self, immediate: bool = False) -> None:
        """当导入态参数变化时重算；默认防抖，必要时可立即执行。"""
        if self.worker_thread is not None:
            return
        if self._import_rgb_image is None:
            return

        if immediate:
            self._do_reprocess_imported_frame()
            return

        self._import_reprocess_timer.start()

    def _handle_result_param_change(
        self,
        worker_apply=None,
        release_import_segment: bool = False,
        release_import_refine: bool = False,
    ) -> None:
        """统一处理影响结果的参数变更：运行态更新 worker，导入态重算。"""
        if self.worker is not None:
            if worker_apply is not None:
                worker_apply(self.worker)
            return

        if release_import_segment:
            self._release_import_segment_model()
        if release_import_refine:
            self._release_import_refine_model()
        self._reprocess_imported_frame()

    @staticmethod
    def _pick_depth_file(rgb_path: Path) -> Path | None:
        """在 RGB 同目录按优先级寻找深度图文件。"""
        parent = rgb_path.parent
        for name in ("depth_refined.png", "depth_raw.png", "rgb_raw.png"):
            candidate = parent / name
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _load_depth_file(
        path: Path, target_shape: tuple[int, int]
    ) -> np.ndarray | None:
        """加载深度图并对齐到目标分辨率。

        兼容以下四种常见文件形态：
        - 单通道 uint16/uint8 深度图（H, W）
        - 单通道被 imread 展平为 3D 的情况（H, W, 1）
        - 三通道 BGR 图（H, W, 3）——深度可能编码在 R/G/B 中，直接转灰度
        - 四通道 BGRA 图（H, W, 4）——去掉 alpha 后转灰度
        """
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            return None

        if depth.ndim == 3:
            channels = depth.shape[2]
            if channels == 1:
                depth = depth[:, :, 0]
            elif channels == 3:
                depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
            elif channels == 4:
                depth = cv2.cvtColor(depth, cv2.COLOR_BGRA2GRAY)
            else:
                # 未知通道数，取第一个通道兜底
                depth = depth[:, :, 0]

        target_h, target_w = target_shape
        if depth.shape[:2] != (target_h, target_w):
            depth = cv2.resize(
                depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST
            )
        return depth

    @staticmethod
    def _is_camera_connected() -> bool:
        """检测是否存在已连接的相机设备。"""
        try:
            ctx = Context()
            device_list = ctx.query_devices()
            return len(device_list) > 0
        except Exception:
            return False

    @staticmethod
    def _compute_aspect_fit_size(label_size: QSize, src_w: int, src_h: int) -> QSize:
        """根据控件尺寸和源图尺寸计算等比例适配目标尺寸。"""
        if src_w <= 0 or src_h <= 0:
            return QSize(1, 1)

        lw = max(1, int(label_size.width()))
        lh = max(1, int(label_size.height()))
        scale = min(lw / src_w, lh / src_h)

        tw = max(1, int(src_w * scale))
        th = max(1, int(src_h * scale))
        return QSize(tw, th)

    def _update_view_target_sizes(
        self,
        aligned_image: np.ndarray,
        seg_image: np.ndarray,
    ) -> None:
        """仅在尺寸变化时更新缩放目标，避免逐帧重算比例。"""
        ah, aw = aligned_image.shape[:2]
        sh, sw = seg_image.shape[:2]

        aligned_label_size = self.aligned_label.size()
        seg_label_size = self.segment_label.size()

        aligned_src = (aw, ah)
        seg_src = (sw, sh)

        if (
            self._aligned_view_dirty
            or aligned_label_size != self._aligned_label_size
            or aligned_src != self._aligned_src_size
        ):
            self._aligned_target_size = self._compute_aspect_fit_size(
                aligned_label_size, aw, ah
            )
            self._aligned_label_size = QSize(aligned_label_size)
            self._aligned_src_size = aligned_src
            self._aligned_view_dirty = False

        if (
            self._seg_view_dirty
            or seg_label_size != self._seg_label_size
            or seg_src != self._seg_src_size
        ):
            self._seg_target_size = self._compute_aspect_fit_size(
                seg_label_size, sw, sh
            )
            self._seg_label_size = QSize(seg_label_size)
            self._seg_src_size = seg_src
            self._seg_view_dirty = False

    def _apply_current_controls_to_worker(self) -> None:
        if self.worker is None:
            return
        self.worker.set_segment_enabled(self.chk_segment.isChecked())
        self.worker.set_show_depth(self.chk_show_depth.isChecked())
        self.worker.set_depth_display_mode(self.cmb_depth_display.currentText())
        self.worker.set_depth_refinement(self.chk_refine.isChecked())
        self.worker.set_use_semantic_segment(self.chk_use_semantic_segment.isChecked())
        self.worker.set_area_method(self.cmb_area_method.currentText())
        self.worker.set_min_region_area(self.spin_min_area.value())
        self.worker.set_load_refine_model(self.chk_load_refine_model.isChecked())
        self.worker.set_load_segment_model(self.chk_load_segment_model.isChecked())
        self.worker.set_refine_model_path(self.edt_refine_model_path.text().strip())
        self.worker.set_segment_model_path(self.edt_segment_model_path.text().strip())

    @Slot()
    def on_start(self) -> None:
        if self.worker_thread is not None:
            return

        if not self._is_camera_connected():
            self._append_log("[Camera] 未检测到相机，请检查相机连接")
            QMessageBox.warning(self, "未检测到相机", "请检查相机连接")
            return

        if (
            self.chk_load_refine_model.isChecked()
            and not self.edt_refine_model_path.text().strip()
        ):
            QMessageBox.warning(self, "模型路径为空", "请填写深度优化模型路径")
            return
        if (
            self.chk_load_segment_model.isChecked()
            and not self.edt_segment_model_path.text().strip()
        ):
            QMessageBox.warning(self, "模型路径为空", "请填写语义分割模型路径")
            return
        if self.chk_load_segment_model.isChecked():
            seg_path = self.edt_segment_model_path.text().strip().lower()
            try:
                infer_segment_backend(seg_path)
            except ValueError:
                QMessageBox.warning(
                    self,
                    "模型后缀不支持",
                    "语义分割模型仅支持 .onnx 或 .rknn 文件",
                )
                return

        self.worker_thread = QThread(self)
        self.worker = CameraWorker(self._calibration_cache)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.status.connect(self.on_worker_status)
        self.worker.error.connect(self.on_worker_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self._apply_current_controls_to_worker()
        self._set_model_controls_enabled(False)
        self._sync_model_control_states()
        self._release_import_segment_model()
        self._release_import_refine_model()
        self._import_reprocess_timer.stop()
        self._current_source_name = None
        self.worker_thread.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.btn_import.setEnabled(False)
        self.btn_export_json.setEnabled(False)
        self._append_log("[UI] 后台线程已启动")

    @Slot()
    def on_stop(self) -> None:
        if self.worker is not None:
            self.worker.stop()
        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_import.setEnabled(False)
        self._append_log("[UI] 请求停止")

    @Slot()
    def on_worker_finished(self) -> None:
        self._append_log("[UI] 后台线程已停止")

        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

        self.worker_thread = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_import.setEnabled(True)
        self._set_model_controls_enabled(True)
        self._sync_model_control_states()

    @Slot()
    def on_save(self) -> None:
        if self.worker_thread is None:
            QMessageBox.information(self, "保存失败", "相机未运行，无法保存")
            return

        if self._latest_raw_rgb is None or self._latest_raw_depth is None:
            QMessageBox.information(self, "保存失败", "当前暂无可保存帧，请稍后再试")
            return

        save_root = Path("output") / "saved_frames"
        save_root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        millis = int((time.time() % 1) * 1000)
        save_dir = save_root / f"{stamp}_{millis:03d}"
        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            rgb_path = save_dir / "rgb_raw.png"
            depth_raw_path = save_dir / "depth_raw.png"

            cv2.imwrite(str(rgb_path), self._latest_raw_rgb)
            cv2.imwrite(
                str(depth_raw_path),
                self._to_uint16_depth(self._latest_raw_depth),
            )

            saved_items = [str(rgb_path), str(depth_raw_path)]

            if self._latest_refined_depth is not None:
                depth_refined_path = save_dir / "depth_refined.png"
                cv2.imwrite(
                    str(depth_refined_path),
                    self._to_uint16_depth(self._latest_refined_depth),
                )
                saved_items.append(str(depth_refined_path))

            if self._latest_n_regions > 0 and self._latest_seg_image is not None:
                seg_path = save_dir / "segmentation_regions.png"
                cv2.imwrite(str(seg_path), self._latest_seg_image)
                saved_items.append(str(seg_path))

                csv_path = save_dir / "area_results.csv"
                with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    writer.writerow(["region_id", "area_cm2"])
                    for region_id, area_cm2 in self._latest_area_results:
                        writer.writerow([region_id, f"{area_cm2:.6f}"])
                saved_items.append(str(csv_path))

                json_path = save_dir / "area_report.json"
                report = self._build_area_report(source_name=f"{stamp}_{millis:03d}")
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                saved_items.append(str(json_path))

            self._append_log("[Save] 保存成功:")
            for item in saved_items:
                self._append_log(f"[Save] {item}")
            QMessageBox.information(self, "保存成功", f"已保存到目录:\n{save_dir}")
        except Exception as exc:
            self._append_log(f"[Save] 保存失败: {exc}")
            QMessageBox.critical(self, "保存失败", str(exc))

    def _build_area_report(self, source_name: str | None = None) -> dict:
        """构造涂层损坏面积检测的结构化 JSON 报告。"""
        regions = [
            {"region_id": int(rid), "area_cm2": round(float(area), 6)}
            for rid, area in self._latest_area_results
        ]
        total_area_cm2 = float(sum(a for _, a in self._latest_area_results))
        name = source_name or self._current_source_name or time.strftime(
            "%Y%m%d_%H%M%S"
        )
        return {
            "module": "coating_damage",
            "image": name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "area_method": self._latest_area_method,
            "unit": "cm^2",
            "n_regions": int(self._latest_n_regions),
            "regions": regions,
            "total_area_cm2": round(total_area_cm2, 6),
        }

    @Slot()
    def on_export_json(self) -> None:
        if self._latest_n_regions <= 0 or not self._latest_area_results:
            QMessageBox.information(
                self, "导出失败", "当前无可导出的面积结果，请先完成一次分割"
            )
            return

        default_name = (
            f"{Path(self._current_source_name).stem}_area_report.json"
            if self._current_source_name
            else f"area_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        save_dir = Path("output") / "reports"
        save_dir.mkdir(parents=True, exist_ok=True)
        default_path = str(save_dir / default_name)

        target_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出 JSON 报告",
            default_path,
            "JSON Files (*.json)",
        )
        if not target_path:
            return

        try:
            report = self._build_area_report()
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            self._append_log(f"[Export] JSON 报告已导出: {target_path}")
            QMessageBox.information(
                self,
                "导出成功",
                f"已导出到:\n{target_path}\n\n"
                f"共 {report['n_regions']} 处损坏，总面积 "
                f"{report['total_area_cm2']:.2f} cm²",
            )
        except Exception as exc:
            self._append_log(f"[Export] 导出失败: {exc}")
            QMessageBox.critical(self, "导出失败", str(exc))

    @Slot()
    def on_import(self) -> None:
        if self.worker_thread is not None:
            QMessageBox.information(self, "导入不可用", "相机运行中无法导入，请先停止")
            return

        start_dir = str(Path.cwd())
        rgb_path_str, _ = QFileDialog.getOpenFileName(
            self,
            "选择 RGB 图像",
            start_dir,
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not rgb_path_str:
            return

        rgb_path = Path(rgb_path_str)
        depth_path = self._pick_depth_file(rgb_path)
        if depth_path is None:
            QMessageBox.warning(
                self,
                "未找到深度图",
                "同目录未找到可用深度图，查找顺序: depth_refined.png -> depth_raw.png -> rgb_raw.png",
            )
            return

        color_image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if color_image is None:
            QMessageBox.warning(self, "导入失败", "RGB 图像读取失败")
            return

        target_shape = color_image.shape[:2]
        depth_refined_path = rgb_path.parent / "depth_refined.png"
        depth_raw_path = None
        for name in ("depth_raw.png", "rgb_raw.png"):
            candidate = rgb_path.parent / name
            if candidate.exists() and candidate.is_file():
                depth_raw_path = candidate
                break

        self._import_rgb_image = color_image
        self._import_depth_refined = (
            self._load_depth_file(depth_refined_path, target_shape)
            if depth_refined_path.exists()
            else None
        )
        self._import_depth_raw = (
            self._load_depth_file(depth_raw_path, target_shape)
            if depth_raw_path is not None
            else None
        )
        self._import_source_id += 1
        self._current_source_name = rgb_path.name

        if self._import_depth_raw is None and self._import_depth_refined is None:
            QMessageBox.warning(self, "导入失败", f"深度图读取失败: {depth_path}")
            return

        if self._import_depth_raw is None and depth_path is not None:
            fallback_depth = self._load_depth_file(depth_path, target_shape)
            if fallback_depth is not None:
                self._import_depth_raw = fallback_depth

        try:
            self._reprocess_imported_frame(immediate=True)
            chosen_depth = (
                "depth_refined.png"
                if self.chk_refine.isChecked()
                and self._import_depth_refined is not None
                else (
                    "depth_raw.png/rgb_raw.png"
                    if self._import_depth_raw is not None
                    else "depth_refined.png"
                )
            )
            self._append_log(
                f"[Import] 导入完成: RGB={rgb_path}, DepthSource={chosen_depth}"
            )
        except Exception as exc:
            self._append_log(f"[Import] 处理失败: {exc}")
            QMessageBox.critical(self, "导入处理失败", str(exc))

    @Slot(str)
    def on_worker_status(self, text: str) -> None:
        self._append_log(text)

    @Slot(str)
    def on_worker_error(self, detail: str) -> None:
        self._append_log("[Error] 处理线程异常")
        self._append_log(detail)
        QMessageBox.critical(self, "处理线程异常", detail)

    @Slot(object)
    def on_import_process_result(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        request_id = int(payload.get("request_id", -1))
        if request_id != self._latest_import_request_id:
            return
        if self.worker_thread is not None:
            return

        self.on_frame_ready(
            payload["aligned_image"],
            payload["seg_image"],
            0.0,
            int(payload["n_regions"]),
            str(payload["method_name"]),
            str(payload["seg_display"]),
            payload["raw_rgb"],
            payload["raw_depth"],
            payload.get("refined_depth"),
            payload["area_results"],
        )
        # self._append_log("[Import] 参数变更，已自动重算")

    @Slot(object)
    def on_import_process_error(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        request_id = int(payload.get("request_id", -1))
        if request_id != self._latest_import_request_id:
            return
        detail = str(payload.get("detail", "导入处理失败"))
        self._append_log(f"[Import] 参数变更后重算失败: {detail}")
        QMessageBox.warning(self, "导入重算失败", detail)

    @Slot(object, object, float, int, str, str, object, object, object, object)
    def on_frame_ready(
        self,
        aligned_image: np.ndarray,
        seg_image: np.ndarray,
        fps: float,
        n_regions: int,
        method_name: str,
        seg_display: str,
        raw_rgb: np.ndarray,
        raw_depth: np.ndarray,
        refined_depth: np.ndarray | None,
        area_results: list[tuple[int, float]],
    ) -> None:
        self._update_view_target_sizes(aligned_image, seg_image)

        self._latest_raw_rgb = raw_rgb
        self._latest_raw_depth = raw_depth
        self._latest_refined_depth = refined_depth
        self._latest_seg_image = seg_image
        self._latest_area_results = list(area_results)
        self._latest_n_regions = int(n_regions)
        self._latest_area_method = str(method_name)
        self.btn_export_json.setEnabled(
            self._latest_n_regions > 0 and bool(self._latest_area_results)
        )

        aligned_pix = bgr_to_qpixmap(aligned_image, self._aligned_target_size)
        seg_pix = bgr_to_qpixmap(seg_image, self._seg_target_size)

        if not aligned_pix.isNull():
            self.aligned_label.setPixmap(aligned_pix)

        if not seg_pix.isNull():
            self.segment_label.setPixmap(seg_pix)

        self.lbl_fps.setText(f"{fps:.1f}")
        self.lbl_regions.setText(str(n_regions))
        self.lbl_method.setText(method_name)
        self.lbl_seg_method.setText(seg_display)

    @Slot(bool)
    def on_segment_toggled(self, enabled: bool) -> None:
        self._handle_result_param_change(
            worker_apply=lambda worker: worker.set_segment_enabled(enabled)
        )

    @Slot(bool)
    def on_show_depth_toggled(self, enabled: bool) -> None:
        self.cmb_depth_display.setEnabled(enabled)
        self._handle_result_param_change(
            worker_apply=lambda worker: worker.set_show_depth(enabled)
        )

    @Slot(bool)
    def on_load_refine_model_toggled(self, enabled: bool) -> None:
        self._sync_model_control_states()
        self._handle_result_param_change(
            worker_apply=lambda worker: worker.set_load_refine_model(enabled),
            release_import_refine=not enabled,
        )

    @Slot(bool)
    def on_load_segment_model_toggled(self, enabled: bool) -> None:
        self._sync_model_control_states()
        self._handle_result_param_change(
            worker_apply=lambda worker: worker.set_load_segment_model(enabled),
            release_import_segment=not enabled,
        )

    @Slot()
    def on_refine_model_path_edit_finished(self) -> None:
        model_path = self.edt_refine_model_path.text().strip()
        self._handle_result_param_change(
            worker_apply=lambda worker: worker.set_refine_model_path(model_path),
            release_import_refine=True,
        )

    @Slot()
    def on_segment_model_path_edit_finished(self) -> None:
        model_path = self.edt_segment_model_path.text().strip()
        self._handle_result_param_change(
            worker_apply=lambda worker: worker.set_segment_model_path(model_path),
            release_import_segment=True,
        )

    @Slot()
    def on_browse_refine_model(self) -> None:
        start_dir = self.edt_refine_model_path.text().strip() or str(Path.cwd())
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择深度优化模型",
            start_dir,
            "Model Files (*.pt *.pth);;All Files (*)",
        )
        if file_path:
            self.edt_refine_model_path.setText(file_path)
            self._handle_result_param_change(
                worker_apply=lambda worker: worker.set_refine_model_path(file_path),
                release_import_refine=True,
            )

    @Slot()
    def on_browse_segment_model(self) -> None:
        start_dir = self.edt_segment_model_path.text().strip() or str(Path.cwd())
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择语义分割模型",
            start_dir,
            "Model Files (*.onnx *.rknn);;All Files (*)",
        )
        if file_path:
            self.edt_segment_model_path.setText(file_path)
            self._handle_result_param_change(
                worker_apply=lambda worker: worker.set_segment_model_path(file_path),
                release_import_segment=True,
            )

    @Slot(str)
    def on_depth_display_mode_changed(self, mode_name: str) -> None:
        self._handle_result_param_change(
            worker_apply=lambda worker: worker.set_depth_display_mode(mode_name)
        )

    @Slot(bool)
    def on_refine_toggled(self, enabled: bool) -> None:
        self._handle_result_param_change(
            worker_apply=lambda worker: worker.set_depth_refinement(enabled)
        )

    @Slot(bool)
    def on_use_semantic_segment_toggled(self, enabled: bool) -> None:
        self._handle_result_param_change(
            worker_apply=lambda worker: worker.set_use_semantic_segment(enabled),
            release_import_segment=not enabled,
        )

    @Slot(str)
    def on_area_method_changed(self, method_name: str) -> None:
        self.lbl_method.setText(method_name)
        self._handle_result_param_change(
            worker_apply=lambda worker: worker.set_area_method(method_name)
        )

    @Slot(int)
    def on_min_area_changed(self, area: int) -> None:
        self._handle_result_param_change(
            worker_apply=lambda worker: worker.set_min_region_area(area)
        )

    def resizeEvent(self, event) -> None:
        # 仅在窗口尺寸变化时将缩放目标标记为需要重算
        self._aligned_view_dirty = True
        self._seg_view_dirty = True
        super().resizeEvent(event)

    def shutdown(self) -> None:
        """由父页面在离开检测页或应用退出时调用，安全释放线程与模型。"""
        self._release_import_segment_model()
        self._release_import_refine_model()
        self.import_control_requested.emit({"action": "clear_all"})
        self.import_thread.quit()
        self.import_thread.wait(2000)
        self.on_stop()
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait(2000)


def main() -> None:
    """本文件独立运行时的入口（调试用）。"""
    app = QApplication(sys.argv)
    window = CoatingWidget()
    window.setWindowTitle("涂层损坏面积量算（独立调试）")
    window.resize(1400, 820)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
