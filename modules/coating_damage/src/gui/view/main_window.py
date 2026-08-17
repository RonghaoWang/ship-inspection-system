from __future__ import annotations

from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from config import RuntimeConfig, AreaMethod

try:
    from PySide6.QtCore import QSize, Qt, Signal, Slot
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QPushButton,
        QPlainTextEdit,
        QSlider,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:
    raise SystemExit("PySide6 未安装，请先执行: pip install PySide6") from exc


GUI_DEFAULT_LOAD_REFINE_MODEL = False
GUI_DEFAULT_LOAD_SEGMENT_MODEL = False
GUI_DEFAULT_USE_SEMANTIC_SEGMENT = False
GUI_DEFAULT_REFINE_MODEL_PATH = "models/lingbot-depth-pretrain-vitl-14-v0.5.pt"
GUI_DEFAULT_SEGMENT_MODEL_PATH = "models/segformer_onnx_512x512.onnx"

AREA_METHOD_DISPLAY_NAMES = {
    AreaMethod.DEPTH_CENTER.value: "深度重心法",
    AreaMethod.RANSAC.value: "平面投影法",
    AreaMethod.AUTO.value: "自动",
}


def area_method_display_name(method_name: str) -> str:
    return AREA_METHOD_DISPLAY_NAMES.get(str(method_name), str(method_name))


class RunUiState(str, Enum):
    IDLE = "idle"
    REALTIME = "realtime"
    OFFLINE = "offline"


def bgr_to_qpixmap(image_bgr: np.ndarray, target_size: QSize | None = None) -> QPixmap:
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


class MainWindowView(QMainWindow):
    start_requested = Signal()
    stop_requested = Signal()
    save_requested = Signal()
    import_requested = Signal()
    options_changed = Signal(dict)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Area Measurement GUI")
        self.resize(1400, 820)

        self._latest_raw_rgb: np.ndarray | None = None
        self._latest_raw_depth: np.ndarray | None = None
        self._latest_refined_depth: np.ndarray | None = None
        self._latest_seg_image: np.ndarray | None = None

        self._aligned_view_dirty = True
        self._seg_view_dirty = True
        self._aligned_label_size = QSize()
        self._seg_label_size = QSize()
        self._aligned_src_size: tuple[int, int] | None = None
        self._seg_src_size: tuple[int, int] | None = None
        self._aligned_target_size = QSize()
        self._seg_target_size = QSize()

        self._build_ui()
        self._run_ui_state = RunUiState.IDLE
        self._apply_run_ui_state(RunUiState.IDLE)

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)

        main_layout = QHBoxLayout(root)

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

        panel_layout = QVBoxLayout()

        run_box = QGroupBox("运行控制")
        run_form = QFormLayout(run_box)
        self.btn_start = QPushButton("开始")
        self.btn_stop = QPushButton("停止")
        self.btn_save = QPushButton("保存")
        self.btn_import = QPushButton("导入")
        run_btn_row = QWidget()
        run_btn_layout = QHBoxLayout(run_btn_row)
        run_btn_layout.setContentsMargins(0, 0, 0, 0)
        run_btn_layout.addWidget(self.btn_start)
        run_btn_layout.addWidget(self.btn_stop)
        run_btn_layout.addWidget(self.btn_save)
        run_btn_layout.addWidget(self.btn_import)
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
        option_form.addRow(self.chk_refine)

        self.chk_use_semantic_segment = QCheckBox("启用语义分割模型")
        self.chk_use_semantic_segment.setChecked(GUI_DEFAULT_USE_SEMANTIC_SEGMENT)
        option_form.addRow(self.chk_use_semantic_segment)

        self.cmb_area_method = QComboBox()
        for method in (AreaMethod.DEPTH_CENTER, AreaMethod.RANSAC, AreaMethod.AUTO):
            self.cmb_area_method.addItem(
                area_method_display_name(method.value), method.value
            )
        default_area_method = RuntimeConfig.area_method.value
        if default_area_method not in {
            AreaMethod.RANSAC.value,
            AreaMethod.DEPTH_CENTER.value,
            AreaMethod.AUTO.value,
        }:
            default_area_method = AreaMethod.DEPTH_CENTER.value
        default_area_method_index = self.cmb_area_method.findData(default_area_method)
        if default_area_method_index >= 0:
            self.cmb_area_method.setCurrentIndex(default_area_method_index)
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
        option_form.addRow("连通域像素阈值", min_area_row)

        panel_layout.addWidget(option_box)

        status_box = QGroupBox("运行状态")
        status_form = QFormLayout(status_box)
        self.lbl_fps = QLabel("0.0")
        self.lbl_regions = QLabel("0")
        self.lbl_method = QLabel(area_method_display_name(default_area_method))
        self.lbl_seg_method = QLabel("Caddy")
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

        self._connect_ui_signals()
        self._sync_model_control_states()

    def _connect_ui_signals(self) -> None:
        self.btn_start.clicked.connect(self.start_requested.emit)
        self.btn_stop.clicked.connect(self.stop_requested.emit)
        self.btn_save.clicked.connect(self.save_requested.emit)
        self.btn_import.clicked.connect(self.import_requested.emit)

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

        self.chk_show_depth.toggled.connect(self.on_show_depth_toggled)
        self.chk_segment.toggled.connect(self._emit_options_changed)
        self.cmb_depth_display.currentTextChanged.connect(self._emit_options_changed)
        self.chk_refine.toggled.connect(self._emit_options_changed)
        self.chk_use_semantic_segment.toggled.connect(self._emit_options_changed)
        self.cmb_area_method.currentTextChanged.connect(self.on_area_method_changed)

        self.slider_min_area.valueChanged.connect(self.spin_min_area.setValue)
        self.spin_min_area.valueChanged.connect(self.slider_min_area.setValue)
        self.spin_min_area.valueChanged.connect(self._emit_options_changed)

    def _set_model_controls_enabled(self, enabled: bool) -> None:
        self.chk_load_refine_model.setEnabled(enabled)
        self.edt_refine_model_path.setEnabled(enabled)
        self.btn_browse_refine_model.setEnabled(enabled)
        self.chk_load_segment_model.setEnabled(enabled)
        self.edt_segment_model_path.setEnabled(enabled)
        self.btn_browse_segment_model.setEnabled(enabled)

    def _apply_run_ui_state(self, state: RunUiState) -> None:
        self._run_ui_state = state
        is_idle = state == RunUiState.IDLE
        is_realtime = state == RunUiState.REALTIME
        is_offline = state == RunUiState.OFFLINE

        self.btn_start.setEnabled(is_idle or is_offline)
        self.btn_stop.setEnabled(is_realtime)
        self.btn_save.setEnabled(is_realtime)
        self.btn_import.setEnabled(is_idle or is_offline)

        self._set_model_controls_enabled(is_idle or is_offline)
        self._sync_model_control_states()

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

    def _collect_runtime_options(self) -> dict:
        return {
            "show_depth": self.chk_show_depth.isChecked(),
            "segment_enabled": self.chk_segment.isChecked(),
            "depth_display_mode": self.cmb_depth_display.currentText(),
            "refine_enabled": self.chk_refine.isChecked(),
            "use_semantic_segment": self.chk_use_semantic_segment.isChecked(),
            "area_method": self.cmb_area_method.currentData()
            or self.cmb_area_method.currentText(),
            "min_region_area": self.spin_min_area.value(),
            "load_refine_model": self.chk_load_refine_model.isChecked(),
            "refine_model_path": self.edt_refine_model_path.text().strip(),
            "load_segment_model": self.chk_load_segment_model.isChecked(),
            "segment_model_path": self.edt_segment_model_path.text().strip(),
        }

    def collect_runtime_options(self) -> dict:
        return self._collect_runtime_options()

    @Slot()
    def _emit_options_changed(self, *_args) -> None:
        self.options_changed.emit(self._collect_runtime_options())

    @Slot(bool)
    def on_show_depth_toggled(self, enabled: bool) -> None:
        self.cmb_depth_display.setEnabled(enabled)
        self._emit_options_changed()

    @Slot(bool)
    def on_load_refine_model_toggled(self, _enabled: bool) -> None:
        self._sync_model_control_states()
        self._emit_options_changed()

    @Slot(bool)
    def on_load_segment_model_toggled(self, _enabled: bool) -> None:
        self._sync_model_control_states()
        self._emit_options_changed()

    @Slot()
    def on_refine_model_path_edit_finished(self) -> None:
        self._emit_options_changed()

    @Slot()
    def on_segment_model_path_edit_finished(self) -> None:
        self._emit_options_changed()

    @Slot()
    def on_browse_refine_model(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        start_dir = self.edt_refine_model_path.text().strip() or str(Path.cwd())
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择深度优化模型",
            start_dir,
            "Model Files (*.pt *.pth);;All Files (*)",
        )
        if file_path:
            self.edt_refine_model_path.setText(file_path)
            self._emit_options_changed()

    @Slot()
    def on_browse_segment_model(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        start_dir = self.edt_segment_model_path.text().strip() or str(Path.cwd())
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择语义分割模型",
            start_dir,
            "Model Files (*.onnx *.rknn);;All Files (*)",
        )
        if file_path:
            self.edt_segment_model_path.setText(file_path)
            self._emit_options_changed()

    def pick_import_rgb_path(self) -> str:
        from PySide6.QtWidgets import QFileDialog

        start_dir = str(Path.cwd())
        rgb_path_str, _ = QFileDialog.getOpenFileName(
            self,
            "选择 RGB 图像",
            start_dir,
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        return rgb_path_str or ""

    @Slot(str)
    def on_area_method_changed(self, method_name: str) -> None:
        self.lbl_method.setText(method_name)
        self._emit_options_changed()

    @staticmethod
    def _compute_aspect_fit_size(label_size: QSize, src_w: int, src_h: int) -> QSize:
        if src_w <= 0 or src_h <= 0:
            return QSize(1, 1)
        lw = max(1, int(label_size.width()))
        lh = max(1, int(label_size.height()))
        scale = min(lw / src_w, lh / src_h)
        return QSize(max(1, int(src_w * scale)), max(1, int(src_h * scale)))

    def _update_view_target_sizes(
        self,
        aligned_image: np.ndarray,
        seg_image: np.ndarray,
    ) -> None:
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

    def render_frame(self, frame: object) -> None:
        aligned_image = getattr(frame, "aligned_image", None)
        seg_image = getattr(frame, "seg_image", None)
        if aligned_image is None or seg_image is None:
            return

        self._update_view_target_sizes(aligned_image, seg_image)
        self._latest_raw_rgb = getattr(frame, "raw_rgb", None)
        self._latest_raw_depth = getattr(frame, "raw_depth", None)
        self._latest_refined_depth = getattr(frame, "refined_depth", None)
        self._latest_seg_image = seg_image

        aligned_pix = bgr_to_qpixmap(aligned_image, self._aligned_target_size)
        seg_pix = bgr_to_qpixmap(seg_image, self._seg_target_size)
        if not aligned_pix.isNull():
            self.aligned_label.setPixmap(aligned_pix)
        if not seg_pix.isNull():
            self.segment_label.setPixmap(seg_pix)

        self.lbl_fps.setText(f"{float(getattr(frame, 'fps', 0.0)):.1f}")
        self.lbl_regions.setText(str(int(getattr(frame, "n_regions", 0))))
        self.lbl_method.setText(
            area_method_display_name(
                str(getattr(frame, "method_name", self.lbl_method.text()))
            )
        )
        self.lbl_seg_method.setText(
            str(getattr(frame, "seg_display", self.lbl_seg_method.text()))
        )

    def render_status(self, message: str) -> None:
        self._append_log(message)

    def show_error(self, message: str) -> None:
        self._append_log(f"[Error] {message}")

    def render_state(self, state: str | RunUiState) -> None:
        if isinstance(state, RunUiState):
            self._apply_run_ui_state(state)
            return
        state_text = str(state).lower()
        if state_text == RunUiState.REALTIME.value:
            self._apply_run_ui_state(RunUiState.REALTIME)
        elif state_text == RunUiState.OFFLINE.value:
            self._apply_run_ui_state(RunUiState.OFFLINE)
        else:
            self._apply_run_ui_state(RunUiState.IDLE)

    def clear_display(self) -> None:
        self._latest_raw_rgb = None
        self._latest_raw_depth = None
        self._latest_refined_depth = None
        self._latest_seg_image = None

        self.aligned_label.clear()
        self.segment_label.clear()
        self.aligned_label.setText("Align Viewer")
        self.segment_label.setText("Segmentation Regions")

        self.lbl_fps.setText("0.0")
        self.lbl_regions.setText("0")
        self.lbl_seg_method.setText("Caddy")

        self._aligned_view_dirty = True
        self._seg_view_dirty = True
        self._aligned_src_size = None
        self._seg_src_size = None
        self._aligned_target_size = QSize()
        self._seg_target_size = QSize()

    def resizeEvent(self, event) -> None:
        self._aligned_view_dirty = True
        self._seg_view_dirty = True
        super().resizeEvent(event)
