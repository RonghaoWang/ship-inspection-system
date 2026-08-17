"""船舶智能检测系统 - 统一 GUI 入口。

向导式 4 层导航：
    首页 → 阶段页（建造/运营） → 检测页（三选一）

运行：
    python system_gui.py

依赖：
    pip install PySide6

本文件仅骨架 + 页面结构。算法接入由各 detection 页面单独完成。
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QRadioButton,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).parent))

from gui_common import (
    BasePage,
    HomePage,
    Router,
    STYLE_SHEET,
    big_card,
    big_result,
    image_display,
    primary_button,
    secondary_button,
)


# ============================================================
# 首页
# ============================================================
class Home(HomePage):
    def __init__(self) -> None:
        super().__init__()
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(48, 32, 48, 32)
        root.setSpacing(0)

        # 标题
        title = QLabel("面向船舶建造与运营的\n智能辅助检测系统")
        title.setObjectName("homeTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        subtitle = QLabel("— 一套底座,覆盖建造与运营 —")
        subtitle.setObjectName("homeSubtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(subtitle)

        # 两大卡片
        cards = QHBoxLayout()
        cards.setSpacing(32)
        cards.addStretch(1)
        cards.addWidget(
            big_card(
                "🏗",
                "建造期检测",
                "焊缝外观 · 气密性",
                "进入 →",
                lambda: self._router.push("stage_construction"),
            )
        )
        cards.addWidget(
            big_card(
                "⛴",
                "运营期检测",
                "涂层损坏面积量算",
                "进入 →",
                lambda: self._router.push("stage_operation"),
            )
        )
        cards.addStretch(1)
        root.addLayout(cards)

        root.addStretch(1)

        # 底部
        bottom = QFrame()
        bottom.setObjectName("bottomBar")
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(24, 0, 24, 0)

        for text in ("关于系统", "文档", "导出示例报告"):
            lbl = QLabel(text)
            lbl.setObjectName("bottomHint")
            bottom_layout.addWidget(lbl)
        bottom_layout.addStretch(1)
        ver = QLabel("v1.0")
        ver.setObjectName("bottomHint")
        bottom_layout.addWidget(ver)
        root.addWidget(bottom)


# ============================================================
# 阶段页 - 建造期
# ============================================================
class StageConstruction(BasePage):
    def __init__(self) -> None:
        super().__init__(["首页", "建造期检测"])

    def build_content(self, layout: QVBoxLayout) -> None:
        title = QLabel("建造期检测")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        desc = QLabel("船体焊接完成后、交付前的关键质量把关环节")
        desc.setObjectName("sectionDesc")
        layout.addWidget(desc)

        cards = QHBoxLayout()
        cards.setSpacing(24)
        cards.addWidget(
            big_card(
                "🔍",
                "焊缝外观缺陷检测",
                "面向船厂小组立焊缝的\n多类型外观缺陷智能识别",
                "开始检测",
                lambda: self._router.push("detect_weld"),
            )
        )
        cards.addWidget(
            big_card(
                "💧",
                "焊缝气密性泄漏检测",
                "面向管系 / 舱室密性试验的\n气泡视频泄漏自动判定",
                "开始检测",
                lambda: self._router.push("detect_air"),
            )
        )
        layout.addLayout(cards)

        layout.addStretch(1)


# ============================================================
# 阶段页 - 运营期
# ============================================================
class StageOperation(BasePage):
    def __init__(self) -> None:
        super().__init__(["首页", "运营期检测"])

    def build_content(self, layout: QVBoxLayout) -> None:
        title = QLabel("运营期检测")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        desc = QLabel("船舶入役后的坞修检修关键量算环节")
        desc.setObjectName("sectionDesc")
        layout.addWidget(desc)

        cards = QHBoxLayout()
        cards.setSpacing(24)
        cards.addStretch(1)
        cards.addWidget(
            big_card(
                "🎨",
                "涂层损坏面积量算",
                "面向坞修现场的\nRGB-D 相机智能面积量算",
                "开始检测",
                lambda: self._router.push("detect_coating"),
            )
        )
        cards.addStretch(1)

        layout.addLayout(cards)
        layout.addStretch(1)


# ============================================================
# 检测页 - 焊缝外观
# ============================================================
class WeldWorker(QThread):
    """后台跑焊缝检测，避免阻塞 UI。"""
    finished_ok = Signal(object)   # DetectResult
    failed = Signal(str)

    def __init__(self, adapter, image_path: str, overlap: float, conf: float) -> None:
        super().__init__()
        self._adapter = adapter
        self._image_path = image_path
        self._overlap = overlap
        self._conf = conf

    def run(self) -> None:
        try:
            result = self._adapter.detect(
                image_path=self._image_path,
                overlap=self._overlap,
                conf=self._conf,
            )
            self.finished_ok.emit(result)
        except Exception as e:
            import traceback
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


class DetectWeld(BasePage):
    def __init__(self) -> None:
        self._adapter = None
        self._current_image: str | None = None
        self._current_result = None
        self._worker: WeldWorker | None = None
        super().__init__(["首页", "建造期", "焊缝外观缺陷检测"])

    def build_content(self, layout: QVBoxLayout) -> None:
        # 主区：左右
        main_row = QHBoxLayout()
        main_row.setSpacing(16)

        # 左：输入 + 参数
        left = QVBoxLayout()

        input_grp = QGroupBox("输入")
        input_layout = QVBoxLayout(input_grp)
        input_layout.addWidget(secondary_button("选择焊缝图像...", self._pick))
        self._file_label = QLabel("尚未选择文件")
        self._file_label.setStyleSheet("color: #5f6368; font-size: 12px;")
        input_layout.addWidget(self._file_label)
        self._preview = image_display("图像预览")
        input_layout.addWidget(self._preview)
        left.addWidget(input_grp)

        param_grp = QGroupBox("参数")
        param_layout = QGridLayout(param_grp)
        param_layout.addWidget(QLabel("切片重叠率"), 0, 0)
        self._overlap = QSlider(Qt.Orientation.Horizontal)
        self._overlap.setRange(0, 100)
        self._overlap.setValue(20)
        self._overlap_label = QLabel("0.20")
        self._overlap.valueChanged.connect(
            lambda v: self._overlap_label.setText(f"{v/100:.2f}")
        )
        param_layout.addWidget(self._overlap, 0, 1)
        param_layout.addWidget(self._overlap_label, 0, 2)

        param_layout.addWidget(QLabel("置信度阈值"), 1, 0)
        self._conf = QSlider(Qt.Orientation.Horizontal)
        self._conf.setRange(1, 99)
        self._conf.setValue(25)
        self._conf_label = QLabel("0.25")
        self._conf.valueChanged.connect(
            lambda v: self._conf_label.setText(f"{v/100:.2f}")
        )
        param_layout.addWidget(self._conf, 1, 1)
        param_layout.addWidget(self._conf_label, 1, 2)
        left.addWidget(param_grp)

        self._run_btn = primary_button("开始检测", self._run)
        left.addWidget(self._run_btn)
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #5f6368; font-size: 12px;")
        left.addWidget(self._status_label)
        left.addStretch(1)

        left_wrap = QWidget()
        left_wrap.setLayout(left)
        left_wrap.setMaximumWidth(360)
        main_row.addWidget(left_wrap)

        # 右：结果可视化
        right = QVBoxLayout()
        result_grp = QGroupBox("结果可视化")
        rg = QVBoxLayout(result_grp)
        self._result_image = image_display("检测结果将在这里显示")
        rg.addWidget(self._result_image)
        right.addWidget(result_grp, 2)

        table_grp = QGroupBox("检测结果")
        tg = QVBoxLayout(table_grp)
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["#", "缺陷类型", "位置 (x,y,w,h)", "置信度"])
        self._table.horizontalHeader().setStretchLastSection(True)
        tg.addWidget(self._table)
        self._summary_label = QLabel("")
        self._summary_label.setStyleSheet("color: #5f6368; padding: 4px;")
        tg.addWidget(self._summary_label)
        right.addWidget(table_grp, 1)

        main_row.addLayout(right, 1)
        layout.addLayout(main_row, 1)

        # 底部动作行
        actions = QHBoxLayout()
        actions.addWidget(secondary_button("导出报告 JSON", self._export_json))
        actions.addWidget(secondary_button("保存标注图", self._save_vis))
        actions.addWidget(secondary_button("清空", self._clear))
        actions.addStretch(1)
        layout.addLayout(actions)

    def _get_adapter(self):
        if self._adapter is None:
            from modules.weld_defect.gui_adapter import WeldDefectAdapter
            self._adapter = WeldDefectAdapter()
        return self._adapter

    def _pick(self) -> None:
        from pathlib import Path
        start_dir = str(Path(__file__).parent / "modules" / "weld_defect" / "samples")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择焊缝图像",
            start_dir,
            "图像 (*.jpg *.jpeg *.png *.bmp)",
        )
        if not path:
            return
        self._current_image = path
        self._file_label.setText(Path(path).name)
        # 预览缩略图
        pix = QPixmap(path)
        if not pix.isNull():
            self._preview.setPixmap(
                pix.scaled(
                    self._preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            self._preview.setText("")

    def _run(self) -> None:
        if not self._current_image:
            QMessageBox.information(self, "提示", "请先选择一张焊缝图像。")
            return

        adapter = self._get_adapter()
        ok, msg = adapter.check_environment()
        if not ok:
            QMessageBox.warning(self, "环境未就绪", msg)
            return

        # 禁用按钮避免重复触发
        self._run_btn.setEnabled(False)
        self._status_label.setText("检测中…（首次运行会加载模型，可能需要几秒）")
        QApplication.processEvents()

        self._worker = WeldWorker(
            adapter,
            self._current_image,
            self._overlap.value() / 100.0,
            self._conf.value() / 100.0,
        )
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_done(self, result) -> None:
        self._current_result = result
        # 结果图
        if result.vis_image_path:
            pix = QPixmap(result.vis_image_path)
            if not pix.isNull():
                self._result_image.setPixmap(
                    pix.scaled(
                        self._result_image.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
                self._result_image.setText("")
        # 表格
        self._table.setRowCount(len(result.detections))
        for i, det in enumerate(result.detections):
            x1, y1, x2, y2 = det.bbox_xyxy
            self._table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self._table.setItem(i, 1, QTableWidgetItem(det.category))
            self._table.setItem(
                i,
                2,
                QTableWidgetItem(f"({int(x1)}, {int(y1)}, {int(x2 - x1)}, {int(y2 - y1)})"),
            )
            self._table.setItem(i, 3, QTableWidgetItem(f"{det.score:.2f}"))
        self._summary_label.setText(
            f"共检出 {len(result.detections)} 处缺陷，耗时 {result.elapsed_s:.2f} 秒"
        )
        self._status_label.setText("")
        self._run_btn.setEnabled(True)

    def _on_failed(self, msg: str) -> None:
        self._status_label.setText("")
        self._run_btn.setEnabled(True)
        QMessageBox.critical(self, "检测失败", msg)

    def _export_json(self) -> None:
        if not self._current_result or not self._current_result.json_path:
            QMessageBox.information(self, "提示", "请先运行检测。")
            return
        from pathlib import Path
        src = Path(self._current_result.json_path)
        if not src.exists():
            QMessageBox.warning(self, "文件不存在", str(src))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出报告 JSON", src.name, "JSON (*.json)"
        )
        if not path:
            return
        Path(path).write_bytes(src.read_bytes())
        QMessageBox.information(self, "完成", f"已导出至 {path}")

    def _save_vis(self) -> None:
        if not self._current_result or not self._current_result.vis_image_path:
            QMessageBox.information(self, "提示", "请先运行检测。")
            return
        from pathlib import Path
        src = Path(self._current_result.vis_image_path)
        if not src.exists():
            QMessageBox.warning(self, "文件不存在", str(src))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存标注图", src.name, "JPEG (*.jpg);;PNG (*.png)"
        )
        if not path:
            return
        Path(path).write_bytes(src.read_bytes())
        QMessageBox.information(self, "完成", f"已保存至 {path}")

    def _clear(self) -> None:
        self._current_image = None
        self._current_result = None
        self._file_label.setText("尚未选择文件")
        self._preview.setPixmap(QPixmap())
        self._preview.setText("图像预览")
        self._result_image.setPixmap(QPixmap())
        self._result_image.setText("检测结果将在这里显示")
        self._table.setRowCount(0)
        self._summary_label.setText("")
        self._status_label.setText("")


# ============================================================
# 检测页 - 气密性
# ============================================================
class AirWorker(QThread):
    """后台跑气密性检测，逐帧发信号。"""
    frame_ready = Signal(object)   # FrameResult
    finished_ok = Signal(object)   # VideoSummary
    failed = Signal(str)

    def __init__(self, adapter, video_path: str, params, writer_output_path=None) -> None:
        super().__init__()
        self._adapter = adapter
        self._video_path = video_path
        self._params = params
        self._writer_output_path = writer_output_path
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def _should_stop(self) -> bool:
        return self._stop_requested

    def run(self) -> None:
        try:
            gen = self._adapter.process_video(
                self._video_path,
                self._params,
                should_stop=self._should_stop,
                writer_output_path=self._writer_output_path,
            )
            summary = None
            try:
                while True:
                    frame_result = next(gen)
                    self.frame_ready.emit(frame_result)
            except StopIteration as stop:
                summary = stop.value
            self.finished_ok.emit(summary)
        except Exception as e:
            import traceback
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


class DetectAir(BasePage):
    def __init__(self) -> None:
        self._adapter = None
        self._current_video: str | None = None
        self._current_summary = None
        self._worker: AirWorker | None = None
        self._last_frame_bgr = None  # 供保存标注视频截图用（简化：只留最后一帧）
        self._frame_skip = 2         # UI 刷新降采样，减少压力
        self._frame_count = 0
        self._output_video_path: str | None = None
        self._last_output_video_path: str | None = None
        super().__init__(["首页", "建造期", "焊缝气密性泄漏检测"])

    def build_content(self, layout: QVBoxLayout) -> None:
        main_row = QHBoxLayout()
        main_row.setSpacing(16)

        # 左
        left = QVBoxLayout()
        input_grp = QGroupBox("输入")
        input_layout = QVBoxLayout(input_grp)
        input_layout.addWidget(secondary_button("选择视频文件...", self._pick))
        self._file_label = QLabel("尚未选择文件")
        self._file_label.setStyleSheet("color: #5f6368; font-size: 12px;")
        input_layout.addWidget(self._file_label)
        self._meta_label = QLabel("")
        self._meta_label.setStyleSheet("color: #5f6368; font-size: 12px;")
        input_layout.addWidget(self._meta_label)
        left.addWidget(input_grp)

        param_grp = QGroupBox("参数")
        pg = QGridLayout(param_grp)
        pg.addWidget(QLabel("置信度阈值"), 0, 0)
        self._conf = QDoubleSpinBox()
        self._conf.setRange(0.0, 1.0)
        self._conf.setSingleStep(0.05)
        self._conf.setValue(0.3)
        pg.addWidget(self._conf, 0, 1)
        pg.addWidget(QLabel("IoU 阈值"), 1, 0)
        self._iou = QDoubleSpinBox()
        self._iou.setRange(0.0, 1.0)
        self._iou.setSingleStep(0.05)
        self._iou.setValue(0.5)
        pg.addWidget(self._iou, 1, 1)
        pg.addWidget(QLabel("确认帧数"), 2, 0)
        self._confirm = QSpinBox()
        self._confirm.setRange(1, 30)
        self._confirm.setValue(3)
        pg.addWidget(self._confirm, 2, 1)
        left.addWidget(param_grp)

        # 输出选项
        output_grp = QGroupBox("输出")
        og = QVBoxLayout(output_grp)
        self._save_video_cb = QCheckBox("检测时同步保存可视化视频")
        self._save_video_cb.setChecked(False)
        og.addWidget(self._save_video_cb)
        self._output_path_label = QLabel("尚未指定输出路径")
        self._output_path_label.setStyleSheet("color: #5f6368; font-size: 11px;")
        self._output_path_label.setWordWrap(True)
        og.addWidget(self._output_path_label)
        self._pick_output_btn = secondary_button("指定输出路径…", self._pick_output)
        og.addWidget(self._pick_output_btn)
        left.addWidget(output_grp)

        self._run_btn = primary_button("开始检测", self._run)
        left.addWidget(self._run_btn)
        self._stop_btn = secondary_button("停止", self._stop)
        self._stop_btn.setEnabled(False)
        left.addWidget(self._stop_btn)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #5f6368; font-size: 12px;")
        left.addWidget(self._status_label)
        left.addStretch(1)

        left_wrap = QWidget()
        left_wrap.setLayout(left)
        left_wrap.setMaximumWidth(360)
        main_row.addWidget(left_wrap)

        # 右：预览 + 结果
        right = QVBoxLayout()
        preview_grp = QGroupBox("实时预览")
        pv = QVBoxLayout(preview_grp)
        self._preview = image_display("视频帧将在这里显示")
        pv.addWidget(self._preview)
        self._frame_info = QLabel("")
        self._frame_info.setStyleSheet("color: #5f6368; font-size: 12px; padding: 4px;")
        pv.addWidget(self._frame_info)
        right.addWidget(preview_grp, 2)

        result_grp = QGroupBox("泄漏判定结果")
        rv = QVBoxLayout(result_grp)
        self._verdict = big_result("—", "综合判定")
        rv.addWidget(self._verdict)
        self._score_label = QLabel("综合评分：—")
        self._score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._score_label.setStyleSheet("color: #202124; font-size: 14px; font-weight: 600;")
        rv.addWidget(self._score_label)
        self._components_label = QLabel("聚合度 A = —   持续性 P = —   一致性 C = —")
        self._components_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._components_label.setStyleSheet("color: #5f6368; font-size: 12px;")
        rv.addWidget(self._components_label)
        self._formula_label = QLabel("综合评分 = 聚合度×0.2 + 持续性×0.5 + 一致性×0.3")
        self._formula_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._formula_label.setStyleSheet("color: #5f6368; font-size: 11px; padding: 4px 0;")
        rv.addWidget(self._formula_label)
        self._leak_points_label = QLabel("")
        self._leak_points_label.setStyleSheet("color: #d93025; font-size: 12px; padding: 4px;")
        rv.addWidget(self._leak_points_label)
        right.addWidget(result_grp, 1)

        main_row.addLayout(right, 1)
        layout.addLayout(main_row, 1)

        actions = QHBoxLayout()
        actions.addWidget(secondary_button("导出报告 JSON", self._export_json))
        actions.addWidget(secondary_button("打开输出视频", self._open_output_video))
        actions.addWidget(secondary_button("清空", self._clear))
        actions.addStretch(1)
        layout.addLayout(actions)

    def _get_adapter(self):
        if self._adapter is None:
            from modules.airtightness.gui_adapter import AirtightnessAdapter
            self._adapter = AirtightnessAdapter()
        return self._adapter

    def _pick(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择气密性试验视频",
            "",
            "视频 (*.mp4 *.avi *.mov *.mkv)",
        )
        if not path:
            return
        self._current_video = path
        from pathlib import Path
        self._file_label.setText(Path(path).name)

        # 探视频元数据
        from modules.airtightness.gui_adapter import AirtightnessAdapter
        info = AirtightnessAdapter.probe(path)
        if info.get("ok"):
            self._meta_label.setText(
                f"时长 {info['duration_s']:.1f}s · {info['width']}×{info['height']} · {info['fps']:.1f} FPS · {info['frame_count']} 帧"
            )
        else:
            self._meta_label.setText(info.get("error", ""))

    def _pick_output(self) -> None:
        from pathlib import Path
        default = "leak_detection_output.mp4"
        if self._current_video:
            stem = Path(self._current_video).stem
            default = f"{stem}_leak_detection.mp4"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "选择输出视频路径",
            default,
            "MP4 (*.mp4);;AVI (*.avi)",
        )
        if not path:
            return
        self._output_video_path = path
        self._output_path_label.setText(path)
        self._save_video_cb.setChecked(True)

    def _run(self) -> None:
        if not self._current_video:
            QMessageBox.information(self, "提示", "请先选择一段视频。")
            return

        adapter = self._get_adapter()
        ok, msg = adapter.check_environment()
        if not ok:
            QMessageBox.warning(self, "环境未就绪", msg)
            return

        # 视频输出路径
        writer_path = None
        if self._save_video_cb.isChecked():
            if not self._output_video_path:
                # 未指定路径：弹选择框
                self._pick_output()
                if not self._output_video_path:
                    # 用户取消
                    QMessageBox.information(self, "提示", "已取消。若不需要保存视频，请取消勾选。")
                    return
            writer_path = self._output_video_path

        # 组装参数
        from modules.airtightness.gui_adapter import LeakDetectionParams
        params = LeakDetectionParams(
            conf=float(self._conf.value()),
            iou=float(self._iou.value()),
            confirm_frames=int(self._confirm.value()),
        )

        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        status = "检测中…（首次加载模型可能需要几秒）"
        if writer_path:
            status += f"\n可视化视频保存到：{writer_path}"
        self._status_label.setText(status)
        self._frame_count = 0
        # 清结果
        self._verdict.findChild(QLabel).setText("检测中")

        self._last_output_video_path = writer_path

        self._worker = AirWorker(
            adapter, self._current_video, params, writer_output_path=writer_path
        )
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _stop(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()
            self._status_label.setText("正在停止…")

    def _on_frame(self, frame_result) -> None:
        self._frame_count += 1
        # 降采样刷新 UI
        if self._frame_count % self._frame_skip != 0:
            return
        self._last_frame_bgr = frame_result.frame_bgr

        # BGR ndarray -> QPixmap
        import numpy as np
        from PySide6.QtGui import QImage
        img = frame_result.frame_bgr
        h, w = img.shape[:2]
        rgb = img[:, :, ::-1].copy()
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self._preview.setPixmap(
            pix.scaled(
                self._preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._preview.setText("")

        self._frame_info.setText(
            f"帧 {frame_result.frame_idx} · 气泡数 {frame_result.num_bubbles} · 当前评分 {frame_result.smoothed_score:.3f}"
        )

        # 实时更新结果面板
        verdict_label = self._verdict.findChild(QLabel)
        if frame_result.is_leaking:
            verdict_label.setText("⚠ 检测到泄漏")
            verdict_label.setStyleSheet("color: #d93025;")
        else:
            verdict_label.setText("暂未检测到")
            verdict_label.setStyleSheet("color: #188038;")

        self._score_label.setText(f"综合评分：{frame_result.smoothed_score:.3f}")
        self._components_label.setText(
            f"聚合度 A = {frame_result.aggregation:.3f}   "
            f"持续性 P = {frame_result.persistence:.3f}   "
            f"一致性 C = {frame_result.consistency:.3f}"
        )

    def _on_done(self, summary) -> None:
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._current_summary = summary

        if summary is None:
            self._status_label.setText("已停止")
            return

        verdict_label = self._verdict.findChild(QLabel)
        if summary.final_verdict:
            verdict_label.setText("⚠ 检测到泄漏")
            verdict_label.setStyleSheet("color: #d93025;")
            self._leak_points_label.setText(
                f"泄漏帧占比 {summary.leak_ratio*100:.1f}%"
            )
        else:
            verdict_label.setText("✓ 未检测到泄漏")
            verdict_label.setStyleSheet("color: #188038;")
            self._leak_points_label.setText("")

        self._score_label.setText(f"最高评分：{summary.max_score:.3f}")
        done_msg = f"完成 · 处理 {summary.total_frames} 帧 · 泄漏帧 {summary.leak_frames}"
        if self._last_output_video_path:
            done_msg += f"\n可视化视频已保存：{self._last_output_video_path}"
        self._status_label.setText(done_msg)

    def _on_failed(self, msg: str) -> None:
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status_label.setText("")
        QMessageBox.critical(self, "检测失败", msg)

    def _open_output_video(self) -> None:
        if not self._last_output_video_path:
            QMessageBox.information(
                self, "提示",
                "未生成可视化视频。请勾选“检测时同步保存可视化视频”，指定输出路径后重新运行检测。",
            )
            return
        from pathlib import Path
        p = Path(self._last_output_video_path)
        if not p.exists():
            QMessageBox.warning(self, "文件不存在", f"未找到 {p}")
            return
        import os
        try:
            os.startfile(str(p))  # type: ignore[attr-defined]
        except AttributeError:
            import subprocess, sys as _sys
            if _sys.platform == "darwin":
                subprocess.Popen(["open", str(p.parent)])
            else:
                subprocess.Popen(["xdg-open", str(p.parent)])

    def _export_json(self) -> None:
        if self._current_summary is None:
            QMessageBox.information(self, "提示", "请先运行检测。")
            return
        summary = self._current_summary
        report = {
            "task": "airtightness",
            "video_path": self._current_video,
            "params": {
                "conf": float(self._conf.value()),
                "iou": float(self._iou.value()),
                "confirm_frames": int(self._confirm.value()),
                "weights": {
                    "w_aggregation": 0.2,
                    "w_persistence": 0.5,
                    "w_consistency": 0.3,
                },
            },
            "summary": {
                "total_frames": summary.total_frames,
                "leak_frames": summary.leak_frames,
                "leak_ratio": summary.leak_ratio,
                "max_score": summary.max_score,
                "final_verdict": summary.final_verdict,
                "leak_points": summary.leak_points,
            },
        }
        path, _ = QFileDialog.getSaveFileName(
            self, "导出报告 JSON", "airtightness_report.json", "JSON (*.json)"
        )
        if not path:
            return
        import json
        from pathlib import Path
        Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        QMessageBox.information(self, "完成", f"已导出至 {path}")

    def _clear(self) -> None:
        self._current_video = None
        self._current_summary = None
        self._file_label.setText("尚未选择文件")
        self._meta_label.setText("")
        self._preview.setPixmap(QPixmap())
        self._preview.setText("视频帧将在这里显示")
        self._frame_info.setText("")
        verdict_label = self._verdict.findChild(QLabel)
        verdict_label.setText("—")
        verdict_label.setStyleSheet("")
        self._score_label.setText("综合评分：—")
        self._components_label.setText("聚合度 A = —   持续性 P = —   一致性 C = —")
        self._leak_points_label.setText("")
        self._status_label.setText("")
        self._output_video_path = None
        self._last_output_video_path = None
        self._output_path_label.setText("尚未指定输出路径")
        self._save_video_cb.setChecked(False)


# ============================================================
# 检测页 - 涂层
# ============================================================
class DetectCoating(BasePage):
    def __init__(self) -> None:
        self._coating_widget = None
        self._load_error: str | None = None
        super().__init__(["首页", "运营期", "涂层损坏面积量算"])

    def build_content(self, layout: QVBoxLayout) -> None:
        # 顶部提示条
        hint = QLabel(
            "支持实时相机（Orbbec RGB-D）与离线 RGB-D 图像导入两种模式。"
            "面积算法可在参数区切换：RANSAC 平面拟合 / Depth Center / Auto。"
        )
        hint.setStyleSheet("color: #5f6368; padding: 2px 0 8px 0;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # 嵌入 CoatingWidget
        try:
            from modules.coating_damage.coating_widget import CoatingWidget
            self._coating_widget = CoatingWidget(parent=self)
            layout.addWidget(self._coating_widget, 1)
        except Exception as e:
            import traceback
            self._load_error = f"{e}\n\n{traceback.format_exc()}"
            fallback = QLabel(
                "涂层检测模块加载失败。请检查以下依赖是否安装：\n"
                "• PySide6\n"
                "• pyorbbecsdk（Orbbec 相机 SDK）\n"
                "• opencv-python、numpy、torch、onnxruntime\n\n"
                "详见错误信息：\n\n" + self._load_error
            )
            fallback.setStyleSheet(
                "color: #d93025; background: #fce8e6; "
                "border: 1px solid #f5b7b1; border-radius: 4px; padding: 12px;"
            )
            fallback.setWordWrap(True)
            layout.addWidget(fallback, 1)

    def _on_back(self) -> None:
        """离开涂层页时，安全释放相机与线程。"""
        if self._coating_widget is not None:
            try:
                self._coating_widget.shutdown()
            except Exception:
                pass
        super()._on_back()


# ============================================================
# 主窗口 + 路由注册
# ============================================================
class SystemWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("船舶智能检测系统")
        self.resize(1280, 800)

        router = Router()
        router.register("home", Home)
        router.register("stage_construction", StageConstruction)
        router.register("stage_operation", StageOperation)
        router.register("detect_weld", DetectWeld)
        router.register("detect_air", DetectAir)
        router.register("detect_coating", DetectCoating)

        router.push("home")
        self.setCentralWidget(router)


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE_SHEET)
    win = SystemWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
