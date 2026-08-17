"""
主入口：实时获取 RGB-D 帧 → 分割 → 面积计算 → 可视化。

用法：
    python main.py
"""

import time
import cv2
import numpy as np

from pyorbbecsdk import (
    Pipeline,
    AlignFilter,
    OBStreamType,
)

from config import (
    AreaMethod,
    MIN_DEPTH,
    MAX_DEPTH,
    ALIGN_MODE,
    HW_DENOISE,
    DEPTH_CMAP_MIN_MM,
    DEPTH_CMAP_MAX_MM,
    DEPTH_COLORMAP,
    SAVE_PATH,
    LOAD_MODEL,
    LOAD_REFINE_MODEL,
    LOAD_SEGMENT_MODEL,
    REFINE_MODEL_PATH,
    SEGMENT_MODEL_PATH,
    SEGMENT_METHOD,
    set_calibration_params,
    get_calibration_params,
    RuntimeConfig,
)
from src.camera import get_stream_config, get_camera_parameters, enable_hw_denoise
from src.processing import (
    split_mask,
    refine_depth_frame,
    predict_ort,
    predict_cv,
    predict_rknn,
)
from src.calc import calc_ransac, calc_depth_center, calc_point_cloud, calc_auto
from src.visualization import (
    draw_labeled_regions,
    draw_annotation,
    draw_fps,
    draw_segmentation_overlay,
)
from src.utils import (
    frame_to_bgr_image,
    get_depth_data,
    depth_to_colormap_fixed_window,
    model_loader,
    get_calibration_parameters,
)
from src.io import ExcelRecorder, handle_key, print_keyboard_help


# ─── 主循环 ─────────────────────────────────────────────────


def main():
    cfg = RuntimeConfig
    pipeline = Pipeline()
    recorder = ExcelRecorder(area_method=cfg.area_method)

    # 硬件降噪
    if HW_DENOISE:
        enable_hw_denoise(pipeline)

    # 深度图精细化模型加载
    refine_model = None
    segment_model = None
    if LOAD_MODEL:
        if LOAD_REFINE_MODEL:
            print("[Info] Depth refinement is enabled. Loading model...")
            refine_model = model_loader(
                REFINE_MODEL_PATH,
                type="MDM",
            )
            print("[Info] Model loaded successfully.")
        if LOAD_SEGMENT_MODEL:
            print("[Info] Segment model is enabled. Loading model...")
            segment_model = model_loader(
                SEGMENT_MODEL_PATH,
                type=LOAD_SEGMENT_MODEL,
            )
            print("[Info] Segment model loaded successfully.")

    # 配置流
    result = get_stream_config(pipeline)
    if result is None:
        print("无法获取流配置，退出。")
        return
    depth_profile, color_profile, config = result

    # 按当前流分辨率自动加载校准参数
    resolution = None
    try:
        resolution = f"{depth_profile.get_width()}*{depth_profile.get_height()}"
    except Exception:
        pass

    if resolution:
        try:
            kc, kv = get_calibration_parameters(resolution)
            set_calibration_params(kc, kv)
            print(f"[Calibration] resolution={resolution}, Kc={kc}, Kv={kv}")
        except Exception as e:
            default_kc, default_kv = get_calibration_params()
            print(f"[Calibration] Failed to load by resolution {resolution}: {e}")
            print(f"[Calibration] Use default Kc={default_kc}, Kv={default_kv}")
    else:
        print("[Calibration] Cannot determine stream resolution. Use default Kc/Kv.")

    # 相机内参
    K = get_camera_parameters(color_profile, depth_profile)

    # 启动管线
    pipeline.start(config)
    print("Pipeline started.")

    # 打印帮助
    _print_startup_help()

    align_filter = None
    if ALIGN_MODE == "SW":
        align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    try:
        _run_loop(pipeline, align_filter, K, cfg, recorder, refine_model, segment_model)
    finally:
        cv2.destroyAllWindows()
        pipeline.stop()
        if cfg.save_in_xlsx:
            recorder.save(SAVE_PATH)


def _run_loop(pipeline, align_filter, K, cfg, recorder, refine_model, segment_model):
    """核心帧处理循环。"""
    prev_time = time.time()
    fps = 0.0
    fps_ema = None
    fps_alpha = 0.15
    while True:
        frames = pipeline.wait_for_frames(1000)
        if frames is None:
            continue

        # ── 获取对齐后的帧 ────────────────────────────
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
            print(f"不支持的 ALIGN_MODE: {ALIGN_MODE}，请使用 'HW' 或 'SW'")
            break

        # ── 深度图精细化 ───────────────────────────────
        if LOAD_MODEL & LOAD_REFINE_MODEL & cfg.depth_refinement:
            depth_data = refine_depth_frame(refine_model, color_frame, depth_frame, K)
        else:
            depth_data = get_depth_data(depth_frame, MIN_DEPTH, MAX_DEPTH)

        # ── 转换彩色图 ───────────────────────────────
        color_image = frame_to_bgr_image(color_frame)
        if color_image is None:
            print("Failed to convert frame to image")
            continue

        # ── 深度伪彩叠加显示 ───────────────────
        # 固定比例尺显示深度图
        if cfg.use_fixed_depth_scale:
            depth_image = depth_to_colormap_fixed_window(
                depth_data, DEPTH_CMAP_MIN_MM, DEPTH_CMAP_MAX_MM, DEPTH_COLORMAP
            )
            aligned_image = cv2.addWeighted(color_image, 0.5, depth_image, 0.5, 0)
            cv2.imshow("Align Viewer", aligned_image)
        # 自适应比例尺显示深度图
        else:
            depth_image = cv2.normalize(
                depth_data, None, 0, 255, cv2.NORM_MINMAX
            )  # 归一化到 0-255
            depth_image = cv2.applyColorMap(
                depth_image.astype(np.uint8), cv2.COLORMAP_JET
            )
            aligned_image = cv2.addWeighted(color_image, 0.5, depth_image, 0.5, 0)
            cv2.imshow("Align Viewer", aligned_image)

        # ── 区域分割 ─────────────────────────────────
        if cfg.segment_image:
            if SEGMENT_METHOD == "OpenCV":
                pred_mask = predict_cv(color_image)
            elif (
                SEGMENT_METHOD == "Model"
                and LOAD_MODEL
                and LOAD_SEGMENT_MODEL == "ONNX"
            ):
                pred_mask = predict_ort(segment_model, color_image)
            elif (
                SEGMENT_METHOD == "Model"
                and LOAD_MODEL
                and LOAD_SEGMENT_MODEL == "RKNN"
            ):
                pred_mask = predict_rknn(segment_model, color_image)
            else:
                print(f"不支持的 segment_method: {SEGMENT_METHOD}")
                break

            # if SEGMENT_METHOD == "Model" and pred_mask is not None:
            #     pred_vis = draw_segmentation_overlay(color_image, pred_mask)
            #     cv2.imshow("Segmentation Overlay", pred_vis)

            # 掩码区域划分
            _, labeled, n = split_mask(pred_mask, cfg.min_region_area)
            blended = draw_labeled_regions(
                color_image, labeled, n, pred_mask
            )  # 区域着色显示

            # ── 逐区域计算面积 ───────────────────────────
            for label in range(1, n + 1):
                region_mask = labeled == label  # 当前区域的二值掩膜
                region_depth_data = np.where(
                    region_mask, depth_data, 0
                )  # 当前区域深度数据
                num_pixels = int(np.sum(region_mask))  # 区域像素数量
                if num_pixels < 10:
                    continue

                if cfg.area_method == AreaMethod.RANSAC:
                    area, area_fixed = calc_ransac(region_depth_data, K)

                elif cfg.area_method == AreaMethod.DEPTH_CENTER:
                    area, area_fixed, info = calc_depth_center(
                        region_depth_data,
                        K,
                        num_pixels,
                        use_sampling=cfg.using_sampling,
                    )

                elif cfg.area_method == AreaMethod.POINT_CLOUD:
                    area = calc_point_cloud(region_depth_data, K)

                elif cfg.area_method == AreaMethod.AUTO:
                    area, area_fixed, info = calc_auto(
                        region_depth_data,
                        K,
                        num_pixels,
                        use_sampling=cfg.using_sampling,
                    )
                else:
                    print(f"不支持的 area_method: {cfg.area_method}")
                    continue

                text = f"{area:.2f}cm2"
                if cfg.area_method == AreaMethod.DEPTH_CENTER:
                    draw_annotation(
                        blended,
                        region_mask,
                        text,
                        cx=info["center_x"],
                        cy=info["center_y"],
                    )  # 标注面积并显示中心点位置
                else:
                    draw_annotation(blended, region_mask, text)  # 标注面积

        else:
            pass  # FPS 绘制和 imshow 统一在下方处理

        # ── FPS 计算 ──────────────────────────────────
        curr_time = time.time()
        inst_fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        # 指数滑动平均，减少帧间抖动带来的显示跳变
        if fps_ema is None:
            fps_ema = inst_fps
        else:
            fps_ema = fps_alpha * inst_fps + (1.0 - fps_alpha) * fps_ema
        fps = fps_ema

        # 在 Segmentation Regions 窗口左上角绘制 FPS
        # 获取当前显示的图像（分割模式用 blended，否则用 color_image）
        seg_img = blended if cfg.segment_image else color_image
        draw_fps(seg_img, fps)
        cv2.imshow("Segmentation Regions", seg_img)

        # ── 键盘交互 ─────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if handle_key(key):
            if LOAD_MODEL and LOAD_SEGMENT_MODEL == "RKNN":
                segment_model.release()  # 释放 RKNN 模型资源
            break


def _print_startup_help():
    """在启动时打印帮助信息。"""
    print("[Startup] 按 h 可再次查看帮助。")
    print_keyboard_help()


if __name__ == "__main__":
    main()
