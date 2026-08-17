import os
import json
from pathlib import Path
import time
import cv2
import numpy as np
from ultralytics import YOLO

# -------------------- 几何辅助函数 --------------------

def rotate90_cw(img: np.ndarray) -> np.ndarray:
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=float)

def xywh_to_xyxy(box):
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=float)

def bbox_scale(box_xywh, scale):
    x, y, w, h = box_xywh
    return np.array([x * scale, y * scale, w * scale, h * scale], dtype=float)

def bbox_translate(box_xywh, tx, ty):
    x, y, w, h = box_xywh
    return np.array([x + tx, y + ty, w, h], dtype=float)

def bbox_from_rotated_to_unrotated_xywh(box_xywh, w0, h0):
    """
    这里保持你原逻辑：rotated 图的 box -> 还原到未旋转图的 box
    注意：你原实现里参数名 w0/h0 其实没用到 w0，仅用 h0/cw 等；
    这里不改动，避免破坏你已有映射行为。
    """
    x, y, w, h = box_xywh
    x_u = y
    y_u = h0 - (x + w)
    w_u = h
    h_u = w
    return np.array([x_u, y_u, w_u, h_u], dtype=float)

def clip_box_xyxy(box, W, H):
    x1, y1, x2, y2 = box
    x1 = clamp(x1, 0, W - 1)
    y1 = clamp(y1, 0, H - 1)
    x2 = clamp(x2, 0, W - 1)
    y2 = clamp(y2, 0, H - 1)
    return np.array([x1, y1, x2, y2], dtype=float)


# -------------------- 亮度/对比度自适应预处理（新增） --------------------

def adaptive_normalize_tile(tile_bgr: np.ndarray,
                            clahe_clip: float = 2.0,
                            clahe_grid: int = 8,
                            highlight_compress: bool = True,
                            hi_thresh: int = 220,
                            hi_compress: float = 0.45,
                            shadow_lift: bool = True,
                            shadow_q: float = 0.15,      # 看暗部用的分位数
                            shadow_target: int = 95,     # 希望暗部抬到的水平(0-255)
                            max_lift: int = 70) -> np.ndarray:
    """
    适合焊缝：强反光 + 局部暗
    - CLAHE：提升局部对比度（主力）
    - 高亮压缩：避免反光区域过曝
    - 阴影提升（按暗部分位数）：只抬暗部，不把亮部推爆
    """
    out = tile_bgr.copy()

    # --- LAB ---
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # 1) CLAHE
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip),
                            tileGridSize=(int(clahe_grid), int(clahe_grid)))
    L2 = clahe.apply(L)

    # 2) Highlight compress
    if highlight_compress:
        L2 = np.where(L2 > hi_thresh,
                      hi_thresh + (L2 - hi_thresh) * float(hi_compress),
                      L2).astype(np.uint8)

    # 3) Shadow lift (percentile-based)
    if shadow_lift:
        Lf = L2.astype(np.float32)
        qv = float(np.quantile(Lf, float(shadow_q)))  # 暗部分位亮度
        # 需要抬升的量：把暗部 qv 拉到 shadow_target（但限制最大提升）
        lift = float(shadow_target) - qv
        lift = float(np.clip(lift, 0.0, float(max_lift)))

        if lift > 1e-3:
            # 只对暗部提升：越暗提升越多，越亮提升越少
            # w 在 [0,1]，L 越小 w 越大
            w = (1.0 - (Lf / 255.0)) ** 2  # 二次增强暗部权重
            Lf2 = Lf + lift * w
            L2 = np.clip(Lf2, 0, 255).astype(np.uint8)

    lab2 = cv2.merge([L2, A, B])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return out




# -------------------- 辅助工具函数 --------------------

def detect_weld_area(area_model: YOLO, image_bgr, conf=0.25):
    H, W = image_bgr.shape[:2]
    res = area_model.predict(source=image_bgr, conf=conf, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    xyxy = res.boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    idx = int(np.argmax(areas))
    x1, y1, x2, y2 = xyxy[idx]
    return clip_box_xyxy(np.array([x1, y1, x2, y2]), W, H)

def tile_width_positions(total_w, tile_w=640, overrate=0.2):
    overrate = min(max(float(overrate), 0.0), 0.9)
    stride = max(1, int(round(tile_w * (1.0 - overrate))))
    if total_w <= tile_w:
        return [0]
    xs, x = [0], 0
    while x + tile_w < total_w:
        x_next = x + stride
        if x_next + tile_w >= total_w:
            xs.append(total_w - tile_w)
            break
        xs.append(x_next)
        x = x_next
    return xs

def draw_boxes(img_bgr, boxes_xyxy, labels=None, colors=None, thickness=2):
    out = img_bgr.copy()
    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0) if colors is None else colors[i]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if labels is not None:
            txt = labels[i]
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, txt, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return out


# -------------------- 主流程函数 --------------------

def run_pipeline(
        image_path,
        area_model,
        defect_model,
        out_dir="outputs",
        overrate=0.2,
        conf=0.25,
        visualize=True,
        # 预处理参数（可按需调整）
        normalize_tiles=True,
        target_mean_L=150.0,
        clahe_clip=2.0,
        clahe_grid=8,
        gamma_only_if_dark=True,
        dark_mean_L_thresh=115.0
):
    time_start = time.time()
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"⚠️ 无法读取图像: {image_path}")
        return
    H0, W0 = img_bgr.shape[:2]

    # 5类对应5色（BGR）
    CLASS_COLOR = {
        "undercut": (0, 0, 255),        # 红
        "slaginclusion": (0, 255, 0),   # 绿
        "porosity": (255, 0, 0),        # 蓝
        "crack": (255, 255, 0),         # 亮青
        "overlap": (0, 165, 255),       # 亮橙
    }
    DEFAULT_COLOR = (0, 128, 255)  # 橙色，防止没匹配上

    # ✅ 检测焊缝区域
    area_xyxy = detect_weld_area(area_model, img_bgr, conf=conf)
    if area_xyxy is None:
        print("❌ 未检测到焊缝区域，保存原图和JSON（文件名保持一致）。")
        os.makedirs(out_dir, exist_ok=True)
        vis_path = os.path.join(out_dir, f"{Path(image_path).stem}_vis.jpg")
        json_path = os.path.join(out_dir, f"{Path(image_path).stem}_result.json")

        cv2.imwrite(vis_path, img_bgr)
        meta = {
            "image_path": image_path,
            "image_size": [int(W0), int(H0)],
            "message": "No weld area detected"
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return

    # ✅ 检测到焊缝区域后保存焊缝区域图像
    ax1, ay1, ax2, ay2 = map(int, area_xyxy)
    if ax2 - ax1 >= ay2 - ay1:
        ax1 = 0
        ax2 = W0
    else:
        ay1 = 0
        ay2 = H0
    crop = img_bgr[ay1:ay2, ax1:ax2].copy()

    # 新增：保存焊缝区域图片
    save_dir = r"F:\weldtoolv2.1\process\weld_area_detect"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{Path(image_path).stem}_weld_area.jpg")
    cv2.imwrite(save_path, crop)
    print(f"📸 焊缝区域已保存: {save_path}")

    ch, cw = crop.shape[:2]
    rotated = False
    if cw < ch:
        crop = rotate90_cw(crop)
        rotated = True
        ch, cw = crop.shape[:2]

    target_h = 640
    scale = target_h / float(ch)
    new_w = max(1, int(round(cw * scale)))
    resized = cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_LINEAR)

    tile_w = 640
    xs = tile_width_positions(new_w, tile_w=tile_w, overrate=overrate)
    det_records = []

    # 保存 640×640 tile（预处理后）
    TILE_SAVE_ROOT = r"F:\weldtoolv2.1\process\tiles_640"
    os.makedirs(TILE_SAVE_ROOT, exist_ok=True)
    # 保存带框 tile
    TILE_VIS_ROOT = r"F:\weldtoolv2.1\process\tiles_640_detected"
    os.makedirs(TILE_VIS_ROOT, exist_ok=True)

    for i_tile, x0 in enumerate(xs):
        tile = resized[:, x0:x0 + tile_w].copy()  # 640×640 原瓦片
        lab_dbg = cv2.cvtColor(tile, cv2.COLOR_BGR2LAB)
        print("前tile mean_L =", float(lab_dbg[:, :, 0].mean()))

        # ✅ 新增：推理前亮度/对比度自适应统一
        if normalize_tiles:
            tile = adaptive_normalize_tile(
                tile,
                clahe_clip=2.0,
                clahe_grid=8,
                highlight_compress=True,
                hi_thresh=220,
                hi_compress=0.45,
                shadow_lift=True,
                shadow_q=0.15,
                shadow_target=95,
                max_lift=70
            )

            lab_dbg = cv2.cvtColor(tile, cv2.COLOR_BGR2LAB)
            print("后tile mean_L =", float(lab_dbg[:, :, 0].mean()))

        tile_name = f"{Path(image_path).stem}_tile{i_tile:03d}.jpg"
        tile_path = os.path.join(TILE_SAVE_ROOT, tile_name)
        cv2.imwrite(tile_path, tile)

        # ① 缺陷检测
        res = defect_model.predict(source=tile, conf=conf, verbose=False)[0]
        det_records_tile = []
        if res.boxes is not None and len(res.boxes) > 0:
            b_xyxy = res.boxes.xyxy.cpu().numpy()
            b_cls = res.boxes.cls.cpu().numpy().astype(int)
            b_conf = res.boxes.conf.cpu().numpy()

            # 过滤逻辑保留
            mask = (b_xyxy[:, 3] < 600) & (b_xyxy[:, 1] > 40)
            b_xyxy = b_xyxy[mask]
            b_cls = b_cls[mask]
            b_conf = b_conf[mask]

            for k in range(len(b_xyxy)):
                x1_t, y1_t, x2_t, y2_t = b_xyxy[k]
                cls_name = defect_model.names.get(int(b_cls[k]), str(b_cls[k]))
                cfd = float(b_conf[k])
                det_records_tile.append({
                    "bbox_xyxy": [float(v) for v in [x1_t, y1_t, x2_t, y2_t]],
                    "category": cls_name,
                    "score": cfd
                })

        # ② 画框
        tile_vis = draw_boxes(
            tile,
            boxes_xyxy=[r["bbox_xyxy"] for r in det_records_tile],
            labels=[f"{r['category']} {r['score']:.2f}" for r in det_records_tile],
            colors=[CLASS_COLOR.get(r["category"], DEFAULT_COLOR) for r in det_records_tile]
        )

        # ③ 保存带框 640×640 图
        vis_name = f"{Path(image_path).stem}_tile{i_tile:03d}_detected.jpg"
        cv2.imwrite(os.path.join(TILE_VIS_ROOT, vis_name), tile_vis)

        # ④ 框换算回原图坐标，塞进全局 det_records
        for r in det_records_tile:
            # 局部坐标 → resized 坐标
            box_r_xywh = xyxy_to_xywh(r["bbox_xyxy"])
            box_r_xywh[0] += x0  # 加上瓦片左上角 x 偏移

            # resized → 原 crop 尺度
            box_u_xywh = bbox_scale(box_r_xywh, 1.0 / scale)

            # 如果曾旋转，再转回来
            if rotated:
                box_pre_xywh = bbox_from_rotated_to_unrotated_xywh(box_u_xywh, w0=ch, h0=cw)
            else:
                box_pre_xywh = box_u_xywh

            # crop 坐标 → 原图坐标
            box_img_xywh = bbox_translate(box_pre_xywh, ax1, ay1)
            box_img_xyxy = xywh_to_xyxy(box_img_xywh)
            box_img_xyxy = clip_box_xyxy(box_img_xyxy, W0, H0)

            det_records.append({
                "bbox_xyxy": [float(v) for v in box_img_xyxy],
                "category": r["category"],
                "score": r["score"]
            })

    # ✅ 可视化整图
    os.makedirs(out_dir, exist_ok=True)
    if visualize:
        boxes = [r["bbox_xyxy"] for r in det_records]
        labels = [f"{r['category']} {r['score']:.2f}" for r in det_records]
        colors = [CLASS_COLOR.get(r["category"], DEFAULT_COLOR) for r in det_records]
        vis = draw_boxes(img_bgr, boxes, labels=labels, colors=colors)
        vis_path = os.path.join(out_dir, f"{Path(image_path).stem}_vis.jpg")
        cv2.imwrite(vis_path, vis)

    # ✅ 保存 JSON
    json_path = os.path.join(out_dir, f"{Path(image_path).stem}_result.json")
    meta = {
        "image_path": image_path,
        "image_size": [int(W0), int(H0)],
        "detections": det_records
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] {os.path.basename(image_path)} 处理完成，用时 {time.time() - time_start:.2f} 秒")


# -------------------- 主入口 --------------------

def main():
    input_dir = "guitest"
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    print("🚀 正在加载模型 ...")
    area_model = YOLO("weights/weld_area.pt")
    defect_model = YOLO("weights/liangdutongyi.pt")

    # 预热
    _ = area_model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    _ = defect_model.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    print("✅ 模型加载并预热完成。")

    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = [f for f in Path(input_dir).glob("*") if f.suffix.lower() in exts]

    for i, img_path in enumerate(image_files, start=1):
        print(f"\n[{i}/{len(image_files)}] 正在处理：{img_path.name}")
        run_pipeline(
            image_path=str(img_path),
            area_model=area_model,
            defect_model=defect_model,
            out_dir=out_dir,
            overrate=0.0,
            conf=0.25,
            visualize=True,

            # ====== 亮度/对比度统一参数（可以从这里调）======
            normalize_tiles=True,
            target_mean_L=150.0,     # 想更亮：155；想更保守：145~150
            clahe_clip=2.0,          # 暗图噪声大就降到 1.5
            clahe_grid=8,
            gamma_only_if_dark=True, # 只对暗 tile 做 gamma，更稳
            dark_mean_L_thresh=110.0 # 暗阈值：越大=越容易触发 gamma
        )

    print("\n✅ 所有图像处理完成！结果已保存至：", os.path.abspath(out_dir))


if __name__ == "__main__":
    main()
