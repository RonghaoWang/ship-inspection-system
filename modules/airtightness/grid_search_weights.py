import os
import json
import math
import itertools
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

VIDEO_DIR = "/root/autodl-tmp/ultralytics-main/leak_videos"
MODEL_PATH = "/root/autodl-tmp/ultralytics-main/runs/detect/train-13/weights/best.pt"
CACHE_DIR = "/root/autodl-tmp/ultralytics-main/grid_search_cache"
WINDOW = 30
ALPHA_EMA = 0.7
THRESHOLD = 0.40
CONFIRM_FRAMES = 3
DBSCAN_EPS = 120
DBSCAN_MIN_SAMPLES = 3


def get_frame_results(video_path):
    """
    对单个视频运行检测+跟踪，返回逐帧结果列表。
    每帧: [(center_x, center_y, w, h, conf, track_id), ...]
    """
    name = os.path.splitext(os.path.basename(video_path))[0]
    cache_path = os.path.join(CACHE_DIR, f"{name}.json")

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    model = YOLO(MODEL_PATH)
    results = model.track(
        source=video_path,
        conf=0.3,
        iou=0.5,
        persist=True,
        tracker="botsort.yaml",
        verbose=False,
    )

    frame_results = []
    for r in results:
        if r.boxes is None or r.boxes.id is None:
            frame_results.append([])
            continue
        boxes = r.boxes.xywh.cpu().tolist()
        confs = r.boxes.conf.cpu().tolist()
        ids = r.boxes.id.int().cpu().tolist()
        frame_results.append([
            (float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c), int(i))
            for b, c, i in zip(boxes, confs, ids)
        ])

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(frame_results, f)

    return frame_results


def extract_features(frame_results):
    """
    从逐帧跟踪结果中提取每帧的 A、P、C 三维特征。
    返回: list of dict，每帧包含 {"A": float, "P": float, "C": float}
    """
    T = len(frame_results)
    features = []
    track_first_appear = {}
    track_age = defaultdict(list)
    track_velocities = defaultdict(list)

    for t in range(T):
        active_bubbles = frame_results[t]
        active_ids = set(b[5] for b in active_bubbles)

        for bid in active_ids:
            if bid not in track_first_appear:
                track_first_appear[bid] = t
            track_age[bid].append(t)

        # ================= 聚集程度 A =================
        origin_points = []
        for b in active_bubbles:
            cx, cy, _, _, _, bid = b
            origin_t = track_first_appear[bid]
            first_frame_bubbles = frame_results[origin_t]
            for fb in first_frame_bubbles:
                if fb[5] == bid:
                    origin_points.append((fb[0], fb[1]))
                    break

        N = len(origin_points)
        if N >= DBSCAN_MIN_SAMPLES:
            pts = np.array(origin_points)
            db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(pts)
            labels = db.labels_
            valid = labels[labels != -1]
            if len(valid) == 0:
                A = 0.0
            else:
                best_label = np.bincount(valid).argmax()
                cluster_mask = labels == best_label
                N_cluster = cluster_mask.sum()
                R_cluster = N_cluster / N
                cluster_pts = pts[cluster_mask]
                if len(cluster_pts) >= 2:
                    from sklearn.metrics import pairwise_distances
                    d = pairwise_distances(cluster_pts).flatten()
                    d = d[d > 0]
                    d_nn = d.min() if len(d) > 0 else DBSCAN_EPS
                else:
                    d_nn = DBSCAN_EPS
                C_density = max(0.0, 1.0 - d_nn / DBSCAN_EPS)
                A = R_cluster * C_density
        else:
            A = 0.0

        # ================= 运动持续性 P =================
        win_start = max(0, t - WINDOW + 1)
        frames_with_bubble = 0
        for ft in range(win_start, t + 1):
            if len(frame_results[ft]) > 0:
                frames_with_bubble += 1
        P_det = frames_with_bubble / (t - win_start + 1)

        N_stable = sum(1 for bid in active_ids if sum(1 for x in track_age[bid] if x <= t) >= 5)
        P_stable = N_stable / max(1, len(active_ids))
        P = 0.5 * P_det + 0.5 * P_stable

        # ================= 运动一致性 C =================
        velocity_vectors = []
        for b in active_bubbles:
            cx, cy, w, h, conf, bid = b
            ages = sorted([a for a in track_age[bid] if a <= t])
            if len(ages) < 2:
                continue
            idx_prev = ages[-2]
            bubbles_prev = frame_results[idx_prev]
            for pb in bubbles_prev:
                if pb[5] == bid:
                    vx = cx - pb[0]
                    vy = cy - pb[1]
                    norm = math.sqrt(vx**2 + vy**2 + 1e-12)
                    velocity_vectors.append((vx / norm, vy / norm))
                    break

        if len(velocity_vectors) >= 2:
            v_arr = np.array(velocity_vectors)
            v_mean = v_arr.mean(axis=0)
            v_mean_norm = np.linalg.norm(v_mean)
            if v_mean_norm > 1e-12:
                v_mean /= v_mean_norm
            cos_sim = np.dot(v_arr, v_mean)
            C = float((cos_sim.mean() + 1.0) / 2.0)
        elif len(velocity_vectors) == 1:
            C = 0.5
        else:
            C = 0.5

        features.append({"A": A, "P": P, "C": C})

    return features


def evaluate_weights(features, wA, wP, wC):
    """
    给定权重，对一段视频的特征序列执行加权融合 + EMA + 连续帧确认，
    返回是否判定为泄漏 (True/False)。
    """
    S_smooth = 0.0
    confirm_count = 0

    for f in features:
        S_frame = wA * f["A"] + wP * f["P"] + wC * f["C"]
        S_smooth = ALPHA_EMA * S_frame + (1 - ALPHA_EMA) * S_smooth

        if S_smooth >= THRESHOLD:
            confirm_count += 1
        else:
            confirm_count = 0

        if confirm_count >= CONFIRM_FRAMES:
            return True

    return False


def grid_search(video_paths):
    """
    对全部视频执行网格搜索，找到使召回率最高的权重组合。
    """
    all_features = []
    ground_truth = []

    for vp in video_paths:
        name = os.path.basename(vp)
        cache_feat = os.path.join(CACHE_DIR, f"{os.path.splitext(name)[0]}_feat.json")

        if os.path.exists(cache_feat):
            with open(cache_feat) as f:
                feats = json.load(f)
        else:
            print(f"  提取特征: {name}")
            frame_results = get_frame_results(vp)
            feats = extract_features(frame_results)
            with open(cache_feat, "w") as f:
                json.dump(feats, f)

        all_features.append(feats)
        ground_truth.append(True)

    step = 0.1
    candidates = []
    for i in range(0, 11):
        wA = round(i * step, 2)
        for j in range(0, 11 - i):
            wP = round(j * step, 2)
            wC = round(1.0 - wA - wP, 2)
            candidates.append((wA, wP, wC))

    print(f"\n共 {len(candidates)} 组权重待搜索")
    best_recall = 0
    best_weights = None
    results_all = []

    for idx, (wA, wP, wC) in enumerate(candidates):
        correct = 0
        for feats, gt in zip(all_features, ground_truth):
            pred = evaluate_weights(feats, wA, wP, wC)
            if pred == gt:
                correct += 1
        recall = correct / len(ground_truth)
        results_all.append((wA, wP, wC, recall))

        if recall > best_recall:
            best_recall = recall
            best_weights = (wA, wP, wC)

        if (idx + 1) % 10 == 0:
            print(f"  [{idx + 1}/{len(candidates)}] 当前最佳: wA={best_weights[0]}, "
                  f"wP={best_weights[1]}, wC={best_weights[2]}, recall={best_recall:.4f}")

    print(f"\n========== 网格搜索结果 ==========")
    results_all.sort(key=lambda x: x[3], reverse=True)

    print(f"\n{'排名':<5} {'wA':<8} {'wP':<8} {'wC':<8} {'召回率':<10}")
    print("-" * 45)
    for rank, (wA, wP, wC, rec) in enumerate(results_all[:10], 1):
        marker = " <<<" if rank == 1 else ""
        print(f"{rank:<5} {wA:<8.2f} {wP:<8.2f} {wC:<8.2f} {rec:<10.4f}{marker}")

    print(f"\n最终推荐: wA={best_weights[0]}, wP={best_weights[1]}, wC={best_weights[2]}, "
          f"召回率={best_recall:.4f} ({int(best_recall * len(ground_truth))}/{len(ground_truth)})")
    print(f"\n公式: S = {best_weights[0]}A + {best_weights[1]}P + {best_weights[2]}C")

    return best_weights, best_recall


if __name__ == "__main__":
    import sys

    if not os.path.isdir(VIDEO_DIR):
        print(f"错误: 视频目录不存在 {VIDEO_DIR}")
        print("请将 42 段泄漏视频放入该目录，或修改 VIDEO_DIR 路径")
        sys.exit(1)

    videos = sorted([
        os.path.join(VIDEO_DIR, f)
        for f in os.listdir(VIDEO_DIR)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ])

    if not videos:
        print(f"错误: {VIDEO_DIR} 中未找到视频文件")
        sys.exit(1)

    print(f"找到 {len(videos)} 段视频，开始处理...\n")
    grid_search(videos)