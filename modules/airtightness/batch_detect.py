import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
import ultralytics.nn.modules.conv as conv_module

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leak_detector import LeakDetector


def patch_conv_layer():
    def universal_forward(self, x):
        x = self.conv(x)
        if hasattr(self, "bn") and self.bn is not None:
            x = self.bn(x)
        return self.act(x)

    conv_module.Conv.forward = universal_forward
    conv_module.Conv.forward_fuse = universal_forward


patch_conv_layer()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SAMPLE_DIR = os.path.join(PROJECT_DIR, "sample")
MODEL_PATH = os.path.abspath(os.path.join(PROJECT_DIR, "result", "baseline", "weights", "best.pt"))
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "batch_results.csv")
SUMMARY_TXT = os.path.join(SCRIPT_DIR, "batch_summary.txt")

DETECTOR_CONFIG = {
    "window_size": 30,
    "cluster_eps": 180.0,
    "cluster_min_samples": 3,
    "min_track_age": 5,
    "leak_score_threshold": 0.5,
    "leak_confirm_frames": 3,
    "w_aggregation": 0.2,
    "w_persistence": 0.5,
    "w_consistency": 0.3,
}

LEAK_FRAME_RATIO_THRESHOLD = 0.05


def process_video(video_path, model):
    video_name = os.path.basename(video_path)
    detector = LeakDetector(**DETECTOR_CONFIG)

    results_gen = model.track(
        source=video_path,
        conf=0.3,
        iou=0.5,
        persist=True,
        tracker="botsort.yaml",
        stream=True,
        show=False,
        save=False,
        verbose=False,
        imgsz=416,
    )

    total_frames = 0
    frames_with_bubbles = 0
    frames_leaking = 0
    max_score = 0.0
    score_sum = 0.0
    score_count = 0
    ever_leaking = False

    for frame_idx, result in enumerate(results_gen):
        total_frames += 1
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
            track_ids = boxes.id.cpu().numpy() if boxes.is_track and hasattr(boxes.id, "cpu") else None
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
            classes = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls

            if track_ids is not None:
                leak_result = detector.update(xyxy, track_ids, confs, classes)
            else:
                leak_result = detector.update(
                    np.array([]).reshape(0, 4), np.array([]), np.array([]), np.array([])
                )
        else:
            leak_result = detector.update(
                np.array([]).reshape(0, 4), np.array([]), np.array([]), np.array([])
            )

        score = leak_result["smoothed_score"]
        max_score = max(max_score, score)
        if leak_result["num_bubbles"] > 0:
            frames_with_bubbles += 1
            score_sum += score
            score_count += 1

        if leak_result["is_leaking"]:
            frames_leaking += 1
            ever_leaking = True

    if total_frames == 0:
        return {
            "video": video_name,
            "total_frames": 0,
            "frames_with_bubbles": 0,
            "bubble_ratio": 0.0,
            "frames_leaking": 0,
            "leak_ratio": 0.0,
            "max_score": 0.0,
            "avg_score": 0.0,
            "ever_leaking": False,
            "detected_as_leak": False,
        }

    bubble_ratio = frames_with_bubbles / total_frames
    leak_ratio = frames_leaking / total_frames
    avg_score = score_sum / score_count if score_count > 0 else 0.0
    detected_as_leak = leak_ratio >= LEAK_FRAME_RATIO_THRESHOLD and ever_leaking

    return {
        "video": video_name,
        "total_frames": total_frames,
        "frames_with_bubbles": frames_with_bubbles,
        "bubble_ratio": round(bubble_ratio, 4),
        "frames_leaking": frames_leaking,
        "leak_ratio": round(leak_ratio, 4),
        "max_score": round(max_score, 4),
        "avg_score": round(avg_score, 4),
        "ever_leaking": ever_leaking,
        "detected_as_leak": detected_as_leak,
    }


def main():
    print(f"Model: {MODEL_PATH}")
    print(f"Sample dir: {SAMPLE_DIR}")
    print(f"Loading model...")
    model = YOLO(MODEL_PATH)

    video_files = sorted(
        [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(".mp4")]
    )
    if not video_files:
        print("No .mp4 videos found in sample/")
        return

    print(f"Found {len(video_files)} videos\n")

    results = []
    for idx, video_name in enumerate(video_files, 1):
        video_path = os.path.join(SAMPLE_DIR, video_name)
        print(f"[{idx}/{len(video_files)}] Processing: {video_name} ...", end=" ", flush=True)
        t0 = time.time()
        r = process_video(video_path, model)
        elapsed = time.time() - t0

        status = "LEAK" if r["detected_as_leak"] else "NO_LEAK"
        print(
            f"done ({elapsed:.1f}s) | frames={r['total_frames']} | "
            f"bubbles={r['bubble_ratio']:.3f} | "
            f"result={status} | max_score={r['max_score']:.3f}"
        )
        results.append(r)

    fieldnames = [
        "video", "total_frames", "frames_with_bubbles", "bubble_ratio",
        "frames_leaking", "leak_ratio", "max_score", "avg_score",
        "ever_leaking", "detected_as_leak",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    total_videos = len(results)
    detected_count = sum(1 for r in results if r["detected_as_leak"])
    accuracy = detected_count / total_videos if total_videos > 0 else 0.0
    ever_count = sum(1 for r in results if r["ever_leaking"])
    no_bubbles_count = sum(1 for r in results if r["frames_with_bubbles"] == 0)

    summary_lines = [
        "=" * 60,
        "  BATCH LEAK DETECTION SUMMARY",
        "=" * 60,
        f"  Total videos:           {total_videos}",
        f"  Detected as leaking:    {detected_count}",
        f"  Detection accuracy:     {accuracy:.2%}  ({detected_count}/{total_videos})",
        "",
        f"  Ever had leak signal:   {ever_count}",
        f"  Never had leak signal:  {total_videos - ever_count}",
        "",
        f"  Videos with no bubbles: {no_bubbles_count}",
        "",
        "  Per-video details saved to:",
        f"  {OUTPUT_CSV}",
        "=" * 60,
    ]

    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print("\n" + "\n".join(summary_lines))


if __name__ == "__main__":
    main()