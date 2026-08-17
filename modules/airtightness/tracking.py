import os
import sys

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import ultralytics.nn.modules.conv as conv_module

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leak_detector import LeakDetector


def patch_conv_layer():
    def universal_forward(self, x):
        x = self.conv(x)
        if hasattr(self, 'bn') and self.bn is not None:
            x = self.bn(x)
        return self.act(x)

    conv_module.Conv.forward = universal_forward
    conv_module.Conv.forward_fuse = universal_forward


patch_conv_layer()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(SCRIPT_DIR, "weights", "best.pt")
video_path = os.path.join(SCRIPT_DIR, "your_video.mp4")

model = YOLO(model_path)

results_gen = model.track(
    source=video_path,
    conf=0.3,
    iou=0.5,
    persist=True,
    tracker="botsort.yaml",
    stream=True,
    show=False,
    save=False,
    imgsz=416,
)

detector = LeakDetector(
    window_size=30,
    cluster_eps=180.0,
    cluster_min_samples=3,
    min_track_age=5,
    leak_score_threshold=0.5,
    leak_confirm_frames=3,
    w_aggregation=0.2,
    w_persistence=0.5,
    w_consistency=0.3,
)

frame_w = None
frame_h = None
fps = 30
out_writer = None
output_path = os.path.join(SCRIPT_DIR, "leak_detection_output.avi")
fourcc = cv2.VideoWriter_fourcc(*"XVID")

for frame_idx, result in enumerate(results_gen):
    orig_img = result.plot()

    frame_h, frame_w = orig_img.shape[:2]
    if out_writer is None:
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
        track_ids = boxes.id.cpu().numpy() if boxes.is_track and hasattr(boxes.id, "cpu") else None
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
        classes = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls

        if track_ids is not None:
            leak_result = detector.update(xyxy, track_ids, confs, classes)

            if leak_result["is_leaking"]:
                cv2.putText(
                    orig_img,
                    f"LEAK DETECTED! Score: {leak_result['smoothed_score']:.3f}",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    orig_img,
                    f"A={leak_result['aggregation']:.3f} P={leak_result['persistence']:.3f} C={leak_result['consistency']:.3f}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

                leak_source = leak_result.get("leak_source")
                if leak_source is not None:
                    sx, sy = int(leak_source[0]), int(leak_source[1])
                    cv2.drawMarker(
                        orig_img,
                        (sx, sy),
                        (0, 0, 255),
                        markerType=cv2.MARKER_CROSS,
                        markerSize=20,
                        thickness=3,
                    )
                    cv2.circle(orig_img, (sx, sy), 12, (0, 0, 255), 2)
                    cv2.putText(
                        orig_img,
                        "LEAK SOURCE",
                        (sx - 55, sy - 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2,
                    )
            else:
                cv2.putText(
                    orig_img,
                    f"No Leak | Score: {leak_result['smoothed_score']:.3f}",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    orig_img,
                    f"A={leak_result['aggregation']:.3f} P={leak_result['persistence']:.3f} C={leak_result['consistency']:.3f}",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
        else:
            leak_result = detector.update(
                np.array([]).reshape(0, 4), np.array([]), np.array([]), np.array([])
            )
            cv2.putText(
                orig_img,
                "Tracking ID unavailable",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2,
            )
    else:
        leak_result = detector.update(
            np.array([]).reshape(0, 4), np.array([]), np.array([]), np.array([])
        )
        cv2.putText(
            orig_img,
            f"No Bubbles | Score: {leak_result['smoothed_score']:.3f}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Leak Detection", orig_img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    out_writer.write(orig_img)

    if frame_idx % 30 == 0:
        print(
            f"Frame {frame_idx}: "
            f"score={leak_result.get('smoothed_score', 0):.3f}, "
            f"leaking={leak_result.get('is_leaking', False)}, "
            f"bubbles={leak_result.get('num_bubbles', 0)}"
        )

if out_writer is not None:
    out_writer.release()
cv2.destroyAllWindows()
print(f"\nLeak detection output saved to: {output_path}")
print("Done.")