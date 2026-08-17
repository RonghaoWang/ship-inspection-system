from __future__ import annotations
from collections import defaultdict, deque
from typing import Optional

import numpy as np


def _simple_dbscan(points: np.ndarray, eps: float, min_samples: int):
    if len(points) == 0:
        return np.array([], dtype=np.int32)

    n = len(points)
    labels = np.full(n, -1, dtype=np.int32)

    neighbors = []
    for i in range(n):
        dists = np.linalg.norm(points - points[i], axis=1)
        nbrs = set(np.where(dists <= eps)[0])
        nbrs.discard(i)
        neighbors.append(nbrs)

    cluster_id = 0
    for i in range(n):
        if labels[i] != -1 or len(neighbors[i]) < min_samples:
            continue

        labels[i] = cluster_id
        seeds = list(neighbors[i])
        visited = {i}

        while seeds:
            q = seeds.pop()
            if q in visited:
                continue
            visited.add(q)

            if len(neighbors[q]) >= min_samples:
                for nb in neighbors[q]:
                    if nb not in visited and labels[nb] == -1:
                        seeds.append(nb)

            if labels[q] == -1:
                labels[q] = cluster_id

        cluster_id += 1

    return labels


class LeakDetector:
    def __init__(
        self,
        window_size: int = 30,
        cluster_eps: float = 80.0,
        cluster_min_samples: int = 3,
        min_track_age: int = 5,
        leak_score_threshold: float = 0.5,
        leak_confirm_frames: int = 3,
        w_aggregation: float = 0.40,
        w_persistence: float = 0.35,
        w_consistency: float = 0.25,
        smooth_alpha: float = 0.7,
    ):
        self.window_size = window_size
        self.cluster_eps = cluster_eps
        self.cluster_min_samples = cluster_min_samples
        self.min_track_age = min_track_age
        self.leak_score_threshold = leak_score_threshold
        self.leak_confirm_frames = leak_confirm_frames
        self.w_a = w_aggregation
        self.w_p = w_persistence
        self.w_c = w_consistency
        self.smooth_alpha = smooth_alpha

        self.frame_id = 0
        self.smoothed_score = 0.0
        self.leak_counter = 0
        self.is_leaking = False
        self.leak_region: Optional[tuple[float, float, float, float]] = None

        self.frame_has_detection: deque[bool] = deque(maxlen=window_size)
        self.track_history: dict[int, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self.track_age: dict[int, int] = {}
        self.track_last_pos: dict[int, tuple[float, float]] = {}
        self.track_velocities: dict[int, tuple[float, float]] = {}
        self.track_origin: dict[int, tuple[float, float]] = {}

    def reset(self):
        self.frame_id = 0
        self.smoothed_score = 0.0
        self.leak_counter = 0
        self.is_leaking = False
        self.leak_region = None
        self.frame_has_detection.clear()
        self.track_history.clear()
        self.track_age.clear()
        self.track_last_pos.clear()
        self.track_velocities.clear()
        self.track_origin.clear()

    def compute_aggregation(self, origins: np.ndarray) -> float:
        n = len(origins)
        if n < 2:
            return 0.0

        if n < self.cluster_min_samples:
            nn_dists = []
            for i in range(n):
                others = np.delete(origins, i, axis=0)
                dists = np.linalg.norm(others - origins[i], axis=1)
                nn_dists.append(np.min(dists))
            mean_nn = np.mean(nn_dists)
            return max(0.0, 1.0 - mean_nn / (self.cluster_eps * 2))

        clustering = _simple_dbscan(
            origins, eps=self.cluster_eps, min_samples=self.cluster_min_samples
        )
        labels = clustering

        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        if not unique_labels:
            return 0.0

        largest_cluster_size = max(
            np.sum(labels == label) for label in unique_labels
        )
        cluster_ratio = largest_cluster_size / n

        cluster_points = origins[labels != -1]
        if len(cluster_points) < 2:
            return cluster_ratio * 0.5

        all_labels_valid = labels[labels != -1]
        main_label = max(unique_labels, key=lambda l: np.sum(all_labels_valid == l))
        main_cluster = cluster_points[all_labels_valid == main_label]

        nn_dists = []
        for i in range(len(main_cluster)):
            others = np.delete(main_cluster, i, axis=0)
            if len(others) > 0:
                dists = np.linalg.norm(others - main_cluster[i], axis=1)
                nn_dists.append(np.min(dists))

        if not nn_dists:
            return cluster_ratio * 0.5

        mean_nn = np.mean(nn_dists)
        concentration = max(0.0, 1.0 - mean_nn / self.cluster_eps)
        return cluster_ratio * concentration

    def compute_persistence(self, active_track_ids: list[int]) -> float:
        if not self.frame_has_detection:
            p_det = 0.0
        else:
            p_det = sum(self.frame_has_detection) / len(self.frame_has_detection)

        if len(active_track_ids) == 0:
            p_stable = 0.0
        else:
            stable_count = sum(
                1 for tid in active_track_ids
                if self.track_age.get(tid, 0) >= self.min_track_age
            )
            p_stable = stable_count / len(active_track_ids)

        return 0.5 * p_det + 0.5 * p_stable

    def compute_consistency(
        self, active_track_ids: list[int], centers: np.ndarray
    ) -> float:
        velocities = []
        for i, tid in enumerate(active_track_ids):
            vel = self.track_velocities.get(tid)
            if vel is not None:
                velocities.append(vel)
            elif self.track_age.get(tid, 0) >= 2:
                history = self.track_history[tid]
                if len(history) >= 2:
                    pos_curr = np.array(history[-1])
                    pos_prev = np.array(history[-2])
                    vel = pos_curr - pos_prev
                    vel_norm = np.linalg.norm(vel)
                    if vel_norm > 1e-6:
                        vel = vel / vel_norm
                    velocities.append(tuple(vel))

        if len(velocities) < 2:
            return 0.5

        vel_array = np.array(velocities)
        mean_vel = np.mean(vel_array, axis=0)
        mean_norm = np.linalg.norm(mean_vel)
        if mean_norm < 1e-6:
            return 0.5

        mean_vel = mean_vel / mean_norm
        similarities = np.dot(vel_array, mean_vel)
        similarities = np.clip(similarities, -1.0, 1.0)

        return float(np.mean(similarities) * 0.5 + 0.5)

    def compute_leak_score(
        self,
        centers: np.ndarray,
        origins: np.ndarray,
        active_track_ids: list[int],
    ) -> dict:
        a = self.compute_aggregation(origins)
        p = self.compute_persistence(active_track_ids)
        c = self.compute_consistency(active_track_ids, centers)

        raw_score = self.w_a * a + self.w_p * p + self.w_c * c
        self.smoothed_score = (
            self.smooth_alpha * raw_score
            + (1 - self.smooth_alpha) * self.smoothed_score
        )

        if self.smoothed_score >= self.leak_score_threshold:
            self.leak_counter += 1
        else:
            self.leak_counter = max(0, self.leak_counter - 1)

        self.is_leaking = self.leak_counter >= self.leak_confirm_frames

        leak_source = None
        if self.is_leaking and len(origins) >= self.cluster_min_samples:
            clustering = _simple_dbscan(
                origins, eps=self.cluster_eps, min_samples=self.cluster_min_samples
            )
            labels = clustering
            valid_labels = set(labels) - {-1}
            if valid_labels:
                main_label = max(valid_labels, key=lambda l: np.sum(labels == l))
                main_points = origins[labels == main_label]
                leak_cx = float(np.mean(main_points[:, 0]))
                leak_cy = float(np.mean(main_points[:, 1]))
                leak_source = (leak_cx, leak_cy)

        return {
            "raw_score": round(raw_score, 4),
            "smoothed_score": round(self.smoothed_score, 4),
            "aggregation": round(a, 4),
            "persistence": round(p, 4),
            "consistency": round(c, 4),
            "is_leaking": self.is_leaking,
            "leak_source": leak_source,
            "num_bubbles": len(centers),
        }

    def update(
        self, boxes_xyxy, track_ids, confs, classes
    ) -> dict:
        self.frame_id += 1

        n = len(boxes_xyxy)
        if n == 0:
            self.frame_has_detection.append(False)
            return self.compute_leak_score(
                np.array([]).reshape(0, 2),
                np.array([]).reshape(0, 2),
                [],
            )

        self.frame_has_detection.append(True)

        centers = np.zeros((n, 2), dtype=np.float32)
        origins = np.zeros((n, 2), dtype=np.float32)
        active_track_ids = []

        for i in range(n):
            x1, y1, x2, y2 = boxes_xyxy[i]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            centers[i] = [cx, cy]

            tid = int(track_ids[i])
            active_track_ids.append(tid)

            if tid in self.track_last_pos:
                prev_pos = self.track_last_pos[tid]
                vel = (cx - prev_pos[0], cy - prev_pos[1])
                vel_norm = np.sqrt(vel[0] ** 2 + vel[1] ** 2)
                if vel_norm > 1e-6:
                    vel = (vel[0] / vel_norm, vel[1] / vel_norm)
                self.track_velocities[tid] = vel
                self.track_age[tid] = self.track_age.get(tid, 0) + 1
            else:
                self.track_age[tid] = 1
                self.track_velocities.pop(tid, None)
                self.track_origin[tid] = (cx, cy)

            self.track_last_pos[tid] = (cx, cy)
            self.track_history[tid].append((cx, cy))

        for i, tid in enumerate(active_track_ids):
            org = self.track_origin.get(tid)
            if org is not None:
                origins[i] = org
            else:
                origins[i] = centers[i]

        aged_out = [
            tid for tid in self.track_age
            if tid not in active_track_ids
            and self.frame_id - self.track_age.get(tid, 0) > self.window_size
        ]
        for tid in aged_out:
            self.track_history.pop(tid, None)
            self.track_age.pop(tid, None)
            self.track_last_pos.pop(tid, None)
            self.track_velocities.pop(tid, None)
            self.track_origin.pop(tid, None)

        result = self.compute_leak_score(centers, origins, active_track_ids)
        result["centers"] = [(float(cx), float(cy)) for cx, cy in centers]
        return result