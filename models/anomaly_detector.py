"""
Behavioral Anomaly Detection Engine.
Buffers a sliding temporal window of surveillance frames sampled at 2-4 FPS,
computes spatio-temporal optical flow energy and motion entropy, and produces
anomaly confidence scores for perimeter intrusions, rapid rushes, and abnormal maneuvers.
"""

import cv2
import numpy as np
from collections import deque
from typing import Tuple, Dict, List, Optional
import time


class BehavioralAnomalyDetector:
    """
    Spatio-temporal anomaly analyzer operating over a sliding temporal buffer.
    Computes optical flow vectors, kinetic energy gradients, and directional variance.
    """
    def __init__(self, window_size: int = 16, sample_fps: int = 3, score_threshold: float = 0.68):
        self.window_size = window_size
        self.sample_fps = sample_fps
        self.score_threshold = score_threshold
        
        self.frame_buffer = deque(maxlen=window_size)
        self.timestamp_buffer = deque(maxlen=window_size)
        self.last_sample_time = 0.0
        self.min_sample_interval = 1.0 / float(sample_fps)

        # Optical flow tracker state
        self.prev_gray = None

    def should_sample(self, current_time: float) -> bool:
        """Determines if enough time has elapsed to capture a frame for the 2-4 FPS buffer."""
        return (current_time - self.last_sample_time) >= (self.min_sample_interval - 1e-4)

    def push_frame(self, frame_bgr: np.ndarray, timestamp: Optional[float] = None, force: bool = False) -> None:
        """Adds a downsampled grayscale frame to the temporal sliding window."""
        if frame_bgr is None or frame_bgr.size == 0:
            return

        now = timestamp if timestamp is not None else time.time()
        if not force and not self.should_sample(now):
            return

        self.last_sample_time = now
        # Downscale for ultra-fast spatio-temporal flow calculation (< 3ms)
        small_gray = cv2.cvtColor(cv2.resize(frame_bgr, (160, 90)), cv2.COLOR_BGR2GRAY)
        self.frame_buffer.append(small_gray)
        self.timestamp_buffer.append(now)

    def compute_anomaly_score(self) -> Dict:
        """
        Processes current sliding temporal window to evaluate motion dynamics and anomaly score.
        Returns:
            {
                "anomaly_score": float (0.0 to 1.0),
                "is_anomalous": bool,
                "anomaly_type": str,
                "motion_energy": float,
                "directional_entropy": float,
                "ready": bool
            }
        """
        # Require at least 8 frames for stable temporal dynamics
        if len(self.frame_buffer) < 8:
            return {
                "anomaly_score": 0.0,
                "is_anomalous": False,
                "anomaly_type": "BUFFERING",
                "motion_energy": 0.0,
                "directional_entropy": 0.0,
                "ready": False
            }

        frames = list(self.frame_buffer)
        n = len(frames)

        flow_magnitudes = []
        flow_angles = []

        # Compute dense optical flow between successive temporal samples
        for i in range(1, n):
            prev_f = frames[i - 1]
            curr_f = frames[i]

            flow = cv2.calcOpticalFlowFarneback(
                prev_f, curr_f, None,
                pyr_scale=0.5, levels=2, winsize=13,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0
            )

            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_magnitudes.append(np.mean(mag))
            flow_angles.append(ang)

        avg_magnitude = float(np.mean(flow_magnitudes))
        magnitude_std = float(np.std(flow_magnitudes))
        max_surge = float(np.max(flow_magnitudes))

        # Kinetic acceleration / jerk: difference in velocity between frame pairs
        accel_spikes = [abs(flow_magnitudes[i] - flow_magnitudes[i-1]) for i in range(1, len(flow_magnitudes))]
        max_accel = float(np.max(accel_spikes)) if accel_spikes else 0.0

        # Angular entropy (detects erratic, disorderly movement vs smooth convoy/patrol)
        if len(flow_angles) > 0:
            hist, _ = np.histogram(flow_angles[-1], bins=8, range=(0, 2 * np.pi))
            prob = hist / (np.sum(hist) + 1e-6)
            prob = prob[prob > 0]
            entropy = -np.sum(prob * np.log2(prob))  # 0.0 (unidirectional) to ~3.0 (chaotic)
        else:
            entropy = 0.0

        # Composite anomaly score formula normalized into [0.0, 1.0]
        # Baseline normal patrol: avg_magnitude ~ 0.5 - 2.0, max_accel < 1.0, entropy < 2.0
        norm_mag = np.tanh(avg_magnitude / 4.0)
        norm_accel = np.tanh(max_accel / 3.0)
        norm_entropy = min(1.0, entropy / 2.8)

        # Weighted anomaly index
        raw_score = 0.40 * norm_accel + 0.35 * norm_mag + 0.25 * norm_entropy
        anomaly_score = float(np.clip(raw_score, 0.0, 1.0))

        # Anomaly classification logic
        anomaly_type = "NORMAL_PATROL"
        if anomaly_score >= self.score_threshold:
            if norm_accel > 0.75:
                anomaly_type = "RAPID_PERIMETER_RUSH"
            elif norm_entropy > 0.80:
                anomaly_type = "CHAOTIC_FENCE_PROWLING"
            else:
                anomaly_type = "ERRATIC_TACTICAL_MANEUVER"

        return {
            "anomaly_score": round(anomaly_score, 3),
            "is_anomalous": anomaly_score >= self.score_threshold,
            "anomaly_type": anomaly_type,
            "motion_energy": round(avg_magnitude, 2),
            "directional_entropy": round(entropy, 2),
            "ready": True
        }
