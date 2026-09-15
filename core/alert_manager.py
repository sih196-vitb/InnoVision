"""
Central Alert Management and Dispatch Engine.
Standardizes alert payloads, applies temporal suppression filters (deduplication),
logs events to disk, and pushes notifications to Redis pub/sub and in-memory brokers.
"""

import time
import json
import base64
import cv2
import numpy as np
import threading
from pathlib import Path
from typing import Dict, Optional, List, Callable
from config.settings import LOGS_DIR, QUEUE_ALERTS, CHANNEL_ALERTS


class AlertManager:
    """
    Orchestrates alert aggregation, rate limiting, persistence, and distribution.
    """
    def __init__(self, broker=None, suppression_window_sec: float = 4.0):
        self.broker = broker
        self.suppression_window_sec = suppression_window_sec
        self.recent_alerts: Dict[str, float] = {}  # key: f"{alert_type}:{track_id or entity}" -> last_time
        self.alert_subscribers: List[Callable[[Dict], None]] = []
        self.log_file = LOGS_DIR / "alerts.jsonl"
        self._file_lock = threading.Lock()

    def register_callback(self, callback: Callable[[Dict], None]):
        """Registers a direct Python callable (e.g., Flask-SocketIO emit handler)."""
        self.alert_subscribers.append(callback)

    def is_suppressed(self, dedup_key: str, now: float) -> bool:
        """Determines if this entity has triggered an identical alert recently."""
        last_time = self.recent_alerts.get(dedup_key, 0.0)
        if (now - last_time) < self.suppression_window_sec:
            return True
        self.recent_alerts[dedup_key] = now
        return False

    @staticmethod
    def encode_crop_to_base64(crop_bgr: np.ndarray) -> str:
        """Encodes an OpenCV image crop to a base64 JPEG string for dashboard display."""
        if crop_bgr is None or crop_bgr.size == 0:
            return ""
        # Resize crop if too large to conserve WebSocket bandwidth
        h, w = crop_bgr.shape[:2]
        if max(h, w) > 250:
            scale = 250.0 / max(h, w)
            crop_bgr = cv2.resize(crop_bgr, (int(w * scale), int(h * scale)))
        
        _, buffer = cv2.imencode('.jpg', crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return base64.b64encode(buffer).decode('utf-8')

    def dispatch(self, alert_data: Dict, crop_bgr: Optional[np.ndarray] = None) -> Optional[Dict]:
        """
        Dispatches an alert after verifying deduplication window.
        """
        now = time.time()
        alert_type = alert_data.get("alert_type", "GENERIC_ALERT")
        track_id = alert_data.get("track_id", alert_data.get("fence_id", "default"))
        dedup_key = f"{alert_type}:{track_id}"

        if self.is_suppressed(dedup_key, now):
            return None

        # Assemble full normalized payload
        payload = {
            "id": f"ALT-{int(now * 1000)}",
            "timestamp": now,
            "datetime_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "alert_type": alert_type,
            "threat_level": alert_data.get("threat_level", "HIGH"),
            "camera_id": alert_data.get("camera_id", "BOP-ALPHA-SECTOR-01"),
            "message": alert_data.get("message", "Security event detected"),
            "details": alert_data,
            "crop_base64": self.encode_crop_to_base64(crop_bgr) if crop_bgr is not None else ""
        }

        # 1. Log to JSON Lines file
        try:
            with self._file_lock:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload) + "\n")
        except Exception as e:
            print(f"[AlertManager] Logging error: {e}")

        # 2. Push to Redis broker (or in-memory fallback)
        if self.broker is not None:
            try:
                self.broker.push(QUEUE_ALERTS, payload)
                self.broker.publish(CHANNEL_ALERTS, payload)
            except Exception as e:
                print(f"[AlertManager] Broker dispatch error: {e}")

        # 3. Direct subscribers (WebSockets)
        for cb in self.alert_subscribers:
            try:
                cb(payload)
            except Exception as e:
                print(f"[AlertManager] Subscriber callback error: {e}")

        return payload
