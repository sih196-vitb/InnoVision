"""
Asynchronous Behavioral Anomaly Specialist Worker.
Buffers a sliding temporal window of video frames sampled at 2-4 FPS from 'queue:anomaly',
computes spatio-temporal motion dynamics, and detects abnormal tactical maneuvers or perimeter rushes.
"""

import cv2
import numpy as np
import base64
import time
from typing import Dict, Any, Optional
from workers.base_worker import BaseWorker, MessageBroker
from models.anomaly_detector import BehavioralAnomalyDetector
from core.alert_manager import AlertManager
from config.settings import (
    QUEUE_ANOMALY,
    ANOMALY_WINDOW_SIZE,
    ANOMALY_SAMPLE_FPS,
    ANOMALY_SCORE_THRESHOLD
)


class AnomalyWorker(BaseWorker):
    """
    Worker analyzing temporal sliding sequences to detect anomalous behavioral signatures.
    """
    def __init__(self, broker: Optional[MessageBroker] = None, alert_manager: Optional[AlertManager] = None):
        super().__init__(name="Anomaly-Worker", queue_name=QUEUE_ANOMALY, broker=broker)
        self.detector = BehavioralAnomalyDetector(
            window_size=ANOMALY_WINDOW_SIZE,
            sample_fps=ANOMALY_SAMPLE_FPS,
            score_threshold=ANOMALY_SCORE_THRESHOLD
        )
        self.alert_manager = alert_manager if alert_manager is not None else AlertManager(broker=self.broker)

    @staticmethod
    def decode_base64_to_image(b64_str: str) -> Optional[np.ndarray]:
        """Decodes base64 string to OpenCV BGR image."""
        try:
            img_bytes = base64.b64decode(b64_str)
            nparr = np.frombuffer(img_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def process_item(self, item: Dict[str, Any]):
        """
        Processes sampled frame payload:
        item = {
            "camera_id": str,
            "timestamp": float,
            "frame_base64": str
        }
        """
        frame_bgr = self.decode_base64_to_image(item.get("frame_base64", ""))
        if frame_bgr is None or frame_bgr.size == 0:
            return

        ts = item.get("timestamp", time.time())
        self.detector.push_frame(frame_bgr, timestamp=ts)

        # Evaluate current temporal sequence
        analysis = self.detector.compute_anomaly_score()

        if analysis.get("ready") and analysis.get("is_anomalous"):
            score = analysis["anomaly_score"]
            atype = analysis["anomaly_type"]
            energy = analysis["motion_energy"]

            alert_payload = {
                "alert_type": "BEHAVIORAL_ANOMALY",
                "threat_level": "CRITICAL" if score > 0.85 else "HIGH",
                "camera_id": item.get("camera_id", "BOP-ALPHA"),
                "anomaly_type": atype,
                "anomaly_score": score,
                "motion_energy": energy,
                "message": f"ANOMALY DETECTED: [{atype}] - Kinetic Energy: {energy:.1f} (Score: {score*100:.0f}%)",
                "details": analysis
            }

            self.alert_manager.dispatch(alert_payload, crop_bgr=frame_bgr)
            print(f"[Anomaly-Worker] ALERT! {atype} | Score: {score:.2f} | Kinetic Energy: {energy:.1f}")


if __name__ == "__main__":
    worker = AnomalyWorker()
    worker.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        worker.stop()
