"""
Asynchronous Facial Recognition Subsystem (FRS) Specialist Worker.
Pulls person crops from 'queue:faces', isolates face regions, extracts 512-D embeddings,
and calculates cosine similarity against BOP security watchlists.
"""

import cv2
import numpy as np
import base64
import time
from typing import Dict, Any, Optional
from workers.base_worker import BaseWorker, MessageBroker
from models.face_engine import FaceEngine
from core.alert_manager import AlertManager
from config.settings import (
    QUEUE_FACES,
    WATCHLIST_FILE,
    FRS_MATCH_THRESHOLD,
    WEIGHTS_DIR
)


class FRSWorker(BaseWorker):
    """
    Asynchronous FRS worker executing deep facial matching decoupled from stream ingestion.
    """
    def __init__(self, broker: Optional[MessageBroker] = None, alert_manager: Optional[AlertManager] = None):
        super().__init__(name="FRS-Worker", queue_name=QUEUE_FACES, broker=broker)
        arcface_path = WEIGHTS_DIR / "arcface_r50.onnx"
        self.face_engine = FaceEngine(
            watchlist_path=WATCHLIST_FILE,
            arcface_onnx_path=arcface_path if arcface_path.exists() else None,
            match_threshold=FRS_MATCH_THRESHOLD
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
        Processes person crop payload:
        item = {
            "track_id": str,
            "camera_id": str,
            "timestamp": float,
            "bbox": [x1, y1, x2, y2],
            "crop_base64": str
        }
        """
        person_bgr = self.decode_base64_to_image(item.get("crop_base64", ""))
        if person_bgr is None or person_bgr.size == 0:
            return

        # 1. Refine face / head-and-shoulders region
        face_roi = self.face_engine.detect_face(person_bgr)
        if face_roi is None or face_roi.size == 0:
            face_roi = person_bgr

        # 2. Extract deep 512-D ArcFace embedding
        embedding = self.face_engine.extract_embedding(face_roi)

        # 3. Match against loaded security watchlist
        match = self.face_engine.match_watchlist(embedding)

        if match:
            category = match.get("category", "UNKNOWN")
            threat = match.get("threat_level", "HIGH")
            name = match.get("name", "Unknown Person")
            sim = match.get("similarity", 0.0)

            alert_payload = {
                "alert_type": "FRS_WATCHLIST_HIT",
                "threat_level": threat,
                "camera_id": item.get("camera_id", "BOP-ALPHA"),
                "track_id": item.get("track_id", "P-001"),
                "suspect_id": match.get("id"),
                "name": name,
                "category": category,
                "similarity": sim,
                "message": f"FRS MATCH: [{category}] {name} (Confidence: {sim*100:.1f}%) - {match.get('designation', '')}",
                "details": match
            }
            self.alert_manager.dispatch(alert_payload, crop_bgr=face_roi)
            print(f"[FRS-Worker] Watchlist Hit! {name} ({sim*100:.1f}%) Threat: {threat}")
        else:
            # Unidentified person in monitored zone
            pass


if __name__ == "__main__":
    worker = FRSWorker()
    worker.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        worker.stop()
