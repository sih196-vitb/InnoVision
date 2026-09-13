"""
Asynchronous Automatic Number Plate Recognition (ANPR) Specialist Worker.
Pulls vehicle crops from 'queue:plates', isolates plate ROIs, extracts alphanumeric
text via PaddleOCR, and validates registration against security registries.
"""

import cv2
import numpy as np
import base64
import time
from typing import Dict, Any, Optional
from workers.base_worker import BaseWorker, MessageBroker
from models.plate_engine import PlateEngine
from core.alert_manager import AlertManager
from config.settings import (
    QUEUE_PLATES,
    BLACKLIST_PLATES,
    WHITELIST_PLATES,
    ANPR_OCR_LANG
)


class ANPRWorker(BaseWorker):
    """
    Worker dedicated to ANPR tasks. Decoupled from video ingestion to prevent frame drops.
    """
    def __init__(self, broker: Optional[MessageBroker] = None, alert_manager: Optional[AlertManager] = None):
        super().__init__(name="ANPR-Worker", queue_name=QUEUE_PLATES, broker=broker)
        self.plate_engine = PlateEngine(lang=ANPR_OCR_LANG)
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
        Processes cropped vehicle payload:
        item = {
            "track_id": str,
            "camera_id": str,
            "timestamp": float,
            "bbox": [x1, y1, x2, y2],
            "crop_base64": str
        }
        """
        crop_bgr = self.decode_base64_to_image(item.get("crop_base64", ""))
        if crop_bgr is None or crop_bgr.size == 0:
            return

        # 1. Locate license plate candidate within vehicle crop
        plate_roi = self.plate_engine.locate_plate_roi(crop_bgr)
        if plate_roi is None or plate_roi.size == 0:
            plate_roi = crop_bgr

        # 2. Extract alphanumeric text using PaddleOCR / fallback
        plate_text, conf = self.plate_engine.read_plate(plate_roi)
        
        if not plate_text or len(plate_text) < 3:
            return

        # 3. Validate registration string against blacklist & whitelist
        verification = self.plate_engine.validate_registration(
            plate_text,
            BLACKLIST_PLATES,
            WHITELIST_PLATES
        )

        status = verification["status"]
        threat_level = verification["threat_level"]
        description = verification["description"]

        # 4. Dispatch alert if Blacklisted or Unknown vehicle detected
        alert_payload = {
            "alert_type": "ANPR_DETECTION",
            "threat_level": threat_level,
            "camera_id": item.get("camera_id", "BOP-ALPHA"),
            "track_id": item.get("track_id", "V-001"),
            "plate_number": plate_text,
            "registration_status": status,
            "confidence": round(conf, 2),
            "message": f"ANPR: [{status}] Plate '{plate_text}' ({conf*100:.0f}%) - {description}",
            "details": {
                "plate": plate_text,
                "status": status,
                "vehicle_bbox": item.get("bbox")
            }
        }

        # Dispatch via alert manager
        self.alert_manager.dispatch(alert_payload, crop_bgr=plate_roi)
        print(f"[ANPR-Worker] Processed Plate: {plate_text} | Status: {status} | Conf: {conf:.2f}")


if __name__ == "__main__":
    worker = ANPRWorker()
    worker.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        worker.stop()
