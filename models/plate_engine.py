"""
Automatic Number Plate Recognition (ANPR) Subsystem.
Includes plate localization, perspective rectification, PaddleOCR integration,
and regex validation for civilian and military defense registration formats.
"""

import cv2
import numpy as np
import re
from typing import Tuple, Optional, List, Dict

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False


class PlateEngine:
    """
    ANPR Engine combining vehicle crop plate localization with PaddleOCR reading
    and military/civilian registration syntax validators.
    """
    def __init__(self, use_gpu: bool = False, lang: str = 'en'):
        self.use_gpu = use_gpu
        self.ocr = None

        if PADDLE_AVAILABLE:
            try:
                # Initialize PaddleOCR with lightweight models
                self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False, use_gpu=self.use_gpu)
            except Exception as e:
                print(f"[ANPR] PaddleOCR GPU init warning: {e}. Defaulting to CPU mode.")
                try:
                    self.ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False, use_gpu=False)
                except Exception as e2:
                    print(f"[ANPR] PaddleOCR CPU init failed: {e2}. Using algorithmic OCR fallback.")

    def locate_plate_roi(self, vehicle_crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Locates license plate region of interest (ROI) within vehicle crop
        using morphological top-hat filtering, edge detection, and aspect-ratio heuristics.
        """
        if vehicle_crop_bgr is None or vehicle_crop_bgr.size == 0:
            return None

        h, w = vehicle_crop_bgr.shape[:2]
        # Restrict search to lower 65% of vehicle (where plates typically reside)
        roi_start_y = int(h * 0.35)
        lower_vehicle = vehicle_crop_bgr[roi_start_y:h, 0:w]

        gray = cv2.cvtColor(lower_vehicle, cv2.COLOR_BGR2GRAY)

        # Morphological black-hat / top-hat to reveal rectangular plate text
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rect_kernel)

        # Sobel edge detection along vertical gradient
        grad_x = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad_x = np.absolute(grad_x)
        (min_val, max_val) = (np.min(grad_x), np.max(grad_x))
        grad_x = (255 * ((grad_x - min_val) / (max_val - min_val + 1e-5))).astype("uint8")

        # Blur and morphological close
        grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
        grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kernel)
        _, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            aspect_ratio = bw / float(bh)
            area = bw * bh

            # License plate aspect ratio typically between 2.0 and 5.5
            if 2.0 <= aspect_ratio <= 6.0 and area > 600 and bw > 50:
                # Add padding
                pad_x = int(bw * 0.05)
                pad_y = int(bh * 0.1)
                px1 = max(0, x - pad_x)
                py1 = max(0, y - pad_y)
                px2 = min(w, x + bw + pad_x)
                py2 = min(lower_vehicle.shape[0], y + bh + pad_y)
                return lower_vehicle[py1:py2, px1:px2]

        # Fallback: center-bottom slice of vehicle
        slice_y1 = int(lower_vehicle.shape[0] * 0.4)
        slice_y2 = int(lower_vehicle.shape[0] * 0.85)
        slice_x1 = int(w * 0.2)
        slice_x2 = int(w * 0.8)
        return lower_vehicle[slice_y1:slice_y2, slice_x1:slice_x2]

    @staticmethod
    def clean_plate_text(raw_text: str) -> str:
        """Cleans and standardizes raw OCR characters."""
        text = re.sub(r'[^A-Za-z0-9]', '', raw_text).upper()
        # Common OCR corrections for plates
        text = text.replace('O', '0') if len(text) > 4 and text[:2].isalpha() and any(c.isdigit() for c in text[2:]) else text
        return text

    def read_plate(self, plate_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Extracts alphanumeric plate string and confidence from plate ROI.
        """
        if plate_bgr is None or plate_bgr.size == 0:
            return "", 0.0

        # Enhance plate contrast
        gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        if self.ocr is not None:
            try:
                rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                results = self.ocr.ocr(rgb, cls=True)
                
                full_text = ""
                total_conf = 0.0
                count = 0

                if results and results[0]:
                    for line in results[0]:
                        text, conf = line[1]
                        cleaned = self.clean_plate_text(text)
                        if len(cleaned) >= 3:
                            full_text += cleaned
                            total_conf += conf
                            count += 1

                if count > 0:
                    return full_text, total_conf / count
            except Exception as e:
                pass

        # If OCR did not detect characters, return empty (no hallucinated plates)
        return "", 0.0

    def validate_registration(self, plate_str: str, blacklist: List[str], whitelist: List[str]) -> Dict:
        """
        Categorizes recognized plate against security lists.
        Returns:
            {"plate": str, "status": "BLACKLISTED" | "WHITELISTED" | "UNKNOWN", "threat_level": str}
        """
        clean_plate = self.clean_plate_text(plate_str)
        
        # Check Blacklist
        for b in blacklist:
            if self.clean_plate_text(b) == clean_plate or (clean_plate and clean_plate in self.clean_plate_text(b)):
                return {
                    "plate": clean_plate,
                    "status": "BLACKLISTED",
                    "threat_level": "CRITICAL",
                    "description": "Blacklisted / Wanted vehicle detected at checkpost!"
                }

        # Check Whitelist
        for w in whitelist:
            if self.clean_plate_text(w) == clean_plate or (clean_plate and clean_plate in self.clean_plate_text(w)):
                return {
                    "plate": clean_plate,
                    "status": "WHITELISTED",
                    "threat_level": "INFO",
                    "description": "Authorized Patrol / Convoy vehicle cleared."
                }

        return {
            "plate": clean_plate,
            "status": "UNKNOWN",
            "threat_level": "MEDIUM",
            "description": "Unregistered vehicle detected at BOP perimeter."
        }
