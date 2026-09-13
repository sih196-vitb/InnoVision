"""
Automatic Perimeter & Boundary Detection Engine.
Analyzes CCTV surveillance scenes to automatically extract:
1. Physical boundary / horizon lines (restricted perimeter fence line).
2. Primary transit corridors & checkpost roadways via Hough vector transforms and edge gradients.
Automatically synthesizes normalized Shapely virtual fence polygon configurations.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


class AutoPerimeterDetector:
    """
    Computer vision scene analyzer that automatically estimates perimeter fence lines
    and checkpost approach corridors without requiring manual calibration.
    """
    def __init__(self, frame_width: int = 1280, frame_height: int = 720):
        self.frame_width = frame_width
        self.frame_height = frame_height

    def detect_perimeters(self, frame_bgr: np.ndarray) -> List[Dict]:
        """
        Processes a surveillance frame and produces automated tactical fence zones.
        Returns:
            List[Dict] in standard virtual fence configuration format.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return self._fallback_perimeters()

        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 1. Estimate Horizon / Physical Fence Line across upper terrain
        # Calculate horizontal edge projection
        sobel_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        abs_sobel_y = np.absolute(sobel_y)
        row_energy = np.mean(abs_sobel_y, axis=1)

        # Search for primary perimeter transition between 20% and 45% of frame height
        search_start = int(h * 0.20)
        search_end = int(h * 0.45)
        if search_end > search_start:
            horizon_y = search_start + int(np.argmax(row_energy[search_start:search_end]))
        else:
            horizon_y = int(h * 0.30)

        # Convert to normalized coordinate (0.0 to 1.0)
        norm_h_y = round(horizon_y / float(h), 3)
        fence_depth = 0.15  # Buffer corridor thickness

        # 2. Estimate Transit Roadway / Approach Corridor via Hough Transform
        edges = cv2.Canny(blurred, 50, 150)
        # Focus on lower 60% of frame for road / vehicle corridor
        roi_edges = np.zeros_like(edges)
        roi_edges[horizon_y:, :] = edges[horizon_y:, :]

        lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=40, minLineLength=60, maxLineGap=20)
        
        left_lane_x = int(w * 0.25)
        right_lane_x = int(w * 0.75)
        apex_left_x = int(w * 0.40)
        apex_right_x = int(w * 0.60)

        if lines is not None and len(lines) > 0:
            lines = lines.reshape(-1, 4)
            left_slopes = []
            right_slopes = []
            for line in lines:
                x1, y1, x2, y2 = int(line[0]), int(line[1]), int(line[2]), int(line[3])
                if x2 == x1:
                    continue
                slope = (y2 - y1) / float(x2 - x1)
                # Left lane boundaries have negative slope in image coordinates
                if slope < -0.4 and x1 < w * 0.5:
                    left_slopes.append(line)
                # Right lane boundaries have positive slope in image coordinates
                elif slope > 0.4 and x2 > w * 0.5:
                    right_slopes.append(line)

            if left_slopes:
                avg_l = np.mean(left_slopes, axis=0).astype(int)
                left_lane_x = int(np.clip(avg_l[0], w * 0.10, w * 0.45))
            if right_slopes:
                avg_r = np.mean(right_slopes, axis=0).astype(int)
                right_lane_x = int(np.clip(avg_r[2], w * 0.55, w * 0.90))

        # 3. Assemble Calibrated Automatic Perimeter Zones
        auto_zones = [
            {
                "id": "auto_zone_perimeter",
                "name": "Auto-Detected Northern Boundary (Restricted)",
                "zone_type": "RESTRICTED_PERIMETER",
                "color": "#EF4444",
                "is_auto": True,
                "polygon": [
                    [0.02, max(0.05, norm_h_y - 0.05)],
                    [0.98, max(0.05, norm_h_y - 0.07)],
                    [0.98, min(0.95, norm_h_y + fence_depth)],
                    [0.02, min(0.95, norm_h_y + fence_depth + 0.02)]
                ],
                "allowed_classes": [],
                "loiter_threshold_sec": 3.5
            },
            {
                "id": "auto_zone_corridor",
                "name": "Auto-Detected Transit Corridor & Barrier",
                "zone_type": "CHECKPOST_CONTROL",
                "color": "#F59E0B",
                "is_auto": True,
                "polygon": [
                    [round(apex_left_x / w, 3), min(0.90, norm_h_y + 0.08)],
                    [round(apex_right_x / w, 3), min(0.90, norm_h_y + 0.08)],
                    [round(right_lane_x / w, 3), 0.95],
                    [round(left_lane_x / w, 3), 0.95]
                ],
                "allowed_classes": [0, 2, 7],  # Person, Car, Truck
                "loiter_threshold_sec": 10.0
            }
        ]

        return auto_zones

    def _fallback_perimeters(self) -> List[Dict]:
        """Default synthetic tactical zones if frame analysis is unavailable."""
        return [
            {
                "id": "auto_zone_perimeter",
                "name": "Auto-Detected Northern Boundary (Restricted)",
                "zone_type": "RESTRICTED_PERIMETER",
                "color": "#EF4444",
                "is_auto": True,
                "polygon": [
                    [0.05, 0.20],
                    [0.95, 0.16],
                    [0.95, 0.40],
                    [0.05, 0.44]
                ],
                "allowed_classes": [],
                "loiter_threshold_sec": 4.0
            },
            {
                "id": "auto_zone_corridor",
                "name": "Auto-Detected Transit Corridor & Barrier",
                "zone_type": "CHECKPOST_CONTROL",
                "color": "#F59E0B",
                "is_auto": True,
                "polygon": [
                    [0.28, 0.48],
                    [0.72, 0.48],
                    [0.85, 0.92],
                    [0.15, 0.92]
                ],
                "allowed_classes": [0, 2, 7],
                "loiter_threshold_sec": 10.0
            }
        ]
