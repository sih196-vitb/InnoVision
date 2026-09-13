"""
Virtual Fence and Loitering Detection Engine.
Utilizes Shapely 2.0 polygon collision vectors and temporal dwell trackers
to identify perimeter breaches, directional boundary crossings, and unauthorized loitering.
"""

import time
from typing import List, Dict, Tuple, Optional, Set
from shapely.geometry import Point, Polygon, LineString


class VirtualFenceManager:
    """
    Manages defined virtual fence zones, evaluates spatial tracklet collisions,
    and maintains temporal dwelling state for loitering detection.
    """
    def __init__(self, fence_configs: List[Dict], frame_width: int = 1280, frame_height: int = 720):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fences: Dict[str, Dict] = {}
        self.dwell_trackers: Dict[str, Dict[str, float]] = {}  # {track_id: {fence_id: first_seen_time}}
        self.active_tracks_last_pos: Dict[str, Tuple[float, float]] = {}  # {track_id: (x, y)}
        
        self.update_fences(fence_configs)

    def update_fences(self, fence_configs: List[Dict]):
        """Parses and updates fence polygon geometries."""
        self.fences.clear()
        for cfg in fence_configs:
            f_id = cfg.get("id", f"fence_{int(time.time() * 1000)}")
            raw_poly = cfg.get("polygon") or cfg.get("polygon_pixels", [])
            if not raw_poly:
                continue

            # Convert normalized coordinates [0.0, 1.0] to pixel coordinates if needed
            pixel_poly = []
            for pt in raw_poly:
                px = pt[0] * self.frame_width if pt[0] <= 1.0 else pt[0]
                py = pt[1] * self.frame_height if pt[1] <= 1.0 else pt[1]
                pixel_poly.append((float(px), float(py)))

            if len(pixel_poly) >= 3:
                shapely_poly = Polygon(pixel_poly)
                self.fences[f_id] = {
                    "id": f_id,
                    "name": cfg.get("name", f_id),
                    "zone_type": cfg.get("zone_type", "RESTRICTED"),
                    "color": cfg.get("color", "#EF4444"),
                    "polygon": raw_poly,
                    "polygon_pixels": pixel_poly,
                    "shapely_poly": shapely_poly,
                    "allowed_classes": cfg.get("allowed_classes", []),
                    "loiter_threshold_sec": float(cfg.get("loiter_threshold_sec", 5.0)),
                    "is_auto": cfg.get("is_auto", False)
                }

    def update_frame_dimensions(self, width: int, height: int):
        """Updates resolution and re-scales polygons if stream resolution changes."""
        if width != self.frame_width or height != self.frame_height:
            self.frame_width = width
            self.frame_height = height

    def check_tracklets(self, tracklets: List[Dict]) -> List[Dict]:
        """
        Evaluates a batch of active tracklets against all virtual fences.
        Each tracklet dict: {"track_id": str, "class_id": int, "class_name": str, "bbox": [x1, y1, x2, y2]}
        Returns list of generated breach / loitering alert records.
        """
        now = time.time()
        alerts = []
        current_active_ids = set()

        for track in tracklets:
            track_id = str(track["track_id"])
            class_id = track.get("class_id", -1)
            class_name = track.get("class_name", "unknown")
            bbox = track["bbox"]  # [x1, y1, x2, y2]
            current_active_ids.add(track_id)

            # Test both feet position (bottom center) and body center point
            # to handle full-body CCTV, seated persons, and webcam close-ups
            ground_x = (bbox[0] + bbox[2]) / 2.0
            ground_y = float(bbox[3])
            center_x = ground_x
            center_y = (bbox[1] + bbox[3]) / 2.0

            ground_pt = Point(ground_x, ground_y)
            center_pt = Point(center_x, center_y)

            prev_pos = self.active_tracks_last_pos.get(track_id)
            self.active_tracks_last_pos[track_id] = (ground_x, ground_y)

            if track_id not in self.dwell_trackers:
                self.dwell_trackers[track_id] = {}

            # Test points against each configured virtual fence
            for fence_id, fence in self.fences.items():
                poly: Polygon = fence["shapely_poly"]
                is_inside = poly.contains(ground_pt) or poly.contains(center_pt)
                
                # Check directional perimeter crossing (vector from prev_pos to curr_pos)
                crossed_boundary = False
                if prev_pos is not None:
                    motion_vector = LineString([prev_pos, (ground_x, ground_y)])
                    crossed_boundary = poly.exterior.intersects(motion_vector)

                if is_inside or crossed_boundary:
                    is_allowed = class_id in fence.get("allowed_classes", [])
                    is_first_entry = fence_id not in self.dwell_trackers[track_id]

                    if is_first_entry:
                        self.dwell_trackers[track_id][fence_id] = now
                        
                        # Unauthorized breach: immediate perimeter intrusion alert
                        if not is_allowed:
                            alerts.append({
                                "alert_type": "PERIMETER_INTRUSION",
                                "threat_level": "CRITICAL" if fence["zone_type"] in ["RESTRICTED_PERIMETER", "RESTRICTED"] else "HIGH",
                                "fence_id": fence_id,
                                "fence_name": fence["name"],
                                "track_id": track_id,
                                "class_name": class_name,
                                "bbox": bbox,
                                "timestamp": now,
                                "message": f"Unauthorized {class_name} breached {fence['name']}!"
                            })
                    else:
                        # Entity is dwelling / lingering inside the zone
                        dwell_time = now - self.dwell_trackers[track_id][fence_id]
                        loiter_threshold = fence.get("loiter_threshold_sec", 5.0)

                        if dwell_time >= loiter_threshold:
                            alerts.append({
                                "alert_type": "LOITERING_VIOLATION",
                                "threat_level": "HIGH",
                                "fence_id": fence_id,
                                "fence_name": fence["name"],
                                "track_id": track_id,
                                "class_name": class_name,
                                "bbox": bbox,
                                "dwell_time": round(dwell_time, 1),
                                "timestamp": now,
                                "message": f"{class_name.capitalize()} (#{track_id}) loitering in {fence['name']} for {dwell_time:.1f}s"
                            })
                else:
                    # Entity exited the zone
                    if fence_id in self.dwell_trackers[track_id]:
                        del self.dwell_trackers[track_id][fence_id]

        # Purge stale tracks that left the camera viewport
        stale_tracks = set(self.dwell_trackers.keys()) - current_active_ids
        for tid in stale_tracks:
            del self.dwell_trackers[tid]
            if tid in self.active_tracks_last_pos:
                del self.active_tracks_last_pos[tid]

        return alerts
