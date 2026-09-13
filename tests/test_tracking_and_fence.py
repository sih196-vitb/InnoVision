"""
Comprehensive Integration Tests for Detection Tracking, Intrusion, and Loitering.
"""

import unittest
import time
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import CLASS_PERSON, CLASS_CAR
from core.virtual_fence import VirtualFenceManager
from core.alert_manager import AlertManager
from core.stream_engine import StreamEngine
from workers.base_worker import MessageBroker


class TestTrackingAndFence(unittest.TestCase):

    def test_01_loitering_for_authorized_classes(self):
        """Verifies authorized classes at checkpost trigger loitering after threshold."""
        fences = [{
            "id": "gate_transit",
            "name": "Checkpost Gate",
            "zone_type": "CHECKPOST_CONTROL",
            "color": "#F59E0B",
            "polygon": [[100, 100], [400, 100], [400, 400], [100, 400]],
            "allowed_classes": [CLASS_PERSON, CLASS_CAR],
            "loiter_threshold_sec": 0.3
        }]
        mgr = VirtualFenceManager(fences, 640, 480)

        person_track = [{
            "track_id": "1",
            "class_id": CLASS_PERSON,
            "class_name": "person",
            "bbox": [150, 150, 250, 350]
        }]

        # Frame 1: Entry
        alerts_entry = mgr.check_tracklets(person_track)
        # Person is authorized at checkpost -> NO intrusion alert
        self.assertEqual(len(alerts_entry), 0, "Authorized person should not trigger PERIMETER_INTRUSION")

        # Frame 2: Stay longer than loiter_threshold_sec (0.3s)
        time.sleep(0.35)
        alerts_loiter = mgr.check_tracklets(person_track)
        self.assertEqual(len(alerts_loiter), 1)
        self.assertEqual(alerts_loiter[0]["alert_type"], "LOITERING_VIOLATION")
        self.assertIn("loitering", alerts_loiter[0]["message"])

    def test_02_intrusion_and_loitering_for_restricted_perimeter(self):
        """Verifies unauthorized breach triggers PERIMETER_INTRUSION and subsequent loitering."""
        fences = [{
            "id": "restricted_border",
            "name": "Restricted Border",
            "zone_type": "RESTRICTED_PERIMETER",
            "color": "#EF4444",
            "polygon": [[50, 50], [350, 50], [350, 350], [50, 350]],
            "allowed_classes": [],
            "loiter_threshold_sec": 0.3
        }]
        mgr = VirtualFenceManager(fences, 640, 480)

        intruder_track = [{
            "track_id": "99",
            "class_id": CLASS_PERSON,
            "class_name": "person",
            "bbox": [100, 100, 180, 260]
        }]

        # First frame -> immediate intrusion alert
        alerts_1 = mgr.check_tracklets(intruder_track)
        self.assertEqual(len(alerts_1), 1)
        self.assertEqual(alerts_1[0]["alert_type"], "PERIMETER_INTRUSION")

        # After dwelling -> loitering alert
        time.sleep(0.35)
        alerts_2 = mgr.check_tracklets(intruder_track)
        self.assertTrue(any(a["alert_type"] == "LOITERING_VIOLATION" for a in alerts_2))

    def test_03_center_point_detection_for_webcams(self):
        """Verifies a person whose feet are below the fence is still detected by center of mass."""
        fences = [{
            "id": "desk_zone",
            "name": "Restricted Zone",
            "zone_type": "RESTRICTED_PERIMETER",
            "color": "#EF4444",
            "polygon": [[100, 100], [500, 100], [500, 400], [100, 400]],
            "allowed_classes": [],
            "loiter_threshold_sec": 1.0
        }]
        mgr = VirtualFenceManager(fences, 640, 480)

        # Upper body bbox: bbox y1=150, y2=450 (feet y2=450 is OUTSIDE fence y:100-400, but center y=300 is INSIDE)
        seated_person = [{
            "track_id": "42",
            "class_id": CLASS_PERSON,
            "class_name": "person",
            "bbox": [200, 150, 300, 450]
        }]

        alerts = mgr.check_tracklets(seated_person)
        self.assertEqual(len(alerts), 1, "Should be detected by center point inside fence")
        self.assertEqual(alerts[0]["alert_type"], "PERIMETER_INTRUSION")

    def test_04_stream_engine_safe_detection(self):
        """Verifies StreamEngine executes real YOLO tracking without crashing."""
        broker = MessageBroker()
        engine = StreamEngine(stream_source="synthetic", broker=broker)
        blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        tracks = engine._run_detection_and_tracking(blank_frame)
        self.assertIsInstance(tracks, list)


if __name__ == "__main__":
    unittest.main()
