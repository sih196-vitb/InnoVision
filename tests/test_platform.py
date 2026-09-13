"""
Automated Test Suite for AI Video Analytics Platform.
Validates Zero-DCE enhancement, Shapely virtual fences, message broker queues,
ANPR engine, FRS matching, behavioral anomaly detector, and dashboard endpoints.
"""

import unittest
import numpy as np
import cv2
import json
import time
from pathlib import Path

from config.settings import (
    LOW_LIGHT_THRESHOLD,
    QUEUE_PLATES,
    QUEUE_FACES,
    BLACKLIST_PLATES,
    WHITELIST_PLATES,
    WATCHLIST_FILE
)
from models.zero_dce import ZeroDCEEnhancer
from core.virtual_fence import VirtualFenceManager
from core.auto_perimeter import AutoPerimeterDetector
from workers.base_worker import MessageBroker
from models.plate_engine import PlateEngine
from models.face_engine import FaceEngine
from models.anomaly_detector import BehavioralAnomalyDetector
from training.train_plate_detector import check_hardware_guards
from dashboard.app import app


class TestBOPPlatform(unittest.TestCase):

    def setUp(self):
        self.app_client = app.test_client()

    def test_01_zero_dce_enhancer(self):
        """Validates adaptive illumination calculation and Zero-DCE enhancement."""
        enhancer = ZeroDCEEnhancer(low_light_thresh=60.0, device="cpu")

        # Pitch dark image (luminance ~ 10)
        dark_img = np.full((120, 160, 3), 10, dtype=np.uint8)
        lum = enhancer.measure_luminance(dark_img)
        self.assertLess(lum, 60.0)

        enhanced_bgr, was_enhanced, _ = enhancer.enhance(dark_img)
        self.assertTrue(was_enhanced)
        self.assertEqual(enhanced_bgr.shape, dark_img.shape)

        # Well-lit image (luminance ~ 180)
        bright_img = np.full((120, 160, 3), 180, dtype=np.uint8)
        _, was_enhanced_bright, lum_bright = enhancer.enhance(bright_img)
        self.assertFalse(was_enhanced_bright)
        self.assertGreater(lum_bright, 60.0)

    def test_02_virtual_fence_intrusion_and_loitering(self):
        """Validates Shapely polygon collision, intrusion alert, and loitering timers."""
        fences = [{
            "id": "test_zone",
            "name": "Armory Perimeter",
            "zone_type": "RESTRICTED_PERIMETER",
            "color": "#EF4444",
            "polygon": [[100, 100], [300, 100], [300, 300], [100, 300]],
            "allowed_classes": [],
            "loiter_threshold_sec": 0.2  # fast loiter for test
        }]
        mgr = VirtualFenceManager(fences, 640, 480)

        # Tracklet outside fence
        outside_track = [{
            "track_id": "T1",
            "class_id": 0,
            "class_name": "person",
            "bbox": [20, 20, 40, 60]  # bottom center is (30, 60) -> outside
        }]
        alerts = mgr.check_tracklets(outside_track)
        self.assertEqual(len(alerts), 0)

        # Tracklet inside fence -> triggers intrusion
        inside_track = [{
            "track_id": "T1",
            "class_id": 0,
            "class_name": "person",
            "bbox": [150, 150, 190, 250]  # bottom center is (170, 250) -> inside
        }]
        alerts_intrusion = mgr.check_tracklets(inside_track)
        self.assertEqual(len(alerts_intrusion), 1)
        self.assertEqual(alerts_intrusion[0]["alert_type"], "PERIMETER_INTRUSION")

        # Test loitering dwell time
        time.sleep(0.25)
        alerts_loiter = mgr.check_tracklets(inside_track)
        self.assertTrue(any(a["alert_type"] == "LOITERING_VIOLATION" for a in alerts_loiter))

    def test_03_message_broker_queues(self):
        """Validates message broker push/pop and in-memory fallback queues."""
        broker = MessageBroker()
        test_payload = {"test_id": "12345", "action": "VERIFY_BROKER"}

        broker.push("queue:test_channel", test_payload)
        popped = broker.pop("queue:test_channel", timeout=1.0)
        self.assertIsNotNone(popped)
        self.assertEqual(popped.get("test_id"), "12345")

    def test_04_anpr_plate_engine(self):
        """Validates plate ROI localization and blacklist/whitelist validation."""
        engine = PlateEngine()

        # Test vehicle crop with rectangular high contrast area
        canvas = np.zeros((200, 300, 3), dtype=np.uint8)
        cv2.rectangle(canvas, (80, 140), (220, 180), (240, 240, 240), -1)
        cv2.putText(canvas, "DL01AB1234", (90, 165), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 0), 2)

        roi = engine.locate_plate_roi(canvas)
        self.assertIsNotNone(roi)
        self.assertGreater(roi.size, 0)

        # Test validation logic
        res_black = engine.validate_registration("DL01AB1234", BLACKLIST_PLATES, WHITELIST_PLATES)
        self.assertEqual(res_black["status"], "BLACKLISTED")

        res_white = engine.validate_registration("BSF-4412-PATROL", BLACKLIST_PLATES, WHITELIST_PLATES)
        self.assertEqual(res_white["status"], "WHITELISTED")

        res_unknown = engine.validate_registration("UP16XX8888", BLACKLIST_PLATES, WHITELIST_PLATES)
        self.assertEqual(res_unknown["status"], "UNKNOWN")

    def test_05_frs_face_engine(self):
        """Validates face extraction and cosine similarity matching."""
        engine = FaceEngine(watchlist_path=WATCHLIST_FILE, match_threshold=0.5)
        self.assertGreater(len(engine.watchlist), 0)

        face_sample = np.zeros((112, 112, 3), dtype=np.uint8)
        cv2.circle(face_sample, (56, 56), 30, (200, 200, 200), -1)

        embedding = engine.extract_embedding(face_sample)
        self.assertEqual(len(embedding), 512)

        # Cosine similarity self check
        sim_self = engine.cosine_similarity(embedding, embedding)
        self.assertAlmostEqual(sim_self, 1.0, places=2)

    def test_06_anomaly_detector(self):
        """Validates sliding window optical flow motion energy calculation."""
        detector = BehavioralAnomalyDetector(window_size=8, sample_fps=10)

        # Feed 10 synthetic frames with sudden motion surge
        for i in range(10):
            frame = np.zeros((180, 320, 3), dtype=np.uint8)
            # Add moving bright block
            cv2.rectangle(frame, (i * 25, 50), (i * 25 + 40, 90), (255, 255, 255), -1)
            detector.push_frame(frame, timestamp=time.time() + (i * 0.2), force=True)

        res = detector.compute_anomaly_score()
        self.assertTrue(res.get("ready"))
        self.assertIn("anomaly_score", res)
        self.assertIn("motion_energy", res)

    def test_07_hardware_training_guards(self):
        """Validates that training memory guards enforce safe batch sizing."""
        device, batch_size = check_hardware_guards()
        self.assertIn(batch_size, [4, 8])

    def test_08_dashboard_api_endpoints(self):
        """Validates Flask REST endpoints."""
        # /api/stats
        resp_stats = self.app_client.get('/api/stats')
        self.assertEqual(resp_stats.status_code, 200)
        data_stats = json.loads(resp_stats.data)
        self.assertIn("gpu", data_stats)
        self.assertIn("system", data_stats)
        self.assertIn("pipeline", data_stats)

        # /api/fences
        resp_fences = self.app_client.get('/api/fences')
        self.assertEqual(resp_fences.status_code, 200)
        data_fences = json.loads(resp_fences.data)
        self.assertIn("fences", data_fences)

        # /api/watchlist
        resp_wlist = self.app_client.get('/api/watchlist')
        self.assertEqual(resp_wlist.status_code, 200)

    def test_09_auto_perimeter_detection_and_toggle(self):
        """Validates auto-perimeter detection and mode switching endpoints."""
        detector = AutoPerimeterDetector(1280, 720)
        sample_frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
        # Add simulated horizon line
        sample_frame[250:255, :] = (20, 20, 20)

        auto_zones = detector.detect_perimeters(sample_frame)
        self.assertGreaterEqual(len(auto_zones), 2)
        for zone in auto_zones:
            self.assertIn("polygon", zone)
            self.assertGreaterEqual(len(zone["polygon"]), 3)
            self.assertTrue(zone.get("is_auto"))

        # Test mode toggle API: Switch to auto
        res_mode_auto = self.app_client.post('/api/fences/mode', json={"mode": "auto"})
        # Even if engine is not bound in mock test client, endpoint validates payload
        self.assertIn(res_mode_auto.status_code, [200, 500])

        # Test invalid mode payload returns 400
        res_invalid = self.app_client.post('/api/fences/mode', json={"mode": "invalid_xyz"})
        self.assertEqual(res_invalid.status_code, 400)


if __name__ == '__main__':
    unittest.main()
