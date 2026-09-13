"""
Core Stream Ingestion & Perception Engine.
Decouples RTSP network streaming, performs adaptive Zero-DCE low-light enhancement,
runs YOLOv10 object detection and ByteTrack tracking, enforces Shapely virtual fence
collision vectors, and asynchronously dispatches crops to specialist Redis queues.
"""

import cv2
import numpy as np
import time
import threading
import base64
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from config.settings import (
    DEFAULT_STREAM_SOURCE,
    CAMERA_ID,
    STREAM_WIDTH,
    STREAM_HEIGHT,
    STREAM_FPS,
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    CLASS_PERSON,
    VEHICLE_CLASSES,
    VIRTUAL_FENCES,
    QUEUE_FACES,
    QUEUE_PLATES,
    QUEUE_ANOMALY,
    DEVICE,
    YOLO_MODEL_NAME,
    YOLO_FALLBACK_MODEL
)
from models.zero_dce import ZeroDCEEnhancer
from core.virtual_fence import VirtualFenceManager
from core.auto_perimeter import AutoPerimeterDetector
from core.alert_manager import AlertManager
from workers.base_worker import MessageBroker

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class VideoStreamReader:
    """
    Threaded video ingestion worker decoupling RTSP stream network latency
    from GPU inference loop. Includes automatic reconnection and synthetic BOP generator.
    """
    def __init__(self, source: str = DEFAULT_STREAM_SOURCE, width: int = STREAM_WIDTH, height: int = STREAM_HEIGHT):
        self.source = source
        self.width = width
        self.height = height
        self.is_synthetic = (str(source).lower() == "synthetic")
        
        self.latest_frame: Optional[np.ndarray] = None
        self.running = False
        self.connected = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.cap = None

        # Synthetic generator variables
        self._sim_frame_count = 0

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, name="Stream-Capture", daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def _capture_loop(self):
        while self.running:
            if self.is_synthetic:
                frame = self._generate_synthetic_bop_frame()
                with self.lock:
                    self.latest_frame = frame
                    self.connected = True
                time.sleep(1.0 / STREAM_FPS)
                continue

            # Physical RTSP / Webcam / File ingestion
            src = int(self.source) if str(self.source).isdigit() else self.source
            is_file = isinstance(src, str) and (Path(src).is_file() or any(src.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mkv', '.mov', '.ts']))
            
            if isinstance(src, str) and src.lower().startswith("rtsp://"):
                import os
                # Enforce TCP transport and 5-second socket timeout for RTSP
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
                self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            elif isinstance(src, int) or (isinstance(src, str) and str(src).isdigit()):
                import platform
                dev_idx = int(src)
                # On Windows, cv2.CAP_DSHOW initializes webcams instantaneously (0.5s vs 15-20s MSMF hang)
                if platform.system() == "Windows":
                    print(f"[StreamReader] Opening camera {dev_idx} using DirectShow backend...")
                    self.cap = cv2.VideoCapture(dev_idx, cv2.CAP_DSHOW)
                    if not self.cap.isOpened():
                        print(f"[StreamReader] DirectShow failed; falling back to default backend for camera {dev_idx}...")
                        self.cap = cv2.VideoCapture(dev_idx)
                else:
                    self.cap = cv2.VideoCapture(dev_idx)
            else:
                self.cap = cv2.VideoCapture(src)
            
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            if not self.cap.isOpened():
                print(f"[StreamReader] Unable to connect to source '{self.source}'. Retrying in 3s...")
                time.sleep(3.0)
                continue

            # Read native video FPS for realistic playback pacing
            native_fps = self.cap.get(cv2.CAP_PROP_FPS)
            target_delay = (1.0 / native_fps) if (is_file and native_fps > 5 and native_fps <= 60) else (1.0 / STREAM_FPS)

            self.connected = True
            print(f"[StreamReader] Ingesting stream from: {self.source} (Target: {1.0/target_delay:.1f} FPS, File Mode: {is_file})")

            consecutive_failures = 0
            while self.running and self.cap.isOpened():
                t_frame_start = time.time()
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 10:
                        if is_file:
                            # Seamlessly loop recorded video file
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = self.cap.read()
                            if not ret or frame is None:
                                break
                            consecutive_failures = 0
                        else:
                            print("[StreamReader] Stream disconnected or EOF after repeated attempts. Reconnecting...")
                            break
                    time.sleep(0.02)
                    continue

                consecutive_failures = 0
                
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))

                with self.lock:
                    self.latest_frame = frame

                # Pacing delay for real-time video playback simulation
                elapsed = time.time() - t_frame_start
                sleep_time = max(0.001, target_delay - elapsed)
                time.sleep(sleep_time)

            self.connected = False
            if self.cap:
                self.cap.release()
            time.sleep(1.0)

    def _generate_synthetic_bop_frame(self) -> np.ndarray:
        """
        Generates realistic synthetic tactical surveillance video frames
        simulating a Border Out Post checkpost with day/night illumination cycles,
        moving patrol personnel, and approaching tactical vehicles.
        """
        self._sim_frame_count += 1
        t = self._sim_frame_count

        # Simulate dynamic day/night light cycle (every 300 frames drops to night)
        cycle_phase = (t % 300) / 300.0
        is_night = cycle_phase > 0.45 and cycle_phase < 0.85
        ambient_base = 35 if is_night else 160

        # Background: Outpost terrain and sky
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Horizon & ground gradient
        ground_y = int(self.height * 0.35)
        frame[0:ground_y, :] = (int(ambient_base * 0.7), int(ambient_base * 0.6), int(ambient_base * 0.5))  # Sky
        frame[ground_y:, :] = (int(ambient_base * 0.35), int(ambient_base * 0.45), int(ambient_base * 0.40))  # Ground terrain

        # Road / Barrier
        road_pts = np.array([[int(self.width * 0.3), self.height], [int(self.width * 0.7), self.height],
                             [int(self.width * 0.55), ground_y], [int(self.width * 0.45), ground_y]], np.int32)
        cv2.fillPoly(frame, [road_pts], (int(ambient_base * 0.25), int(ambient_base * 0.25), int(ambient_base * 0.25)))

        # Watchtower / Security Bunker silhouette
        cv2.rectangle(frame, (80, ground_y - 120), (180, ground_y + 40), (40, 50, 45), -1)
        cv2.putText(frame, "BOP-ALPHA", (85, ground_y - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

        # Barbed wire fence posts
        for fx in range(50, self.width, 90):
            cv2.line(frame, (fx, ground_y - 40), (fx, ground_y + 60), (30, 30, 30), 2)
        cv2.line(frame, (50, ground_y - 20), (self.width - 50, ground_y - 20), (50, 50, 50), 1)
        cv2.line(frame, (50, ground_y + 10), (self.width - 50, ground_y + 10), (50, 50, 50), 1)

        # Draw Moving Target 1: Infiltrator / Person moving along northern perimeter
        # Breaches the top virtual fence zone
        px = int((self.width * 0.1) + ((t * 4) % int(self.width * 0.8)))
        py = int(self.height * 0.28 + np.sin(t * 0.05) * 20)
        pw, ph = 40, 85
        # Draw simulated human figure
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (60, 70, 65), -1)
        cv2.circle(frame, (px + pw // 2, py - 12), 12, (70, 80, 75), -1)

        # Draw Moving Target 2: Vehicle approaching BOP checkpost barrier
        vx = int((self.width * 0.45) + np.sin(t * 0.02) * 50)
        vy = int(self.height * 0.55 + ((t * 3) % int(self.height * 0.3)))
        vw, vh = 130, 80
        cv2.rectangle(frame, (vx - vw // 2, vy), (vx + vw // 2, vy + vh), (50, 55, 60), -1)
        # License plate on vehicle rear
        plate_w, plate_h = 44, 14
        plate_x = vx - plate_w // 2
        plate_y = vy + vh - 22
        cv2.rectangle(frame, (plate_x, plate_y), (plate_x + plate_w, plate_y + plate_h), (220, 220, 220), -1)
        cv2.putText(frame, "PB08", (plate_x + 2, plate_y + 10), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), 1)

        # Add camera sensor noise
        noise = np.random.normal(0, 4 if not is_night else 12, frame.shape).astype(np.int16)
        noisy_frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return noisy_frame


class StreamEngine:
    """
    Core Perception Pipeline orchestrating:
    - RTSP ingestion decoupling
    - Adaptive Zero-DCE illumination enhancement
    - YOLOv10 object detection and ByteTrack tracking
    - Shapely virtual fence & loitering analysis
    - Asynchronous crop dispatch to Redis queues (queue:faces, queue:plates, queue:anomaly)
    """
    def __init__(self, stream_source: str = DEFAULT_STREAM_SOURCE, broker: Optional[MessageBroker] = None):
        self.camera_id = CAMERA_ID
        self.stream_reader = VideoStreamReader(source=stream_source)
        self.broker = broker if broker is not None else MessageBroker()
        self.alert_manager = AlertManager(broker=self.broker)
        
        # Models and sub-modules
        self.zero_dce = ZeroDCEEnhancer(device=DEVICE)
        self.fence_manager = VirtualFenceManager(VIRTUAL_FENCES, STREAM_WIDTH, STREAM_HEIGHT)
        self.perimeter_mode = "manual"  # "manual" or "auto"
        self.manual_fences = list(VIRTUAL_FENCES)
        self.auto_detector = AutoPerimeterDetector(STREAM_WIDTH, STREAM_HEIGHT)

        # Object Detection & ByteTrack Model
        self.yolo_model = None
        self._init_yolo()

        # Engine state & telemetry
        self.running = False
        self.fps = 0.0
        self.last_fps_calc = time.time()
        self.frame_counter = 0
        self.active_tracks_count = 0
        self.low_light_active = False
        self.current_annotated_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()

        # Anomaly temporal subsampler state
        self.last_anomaly_sample = 0.0

    def _init_yolo(self):
        """Initializes YOLOv8/v10 with graceful fallback."""
        if not ULTRALYTICS_AVAILABLE:
            print("[StreamEngine] 'ultralytics' not installed. Running algorithmic simulation mode.")
            return

        try:
            print(f"[StreamEngine] Loading detection model: {YOLO_MODEL_NAME}...")
            self.yolo_model = YOLO(YOLO_MODEL_NAME)
            dev_str = DEVICE if (TORCH_AVAILABLE and torch.cuda.is_available() and "cuda" in str(DEVICE)) else "cpu"
            print(f"[StreamEngine] Detection model loaded successfully (Target Device: {dev_str})")
        except Exception as e:
            print(f"[StreamEngine] Primary model load failed ({e}). Loading fallback: {YOLO_FALLBACK_MODEL}...")
            try:
                self.yolo_model = YOLO(YOLO_FALLBACK_MODEL)
            except Exception as e2:
                print(f"[StreamEngine] YOLO fallback error: {e2}. Proceeding in simulation mode.")

    def start(self):
        """Starts stream reader and main perception loop."""
        if self.running:
            return
        self.running = True
        self.stream_reader.start()
        self.thread = threading.Thread(target=self._process_loop, name="Perception-Engine", daemon=True)
        self.thread.start()
        print(f"[StreamEngine] Perception Engine activated for camera '{self.camera_id}'")

    def stop(self):
        """Stops perception pipeline."""
        self.running = False
        self.stream_reader.stop()
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        print("[StreamEngine] Perception Engine stopped.")

    def set_perimeter_mode(self, mode: str) -> List[Dict]:
        """Toggles between 'manual' and 'auto' perimeter detection modes."""
        self.perimeter_mode = mode.lower()
        if self.perimeter_mode == "auto":
            return self.trigger_auto_detection()
        else:
            self.fence_manager.update_fences(self.manual_fences)
            return self.manual_fences

    def trigger_auto_detection(self) -> List[Dict]:
        """Automatically estimates and updates natural perimeter boundaries on the active video feed."""
        frame = self.stream_reader.get_frame()
        if frame is None:
            frame = np.zeros((STREAM_HEIGHT, STREAM_WIDTH, 3), dtype=np.uint8)

        detected_fences = self.auto_detector.detect_perimeters(frame)
        self.fence_manager.update_fences(detected_fences)
        print(f"[StreamEngine] Auto-detected {len(detected_fences)} perimeter zones.")
        return detected_fences

    def update_manual_fences(self, fences: List[Dict]):
        """Saves user-defined manual fences."""
        self.manual_fences = fences
        if self.perimeter_mode == "manual":
            self.fence_manager.update_fences(fences)

    def get_latest_annotated_frame(self) -> Optional[np.ndarray]:
        """Returns the latest annotated visualization frame."""
        with self.frame_lock:
            if self.current_annotated_frame is None:
                return None
            return self.current_annotated_frame.copy()

    def _process_loop(self):
        while self.running:
            raw_frame = self.stream_reader.get_frame()
            if raw_frame is None:
                time.sleep(0.01)
                continue

            t_start = time.time()

            # 1. Adaptive Illumination Check: route through Zero-DCE if dark
            enhanced_frame, was_enhanced, luminance = self.zero_dce.enhance(raw_frame)
            self.low_light_active = was_enhanced

            # 2. Run Object Detection & ByteTrack
            tracklets = self._run_detection_and_tracking(enhanced_frame)
            self.active_tracks_count = len(tracklets)

            # 3. Virtual Fence Intrusion and Loitering Collision Checks
            fence_alerts = self.fence_manager.check_tracklets(tracklets)
            for alert in fence_alerts:
                alert["camera_id"] = self.camera_id
                # Crop offending entity
                bbox = alert.get("bbox")
                crop = self._crop_box(enhanced_frame, bbox) if bbox else None
                self.alert_manager.dispatch(alert, crop_bgr=crop)

            # 4. Asynchronous Queue Dispatching (queue:faces & queue:plates)
            self._dispatch_specialist_crops(enhanced_frame, tracklets)

            # 5. Temporal Anomaly Buffer Dispatch (queue:anomaly at 2-4 FPS)
            self._dispatch_anomaly_frame(enhanced_frame)

            # 6. Render Tactical HUD & Annotations for Dashboard
            annotated = self._render_tactical_hud(enhanced_frame, tracklets, was_enhanced, luminance)

            with self.frame_lock:
                self.current_annotated_frame = annotated

            # Calculate FPS metrics
            self.frame_counter += 1
            elapsed = time.time() - self.last_fps_calc
            if elapsed >= 1.0:
                self.fps = self.frame_counter / elapsed
                self.frame_counter = 0
                self.last_fps_calc = time.time()

            # Target 25-30 FPS pacing
            compute_time = time.time() - t_start
            sleep_time = max(0.001, (1.0 / STREAM_FPS) - compute_time)
            time.sleep(sleep_time)

    def _run_detection_and_tracking(self, frame_bgr: np.ndarray) -> List[Dict]:
        """Runs YOLO + ByteTrack on the frame."""
        tracklets = []

        if self.yolo_model is not None:
            try:
                # Safe device string check (never pass cuda if torch is CPU-only)
                track_device = "cpu"
                if TORCH_AVAILABLE and torch.cuda.is_available() and "cuda" in str(DEVICE):
                    track_device = DEVICE

                results = self.yolo_model.track(
                    frame_bgr,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=CONF_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    classes=[CLASS_PERSON] + VEHICLE_CLASSES,
                    verbose=False,
                    device=track_device
                )

                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for i in range(len(boxes)):
                        xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                        cls_id = int(boxes.cls[i].item())
                        conf = float(boxes.conf[i].item())
                        track_id = int(boxes.id[i].item()) if (boxes.id is not None and i < len(boxes.id) and boxes.id[i] is not None) else (i + 1)
                        cls_name = results[0].names.get(cls_id, "object")

                        tracklets.append({
                            "track_id": str(track_id),
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "conf": round(conf, 2),
                            "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                        })
                return tracklets
            except Exception as e:
                print(f"[StreamEngine] Tracking error: {e}")

        # Only inject simulated tracklets if source is explicitly 'synthetic'
        if self.stream_reader.is_synthetic:
            return self._simulated_tracklets(frame_bgr)

        # On actual video / RTSP feeds, return only real detections (never hallucinate fake objects)
        return tracklets

    def _simulated_tracklets(self, frame_bgr: np.ndarray) -> List[Dict]:
        """Provides deterministic tracklets for synthetic testing if YOLO is initializing."""
        h, w = frame_bgr.shape[:2]
        # Target 1: Simulated person in upper fence zone
        t = time.time()
        px = int((w * 0.1) + ((t * 80) % int(w * 0.75)))
        py = int(h * 0.28 + np.sin(t * 2) * 15)
        
        # Target 2: Simulated vehicle in checkpost zone
        vx = int((w * 0.45) + np.sin(t) * 30)
        vy = int(h * 0.58 + ((t * 40) % int(h * 0.25)))

        return [
            {
                "track_id": "P-101",
                "class_id": CLASS_PERSON,
                "class_name": "person",
                "conf": 0.91,
                "bbox": [px, py, px + 45, py + 95]
            },
            {
                "track_id": "V-202",
                "class_id": 2,  # Car
                "class_name": "car",
                "conf": 0.88,
                "bbox": [vx - 65, vy, vx + 65, vy + 80]
            }
        ]

    def _crop_box(self, frame_bgr: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """Crops bounding box safely with boundary clamping."""
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame_bgr[y1:y2, x1:x2]

    def _dispatch_specialist_crops(self, frame_bgr: np.ndarray, tracklets: List[Dict]):
        """Crops detected entities and pushes to specialist worker queues."""
        now = time.time()
        for trk in tracklets:
            cls_id = trk["class_id"]
            bbox = trk["bbox"]
            track_id = trk["track_id"]
            crop = self._crop_box(frame_bgr, bbox)

            if crop is None or crop.size == 0:
                continue

            crop_b64 = AlertManager.encode_crop_to_base64(crop)

            # Person -> dispatch to queue:faces for FRS Worker
            if cls_id == CLASS_PERSON:
                payload = {
                    "camera_id": self.camera_id,
                    "track_id": track_id,
                    "timestamp": now,
                    "bbox": bbox,
                    "crop_base64": crop_b64
                }
                self.broker.push(QUEUE_FACES, payload)

            # Vehicle -> dispatch to queue:plates for ANPR Worker
            elif cls_id in VEHICLE_CLASSES:
                payload = {
                    "camera_id": self.camera_id,
                    "track_id": track_id,
                    "timestamp": now,
                    "bbox": bbox,
                    "crop_base64": crop_b64
                }
                self.broker.push(QUEUE_PLATES, payload)

    def _dispatch_anomaly_frame(self, frame_bgr: np.ndarray):
        """Samples frames at 2-4 FPS for the behavioral anomaly worker."""
        now = time.time()
        if (now - self.last_anomaly_sample) >= 0.30:  # ~3.3 FPS
            self.last_anomaly_sample = now
            small = cv2.resize(frame_bgr, (320, 180))
            frame_b64 = AlertManager.encode_crop_to_base64(small)
            payload = {
                "camera_id": self.camera_id,
                "timestamp": now,
                "frame_base64": frame_b64
            }
            self.broker.push(QUEUE_ANOMALY, payload)

    def _render_tactical_hud(self, frame_bgr: np.ndarray, tracklets: List[Dict], was_enhanced: bool, luminance: float) -> np.ndarray:
        """Renders military tactical HUD overlay with virtual fences, tracklet boxes, and night vision badge."""
        canvas = frame_bgr.copy()
        h, w = canvas.shape[:2]

        # 1. Render Virtual Fence Polygons
        for fence_id, f in self.fence_manager.fences.items():
            pts = np.array(f["polygon_pixels"], np.int32).reshape((-1, 1, 2))
            hex_col = f.get("color", "#EF4444").lstrip('#')
            # Convert HEX to BGR
            r, g, b = tuple(int(hex_col[i:i+2], 16) for i in (0, 2, 4))
            color_bgr = (b, g, r)

            # Draw semi-transparent polygon fill
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], color_bgr)
            cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
            cv2.polylines(canvas, [pts], isClosed=True, color=color_bgr, thickness=2, lineType=cv2.LINE_AA)

            # Label top vertex
            top_pt = min(f["polygon_pixels"], key=lambda p: p[1])
            cv2.putText(canvas, f["name"], (int(top_pt[0]), int(top_pt[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1, cv2.LINE_AA)

        # 2. Render Tracklet Bounding Boxes
        for trk in tracklets:
            bbox = trk["bbox"]
            tid = trk["track_id"]
            cname = trk["class_name"]
            conf = trk["conf"]

            color = (0, 240, 255) if cname == "person" else (255, 180, 0)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            # Corner brackets for tactical HUD look
            corner_len = min(15, (x2 - x1) // 3)
            cv2.line(canvas, (x1, y1), (x1 + corner_len, y1), color, 3)
            cv2.line(canvas, (x1, y1), (x1, y1 + corner_len), color, 3)
            cv2.line(canvas, (x2, y2), (x2 - corner_len, y2), color, 3)
            cv2.line(canvas, (x2, y2), (x2, y2 - corner_len), color, 3)

            label = f"#{tid} {cname.upper()} {int(conf*100)}%"
            cv2.rectangle(canvas, (x1, y1 - 20), (x1 + len(label)*9, y1), color, -1)
            cv2.putText(canvas, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

        # 3. Top HUD Banner
        cv2.rectangle(canvas, (0, 0), (w, 36), (20, 24, 28), -1)
        cv2.line(canvas, (0, 36), (w, 36), (40, 48, 56), 1)

        # Tactical Status Text
        status_text = f"CAM: {self.camera_id}  |  FPS: {self.fps:.1f}  |  TRACKS: {len(tracklets)}  |  FENCE: [{self.perimeter_mode.upper()}]"
        cv2.putText(canvas, status_text, (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 220, 240), 1, cv2.LINE_AA)

        # Night Vision / Zero-DCE status badge
        if was_enhanced:
            badge_text = f"[!] ZERO-DCE LOW-LIGHT BOOST (LUM: {luminance:.0f})"
            cv2.putText(canvas, badge_text, (w - 380, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 120), 2, cv2.LINE_AA)
        else:
            badge_text = f"NORMAL ILLUMINATION (LUM: {luminance:.0f})"
            cv2.putText(canvas, badge_text, (w - 340, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

        return canvas
