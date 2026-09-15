"""
Configuration module for AI-driven Software-Defined Video Analytics Platform.
Optimized for Border Out Posts (BOPs), check posts, and strategic installations.
Hardware Profile: NVIDIA RTX 3050 Laptop GPU (4GB VRAM), Intel 11th Gen i5 (4 workers), 24GB RAM.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
WEIGHTS_DIR = MODELS_DIR / "weights"
LOGS_DIR = BASE_DIR / "logs"

for directory in [DATA_DIR, MODELS_DIR, WEIGHTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Hardware & VRAM Guardrails (RTX 3050 4GB VRAM Protection)
try:
    import torch
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

CUDA_AMP_ENABLED = (DEVICE != "cpu") # FP16 Automatic Mixed Precision
MAX_TRAINING_BATCH = 8              # Strict memory ceiling for 4GB VRAM
INFERENCE_BATCH = 1                 # Real-time streaming inference
DATALOADER_WORKERS = 4              # Intel 11th Gen i5 CPU optimization
RAM_DATASET_CACHE = True            # Exploit 24GB System RAM to eliminate disk I/O bottlenecks
MAX_VRAM_TARGET_MB = 3400           # Keep safety cushion under 4096MB

# Stream & Ingestion Config
DEFAULT_STREAM_SOURCE = os.getenv("STREAM_SOURCE", "synthetic")  # "synthetic", "0", or "rtsp://..."
CAMERA_ID = "BOP-ALPHA-SECTOR-01"
STREAM_WIDTH = 1280
STREAM_HEIGHT = 720
STREAM_FPS = 25
RECONNECT_INTERVAL_SEC = 3.0

# Adaptive Low-Light Illumination Engine (Zero-DCE)
# If average scene luminance in HSV V channel (0-255) drops below threshold, route to Zero-DCE
LOW_LIGHT_THRESHOLD = 60.0
ZERO_DCE_SCALE_FACTOR = 1.0

# Detection & Tracking Config (YOLOv8/v10 + ByteTrack)
YOLO_MODEL_NAME = "yolov8n.pt"      # Ultralytics lightweight model (lightest VRAM footprint)
YOLO_FALLBACK_MODEL = "yolov8n.pt"  # Resilient fallback
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45

# Class indices (COCO standard)
CLASS_PERSON = 0
CLASS_BICYCLE = 1
CLASS_CAR = 2
CLASS_MOTORCYCLE = 3
CLASS_BUS = 5
CLASS_TRUCK = 7
VEHICLE_CLASSES = [CLASS_BICYCLE, CLASS_CAR, CLASS_MOTORCYCLE, CLASS_BUS, CLASS_TRUCK]

# Virtual Fence & Loitering Parameters
# Normalized polygon coordinates [(x_norm, y_norm), ...] for camera viewport (0.0 to 1.0)
VIRTUAL_FENCES = [
    {
        "id": "fence_perimeter_north",
        "name": "Restricted Perimeter Zone (Breach Alert)",
        "zone_type": "RESTRICTED_PERIMETER",
        "color": "#EF4444",  # Red
        "polygon": [
            [0.05, 0.10],
            [0.95, 0.10],
            [0.95, 0.45],
            [0.05, 0.45]
        ],
        "allowed_classes": [],
        "loiter_threshold_sec": 4.0
    },
    {
        "id": "fence_checkpost_gate",
        "name": "BOP Main Barrier & Transit Zone",
        "zone_type": "CHECKPOST_CONTROL",
        "color": "#F59E0B",  # Amber
        "polygon": [
            [0.15, 0.35],
            [0.85, 0.35],
            [0.90, 0.95],
            [0.10, 0.95]
        ],
        "allowed_classes": [CLASS_PERSON, CLASS_CAR, CLASS_TRUCK],
        "loiter_threshold_sec": 5.0
    }
]

FENCES_FILE = CONFIG_DIR / "fences.json"

# Redis Messaging & Broker
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

QUEUE_FACES = "queue:faces"
QUEUE_PLATES = "queue:plates"
QUEUE_ALERTS = "queue:alerts"
QUEUE_ANOMALY = "queue:anomaly"
CHANNEL_ALERTS = "channel:alerts"

# Face Recognition Subsystem (FRS)
WATCHLIST_FILE = CONFIG_DIR / "watchlist.json"
FRS_MATCH_THRESHOLD = 0.55         # Cosine distance threshold (lower = stricter)
FACE_CROP_MIN_SIZE = 48             # Minimum width/height in pixels for face processing

# Automatic Number Plate Recognition (ANPR)
ANPR_OCR_LANG = "en"
PLATE_CONF_THRESHOLD = 0.40
BLACKLIST_PLATES = [
    "DL01AB1234",
    "JK02X9999",
    "PB08CC4040",
    "SUSPECT-01",
    "UNKNOWN-ROUGE"
]
WHITELIST_PLATES = [
    "DEF-7890-ARMY",
    "BSF-4412-PATROL",
    "BOP-HQ-001",
    "DL04CA9001"
]

# Behavioral Anomaly Subsystem
ANOMALY_WINDOW_SIZE = 16            # Temporal sliding window frames
ANOMALY_SAMPLE_FPS = 3              # Subsample FPS to buffer motion dynamics
ANOMALY_SCORE_THRESHOLD = 0.68      # Alert trigger threshold (0.0 to 1.0)

# Central Command Dashboard
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 5000
DASHBOARD_DEBUG = False
SECRET_KEY = "bop_tactical_defense_secret_key"
