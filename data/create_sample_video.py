"""
Utility to generate a realistic test video file (data/test_media/sample_cctv.mp4)
to test video file ingestion, virtual fence intrusions, Zero-DCE, ANPR, and loitering.
"""

import cv2
import numpy as np
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR

OUTPUT_DIR = DATA_DIR / "test_media"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "sample_cctv.mp4"


def generate_test_video(duration_sec: int = 15, fps: int = 25, width: int = 1280, height: int = 720):
    total_frames = duration_sec * fps
    print(f"[*] Generating {duration_sec}s test video ({total_frames} frames) at: {OUTPUT_FILE}")

    # Use mp4v fourcc
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(OUTPUT_FILE), fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"[ERROR] Could not open VideoWriter for {OUTPUT_FILE}")
        return None

    # Load real textures from ultralytics assets if available
    person_sprite = None
    vehicle_sprite = None
    try:
        from ultralytics import ASSETS
        bus_asset = cv2.imread(str(ASSETS / 'bus.jpg'))
        zidane_asset = cv2.imread(str(ASSETS / 'zidane.jpg'))
        if bus_asset is not None:
            # Person crop from bus.jpg
            person_sprite = bus_asset[398:902, 48:245]
            # Bus/vehicle crop
            vehicle_sprite = bus_asset[231:756, 22:805]
        elif zidane_asset is not None:
            person_sprite = zidane_asset[197:711, 114:740]
    except Exception as e:
        print(f"[*] Note: Asset loading notice: {e}")

    for t in range(total_frames):
        # Day to night transition in the middle (frames 120 to 260)
        is_night = (120 <= t <= 260)
        ambient = 40 if is_night else 170

        frame = np.zeros((height, width, 3), dtype=np.uint8)
        ground_y = int(height * 0.35)

        # Sky and ground
        frame[0:ground_y, :] = (int(ambient * 0.7), int(ambient * 0.6), int(ambient * 0.5))
        frame[ground_y:, :] = (int(ambient * 0.35), int(ambient * 0.45), int(ambient * 0.40))

        # Road
        road_pts = np.array([[int(width * 0.25), height], [int(width * 0.75), height],
                             [int(width * 0.55), ground_y], [int(width * 0.45), ground_y]], np.int32)
        cv2.fillPoly(frame, [road_pts], (int(ambient * 0.25), int(ambient * 0.25), int(ambient * 0.25)))

        # Watchtower
        cv2.rectangle(frame, (80, ground_y - 120), (180, ground_y + 40), (40, 50, 45), -1)
        cv2.putText(frame, "BOP-ALPHA", (85, ground_y - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

        # Perimeter fence wire
        for fx in range(50, width, 90):
            cv2.line(frame, (fx, ground_y - 40), (fx, ground_y + 60), (30, 30, 30), 2)
        cv2.line(frame, (50, ground_y - 20), (width - 50, ground_y - 20), (50, 50, 50), 1)

        # Target 1: Intruder person moving across the restricted northern perimeter (y: 0.15 - 0.40)
        px1 = int((width * 0.10) + ((t * 5) % int(width * 0.75)))
        py1 = int(height * 0.18 + np.sin(t * 0.05) * 15)
        pw1, ph1 = 70, 160

        if person_sprite is not None:
            sprite1 = cv2.resize(person_sprite, (pw1, ph1))
            if py1 + ph1 < height and px1 + pw1 < width:
                frame[py1:py1 + ph1, px1:px1 + pw1] = sprite1
        else:
            cv2.rectangle(frame, (px1, py1), (px1 + pw1, py1 + ph1), (60, 70, 65), -1)
            cv2.circle(frame, (px1 + pw1 // 2, py1 - 15), 15, (70, 80, 75), -1)

        # Target 2: Loitering person dwelling in the checkpost gate zone (y: 0.45 - 0.85)
        # Stays in the zone for the entire duration -> triggers LOITERING_VIOLATION after threshold
        px2 = int(width * 0.42 + np.sin(t * 0.03) * 20)
        py2 = int(height * 0.45 + np.cos(t * 0.02) * 10)
        pw2, ph2 = 90, 200

        if person_sprite is not None:
            sprite2 = cv2.resize(person_sprite, (pw2, ph2))
            if py2 + ph2 < height and px2 + pw2 < width:
                frame[py2:py2 + ph2, px2:px2 + pw2] = sprite2
        else:
            cv2.rectangle(frame, (px2, py2), (px2 + pw2, py2 + ph2), (70, 60, 65), -1)
            cv2.circle(frame, (px2 + pw2 // 2, py2 - 15), 18, (80, 70, 75), -1)

        # Target 3: Tactical Vehicle in roadway
        vx = int((width * 0.58) + np.sin(t * 0.02) * 20)
        vy = int(height * 0.60 + ((t * 2) % int(height * 0.2)))
        vw, vh = 160, 100

        if vehicle_sprite is not None:
            v_resized = cv2.resize(vehicle_sprite, (vw, vh))
            if vy + vh < height and vx + vw < width:
                frame[vy:vy + vh, vx:vx + vw] = v_resized
        else:
            cv2.rectangle(frame, (vx, vy), (vx + vw, vy + vh), (50, 55, 60), -1)

        # License plate on vehicle
        plate_w, plate_h = 56, 18
        plate_x = vx + 10
        plate_y = vy + vh - 26
        if plate_y + plate_h < height and plate_x + plate_w < width:
            cv2.rectangle(frame, (plate_x, plate_y), (plate_x + plate_w, plate_y + plate_h), (240, 240, 240), -1)
            cv2.putText(frame, "PB08", (plate_x + 4, plate_y + 14), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), 2)

        # Low light attenuation if night
        if is_night:
            frame = (frame.astype(np.float32) * 0.35).astype(np.uint8)

        # Sensor noise
        noise = np.random.normal(0, 4 if not is_night else 8, frame.shape).astype(np.int16)
        noisy = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        out.write(noisy)

    out.release()
    print(f"[+] Successfully generated test video: {OUTPUT_FILE} ({total_frames} frames)")
    return OUTPUT_FILE


if __name__ == "__main__":
    generate_test_video(duration_sec=15)
