"""
Dataset Preparation & Synthesizer for BOP Threat & Plate Training.
Prepares standard YOLO-format directory structure and generates sample data
for validating the 4GB VRAM-safe training pipeline.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import yaml
from config.settings import DATA_DIR

DATASET_ROOT = DATA_DIR / "datasets" / "plate_dataset"


def prepare_plate_dataset(num_train: int = 24, num_val: int = 6):
    """
    Creates sample dataset in YOLO format:
      images/train, images/val
      labels/train, labels/val
      data.yaml
    """
    print(f"[Dataset] Preparing sample YOLO plate dataset at: {DATASET_ROOT}")
    
    dirs = [
        DATASET_ROOT / "images" / "train",
        DATASET_ROOT / "images" / "val",
        DATASET_ROOT / "labels" / "train",
        DATASET_ROOT / "labels" / "val",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # Generate sample synthetic vehicle/plate pairs
    splits = [("train", num_train), ("val", num_val)]
    for split_name, count in splits:
        img_dir = DATASET_ROOT / "images" / split_name
        lbl_dir = DATASET_ROOT / "labels" / split_name

        for i in range(count):
            img_w, img_h = 640, 480
            canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            # Background
            canvas[:, :] = (np.random.randint(40, 80), np.random.randint(40, 80), np.random.randint(40, 80))

            # Draw vehicle body
            vx = np.random.randint(100, 300)
            vy = np.random.randint(150, 250)
            vw = np.random.randint(220, 300)
            vh = np.random.randint(140, 180)
            cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), (120, 130, 140), -1)

            # Draw license plate
            pw = np.random.randint(80, 110)
            ph = np.random.randint(25, 35)
            px = vx + (vw - pw) // 2
            py = vy + vh - ph - 15
            cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (230, 230, 230), -1)
            cv2.putText(canvas, f"BOP{i:02d}", (px + 5, py + 20), cv2.FONT_HERSHEY_PLAIN, 1.2, (10, 10, 10), 2)

            # Save image
            img_file = img_dir / f"plate_sample_{split_name}_{i:03d}.jpg"
            cv2.imwrite(str(img_file), canvas)

            # Save YOLO annotation: <class_id> <x_center> <y_center> <width> <height> (normalized)
            x_center = (px + pw / 2.0) / img_w
            y_center = (py + ph / 2.0) / img_h
            norm_w = pw / img_w
            norm_h = ph / img_h

            lbl_file = lbl_dir / f"plate_sample_{split_name}_{i:03d}.txt"
            with open(lbl_file, "w", encoding="utf-8") as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

    # Generate data.yaml
    data_yaml = {
        "path": str(DATASET_ROOT.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "license_plate"
        }
    }

    yaml_path = DATASET_ROOT / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"[Dataset] Generated {num_train} train and {num_val} val samples. Data config: {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    prepare_plate_dataset()
