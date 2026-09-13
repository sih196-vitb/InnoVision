"""
General Purpose VRAM-Guarded YOLO Fine-Tuning Pipeline for BOP Threat Detection.
Enforces batch=8 (or 4), amp=True (FP16), workers=4, and cache=True for RTX 3050 4GB VRAM.
"""

import os
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Prevent CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from config.settings import (
    WEIGHTS_DIR,
    DATA_DIR,
    MAX_TRAINING_BATCH,
    DATALOADER_WORKERS,
    RAM_DATASET_CACHE,
    CUDA_AMP_ENABLED
)
from training.train_plate_detector import check_hardware_guards

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


def train_custom_threat_model(
    data_yaml_path: str,
    model_variant: str = "yolov8n.pt",
    epochs: int = 25,
    imgsz: int = 640
):
    """Fine-tunes YOLO model with explicit 4GB VRAM constraints."""
    device, batch_size = check_hardware_guards()

    if not ULTRALYTICS_AVAILABLE:
        print("[ERROR] 'ultralytics' not available. Please install it.")
        return

    print(f"[Training] Initializing model {model_variant} on dataset {data_yaml_path}...")
    model = YOLO(model_variant)

    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        workers=DATALOADER_WORKERS,
        amp=CUDA_AMP_ENABLED,
        cache=RAM_DATASET_CACHE,
        project=str(DATA_DIR / "runs" / "threat_train"),
        name="bop_threat_model",
        exist_ok=True,
        verbose=True
    )

    print(f"[Training] Model training complete. Artifacts saved in: {results.save_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VRAM-Safe BOP Threat Model Fine-Tuning")
    parser.add_argument("--data", type=str, default=str(DATA_DIR / "datasets" / "plate_dataset" / "data.yaml"))
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    train_custom_threat_model(
        data_yaml_path=args.data,
        model_variant=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz
    )
