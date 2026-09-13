"""
VRAM-Safe License Plate Detector Fine-Tuning Pipeline.
Strictly configured for NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM limit),
Intel 11th Gen i5 CPU (4 workers), and 24GB System RAM caching.
"""

import os
import sys
import psutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Hardware Memory Management Guardrails
# Prevent PyTorch memory fragmentation on 4GB VRAM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from config.settings import (
    WEIGHTS_DIR,
    DATA_DIR,
    MAX_TRAINING_BATCH,
    DATALOADER_WORKERS,
    RAM_DATASET_CACHE,
    CUDA_AMP_ENABLED
)
from training.prepare_dataset import prepare_plate_dataset, DATASET_ROOT

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


def check_hardware_guards():
    """Validates VRAM budget and hardware constraints prior to training launch."""
    print("=" * 70)
    print("  PRE-FLIGHT HARDWARE & VRAM MEMORY AUDIT")
    print("=" * 70)

    # 1. System RAM Check
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    ram_avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    print(f"[*] System RAM: Total {ram_gb:.1f} GB | Available: {ram_avail_gb:.1f} GB")
    if ram_gb >= 16.0:
        print("    [+] High-capacity RAM detected. Dataset caching in RAM (cache=True) enabled.")

    # 2. CPU Workers
    cpu_cores = os.cpu_count() or 4
    print(f"[*] CPU Cores: {cpu_cores} | Dataloader Workers assigned: {DATALOADER_WORKERS}")

    # 3. GPU & VRAM Audit
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        print("[!] Warning: CUDA device not detected. Training will execute in CPU fallback mode.")
        return "cpu", 4

    gpu_name = torch.cuda.get_device_name(0)
    total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    free_vram_mb, _ = torch.cuda.mem_get_info(0)
    free_vram_mb = free_vram_mb / (1024 ** 2)

    print(f"[*] GPU Device: {gpu_name}")
    print(f"[*] Total VRAM: {total_vram_mb:.0f} MB | Free VRAM: {free_vram_mb:.0f} MB")

    # Strict batch sizing rule for 4GB VRAM
    # If free VRAM < 2500 MB (due to OS/display overhead), step down to batch=4 to prevent OOM
    if free_vram_mb < 2500.0:
        batch_size = 4
        print(f"[!] Available VRAM is below 2.5GB. Safely adjusting batch={batch_size} to prevent OOM.")
    else:
        batch_size = MAX_TRAINING_BATCH  # batch=8
        print(f"[+] Sufficient VRAM available. Enforcing optimal batch={batch_size}.")

    print(f"[*] FP16 Automatic Mixed Precision (amp): {CUDA_AMP_ENABLED}")
    print("=" * 70)

    return 0, batch_size


def train_plate_model(
    epochs: int = 15,
    imgsz: int = 640,
    pretrained_model: str = "yolov8n.pt"
):
    """
    Executes VRAM-guarded training session.
    Enforces: batch=8 (or 4), amp=True, workers=4, cache=True.
    """
    device, batch_size = check_hardware_guards()

    if not ULTRALYTICS_AVAILABLE:
        print("[ERROR] 'ultralytics' package not installed. Run 'pip install ultralytics'.")
        return None

    # Ensure dataset is prepared
    yaml_config = DATASET_ROOT / "data.yaml"
    if not yaml_config.exists():
        yaml_config = prepare_plate_dataset()

    print(f"\n[Training] Loading base model: {pretrained_model}...")
    model = YOLO(pretrained_model)

    print("\n[Training] Launching fine-tuning with strict memory guards:")
    print(f"  - batch: {batch_size}")
    print(f"  - amp (FP16): {CUDA_AMP_ENABLED}")
    print(f"  - workers: {DATALOADER_WORKERS}")
    print(f"  - cache: {RAM_DATASET_CACHE}")
    print(f"  - imgsz: {imgsz}")
    print(f"  - epochs: {epochs}\n")

    # Clear CUDA cache before starting
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        results = model.train(
            data=str(yaml_config),
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            workers=DATALOADER_WORKERS,
            amp=CUDA_AMP_ENABLED,
            cache=RAM_DATASET_CACHE,
            save=True,
            project=str(DATA_DIR / "runs" / "plate_train"),
            name="bop_plate_model",
            exist_ok=True,
            verbose=True
        )

        # Copy best model to weights dir
        best_pt = Path(results.save_dir) / "weights" / "best.pt"
        target_pt = WEIGHTS_DIR / "plate_detector_best.pt"
        if best_pt.exists():
            import shutil
            shutil.copy(best_pt, target_pt)
            print(f"\n[Training Complete] Optimized model weights saved to: {target_pt}")

        return results

    except torch.cuda.OutOfMemoryError as oom:
        print(f"\n[CUDA OOM GUARD TRIGGERED]: {oom}")
        print("Re-trying with conservative batch=4...")
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model.train(
            data=str(yaml_config),
            epochs=epochs,
            batch=4,
            imgsz=480,  # Scaled down resolution for guaranteed safety
            device=device,
            workers=DATALOADER_WORKERS,
            amp=True,
            cache=True
        )


if __name__ == "__main__":
    train_plate_model(epochs=3)
