"""
Master Orchestration Script for AI Video Analytics Platform.
Initializes the Message Broker, Core Perception Engine, Specialist Workers
(ANPR, FRS, Anomaly), and Central Command Dashboard with graceful shutdown.
"""

import sys
import time
import signal
import argparse
from pathlib import Path

from config.settings import (
    DEFAULT_STREAM_SOURCE,
    DASHBOARD_HOST,
    DASHBOARD_PORT,
    DEVICE
)
from workers.base_worker import MessageBroker
from core.stream_engine import StreamEngine
from workers.anpr_worker import ANPRWorker
from workers.frs_worker import FRSWorker
from workers.anomaly_worker import AnomalyWorker
from dashboard.app import app, socketio, set_engine_and_broker


def print_banner():
    print("""
================================================================================
     DEFENSE-AI SOFTWARE-DEFINED VIDEO ANALYTICS PLATFORM (BOP-TACTICAL)
  Perimeter Defense, Zero-DCE Low-Light, ByteTrack, ANPR, FRS & Anomaly Suite
  Hardware Profile: NVIDIA RTX 3050 (4GB VRAM Guard) | Intel i5 | 24GB RAM
================================================================================
    """)


def main():
    parser = argparse.ArgumentParser(description="AI Video Analytics Platform Master Launcher")
    parser.add_argument("--source", type=str, default=DEFAULT_STREAM_SOURCE,
                        help="Video source: 'synthetic', webcam index ('0'), or RTSP URL ('rtsp://...')")
    parser.add_argument("--port", type=int, default=DASHBOARD_PORT, help="Dashboard port (default: 5000)")
    parser.add_argument("--host", type=str, default=DASHBOARD_HOST, help="Dashboard host (default: 0.0.0.0)")
    args = parser.parse_args()

    print_banner()

    # 1. Initialize Resilient Message Broker (Redis or In-Memory fallback)
    print("[1/5] Initializing Message Broker...")
    broker = MessageBroker()

    # 2. Initialize Core Perception Engine
    print(f"[2/5] Initializing Core Perception Engine (Source: '{args.source}', Device: '{DEVICE}')...")
    engine = StreamEngine(stream_source=args.source, broker=broker)

    # 3. Initialize Asynchronous Specialist Workers
    print("[3/5] Initializing Asynchronous Specialist Workers...")
    anpr_worker = ANPRWorker(broker=broker, alert_manager=engine.alert_manager)
    frs_worker = FRSWorker(broker=broker, alert_manager=engine.alert_manager)
    anomaly_worker = AnomalyWorker(broker=broker, alert_manager=engine.alert_manager)

    # 4. Start Core Pipeline and Workers
    print("[4/5] Launching Engine & Workers...")
    engine.start()
    anpr_worker.start()
    frs_worker.start()
    anomaly_worker.start()

    # 5. Bind Engine & Broker to Dashboard
    print(f"[5/5] Launching Central Command Dashboard at http://localhost:{args.port}...")
    set_engine_and_broker(engine, broker)

    def graceful_shutdown(sig=None, frame=None):
        print("\n[SHUTDOWN] Intercepted termination signal. Gracefully stopping workers...")
        engine.stop()
        anpr_worker.stop()
        frs_worker.stop()
        anomaly_worker.stop()
        print("[SHUTDOWN] All subsystems halted cleanly. Exiting.")
        sys.exit(0)

    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    try:
        socketio.run(app, host=args.host, port=args.port, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        graceful_shutdown()


if __name__ == "__main__":
    main()
