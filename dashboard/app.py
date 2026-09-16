"""
Central Command Dashboard - Flask & Flask-SocketIO Server.
Provides real-time MJPEG video streaming, WebSocket alert dispatch,
interactive virtual fence polygon configuration, and hardware telemetry monitoring.
"""

import os
import cv2
import json
import time
import psutil
import numpy as np
from flask import Flask, Response, request, jsonify
from flask_socketio import SocketIO, emit
from pathlib import Path
from typing import Optional

from config.settings import (
    DASHBOARD_HOST,
    DASHBOARD_PORT,
    DASHBOARD_DEBUG,
    SECRET_KEY,
    VIRTUAL_FENCES,
    WATCHLIST_FILE,
    LOGS_DIR
)
from core.stream_engine import StreamEngine
from workers.base_worker import MessageBroker

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global references
stream_engine: Optional[StreamEngine] = None
message_broker: Optional[MessageBroker] = None


def set_engine_and_broker(engine: StreamEngine, broker: MessageBroker):
    """Binds active perception engine and broker to Flask."""
    global stream_engine, message_broker
    stream_engine = engine
    message_broker = broker

    # Register AlertManager callback to instantly broadcast alerts to WebSockets
    if stream_engine and stream_engine.alert_manager:
        stream_engine.alert_manager.register_callback(broadcast_alert_to_clients)


def broadcast_alert_to_clients(alert_payload: dict):
    """Pushes new alert to all connected dashboard WebSockets."""
    try:
        socketio.emit('alert_event', alert_payload)
    except Exception as e:
        print(f"[Dashboard] SocketIO broadcast error: {e}")



def generate_mjpeg_stream():
    """Generates continuous multipart JPEG stream for low-latency web viewing."""
    standby_counter = 0
    while True:
        frame = None
        if stream_engine is not None:
            # 1. Prefer tactical annotated HUD frame
            frame = stream_engine.get_latest_annotated_frame()
            # 2. If annotated HUD is still warming up, immediately stream raw live camera frame
            if frame is None and stream_engine.stream_reader is not None:
                frame = stream_engine.stream_reader.get_frame()
        
        if frame is None:
            standby_counter = (standby_counter + 1) % 60
            standby_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            dot_str = "." * ((standby_counter // 15) + 1)
            cv2.putText(standby_frame, f"TACTICAL STREAM CONNECTING / NO SIGNAL{dot_str}", (320, 340),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 240, 255), 2, cv2.LINE_AA)
            cv2.putText(standby_frame, "Awaiting RTSP stream or video feed initialization", (400, 385),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (140, 160, 180), 1, cv2.LINE_AA)
            frame = standby_frame

        # Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' +
                   frame_bytes + b'\r\n')
        time.sleep(0.033)  # ~30 FPS


@app.route('/video_feed')
def video_feed():
    """Video streaming route for CCTV / perception canvas."""
    response = Response(generate_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, pre-check=0, post-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@app.route('/api/stats')
def get_system_stats():
    """Returns real-time hardware telemetry and pipeline performance metrics."""
    # GPU / VRAM Telemetry
    gpu_data = {
        "name": "NVIDIA GeForce RTX 3050 Laptop GPU",
        "vram_total_mb": 4096,
        "vram_used_mb": 0,
        "vram_percent": 0.0,
        "cuda_available": False
    }

    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_data["cuda_available"] = True
        gpu_data["name"] = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        free, _ = torch.cuda.mem_get_info(0)
        free = free / (1024 ** 2)
        used = total - free
        gpu_data["vram_total_mb"] = round(total)
        gpu_data["vram_used_mb"] = round(used)
        gpu_data["vram_percent"] = round((used / total) * 100.0, 1)

    # CPU & RAM Telemetry
    ram = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=None)

    fps = stream_engine.fps if stream_engine else 0.0
    tracks = stream_engine.active_tracks_count if stream_engine else 0
    zero_dce = stream_engine.low_light_active if stream_engine else False

    return jsonify({
        "gpu": gpu_data,
        "system": {
            "cpu_percent": cpu_percent,
            "ram_total_gb": round(ram.total / (1024 ** 3), 1),
            "ram_used_gb": round(ram.used / (1024 ** 3), 1),
            "ram_percent": ram.percent
        },
        "pipeline": {
            "fps": round(fps, 1),
            "active_tracks": tracks,
            "zero_dce_active": zero_dce,
            "camera_id": stream_engine.camera_id if stream_engine else "UNKNOWN"
        }
    })


@app.route('/api/fences', methods=['GET', 'POST'])
def manage_fences():
    """Retrieves or updates virtual fence polygon coordinates."""
    if request.method == 'GET':
        mode = stream_engine.perimeter_mode if stream_engine else "manual"
        if stream_engine and stream_engine.fence_manager:
            fences_list = []
            for fid, f in stream_engine.fence_manager.fences.items():
                fences_list.append({
                    "id": fid,
                    "name": f["name"],
                    "zone_type": f["zone_type"],
                    "color": f["color"],
                    "polygon": f.get("polygon", []),
                    "polygon_pixels": f["polygon_pixels"],
                    "allowed_classes": f.get("allowed_classes", []),
                    "loiter_threshold_sec": f["loiter_threshold_sec"],
                    "is_auto": f.get("is_auto", False)
                })
            return jsonify({"fences": fences_list, "mode": mode})
        return jsonify({"fences": VIRTUAL_FENCES, "mode": mode})

    elif request.method == 'POST':
        # Interactive drawing updates from UI canvas
        data = request.json
        if not data or "fences" not in data:
            return jsonify({"status": "error", "message": "Invalid fence payload"}), 400

        new_fences = data["fences"]
        if stream_engine and stream_engine.fence_manager:
            stream_engine.update_manual_fences(new_fences)
            # Broadcast updated fences to all clients
            socketio.emit('fences_updated', {"fences": new_fences, "mode": "manual"})
            return jsonify({"status": "success", "count": len(new_fences), "mode": "manual"})

        return jsonify({"status": "error", "message": "Engine not initialized"}), 500


@app.route('/api/fences/mode', methods=['POST'])
def set_fence_mode():
    """Toggles perimeter detection mode between 'auto' and 'manual'."""
    data = request.json or {}
    mode = data.get("mode", "manual").lower()
    if mode not in ["auto", "manual"]:
        return jsonify({"status": "error", "message": "Invalid mode. Choose 'auto' or 'manual'"}), 400

    if stream_engine:
        active_fences = stream_engine.set_perimeter_mode(mode)
        socketio.emit('fences_updated', {"fences": active_fences, "mode": mode})
        return jsonify({"status": "success", "mode": mode, "count": len(active_fences)})

    return jsonify({"status": "error", "message": "Engine not initialized"}), 500


@app.route('/api/fences/auto_detect', methods=['POST'])
def force_auto_detect():
    """Forces an immediate computer-vision re-scan of perimeter boundaries."""
    if stream_engine:
        detected = stream_engine.trigger_auto_detection()
        socketio.emit('fences_updated', {"fences": detected, "mode": "auto"})
        return jsonify({"status": "success", "mode": "auto", "count": len(detected), "fences": detected})

    return jsonify({"status": "error", "message": "Engine not initialized"}), 500


@app.route('/api/alerts', methods=['GET'])
def get_recent_alerts():
    """Retrieves recent alerts from disk log."""
    log_file = LOGS_DIR / "alerts.jsonl"
    alerts = []
    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()[-40:]  # Last 40 alerts
                for line in reversed(lines):
                    if line.strip():
                        alerts.append(json.loads(line))
        except Exception as e:
            print(f"[Dashboard] Read alerts error: {e}")
    return jsonify({"alerts": alerts})


@app.route('/api/watchlist', methods=['GET', 'POST'])
def handle_watchlist():
    """Returns or adds entries to security watchlist."""
    if request.method == 'GET':
        if WATCHLIST_FILE.exists():
            with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
        return jsonify({"watchlist": []})

    elif request.method == 'POST':
        new_entry = request.json
        if not new_entry or "name" not in new_entry:
            return jsonify({"status": "error", "message": "Missing name"}), 400

        data = {"watchlist": []}
        if WATCHLIST_FILE.exists():
            with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

        new_entry["id"] = f"WLIST-{int(time.time()*1000)}"
        data["watchlist"].append(new_entry)

        with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return jsonify({"status": "success", "entry": new_entry})


# WebSocket Event Handlers
@socketio.on('connect')
def handle_connect():
    emit('connected', {'status': 'ONLINE', 'timestamp': time.time()})


@socketio.on('client_update_fence')
def handle_ws_fence_update(data):
    """Handles real-time polygon creation directly over WebSockets."""
    if data and "fence" in data and stream_engine:
        new_fence = data["fence"]
        # Append or update fence
        current_cfgs = list(stream_engine.manual_fences)
        
        # Check if updating an existing fence
        updated = False
        for i, f in enumerate(current_cfgs):
            if f.get("id") == new_fence.get("id"):
                current_cfgs[i] = new_fence
                updated = True
                break
                
        if not updated:
            current_cfgs.append(new_fence)
            
        stream_engine.update_manual_fences(current_cfgs)
        emit('fences_updated', {'status': 'updated', 'fences': current_cfgs, 'mode': 'manual'}, broadcast=True)


def run_dashboard():
    """Runs Flask-SocketIO app."""
    print(f"[Dashboard] Launching Central Command Server at http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    socketio.run(app, host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=DASHBOARD_DEBUG, use_reloader=False, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    run_dashboard()
