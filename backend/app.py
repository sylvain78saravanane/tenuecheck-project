"""
Application Flask — ENSITECH TenueCheck
Intégration Supabase (table alerts + Storage)
Upload direct en mémoire — aucun fichier local créé
Branche: dev-sylvain
"""
from __future__ import annotations

from flask import Flask, render_template, Response, jsonify
import cv2
import threading
from datetime import datetime, timezone
import os

from detector import DressCodeDetector
from alert_system import AlertSystem
from config import DETECTION_CONFIG, INTERFACE_CONFIG
from supabase_client import get_supabase

app = Flask(__name__)

# ── Configuration Supabase Storage ───────────────────────────
STORAGE_BUCKET = os.environ.get("STORAGE_BUCKET", "violations-snapshots")

# ── État global ───────────────────────────────────────────────
detector         = None
alert_system     = None
camera           = None
detection_active = True
current_frame    = None
frame_lock       = threading.Lock()

# Cache mémoire léger pour l'UI Flask
stats = {
    "total_detections": 0,
    "total_alerts":     0,
    "violations":       []
}


# ── Initialisation ────────────────────────────────────────────

def initialize_system():
    global detector, alert_system

    print("Initialisation du système...")
    detector     = DressCodeDetector()
    alert_system = AlertSystem()
    print("Système initialisé avec succès!")


# ── Caméra ────────────────────────────────────────────────────

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(INTERFACE_CONFIG["camera_index"])
        camera.set(cv2.CAP_PROP_FRAME_WIDTH,  INTERFACE_CONFIG["frame_width"])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, INTERFACE_CONFIG["frame_height"])
    return camera


# ── Supabase Storage ──────────────────────────────────────────

def upload_frame_to_storage(frame, violation_type: str) -> str | None:
    """
    Encode la frame en JPEG en mémoire et l'uploade directement
    vers Supabase Storage — aucun fichier local créé.
    Retourne le storage key ou None en cas d'erreur.
    """
    try:
        date_str    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        ts          = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        slug        = violation_type.replace(" ", "_").replace("/", "-")[:20]
        storage_key = f"{date_str}/{ts}_{slug}.jpg"

        # Encodage JPEG directement en mémoire (pas de fichier local)
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            print("[WARN] Encodage JPEG échoué")
            return None

        get_supabase().storage.from_(STORAGE_BUCKET).upload(
            path         = storage_key,
            file         = buffer.tobytes(),
            file_options = {"content-type": "image/jpeg"},
        )

        return storage_key

    except Exception as e:
        print(f"[WARN] Upload Storage échoué : {e}")
        return None


# ── Persistance ───────────────────────────────────────────────

def persist_alert(violation: dict, storage_key: str | None) -> str | None:
    """Insère l'alerte dans la table Supabase `alerts`."""
    try:
        res = get_supabase().table("alerts").insert({
            "violation_type": violation["type"],
            "confidence":     round(float(violation["confidence"]), 4),
            "image_s3_key":   storage_key,
        }).execute()
        return res.data[0]["id"] if res.data else None
    except Exception as e:
        print(f"[WARN] Impossible de persister l'alerte : {e}")
        return None


# ── Streaming vidéo ───────────────────────────────────────────

def generate_frames():
    global current_frame, detection_active, stats

    cam = get_camera()

    while True:
        success, frame = cam.read()
        if not success:
            frame = _create_error_frame("Camera non disponible")
        else:
            if detection_active and detector is not None:
                annotated_frame, violations = detector.process_frame(frame)

                if violations:
                    stats["total_detections"] += len(violations)

                    # Upload une seule image pour le lot de violations
                    storage_key = upload_frame_to_storage(
                        annotated_frame, violations[0]["type"]
                    )

                    for v in violations:
                        alert_id = persist_alert(v, storage_key)

                        stats["violations"].insert(0, {
                            "id":              alert_id,
                            "type":            v["type"],
                            "confidence":      v["confidence"],
                            "timestamp":       datetime.now().strftime("%H:%M:%S"),
                            "high_confidence": v.get("high_confidence", True),
                        })

                    # Garde les 50 dernières en cache
                    stats["violations"] = stats["violations"][:50]

                    stats["total_alerts"] += 1
                    alert_system.log_alert(violations)

                frame = annotated_frame
            else:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, f"ENSITECH - {ts}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Detection: PAUSE",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

        with frame_lock:
            current_frame = frame.copy()

        ret, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buffer.tobytes() + b"\r\n")


def _create_error_frame(message: str):
    import numpy as np
    frame    = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (30, 30, 50)
    cv2.putText(frame, message,
                (frame.shape[1] // 2 - 150, frame.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 2)
    return frame


# ── Routes Flask ──────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/violations")
def get_violations():
    """Retourne les 10 dernières alertes depuis Supabase."""
    try:
        res = (
            get_supabase()
            .table("alerts")
            .select("id, violation_type, confidence, detected_at, image_s3_key")
            .order("detected_at", desc=True)
            .limit(10)
            .execute()
        )
        return jsonify({
            "violations":       res.data,
            "total_detections": stats["total_detections"],
            "total_alerts":     stats["total_alerts"],
        })
    except Exception:
        return jsonify({
            "violations":       stats["violations"][:10],
            "total_detections": stats["total_detections"],
            "total_alerts":     stats["total_alerts"],
        })


@app.route("/api/toggle", methods=["POST"])
def toggle_detection():
    global detection_active
    detection_active = not detection_active
    return jsonify({"active": detection_active})


@app.route("/api/capture", methods=["POST"])
def capture_image():
    """Capture la frame courante et l'uploade directement dans Supabase Storage."""
    global current_frame
    if current_frame is None:
        return jsonify({"success": False, "error": "No frame available"})

    with frame_lock:
        frame_copy = current_frame.copy()

    # Upload direct en mémoire — aucun fichier local
    storage_key = upload_frame_to_storage(frame_copy, "capture_manuelle")

    # Persist dans la table alerts
    alert_id = None
    if storage_key:
        alert_id = persist_alert(
            {"type": "Capture manuelle", "confidence": 1.0},
            storage_key
        )

    return jsonify({
        "success":     storage_key is not None,
        "storage_key": storage_key,
        "alert_id":    alert_id,
    })


@app.route("/api/test_alert", methods=["POST"])
def test_alert():
    test_violations = [
        {"type": "Casquette", "confidence": 0.85},
        {"type": "Short",     "confidence": 0.72},
    ]
    alert_system.log_alert(test_violations)

    for v in test_violations:
        persist_alert(v, None)

    return jsonify({"success": True, "message": "Alertes de test enregistrées."})


@app.route("/api/stats")
def get_stats():
    return jsonify(stats)


@app.route("/api/config")
def get_config():
    from config import VETEMENTS_INTERDITS
    return jsonify({
        "vetements_interdits": list(set(VETEMENTS_INTERDITS.values())),
        "detection_config":    DETECTION_CONFIG,
    })


# ── Cleanup ───────────────────────────────────────────────────

def cleanup():
    global camera
    if camera is not None:
        camera.release()


# ── Point d'entrée ────────────────────────────────────────────

if __name__ == "__main__":
    initialize_system()

    print("\n" + "=" * 60)
    print("ENSITECH - Système de Contrôle du Code Vestimentaire")
    print("=" * 60)
    print("Interface : http://localhost:5001")
    print("=" * 60 + "\n")

    try:
        app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
    finally:
        cleanup()