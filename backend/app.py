"""
Application Flask pour le système de détection de code vestimentaire ENSITECH
Interface web avec streaming vidéo en temps réel
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import threading
from datetime import datetime
import os
import json

from detector import DressCodeDetector
from alert_system import AlertSystem
from config import DETECTION_CONFIG, INTERFACE_CONFIG

app = Flask(__name__)

# Variables globales
detector = None
alert_system = None
camera = None
detection_active = True
current_frame = None
frame_lock = threading.Lock()

# Statistiques
stats = {
    "total_detections": 0,
    "total_alerts": 0,
    "violations": []
}


def initialize_system():
    """
    Initialise le détecteur et le système d'alertes
    """
    global detector, alert_system

    print("Initialisation du système...")
    detector = DressCodeDetector()
    alert_system = AlertSystem()
    print("Système initialisé avec succès!")


def get_camera():
    """
    Retourne l'instance de la caméra
    """
    global camera

    if camera is None:
        camera = cv2.VideoCapture(INTERFACE_CONFIG["camera_index"])
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, INTERFACE_CONFIG["frame_width"])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, INTERFACE_CONFIG["frame_height"])

    return camera


def generate_frames():
    """
    Générateur de frames pour le streaming vidéo
    """
    global current_frame, detection_active, stats

    cam = get_camera()

    while True:
        success, frame = cam.read()

        if not success:
            print("Erreur: Impossible de lire la caméra")
            # Créer une frame noire avec message d'erreur
            frame = create_error_frame("Camera non disponible")
        else:
            if detection_active and detector is not None:
                # Traiter la frame pour la détection
                annotated_frame, violations = detector.process_frame(frame)

                if violations:
                    stats["total_detections"] += len(violations)

                    # Ajouter les violations à la liste
                    for v in violations:
                        violation_entry = {
                            "type": v["type"],
                            "confidence": v["confidence"],
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "high_confidence": v.get("high_confidence", True)
                        }
                        stats["violations"].insert(0, violation_entry)

                        # Garder seulement les 50 dernières violations
                        if len(stats["violations"]) > 50:
                            stats["violations"] = stats["violations"][:50]

                    # Sauvegarder l'image et envoyer l'alerte
                    image_path = detector.save_alert_image(annotated_frame, violations)

                    if image_path:
                        stats["total_alerts"] += 1
                        # Log l'alerte (l'envoi d'email peut être activé dans alert_system)
                        alert_system.log_alert(violations, image_path)

                frame = annotated_frame
            else:
                # Ajouter juste le timestamp si détection désactivée
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, f"ENSITECH - {timestamp}",
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Detection: PAUSE",
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

        # Encoder la frame en JPEG
        with frame_lock:
            current_frame = frame.copy()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def create_error_frame(message):
    """
    Crée une frame avec un message d'erreur
    """
    frame = cv2.imread('static/no_camera.png') if os.path.exists('static/no_camera.png') else None

    if frame is None:
        frame = 0 * cv2.imread('static/placeholder.png') if os.path.exists('static/placeholder.png') else None

    if frame is None:
        # Créer une image noire
        frame = (50 * (1 + 0*cv2.UMat(480, 640, cv2.CV_8UC3))).get()
        frame = 50 * (1 + 0*(__import__('numpy').zeros((480, 640, 3), dtype=__import__('numpy').uint8)))

    import numpy as np
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (30, 30, 50)  # Fond sombre

    # Ajouter le message
    cv2.putText(frame, message,
              (frame.shape[1]//2 - 150, frame.shape[0]//2),
              cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 2)

    cv2.putText(frame, "Verifiez la connexion de la camera",
              (frame.shape[1]//2 - 200, frame.shape[0]//2 + 40),
              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    return frame


# Routes Flask
@app.route('/')
def index():
    """
    Page principale
    """
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """
    Route pour le streaming vidéo
    """
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/violations')
def get_violations():
    """
    API pour récupérer les violations
    """
    return jsonify({
        "violations": stats["violations"][:10],
        "total_detections": stats["total_detections"],
        "total_alerts": stats["total_alerts"]
    })


@app.route('/api/toggle', methods=['POST'])
def toggle_detection():
    """
    Active/désactive la détection
    """
    global detection_active
    detection_active = not detection_active
    return jsonify({"active": detection_active})


@app.route('/api/capture', methods=['POST'])
def capture_image():
    """
    Capture une image
    """
    global current_frame

    if current_frame is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        filepath = os.path.join("alerts", filename)
        os.makedirs("alerts", exist_ok=True)

        with frame_lock:
            cv2.imwrite(filepath, current_frame)

        return jsonify({"success": True, "filename": filename})

    return jsonify({"success": False, "error": "No frame available"})


@app.route('/api/test_alert', methods=['POST'])
def test_alert():
    """
    Envoie une alerte de test
    """
    test_violations = [
        {"type": "Casquette", "confidence": 0.85},
        {"type": "Short", "confidence": 0.72}
    ]

    alert_system.log_alert(test_violations)

    return jsonify({
        "success": True,
        "message": "Alerte de test enregistrée dans alerts/alerts_log.txt"
    })


@app.route('/api/stats')
def get_stats():
    """
    Retourne les statistiques
    """
    return jsonify(stats)


@app.route('/api/config')
def get_config():
    """
    Retourne la configuration
    """
    from config import VETEMENTS_INTERDITS
    return jsonify({
        "vetements_interdits": list(set(VETEMENTS_INTERDITS.values())),
        "detection_config": DETECTION_CONFIG
    })


def cleanup():
    """
    Nettoie les ressources
    """
    global camera
    if camera is not None:
        camera.release()


if __name__ == '__main__':
    # Initialiser le système
    initialize_system()

    # Créer le dossier alerts
    os.makedirs("alerts", exist_ok=True)

    print("\n" + "="*60)
    print("ENSITECH - Système de Contrôle du Code Vestimentaire")
    print("="*60)
    print("\nDémarrage du serveur web...")
    print("Accédez à l'interface: http://localhost:5000")
    print("\nAppuyez sur Ctrl+C pour arrêter le serveur")
    print("="*60 + "\n")

    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        cleanup()
