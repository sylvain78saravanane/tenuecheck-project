from __future__ import annotations

import os
import sys
import cv2
import torch

# Fix pour PyTorch 2.6+
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

from ultralytics import YOLO
from datetime import datetime
from config import DETECTION_CONFIG


# ── Classes du modèle ENSITECH ────────────────────────────────

DRESSCODE_CLASSES = [
    "cap",
    "hat",
    "baseball_cap",
    "beanie",
    "bandana",
    "headwear"

]

DRESSCODE_DISPLAY_NAMES = {
    "cap": "Casquette",
    "baseball_cap": "Casquette",
    "hat": "Chapeau",
    "beanie": "Bonnet",
    "bandana": "Bandana",
    "headwear": "Couvre-chef"
}

MODEL_PATH       = "dresscode_yolo.pt"
CONF_THRESHOLD   = DETECTION_CONFIG.get("confidence_threshold", 0.5)
HIGH_CONF_THRESH = 0.70


# ── Détecteur ─────────────────────────────────────────────────

class DressCodeDetector:
    """
    Détecteur de code vestimentaire ENSITECH.
    Charge dresscode_yolo.pt et détecte les 10 classes de vêtements interdits.
    Arrête le programme si le modèle est absent.
    """

    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            print(f"[ERREUR] Modèle introuvable : {MODEL_PATH}")
            print("Placez dresscode_yolo.pt dans le dossier backend/ et relancez.")
            sys.exit(1)

        print(f"Chargement du modèle ENSITECH ({MODEL_PATH})...")
        self.model           = YOLO(MODEL_PATH)
        self.frame_count     = 0
        self.last_detections = []

        print("[MODE] Detection avec modèle personnalisé ENSITECH")
        print(f"Classes surveillées : {list(DRESSCODE_DISPLAY_NAMES.values())}")


    # ── Détection ─────────────────────────────────────────────

    def _detect(self, frame) -> list[dict]:
        """
        Lance l'inférence YOLO sur la frame.
        Retourne une liste de détections :
            { bbox, class_name, display_name, confidence }
        """
        results    = self.model(frame, conf=CONF_THRESHOLD, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                if cls_id >= len(DRESSCODE_CLASSES):
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name      = DRESSCODE_CLASSES[cls_id]
                detections.append({
                    "bbox":         (x1, y1, x2, y2),
                    "class_name":   class_name,
                    "display_name": DRESSCODE_DISPLAY_NAMES[class_name],
                    "confidence":   conf,
                })

        return detections


    # ── Annotation ────────────────────────────────────────────

    def _annotate(self, frame, detections) -> tuple:
        """
        Dessine les bounding boxes et labels sur la frame.
        Retourne (frame_annotée, liste_violations).
        """
        annotated      = frame.copy()
        all_violations = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf            = det["confidence"]
            label           = det["display_name"]
            is_high         = conf >= HIGH_CONF_THRESH

            color   = (0, 0, 255) if is_high else (0, 165, 255)
            prefix  = "INTERDIT" if is_high else "SUSPECT"
            message = f"{prefix}: {label} ({conf*100:.0f}%)"

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3 if is_high else 2)

            # Fond du label
            text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated,
                          (x1, y1 - text_size[1] - 10),
                          (x1 + text_size[0], y1),
                          color, -1)
            cv2.putText(annotated, message,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if is_high:
                all_violations.append({
                    "type":            label,
                    "confidence":      conf,
                    "bbox":            det["bbox"],
                    "timestamp":       datetime.now(),
                    "high_confidence": True,
                })

        # Overlay timestamp + statut
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated, f"ENSITECH - {ts}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        status_text  = f"ALERTES: {len(all_violations)} violation(s)" if all_violations else "Statut: Aucune violation"
        status_color = (0, 0, 255) if all_violations else (0, 255, 0)
        cv2.putText(annotated, status_text,
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        cv2.putText(annotated, f"Mode: ENSITECH Custom YOLO ({MODEL_PATH})",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        return annotated, all_violations


    # ── API publique ──────────────────────────────────────────

    def process_frame(self, frame) -> tuple:
        """
        Traite une frame et retourne (frame_annotée, violations).
        violations est une liste vide sur les frames skippées.
        """
        self.frame_count += 1

        # Skip frames pour optimiser les performances
        if self.frame_count % DETECTION_CONFIG.get("frame_skip", 2) != 0:
            annotated, _ = self._annotate(frame, self.last_detections)
            return annotated, []

        self.last_detections = self._detect(frame)
        return self._annotate(frame, self.last_detections)


# ── Test standalone ───────────────────────────────────────────

def main():
    """Lance la détection en mode standalone (sans Flask)."""
    detector = DressCodeDetector()
    cap      = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la caméra")
        return

    print("Détection démarrée — appuyez sur 'q' pour quitter")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated, violations = detector.process_frame(frame)
        if violations:
            print(f"[ALERTE] {[v['type'] for v in violations]}")
        cv2.imshow("ENSITECH - Contrôle Code Vestimentaire", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()