"""
Module de detection de vetements interdits
Utilise YOLOv8 pour detecter les personnes + modele personnalise ENSITECH pour les vetements

Modes de fonctionnement:
1. Modele YOLO personnalise (dresscode_yolo.pt) - Recommande, detecte tous les vetements interdits
2. Fashion-MNIST (fashion_classifier.pth) - Classification basique vetements
3. Analyse visuelle seule - Mode de secours
"""

import os
import cv2
import numpy as np
import torch

# Fix pour PyTorch 2.6+
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

from ultralytics import YOLO
from datetime import datetime
from config import (
    VETEMENTS_INTERDITS,
    DETECTION_CONFIG,
    MESSAGES_ALERTE
)


# Classes du modele personnalise ENSITECH
DRESSCODE_CLASSES = [
    "short",        # 0
    "mini_skirt",   # 1
    "crop_top",     # 2
    "sportswear",   # 3
    "ripped_jeans", # 4
    "flip_flops",   # 5
    "cap",          # 6
    "hat",          # 7
    "beanie",       # 8
    "bandana",      # 9
]

# Noms d'affichage pour les classes
DRESSCODE_DISPLAY_NAMES = {
    "short": "Short/Bermuda",
    "mini_skirt": "Mini-jupe",
    "crop_top": "Crop top",
    "sportswear": "Tenue de sport",
    "ripped_jeans": "Jean troue",
    "flip_flops": "Tongs/Sandales",
    "cap": "Casquette",
    "hat": "Chapeau",
    "beanie": "Bonnet",
    "bandana": "Bandana",
}

# Classes Fashion-MNIST (mode de secours)
FASHION_LABELS = ["top", "trouser", "pullover", "dress", "coat",
                  "sandal", "shirt", "sneaker", "bag", "ankle_boot"]

# Mapping vers les vetements interdits
FASHION_TO_DRESSCODE = {
    "sandal": ("Sandales/Tongs", True),       # INTERDIT
    "top": ("Top", False),                     # Autorise (sauf si court)
    "dress": ("Robe", False),                  # Autorise (sauf si courte)
    "trouser": ("Pantalon", False),            # Autorise
    "pullover": ("Pull", False),               # Autorise
    "coat": ("Manteau", False),                # Autorise
    "shirt": ("Chemise", False),               # Autorise
    "sneaker": ("Baskets", False),             # Autorise
    "bag": ("Sac", False),                     # Autorise
    "ankle_boot": ("Bottines", False),         # Autorise
}


class DressCodeDetector:
    """
    Detecteur de code vestimentaire utilisant:
    - YOLOv8 personnalise (dresscode_yolo.pt) pour detecter les vetements interdits
    - OU YOLOv8 standard pour la detection de personnes + Fashion-MNIST
    - Analyse visuelle pour les cas non couverts
    """

    def __init__(self):
        # Mode de detection
        self.use_custom_model = False
        self.dresscode_model = None
        self.person_model = None

        # Essayer de charger le modele personnalise ENSITECH
        custom_model_path = "dresscode_yolo.pt"
        if os.path.exists(custom_model_path):
            try:
                print("Chargement du modele personnalise ENSITECH...")
                self.dresscode_model = YOLO(custom_model_path)
                self.use_custom_model = True
                print("Modele personnalise charge avec succes!")
            except Exception as e:
                print(f"Erreur chargement modele personnalise: {e}")
                self.use_custom_model = False

        # Si pas de modele personnalise, utiliser YOLOv8 standard pour les personnes
        if not self.use_custom_model:
            print("Chargement du modele YOLOv8 standard...")
            self.person_model = YOLO('yolov8n.pt')
            self.person_class_id = 0  # COCO: 0=person
            print("Modele personnalise non trouve. Utilisation du mode standard.")
            print("Pour entrainer le modele: python train_custom_yolo.py train")

        # Charger le classificateur Fashion-MNIST si disponible (mode standard)
        self.fashion_model = None
        self.use_fashion_model = False
        if not self.use_custom_model:
            self._load_fashion_model()

        # Configuration
        self.alert_history = {}
        self.frame_count = 0
        self.vetements_interdits_labels = list(set(VETEMENTS_INTERDITS.values()))
        self.last_detections = []

        # Afficher le mode actif
        if self.use_custom_model:
            print("\n[MODE] Detection avec modele personnalise ENSITECH")
            print(f"Classes detectees: {DRESSCODE_CLASSES}")
        elif self.use_fashion_model:
            print("\n[MODE] Detection avec YOLO + Fashion-MNIST")
        else:
            print("\n[MODE] Detection avec YOLO + Analyse visuelle")

        print(f"Vetements interdits surveilles: {list(DRESSCODE_DISPLAY_NAMES.values())}")

    def _load_fashion_model(self):
        """Charge le modele Fashion-MNIST (PyTorch) si disponible"""
        model_path = "fashion_classifier.pth"

        if os.path.exists(model_path):
            try:
                from train_fashion_mnist import FashionCNN
                print("Chargement du classificateur Fashion-MNIST...")

                self.fashion_model = FashionCNN(num_classes=10)
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                self.fashion_model.load_state_dict(checkpoint['model_state_dict'])
                self.fashion_model.eval()
                self.use_fashion_model = True
                print("Classificateur Fashion-MNIST charge avec succes!")
            except Exception as e:
                print(f"Erreur chargement modele Fashion: {e}")
                self.fashion_model = None
                self.use_fashion_model = False
        else:
            print("Modele Fashion-MNIST non trouve.")
            print("Pour l'entrainer: python train_fashion_mnist.py")
            self.use_fashion_model = False

    def detect_with_custom_model(self, frame):
        """
        Detection avec le modele personnalise ENSITECH
        Detecte directement tous les vetements interdits
        """
        results = self.dresscode_model(frame, conf=0.5, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if cls_id < len(DRESSCODE_CLASSES):
                    class_name = DRESSCODE_CLASSES[cls_id]
                    display_name = DRESSCODE_DISPLAY_NAMES.get(class_name, class_name)

                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "class": class_name,
                        "display_name": display_name,
                        "confidence": conf
                    })

        return detections

    def detect_persons(self, frame):
        """Detecte les personnes dans l'image avec YOLOv8 standard"""
        if self.person_model is None:
            return []

        results = self.person_model(frame, conf=DETECTION_CONFIG["confidence_threshold"], verbose=False)
        persons = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls == self.person_class_id:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    persons.append({
                        "bbox": (x1, y1, x2, y2),
                        "confidence": conf
                    })

        return persons

    def _classify_clothing_region(self, region):
        """
        Classifie une region de vetement avec le modele Fashion-MNIST (PyTorch)
        Retourne (classe, confiance) ou None si pas de modele
        """
        if not self.use_fashion_model or self.fashion_model is None:
            return None

        try:
            # Preparer l'image pour Fashion-MNIST (28x28 grayscale)
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (28, 28))
            normalized = resized.astype("float32") / 255.0

            # Convertir en tensor PyTorch [1, 1, 28, 28]
            input_tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)

            # Predire
            with torch.no_grad():
                outputs = self.fashion_model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, pred_class = probs.max(1)

            return (FASHION_LABELS[pred_class.item()], confidence.item())

        except Exception:
            return None

    def analyze_clothing(self, frame, person_bbox):
        """
        Analyse les vetements d'une personne detectee.
        Combine classification Fashion-MNIST + analyse visuelle.
        """
        x1, y1, x2, y2 = person_bbox
        person_roi = frame[y1:y2, x1:x2]

        if person_roi.size == 0:
            return []

        violations = []
        height, width = person_roi.shape[:2]

        # Diviser la personne en zones
        head_region = person_roi[0:int(height*0.2), :]
        upper_region = person_roi[int(height*0.2):int(height*0.5), :]
        lower_region = person_roi[int(height*0.5):int(height*0.85), :]
        feet_region = person_roi[int(height*0.85):, :]

        # === CLASSIFICATION AVEC FASHION-MNIST ===
        if self.use_fashion_model:
            # Classifier la zone des pieds (sandales/tongs)
            if feet_region.size > 0 and feet_region.shape[0] > 10 and feet_region.shape[1] > 10:
                feet_result = self._classify_clothing_region(feet_region)
                if feet_result:
                    label, conf = feet_result
                    if label == "sandal" and conf > 0.6:
                        violations.append(("Sandales/Tongs", conf))

            # Classifier la zone du haut
            if upper_region.size > 0 and upper_region.shape[0] > 10 and upper_region.shape[1] > 10:
                upper_result = self._classify_clothing_region(upper_region)
                if upper_result:
                    label, conf = upper_result
                    # Verifier si c'est un top court (crop top)
                    if label == "top" and conf > 0.5:
                        # Analyse supplementaire pour crop top
                        crop_violations = self._detect_crop_top(upper_region)
                        violations.extend(crop_violations)

            # Classifier la zone du bas
            if lower_region.size > 0 and lower_region.shape[0] > 10 and lower_region.shape[1] > 10:
                lower_result = self._classify_clothing_region(lower_region)
                if lower_result:
                    label, conf = lower_result
                    # Les robes courtes sont interdites
                    if label == "dress" and conf > 0.5:
                        # Verifier si la robe est courte
                        short_violations = self._detect_short_dress(lower_region)
                        violations.extend(short_violations)

        # === ANALYSE VISUELLE (complement) ===

        # Detection des couvre-chefs
        head_violations = self._detect_headwear(head_region)
        violations.extend(head_violations)

        # Detection crop top par analyse visuelle
        if not self.use_fashion_model:
            upper_violations = self._detect_upper_clothing(upper_region)
            violations.extend(upper_violations)

        # Detection short/mini-jupe
        lower_violations = self._detect_lower_clothing(lower_region)
        violations.extend(lower_violations)

        # Detection jean troue
        jean_violations = self._detect_ripped_jeans(lower_region)
        violations.extend(jean_violations)

        # Detection tongs par analyse visuelle (si pas de modele fashion)
        if not self.use_fashion_model:
            feet_violations = self._detect_sandals(feet_region)
            violations.extend(feet_violations)

        return violations

    def _detect_headwear(self, head_region):
        """Detecte les couvre-chefs par analyse visuelle"""
        violations = []

        if head_region.size == 0 or head_region.shape[0] < 10 or head_region.shape[1] < 10:
            return violations

        try:
            hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
            height, width = head_region.shape[:2]
            top_region = hsv[0:int(height*0.6), :]

            if top_region.size == 0:
                return violations

            h_std = np.std(top_region[:, :, 0])
            s_mean = np.mean(top_region[:, :, 1])

            if h_std < 30 and s_mean > 30:
                v_mean = np.mean(top_region[:, :, 2])
                if v_mean > 60:
                    violations.append(("Couvre-chef", 0.55))

        except Exception:
            pass

        return violations

    def _detect_crop_top(self, upper_region):
        """Detecte les crop tops (ventre visible)"""
        violations = []

        if upper_region.size == 0:
            return violations

        height = upper_region.shape[0]
        lower_torso = upper_region[int(height*0.6):, :]

        if lower_torso.size > 0:
            hsv = cv2.cvtColor(lower_torso, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_ratio = np.sum(skin_mask > 0) / skin_mask.size

            if skin_ratio > 0.4:
                violations.append(("Crop top", 0.75))

        return violations

    def _detect_upper_clothing(self, upper_region):
        """Detecte les vetements du haut non conformes"""
        violations = []

        if upper_region.size == 0:
            return violations

        height = upper_region.shape[0]
        lower_torso = upper_region[int(height*0.6):, :]

        if lower_torso.size > 0:
            hsv = cv2.cvtColor(lower_torso, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_ratio = np.sum(skin_mask > 0) / skin_mask.size

            if skin_ratio > 0.4:
                violations.append(("Crop top", 0.6))

        # Detection tenue de sport
        hsv_full = cv2.cvtColor(upper_region, cv2.COLOR_BGR2HSV)
        bright_colors = cv2.inRange(hsv_full, np.array([0, 100, 100]), np.array([180, 255, 255]))
        bright_ratio = np.sum(bright_colors > 0) / bright_colors.size
        std_dev = np.std(upper_region)

        if bright_ratio > 0.6 and std_dev < 40:
            violations.append(("Tenue de sport", 0.5))

        return violations

    def _detect_lower_clothing(self, lower_region):
        """Detecte shorts et mini-jupes"""
        violations = []

        if lower_region.size == 0:
            return violations

        height = lower_region.shape[0]
        hsv = cv2.cvtColor(lower_region, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Detection short (jambes nues)
        leg_region = skin_mask[int(height*0.5):, :]
        if leg_region.size > 0:
            skin_ratio = np.sum(leg_region > 0) / leg_region.size
            if skin_ratio > 0.5:
                violations.append(("Short/Bermuda", 0.75))

        # Detection mini-jupe
        thigh_region = skin_mask[int(height*0.2):int(height*0.6), :]
        if thigh_region.size > 0:
            thigh_skin_ratio = np.sum(thigh_region > 0) / thigh_region.size
            if thigh_skin_ratio > 0.6:
                violations.append(("Mini-jupe", 0.70))

        return violations

    def _detect_short_dress(self, lower_region):
        """Detecte les robes courtes"""
        violations = []

        if lower_region.size == 0:
            return violations

        height = lower_region.shape[0]
        hsv = cv2.cvtColor(lower_region, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Si beaucoup de peau visible dans la partie basse = robe courte
        lower_half = skin_mask[int(height*0.5):, :]
        if lower_half.size > 0:
            skin_ratio = np.sum(lower_half > 0) / lower_half.size
            if skin_ratio > 0.4:
                violations.append(("Robe courte", 0.70))

        return violations

    def _detect_ripped_jeans(self, lower_region):
        """Detecte les jeans troues"""
        violations = []

        if lower_region.size == 0:
            return violations

        hsv = cv2.cvtColor(lower_region, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Zones bleues (jean)
        blue_mask = cv2.inRange(lower_region, np.array([100, 50, 0]), np.array([255, 150, 100]))
        blue_ratio = np.sum(blue_mask > 0) / blue_mask.size

        if blue_ratio > 0.3:  # C'est probablement un jean
            jean_with_skin = cv2.bitwise_and(skin_mask, blue_mask)
            holes_ratio = np.sum(jean_with_skin > 0) / (np.sum(blue_mask > 0) + 1)
            if holes_ratio > 0.1:
                violations.append(("Jean troue", 0.60))

        return violations

    def _detect_sandals(self, feet_region):
        """Detecte les sandales/tongs par analyse visuelle"""
        violations = []

        if feet_region.size == 0 or feet_region.shape[0] < 5 or feet_region.shape[1] < 5:
            return violations

        try:
            hsv = cv2.cvtColor(feet_region, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_ratio = np.sum(skin_mask > 0) / skin_mask.size

            # Beaucoup de peau visible = pieds nus ou sandales
            if skin_ratio > 0.3:
                violations.append(("Sandales/Tongs", 0.55))

        except Exception:
            pass

        return violations

    def _draw_detections(self, frame, detections):
        """Dessine les detections sur la frame"""
        annotated_frame = frame.copy()
        all_violations = []

        for detection in detections:
            bbox = detection["bbox"]
            x1, y1, x2, y2 = bbox
            violations_haute = detection.get("violations_haute", [])

            if violations_haute:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                y_offset = y1 - 10
                for violation, conf in violations_haute:
                    message = f"INTERDIT: {violation} ({conf*100:.0f}%)"

                    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(annotated_frame,
                                (x1, y_offset - text_size[1] - 5),
                                (x1 + text_size[0], y_offset + 5),
                                (0, 0, 255), -1)

                    cv2.putText(annotated_frame, message,
                              (x1, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset -= 25

                    all_violations.append({
                        "type": violation,
                        "confidence": conf,
                        "bbox": bbox,
                        "timestamp": datetime.now(),
                        "high_confidence": True
                    })
            else:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, "CONFORME",
                          (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Timestamp et statut
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated_frame, f"ENSITECH - {timestamp}",
                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if all_violations:
            status = f"ALERTES: {len(all_violations)} violation(s)"
            cv2.putText(annotated_frame, status,
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(annotated_frame, "Statut: Aucune violation",
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Indicateur du mode actif
        if self.use_custom_model:
            cv2.putText(annotated_frame, "Mode: ENSITECH Custom YOLO",
                      (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        elif self.use_fashion_model:
            cv2.putText(annotated_frame, "Mode: Fashion-MNIST + YOLO",
                      (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(annotated_frame, "Mode: YOLO + Analyse visuelle",
                      (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return annotated_frame, all_violations

    def process_frame(self, frame):
        """Traite une frame et retourne les violations detectees"""
        self.frame_count += 1

        # Skip frames pour optimiser
        if self.frame_count % DETECTION_CONFIG["frame_skip"] != 0:
            annotated_frame, _ = self._draw_detections(frame, self.last_detections)
            return annotated_frame, []

        new_detections = []

        # === MODE 1: Modele personnalise ENSITECH ===
        if self.use_custom_model:
            custom_detections = self.detect_with_custom_model(frame)

            for det in custom_detections:
                # Seuil de confiance a 70%
                if det["confidence"] >= 0.70:
                    new_detections.append({
                        "bbox": det["bbox"],
                        "violations_haute": [(det["display_name"], det["confidence"])],
                        "violations_basse": []
                    })
                else:
                    new_detections.append({
                        "bbox": det["bbox"],
                        "violations_haute": [],
                        "violations_basse": [(det["display_name"], det["confidence"])]
                    })

        # === MODE 2: Detection standard (YOLO personnes + analyse) ===
        else:
            persons = self.detect_persons(frame)

            for person in persons:
                bbox = person["bbox"]
                violations = self.analyze_clothing(frame, bbox)

                # Seuil de confiance a 70%
                violations_haute = [(v, c) for v, c in violations if c >= 0.70]
                violations_basse = [(v, c) for v, c in violations if c < 0.70]

                new_detections.append({
                    "bbox": bbox,
                    "violations_haute": violations_haute,
                    "violations_basse": violations_basse
                })

        self.last_detections = new_detections
        return self._draw_detections(frame, new_detections)

    def save_alert_image(self, frame, violations):
        """Sauvegarde l'image de l'alerte"""
        if not DETECTION_CONFIG["save_alerts"]:
            return None

        alerts_folder = DETECTION_CONFIG["alerts_folder"]
        os.makedirs(alerts_folder, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        violation_types = "_".join([v["type"].replace(" ", "_").replace("/", "-") for v in violations[:3]])
        filename = f"alert_{timestamp}_{violation_types}.jpg"
        filepath = os.path.join(alerts_folder, filename)

        cv2.imwrite(filepath, frame)
        return filepath


def main():
    """Test du detecteur en mode standalone"""
    detector = DressCodeDetector()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la camera")
        return

    print("Demarrage de la detection...")
    print("Appuyez sur 'q' pour quitter")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        annotated_frame, violations = detector.process_frame(frame)
        cv2.imshow("ENSITECH - Controle Code Vestimentaire", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
