"""
Test rapide du modele dresscode_yolo.pt sur des images de validation.
Usage:
  python test_model.py                          # teste sur 5 images du dataset val
  python test_model.py --image photo.jpg        # teste sur une image specifique
  python test_model.py --webcam                 # teste avec la webcam
"""

import os
import sys
import cv2
from ultralytics import YOLO

MODEL_PATH = "dresscode_yolo.pt"
CLASSES = {0: "couvre_chef"}
COLORS = {0: (255, 165, 0)}  # orange


def test_on_image(model, image_path):
    """Teste le modele sur une image et affiche le resultat."""
    print(f"\n--- {os.path.basename(image_path)} ---")
    results = model(image_path, conf=0.8, verbose=False)

    img = cv2.imread(image_path)
    detections = 0

    for r in results:
        for box in r.boxes:
            detections += 1
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = CLASSES.get(cls_id, f"classe_{cls_id}")
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = COLORS.get(cls_id, (0, 255, 0))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.0%}"
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            print(f"  {cls_name}: {conf:.1%} [{x1},{y1},{x2},{y2}]")

    if detections == 0:
        print("  Aucune detection")

    # Sauvegarder le resultat
    out_path = f"test_result_{os.path.basename(image_path)}"
    cv2.imwrite(out_path, img)
    print(f"  Resultat sauvegarde: {out_path}")

    # Afficher
    cv2.imshow(f"TenueCheck - {os.path.basename(image_path)}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_webcam(model):
    """Teste le modele en temps reel avec la webcam."""
    print("Ouverture de la webcam... (appuyez sur 'q' pour quitter)")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erreur: impossible d'ouvrir la webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.8, verbose=False)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = CLASSES.get(cls_id, f"classe_{cls_id}")
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = COLORS.get(cls_id, (0, 255, 0))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name} {conf:.0%}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("TenueCheck - Webcam (q=quitter)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Erreur: {MODEL_PATH} non trouve!")
        print("Placez le fichier dresscode_yolo.pt a la racine du projet.")
        sys.exit(1)

    print(f"Chargement du modele: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"Classes: {CLASSES}")

    # Mode webcam
    if "--webcam" in sys.argv:
        test_webcam(model)

    # Mode image specifique
    elif "--image" in sys.argv:
        idx = sys.argv.index("--image")
        if idx + 1 < len(sys.argv):
            test_on_image(model, sys.argv[idx + 1])
        else:
            print("Erreur: specifiez le chemin de l'image")

    # Mode par defaut: tester sur les images de validation
    else:
        val_dir = "dataset_tenuecheck/images/val"
        if os.path.exists(val_dir):
            images = [f for f in os.listdir(val_dir) if f.endswith((".jpg", ".png"))][:5]
            print(f"\nTest sur {len(images)} images de validation:\n")
            for img_file in images:
                test_on_image(model, os.path.join(val_dir, img_file))
        else:
            print(f"Dossier {val_dir} non trouve.")
            print("Utilisez: python test_model.py --image <chemin_image>")
            print("Ou:       python test_model.py --webcam")
