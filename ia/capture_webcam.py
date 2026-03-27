import cv2
import os
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime


def main():
    BASE_DIR = Path(__file__).parent
    MODEL_PATH = BASE_DIR / "dresscode_yolo.pt"
    SAVE_IMG_DIR = BASE_DIR / "dataset_couvre_chef" / "images" / "train"
    SAVE_LBL_DIR = BASE_DIR / "dataset_couvre_chef" / "labels" / "train"

    SAVE_IMG_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_LBL_DIR.mkdir(parents=True, exist_ok=True)

    # Charger le modele pour auto-annotation
    model = None
    if MODEL_PATH.exists():
        model = YOLO(str(MODEL_PATH))
        print("Modele charge pour auto-annotation")

    # Ouvrir la webcam (meme index que config.py)
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    count_positive = 0
    count_negative = 0

    print("\n" + "=" * 50)
    print("CAPTURE D'IMAGES WEBCAM")
    print("=" * 50)
    print("ESPACE = capturer AVEC couvre-chef")
    print("N      = capturer SANS couvre-chef (negative)")
    print("Q      = quitter")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        # Afficher les detections en temps reel
        if model:
            results = model(frame, conf=0.2, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display, f"couvre_chef {conf:.0%}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(display, f"Positives: {count_positive} | Negatives: {count_negative}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "ESPACE=capture avec chapeau | N=sans | Q=quitter",
                    (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Capture webcam - TenueCheck", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord(' '):
            # Capture POSITIVE (avec couvre-chef)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            img_name = f"webcam_pos_{timestamp}"

            # Sauvegarder l'image
            cv2.imwrite(str(SAVE_IMG_DIR / f"{img_name}.jpg"), frame)

            # Auto-annoter avec le modele OU annotation manuelle plein cadre
            h, w = frame.shape[:2]
            annotations = []

            if model:
                results = model(frame, conf=0.15, verbose=False)
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # Convertir en format YOLO normalise
                        xc = ((x1 + x2) / 2) / w
                        yc = ((y1 + y2) / 2) / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        annotations.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            if not annotations:
                # Si le modele n'a rien detecte, annoter la zone haute (tete)
                # Approximation: couvre-chef dans le tiers superieur central
                annotations.append("0 0.500000 0.150000 0.400000 0.200000")

            with open(SAVE_LBL_DIR / f"{img_name}.txt", "w") as f:
                f.write("\n".join(annotations))

            count_positive += 1
            print(f"  [+] Capture positive #{count_positive} ({len(annotations)} annotations)")

        elif key == ord('n'):
            # Capture NEGATIVE (sans couvre-chef) - fichier label vide
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            img_name = f"webcam_neg_{timestamp}"

            cv2.imwrite(str(SAVE_IMG_DIR / f"{img_name}.jpg"), frame)
            # Label vide = pas de couvre-chef (image negative)
            with open(SAVE_LBL_DIR / f"{img_name}.txt", "w") as f:
                f.write("")

            count_negative += 1
            print(f"  [-] Capture negative #{count_negative}")

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nTermine ! {count_positive} positives + {count_negative} negatives ajoutees au dataset")
    print("Relance l'entrainement pour ameliorer le modele.")


if __name__ == "__main__":
    main()
