"""Test rapide du modele dresscode_yolo.pt sur quelques images"""

import os
from pathlib import Path
from ultralytics import YOLO


def main():
    BASE_DIR = Path(__file__).parent
    MODEL_PATH = BASE_DIR / "dresscode_yolo.pt"
    TEST_DIR = BASE_DIR / "dataset_couvre_chef" / "images" / "test"

    if not MODEL_PATH.exists():
        print("ERREUR: dresscode_yolo.pt introuvable !")
        return

    model = YOLO(str(MODEL_PATH))

    # Evaluation sur le jeu de test
    yaml_path = BASE_DIR / "dataset_couvre_chef" / "dataset.yaml"
    if yaml_path.exists():
        print("Evaluation sur le jeu de test...\n")
        metrics = model.val(data=str(yaml_path), split="test", workers=0, verbose=False)
        print(f"  mAP50      : {metrics.box.map50:.3f}")
        print(f"  mAP50-95   : {metrics.box.map:.3f}")
        print(f"  Precision  : {metrics.box.mp:.3f}")
        print(f"  Recall     : {metrics.box.mr:.3f}")

    # Test visuel sur 5 images
    print("\nTest sur 5 images:")
    test_images = list(TEST_DIR.glob("*.jpg"))[:5]
    for img_path in test_images:
        results = model(str(img_path), conf=0.3, verbose=False)
        for r in results:
            n = len(r.boxes)
            if n > 0:
                confs = [f"{float(b.conf[0]):.0%}" for b in r.boxes]
                print(f"  {img_path.name} -> {n} couvre_chef detecte(s) (conf: {', '.join(confs)})")
            else:
                print(f"  {img_path.name} -> rien detecte")

    # Sauvegarder les predictions annotees
    print("\nSauvegarde des images annotees dans test_results/...")
    import cv2
    os.makedirs("test_results", exist_ok=True)
    for img_path in test_images:
        results = model(str(img_path), conf=0.3, verbose=False)
        for r in results:
            annotated = r.plot()
            cv2.imwrite(f"test_results/{img_path.name}", annotated)

    print("Done ! Regarde le dossier test_results/")


if __name__ == "__main__":
    main()
