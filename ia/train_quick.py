"""Entrainement rapide sur le dataset existant (sans re-telecharger)"""

import os
import shutil
import torch
from pathlib import Path
from ultralytics import YOLO


def main():
    BASE_DIR = Path(__file__).parent
    DATASET = BASE_DIR / "dataset_couvre_chef"
    yaml_path = DATASET / "dataset.yaml"

    if not yaml_path.exists():
        print("ERREUR: dataset_couvre_chef/dataset.yaml introuvable")
        print("Lance d'abord: python download_roboflow_and_train.py")
        return

    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Compter les images
    train_count = len(list((DATASET / "images" / "train").glob("*.*")))
    val_count = len(list((DATASET / "images" / "val").glob("*.*")))
    print(f"Dataset: {train_count} train / {val_count} val")

    # Reprendre depuis le meilleur modele existant
    model_path = BASE_DIR / "dresscode_yolo.pt"
    if model_path.exists():
        print(f"Reprise depuis dresscode_yolo.pt")
        model = YOLO(str(model_path))
    else:
        print("Pas de modele existant, demarrage depuis yolov8n.pt")
        model = YOLO("yolov8n.pt")

    model.train(
        data=str(yaml_path),
        epochs=30,
        imgsz=640,
        batch=32,
        name="tenuecheck_couvre_chef",
        patience=30,
        save=True,
        plots=True,
        device=DEVICE,
        workers=0,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        augment=True,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=20,
        translate=0.2,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.15,
    )

    # Copier le meilleur modele
    detect_dir = Path("runs/detect")
    results_dir = max(detect_dir.glob("tenuecheck_couvre_chef*"), key=os.path.getmtime)
    best_weights = results_dir / "weights" / "best.pt"
    output = BASE_DIR / "dresscode_yolo.pt"
    shutil.copy(best_weights, output)
    print(f"\nModele copie dans dresscode_yolo.pt")


if __name__ == "__main__":
    main()
