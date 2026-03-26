"""
TenueCheck - Telecharge des datasets Roboflow + Fashionpedia et entraine YOLOv8n
Focus: casquette, bonnet, chapeau, capuche -> classe unique "couvre_chef"
"""

import os
import gc
import shutil
import random
import glob
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO


# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).parent
DATASET_FINAL = BASE_DIR / "dataset_couvre_chef"
ROBOFLOW_DIR = BASE_DIR / "roboflow_downloads"

# Hyperparametres pour gros entrainement (~5h sur RTX 3070)
EPOCHS = 30
IMGSZ = 640
BATCH = 32
PATIENCE = 30


def download_roboflow_datasets():
    """Telecharge les datasets depuis Roboflow Universe (open source, pas besoin d'API key)."""
    from roboflow import Roboflow

    print("=" * 60)
    print("ETAPE 1a : Telechargement datasets Roboflow")
    print("=" * 60)

    ROBOFLOW_DIR.mkdir(exist_ok=True)

    # Liste des datasets a telecharger (format YOLO)
    datasets = [
        {
            "name": "Headwear Detection (ATM)",
            "workspace": "atm",
            "project": "headware-detection",
            "version": 1,
        },
        {
            "name": "Cap Dataset (ny)",
            "workspace": "ny",
            "project": "cap-ghxtg",
            "version": 1,
        },
        {
            "name": "Cap Dataset (PlayRoom)",
            "workspace": "playroom",
            "project": "cap-dataset-fnmsl",
            "version": 1,
        },
    ]

    downloaded = []
    for ds_info in datasets:
        print(f"\nTelechargement: {ds_info['name']}...")
        try:
            rf = Roboflow(api_key="G0Yo4MeizcErw1YczA8b")
            project = rf.workspace(ds_info["workspace"]).project(ds_info["project"])
            version = project.version(ds_info["version"])
            dataset = version.download("yolov8", location=str(ROBOFLOW_DIR / ds_info["project"]))
            downloaded.append(ds_info)
            print(f"  OK -> {ROBOFLOW_DIR / ds_info['project']}")
        except Exception as e:
            print(f"  ERREUR: {e}")
            print(f"  On continue sans ce dataset...")

    return downloaded


def download_fashionpedia():
    """Telecharge les couvre_chef depuis Fashionpedia (train + val)."""
    from datasets import load_dataset

    print("\n" + "=" * 60)
    print("ETAPE 1b : Telechargement Fashionpedia")
    print("=" * 60)

    FASHIONPEDIA_TO_TARGET = {
        14: 0,  # hat -> couvre_chef
        15: 0,  # headband, head covering -> couvre_chef
        27: 0,  # hood -> couvre_chef
    }
    target_categories = set(FASHIONPEDIA_TO_TARGET.keys())
    MAX_IMAGES = 5000

    fp_dir = ROBOFLOW_DIR / "fashionpedia"
    img_dir = fp_dir / "images"
    lbl_dir = fp_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    total_seen = 0

    for split_name in ["train", "val"]:
        if len(saved) >= MAX_IMAGES:
            break

        print(f"\nFashionpedia split '{split_name}' (streaming)...")
        ds = load_dataset("detection-datasets/fashionpedia", split=split_name, streaming=True)

        for sample in tqdm(ds, desc=f"Collecte {split_name}", total=MAX_IMAGES - len(saved)):
            total_seen += 1
            categories = sample["objects"]["category"]

            if not any(cat in target_categories for cat in categories):
                continue

            image = sample["image"]
            objects = sample["objects"]
            img_w, img_h = image.size

            yolo_annotations = []
            for cat, bbox in zip(objects["category"], objects["bbox"]):
                if cat not in target_categories:
                    continue
                x_min, y_min, w, h = bbox
                x_center = max(0, min(1, (x_min + w / 2) / img_w))
                y_center = max(0, min(1, (y_min + h / 2) / img_h))
                w_norm = max(0, min(1, w / img_w))
                h_norm = max(0, min(1, h / img_h))
                if w_norm > 0.01 and h_norm > 0.01:
                    yolo_annotations.append(
                        f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                    )

            if yolo_annotations:
                i = len(saved)
                img_filename = f"fp_{i:05d}.jpg"
                image.convert("RGB").save(str(img_dir / img_filename), "JPEG", quality=95)
                with open(lbl_dir / f"fp_{i:05d}.txt", "w") as f:
                    f.write("\n".join(yolo_annotations))
                saved.append(img_filename)

                if len(saved) % 500 == 0:
                    print(f"  -> {len(saved)} images sauvegardees...")

            if len(saved) >= MAX_IMAGES:
                break

        del ds
        gc.collect()

    print(f"\n{len(saved)} images Fashionpedia sauvegardees")
    return len(saved)


def merge_datasets():
    """Fusionne tous les datasets telecharges en un seul dataset YOLO."""
    print("\n" + "=" * 60)
    print("ETAPE 2 : Fusion de tous les datasets")
    print("=" * 60)

    # Nettoyer le dossier final
    for split in ["train", "val", "test"]:
        for subdir in ["images", "labels"]:
            d = DATASET_FINAL / subdir / split
            d.mkdir(parents=True, exist_ok=True)
            for f in d.iterdir():
                if f.is_file():
                    f.unlink()

    all_images = []
    all_labels = []

    # 1. Ajouter Fashionpedia
    fp_img_dir = ROBOFLOW_DIR / "fashionpedia" / "images"
    fp_lbl_dir = ROBOFLOW_DIR / "fashionpedia" / "labels"
    if fp_img_dir.exists():
        for img in sorted(fp_img_dir.glob("*.jpg")):
            lbl = fp_lbl_dir / (img.stem + ".txt")
            if lbl.exists():
                all_images.append(img)
                all_labels.append(lbl)
        print(f"  Fashionpedia : {len(all_images)} images")

    # 2. Ajouter les datasets Roboflow
    for project_dir in ROBOFLOW_DIR.iterdir():
        if project_dir.name == "fashionpedia" or not project_dir.is_dir():
            continue

        count_before = len(all_images)

        # Roboflow datasets ont la structure: train/images, valid/images, test/images
        for split in ["train", "valid", "test"]:
            rb_img_dir = project_dir / split / "images"
            rb_lbl_dir = project_dir / split / "labels"

            if not rb_img_dir.exists():
                continue

            for img in rb_img_dir.glob("*"):
                if img.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                    continue
                lbl = rb_lbl_dir / (img.stem + ".txt")
                if lbl.exists():
                    # Remap toutes les classes vers 0 (couvre_chef)
                    with open(lbl) as f:
                        lines = f.readlines()
                    remapped = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            # Forcer class_id = 0
                            remapped.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
                    if remapped:
                        # Sauvegarder le label remapped dans un fichier temporaire
                        tmp_lbl = lbl.parent / (lbl.stem + "_remapped.txt")
                        with open(tmp_lbl, "w") as f:
                            f.write("\n".join(remapped))
                        all_images.append(img)
                        all_labels.append(tmp_lbl)

        count_after = len(all_images)
        print(f"  {project_dir.name} : {count_after - count_before} images")

    print(f"\n  TOTAL : {len(all_images)} images")

    # Shuffle et split
    combined = list(zip(all_images, all_labels))
    random.shuffle(combined)

    n = len(combined)
    n_train = int(n * 0.80)
    n_val = int(n * 0.15)

    splits = {
        "train": combined[:n_train],
        "val": combined[n_train:n_train + n_val],
        "test": combined[n_train + n_val:],
    }

    stats = {}
    for split, pairs in splits.items():
        dst_img = DATASET_FINAL / "images" / split
        dst_lbl = DATASET_FINAL / "labels" / split

        for i, (img_path, lbl_path) in enumerate(pairs):
            # Copier avec un nom unique pour eviter les collisions
            ext = img_path.suffix
            new_name = f"img_{split}_{i:05d}"
            shutil.copy2(str(img_path), str(dst_img / f"{new_name}{ext}"))
            shutil.copy2(str(lbl_path), str(dst_lbl / f"{new_name}.txt"))

        stats[split] = len(pairs)
        print(f"  {split:5s} : {len(pairs)} images")

    return stats


def train_model(stats):
    """Entraine YOLOv8n sur le dataset fusionne."""
    print("\n" + "=" * 60)
    print("ETAPE 3 : Entrainement YOLOv8n")
    print("=" * 60)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU : {gpu_name} ({vram:.1f} GB VRAM)")
        DEVICE = 0
    else:
        print("Pas de GPU, utilisation du CPU")
        DEVICE = "cpu"

    # YAML
    yaml_path = DATASET_FINAL / "dataset.yaml"
    yaml_content = f"""# TenueCheck - couvre_chef (casquette, bonnet, chapeau, capuche)
path: {DATASET_FINAL.resolve()}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: couvre_chef
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    total = sum(stats.values())
    print(f"\nDataset : {total} images ({stats.get('train',0)} train / {stats.get('val',0)} val / {stats.get('test',0)} test)")
    print(f"Epochs: {EPOCHS} | Batch: {BATCH} | Patience: {PATIENCE}")
    print()

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        name="tenuecheck_couvre_chef",
        patience=PATIENCE,
        save=True,
        plots=True,
        device=DEVICE,
        workers=0,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,
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

    # Evaluation
    print("\n" + "=" * 60)
    print("EVALUATION SUR LE JEU DE TEST")
    print("=" * 60)

    detect_dir = Path("runs/detect")
    results_dir = max(detect_dir.glob("tenuecheck_couvre_chef*"), key=os.path.getmtime)
    best_weights = results_dir / "weights" / "best.pt"
    best_model = YOLO(str(best_weights))

    metrics = best_model.val(data=str(yaml_path), split="test", workers=0, verbose=False)

    print(f"  mAP50      : {metrics.box.map50:.3f}")
    print(f"  mAP50-95   : {metrics.box.map:.3f}")
    print(f"  Precision  : {metrics.box.mp:.3f}")
    print(f"  Recall     : {metrics.box.mr:.3f}")

    # Export
    output_model = BASE_DIR / "dresscode_yolo.pt"
    shutil.copy(best_weights, output_model)
    size_mb = best_weights.stat().st_size / 1024 / 1024

    print(f"\n{'=' * 60}")
    print("ENTRAINEMENT TERMINE")
    print(f"{'=' * 60}")
    print(f"  mAP50      : {metrics.box.map50:.3f}")
    print(f"  Precision  : {metrics.box.mp:.3f}")
    print(f"  Recall     : {metrics.box.mr:.3f}")
    print(f"  Modele     : dresscode_yolo.pt ({size_mb:.1f} MB)")


if __name__ == "__main__":
    download_roboflow_datasets()
    download_fashionpedia()
    stats = merge_datasets()
    train_model(stats)
