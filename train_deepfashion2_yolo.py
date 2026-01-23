"""
Transfer Learning: YOLOv8 + DeepFashion2
Entrainement d'un modele de detection de vetements interdits ENSITECH

DeepFashion2 categories:
1: short_sleeve_top    -> crop_top (si court)
2: long_sleeve_top     -> (ignore)
3: short_sleeve_outwear -> (ignore)
4: long_sleeve_outwear -> (ignore)
5: vest                -> crop_top
6: sling               -> crop_top
7: shorts              -> short
8: trousers            -> (ignore, sauf si troue)
9: skirt               -> mini_skirt (si courte)
10: short_sleeve_dress -> (ignore, sauf si courte)
11: long_sleeve_dress  -> (ignore)
12: vest_dress         -> (ignore)
13: sling_dress        -> (ignore)

Classes ENSITECH cibles:
0: short
1: mini_skirt
2: crop_top
3: sportswear (detection par couleurs vives + texture)
"""

import os
import json
import shutil
import random
from pathlib import Path
from PIL import Image
import yaml
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# Chemins
DEEPFASHION2_DIR = "deepfashion2"
OUTPUT_DIR = "dataset_yolo_dresscode"
MODEL_OUTPUT = "dresscode_yolo.pt"

# Classes cibles (simplifiees pour DeepFashion2)
# On garde seulement les classes disponibles dans DeepFashion2
TARGET_CLASSES = {
    0: "short",
    1: "mini_skirt",
    2: "crop_top",
}

# Mapping DeepFashion2 -> Classes ENSITECH
# DeepFashion2 utilise des indices 1-13
CATEGORY_MAPPING = {
    7: 0,   # shorts -> short
    9: 1,   # skirt -> mini_skirt (filtrer par taille)
    5: 2,   # vest -> crop_top
    6: 2,   # sling -> crop_top
    1: 2,   # short_sleeve_top -> crop_top (filtrer par taille)
}

# Parametres d'entrainement
TRAIN_CONFIG = {
    "epochs": 100,
    "batch_size": 16,
    "img_size": 640,
    "patience": 20,
    "model_base": "yolov8n.pt",  # nano = rapide, yolov8s.pt = plus precis
}

# Filtres pour selectionner les vetements "interdits"
# Ratio hauteur/largeur et position pour determiner si c'est court
FILTERS = {
    "mini_skirt_max_height_ratio": 0.25,  # < 25% de la hauteur image = mini
    "crop_top_min_y_ratio": 0.15,         # Commence dans les 15% du haut
    "crop_top_max_height_ratio": 0.30,    # < 30% de hauteur = crop
}


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def setup_directories():
    """Cree la structure de dossiers YOLO"""
    dirs = [
        f"{OUTPUT_DIR}/images/train",
        f"{OUTPUT_DIR}/images/val",
        f"{OUTPUT_DIR}/labels/train",
        f"{OUTPUT_DIR}/labels/val",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("[OK] Dossiers crees")


def create_yaml_config():
    """Cree le fichier dataset.yaml pour YOLO"""
    config = {
        'path': os.path.abspath(OUTPUT_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'names': TARGET_CLASSES
    }

    yaml_path = f"{OUTPUT_DIR}/dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"[OK] Configuration: {yaml_path}")
    return yaml_path


def bbox_to_yolo(bbox, img_w, img_h):
    """
    Convertit [x1, y1, x2, y2] vers format YOLO [x_center, y_center, width, height]
    Normalise entre 0 et 1
    """
    x1, y1, x2, y2 = bbox

    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h

    # Clamp
    x_center = max(0.001, min(0.999, x_center))
    y_center = max(0.001, min(0.999, y_center))
    width = max(0.001, min(0.999, width))
    height = max(0.001, min(0.999, height))

    return x_center, y_center, width, height


def is_short_garment(bbox, img_h, category_id):
    """
    Determine si un vetement est "court" (donc interdit)
    """
    x1, y1, x2, y2 = bbox
    garment_height = (y2 - y1) / img_h
    garment_y_start = y1 / img_h

    # Pour les jupes: mini si hauteur < 25%
    if category_id == 9:  # skirt
        return garment_height < FILTERS["mini_skirt_max_height_ratio"]

    # Pour les hauts: crop si dans la partie haute et court
    if category_id in [1, 5, 6]:  # tops
        is_high = garment_y_start < FILTERS["crop_top_min_y_ratio"] + 0.3
        is_short = garment_height < FILTERS["crop_top_max_height_ratio"]
        return is_high and is_short

    # Pour les shorts: toujours inclus
    if category_id == 7:
        return True

    return False


# ============================================================================
# CONVERSION DU DATASET
# ============================================================================

def convert_deepfashion2():
    """
    Convertit DeepFashion2 vers format YOLO
    """
    print("\n" + "="*60)
    print("CONVERSION DEEPFASHION2 -> YOLO")
    print("="*60)

    if not os.path.exists(DEEPFASHION2_DIR):
        print(f"\n[ERREUR] DeepFashion2 non trouve: {DEEPFASHION2_DIR}")
        print_download_instructions()
        return None

    stats = {cls_name: {"train": 0, "val": 0} for cls_name in TARGET_CLASSES.values()}
    total_images = {"train": 0, "val": 0}

    # Traiter train et validation
    splits = [
        ("train", "train"),
        ("validation", "val"),
    ]

    for src_split, dst_split in splits:
        annos_dir = os.path.join(DEEPFASHION2_DIR, src_split, "annos")
        images_dir = os.path.join(DEEPFASHION2_DIR, src_split, "image")

        if not os.path.exists(annos_dir):
            print(f"[!] Dossier non trouve: {annos_dir}")
            continue

        print(f"\nTraitement {src_split}...")

        anno_files = sorted([f for f in os.listdir(annos_dir) if f.endswith('.json')])

        for anno_file in tqdm(anno_files, desc=f"  {src_split}"):
            anno_path = os.path.join(annos_dir, anno_file)

            # Nom de l'image (peut etre .jpg ou .png)
            base_name = os.path.splitext(anno_file)[0]
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                test_path = os.path.join(images_dir, base_name + ext)
                if os.path.exists(test_path):
                    image_path = test_path
                    break

            if image_path is None:
                continue

            # Lire l'annotation
            try:
                with open(anno_path, 'r') as f:
                    data = json.load(f)
            except Exception:
                continue

            # Dimensions de l'image
            try:
                with Image.open(image_path) as img:
                    img_w, img_h = img.size
            except Exception:
                continue

            # Convertir les annotations
            yolo_lines = []

            for key, value in data.items():
                if not key.startswith('item'):
                    continue

                category_id = value.get('category_id', 0)
                bbox = value.get('bounding_box', [])

                if len(bbox) != 4 or category_id not in CATEGORY_MAPPING:
                    continue

                # Verifier si c'est un vetement "court" (interdit)
                if not is_short_garment(bbox, img_h, category_id):
                    continue

                # Mapper vers notre classe
                target_class = CATEGORY_MAPPING[category_id]

                # Convertir la bbox
                x_c, y_c, w, h = bbox_to_yolo(bbox, img_w, img_h)
                yolo_lines.append(f"{target_class} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

                # Stats
                stats[TARGET_CLASSES[target_class]][dst_split] += 1

            # Sauvegarder si on a des annotations
            if yolo_lines:
                # Copier l'image
                ext = os.path.splitext(image_path)[1]
                new_name = f"{base_name}{ext}"
                dst_img = os.path.join(OUTPUT_DIR, "images", dst_split, new_name)
                shutil.copy(image_path, dst_img)

                # Sauvegarder les labels
                label_name = f"{base_name}.txt"
                dst_label = os.path.join(OUTPUT_DIR, "labels", dst_split, label_name)
                with open(dst_label, 'w') as f:
                    f.write('\n'.join(yolo_lines))

                total_images[dst_split] += 1

    # Afficher les statistiques
    print("\n" + "-"*50)
    print("STATISTIQUES DU DATASET")
    print("-"*50)
    print(f"{'Classe':<15} {'Train':>8} {'Val':>8} {'Total':>8}")
    print("-"*50)

    for cls_name in TARGET_CLASSES.values():
        train = stats[cls_name]["train"]
        val = stats[cls_name]["val"]
        print(f"{cls_name:<15} {train:>8} {val:>8} {train+val:>8}")

    print("-"*50)
    print(f"{'IMAGES':<15} {total_images['train']:>8} {total_images['val']:>8} {sum(total_images.values()):>8}")

    return stats


# ============================================================================
# ENTRAINEMENT
# ============================================================================

def train_model(yaml_path):
    """
    Entraine YOLOv8 avec transfer learning sur le dataset
    """
    print("\n" + "="*60)
    print("ENTRAINEMENT YOLOV8 - TRANSFER LEARNING")
    print("="*60)

    from ultralytics import YOLO
    import torch

    # Verifier GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Device: {device}")

    if device == "cpu":
        print("[INFO] Pas de GPU detecte. L'entrainement sera plus lent.")
        print("[INFO] Pour utiliser un GPU: installez CUDA + pytorch-cuda")

    # Charger le modele pre-entraine
    print(f"\n[INFO] Chargement du modele: {TRAIN_CONFIG['model_base']}")
    model = YOLO(TRAIN_CONFIG["model_base"])

    # Configuration de l'entrainement
    print(f"\n[INFO] Configuration:")
    print(f"  - Epochs: {TRAIN_CONFIG['epochs']}")
    print(f"  - Batch size: {TRAIN_CONFIG['batch_size']}")
    print(f"  - Image size: {TRAIN_CONFIG['img_size']}")
    print(f"  - Early stopping: {TRAIN_CONFIG['patience']} epochs")

    # Lancer l'entrainement
    print("\n[INFO] Demarrage de l'entrainement...")

    try:
        results = model.train(
            data=yaml_path,
            epochs=TRAIN_CONFIG["epochs"],
            batch=TRAIN_CONFIG["batch_size"],
            imgsz=TRAIN_CONFIG["img_size"],
            patience=TRAIN_CONFIG["patience"],
            project="runs/dresscode",
            name="train",
            exist_ok=True,
            pretrained=True,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            warmup_epochs=3,
            augment=True,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            verbose=True,
        )

        # Copier le meilleur modele
        best_path = Path("runs/dresscode/train/weights/best.pt")
        if best_path.exists():
            shutil.copy(best_path, MODEL_OUTPUT)
            print(f"\n[OK] Modele sauvegarde: {MODEL_OUTPUT}")

        return results

    except Exception as e:
        print(f"\n[ERREUR] Entrainement echoue: {e}")
        raise


def evaluate_model():
    """Evalue le modele entraine"""
    print("\n" + "="*60)
    print("EVALUATION DU MODELE")
    print("="*60)

    if not os.path.exists(MODEL_OUTPUT):
        print(f"[ERREUR] Modele non trouve: {MODEL_OUTPUT}")
        return

    from ultralytics import YOLO

    model = YOLO(MODEL_OUTPUT)
    yaml_path = f"{OUTPUT_DIR}/dataset.yaml"

    results = model.val(data=yaml_path)

    print(f"\n[INFO] Resultats:")
    print(f"  - mAP50: {results.box.map50:.4f}")
    print(f"  - mAP50-95: {results.box.map:.4f}")

    # Resultats par classe
    print(f"\n[INFO] Par classe:")
    for i, cls_name in TARGET_CLASSES.items():
        if i < len(results.box.ap50):
            print(f"  - {cls_name}: AP50={results.box.ap50[i]:.4f}")


def test_on_image(image_path):
    """Teste le modele sur une image"""
    import cv2
    from ultralytics import YOLO

    if not os.path.exists(MODEL_OUTPUT):
        print(f"[ERREUR] Modele non trouve: {MODEL_OUTPUT}")
        return

    model = YOLO(MODEL_OUTPUT)
    results = model(image_path)

    print(f"\nDetections sur {image_path}:")
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = TARGET_CLASSES.get(cls_id, f"class_{cls_id}")
            print(f"  - {cls_name}: {conf:.2%}")

    # Sauvegarder l'image annotee
    annotated = results[0].plot()
    output_path = "test_result.jpg"
    cv2.imwrite(output_path, annotated)
    print(f"\n[OK] Image annotee: {output_path}")


# ============================================================================
# INSTRUCTIONS
# ============================================================================

def print_download_instructions():
    """Instructions pour telecharger DeepFashion2"""
    print("""
================================================================================
INSTRUCTIONS POUR OBTENIR DEEPFASHION2
================================================================================

DeepFashion2 est un dataset academique qui necessite une demande d'acces.

ETAPES:

1. Allez sur: https://github.com/switchablenorms/DeepFashion2

2. Remplissez le formulaire de demande d'acces:
   https://docs.google.com/forms/d/e/1FAIpQLSeIoGaFfCQILrtIZPykkr8q_h9qQ5BoTYbjvf95aXbid0v2Bw/viewform

3. Attendez l'approbation (generalement quelques jours)

4. Telechargez les fichiers:
   - train.zip
   - validation.zip

5. Extrayez dans le dossier 'deepfashion2/':

   deepfashion2/
   ├── train/
   │   ├── annos/
   │   │   ├── 000001.json
   │   │   └── ...
   │   └── image/
   │       ├── 000001.jpg
   │       └── ...
   └── validation/
       ├── annos/
       └── image/

6. Relancez ce script

================================================================================
ALTERNATIVE RAPIDE (subset)
================================================================================

Vous pouvez aussi trouver des subsets sur Kaggle:
- https://www.kaggle.com/search?q=deepfashion2

Ou utiliser un petit dataset pour tester d'abord.
""")


def print_usage():
    """Affiche l'aide"""
    print("""
================================================================================
TRANSFER LEARNING: YOLOV8 + DEEPFASHION2
================================================================================

UTILISATION:

  python train_deepfashion2_yolo.py [commande]

COMMANDES:

  convert   - Convertit DeepFashion2 vers format YOLO
  train     - Convertit + entraine le modele
  eval      - Evalue le modele entraine
  test      - Teste sur une image
  full      - Pipeline complet (convert + train + eval)

EXEMPLES:

  # Pipeline complet
  python train_deepfashion2_yolo.py full

  # Ou etape par etape
  python train_deepfashion2_yolo.py convert
  python train_deepfashion2_yolo.py train
  python train_deepfashion2_yolo.py eval

  # Tester sur une image
  python train_deepfashion2_yolo.py test mon_image.jpg

PREREQUIS:

  1. Telechargez DeepFashion2:
     https://github.com/switchablenorms/DeepFashion2

  2. Extrayez dans 'deepfashion2/'

  3. Installez les dependances:
     pip install ultralytics pyyaml tqdm pillow
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    import sys

    if len(sys.argv) < 2:
        print_usage()
        return

    command = sys.argv[1].lower()

    if command == "convert":
        setup_directories()
        create_yaml_config()
        convert_deepfashion2()

    elif command == "train":
        setup_directories()
        yaml_path = create_yaml_config()
        stats = convert_deepfashion2()

        if stats is None:
            return

        print("\n" + "-"*60)
        response = input("Lancer l'entrainement? (o/n): ")
        if response.lower() == 'o':
            train_model(yaml_path)
            print("\n[OK] Entrainement termine!")
            print(f"[OK] Modele disponible: {MODEL_OUTPUT}")

    elif command == "eval":
        evaluate_model()

    elif command == "test":
        if len(sys.argv) < 3:
            print("[ERREUR] Specifiez le chemin de l'image")
            print("Usage: python train_deepfashion2_yolo.py test image.jpg")
            return
        test_on_image(sys.argv[2])

    elif command == "full":
        print("="*60)
        print("PIPELINE COMPLET: CONVERT + TRAIN + EVAL")
        print("="*60)

        setup_directories()
        yaml_path = create_yaml_config()
        stats = convert_deepfashion2()

        if stats is None:
            return

        print("\n" + "-"*60)
        response = input("Lancer l'entrainement? (o/n): ")
        if response.lower() != 'o':
            print("Entrainement annule.")
            return

        train_model(yaml_path)
        evaluate_model()

        print("\n" + "="*60)
        print("TERMINE!")
        print("="*60)
        print(f"\nModele entraine: {MODEL_OUTPUT}")
        print("\nPour utiliser le modele:")
        print("  1. Le fichier 'dresscode_yolo.pt' est pret")
        print("  2. Lancez l'application: python app.py")
        print("  3. Le detecteur utilisera automatiquement ce modele")

    else:
        print(f"[ERREUR] Commande inconnue: {command}")
        print_usage()


if __name__ == "__main__":
    main()
