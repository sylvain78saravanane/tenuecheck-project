"""
Construction du dataset ENSITECH Dress Code
Combine plusieurs sources:
- DeepFashion2 (shorts, skirts, crop tops)
- Roboflow datasets (caps, hats, flip flops)
- Open Images (optionnel)

Classes cibles:
0: short           - Short/Bermuda
1: mini_skirt      - Mini-jupe
2: crop_top        - T-shirt au-dessus du nombril
3: sportswear      - Tenue de sport
4: ripped_jeans    - Jean troue/dechire
5: flip_flops      - Tongs/Sandales
6: cap             - Casquette
7: hat             - Chapeau
8: beanie          - Bonnet
9: bandana         - Bandana
"""

import os
import json
import shutil
import random
from pathlib import Path
from PIL import Image
import yaml

# Configuration
OUTPUT_DIR = "dataset_dresscode"
DEEPFASHION2_DIR = "deepfashion2"

# Classes cibles ENSITECH
TARGET_CLASSES = [
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

# Mapping DeepFashion2 categories vers nos classes
# DeepFashion2: 1-13 categories
DEEPFASHION2_MAPPING = {
    7: 0,   # shorts -> short
    9: 1,   # skirt -> mini_skirt (on filtrera par taille)
    5: 2,   # vest -> crop_top
    6: 2,   # sling -> crop_top
}

# Mapping pour datasets Roboflow (a adapter selon le dataset telecharge)
ROBOFLOW_MAPPINGS = {
    # Format: "nom_classe_roboflow": class_id_ensitech
    "cap": 6,
    "baseball_cap": 6,
    "baseball-cap": 6,
    "hat": 7,
    "sun_hat": 7,
    "fedora": 7,
    "cowboy_hat": 7,
    "beanie": 8,
    "winter_hat": 8,
    "knit_cap": 8,
    "bandana": 9,
    "headband": 9,
    "flip_flops": 5,
    "flip-flops": 5,
    "sandals": 5,
    "sandal": 5,
    "slippers": 5,
    "shorts": 0,
    "short": 0,
    "bermuda": 0,
    "miniskirt": 1,
    "mini_skirt": 1,
    "mini-skirt": 1,
    "skirt": 1,
    "crop_top": 2,
    "crop-top": 2,
    "croptop": 2,
    "tank_top": 2,
    "sportswear": 3,
    "leggings": 3,
    "sports_bra": 3,
    "athletic_wear": 3,
    "ripped_jeans": 4,
    "ripped-jeans": 4,
    "torn_jeans": 4,
    "distressed_jeans": 4,
}


def setup_directories():
    """Cree la structure de dossiers"""
    dirs = [
        f"{OUTPUT_DIR}/images/train",
        f"{OUTPUT_DIR}/images/val",
        f"{OUTPUT_DIR}/labels/train",
        f"{OUTPUT_DIR}/labels/val",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("[OK] Structure de dossiers creee")


def create_dataset_yaml():
    """Cree le fichier de configuration YAML"""
    yaml_content = {
        'path': os.path.abspath(OUTPUT_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(TARGET_CLASSES)}
    }

    yaml_path = f"{OUTPUT_DIR}/dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"[OK] Configuration YAML creee: {yaml_path}")
    return yaml_path


def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convertit bbox [x1, y1, x2, y2] vers format YOLO [x_center, y_center, width, height]
    Valeurs normalisees entre 0 et 1
    """
    x1, y1, x2, y2 = bbox

    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height

    # Clamp entre 0 et 1
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))

    return x_center, y_center, width, height


def is_mini_skirt(bbox, img_height):
    """
    Determine si une jupe est une mini-jupe basee sur sa taille relative
    """
    _, y1, _, y2 = bbox
    clothing_height = (y2 - y1) / img_height
    # Mini-jupe = moins de 20% de la hauteur de l'image
    return clothing_height < 0.20


def process_deepfashion2():
    """
    Convertit DeepFashion2 vers notre format
    """
    print("\n" + "="*60)
    print("TRAITEMENT DE DEEPFASHION2")
    print("="*60)

    if not os.path.exists(DEEPFASHION2_DIR):
        print(f"[!] DeepFashion2 non trouve: {DEEPFASHION2_DIR}")
        print_deepfashion2_instructions()
        return 0, 0

    converted_train = 0
    converted_val = 0

    for split_name, output_split in [("train", "train"), ("validation", "val")]:
        annos_dir = os.path.join(DEEPFASHION2_DIR, split_name, "annos")
        images_dir = os.path.join(DEEPFASHION2_DIR, split_name, "image")

        if not os.path.exists(annos_dir):
            print(f"[!] Dossier non trouve: {annos_dir}")
            continue

        print(f"\nTraitement du split '{split_name}'...")
        anno_files = [f for f in os.listdir(annos_dir) if f.endswith('.json')]

        for i, anno_file in enumerate(anno_files):
            if i % 1000 == 0 and i > 0:
                print(f"  {i}/{len(anno_files)} fichiers traites...")

            anno_path = os.path.join(annos_dir, anno_file)
            image_name = anno_file.replace('.json', '.jpg')
            image_path = os.path.join(images_dir, image_name)

            if not os.path.exists(image_path):
                continue

            # Lire l'annotation
            try:
                with open(anno_path, 'r') as f:
                    data = json.load(f)
            except Exception:
                continue

            # Obtenir les dimensions de l'image
            try:
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
            except Exception:
                continue

            # Convertir les annotations
            yolo_annotations = []

            for key, value in data.items():
                if not key.startswith('item'):
                    continue

                category_id = value.get('category_id', 0)
                bbox = value.get('bounding_box', [])

                if len(bbox) != 4:
                    continue

                # Verifier si cette categorie nous interesse
                if category_id not in DEEPFASHION2_MAPPING:
                    continue

                target_class = DEEPFASHION2_MAPPING[category_id]

                # Filtrer les jupes longues (garder seulement mini-jupes)
                if category_id == 9:  # skirt
                    if not is_mini_skirt(bbox, img_height):
                        continue

                # Convertir la bbox
                x_center, y_center, width, height = convert_bbox_to_yolo(bbox, img_width, img_height)
                yolo_annotations.append(f"{target_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            if not yolo_annotations:
                continue

            # Copier l'image
            new_image_name = f"df2_{image_name}"
            dst_image = os.path.join(OUTPUT_DIR, "images", output_split, new_image_name)
            shutil.copy(image_path, dst_image)

            # Sauvegarder les labels
            label_name = f"df2_{os.path.splitext(image_name)[0]}.txt"
            dst_label = os.path.join(OUTPUT_DIR, "labels", output_split, label_name)
            with open(dst_label, 'w') as f:
                f.write('\n'.join(yolo_annotations))

            if output_split == "train":
                converted_train += 1
            else:
                converted_val += 1

    print(f"\n[OK] DeepFashion2: {converted_train} train, {converted_val} val")
    return converted_train, converted_val


def process_roboflow_dataset(dataset_path, prefix="rb"):
    """
    Traite un dataset Roboflow telecharge au format YOLOv8
    """
    print(f"\nTraitement du dataset Roboflow: {dataset_path}")

    if not os.path.exists(dataset_path):
        print(f"[!] Dataset non trouve: {dataset_path}")
        return 0, 0

    converted_train = 0
    converted_val = 0

    # Lire le data.yaml du dataset source
    yaml_path = os.path.join(dataset_path, "data.yaml")
    source_classes = {}

    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            source_config = yaml.safe_load(f)
            names = source_config.get('names', {})
            if isinstance(names, list):
                source_classes = {i: name for i, name in enumerate(names)}
            else:
                source_classes = names
        print(f"  Classes sources: {source_classes}")

    # Traiter train et val
    for split in ['train', 'valid', 'val', 'test']:
        src_images = os.path.join(dataset_path, split, 'images')
        src_labels = os.path.join(dataset_path, split, 'labels')

        # Essayer aussi la structure plate
        if not os.path.exists(src_images):
            src_images = os.path.join(dataset_path, 'images', split)
            src_labels = os.path.join(dataset_path, 'labels', split)

        if not os.path.exists(src_images):
            continue

        output_split = "val" if split in ['valid', 'val', 'test'] else "train"
        print(f"  Traitement {split} -> {output_split}...")

        image_files = [f for f in os.listdir(src_images)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for img_file in image_files:
            src_img = os.path.join(src_images, img_file)
            label_file = os.path.splitext(img_file)[0] + '.txt'
            src_label = os.path.join(src_labels, label_file)

            if not os.path.exists(src_label):
                continue

            # Lire et convertir les annotations
            new_annotations = []
            with open(src_label, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    try:
                        src_class_id = int(parts[0])
                        src_class_name = source_classes.get(src_class_id, "").lower()

                        # Trouver la classe cible
                        target_class = None
                        for key, value in ROBOFLOW_MAPPINGS.items():
                            if key == src_class_name or key in src_class_name or src_class_name in key:
                                target_class = value
                                break

                        if target_class is not None:
                            new_annotations.append(f"{target_class} {' '.join(parts[1:])}")

                    except (ValueError, IndexError):
                        continue

            if not new_annotations:
                continue

            # Copier l'image avec prefixe unique
            new_img_name = f"{prefix}_{img_file}"
            dst_img = os.path.join(OUTPUT_DIR, "images", output_split, new_img_name)
            shutil.copy(src_img, dst_img)

            # Sauvegarder les labels
            new_label_name = f"{prefix}_{os.path.splitext(img_file)[0]}.txt"
            dst_label = os.path.join(OUTPUT_DIR, "labels", output_split, new_label_name)
            with open(dst_label, 'w') as f:
                f.write('\n'.join(new_annotations))

            if output_split == "train":
                converted_train += 1
            else:
                converted_val += 1

    print(f"  [OK] {converted_train} train, {converted_val} val")
    return converted_train, converted_val


def scan_roboflow_datasets():
    """
    Scanne le dossier courant pour trouver des datasets Roboflow
    """
    print("\n" + "="*60)
    print("RECHERCHE DE DATASETS ROBOFLOW")
    print("="*60)

    datasets_found = []

    # Chercher des dossiers avec data.yaml
    for item in os.listdir('.'):
        if os.path.isdir(item) and item not in [OUTPUT_DIR, DEEPFASHION2_DIR, 'runs', '__pycache__', '.git']:
            yaml_path = os.path.join(item, 'data.yaml')
            if os.path.exists(yaml_path):
                datasets_found.append(item)
                print(f"  [+] Dataset trouve: {item}")

    return datasets_found


def count_dataset_stats():
    """Compte les statistiques du dataset"""
    print("\n" + "="*60)
    print("STATISTIQUES DU DATASET")
    print("="*60)

    stats = {cls: {"train": 0, "val": 0} for cls in TARGET_CLASSES}
    image_counts = {"train": 0, "val": 0}

    for split in ['train', 'val']:
        labels_dir = f"{OUTPUT_DIR}/labels/{split}"
        images_dir = f"{OUTPUT_DIR}/images/{split}"

        if os.path.exists(images_dir):
            image_counts[split] = len([f for f in os.listdir(images_dir)
                                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        if not os.path.exists(labels_dir):
            continue

        for label_file in os.listdir(labels_dir):
            if not label_file.endswith('.txt'):
                continue

            with open(os.path.join(labels_dir, label_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(parts[0])
                            if 0 <= class_id < len(TARGET_CLASSES):
                                stats[TARGET_CLASSES[class_id]][split] += 1
                        except ValueError:
                            pass

    # Afficher les statistiques
    print(f"\n{'Classe':<15} {'Train':>8} {'Val':>8} {'Total':>8}")
    print("-" * 45)

    total_train = 0
    total_val = 0
    empty_classes = []

    for cls in TARGET_CLASSES:
        train = stats[cls]["train"]
        val = stats[cls]["val"]
        total = train + val
        total_train += train
        total_val += val

        status = ""
        if total == 0:
            status = " [VIDE]"
            empty_classes.append(cls)
        elif total < 50:
            status = " [FAIBLE]"

        print(f"{cls:<15} {train:>8} {val:>8} {total:>8}{status}")

    print("-" * 45)
    print(f"{'TOTAL':<15} {total_train:>8} {total_val:>8} {total_train + total_val:>8}")
    print(f"\nImages: {image_counts['train']} train, {image_counts['val']} val")

    if empty_classes:
        print(f"\n[!] Classes sans donnees: {', '.join(empty_classes)}")
        print("    Ajoutez des images pour ces classes via Roboflow ou manuellement.")

    return stats


def print_deepfashion2_instructions():
    """Affiche les instructions pour obtenir DeepFashion2"""
    print("""
Pour obtenir DeepFashion2:

1. DEMANDE OFFICIELLE (recommande):
   - Visitez: https://github.com/switchablenorms/DeepFashion2
   - Remplissez le formulaire de demande
   - Telechargez et extrayez dans 'deepfashion2/'

2. STRUCTURE ATTENDUE:
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
""")


def print_roboflow_instructions():
    """Affiche les instructions pour telecharger depuis Roboflow"""
    print("""
Pour completer votre dataset avec Roboflow:

1. Allez sur https://universe.roboflow.com

2. Recherchez et telechargez ces datasets (format YOLOv8):

   COUVRE-CHEFS:
   - "cap detection" ou "baseball cap"
   - "hat detection"
   - "beanie detection"

   CHAUSSURES:
   - "flip flops detection"
   - "sandals detection"

   VETEMENTS:
   - "shorts detection"
   - "sportswear detection"

3. Pour chaque dataset telecharge:
   - Decompressez le ZIP dans ce dossier
   - Relancez ce script

4. Le script detectera automatiquement les nouveaux datasets
""")


def main():
    print("="*60)
    print("CONSTRUCTION DU DATASET - ENSITECH DRESS CODE")
    print("="*60)
    print(f"\nClasses cibles: {TARGET_CLASSES}")

    # Creer la structure
    setup_directories()

    # Creer le YAML
    create_dataset_yaml()

    total_train = 0
    total_val = 0

    # Traiter DeepFashion2
    train, val = process_deepfashion2()
    total_train += train
    total_val += val

    # Scanner et traiter les datasets Roboflow
    roboflow_datasets = scan_roboflow_datasets()

    if roboflow_datasets:
        for i, dataset in enumerate(roboflow_datasets):
            train, val = process_roboflow_dataset(dataset, prefix=f"rb{i}")
            total_train += train
            total_val += val
    else:
        print("\n[INFO] Aucun dataset Roboflow trouve.")
        print_roboflow_instructions()

    # Afficher les statistiques
    stats = count_dataset_stats()

    # Resume
    print("\n" + "="*60)
    print("RESUME")
    print("="*60)
    print(f"Total images: {total_train} train, {total_val} val")

    if total_train == 0:
        print("\n[!] Aucune image convertie.")
        print("[!] Ajoutez DeepFashion2 et/ou des datasets Roboflow.")
        print_deepfashion2_instructions()
        print_roboflow_instructions()
    elif total_train < 100:
        print("\n[!] Dataset tres petit. Ajoutez plus d'images pour de meilleurs resultats.")
    else:
        print("\n[OK] Dataset pret pour l'entrainement!")
        print("\nProchaine etape:")
        print("  python train_custom_yolo.py check")
        print("  python train_custom_yolo.py train")


if __name__ == "__main__":
    main()
