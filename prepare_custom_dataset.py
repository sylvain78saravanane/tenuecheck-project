"""
Preparation d'un dataset personnalise pour YOLO
Detection des vetements interdits ENSITECH

Classes a detecter:
0: short           - Short/Bermuda
1: mini_skirt      - Mini-jupe
2: crop_top        - T-shirt au-dessus du nombril
3: sportswear      - Tenue de sport (leggings, brassiere, etc.)
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
import urllib.request
import zipfile
from pathlib import Path

# Configuration
DATASET_DIR = "dataset_dresscode"
CLASSES = [
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


def create_directory_structure():
    """Cree la structure de dossiers pour YOLO"""
    dirs = [
        f"{DATASET_DIR}/images/train",
        f"{DATASET_DIR}/images/val",
        f"{DATASET_DIR}/labels/train",
        f"{DATASET_DIR}/labels/val",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("Structure de dossiers creee")


def create_dataset_yaml():
    """Cree le fichier de configuration YAML pour YOLO"""
    yaml_content = f"""# ENSITECH Dress Code Detection Dataset
# Entraine avec: python train_custom_yolo.py

path: {os.path.abspath(DATASET_DIR)}
train: images/train
val: images/val

# Classes de vetements interdits
names:
  0: short
  1: mini_skirt
  2: crop_top
  3: sportswear
  4: ripped_jeans
  5: flip_flops
  6: cap
  7: hat
  8: beanie
  9: bandana
"""

    yaml_path = f"{DATASET_DIR}/dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"Fichier de configuration cree: {yaml_path}")
    return yaml_path


def download_roboflow_datasets():
    """
    Telecharge des datasets depuis Roboflow Universe
    Ces datasets sont publics et gratuits
    """
    print("\n" + "="*60)
    print("TELECHARGEMENT DE DATASETS ROBOFLOW")
    print("="*60)

    # Liste de datasets publics utiles sur Roboflow
    datasets_info = """
Les datasets suivants peuvent etre utiles:

1. Caps and Hats Detection:
   https://universe.roboflow.com/search?q=cap+detection

2. Flip Flops / Sandals:
   https://universe.roboflow.com/search?q=sandals+flip+flops

3. Shorts Detection:
   https://universe.roboflow.com/search?q=shorts+clothing

4. Fashion / Clothing Detection:
   https://universe.roboflow.com/search?q=fashion+clothing

Pour telecharger:
1. Creez un compte gratuit sur https://roboflow.com
2. Trouvez un dataset pertinent
3. Cliquez sur "Download" > Format "YOLOv8"
4. Placez les fichiers dans dataset_dresscode/
"""
    print(datasets_info)

    return None


def create_sample_annotations():
    """
    Cree des fichiers d'annotation exemples pour montrer le format
    """
    print("\n" + "="*60)
    print("CREATION D'EXEMPLES D'ANNOTATIONS")
    print("="*60)

    # Exemple de fichier d'annotation YOLO
    example_annotation = """# Format YOLO: class_id x_center y_center width height
# Toutes les valeurs sont normalisees (0-1)
# Exemple pour une image avec un short et une casquette:

# class_id x_center y_center width height
# 0        0.5      0.7      0.3    0.2     <- short au centre-bas
# 6        0.5      0.1      0.15   0.1     <- casquette en haut

# Pour creer vos annotations:
# 1. Utilisez LabelImg: pip install labelImg && labelImg
# 2. Ou utilisez Roboflow: https://roboflow.com (gratuit)
# 3. Ou utilisez CVAT: https://cvat.ai (gratuit)
"""

    example_path = f"{DATASET_DIR}/ANNOTATION_FORMAT.txt"
    with open(example_path, 'w') as f:
        f.write(example_annotation)

    print(f"Exemple d'annotation cree: {example_path}")

    # Creer un fichier README
    readme_content = """# Dataset ENSITECH Dress Code

## Structure requise

```
dataset_dresscode/
├── images/
│   ├── train/      <- Images d'entrainement (.jpg, .png)
│   └── val/        <- Images de validation (.jpg, .png)
├── labels/
│   ├── train/      <- Annotations (.txt, meme nom que l'image)
│   └── val/        <- Annotations (.txt)
└── dataset.yaml    <- Configuration
```

## Classes (0-9)

| ID | Classe | Description |
|----|--------|-------------|
| 0 | short | Short, bermuda |
| 1 | mini_skirt | Mini-jupe |
| 2 | crop_top | T-shirt au-dessus du nombril |
| 3 | sportswear | Tenue de sport (leggings, brassiere) |
| 4 | ripped_jeans | Jean troue/dechire |
| 5 | flip_flops | Tongs, sandales |
| 6 | cap | Casquette |
| 7 | hat | Chapeau |
| 8 | beanie | Bonnet |
| 9 | bandana | Bandana |

## Format des annotations YOLO

Chaque image `image.jpg` doit avoir un fichier `image.txt` avec:
```
class_id x_center y_center width height
```

Exemple:
```
0 0.5 0.7 0.3 0.2
6 0.5 0.1 0.15 0.1
```

## Comment collecter des images

### Option 1: Prendre vos propres photos
- Photographiez des personnes portant les vetements interdits
- Variez les angles, l'eclairage, les fonds
- Minimum recommande: 100 images par classe

### Option 2: Telecharger depuis Roboflow
1. Allez sur https://universe.roboflow.com
2. Cherchez: "cap detection", "shorts", "flip flops", etc.
3. Telechargez au format YOLOv8
4. Fusionnez les datasets

### Option 3: Utiliser des images web
- Google Images (attention aux droits)
- Pexels, Unsplash (images libres)
- Annotez avec LabelImg ou Roboflow

## Annotation avec LabelImg

```bash
pip install labelImg
labelImg dataset_dresscode/images/train
```

1. Ouvrez le dossier d'images
2. Selectionnez le format YOLO
3. Dessinez des rectangles autour des vetements
4. Sauvegardez (fichiers .txt crees automatiquement)

## Lancer l'entrainement

Une fois le dataset pret:
```bash
python train_custom_yolo.py
```
"""

    readme_path = f"{DATASET_DIR}/README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"README cree: {readme_path}")


def download_open_images_subset():
    """
    Telecharge un subset d'Open Images avec les classes pertinentes
    """
    print("\n" + "="*60)
    print("TELECHARGEMENT OPEN IMAGES (Optionnel)")
    print("="*60)

    try:
        # Installer fiftyone si necessaire
        import subprocess
        subprocess.run(["pip", "install", "fiftyone", "-q"], check=True)

        import fiftyone as fo
        import fiftyone.zoo as foz

        # Classes Open Images pertinentes
        classes_to_download = [
            "Shorts",
            "Miniskirt",
            "Flip-flops",
            "Sandal",
            "Hat",
            "Fedora",
            "Sombrero",
            "Sun hat",
            "Beanie",
        ]

        print(f"Telechargement des classes: {classes_to_download}")

        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="train",
            label_types=["detections"],
            classes=classes_to_download,
            max_samples=500,
        )

        print(f"Dataset telecharge: {len(dataset)} images")
        return dataset

    except Exception as e:
        print(f"Erreur: {e}")
        print("\nPour telecharger Open Images manuellement:")
        print("1. pip install fiftyone")
        print("2. Relancez ce script")
        return None


def merge_roboflow_datasets(dataset_paths):
    """
    Fusionne plusieurs datasets Roboflow en un seul
    """
    print("\n" + "="*60)
    print("FUSION DE DATASETS")
    print("="*60)

    # Mapping des classes externes vers nos classes
    class_mapping = {
        # Shorts
        "shorts": 0, "short": 0, "bermuda": 0,
        # Mini-jupes
        "miniskirt": 1, "mini-skirt": 1, "mini_skirt": 1, "skirt": 1,
        # Crop tops
        "crop_top": 2, "crop-top": 2, "croptop": 2,
        # Sportswear
        "sportswear": 3, "leggings": 3, "sports_bra": 3, "athletic": 3,
        # Jeans troues
        "ripped_jeans": 4, "ripped-jeans": 4, "torn_jeans": 4,
        # Tongs
        "flip_flops": 5, "flip-flops": 5, "sandals": 5, "sandal": 5, "slippers": 5,
        # Casquettes
        "cap": 6, "baseball_cap": 6, "baseball-cap": 6,
        # Chapeaux
        "hat": 7, "fedora": 7, "sun_hat": 7,
        # Bonnets
        "beanie": 8, "winter_hat": 8,
        # Bandanas
        "bandana": 9, "headband": 9,
    }

    for dataset_path in dataset_paths:
        if not os.path.exists(dataset_path):
            print(f"Dataset non trouve: {dataset_path}")
            continue

        print(f"Traitement de: {dataset_path}")

        # Lire le data.yaml du dataset source
        yaml_path = os.path.join(dataset_path, "data.yaml")
        if os.path.exists(yaml_path):
            import yaml
            with open(yaml_path, 'r') as f:
                source_config = yaml.safe_load(f)

            source_classes = source_config.get('names', {})
            print(f"  Classes sources: {source_classes}")

        # Copier et convertir les annotations
        for split in ['train', 'val']:
            src_images = os.path.join(dataset_path, 'images', split)
            src_labels = os.path.join(dataset_path, 'labels', split)

            if not os.path.exists(src_images):
                continue

            dst_images = os.path.join(DATASET_DIR, 'images', split)
            dst_labels = os.path.join(DATASET_DIR, 'labels', split)

            for img_file in os.listdir(src_images):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                # Copier l'image
                src_img = os.path.join(src_images, img_file)
                dst_img = os.path.join(dst_images, img_file)
                shutil.copy(src_img, dst_img)

                # Convertir les annotations
                label_file = os.path.splitext(img_file)[0] + '.txt'
                src_label = os.path.join(src_labels, label_file)
                dst_label = os.path.join(dst_labels, label_file)

                if os.path.exists(src_label):
                    convert_annotations(src_label, dst_label, source_classes, class_mapping)

    print("Fusion terminee")


def convert_annotations(src_path, dst_path, source_classes, class_mapping):
    """Convertit les annotations d'un format a l'autre"""
    new_lines = []

    with open(src_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            src_class_id = int(parts[0])
            src_class_name = source_classes.get(src_class_id, "").lower()

            # Trouver la classe correspondante
            new_class_id = None
            for key, value in class_mapping.items():
                if key in src_class_name or src_class_name in key:
                    new_class_id = value
                    break

            if new_class_id is not None:
                new_line = f"{new_class_id} {' '.join(parts[1:])}"
                new_lines.append(new_line)

    if new_lines:
        with open(dst_path, 'w') as f:
            f.write('\n'.join(new_lines))


def print_instructions():
    """Affiche les instructions completes"""
    print("\n" + "="*60)
    print("INSTRUCTIONS POUR CREER VOTRE DATASET")
    print("="*60)

    instructions = """
## METHODE RECOMMANDEE: Roboflow (plus simple)

1. Allez sur https://universe.roboflow.com

2. Telechargez ces datasets (format YOLOv8):
   - Cherchez "cap detection" ou "hat detection"
   - Cherchez "shorts detection"
   - Cherchez "flip flops" ou "sandals"
   - Cherchez "clothing detection"

3. Pour chaque dataset telecharge:
   - Decompressez le ZIP
   - Copiez les images dans dataset_dresscode/images/train/
   - Copiez les labels dans dataset_dresscode/labels/train/

4. Si les classes ne correspondent pas:
   - Editez les fichiers .txt
   - Remplacez les class_id par les notres (0-9)

## METHODE ALTERNATIVE: Creer votre propre dataset

1. Collectez des images:
   - Prenez des photos de personnes avec vetements interdits
   - Ou telechargez depuis Google Images / Pexels

2. Annotez avec LabelImg:
   ```
   pip install labelImg
   labelImg dataset_dresscode/images/train
   ```

3. Pour chaque image:
   - Dessinez un rectangle autour du vetement interdit
   - Selectionnez la classe (0-9)
   - Sauvegardez

4. Repetez pour avoir au moins 50-100 images par classe

## LANCER L'ENTRAINEMENT

Une fois le dataset pret (images + annotations):
```
python train_custom_yolo.py
```

L'entrainement prend environ 1-2 heures sur CPU,
ou 15-30 minutes avec un GPU.
"""
    print(instructions)


def count_dataset_stats():
    """Compte les statistiques du dataset"""
    print("\n" + "="*60)
    print("STATISTIQUES DU DATASET")
    print("="*60)

    stats = {cls: {"train": 0, "val": 0} for cls in CLASSES}

    for split in ['train', 'val']:
        labels_dir = f"{DATASET_DIR}/labels/{split}"
        if not os.path.exists(labels_dir):
            continue

        for label_file in os.listdir(labels_dir):
            if not label_file.endswith('.txt'):
                continue

            with open(os.path.join(labels_dir, label_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if 0 <= class_id < len(CLASSES):
                            stats[CLASSES[class_id]][split] += 1

    print(f"\n{'Classe':<15} {'Train':>8} {'Val':>8} {'Total':>8}")
    print("-" * 45)

    total_train = 0
    total_val = 0

    for cls in CLASSES:
        train = stats[cls]["train"]
        val = stats[cls]["val"]
        total = train + val
        total_train += train
        total_val += val
        print(f"{cls:<15} {train:>8} {val:>8} {total:>8}")

    print("-" * 45)
    print(f"{'TOTAL':<15} {total_train:>8} {total_val:>8} {total_train + total_val:>8}")

    if total_train == 0:
        print("\n[!] Aucune annotation trouvee. Suivez les instructions ci-dessus.")


def main():
    print("="*60)
    print("PREPARATION DATASET - ENSITECH DRESS CODE")
    print("="*60)

    # Creer la structure
    create_directory_structure()

    # Creer le fichier YAML
    create_dataset_yaml()

    # Creer les exemples
    create_sample_annotations()

    # Afficher les statistiques
    count_dataset_stats()

    # Afficher les instructions
    print_instructions()

    # Info sur Roboflow
    download_roboflow_datasets()


if __name__ == "__main__":
    main()
