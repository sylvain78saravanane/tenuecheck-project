"""
Script d'entrainement YOLOv8 sur DeepFashion2
Pour la detection des vetements interdits ENSITECH

DeepFashion2 contient 13 categories:
1: short_sleeve_top, 2: long_sleeve_top, 3: short_sleeve_outwear,
4: long_sleeve_outwear, 5: vest, 6: sling, 7: shorts, 8: trousers,
9: skirt, 10: short_sleeve_dress, 11: long_sleeve_dress,
12: vest_dress, 13: sling_dress
"""

import os
import json
import shutil
from pathlib import Path
from PIL import Image
import yaml

# Configuration
DEEPFASHION2_PATH = "deepfashion2"  # Dossier contenant le dataset
OUTPUT_PATH = "dataset_yolo"  # Dossier de sortie au format YOLO
MODEL_OUTPUT = "models"  # Dossier pour les modeles entraines

# Mapping DeepFashion2 vers nos classes d'interet pour le dress code
# On garde uniquement les vetements potentiellement interdits
CLASSES_DRESSCODE = {
    0: "shorts",           # shorts (categorie 7) - INTERDIT
    1: "mini_skirt",       # skirt courte (categorie 9) - INTERDIT
    2: "tank_top",         # vest/sling (categories 5,6) - INTERDIT
    3: "crop_top",         # short tops courts - INTERDIT
    4: "dress_short",      # robes courtes - A VERIFIER
}

# Mapping des categories DeepFashion2 vers nos classes
DEEPFASHION_TO_DRESSCODE = {
    7: 0,   # shorts -> shorts
    9: 1,   # skirt -> mini_skirt (on verifiera la longueur)
    5: 2,   # vest -> tank_top
    6: 2,   # sling -> tank_top
}


def setup_directories():
    """Cree la structure de dossiers pour YOLO"""
    dirs = [
        f"{OUTPUT_PATH}/images/train",
        f"{OUTPUT_PATH}/images/val",
        f"{OUTPUT_PATH}/labels/train",
        f"{OUTPUT_PATH}/labels/val",
        MODEL_OUTPUT
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("Dossiers crees avec succes")


def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convertit une bounding box DeepFashion2 au format YOLO
    DeepFashion2: [x1, y1, x2, y2]
    YOLO: [x_center, y_center, width, height] (normalise 0-1)
    """
    x1, y1, x2, y2 = bbox

    # Calculer le centre et les dimensions
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height

    # S'assurer que les valeurs sont dans [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))

    return x_center, y_center, width, height


def is_short_clothing(bbox, img_height):
    """
    Determine si un vetement est court (potentiellement interdit)
    Basé sur la hauteur relative de la bounding box
    """
    _, y1, _, y2 = bbox
    clothing_height = (y2 - y1) / img_height
    # Si le vetement fait moins de 25% de la hauteur de l'image, c'est probablement court
    return clothing_height < 0.25


def convert_deepfashion2_annotation(anno_path, image_path, output_path, split):
    """
    Convertit une annotation DeepFashion2 au format YOLO
    """
    with open(anno_path, 'r') as f:
        data = json.load(f)

    # Obtenir les dimensions de l'image
    try:
        with Image.open(image_path) as img:
            img_width, img_height = img.size
    except Exception as e:
        print(f"Erreur lecture image {image_path}: {e}")
        return False

    yolo_annotations = []

    # Parcourir les items dans l'annotation
    for key, value in data.items():
        if not key.startswith('item'):
            continue

        category_id = value.get('category_id', 0)
        bbox = value.get('bounding_box', [])

        if len(bbox) != 4:
            continue

        # Verifier si cette categorie nous interesse
        if category_id not in DEEPFASHION_TO_DRESSCODE:
            continue

        # Mapper vers notre classe
        yolo_class = DEEPFASHION_TO_DRESSCODE[category_id]

        # Pour les jupes, verifier si c'est une mini-jupe
        if category_id == 9:  # skirt
            if not is_short_clothing(bbox, img_height):
                continue  # Ignorer les jupes longues

        # Convertir la bbox au format YOLO
        x_center, y_center, width, height = convert_bbox_to_yolo(bbox, img_width, img_height)

        yolo_annotations.append(f"{yolo_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    if not yolo_annotations:
        return False

    # Copier l'image
    image_name = os.path.basename(image_path)
    image_dst = os.path.join(output_path, "images", split, image_name)
    shutil.copy(image_path, image_dst)

    # Sauvegarder les annotations YOLO
    label_name = os.path.splitext(image_name)[0] + ".txt"
    label_path = os.path.join(output_path, "labels", split, label_name)

    with open(label_path, 'w') as f:
        f.write('\n'.join(yolo_annotations))

    return True


def convert_dataset(deepfashion_path, output_path):
    """
    Convertit le dataset DeepFashion2 complet au format YOLO
    """
    converted_train = 0
    converted_val = 0

    # Traiter le split train
    train_annos = os.path.join(deepfashion_path, "train", "annos")
    train_images = os.path.join(deepfashion_path, "train", "image")

    if os.path.exists(train_annos):
        print("Conversion des donnees d'entrainement...")
        for anno_file in os.listdir(train_annos):
            if not anno_file.endswith('.json'):
                continue

            anno_path = os.path.join(train_annos, anno_file)
            image_name = anno_file.replace('.json', '.jpg')
            image_path = os.path.join(train_images, image_name)

            if os.path.exists(image_path):
                if convert_deepfashion2_annotation(anno_path, image_path, output_path, "train"):
                    converted_train += 1

                    if converted_train % 1000 == 0:
                        print(f"  {converted_train} images converties...")

    # Traiter le split validation
    val_annos = os.path.join(deepfashion_path, "validation", "annos")
    val_images = os.path.join(deepfashion_path, "validation", "image")

    if os.path.exists(val_annos):
        print("Conversion des donnees de validation...")
        for anno_file in os.listdir(val_annos):
            if not anno_file.endswith('.json'):
                continue

            anno_path = os.path.join(val_annos, anno_file)
            image_name = anno_file.replace('.json', '.jpg')
            image_path = os.path.join(val_images, image_name)

            if os.path.exists(image_path):
                if convert_deepfashion2_annotation(anno_path, image_path, output_path, "val"):
                    converted_val += 1

    return converted_train, converted_val


def create_dataset_yaml(output_path):
    """Cree le fichier de configuration YAML pour YOLO"""

    yaml_content = {
        'path': os.path.abspath(output_path),
        'train': 'images/train',
        'val': 'images/val',
        'names': CLASSES_DRESSCODE
    }

    yaml_path = os.path.join(output_path, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"Fichier de configuration cree: {yaml_path}")
    return yaml_path


def train_model(dataset_yaml, epochs=50, img_size=640, batch_size=16):
    """
    Entraine le modele YOLOv8 sur le dataset prepare
    """
    print("\n" + "="*60)
    print("ENTRAINEMENT DU MODELE YOLOv8")
    print("="*60)

    from ultralytics import YOLO

    # Charger le modele pre-entraine
    model = YOLO('yolov8n.pt')

    # Entrainer
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name='dresscode_detector',
        project=MODEL_OUTPUT,
        patience=10,  # Early stopping
        save=True,
        plots=True,
        verbose=True
    )

    # Copier le meilleur modele
    best_model_path = os.path.join(MODEL_OUTPUT, 'dresscode_detector', 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, 'dresscode_model.pt')
        print(f"\nModele entraine sauvegarde: dresscode_model.pt")

    return results


def download_instructions():
    """Affiche les instructions pour telecharger DeepFashion2"""
    print("\n" + "="*60)
    print("INSTRUCTIONS POUR OBTENIR DEEPFASHION2")
    print("="*60)
    print("""
DeepFashion2 est un dataset de recherche qui necessite une demande d'acces.

OPTION 1 - Demande officielle (recommandee):
1. Visitez: https://github.com/switchablenorms/DeepFashion2
2. Remplissez le formulaire de demande d'acces
3. Attendez l'approbation (peut prendre quelques jours)
4. Telechargez et extrayez dans le dossier 'deepfashion2/'

OPTION 2 - Utiliser un subset depuis Kaggle:
1. Visitez: https://www.kaggle.com/datasets
2. Recherchez "DeepFashion2"
3. Telechargez un subset disponible

OPTION 3 - Utiliser Roboflow (plus simple):
1. Visitez: https://universe.roboflow.com
2. Recherchez "fashion detection" ou "clothing detection"
3. Telechargez un dataset au format YOLOv8

Structure attendue:
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


def main():
    print("="*60)
    print("ENTRAINEMENT DRESSCODE - DEEPFASHION2")
    print("="*60)

    # Verifier si le dataset existe
    if not os.path.exists(DEEPFASHION2_PATH):
        print(f"\nDataset non trouve: {DEEPFASHION2_PATH}")
        download_instructions()

        response = input("\nVoulez-vous creer la structure de dossiers? (o/n): ")
        if response.lower() == 'o':
            os.makedirs(f"{DEEPFASHION2_PATH}/train/annos", exist_ok=True)
            os.makedirs(f"{DEEPFASHION2_PATH}/train/image", exist_ok=True)
            os.makedirs(f"{DEEPFASHION2_PATH}/validation/annos", exist_ok=True)
            os.makedirs(f"{DEEPFASHION2_PATH}/validation/image", exist_ok=True)
            print("Structure creee. Placez vos fichiers et relancez le script.")
        return

    # Creer les dossiers de sortie
    setup_directories()

    # Convertir le dataset
    print("\nConversion du dataset DeepFashion2 vers format YOLO...")
    train_count, val_count = convert_dataset(DEEPFASHION2_PATH, OUTPUT_PATH)

    print(f"\nImages d'entrainement converties: {train_count}")
    print(f"Images de validation converties: {val_count}")

    if train_count == 0:
        print("\nAucune donnee convertie. Verifiez le contenu du dataset.")
        return

    # Creer le fichier YAML
    yaml_path = create_dataset_yaml(OUTPUT_PATH)

    # Demander confirmation pour l'entrainement
    print("\n" + "="*60)
    print("PRET POUR L'ENTRAINEMENT")
    print("="*60)
    print(f"- Images d'entrainement: {train_count}")
    print(f"- Images de validation: {val_count}")
    print(f"- Configuration: {yaml_path}")
    print(f"- Classes: {list(CLASSES_DRESSCODE.values())}")

    response = input("\nLancer l'entrainement? (o/n): ")

    if response.lower() == 'o':
        train_model(yaml_path, epochs=50, batch_size=16)
        print("\n" + "="*60)
        print("ENTRAINEMENT TERMINE")
        print("="*60)
        print("Pour utiliser le nouveau modele:")
        print("1. Copiez 'dresscode_model.pt' dans le dossier principal")
        print("2. Modifiez detector.py pour charger ce modele")
    else:
        print("Entrainement annule.")


if __name__ == "__main__":
    main()
