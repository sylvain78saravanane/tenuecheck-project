"""
Script d'entraînement YOLOv8 sur DeepFashion2
Pour améliorer la détection des vêtements interdits
"""

import os
import json
import shutil
from pathlib import Path
from ultralytics import YOLO
import yaml


# Configuration
DEEPFASHION2_PATH = "deepfashion2"  # Chemin vers le dataset DeepFashion2
OUTPUT_PATH = "dataset_yolo"  # Dossier de sortie au format YOLO
MODEL_OUTPUT = "models"  # Dossier pour les modèles entraînés

# Mapping des catégories DeepFashion2 vers nos catégories d'intérêt
# DeepFashion2 categories:
# 1: short_sleeve_top, 2: long_sleeve_top, 3: short_sleeve_outwear,
# 4: long_sleeve_outwear, 5: vest, 6: sling, 7: shorts, 8: trousers,
# 9: skirt, 10: short_sleeve_dress, 11: long_sleeve_dress,
# 12: vest_dress, 13: sling_dress

CATEGORIES_MAPPING = {
    1: "top",              # short_sleeve_top
    2: "top",              # long_sleeve_top
    3: "outwear",          # short_sleeve_outwear
    4: "outwear",          # long_sleeve_outwear
    5: "vest",             # vest (débardeur) - potentiellement interdit
    6: "sling",            # sling (bretelles fines) - potentiellement interdit
    7: "shorts",           # shorts - INTERDIT
    8: "trousers",         # trousers
    9: "skirt",            # skirt - vérifier longueur
    10: "dress",           # short_sleeve_dress
    11: "dress",           # long_sleeve_dress
    12: "vest_dress",      # vest_dress
    13: "sling_dress",     # sling_dress
}

# Classes que nous voulons détecter pour le dress code
CLASSES_INTERET = {
    0: "person",
    1: "shorts",           # Short - INTERDIT
    2: "mini_skirt",       # Mini-jupe - INTERDIT
    3: "crop_top",         # Crop top - INTERDIT
    4: "tank_top",         # Débardeur échancré - INTERDIT
    5: "sportswear",       # Tenue de sport - INTERDIT
    6: "ripped_jeans",     # Jean troué - INTERDIT
    7: "cap",              # Casquette - INTERDIT
    8: "flip_flops",       # Tongs - INTERDIT
}


def setup_directories():
    """Crée la structure de dossiers pour YOLO"""
    dirs = [
        f"{OUTPUT_PATH}/images/train",
        f"{OUTPUT_PATH}/images/val",
        f"{OUTPUT_PATH}/labels/train",
        f"{OUTPUT_PATH}/labels/val",
        MODEL_OUTPUT
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("Dossiers créés avec succès")


def convert_deepfashion2_to_yolo(deepfashion_path, output_path, split="train"):
    """
    Convertit les annotations DeepFashion2 au format YOLO

    DeepFashion2 utilise des annotations JSON avec bounding boxes
    Format YOLO: class_id x_center y_center width height (normalisé)
    """
    annos_path = os.path.join(deepfashion_path, split, "annos")
    images_path = os.path.join(deepfashion_path, split, "image")

    if not os.path.exists(annos_path):
        print(f"Chemin non trouvé: {annos_path}")
        print("Téléchargez DeepFashion2 depuis: https://github.com/switchablenorms/DeepFashion2")
        return 0

    converted = 0

    for anno_file in os.listdir(annos_path):
        if not anno_file.endswith('.json'):
            continue

        with open(os.path.join(annos_path, anno_file), 'r') as f:
            data = json.load(f)

        image_name = anno_file.replace('.json', '.jpg')
        image_src = os.path.join(images_path, image_name)

        if not os.path.exists(image_src):
            continue

        # Copier l'image
        image_dst = os.path.join(output_path, "images", split, image_name)
        shutil.copy(image_src, image_dst)

        # Convertir les annotations
        img_width = data.get('image_width', 1)
        img_height = data.get('image_height', 1)

        yolo_annotations = []

        for item_key, item_data in data.items():
            if not item_key.startswith('item'):
                continue

            category_id = item_data.get('category_id', 0)
            bbox = item_data.get('bounding_box', [])

            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox

            # Convertir en format YOLO (normalisé)
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height

            # Mapper la catégorie
            # Pour l'entraînement, on garde les catégories qui nous intéressent
            yolo_class = map_to_dress_code_class(category_id, item_data)

            if yolo_class is not None:
                yolo_annotations.append(f"{yolo_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Sauvegarder le fichier label
        if yolo_annotations:
            label_file = os.path.join(output_path, "labels", split, anno_file.replace('.json', '.txt'))
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            converted += 1

    return converted


def map_to_dress_code_class(category_id, item_data):
    """
    Mappe une catégorie DeepFashion2 vers nos classes de dress code
    """
    # 7 = shorts dans DeepFashion2 -> classe 1 (shorts) pour nous
    if category_id == 7:
        return 1  # shorts

    # 9 = skirt -> vérifier si c'est une mini-jupe (basé sur la taille)
    if category_id == 9:
        bbox = item_data.get('bounding_box', [0, 0, 0, 0])
        height = bbox[3] - bbox[1] if len(bbox) == 4 else 0
        # Si la jupe est courte (heuristique simple)
        return 2  # mini_skirt

    # 5 = vest, 6 = sling -> potentiellement crop top ou débardeur
    if category_id in [5, 6]:
        return 4  # tank_top

    # Pour les autres catégories, on ne les inclut pas
    # car elles ne correspondent pas à des vêtements interdits
    return None


def create_dataset_yaml(output_path, num_classes):
    """Crée le fichier de configuration YAML pour YOLO"""

    yaml_content = {
        'path': os.path.abspath(output_path),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'person',
            1: 'shorts',
            2: 'mini_skirt',
            3: 'crop_top',
            4: 'tank_top',
            5: 'sportswear',
        }
    }

    yaml_path = os.path.join(output_path, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"Fichier de configuration créé: {yaml_path}")
    return yaml_path


def train_model(dataset_yaml, epochs=50, img_size=640, batch_size=16):
    """
    Entraîne le modèle YOLOv8 sur le dataset préparé
    """
    print("\n" + "="*60)
    print("ENTRAÎNEMENT DU MODÈLE YOLOv8")
    print("="*60)

    # Charger le modèle pré-entraîné
    model = YOLO('yolov8n.pt')

    # Entraîner
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

    # Copier le meilleur modèle
    best_model_path = os.path.join(MODEL_OUTPUT, 'dresscode_detector', 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, 'dresscode_model.pt')
        print(f"\nModèle entraîné sauvegardé: dresscode_model.pt")

    return results


def download_sample_dataset():
    """
    Télécharge un petit échantillon pour tester
    (DeepFashion2 complet nécessite une demande d'accès)
    """
    print("\n" + "="*60)
    print("INSTRUCTIONS POUR OBTENIR DEEPFASHION2")
    print("="*60)
    print("""
1. Visitez: https://github.com/switchablenorms/DeepFashion2
2. Remplissez le formulaire de demande d'accès
3. Téléchargez le dataset une fois approuvé
4. Extrayez dans le dossier 'deepfashion2/' avec la structure:

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

5. Relancez ce script pour convertir et entraîner
    """)


def main():
    print("="*60)
    print("PRÉPARATION DE L'ENTRAÎNEMENT - DÉTECTION DRESS CODE")
    print("="*60)

    # Vérifier si le dataset existe
    if not os.path.exists(DEEPFASHION2_PATH):
        print(f"\nDataset non trouvé: {DEEPFASHION2_PATH}")
        download_sample_dataset()

        # Créer un dataset minimal pour test
        print("\nCréation d'un dataset minimal pour démonstration...")
        setup_directories()

        # Sans données réelles, on ne peut pas continuer
        print("\nPour un entraînement réel, téléchargez DeepFashion2")
        return

    # Créer les dossiers
    setup_directories()

    # Convertir les données
    print("\nConversion des données d'entraînement...")
    train_count = convert_deepfashion2_to_yolo(DEEPFASHION2_PATH, OUTPUT_PATH, "train")
    print(f"Images d'entraînement converties: {train_count}")

    print("\nConversion des données de validation...")
    val_count = convert_deepfashion2_to_yolo(DEEPFASHION2_PATH, OUTPUT_PATH, "validation")
    print(f"Images de validation converties: {val_count}")

    if train_count == 0:
        print("\nAucune donnée convertie. Vérifiez le chemin du dataset.")
        return

    # Créer le fichier YAML
    yaml_path = create_dataset_yaml(OUTPUT_PATH, num_classes=6)

    # Demander confirmation pour l'entraînement
    print("\n" + "="*60)
    print("PRÊT POUR L'ENTRAÎNEMENT")
    print("="*60)
    print(f"- Images d'entraînement: {train_count}")
    print(f"- Images de validation: {val_count}")
    print(f"- Configuration: {yaml_path}")

    response = input("\nLancer l'entraînement? (o/n): ")

    if response.lower() == 'o':
        train_model(yaml_path, epochs=50, batch_size=16)
        print("\n" + "="*60)
        print("ENTRAÎNEMENT TERMINÉ")
        print("="*60)
        print("Utilisez 'dresscode_model.pt' dans detector.py")
        print("Modifiez la ligne: self.model = YOLO('dresscode_model.pt')")
    else:
        print("Entraînement annulé.")


if __name__ == "__main__":
    main()
