"""
Script de téléchargement et d'entraînement automatique
Utilise DeepFashion2 depuis Hugging Face ou Roboflow
"""

import os
import subprocess
import sys


def install_dependencies():
    """Installe les dépendances nécessaires"""
    deps = ["datasets", "huggingface_hub", "roboflow"]
    for dep in deps:
        subprocess.check_call([sys.executable, "-m", "pip", "install", dep, "-q"])


def download_from_huggingface():
    """
    Télécharge DeepFashion2 depuis Hugging Face
    """
    print("="*60)
    print("TÉLÉCHARGEMENT DEPUIS HUGGING FACE")
    print("="*60)

    try:
        from datasets import load_dataset

        print("Téléchargement du dataset (peut prendre du temps)...")
        dataset = load_dataset("sahirp/deepfashion2", split="train")

        # Créer les dossiers
        os.makedirs("deepfashion2/train/images", exist_ok=True)
        os.makedirs("deepfashion2/train/labels", exist_ok=True)

        print(f"Dataset chargé: {len(dataset)} images")
        return dataset

    except Exception as e:
        print(f"Erreur Hugging Face: {e}")
        return None


def download_from_roboflow():
    """
    Télécharge un subset de DeepFashion2 depuis Roboflow
    (Version plus petite mais directement au format YOLO)
    """
    print("="*60)
    print("TÉLÉCHARGEMENT DEPUIS ROBOFLOW")
    print("="*60)

    try:
        from roboflow import Roboflow

        # Dataset public DeepFashion2 sur Roboflow
        # Note: Vous pouvez créer un compte gratuit sur roboflow.com
        print("""
Pour télécharger depuis Roboflow:
1. Créez un compte gratuit sur https://roboflow.com
2. Obtenez votre API key depuis les paramètres
3. Entrez-la ci-dessous
        """)

        api_key = input("Entrez votre API key Roboflow (ou 'skip' pour passer): ")

        if api_key.lower() == 'skip':
            return None

        rf = Roboflow(api_key=api_key)
        project = rf.workspace().project("deepfashion2-m-10k")
        dataset = project.version(2).download("yolov8")

        print(f"Dataset téléchargé dans: {dataset.location}")
        return dataset.location

    except Exception as e:
        print(f"Erreur Roboflow: {e}")
        return None


def create_custom_dataset():
    """
    Crée un dataset personnalisé en capturant des images
    """
    print("="*60)
    print("CRÉATION D'UN DATASET PERSONNALISÉ")
    print("="*60)

    print("""
Option alternative: Créer votre propre dataset

1. Prenez des photos de personnes portant des vêtements interdits:
   - Shorts, bermudas
   - Mini-jupes
   - Crop tops
   - Casquettes, bonnets
   - Tongs
   - Jeans troués

2. Placez les images dans: dataset_custom/images/

3. Annotez-les avec un outil comme:
   - Label Studio (gratuit): https://labelstud.io
   - Roboflow (gratuit): https://roboflow.com
   - CVAT (gratuit): https://cvat.ai

4. Exportez au format YOLO et placez dans: dataset_custom/labels/
    """)

    os.makedirs("dataset_custom/images/train", exist_ok=True)
    os.makedirs("dataset_custom/images/val", exist_ok=True)
    os.makedirs("dataset_custom/labels/train", exist_ok=True)
    os.makedirs("dataset_custom/labels/val", exist_ok=True)

    print("Dossiers créés pour votre dataset personnalisé.")


def download_fashion_mnist_alternative():
    """
    Télécharge Fashion-MNIST comme alternative simple
    (Moins adapté mais disponible immédiatement)
    """
    print("="*60)
    print("TÉLÉCHARGEMENT FASHION-MNIST (Alternative)")
    print("="*60)

    try:
        import urllib.request
        import gzip
        import numpy as np
        from PIL import Image

        base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
        files = {
            "train_images": "train-images-idx3-ubyte.gz",
            "train_labels": "train-labels-idx1-ubyte.gz",
        }

        os.makedirs("fashion_mnist", exist_ok=True)

        for name, filename in files.items():
            url = base_url + filename
            filepath = os.path.join("fashion_mnist", filename)

            if not os.path.exists(filepath):
                print(f"Téléchargement {filename}...")
                urllib.request.urlretrieve(url, filepath)

        print("Fashion-MNIST téléchargé!")
        print("Note: Ce dataset contient des images 28x28 de vêtements simples")
        print("Il est moins adapté que DeepFashion2 pour la détection réelle")

        return "fashion_mnist"

    except Exception as e:
        print(f"Erreur: {e}")
        return None


def use_pretrained_fashion_model():
    """
    Télécharge un modèle pré-entraîné sur la mode
    """
    print("="*60)
    print("UTILISATION D'UN MODÈLE PRÉ-ENTRAÎNÉ")
    print("="*60)

    try:
        from ultralytics import YOLO

        # YOLOv8 avec classes COCO inclut déjà "person"
        # On peut utiliser un modèle plus grand pour de meilleures performances
        print("Téléchargement de YOLOv8 medium...")
        model = YOLO('yolov8m.pt')

        print("Modèle téléchargé: yolov8m.pt")
        print("Ce modèle détecte déjà les personnes avec précision.")
        print("")
        print("Pour améliorer la détection des vêtements spécifiques,")
        print("il faudra fine-tuner avec un dataset de vêtements.")

        return model

    except Exception as e:
        print(f"Erreur: {e}")
        return None


def main():
    print("="*60)
    print("ENSITECH - PRÉPARATION DU MODÈLE DE DÉTECTION")
    print("="*60)
    print("")
    print("Options disponibles:")
    print("1. Télécharger depuis Hugging Face (DeepFashion2)")
    print("2. Télécharger depuis Roboflow (subset YOLO-ready)")
    print("3. Créer un dataset personnalisé")
    print("4. Utiliser un modèle pré-entraîné plus performant")
    print("5. Télécharger Fashion-MNIST (alternative simple)")
    print("")

    choice = input("Choisissez une option (1-5): ")

    if choice == "1":
        print("\nInstallation des dépendances...")
        install_dependencies()
        download_from_huggingface()

    elif choice == "2":
        print("\nInstallation des dépendances...")
        install_dependencies()
        download_from_roboflow()

    elif choice == "3":
        create_custom_dataset()

    elif choice == "4":
        use_pretrained_fashion_model()

    elif choice == "5":
        download_fashion_mnist_alternative()

    else:
        print("Option invalide")
        return

    print("\n" + "="*60)
    print("TERMINÉ")
    print("="*60)


if __name__ == "__main__":
    main()
