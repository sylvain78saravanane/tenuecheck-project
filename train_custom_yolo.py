"""
Entrainement YOLOv8 sur dataset personnalise
Detection des vetements interdits ENSITECH

Classes:
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
import sys
from pathlib import Path

# Configuration de l'entrainement
CONFIG = {
    "model": "yolov8n.pt",      # Modele de base (nano = rapide)
    "epochs": 100,              # Nombre d'epochs
    "batch_size": 16,           # Taille du batch (reduire si manque de RAM)
    "img_size": 640,            # Taille des images
    "patience": 15,             # Early stopping
    "dataset_yaml": "dataset_dresscode/dataset.yaml",
    "project": "runs/dresscode",
    "name": "ensitech_model",
}

# Classes de vetements interdits
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


def check_dataset():
    """Verifie que le dataset est pret pour l'entrainement"""
    print("=" * 60)
    print("VERIFICATION DU DATASET")
    print("=" * 60)

    dataset_dir = "dataset_dresscode"
    issues = []

    # Verifier la structure
    required_dirs = [
        f"{dataset_dir}/images/train",
        f"{dataset_dir}/images/val",
        f"{dataset_dir}/labels/train",
        f"{dataset_dir}/labels/val",
    ]

    for d in required_dirs:
        if not os.path.exists(d):
            issues.append(f"Dossier manquant: {d}")

    # Compter les images et labels
    stats = {"train": {"images": 0, "labels": 0}, "val": {"images": 0, "labels": 0}}

    for split in ["train", "val"]:
        img_dir = f"{dataset_dir}/images/{split}"
        lbl_dir = f"{dataset_dir}/labels/{split}"

        if os.path.exists(img_dir):
            images = [f for f in os.listdir(img_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            stats[split]["images"] = len(images)

        if os.path.exists(lbl_dir):
            labels = [f for f in os.listdir(lbl_dir) if f.endswith('.txt')]
            stats[split]["labels"] = len(labels)

    # Afficher les statistiques
    print(f"\nImages d'entrainement: {stats['train']['images']}")
    print(f"Labels d'entrainement: {stats['train']['labels']}")
    print(f"Images de validation:  {stats['val']['images']}")
    print(f"Labels de validation:  {stats['val']['labels']}")

    # Verifier les problemes
    if stats['train']['images'] == 0:
        issues.append("Aucune image d'entrainement trouvee")

    if stats['train']['labels'] == 0:
        issues.append("Aucun label d'entrainement trouve")

    if stats['train']['images'] != stats['train']['labels']:
        issues.append(f"Nombre d'images ({stats['train']['images']}) != labels ({stats['train']['labels']}) pour train")

    # Verifier le fichier YAML
    yaml_path = f"{dataset_dir}/dataset.yaml"
    if not os.path.exists(yaml_path):
        issues.append(f"Fichier de configuration manquant: {yaml_path}")

    # Compter les annotations par classe
    print("\n" + "-" * 40)
    print("Annotations par classe:")
    print("-" * 40)

    class_counts = {i: 0 for i in range(len(CLASSES))}

    for split in ["train", "val"]:
        lbl_dir = f"{dataset_dir}/labels/{split}"
        if os.path.exists(lbl_dir):
            for lbl_file in os.listdir(lbl_dir):
                if not lbl_file.endswith('.txt'):
                    continue
                with open(os.path.join(lbl_dir, lbl_file), 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                class_id = int(parts[0])
                                if 0 <= class_id < len(CLASSES):
                                    class_counts[class_id] += 1
                            except ValueError:
                                pass

    total_annotations = 0
    for class_id, count in class_counts.items():
        total_annotations += count
        status = "OK" if count >= 50 else "FAIBLE" if count > 0 else "VIDE"
        print(f"  {class_id}: {CLASSES[class_id]:<15} {count:>5} annotations [{status}]")

    print(f"\nTotal: {total_annotations} annotations")

    if total_annotations == 0:
        issues.append("Aucune annotation trouvee dans le dataset")
    elif total_annotations < 100:
        print("\n[AVERTISSEMENT] Moins de 100 annotations. Le modele risque de mal generaliser.")

    # Afficher les problemes
    if issues:
        print("\n" + "=" * 60)
        print("PROBLEMES DETECTES:")
        print("=" * 60)
        for issue in issues:
            print(f"  [!] {issue}")
        return False

    print("\n[OK] Dataset pret pour l'entrainement")
    return True


def train_model():
    """Entraine le modele YOLOv8"""
    print("\n" + "=" * 60)
    print("ENTRAINEMENT YOLOV8 - ENSITECH DRESS CODE")
    print("=" * 60)

    # Importer ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERREUR] ultralytics non installe. Executez: pip install ultralytics")
        return None

    # Verifier le GPU
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Device: {device}")

    if device == "cpu":
        print("[INFO] Entrainement sur CPU (plus lent)")
        print("[INFO] Pour utiliser le GPU, installez CUDA et pytorch-cuda")

    # Charger le modele de base
    print(f"\n[INFO] Chargement du modele de base: {CONFIG['model']}")
    model = YOLO(CONFIG["model"])

    # Verifier le chemin du dataset
    yaml_path = CONFIG["dataset_yaml"]
    if not os.path.exists(yaml_path):
        print(f"[ERREUR] Fichier non trouve: {yaml_path}")
        return None

    # Lancer l'entrainement
    print(f"\n[INFO] Demarrage de l'entrainement...")
    print(f"  - Epochs: {CONFIG['epochs']}")
    print(f"  - Batch size: {CONFIG['batch_size']}")
    print(f"  - Image size: {CONFIG['img_size']}")
    print(f"  - Early stopping patience: {CONFIG['patience']}")
    print()

    try:
        results = model.train(
            data=yaml_path,
            epochs=CONFIG["epochs"],
            batch=CONFIG["batch_size"],
            imgsz=CONFIG["img_size"],
            patience=CONFIG["patience"],
            project=CONFIG["project"],
            name=CONFIG["name"],
            save=True,
            plots=True,
            verbose=True,
            exist_ok=True,
        )

        # Trouver le meilleur modele
        best_model_path = Path(CONFIG["project"]) / CONFIG["name"] / "weights" / "best.pt"

        if best_model_path.exists():
            # Copier vers le dossier principal
            import shutil
            output_path = "dresscode_yolo.pt"
            shutil.copy(best_model_path, output_path)
            print(f"\n[OK] Modele sauvegarde: {output_path}")
            return output_path
        else:
            print(f"\n[ERREUR] Modele best.pt non trouve")
            return None

    except Exception as e:
        print(f"\n[ERREUR] Entrainement echoue: {e}")
        return None


def evaluate_model(model_path):
    """Evalue le modele entraine"""
    print("\n" + "=" * 60)
    print("EVALUATION DU MODELE")
    print("=" * 60)

    from ultralytics import YOLO

    model = YOLO(model_path)

    # Evaluer sur le set de validation
    results = model.val(data=CONFIG["dataset_yaml"])

    print(f"\n[INFO] Resultats:")
    print(f"  - mAP50: {results.box.map50:.4f}")
    print(f"  - mAP50-95: {results.box.map:.4f}")

    return results


def test_on_image(model_path, image_path):
    """Teste le modele sur une image"""
    from ultralytics import YOLO
    import cv2

    model = YOLO(model_path)
    results = model(image_path)

    # Afficher les resultats
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"class_{cls_id}"
            print(f"  Detecte: {cls_name} (confiance: {conf:.2%})")

    # Sauvegarder l'image annotee
    annotated = results[0].plot()
    output_path = "test_detection.jpg"
    cv2.imwrite(output_path, annotated)
    print(f"\nImage annotee sauvegardee: {output_path}")


def print_usage():
    """Affiche l'aide"""
    print("""
UTILISATION:

  python train_custom_yolo.py [commande]

COMMANDES:

  check     - Verifie que le dataset est pret
  train     - Lance l'entrainement
  eval      - Evalue le modele entraine
  test      - Teste sur une image (necessite le chemin de l'image)

EXEMPLES:

  # Verifier le dataset
  python train_custom_yolo.py check

  # Entrainer le modele
  python train_custom_yolo.py train

  # Evaluer le modele
  python train_custom_yolo.py eval

  # Tester sur une image
  python train_custom_yolo.py test mon_image.jpg

PREPARATION DU DATASET:

  Avant de lancer l'entrainement, vous devez:
  1. Executer: python prepare_custom_dataset.py
  2. Ajouter des images dans dataset_dresscode/images/train/
  3. Annoter les images (LabelImg ou Roboflow)
  4. Verifier avec: python train_custom_yolo.py check
""")


def main():
    if len(sys.argv) < 2:
        print_usage()
        return

    command = sys.argv[1].lower()

    if command == "check":
        check_dataset()

    elif command == "train":
        # Verifier d'abord le dataset
        if not check_dataset():
            print("\n[!] Corrigez les problemes avant de lancer l'entrainement")
            print("[!] Executez d'abord: python prepare_custom_dataset.py")
            return

        # Confirmer
        print("\n" + "-" * 60)
        response = input("Lancer l'entrainement? (o/n): ")
        if response.lower() != 'o':
            print("Entrainement annule.")
            return

        # Entrainer
        model_path = train_model()

        if model_path:
            print("\n" + "=" * 60)
            print("ENTRAINEMENT TERMINE AVEC SUCCES")
            print("=" * 60)
            print(f"\nModele sauvegarde: {model_path}")
            print("\nPour utiliser ce modele:")
            print("1. Le fichier 'dresscode_yolo.pt' est pret")
            print("2. Modifiez detector.py pour charger ce modele:")
            print("   self.model = YOLO('dresscode_yolo.pt')")
            print("3. Lancez l'application: python app.py")

    elif command == "eval":
        model_path = "dresscode_yolo.pt"
        if not os.path.exists(model_path):
            print(f"[ERREUR] Modele non trouve: {model_path}")
            print("Lancez d'abord l'entrainement: python train_custom_yolo.py train")
            return
        evaluate_model(model_path)

    elif command == "test":
        if len(sys.argv) < 3:
            print("[ERREUR] Specifiez le chemin de l'image")
            print("Exemple: python train_custom_yolo.py test image.jpg")
            return

        model_path = "dresscode_yolo.pt"
        image_path = sys.argv[2]

        if not os.path.exists(model_path):
            print(f"[ERREUR] Modele non trouve: {model_path}")
            return

        if not os.path.exists(image_path):
            print(f"[ERREUR] Image non trouvee: {image_path}")
            return

        test_on_image(model_path, image_path)

    else:
        print(f"[ERREUR] Commande inconnue: {command}")
        print_usage()


if __name__ == "__main__":
    main()
