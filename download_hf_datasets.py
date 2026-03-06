"""
Script de telechargement et filtrage des datasets Fashionpedia et DeepFashion2
depuis HuggingFace pour le projet TenueCheck.

Categories cibles:
  0 - couvre_chef   (casquette, chapeau, bonnet, bandana, couvre-chef)

Datasets sources:
  - Fashionpedia (detection-datasets/fashionpedia)
  - DeepFashion2 (sahirp/deepfashion2)

Modele: YOLOv8n (nano) - utilise le transfer learning depuis les poids pre-entraines COCO
"""

import os
import json
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# =====================================================================
# MAPPING DES CATEGORIES
# =====================================================================

# Notre classe cible pour la detection de code vestimentaire
# Chaque cle est l'ID de classe YOLO, chaque valeur est le nom lisible
TARGET_CLASSES = {
    0: "couvre_chef",
}

# --- Fashionpedia: 46 categories (index 0-45) ---
# Dictionnaire complet des categories du dataset Fashionpedia
# On n'utilise que certaines d'entre elles (voir FASHIONPEDIA_TO_TARGET)
FASHIONPEDIA_CATEGORIES = {
    0: "shirt, blouse",
    1: "top, t-shirt, sweatshirt",
    2: "sweater",
    3: "cardigan",
    4: "jacket",
    5: "vest",
    6: "pants",
    7: "shorts",
    8: "skirt",
    9: "coat",
    10: "dress",
    11: "jumpsuit",
    12: "cape",
    13: "glasses",
    14: "hat",
    15: "headband, head covering, hair accessory",
    16: "tie",
    17: "glove",
    18: "watch",
    19: "belt",
    20: "leg warmer",
    21: "tights, stockings",
    22: "sock",
    23: "shoe",
    24: "bag, wallet",
    25: "scarf",
    26: "umbrella",
    27: "hood",
    28: "collar",
    29: "lapel",
    30: "epaulette",
    31: "sleeve",
    32: "pocket",
    33: "neckline",
    34: "buckle",
    35: "zipper",
    36: "applique",
    37: "bead",
    38: "bow",
    39: "flower",
    40: "fringe",
    41: "ribbon",
    42: "rivet",
    43: "ruffle",
    44: "sequin",
    45: "tassel",
}

# Mapping Fashionpedia -> nos classes cibles
# Cle = ID categorie Fashionpedia, Valeur = ID classe cible TenueCheck
# Seules les categories pertinentes pour notre detection sont conservees
FASHIONPEDIA_TO_TARGET = {
    14: 0,  # hat -> couvre_chef
    15: 0,  # headband, head covering, hair accessory -> couvre_chef
    27: 0,  # hood -> couvre_chef
}

# --- DeepFashion2: 13 categories (index 1-13) ---
# Second dataset utilise pour enrichir les donnees d'entrainement
DEEPFASHION2_CATEGORIES = {
    1: "short_sleeve_top",
    2: "long_sleeve_top",
    3: "short_sleeve_outwear",
    4: "long_sleeve_outwear",
    5: "vest",
    6: "sling",
    7: "shorts",
    8: "trousers",
    9: "skirt",
    10: "short_sleeve_dress",
    11: "long_sleeve_dress",
    12: "vest_dress",
    13: "sling_dress",
}

# Mapping DeepFashion2 -> nos classes cibles
# Meme principe que pour Fashionpedia : on ne garde que les categories utiles
DEEPFASHION2_TO_TARGET = {
    # Pas de couvre-chefs dans DeepFashion2
}

# =====================================================================
# CONFIGURATION
# =====================================================================

# Repertoire de sortie pour le dataset au format YOLO
OUTPUT_DIR = "dataset_tenuecheck"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")   # Sous-dossiers : train/, val/, test/
LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")    # Sous-dossiers : train/, val/, test/

# Ratios de repartition du dataset (80% entrainement, 15% validation, 5% test)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.15
TEST_RATIO = 0.05

# Nombre maximum d'images a telecharger par source (pour limiter le temps de telechargement)
MAX_IMAGES_PER_SOURCE = 2000


def setup_directories():
    """
    Cree l'arborescence du dataset au format YOLO.
    Structure :
      dataset_tenuecheck/
        images/
          train/ val/ test/
        labels/
          train/ val/ test/
    """
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(IMAGES_DIR, split), exist_ok=True)
        os.makedirs(os.path.join(LABELS_DIR, split), exist_ok=True)
    print(f"Arborescence creee dans {OUTPUT_DIR}/")


def download_fashionpedia():
    """
    Telecharge et filtre le dataset Fashionpedia depuis HuggingFace.

    Etapes :
    1. Chargement du dataset complet (~45000 images)
    2. Filtrage : on ne garde que les images contenant au moins une categorie cible
       (hat, headband, hood, top, shorts)
    3. Conversion des annotations du format COCO [x_min, y_min, w, h]
       vers le format YOLO [x_center, y_center, w, h] normalise (valeurs entre 0 et 1)
    4. Sauvegarde des images en JPEG et des labels en .txt (un fichier par image)

    Retourne la liste des noms de fichiers sauvegardes.
    """
    print("\n" + "=" * 60)
    print("TELECHARGEMENT FASHIONPEDIA (HuggingFace)")
    print("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        print("Installation de la librairie 'datasets'...")
        os.system("pip install datasets")
        from datasets import load_dataset

    print("Chargement du dataset Fashionpedia...")
    ds = load_dataset("detection-datasets/fashionpedia", split="train")

    filtered_samples = []
    target_categories = set(FASHIONPEDIA_TO_TARGET.keys())

    print(f"Filtrage des images contenant les categories: {[FASHIONPEDIA_CATEGORIES[c] for c in target_categories]}")

    # Parcours du dataset pour ne garder que les images pertinentes
    for idx in tqdm(range(len(ds)), desc="Filtrage Fashionpedia"):
        sample = ds[idx]
        objects = sample["objects"]
        categories = objects["category"]

        # Verifier si au moins une annotation correspond a nos classes cibles
        has_target = any(cat in target_categories for cat in categories)
        if has_target:
            filtered_samples.append((idx, sample))

        # Arreter des qu'on a assez d'images
        if len(filtered_samples) >= MAX_IMAGES_PER_SOURCE:
            break

    print(f"\n{len(filtered_samples)} images filtrees depuis Fashionpedia")

    # Conversion et sauvegarde au format YOLO
    saved = []
    for i, (idx, sample) in enumerate(tqdm(filtered_samples, desc="Sauvegarde Fashionpedia")):
        image = sample["image"]
        objects = sample["objects"]
        img_w, img_h = image.size  # Dimensions de l'image pour normaliser les coordonnees

        yolo_annotations = []
        categories = objects["category"]
        bboxes = objects["bbox"]

        for cat, bbox in zip(categories, bboxes):
            # Ignorer les categories qui ne font pas partie de nos cibles
            if cat not in target_categories:
                continue

            target_class = FASHIONPEDIA_TO_TARGET[cat]

            # Conversion COCO -> YOLO :
            #   COCO : [x_min, y_min, largeur, hauteur] en pixels
            #   YOLO : [x_centre, y_centre, largeur, hauteur] normalise entre 0 et 1
            x_min, y_min, w, h = bbox
            x_center = (x_min + w / 2) / img_w
            y_center = (y_min + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            # Clamper les valeurs entre 0 et 1 (securite contre les annotations hors limites)
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w_norm = max(0, min(1, w_norm))
            h_norm = max(0, min(1, h_norm))

            # Filtrer les bounding boxes trop petites (artefacts d'annotation)
            if w_norm > 0.01 and h_norm > 0.01:
                yolo_annotations.append(f"{target_class} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        # Sauvegarder seulement si l'image a au moins une annotation valide
        if yolo_annotations:
            # Sauvegarde de l'image en JPEG (prefixe "fp_" = Fashionpedia)
            img_filename = f"fp_{i:05d}.jpg"
            img_path = os.path.join(IMAGES_DIR, "train", img_filename)
            image.convert("RGB").save(img_path, "JPEG", quality=95)

            # Sauvegarde du label YOLO (une ligne par objet detecte)
            label_filename = f"fp_{i:05d}.txt"
            label_path = os.path.join(LABELS_DIR, "train", label_filename)
            with open(label_path, "w") as f:
                f.write("\n".join(yolo_annotations))

            saved.append(img_filename)

    print(f"{len(saved)} images sauvegardees depuis Fashionpedia")
    return saved


def download_deepfashion2():
    """
    Telecharge et filtre le dataset DeepFashion2 depuis HuggingFace.

    Meme logique que download_fashionpedia() mais adapte au format
    d'annotation different de DeepFashion2 :
    - Les annotations sont stockees dans un champ JSON par image
    - Les bounding boxes sont au format [x1, y1, x2, y2] (coins opposes)
      au lieu du format COCO [x_min, y_min, w, h]

    Ce dataset est utilise pour enrichir les classes crop_top et survetement.

    Retourne la liste des noms de fichiers sauvegardes.
    """
    print("\n" + "=" * 60)
    print("TELECHARGEMENT DEEPFASHION2 (HuggingFace)")
    print("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        os.system("pip install datasets")
        from datasets import load_dataset

    print("Chargement du dataset DeepFashion2...")
    print("(Ce dataset est volumineux, le telechargement peut prendre du temps)")

    # Tentative de chargement du split train, puis validation en fallback
    try:
        ds = load_dataset("sahirp/deepfashion2", split="train")
    except Exception as e:
        print(f"Erreur lors du chargement de DeepFashion2: {e}")
        print("Tentative avec le dataset de validation...")
        try:
            ds = load_dataset("sahirp/deepfashion2", split="validation")
        except Exception as e2:
            print(f"Impossible de charger DeepFashion2: {e2}")
            print("DeepFashion2 necessite parfois une authentification HuggingFace.")
            print("Executez: huggingface-cli login")
            return []

    target_categories = set(DEEPFASHION2_TO_TARGET.keys())
    filtered_samples = []

    print(f"Filtrage pour les categories: {[DEEPFASHION2_CATEGORIES[c] for c in target_categories]}")

    # Parcours et filtrage des images contenant nos categories cibles
    for idx in tqdm(range(len(ds)), desc="Filtrage DeepFashion2"):
        sample = ds[idx]

        # Le format d'annotation de DeepFashion2 varie selon le loader HuggingFace
        # On gere les deux cas : annotation JSON structuree ou format simplifie
        try:
            if "annotation" in sample:
                # Format structure : dictionnaire d'items avec category_id et bounding_box
                annotation = sample["annotation"]
                if isinstance(annotation, str):
                    annotation = json.loads(annotation)

                for item_key in annotation:
                    if isinstance(annotation[item_key], dict):
                        cat_id = annotation[item_key].get("category_id", -1)
                        if cat_id in target_categories:
                            filtered_samples.append((idx, sample))
                            break

            elif "image" in sample:
                # Format simplifie : un seul label par image
                if hasattr(sample, "get"):
                    label = sample.get("label", sample.get("category", -1))
                    if label in target_categories:
                        filtered_samples.append((idx, sample))

        except (json.JSONDecodeError, KeyError, TypeError):
            # Ignorer les echantillons avec des annotations corrompues
            continue

        if len(filtered_samples) >= MAX_IMAGES_PER_SOURCE:
            break

    print(f"\n{len(filtered_samples)} images filtrees depuis DeepFashion2")

    # Conversion et sauvegarde au format YOLO
    saved = []
    for i, (idx, sample) in enumerate(tqdm(filtered_samples, desc="Sauvegarde DeepFashion2")):
        try:
            image = sample["image"]
            img_w, img_h = image.size

            yolo_annotations = []

            if "annotation" in sample:
                annotation = sample["annotation"]
                if isinstance(annotation, str):
                    annotation = json.loads(annotation)

                for item_key in annotation:
                    if not isinstance(annotation[item_key], dict):
                        continue
                    item = annotation[item_key]
                    cat_id = item.get("category_id", -1)

                    if cat_id not in target_categories:
                        continue

                    target_class = DEEPFASHION2_TO_TARGET[cat_id]
                    bbox = item.get("bounding_box", [])

                    if len(bbox) == 4:
                        # Conversion DeepFashion2 -> YOLO :
                        #   DeepFashion2 : [x1, y1, x2, y2] (coins opposes en pixels)
                        #   YOLO : [x_centre, y_centre, largeur, hauteur] normalise
                        x1, y1, x2, y2 = bbox
                        x_center = ((x1 + x2) / 2) / img_w
                        y_center = ((y1 + y2) / 2) / img_h
                        w_norm = (x2 - x1) / img_w
                        h_norm = (y2 - y1) / img_h

                        # Clamper entre 0 et 1
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        w_norm = max(0, min(1, w_norm))
                        h_norm = max(0, min(1, h_norm))

                        # Filtrer les bbox trop petites
                        if w_norm > 0.01 and h_norm > 0.01:
                            yolo_annotations.append(
                                f"{target_class} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                            )

            # Sauvegarder si au moins une annotation valide (prefixe "df2_" = DeepFashion2)
            if yolo_annotations:
                img_filename = f"df2_{i:05d}.jpg"
                img_path = os.path.join(IMAGES_DIR, "train", img_filename)
                image.convert("RGB").save(img_path, "JPEG", quality=95)

                label_filename = f"df2_{i:05d}.txt"
                label_path = os.path.join(LABELS_DIR, "train", label_filename)
                with open(label_path, "w") as f:
                    f.write("\n".join(yolo_annotations))

                saved.append(img_filename)

        except Exception as e:
            # Ignorer les images problematiques (format inattendu, image corrompue, etc.)
            continue

    print(f"{len(saved)} images sauvegardees depuis DeepFashion2")
    return saved


def split_dataset():
    """
    Repartit les images du dossier train/ en train/val/test selon les ratios definis.

    Toutes les images sont d'abord placees dans train/ lors du telechargement,
    puis cette fonction en deplace une partie vers val/ et test/.
    Le melange aleatoire (shuffle) assure une repartition equilibree des classes.
    """
    print("\n" + "=" * 60)
    print("REPARTITION TRAIN / VAL / TEST")
    print("=" * 60)

    train_images_dir = os.path.join(IMAGES_DIR, "train")
    train_labels_dir = os.path.join(LABELS_DIR, "train")

    # Lister et melanger toutes les images
    all_images = [f for f in os.listdir(train_images_dir) if f.endswith((".jpg", ".png"))]
    random.shuffle(all_images)

    # Calculer les tailles de chaque split
    n = len(all_images)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_files = all_images[:n_train]
    val_files = all_images[n_train:n_train + n_val]
    test_files = all_images[n_train + n_val:]  # Le reste va dans test

    # Deplacer les fichiers images + labels correspondants vers val/ et test/
    for split, files in [("val", val_files), ("test", test_files)]:
        for img_file in files:
            # Le fichier label a le meme nom que l'image mais avec extension .txt
            label_file = img_file.rsplit(".", 1)[0] + ".txt"

            # Deplacer l'image
            src_img = os.path.join(train_images_dir, img_file)
            dst_img = os.path.join(IMAGES_DIR, split, img_file)
            if os.path.exists(src_img):
                shutil.move(src_img, dst_img)

            # Deplacer le label YOLO correspondant
            src_label = os.path.join(train_labels_dir, label_file)
            dst_label = os.path.join(LABELS_DIR, split, label_file)
            if os.path.exists(src_label):
                shutil.move(src_label, dst_label)

    print(f"Train: {len(train_files)} images")
    print(f"Val:   {len(val_files)} images")
    print(f"Test:  {len(test_files)} images")
    print(f"Total: {n} images")


def create_yaml_config():
    """
    Cree le fichier dataset.yaml necessaire a YOLOv8 pour l'entrainement.

    Ce fichier YAML definit :
    - Le chemin absolu vers le dataset
    - Les chemins relatifs des splits train/val/test
    - Le nombre de classes (nc) et leurs noms

    YOLOv8 lit ce fichier pour savoir ou trouver les images et labels.
    """
    yaml_content = f"""# TenueCheck Dataset - Configuration YOLOv8n
# Categories: couvre_chef

path: {os.path.abspath(OUTPUT_DIR)}
train: images/train
val: images/val
test: images/test

# Classes
nc: {len(TARGET_CLASSES)}
names:
  0: couvre_chef
"""

    yaml_path = os.path.join(OUTPUT_DIR, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\nFichier YAML cree: {yaml_path}")
    return yaml_path


def print_mapping_summary():
    """Affiche le resume du mapping des categories source -> classes cibles."""
    print("\n" + "=" * 60)
    print("MAPPING DES CATEGORIES")
    print("=" * 60)

    print("\n--- Classes cibles TenueCheck ---")
    for class_id, class_name in TARGET_CLASSES.items():
        print(f"  {class_id}: {class_name}")

    print("\n--- Fashionpedia -> TenueCheck ---")
    for fp_cat, target_cls in FASHIONPEDIA_TO_TARGET.items():
        print(f"  [{fp_cat}] {FASHIONPEDIA_CATEGORIES[fp_cat]:40s} -> [{target_cls}] {TARGET_CLASSES[target_cls]}")

    print("\n--- DeepFashion2 -> TenueCheck ---")
    for df_cat, target_cls in DEEPFASHION2_TO_TARGET.items():
        print(f"  [{df_cat}] {DEEPFASHION2_CATEGORIES[df_cat]:40s} -> [{target_cls}] {TARGET_CLASSES[target_cls]}")

    print("\n--- Note ---")
    print("  Seule la classe couvre_chef est entrainee pour l'instant.")
    print("  Les classes crop_top et survetement seront ajoutees plus tard.")


def train_yolov8n(yaml_path, device="cpu"):
    """
    Lance l'entrainement YOLOv8 nano avec transfer learning.

    Transfer learning : on charge yolov8n.pt (poids pre-entraines sur COCO, 80 classes)
    puis on affine (fine-tune) le modele sur notre classe couvre_chef.
    Les couches basses du reseau (detection de formes, textures) sont deja entraines,
    seules les couches hautes sont reajustees pour nos categories.

    Parametres d'entrainement :
    - epochs=50 : nombre de passages sur le dataset complet
    - patience=20 : arret anticipe si pas d'amelioration pendant 20 epochs (early stopping)
    - imgsz=640 : redimensionnement des images en 640x640 pixels
    - batch=16 : nombre d'images traitees simultanement par le GPU
    - optimizer=AdamW : optimiseur avec weight decay pour eviter le surapprentissage
    - lr0=0.001 : taux d'apprentissage initial
    - lrf=0.01 : facteur de reduction du learning rate en fin d'entrainement
    - augment=True : augmentation de donnees activee (voir parametres ci-dessous)

    Augmentations de donnees (pour enrichir artificiellement le dataset) :
    - hsv_h/s/v : variations de teinte, saturation et luminosite
    - degrees=10 : rotation aleatoire de +/- 10 degres
    - translate=0.1 : translation aleatoire de +/- 10%
    - scale=0.5 : zoom aleatoire
    - fliplr=0.5 : retournement horizontal (50% de chance)
    - mosaic=1.0 : assemblage de 4 images en une seule (augmente la diversite)

    Le meilleur modele (best.pt) est copie en dresscode_yolo.pt a la racine du projet.
    """
    print("\n" + "=" * 60)
    print(f"ENTRAINEMENT YOLOv8n (nano) - device: {device}")
    print("=" * 60)

    from ultralytics import YOLO

    # Chargement du modele pre-entraine sur COCO (transfer learning)
    # yolov8n.pt = YOLOv8 nano (~3.2M parametres, ~6 MB)
    model = YOLO("yolov8n.pt")

    # Lancement de l'entrainement (fine-tuning sur couvre_chef)
    results = model.train(
        data=yaml_path,          # Chemin vers le fichier dataset.yaml
        epochs=50,               # Nombre d'epochs (passages complets sur le dataset)
        imgsz=640,               # Taille des images d'entree (redimensionnees en 640x640)
        batch=16,                # Taille du batch (nombre d'images par iteration)
        name="tenuecheck_yolov8n",  # Nom du run (dossier de sortie dans runs/detect/)
        patience=20,             # Early stopping : arret si pas d'amelioration pendant 20 epochs
        save=True,               # Sauvegarder les poids du modele
        plots=True,              # Generer les graphiques d'entrainement (loss, mAP, etc.)
        device=device,           # Device : "0" pour GPU, "cpu" pour CPU
        workers=4,               # Nombre de threads pour le chargement des donnees
        optimizer="AdamW",       # Optimiseur AdamW (Adam avec weight decay)
        lr0=0.001,               # Learning rate initial
        lrf=0.01,                # Learning rate final = lr0 * lrf (decroissance progressive)
        augment=True,            # Activer l'augmentation de donnees
        hsv_h=0.015,             # Variation de teinte (hue) : +/- 1.5%
        hsv_s=0.7,               # Variation de saturation : +/- 70%
        hsv_v=0.4,               # Variation de luminosite (value) : +/- 40%
        degrees=10,              # Rotation aleatoire : +/- 10 degres
        translate=0.1,           # Translation aleatoire : +/- 10% de l'image
        scale=0.5,               # Zoom aleatoire : +/- 50%
        fliplr=0.5,              # Retournement horizontal : 50% de probabilite
        mosaic=1.0,              # Mosaic augmentation : assemblage de 4 images (100% actif)
    )

    # Recuperer le meilleur modele et le copier a la racine du projet
    best_model = Path("runs/detect/tenuecheck_yolov8n/weights/best.pt")
    if best_model.exists():
        shutil.copy(best_model, "dresscode_yolo.pt")
        print(f"\nModele sauvegarde: dresscode_yolo.pt")
    else:
        print("Modele best.pt non trouve, verifiez runs/detect/tenuecheck_yolov8n/")

    return results


def print_stats():
    """
    Affiche les statistiques du dataset : nombre d'images et d'annotations
    par split (train/val/test) et par classe.
    Utile pour verifier l'equilibre du dataset avant l'entrainement.
    """
    print("\n" + "=" * 60)
    print("STATISTIQUES DU DATASET")
    print("=" * 60)

    class_counts = {cls: 0 for cls in TARGET_CLASSES.values()}

    for split in ["train", "val", "test"]:
        labels_dir = os.path.join(LABELS_DIR, split)
        if not os.path.exists(labels_dir):
            continue

        split_counts = {cls: 0 for cls in TARGET_CLASSES.values()}
        n_images = 0

        # Lire chaque fichier label et compter les annotations par classe
        for label_file in os.listdir(labels_dir):
            if not label_file.endswith(".txt"):
                continue
            n_images += 1

            with open(os.path.join(labels_dir, label_file)) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        # Le premier element de chaque ligne est l'ID de classe
                        cls_id = int(parts[0])
                        cls_name = TARGET_CLASSES.get(cls_id, "unknown")
                        split_counts[cls_name] = split_counts.get(cls_name, 0) + 1
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        print(f"\n  {split.upper()}: {n_images} images")
        for cls, count in split_counts.items():
            print(f"    {cls}: {count} annotations")

    print(f"\n  TOTAL annotations par classe:")
    for cls, count in class_counts.items():
        print(f"    {cls}: {count}")


# =====================================================================
# POINT D'ENTREE - Interface en ligne de commande
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TenueCheck - Telechargement datasets HuggingFace")
    parser.add_argument("action", choices=["download", "train", "all", "stats", "mapping"],
                        help="Action: download (telecharger), train (entrainer), all (tout), stats, mapping")
    parser.add_argument("--max-images", type=int, default=MAX_IMAGES_PER_SOURCE,
                        help=f"Nombre max d'images par source (default: {MAX_IMAGES_PER_SOURCE})")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device pour l'entrainement: '0' (GPU) ou 'cpu'")
    args = parser.parse_args()

    MAX_IMAGES_PER_SOURCE = args.max_images

    # Action "download" ou "all" : telecharger les datasets et preparer le dataset YOLO
    if args.action in ["download", "all"]:
        print_mapping_summary()
        setup_directories()

        # Telecharger depuis Fashionpedia (seule source avec couvre-chefs)
        fp_images = download_fashionpedia()
        # DeepFashion2 n'a pas de couvre-chefs, on skip pour l'instant
        # df2_images = download_deepfashion2()

        # Repartir en train/val/test selon les ratios definis
        split_dataset()

        # Creer le fichier YAML de configuration pour YOLOv8
        yaml_path = create_yaml_config()

        # Afficher les statistiques du dataset constitue
        print_stats()

        print("\n" + "=" * 60)
        print("TELECHARGEMENT TERMINE")
        print("=" * 60)
        print(f"Dataset pret dans: {os.path.abspath(OUTPUT_DIR)}/")
        print(f"Config YAML: {yaml_path}")
        print(f"\nPour entrainer: python download_hf_datasets.py train")

    # Action "train" ou "all" : lancer l'entrainement YOLOv8n
    if args.action in ["train", "all"]:
        yaml_path = os.path.join(OUTPUT_DIR, "dataset.yaml")
        if not os.path.exists(yaml_path):
            print(f"Erreur: {yaml_path} non trouve. Lancez d'abord 'download'.")
        else:
            train_yolov8n(yaml_path, device=args.device)

    # Action "stats" : afficher les statistiques du dataset existant
    if args.action == "stats":
        print_stats()

    # Action "mapping" : afficher le mapping des categories
    if args.action == "mapping":
        print_mapping_summary()
