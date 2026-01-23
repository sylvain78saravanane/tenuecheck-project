"""
Telecharge des datasets depuis Roboflow Universe
Pour completer le dataset ENSITECH Dress Code

Necessite: pip install roboflow
"""

import os
import sys

# Datasets Roboflow publics recommandes
# Format: (workspace, project, version)
RECOMMENDED_DATASETS = [
    # Casquettes et chapeaux
    {
        "name": "Caps Detection",
        "search": "cap detection baseball",
        "classes": ["cap", "hat"],
        "url": "https://universe.roboflow.com/search?q=cap+detection"
    },
    {
        "name": "Hat Detection",
        "search": "hat detection fedora",
        "classes": ["hat", "cap"],
        "url": "https://universe.roboflow.com/search?q=hat+detection"
    },
    # Tongs et sandales
    {
        "name": "Flip Flops Detection",
        "search": "flip flops sandals",
        "classes": ["flip_flops", "sandals"],
        "url": "https://universe.roboflow.com/search?q=flip+flops"
    },
    # Shorts
    {
        "name": "Shorts Detection",
        "search": "shorts clothing",
        "classes": ["shorts"],
        "url": "https://universe.roboflow.com/search?q=shorts+detection"
    },
]


def check_roboflow_installed():
    """Verifie si roboflow est installe"""
    try:
        import roboflow
        return True
    except ImportError:
        return False


def install_roboflow():
    """Installe roboflow"""
    print("Installation de roboflow...")
    os.system(f"{sys.executable} -m pip install roboflow")


def download_with_api(api_key, workspace, project, version, output_dir):
    """
    Telecharge un dataset avec l'API Roboflow
    """
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    dataset = project.version(version).download("yolov8", location=output_dir)

    return dataset


def print_manual_instructions():
    """Affiche les instructions de telechargement manuel"""
    print("""
================================================================================
TELECHARGEMENT MANUEL DEPUIS ROBOFLOW UNIVERSE
================================================================================

Pour chaque dataset necessaire, suivez ces etapes:

1. CAPS/CASQUETTES:
   - Allez sur: https://universe.roboflow.com/search?q=cap+detection
   - Choisissez un dataset avec beaucoup d'images
   - Cliquez sur "Download" > Format "YOLOv8"
   - Extrayez le ZIP dans ce dossier

2. HATS/CHAPEAUX:
   - Allez sur: https://universe.roboflow.com/search?q=hat+detection
   - Telechargez au format YOLOv8
   - Extrayez dans ce dossier

3. FLIP FLOPS/TONGS:
   - Allez sur: https://universe.roboflow.com/search?q=flip+flops+sandals
   - Telechargez au format YOLOv8
   - Extrayez dans ce dossier

4. BEANIE/BONNET:
   - Allez sur: https://universe.roboflow.com/search?q=beanie+winter+hat
   - Telechargez au format YOLOv8
   - Extrayez dans ce dossier

5. SPORTSWEAR:
   - Allez sur: https://universe.roboflow.com/search?q=sportswear+athletic
   - Telechargez au format YOLOv8
   - Extrayez dans ce dossier

================================================================================
DATASETS RECOMMANDES (liens directs):
================================================================================

1. Baseball Caps:
   https://universe.roboflow.com/capsdetection/caps-a3cqe

2. Hat Detection:
   https://universe.roboflow.com/myworkspace-5utbm/hat-detection-1mwfp

3. Flip Flops:
   https://universe.roboflow.com/school-ixbfk/flip-flop-detection

4. Clothing Detection (general):
   https://universe.roboflow.com/roboflow-100/apparel-detection

================================================================================
APRES TELECHARGEMENT:
================================================================================

1. Extrayez tous les ZIPs dans le dossier 'ensitech_dress_code/'

2. Lancez la construction du dataset:
   python build_dataset.py

3. Verifiez le dataset:
   python train_custom_yolo.py check

4. Lancez l'entrainement:
   python train_custom_yolo.py train
""")


def download_with_roboflow_cli():
    """Utilise l'API Roboflow si disponible"""
    if not check_roboflow_installed():
        print("[INFO] Roboflow n'est pas installe.")
        response = input("Voulez-vous l'installer? (o/n): ")
        if response.lower() == 'o':
            install_roboflow()
        else:
            print_manual_instructions()
            return

    print("\n" + "="*60)
    print("TELECHARGEMENT VIA API ROBOFLOW")
    print("="*60)

    print("""
Pour telecharger via l'API, vous avez besoin d'une cle API Roboflow.

1. Creez un compte gratuit sur https://roboflow.com
2. Allez dans Settings > API Keys
3. Copiez votre cle API

Note: Le telechargement manuel est souvent plus simple.
""")

    api_key = input("Entrez votre cle API Roboflow (ou 'skip' pour instructions manuelles): ")

    if api_key.lower() == 'skip' or not api_key:
        print_manual_instructions()
        return

    # Exemple de telechargement (a adapter selon les datasets disponibles)
    print("\n[INFO] Pour telecharger un dataset specifique:")
    print("  from roboflow import Roboflow")
    print("  rf = Roboflow(api_key='VOTRE_CLE')")
    print("  project = rf.workspace('workspace_name').project('project_name')")
    print("  dataset = project.version(1).download('yolov8')")

    print_manual_instructions()


def create_sample_structure():
    """Cree une structure exemple pour montrer le format attendu"""
    print("\n" + "="*60)
    print("CREATION D'UNE STRUCTURE EXEMPLE")
    print("="*60)

    example_dir = "roboflow_example"
    os.makedirs(f"{example_dir}/train/images", exist_ok=True)
    os.makedirs(f"{example_dir}/train/labels", exist_ok=True)
    os.makedirs(f"{example_dir}/valid/images", exist_ok=True)
    os.makedirs(f"{example_dir}/valid/labels", exist_ok=True)

    # Creer un data.yaml exemple
    yaml_content = """# Exemple de data.yaml Roboflow
train: train/images
val: valid/images

nc: 3
names: ['cap', 'hat', 'beanie']
"""
    with open(f"{example_dir}/data.yaml", 'w') as f:
        f.write(yaml_content)

    # Creer un README
    readme_content = """# Structure d'un dataset Roboflow

Quand vous telechargez un dataset depuis Roboflow au format YOLOv8,
vous obtenez cette structure:

```
dataset_name/
├── data.yaml           <- Configuration avec les classes
├── train/
│   ├── images/         <- Images d'entrainement (.jpg)
│   └── labels/         <- Annotations YOLO (.txt)
├── valid/
│   ├── images/         <- Images de validation
│   └── labels/
└── test/               <- (optionnel)
    ├── images/
    └── labels/
```

Format des annotations YOLO (fichier .txt):
```
class_id x_center y_center width height
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.15
```

Toutes les valeurs sont normalisees entre 0 et 1.
"""
    with open(f"{example_dir}/README.txt", 'w') as f:
        f.write(readme_content)

    print(f"[OK] Structure exemple creee dans: {example_dir}/")
    print("     Consultez README.txt pour comprendre le format.")


def main():
    print("="*60)
    print("AIDE AU TELECHARGEMENT - ROBOFLOW DATASETS")
    print("="*60)

    print("""
Ce script vous aide a telecharger des datasets depuis Roboflow Universe
pour completer votre dataset de detection de dress code.

Options:
1. Instructions de telechargement manuel (recommande)
2. Telechargement via API Roboflow
3. Creer une structure exemple

""")

    choice = input("Choisissez une option (1/2/3): ")

    if choice == "1":
        print_manual_instructions()
    elif choice == "2":
        download_with_roboflow_cli()
    elif choice == "3":
        create_sample_structure()
    else:
        print_manual_instructions()


if __name__ == "__main__":
    main()
