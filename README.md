# TenueCheck - Systeme de Detection du Code Vestimentaire

Systeme de detection automatique en temps reel des tenues non conformes au reglement interieur d'ENSITECH (Article 17), base sur YOLOv8n et le transfer learning.

## Modele IA

Le modele `dresscode_yolo.pt` est un **YOLOv8n** (nano) entraine par transfer learning sur **10 155 images** provenant de :
- **Fashionpedia** (5 000 images) — hat, headband, hood
- **Roboflow - Cap Dataset** (3 321 images) — casquettes
- **Roboflow - Headwear Detection** (1 164 images) — chapeaux, casquettes, hijab
- **Roboflow - Cap Dataset PlayRoom** (670 images) — casquettes

### Performances actuelles (couvre_chef)

| Metrique | Valeur |
|----------|--------|
| mAP50 | 0.887 |
| mAP50-95 | 0.672 |
| Precision | 0.776 |
| Recall | 0.821 |

### Classe detectee

| Classe | ID | Exemples |
|--------|----|----|
| couvre_chef | 0 | Casquette, bonnet, chapeau, capuche, bob, beret |

### Entrainement

- **Modele de base** : YOLOv8n pre-entraine sur COCO (transfer learning)
- **GPU** : NVIDIA GeForce RTX 3070 (8 GB VRAM)
- **Epochs** : 500 (early stopping patience 80)
- **Batch** : 32
- **Optimizer** : AdamW (lr=0.001, cos_lr)
- **Augmentation** : mosaic, mixup, copy_paste, rotation, flip, HSV

## Fonctionnalites

- **Detection en temps reel** via webcam ou camera de surveillance
- **Couvre-chefs detectes** : casquette, chapeau, bonnet, capuche, bandana
- **Alertes automatiques** avec capture d'image
- **Interface web** moderne et responsive
- **Envoi d'emails** aux responsables (configurable)

## Installation

### Prerequis
- Python 3.8+
- Webcam ou camera IP
- GPU NVIDIA (recommande pour l'entrainement, pas necessaire pour l'inference)

### Installation rapide (Windows)
```batch
run.bat
```

### Installation manuelle
```bash
pip install -r requirements.txt
python app.py
```

### Installation GPU (pour entrainement)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Utilisation

1. Lancez l'application : `python app.py` ou `run.bat`
2. Ouvrez **http://localhost:5000**
3. La detection demarre automatiquement avec la webcam
4. Les violations sont affichees en temps reel avec un cadre rouge
5. Les alertes sont enregistrees dans `alerts/`

## Entrainer le modele

Pour re-entrainer ou ameliorer le modele :

```bash
python download_roboflow_and_train.py
```

Ce script telecharge les datasets, les fusionne et lance l'entrainement. Le modele `dresscode_yolo.pt` est automatiquement mis a jour.

Pour tester le modele :
```bash
python test_quick.py
```

## Configuration

### Detection (config.py)
```python
DETECTION_CONFIG = {
    "confidence_threshold": 0.5,
    "frame_skip": 2,
    "alert_cooldown": 30,
}
```

### Email (config.py)
```python
EMAIL_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "votre_email@gmail.com",
    "sender_password": "votre_mot_de_passe_app",
    "recipient_email": "responsable@ensitech.com"
}
```

## Structure du projet

```
tenuecheck-project/
├── app.py                          # Application Flask principale
├── detector.py                     # Module de detection YOLO
├── alert_system.py                 # Systeme d'alertes email
├── config.py                       # Configuration
├── dresscode_yolo.pt               # Modele entraine (YOLOv8n)
├── requirements.txt                # Dependances Python
├── run.bat                         # Script de lancement Windows
├── download_roboflow_and_train.py  # Script d'entrainement complet
├── test_quick.py                   # Test rapide du modele
├── templates/
│   └── index.html                  # Interface web
├── dataset_couvre_chef/            # Dataset filtre (couvre_chef)
├── roboflow_downloads/             # Datasets Roboflow telecharges
├── runs/                           # Resultats d'entrainement YOLO
└── alerts/                         # Images des alertes
```

## API REST

| Endpoint | Methode | Description |
|----------|---------|-------------|
| `/` | GET | Interface web principale |
| `/video_feed` | GET | Flux video MJPEG |
| `/api/violations` | GET | Liste des violations |
| `/api/toggle` | POST | Activer/desactiver la detection |
| `/api/capture` | POST | Capturer une image |
| `/api/test_alert` | POST | Envoyer une alerte de test |

## Technologies

- **YOLOv8n** — Detection d'objets en temps reel (Ultralytics)
- **PyTorch + CUDA** — Entrainement GPU
- **OpenCV** — Traitement d'images
- **Flask** — Serveur web
- **Roboflow + HuggingFace** — Sources de datasets

## Auteurs

Projet TenueCheck — ENSITECH Master 2, 2026

---
*Conformement a l'Article 17 du reglement interieur d'ENSITECH*
