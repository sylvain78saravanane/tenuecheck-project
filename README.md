# TenueCheck - Systeme de Detection du Code Vestimentaire

Systeme de detection automatique en temps reel des tenues non conformes au reglement interieur d'ENSITECH (Article 17), base sur YOLOv8n et le transfer learning.

## Modele IA

Le modèle `dresscode_yolo.pt` est un **YOLOv8n** (nano) entraîné par transfer learning sur **10 155 images** provenant de :
- **Fashionpedia** (5 000 images) — hat, headband, hood
- **Roboflow - Cap Dataset** (3 321 images) — casquettes
- **Roboflow - Headwear Detection** (1 164 images) — chapeaux, casquettes, hijab
- **Roboflow - Cap Dataset PlayRoom** (670 images) — casquettes

## Utilisation
 
1. Lancez l'application : `python core/app.py`
2. Ouvrez **http://localhost:5001**
3. La détection démarre automatiquement avec la webcam
4. Les violations sont affichées en temps réel avec un cadre rouge
5. Les alertes sont enregistrées dans Supabase (PostgreSQL + Storage)

## Fonctionnalites

- **Detection en temps reel** via webcam ou camera de surveillance
- **Couvre-chefs detectes** : casquette, chapeau, bonnet, capuche, bandana
- **Alertes automatiques** avec capture d'image
- **Interface web** moderne et responsive
- **Envoi d'emails** aux responsables (en cours de développement)

### Performances actuelles (couvre_chef)

| Metrique | Valeur |
|----------|--------|
| mAP50 | 0.887 |
| mAP50-95 | 0.672 |
| Precision | 0.776 |
| Recall | 0.821 |

## Installation

### Prerequis
- Python 3.8+
- Webcam ou camera IP
- GPU NVIDIA (recommandé pour l'entrainement, pas necessaire pour l'inference)

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

## Structure du projet
 
```
tenuecheck-project/
│
├── ia/                                 # Couche IA
│   ├── detector.py                     # Détecteur YOLOv8 (inference)
│   ├── dresscode_yolo.pt               # Modèle entraîné
│   ├── capture_webcam.py               # Capture d'images pour le dataset
│   ├── download_roboflow_and_train.py  # Pipeline d'entraînement complet
│   ├── train_quick.py                  # Réentraînement rapide
│   ├── test_quick.py                   # Test rapide du modèle
│   ├── TenueCheck_Training_Colab.ipynb # Notebook Colab (pipeline ML complet)
│   └── GUIDE_ML.md                     # Documentation machine learning
│
├── core/                               # Couche API + BDD
│   ├── app.py                          # Application Flask principale
│   ├── alert_system.py                 # Système d'alertes email
│   ├── config.py                       # Configuration
│   ├── supabase_client.py              # Client Supabase (singleton)
│   ├── .env.example                    # Template des variables d'environnement
│   └── templates/
│       └── index.html                  # Interface web
│
├── .gitignore
├── README.md
├── requirements.txt
└── run.bat                             # Lancement rapide Windows
```

## API REST
 
| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Interface web principale |
| `/video_feed` | GET | Flux vidéo MJPEG |
| `/api/violations` | GET | 10 dernières alertes (Supabase) |
| `/api/toggle` | POST | Activer/désactiver la détection |
| `/api/capture` | POST | Capturer et uploader dans Storage |
| `/api/test_alert` | POST | Envoyer une alerte de test |
| `/api/stats` | GET | Statistiques de la session |
| `/api/config` | GET | Configuration active |

## Technologies

## Stack technique
 
- **YOLOv8n** — Détection d'objets en temps réel (Ultralytics)
- **PyTorch + CUDA** — Entraînement GPU
- **OpenCV** — Traitement d'images
- **Flask** — Serveur web
- **Supabase** — PostgreSQL (alertes) + Storage (clichés)
- **Roboflow + HuggingFace** — Sources de datasets



