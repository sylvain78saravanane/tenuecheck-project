# ENSITECH - Système de Contrôle du Code Vestimentaire (TenueCheck)

Système de détection automatique en temps réel des tenues non conformes au règlement intérieur d'ENSITECH (Article 17).

## Fonctionnalités

- **Détection en temps réel** via webcam ou caméra de surveillance.
- **Vêtements interdits détectés** :
  - Bas : Short, Bermuda, Mini-jupe, Jean troué, Pantalon baggy
  - Hauts : Crop top, Brassière de sport, Tenue de sport
  - Chaussures : Tongs
  - Accessoires : Casquette, Chapeau, Bonnet, Bandana, Lunettes
- **Alertes automatiques** avec capture d'image stockées sur base de données (Supabase).
- **Application Mobile** (React Native) pour la réception des alertes en temps réel par les agents.
- **Interface web / API** pour l'administration.

---

## Architecture du Projet

Le projet est divisé en deux parties principales pour séparer les responsabilités :

```text
tenuecheck-project/
├── backend/            # L'IA (YOLO), OpenCV, et l'API Python/Flask
└── mobile/             # L'Application Mobile React Native (Expo)

## Installation

### Prérequis
- Python 3.8 ou supérieur
- Webcam ou caméra IP

### Installation rapide (Windows)
```batch
# Double-cliquez sur run.bat
```

### Installation manuelle
```bash
# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
python app.py
```

## Utilisation

1. Lancez l'application avec `python app.py` ou `run.bat`
2. Ouvrez votre navigateur à l'adresse : **http://localhost:5000**
3. La détection démarre automatiquement avec la webcam
4. Les violations sont affichées en temps réel avec un cadre rouge
5. Les alertes sont enregistrées dans le dossier `alerts/`

## Configuration

### Paramètres email (config.py)
Pour activer les alertes par email, modifiez `EMAIL_CONFIG` dans `config.py` :
```python
EMAIL_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "votre_email@gmail.com",
    "sender_password": "votre_mot_de_passe_app",
    "recipient_email": "responsable@ensitech.com"
}
```

### Paramètres de détection (config.py)
```python
DETECTION_CONFIG = {
    "confidence_threshold": 0.5,  # Seuil de confiance
    "frame_skip": 2,              # Traiter 1 frame sur N
    "alert_cooldown": 30,         # Délai entre alertes
}
```

## Structure du projet

```
ensitech_dress_code/
├── app.py              # Application Flask principale
├── detector.py         # Module de détection YOLO
├── alert_system.py     # Système d'alertes email
├── config.py           # Configuration
├── requirements.txt    # Dépendances Python
├── run.bat            # Script de lancement Windows
├── templates/
│   └── index.html     # Interface web
├── static/            # Fichiers statiques
└── alerts/            # Images des alertes
```

## API REST

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Interface web principale |
| `/video_feed` | GET | Flux vidéo MJPEG |
| `/api/violations` | GET | Liste des violations |
| `/api/toggle` | POST | Activer/désactiver la détection |
| `/api/capture` | POST | Capturer une image |
| `/api/test_alert` | POST | Envoyer une alerte de test |

## Technologies utilisées

- **YOLOv8** : Détection d'objets en temps réel
- **OpenCV** : Traitement d'images
- **Flask** : Serveur web
- **Python** : Langage principal

## Auteurs

Projet ENSITECH 2026 - Traitement d'image et détection de pattern

---
*Conformément à l'Article 17 du règlement intérieur d'ENSITECH*
