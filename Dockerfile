# Utilisation de Python 3.11 slim pour réduire la taille de l'image
FROM python:3.11-slim

# Installation des bibliothèques système requises par OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copie du fichier requirements.txt en premier pour optimiser le cache Docker
COPY requirements.txt .

# Installation des dépendances Python avec OpenCV headless adapté au container
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir supabase gunicorn \
    && pip uninstall -y opencv-python 2>/dev/null; \
       pip install --no-cache-dir opencv-python-headless>=4.8.0

# Installation de Ruff pour garantir la qualité du code au moment du build
RUN pip install --no-cache-dir ruff

# Copie du code source dans le container
COPY . .

# Correction automatique du formatage et vérification du code via Ruff
RUN ruff check . --fix --config pyproject.toml

ENV PYTHONUNBUFFERED=1

EXPOSE 5001

# Démarrage de l'application via Gunicorn avec 1 worker et 4 threads
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "1", "--threads", "4", "--timeout", "120", "core.app:app"]