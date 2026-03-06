"""
Configuration du système de détection de code vestimentaire ENSITECH
"""

# Liste des vêtements/accessoires interdits à détecter
# Mappés aux classes YOLO/Fashion datasets
VETEMENTS_INTERDITS = {
    # Bas du corps
    "shorts": "Short",
    "short": "Short",
    "bermuda": "Bermuda",
    "baggy": "Pantalon baggy",
    "ripped_jeans": "Jean troué/déchiré",
    "mini_skirt": "Mini-jupe",
    "miniskirt": "Mini-jupe",

    # Hauts
    "crop_top": "Crop top (T-shirt au-dessus du nombril)",
    "tank_top": "Haut échancré",
    "sports_bra": "Brassière de sport",
    "leggings": "Leggings",
    "sportswear": "Tenue de sport",

    # Chaussures
    "flip_flops": "Tongs",
    "sandals": "Tongs/Sandales",
    "slippers": "Tongs",

    # Accessoires / couvre-chefs
    "cap": "Casquette",
    "baseball_cap": "Casquette",
    "hat": "Chapeau",
    "beanie": "Bonnet",
    "bandana": "Bandana",
    "headwear": "Couvre-chef"
}

# Classes YOLO COCO qui peuvent correspondre aux vêtements interdits
# YOLO standard détecte ces classes liées à la personne
YOLO_CLASSES_INTERET = {
    "person": True,  # Pour détecter les personnes
    "tie": False,    # Cravate - autorisée
    "backpack": False,  # Sac - autorisé
    "handbag": False,  # Sac à main - autorisé
}

# Configuration email pour les alertes
EMAIL_CONFIG = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "sender_email": "votre_email@gmail.com",
    "sender_password": "votre_mot_de_passe_app",  # Utiliser un mot de passe d'application
    "recipient_email": "responsable@ensitech.com"
}

# Configuration de la détection
DETECTION_CONFIG = {
    "confidence_threshold": 0.5,  # Seuil de confiance minimum
    "frame_skip": 2,  # Analyser 1 frame sur N pour optimiser
    "alert_cooldown": 30,  # Secondes entre deux alertes pour la même personne
    "save_alerts": True,  # Sauvegarder les images d'alerte
    "alerts_folder": "alerts"
}

# Configuration de l'interface
INTERFACE_CONFIG = {
    "window_title": "ENSITECH - Contrôle Code Vestimentaire",
    "camera_index": 0,  # Index de la caméra (0 = webcam par défaut)
    "frame_width": 1280,
    "frame_height": 720
}

# Messages d'alerte par type de vêtement
MESSAGES_ALERTE = {
    "Short": "ALERTE: Short détecté - Vêtement non autorisé",
    "Bermuda": "ALERTE: Bermuda détecté - Vêtement non autorisé",
    "Pantalon baggy": "ALERTE: Pantalon baggy détecté - Vêtement non autorisé",
    "Jean troué/déchiré": "ALERTE: Jean troué détecté - Vêtement non autorisé",
    "Mini-jupe": "ALERTE: Mini-jupe détectée - Vêtement non autorisé",
    "Crop top (T-shirt au-dessus du nombril)": "ALERTE: Crop top détecté - Vêtement non autorisé",
    "Haut échancré": "ALERTE: Haut échancré détecté - Vêtement non autorisé",
    "Brassière de sport": "ALERTE: Brassière de sport détectée - Vêtement non autorisé",
    "Leggings": "ALERTE: Leggings détectés - Vêtement non autorisé",
    "Tenue de sport": "ALERTE: Tenue de sport détectée - Vêtement non autorisé",
    "Tongs": "ALERTE: Tongs détectées - Chaussures non autorisées",
    "Tongs/Sandales": "ALERTE: Tongs/Sandales détectées - Chaussures non autorisées",
    "Casquette": "ALERTE: Casquette détectée - Accessoire non autorisé",
    "Chapeau": "ALERTE: Chapeau détecté - Accessoire non autorisé",
    "Bonnet": "ALERTE: Bonnet détecté - Accessoire non autorisé",
    "Bandana": "ALERTE: Bandana détecté - Accessoire non autorisé",
    "Couvre-chef": "ALERTE: Couvre-chef détecté - Accessoire non autorisé"
}
