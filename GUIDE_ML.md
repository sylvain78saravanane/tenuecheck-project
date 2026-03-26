# Guide Machine Learning - TenueCheck

Documentation pour comprendre comment fonctionne le machine learning dans le projet TenueCheck.

## 1. Pourquoi le Machine Learning ?

Le reglement interieur ENSITECH (Article 17) interdit les couvre-chefs. Surveiller manuellement par camera demande un humain 24h/24. Le **machine learning** automatise cette tache : le modele apprend a reconnaitre les couvre-chefs a partir d'exemples, puis detecte en temps reel sur un flux video sans intervention humaine.

Alternatives sans ML :
- **Detection par couleur/forme** : trop de faux positifs, ne marche pas avec des couvre-chefs varies
- **Regles codees en dur** : impossible de coder "ce qui ressemble a une casquette" manuellement
- **Surveillance humaine** : couteux, fatigue, erreurs

Le ML est la seule approche qui **generalise** : il apprend a partir d'exemples et reconnait des couvre-chefs qu'il n'a jamais vus.

## 2. Pourquoi YOLO et pas un autre modele ?

Il existe plusieurs architectures de detection d'objets :

| Modele | Vitesse | Precision | Temps reel ? |
|---|---|---|---|
| **YOLO** | Tres rapide (6ms) | Bonne | Oui |
| Faster R-CNN | Lent (100ms+) | Tres bonne | Non |
| SSD | Rapide (20ms) | Moyenne | Oui mais moins precis |
| DETR (Transformer) | Lent (50ms+) | Tres bonne | Non |
| EfficientDet | Moyen (30ms) | Bonne | Limite |

**YOLO** (You Only Look Once) regarde l'image **une seule fois** pour detecter tous les objets. Les autres modeles (comme Faster R-CNN) font plusieurs passes. C'est pour ca que YOLO est le plus rapide.

Pour de la **videosurveillance en temps reel**, la vitesse est critique : il faut traiter 30 images par seconde minimum. Seul YOLO le permet tout en gardant une bonne precision.

## 3. Pourquoi YOLOv8 et pas v5, v7, v9, v10, v11 ?

| Version | Annee | Avantage | Inconvenient |
|---|---|---|---|
| YOLOv5 | 2020 | Tres stable, communaute large | Architecture vieillissante |
| YOLOv7 | 2022 | Tres performant | Configuration complexe |
| **YOLOv8** | **2023** | **Meilleur rapport perf/simplicite, API moderne (Ultralytics)** | - |
| YOLOv9 | 2024 | Leger gain de precision | Moins mature, communaute plus petite |
| YOLOv10/v11 | 2024-2025 | Tres recent | Peu teste, pas assez de retours |

**YOLOv8** est le choix optimal :
- **API Ultralytics** simple (`model.train()`, `model.val()`, `model()`) — tout en 3 lignes
- **Transfer learning integre** — charge les poids COCO automatiquement
- **Bien documente** — beaucoup de tutoriels et support communautaire
- **Stable et teste** — utilise en production dans des milliers de projets
- **Export facile** — ONNX, TensorRT, CoreML pour le deploiement mobile

## 4. Pourquoi YOLOv8n (nano) et pas s, m, l, x ?

| Variante | Parametres | Taille | Vitesse (RTX 3070) | Usage |
|---|---|---|---|---|
| **YOLOv8n (nano)** | **3M** | **6 MB** | **6ms** | **1-2 classes, temps reel** |
| YOLOv8s (small) | 11M | 22 MB | 15ms | 5-10 classes |
| YOLOv8m (medium) | 26M | 52 MB | 30ms | Multi-classes complexe |
| YOLOv8l (large) | 44M | 87 MB | 55ms | Competitions, haute precision |
| YOLOv8x (xlarge) | 68M | 131 MB | 80ms | Recherche |

On detecte **1 seule classe** (couvre_chef). Un modele nano suffit largement — un modele plus gros n'apporterait pas plus de precision mais causerait du **surapprentissage** (le modele memorise les images au lieu d'apprendre a generaliser).

De plus :
- **6 MB** = partageable sur GitHub, chargement instantane
- **6ms** = 160 FPS, largement au-dessus des 30 FPS necessaires
- Tourne aussi sur **CPU sans GPU** (plus lent mais fonctionnel)

## 5. Pourquoi le Transfer Learning ?

**From scratch** (depuis zero) :
- Le modele ne sait rien — il doit apprendre ce qu'est un contour, une forme, une texture
- Il faut **100 000+ images** et **des jours d'entrainement**
- Resultat mediocre avec peu de donnees

**Transfer learning** (notre approche) :
- On prend YOLOv8n **pre-entraine sur COCO** (330 000 images, 80 classes)
- Le modele sait deja reconnaitre des formes, textures, contours, objets
- On **remplace la derniere couche** (80 classes -> 1 classe : couvre_chef)
- On **affine** (fine-tune) avec nos 10 155 images
- Resultat : **mAP50 = 0.887 en 30 epochs (1h d'entrainement)**

```
Sans transfer learning :
  100 000 images + 3 jours = mAP50 ~0.70

Avec transfer learning :
  10 000 images + 1 heure = mAP50 0.887
```

C'est comme apprendre le portugais quand on parle deja espagnol — on ne repart pas de zero, on adapte ce qu'on sait deja.

```
Modele COCO (80 classes, 330K images)
         |
         | On garde les couches de base (detection de formes, textures, contours)
         | On remplace la derniere couche (80 classes -> 1 classe : couvre_chef)
         |
         v
Notre modele (1 classe, 10 000+ images)
```

## 6. Le dataset

Le modele a ete entraine sur **10 155+ images** de couvre-chefs provenant de :

| Source | Images | Contenu |
|--------|--------|---------|
| Fashionpedia (HuggingFace) | 5 000 | Photos de mode avec chapeaux, bandeaux, capuches |
| Roboflow - Cap Dataset | 3 321 | Casquettes |
| Roboflow - Headwear Detection | 1 164 | Chapeaux, casquettes, hijab |
| Roboflow - Cap PlayRoom | 670 | Casquettes |
| Captures webcam | ~1 000+ | Images prises avec notre camera pour ameliorer la detection en conditions reelles |

### Format des annotations (YOLO)

Chaque image a un fichier `.txt` associe avec les annotations :

```
0 0.500000 0.150000 0.400000 0.200000
```

Signification : `classe x_centre y_centre largeur hauteur` (valeurs normalisees entre 0 et 1)

### Split du dataset

| Split | Proportion | Utilite |
|-------|-----------|---------|
| Train (80%) | ~8 100 images | Le modele apprend dessus |
| Val (15%) | ~1 500 images | Verifie la performance pendant l'entrainement |
| Test (5%) | ~500 images | Evaluation finale, jamais vu par le modele |

## 7. L'entrainement

### Les hyperparametres

```python
EPOCHS = 30         # Nombre de passages sur tout le dataset
BATCH = 32          # Nombre d'images traitees en meme temps
IMGSZ = 640         # Resolution des images (640x640 pixels)
PATIENCE = 30       # Arret automatique si pas d'amelioration pendant 30 epochs
OPTIMIZER = "AdamW"  # Algorithme d'optimisation
LR = 0.001          # Taux d'apprentissage
```

### L'augmentation de donnees

Pour eviter le surapprentissage, on transforme les images aleatoirement pendant l'entrainement :

| Augmentation | Effet |
|---|---|
| Mosaic | Combine 4 images en une seule |
| Mixup | Melange 2 images avec transparence |
| HSV | Modifie teinte/saturation/luminosite |
| Rotation (20 deg) | Tourne l'image |
| Flip horizontal | Miroir gauche/droite |
| Scale (0.5) | Zoom avant/arriere |
| Copy-paste | Copie des objets d'une image a une autre |

### Ce qui se passe pendant un epoch

```
1. Le modele voit les 8100 images de train (par batch de 32)
2. Pour chaque batch :
   a. Il predit ou sont les couvre-chefs
   b. Il compare avec les vraies annotations
   c. Il calcule l'erreur (loss)
   d. Il ajuste ses parametres pour reduire l'erreur
3. A la fin de l'epoch, on teste sur les 1500 images de val
4. Si le score s'ameliore, on sauvegarde le modele (best.pt)
```

## 8. Comprendre les stats d'un epoch

Exemple :
```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
 30/30    3.86G     0.7857     0.6719     1.187        33          640
          Class     Images  Instances   Box(P    R    mAP50  mAP50-95)
          all        1523     1854      0.793  0.811  0.879   0.66
```

### Ligne d'entrainement

| Stat | Valeur | Signification |
|---|---|---|
| **Epoch 30/30** | Dernier epoch | Le modele a vu les 8124 images 30 fois |
| **GPU_mem 3.86G** | 3.86 GB utilises | Sur les 8 GB de la RTX 3070 (48% utilise) |
| **box_loss 0.7857** | Erreur de localisation | Ou est le couvre-chef ? Tres bas = tres precis |
| **cls_loss 0.6719** | Erreur de classification | Est-ce un couvre-chef ? Tres bas = il se trompe rarement |
| **dfl_loss 1.187** | Erreur des bords | Les contours du rectangle sont-ils precis ? Correct |
| **Instances 33** | 33 couvre-chefs | Dans le dernier batch de 32 images |
| **Size 640** | 640x640 pixels | Resolution de travail |
| **254/254** | Tous les batchs traites | 8124 images / 32 par batch = 254 batchs |
| **1.9it/s** | 1.9 batchs par seconde | Vitesse d'entrainement |
| **2:14** | 2 min 14 par epoch | Temps total pour une passe complete |

### Ligne de validation

| Stat | Valeur | Signification |
|---|---|---|
| **Images 1523** | Taille du jeu de validation | Images jamais vues pendant l'entrainement |
| **Instances 1854** | 1854 couvre-chefs annotes | Certaines images ont plusieurs couvre-chefs |
| **Box(P) 0.793** | **Precision 79.3%** | Quand il dit "couvre-chef", il a raison 4 fois sur 5 |
| **R 0.811** | **Recall 81.1%** | Il trouve 81% des vrais couvre-chefs |
| **mAP50 0.879** | **Score principal 87.9%** | Performance globale — excellent |
| **mAP50-95 0.66** | **Score strict 66%** | Moyenne sur plusieurs niveaux d'exigence |

### Les 3 Loss (erreurs - on veut qu'elles descendent)

| Loss | Question |
|------|----------|
| **box_loss** | Le rectangle est-il bien place sur le couvre-chef ? |
| **cls_loss** | Est-ce bien un couvre-chef et pas autre chose ? |
| **dfl_loss** | Les bords du rectangle sont-ils precis ? |

### Les 4 metriques de performance (on veut qu'elles montent)

| Metrique | Question | Notre score |
|----------|----------|-------------|
| **Precision** | Quand il dit "couvre-chef", a-t-il raison ? | 0.776 (78%) |
| **Recall** | Trouve-t-il tous les couvre-chefs ? | 0.821 (82%) |
| **mAP50** | Score global (seuil IoU 50%) | **0.887 (89%)** |
| **mAP50-95** | Score global strict (seuils 50% a 95%) | 0.672 (67%) |

### C'est quoi l'IoU (Intersection over Union) ?

```
IoU = Surface en commun / Surface totale

  Prediction:  [==========]
  Realite:        [==========]
  Commun:         [======]

  IoU = surface commune / surface totale des 2 rectangles
  Si IoU > 50% -> la detection est consideree comme correcte (mAP50)
  Si IoU > 95% -> la detection est tres precise (mAP95)
```

## 9. Les scripts

### Entrainement complet (depuis zero)
```bash
python download_roboflow_and_train.py
```
Telecharge tous les datasets + entraine. Resultat : `dresscode_yolo.pt`

### Entrainement rapide (amelioration)
```bash
python train_quick.py
```
Reprend depuis `dresscode_yolo.pt` existant et continue l'entrainement. Utile apres avoir ajoute des images webcam.

### Capture d'images webcam
```bash
python capture_webcam.py
```
- ESPACE = capture avec couvre-chef
- N = capture sans couvre-chef (negative)
- Q = quitter

Les images sont ajoutees au dataset de train. Relancer `train_quick.py` ensuite.

### Test du modele
```bash
python test_quick.py
```

### Lancer l'application
```bash
python app.py
```
Ouvre http://localhost:5000

## 10. Comment ameliorer le modele ?

Par ordre d'impact :

1. **Plus de donnees variees** — Ajouter des images webcam avec `capture_webcam.py` (differents angles, lumieres, types de couvre-chefs)
2. **Plus d'epochs** — Modifier `EPOCHS` dans `train_quick.py` (ex: 100 au lieu de 30)
3. **Captures negatives** — Ajouter des images SANS couvre-chef pour reduire les faux positifs
4. **Nouveaux datasets** — Ajouter d'autres datasets Roboflow dans `download_roboflow_and_train.py`

## 11. Copier le meilleur modele apres entrainement

Si le script crash ou est interrompu (Ctrl+C), le modele n'est pas copie automatiquement. Faire :

```bash
copy runs\detect\tenuecheck_couvre_chefX\weights\best.pt dresscode_yolo.pt
```

(Remplacer X par le numero du dernier dossier d'entrainement)

## 12. FAQ

**Q: Ca marche sans GPU ?**
R: Oui, l'application (`app.py`) tourne sur CPU. Seul l'entrainement necessite un GPU NVIDIA.

**Q: Pourquoi la detection est faible de loin ?**
R: YOLOv8n travaille en 640x640 pixels. De loin, le couvre-chef occupe tres peu de pixels et devient difficile a detecter.

**Q: Pourquoi le modele detecte ma chaise ?**
R: Le modele n'a pas assez vu d'images negatives (sans couvre-chef). Faire plus de captures avec la touche N dans `capture_webcam.py`.

**Q: Comment partager le modele ?**
R: Envoyer le fichier `dresscode_yolo.pt` (6 MB). Le destinataire le place a la racine du projet.

**Q: Quelle est la difference entre train_quick.py et download_roboflow_and_train.py ?**
R: `train_quick.py` reprend depuis le modele existant (rapide, 30 epochs). `download_roboflow_and_train.py` retelecharge tout et repart de zero (long mais complet).
