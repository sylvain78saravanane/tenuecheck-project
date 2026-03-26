# Guide Machine Learning - TenueCheck

Documentation pour comprendre comment fonctionne le machine learning dans le projet TenueCheck.

## 1. C'est quoi le machine learning dans ce projet ?

On utilise un modele d'intelligence artificielle appele **YOLOv8n** (You Only Look Once, version 8, nano) pour **detecter les couvre-chefs** (casquettes, bonnets, chapeaux, capuches) dans un flux video en temps reel.

Le modele regarde une image et repond a 2 questions :
- **Ou** se trouve le couvre-chef dans l'image ? (bounding box = rectangle rouge)
- **A quel point** il est sur que c'est un couvre-chef ? (confiance en %)

## 2. Comment ca marche ? Le Transfer Learning

On ne part pas de zero. On utilise le **transfer learning** :

```
Modele COCO (80 classes, 330K images)
         |
         | On garde les couches de base (detection de formes, textures, contours)
         | On remplace la derniere couche (80 classes -> 1 classe : couvre_chef)
         |
         v
Notre modele (1 classe, 10 000+ images)
```

C'est comme apprendre une nouvelle langue quand on en connait deja une : on ne reapprend pas ce qu'est un nom ou un verbe, on apprend juste les nouveaux mots.

### Pourquoi YOLOv8n (nano) ?

| Critere | Pourquoi nano |
|---------|---------------|
| Vitesse | 6ms par image = 160 FPS sur RTX 3070 |
| Taille | 6 MB — facile a partager et deployer |
| 1 seule classe | Un modele plus gros serait du surapprentissage |
| VRAM | Tourne en batch 32 sur une RTX 3070 (8 GB) |
| CPU | Fonctionne aussi sans GPU (plus lent mais OK) |

## 3. Le dataset

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

## 4. L'entrainement

### Les hyperparametres

```python
EPOCHS = 500        # Nombre max de passages sur tout le dataset
BATCH = 32          # Nombre d'images traitees en meme temps
IMGSZ = 640         # Resolution des images (640x640 pixels)
PATIENCE = 80       # Arret automatique si pas d'amelioration pendant 80 epochs
OPTIMIZER = "AdamW"  # Algorithme d'optimisation
LR = 0.001          # Taux d'apprentissage
```

### L'augmentation de donnees

Pour eviter le surapprentissage, on transforme les images aleatoirement :

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

## 5. Comprendre les metriques

### Les 3 Loss (erreurs - on veut qu'elles descendent)

| Loss | Question |
|------|----------|
| **box_loss** | Le rectangle est-il bien place sur le couvre-chef ? |
| **cls_loss** | Est-ce bien un couvre-chef et pas autre chose ? |
| **dfl_loss** | Les bords du rectangle sont-ils precis ? |

### Les 4 metriques de performance (on veut qu'elles montent)

| Metrique | Question | Notre score |
|----------|----------|-------------|
| **Precision** | Quand il dit "couvre-chef", a-t-il raison ? | 0.73 (73%) |
| **Recall** | Trouve-t-il tous les couvre-chefs ? | 0.77 (77%) |
| **mAP50** | Score global (seuil IoU 50%) | **0.82 (82%)** |
| **mAP50-95** | Score global strict (seuils 50% a 95%) | 0.60 (60%) |

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

## 6. Les scripts

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

## 7. Comment ameliorer le modele ?

Par ordre d'impact :

1. **Plus de donnees variees** — Ajouter des images webcam avec `capture_webcam.py` (differents angles, lumieres, types de couvre-chefs)
2. **Plus d'epochs** — Modifier `EPOCHS` dans `train_quick.py` (ex: 100 au lieu de 30)
3. **Captures negatives** — Ajouter des images SANS couvre-chef pour reduire les faux positifs
4. **Nouveaux datasets** — Ajouter d'autres datasets Roboflow dans `download_roboflow_and_train.py`

## 8. Copier le meilleur modele apres entrainement

Si le script crash ou est interrompu (Ctrl+C), le modele n'est pas copie automatiquement. Faire :

```bash
copy runs\detect\tenuecheck_couvre_chefX\weights\best.pt dresscode_yolo.pt
```

(Remplacer X par le numero du dernier dossier d'entrainement)

## 9. FAQ

**Q: Ca marche sans GPU ?**
R: Oui, l'application (`app.py`) tourne sur CPU. Seul l'entrainement necessite un GPU NVIDIA.

**Q: Pourquoi la detection est faible de loin ?**
R: YOLOv8n travaille en 640x640 pixels. De loin, le couvre-chef occupe tres peu de pixels et devient difficile a detecter.

**Q: Pourquoi le modele detecte ma chaise ?**
R: Le modele n'a pas assez vu d'images negatives (sans couvre-chef). Faire plus de captures avec la touche N dans `capture_webcam.py`.

**Q: Comment partager le modele ?**
R: Envoyer le fichier `dresscode_yolo.pt` (6 MB). Le destinataire le place a la racine du projet.
