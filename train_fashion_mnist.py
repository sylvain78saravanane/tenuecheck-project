"""
Entrainement d'un modele de classification de vetements avec Fashion-MNIST
Utilise PyTorch (plus leger que TensorFlow sur Windows)

Fashion-MNIST classes:
0: T-shirt/top
1: Trouser
2: Pullover
3: Dress
4: Coat
5: Sandal      <- INTERDIT
6: Shirt
7: Sneaker
8: Bag
9: Ankle boot
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Configuration
NUM_EPOCHS = 25
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MODEL_PATH = "fashion_classifier.pth"

# Labels Fashion-MNIST
FASHION_LABELS = ["top", "trouser", "pullover", "dress", "coat",
                  "sandal", "shirt", "sneaker", "bag", "ankle_boot"]


class FashionCNN(nn.Module):
    """CNN simple pour la classification Fashion-MNIST"""

    def __init__(self, num_classes=10):
        super(FashionCNN, self).__init__()

        # Couches convolutionnelles
        self.conv_layers = nn.Sequential(
            # Bloc 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            # Bloc 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )

        # Couches denses
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


def download_fashion_mnist():
    """Telecharge Fashion-MNIST manuellement"""
    import urllib.request
    import gzip

    base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    data_dir = "fashion_mnist_data"
    os.makedirs(data_dir, exist_ok=True)

    def load_images(filepath):
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28, 28)

    def load_labels(filepath):
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    # Telecharger les fichiers
    for name, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Telechargement de {filename}...")
            urllib.request.urlretrieve(base_url + filename, filepath)

    # Charger les donnees
    train_images = load_images(os.path.join(data_dir, files["train_images"]))
    train_labels = load_labels(os.path.join(data_dir, files["train_labels"]))
    test_images = load_images(os.path.join(data_dir, files["test_images"]))
    test_labels = load_labels(os.path.join(data_dir, files["test_labels"]))

    return (train_images, train_labels), (test_images, test_labels)


def train_model():
    """Entraine le modele sur Fashion-MNIST"""
    print("="*60)
    print("ENTRAINEMENT - CLASSIFICATEUR DE VETEMENTS (PyTorch)")
    print("="*60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Utilisation de: {device}")

    # Charger le dataset
    print("\n[INFO] Chargement de Fashion-MNIST...")
    (train_images, train_labels), (test_images, test_labels) = download_fashion_mnist()

    # Normaliser et convertir en tenseurs
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    # Ajouter la dimension du canal
    train_images = train_images[:, np.newaxis, :, :]
    test_images = test_images[:, np.newaxis, :, :]

    # Creer les datasets PyTorch
    train_dataset = TensorDataset(
        torch.from_numpy(train_images),
        torch.from_numpy(train_labels).long()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(test_images),
        torch.from_numpy(test_labels).long()
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"[INFO] Dataset charge: {len(train_images)} images d'entrainement")
    print(f"[INFO] Classes: {FASHION_LABELS}")

    # Creer le modele
    print("\n[INFO] Creation du modele...")
    model = FashionCNN(num_classes=10).to(device)

    # Loss et optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Historique pour les graphiques
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    # Entrainement
    print("\n[INFO] Entrainement du modele...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        # Calculer les moyennes
        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # Sauvegarder le modele
    print(f"\n[INFO] Sauvegarde du modele dans {MODEL_PATH}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'labels': FASHION_LABELS
    }, MODEL_PATH)

    # Evaluation finale
    print("\n[INFO] Evaluation finale...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Rapport de classification
    from sklearn.metrics import classification_report
    print("\nRapport de classification:")
    print(classification_report(all_labels, all_preds, target_names=FASHION_LABELS))

    # Graphiques
    print("\n[INFO] Creation des graphiques...")
    plt.style.use("ggplot")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(history["train_acc"], label="Train Acc")
    ax2.plot(history["val_acc"], label="Val Acc")
    ax2.set_title("Training Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_plot.png")
    print("[INFO] Graphique sauvegarde: training_plot.png")

    print("\n" + "="*60)
    print("ENTRAINEMENT TERMINE")
    print("="*60)
    print(f"Modele sauvegarde: {MODEL_PATH}")
    print(f"Precision finale: {val_acc:.2f}%")
    print("\nPour utiliser ce modele, lancez l'application:")
    print("  python app.py")

    return model


if __name__ == "__main__":
    train_model()
