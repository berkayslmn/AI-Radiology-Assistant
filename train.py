import os

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils.dataset import NIHChestXrayDataset, NIH_LABELS
from utils.model import build_densenet121, get_pos_weights

IMG_DIR = "veriseti/images/"
MODEL_SAVE_PATH = "models/densenet121_kalibre_edilmis_model.pth"
SPLIT_DIR = "models/splits"
TRAIN_SPLIT_PATH = os.path.join(SPLIT_DIR, "train_split.csv")
VAL_SPLIT_PATH = os.path.join(SPLIT_DIR, "val_split.csv")

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
EARLY_STOPPING_PATIENCE = 4
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def main():
    print(f"Kullanılan Cihaz: {DEVICE}")

    if not (os.path.exists(TRAIN_SPLIT_PATH) and os.path.exists(VAL_SPLIT_PATH)):
        raise FileNotFoundError(
            f"'{TRAIN_SPLIT_PATH}' ve/veya '{VAL_SPLIT_PATH}' bulunamadı.\n"
            "Önce: python -m utils.split"
        )

    train_df = pd.read_csv(TRAIN_SPLIT_PATH)
    val_df = pd.read_csv(VAL_SPLIT_PATH)
    print(
        f"[BİLGİ] train: {train_df['Patient ID'].nunique()} hasta / {len(train_df)} görüntü, "
        f"val: {val_df['Patient ID'].nunique()} hasta / {len(val_df)} görüntü"
    )

    train_dataset = NIHChestXrayDataset(train_df, IMG_DIR, NIH_LABELS, transform=train_transforms)
    val_dataset = NIHChestXrayDataset(val_df, IMG_DIR, NIH_LABELS, transform=val_transforms)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    pos_weight_tensor = get_pos_weights(train_df, NIH_LABELS).to(DEVICE)

    model = build_densenet121(pretrained=True).to(DEVICE)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    patience_counter = 0

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 15)

        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc="Eğitim"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        epoch_train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Doğrulama"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)

        print(
            f"Eğitim Kaybı (Train Loss): {epoch_train_loss:.4f} | "
            f"Doğrulama Kaybı (Val Loss): {epoch_val_loss:.4f}"
        )

        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f">>> Yeni en iyi model kaydedildi! ({MODEL_SAVE_PATH})")
        else:
            patience_counter += 1
            print(
                f"Uyarı: Doğrulama kaybı iyileşmedi. "
                f"Erken durdurma sayacı: {patience_counter}/{EARLY_STOPPING_PATIENCE}"
            )
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("\n[!] Aşırı öğrenme (Overfitting) tespit edildi. Erken durdurma tetiklendi!")
                break

    print("\nEğitim süreci sonlandırıldı.")
    print(
        "\n[BİLGİ] Sıradaki adımlar:\n"
        "  1) python optimize_thresholds.py\n"
        "  2) python evaluate.py"
    )


if __name__ == "__main__":
    main()