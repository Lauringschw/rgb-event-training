import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import os
import random
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from torchvision import models

load_dotenv(Path(__file__).parent.parent.parent / '.env')

SLIDING_BASE = Path(os.getenv("SLIDING_BASE"))

SEED   = 42
N_BINS = 5

GESTURE_TO_LABEL = {'rock': 0, 'paper': 1, 'scissor': 2}
LABEL_TO_GESTURE = {v: k for k, v in GESTURE_TO_LABEL.items()}


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# == dataset ===================================================================

class MmapVoxelDataset(Dataset):
    def __init__(self, data, labels, indices):
        self.data    = data
        self.labels  = labels
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.data[i].copy().astype(np.float32)
        max_val = np.abs(x).max()
        if max_val > 0:
            x = x / max_val
        x = torch.from_numpy(x)
        y = int(self.labels[i])
        return x, y


# == model =====================================================================

class ResNet18Voxel(nn.Module):
    def __init__(self, num_classes=3, n_bins=N_BINS):
        super().__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(n_bins, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


# == split (recording-level) ===================================================

def get_split(labels, recording_ids, test_size=0.20, val_size=0.10):
    unique_recs = np.unique(recording_ids)
    rec_labels  = np.array([labels[recording_ids == r][0] for r in unique_recs])

    recs_temp, recs_test, _, _ = train_test_split(
        unique_recs, rec_labels, test_size=test_size, random_state=42, stratify=rec_labels)

    rec_labels_temp = np.array([labels[recording_ids == r][0] for r in recs_temp])
    adjusted_val    = val_size / (1.0 - test_size)

    recs_train, recs_val, _, _ = train_test_split(
        recs_temp, rec_labels_temp, test_size=adjusted_val, random_state=123, stratify=rec_labels_temp)

    return recs_train, recs_val, recs_test


# == train / eval ==============================================================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += out.argmax(1).eq(y).sum().item()
        total      += y.size(0)
    return total_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out  = model(X)
            loss = criterion(out, y)
            total_loss += loss.item()
            correct    += out.argmax(1).eq(y).sum().item()
            total      += y.size(0)
    return total_loss / len(loader), 100.0 * correct / total


# == main ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_ms', type=int, required=True,
                        help='Window duration in ms (e.g. 10, 20, 30, ...)')
    args = parser.parse_args()

    seed_everything(SEED)

    MERGED_DIR = SLIDING_BASE / f"{args.window_ms}ms" / "merged"

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print("=" * 60)
    print(f"TRAINING: {args.window_ms}ms Voxel Model (ResNet-18)")
    print("=" * 60)
    print(f"Device     : {device}")
    print(f"Merged dir : {MERGED_DIR}\n")

    # == load ==================================================================
    data_path   = MERGED_DIR / "voxel_data.npy"
    labels_path = MERGED_DIR / "voxel_labels.npy"
    recids_path = MERGED_DIR / "voxel_recording_ids.npy"

    for p in [data_path, labels_path, recids_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}\nRun merge_voxel.py --window_ms {args.window_ms} first.")

    raw_data      = np.load(data_path, mmap_mode='r')
    raw_labels    = np.load(labels_path)
    recording_ids = np.load(recids_path)

    print(f"Loaded {len(raw_data)} samples")
    print(f"Data shape: {raw_data.shape}")

    # == split =================================================================
    # Fixed seeds ensure identical split across all window sizes
    recs_train, recs_val, recs_test = get_split(raw_labels, recording_ids)

    test_ids_path = SLIDING_BASE / "test_recording_ids.npy"
    np.save(test_ids_path, recs_test)
    print(f"Saved test recording IDs: {test_ids_path}")

    train_mask = np.isin(recording_ids, recs_train)
    val_mask   = np.isin(recording_ids, recs_val)
    test_mask  = np.isin(recording_ids, recs_test)

    train_idx = np.sort(np.where(train_mask)[0])
    val_idx   = np.sort(np.where(val_mask)[0])
    test_idx  = np.sort(np.where(test_mask)[0])

    print(f"\nSplit: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)} samples")
    print(f"       ({len(recs_train)} / {len(recs_val)} / {len(recs_test)} recordings)\n")

    # == dataloaders ===========================================================
    def make_loader(indices, shuffle=False):
        ds = MmapVoxelDataset(raw_data, raw_labels, indices)
        g  = torch.Generator()
        g.manual_seed(SEED)
        return DataLoader(ds, batch_size=32, shuffle=shuffle, num_workers=0, generator=g)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader   = make_loader(val_idx)
    test_loader  = make_loader(test_idx)

    # == model =================================================================
    model = ResNet18Voxel(n_bins=N_BINS).to(device)

    class_counts  = np.array([np.sum(raw_labels[train_idx] == i) for i in range(3)])
    class_weights = torch.FloatTensor(1.0 / class_counts) * 3
    criterion     = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer     = optim.Adam(model.parameters(), lr=0.001)
    scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Train class counts: rock={class_counts[0]}, paper={class_counts[1]}, scissor={class_counts[2]}")
    print(f"Model parameters  : {total_params:,}\n")

    # == training loop =========================================================
    MAX_EPOCHS = 50
    PATIENCE   = 10

    best_val_acc      = 0.0
    epochs_no_improve = 0
    model_path        = MERGED_DIR / f'model_voxel_{args.window_ms}ms_best.pth'

    print("=" * 50)
    print(f"Training — {args.window_ms}ms window")
    print("=" * 50 + "\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        print(f"Epoch {epoch}/{MAX_EPOCHS}  "
              f"train loss={train_loss:.4f} acc={train_acc:.2f}%  "
              f"val loss={val_loss:.4f} acc={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc      = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            print(f"  => New best val {best_val_acc:.2f}% — saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # == test ==================================================================
    print("\n" + "=" * 50)
    model.load_state_dict(torch.load(model_path, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test acc : {test_acc:.2f}%   (best val: {best_val_acc:.2f}%)")

    metrics_path = MERGED_DIR / f'metrics_{args.window_ms}ms.txt'
    with open(metrics_path, 'w') as f:
        f.write(f"Window          : {args.window_ms}ms\n")
        f.write(f"Best val acc    : {best_val_acc:.2f}%\n")
        f.write(f"Test acc        : {test_acc:.2f}%\n")
        f.write(f"Test loss       : {test_loss:.4f}\n")
    print(f"Metrics saved   : {metrics_path}")
    print(f"Next step: python3 extract_test_samples.py --repr voxel --window_ms {args.window_ms}")