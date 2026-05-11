import os
import math
from dataclasses import dataclass
import csv
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from conv2dggd import Conv2dGGD
from linearggd import LinearGGD

def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class Config:
    data_dir = './data/'
    output_dir = './outputs/'
    batch_size = 256
    epochs = 100
    lr = 1e-3
    num_workers = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resample_G = False
    base_seeds = [67, 68, 69, 70, 71, 72, 73, 74, 75, 76]  

class model_ggd(nn.Module):
    def __init__(self, base_seed):
        super().__init__()
        self._resample_rng = random.Random(base_seed + 999)

        self.conv1 = Conv2dGGD(3, 32, kernel_size=5, stride=2, padding=1, bias=False, N_scale=2, g_seed=base_seed)
        self.fc1 = LinearGGD(32 * 15 * 15, 32, bias=False, N_scale=2, g_seed=base_seed + 1)
        self.fc2 = LinearGGD(32, 32, bias=False, N_scale=2, g_seed=base_seed + 2)
        self.fc3 = LinearGGD(32, 32, bias=False, N_scale=2, g_seed=base_seed + 3)
        self.head = LinearGGD(32, 10, bias=False, N_scale=2, g_seed=base_seed + 4)

    def resample_G(self):
        for m in self.modules():
            if isinstance(m, (LinearGGD, Conv2dGGD)):
                m.resample_G(self._resample_rng.randint(1, 10000))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.head(x)
        return x

def get_loaders(cfg: Config):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=cfg.data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=cfg.data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    return train_loader, test_loader

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        running_correct += (preds == targets).sum().item()
        running_total += targets.size(0)

    epoch_loss = running_loss / running_total
    epoch_acc = 100.0 * running_correct / running_total
    return epoch_loss, epoch_acc

def init_csv(csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_acc",
        ])

def append_csv(csv_path, row):
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def run_training(cfg: Config, base_seed: int):
    print(f"\nStarting run for seed {base_seed}")
    seed_everything(base_seed)

    train_loader, test_loader = get_loaders(cfg)
    model = model_ggd(base_seed).to(cfg.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
    )

    if cfg.resample_G:
        csv_path = os.path.join(cfg.output_dir, f"training_log_resample_seed_{base_seed}.csv")
    else:
        csv_path = os.path.join(cfg.output_dir, f"training_log_regular_seed_{base_seed}.csv")
    init_csv(csv_path)

    for epoch in range(cfg.epochs):
        if cfg.resample_G:
            if (epoch + 1) in [30, 60]:
                model.resample_G()

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=cfg.device,
        )

        append_csv(csv_path, [
            epoch + 1,
            train_loss,
            train_acc,
        ])

        print(
            f"Seed {base_seed} | "
            f"Epoch [{epoch + 1:03d}/{cfg.epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%"
        )

    print(f"Training complete for seed {base_seed}")
    print(f"CSV log saved to: {csv_path}")

def main():
    cfg = Config()

    for base_seed in cfg.base_seeds:
        run_training(cfg, base_seed)

if __name__ == '__main__':
    main()