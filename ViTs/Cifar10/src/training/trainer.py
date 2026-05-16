import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import random
import os
def create_ema(model):
    ema = deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


@torch.no_grad()
def update_ema(model, ema, decay):
    for (_, p), (_, ep) in zip(model.named_parameters(), ema.named_parameters()):
        ep.data.mul_(decay).add_(p.data, alpha=1.0 - decay)



def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size, device=x.device)

    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    y_a, y_b = y, y[index]

    return x, y_a, y_b, lam

def ggd_reg_loss(model):
    reg = None
    for m in model.modules():
        if hasattr(m, "_reg_loss") and (m._reg_loss is not None):
            reg = m._reg_loss if reg is None else reg + m._reg_loss
    return reg



class Trainer:
    def __init__(
    self,
    model,
    train_loader,
    val_loader,
    optimizer,
    scaler,
    lr_scheduler,
    epochs,
    use_ema=False,
    ema_decay=0.999,
):

        self.device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.mps.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optim = optimizer
        self.scaler = scaler
        self.lr_sch = lr_scheduler
        self.epochs = epochs


        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema = create_ema(model).to(self.device) if use_ema else None

        if self.use_ema:
            for p, ep in zip(self.model.parameters(), self.ema.parameters()):
                assert p.device == ep.device, (
                    f"EMA device mismatch: model on {p.device}, ema on {ep.device}"
                )


    def _train_one_epoch(self):

        use_amp = torch.cuda.is_available()
        amp_dtype = torch.float16

        model = self.model
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
    
        use_mixup = True
        use_cutmix = True
        mix_prob = 0.5
        alpha = 1.0
    
        for image, label in tqdm(self.train_loader, desc='Training'):
            img, label = image.to(self.device), label.to(self.device)
    
            r = np.random.rand()

            if r < mix_prob and (use_mixup or use_cutmix):
                if use_mixup and use_cutmix:
                    if np.random.rand() < 0.5:
                        img, y_a, y_b, lam = mixup_data(img, label, alpha)
                    else:
                        img, y_a, y_b, lam = cutmix_data(img, label, alpha)
                elif use_mixup:
                    img, y_a, y_b, lam = mixup_data(img, label, alpha)
                else:
                    img, y_a, y_b, lam = cutmix_data(img, label, alpha)
            
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    logits = model(img)
                    loss = lam * self.criterion(logits, y_a) + (1 - lam) * self.criterion(logits, y_b)
            
                    reg = ggd_reg_loss(model)
                    if reg is not None:
                        loss = loss + reg
            
            else:
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    logits = model(img)
                    loss = self.criterion(logits, label)
            
                    reg = ggd_reg_loss(model)
                    if reg is not None:
                        loss = loss + reg
    
            self.optim.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
            self.lr_sch.step()
    
            if self.use_ema:
                update_ema(self.model, self.ema, self.ema_decay)
    
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
    
            if label.dim() > 1:
                _, true_labels = torch.max(label, 1)
            else:
                true_labels = label
    
            total_samples += label.size(0)
            total_correct += (predicted == true_labels).sum().item()
    
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * total_correct / total_samples
        return avg_loss, accuracy

    def _validate_model(self, model):
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validating"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * total_correct / total_samples
        return avg_loss, accuracy


    def _run_trainer(self, model_path):
        model = self.model
        model.to(self.device)
    
        best_val_accuracy = 0.0
        history = []
    
        run_dir = os.path.dirname(model_path)
        history_path = os.path.join(run_dir, "training_history.csv")
        latest_ckpt = os.path.join(run_dir, "latest.pth")
    
        print("Starting Training...")
        for epoch in range(self.epochs):
            print(f"--- Epoch {epoch+1}/{self.epochs} ---")
    
            train_loss, train_acc = self._train_one_epoch()
            eval_model = self.ema if self.use_ema else self.model
            val_loss, val_acc = self._validate_model(eval_model)
    
            row = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": self.optim.param_groups[0]["lr"],
            }
            history.append(row)
    
            pd.DataFrame(history).to_csv(history_path, index=False)
    
    
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save(eval_model.state_dict(), model_path)
                print(f"✅ New best model saved ({best_val_accuracy:.2f}%)")
    
            lr = self.optim.param_groups[0]["lr"]
            
            print(
                f"Epoch {epoch+1}: "
                f"lr={lr:.6e} | "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}% | "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%"
            )

    
        print("--- Training Finished ---")
        print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
        return history


    @staticmethod
    def set_seed(seed: int):
      """
      Sets the seed for reproducibility. We will avoid the strictest settings
      to prevent runtime errors on the GPU.
      """
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)

      if torch.cuda.is_available():
          torch.cuda.manual_seed(seed)
          torch.cuda.manual_seed_all(seed)

      torch.backends.cudnn.deterministic = False
      torch.backends.cudnn.benchmark = True