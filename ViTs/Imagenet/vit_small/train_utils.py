import random
import numpy as np
import os
import sys
import math
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms

from datasets import load_dataset
from tqdm import tqdm

from torch.optim.lr_scheduler import LambdaLR, OneCycleLR

from layer_utils import make_G_from_seed


class DataHandler:
    def __init__(
        self,
        config,
        data_dir=None,
        use_fake_data=False,
        stream_imagenet=True,
        distributed=False,
        rank=0,
        world_size=1,
    ):
        self.config = config
        self.image_size = config.image_size
        self.batch_size = config.batch_size
        self.data_dir = data_dir
        self.use_fake_data = use_fake_data
        self.stream_imagenet = stream_imagenet
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def _train_transform(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(
                self.image_size,
                scale=(0.08, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def _val_transform(self):
        return transforms.Compose([
            transforms.Resize(
                int(self.image_size * 256 / 224),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def _make_hf_transform(self, transform):
        def apply_transform(example):
            image = example["image"].convert("RGB")
            label = example["label"]
            return {
                "image": transform(image),
                "label": label,
            }
        return apply_transform

    def _collate_fn(self, batch):
        images = torch.stack([x["image"] for x in batch])
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
        return images, labels

    def get_dataloaders(self):
        if self.stream_imagenet and not self.use_fake_data:
            if self.is_main:
                print("Using Hugging Face ImageNet streaming.")

            train_dataset = load_dataset(
                "ILSVRC/imagenet-1k",
                split="train",
                streaming=True,
                token=True,
            )

            val_dataset = load_dataset(
                "ILSVRC/imagenet-1k",
                split="validation",
                streaming=True,
                token=True,
            )

            train_dataset = train_dataset.shuffle(
                buffer_size=10_000,
                seed=42,
            )

            if self.distributed:
                train_dataset = train_dataset.shard(
                    num_shards=self.world_size,
                    index=self.rank,
                )

                val_dataset = val_dataset.shard(
                    num_shards=self.world_size,
                    index=self.rank,
                )

            train_dataset = train_dataset.map(
                self._make_hf_transform(self._train_transform())
            )

            val_dataset = val_dataset.map(
                self._make_hf_transform(self._val_transform())
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=16,
                pin_memory=True,
                persistent_workers=False,
                prefetch_factor=4,
                collate_fn=self._collate_fn,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=16,
                pin_memory=True,
                persistent_workers=False,
                prefetch_factor=4,
                collate_fn=self._collate_fn,
            )

            if self.is_main:
                print("Streaming DataLoaders created successfully.")
                print(f"Batch size per GPU: {self.batch_size}")
                print(f"World size: {self.world_size}")
                print(f"Global batch size: {self.batch_size * self.world_size}")

            return train_loader, val_loader

        if not self.use_fake_data:
            if self.data_dir is None:
                raise ValueError(
                    "data_dir must be set when using local ImageNet. "
                    "Expected structure: data_dir/train/<class> and data_dir/val/<class>"
                )

            train_dir = os.path.join(self.data_dir, "train")
            val_dir = os.path.join(self.data_dir, "val")

            if not os.path.isdir(train_dir):
                raise FileNotFoundError(f"Missing train directory: {train_dir}")

            if not os.path.isdir(val_dir):
                raise FileNotFoundError(f"Missing val directory: {val_dir}")

            if self.is_main:
                print(f"Using local ImageNet from {self.data_dir}")

            train_dataset = datasets.ImageFolder(
                root=train_dir,
                transform=self._train_transform(),
            )

            val_dataset = datasets.ImageFolder(
                root=val_dir,
                transform=self._val_transform(),
            )

            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True,
            ) if self.distributed else None

            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False,
            ) if self.distributed else None

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=train_sampler is None,
                sampler=train_sampler,
                num_workers=8,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
                drop_last=True,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=8,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
                drop_last=False,
            )

            if self.is_main:
                print("Local ImageNet DataLoaders created successfully.")
                print(f"Train samples: {len(train_dataset)}")
                print(f"Val samples: {len(val_dataset)}")
                print(f"Classes: {len(train_dataset.classes)}")
                print(f"Batch size per GPU: {self.batch_size}")
                print(f"World size: {self.world_size}")
                print(f"Global batch size: {self.batch_size * self.world_size}")

            return train_loader, val_loader

        if self.is_main:
            print("Using FakeData for smoke test.")

        train_dataset = datasets.FakeData(
            size=1024,
            image_size=(3, self.image_size, self.image_size),
            num_classes=self.config.num_classes,
            transform=self._train_transform(),
        )

        val_dataset = datasets.FakeData(
            size=256,
            image_size=(3, self.image_size, self.image_size),
            num_classes=self.config.num_classes,
            transform=self._val_transform(),
        )

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            drop_last=True,
        ) if self.distributed else None

        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
            drop_last=False,
        ) if self.distributed else None

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            drop_last=False,
        )

        return train_loader, val_loader


class Evaluator:
    def __init__(self, model, loader, device, output_dir):
        self.model = model
        self.loader = loader
        self.device = device
        self.output_dir = output_dir

    def evaluate(self, model_path):
        print("\n--- Starting ImageNet Validation ---")

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in tqdm(self.loader, desc="Evaluating"):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(images)
                preds = outputs.argmax(dim=1)

                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        top1 = 100.0 * total_correct / total_samples
        print(f"\nImageNet Top-1 Accuracy: {top1:.2f}%")

        results = {
            "top1_accuracy": top1,
            "num_samples": total_samples,
        }

        os.makedirs(self.output_dir, exist_ok=True)

        with open(os.path.join(self.output_dir, "val_metrics.json"), "w") as f:
            json.dump(results, f, indent=2)

        return results


def save_run_config(
    path: str,
    *,
    model: nn.Module,
    trainer,
    config,
    optimizer,
    scheduler,
    train_loader,
    seed: int,
):
    model_to_inspect = model

    cfg = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "seed": int(seed),
        },
        "data": {
            "dataset": "imagenet-1k",
            "image_size": config.image_size,
            "batch_size": train_loader.batch_size,
            "global_batch_size": train_loader.batch_size * trainer.world_size,
            "num_workers": train_loader.num_workers,
            "distributed": trainer.distributed,
            "world_size": trainer.world_size,
        },
        "model": {
            "name": model_to_inspect.__class__.__name__,
            "patch_size": config.patch_size,
            "dim": config.d_model,
            "depth": config.transformer_layers,
            "heads": config.num_heads,
            "embed_dropout": config.embed_dropout,
            "attn_dropout": config.attn_dropout,
            "mlp_dropout": config.ff_dropout,
        },
        "training": {
            "epochs": config.epochs,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "weight_decay": config.weight_decay,
            "label_smoothing": trainer.criterion.label_smoothing,
            "grad_clip": trainer.grad_clip,
            "use_amp": trainer.use_amp,
            "val_every": trainer.val_every,
            "use_compile": trainer.use_compile,
        },
        "optimizer": {
            "type": optimizer.__class__.__name__,
            "defaults": {
                k: v
                for k, v in optimizer.defaults.items()
                if isinstance(v, (int, float, str, bool, tuple))
            },
            "param_groups": [
                {
                    "lr": pg.get("lr"),
                    "weight_decay": pg.get("weight_decay", 0.0),
                }
                for pg in optimizer.param_groups
            ],
        },
        "scheduler": {
            "type": scheduler.__class__.__name__ if scheduler else None,
            "params": {},
        },
    }

    if scheduler is not None:
        for k, v in scheduler.__dict__.items():
            if isinstance(v, (int, float, str, bool)):
                cfg["scheduler"]["params"][k] = v

    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def build_two_stage_cosine_scheduler(
    optimizer,
    warmup_epochs,
    total_epochs,
    steps_per_epoch,
    mid_epoch=150,
    mid_lr_ratio=0.5,
    min_lr_ratio=0.001,
    warmup_start_ratio=0.1,
):
    warmup_steps = warmup_epochs * steps_per_epoch
    mid_steps = mid_epoch * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def cosine_interp(start, end, progress):
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return end + (start - end) * cosine

    def lr_lambda(step):
        if step < warmup_steps:
            alpha = float(step) / float(max(1, warmup_steps))
            return warmup_start_ratio + (1.0 - warmup_start_ratio) * alpha

        if step < mid_steps:
            progress = float(step - warmup_steps) / float(max(1, mid_steps - warmup_steps))
            return cosine_interp(
                start=1.0,
                end=mid_lr_ratio,
                progress=progress,
            )

        progress = float(step - mid_steps) / float(max(1, total_steps - mid_steps))
        return cosine_interp(
            start=mid_lr_ratio,
            end=min_lr_ratio,
            progress=progress,
        )

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        epochs,
        lr,
        min_lr_ratio,
        weight_decay,
        output_dir,
        warmup_epochs=5,
        label_smoothing=0.1,
        use_amp=True,
        use_compile=True,
        val_every=5,
        grad_clip=0.0,
        *,
        config,
        distributed=False,
        rank=0,
        world_size=1,
        local_rank=0,
    ):
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_main = rank == 0

        self.config = config
        self.warmup_epochs = warmup_epochs
        self.label_smoothing = label_smoothing
        self.optim_betas = (0.9, 0.95)

        if self.distributed:
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.output_dir = output_dir
        self.val_every = val_every
        self.grad_clip = grad_clip

        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        self.raw_model = model.to(self.device)

        self.min_lr_ratio = min_lr_ratio

        decay_params = []
        no_decay_params = []
        no_decay_names = []

        for name, param in self.raw_model.named_parameters():
            if not param.requires_grad:
                continue

            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay_params.append(param)
                no_decay_names.append(name)
            else:
                decay_params.append(param)

        total = sum(p.numel() for p in self.raw_model.parameters())
        trainable = sum(p.numel() for p in self.raw_model.parameters() if p.requires_grad)

        if self.is_main:
            print(f"Optimizer will update {trainable:,}/{total:,} parameters")
            print(f"weight decay applied : {sum(p.numel() for p in decay_params):,} params")
            print(f"no weight decay : {sum(p.numel() for p in no_decay_params):,} params")
            print(f"no-decay includes : {no_decay_names}")

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        self.optim = torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=self.optim_betas,
            eps=1e-8,
        )

        try:
            self.steps_per_epoch = len(train_loader)
        except TypeError:
            self.steps_per_epoch = math.ceil(1_281_167 / (self.config.batch_size * self.world_size))

        self.val_steps = math.ceil(50_000 / (self.config.batch_size * self.world_size))

        self.lr_sch = build_two_stage_cosine_scheduler(
            self.optim,
            warmup_epochs=warmup_epochs,
            total_epochs=epochs,
            steps_per_epoch=self.steps_per_epoch,
            mid_epoch=75,
            mid_lr_ratio=1e-4 / lr,
            min_lr_ratio=self.min_lr_ratio,
            warmup_start_ratio=0.01,
        )

        self.use_amp = bool(use_amp)
        self.use_compile = bool(use_compile)

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp and self.device.type == "cuda",
        )

        os.makedirs(self.output_dir, exist_ok=True)

        self.best_val_acc = 0.0
        self.n_anneal_schedule = []
        self._last_applied_anneal_epoch = None

        model_for_forward = self.raw_model

        if self.use_compile:
            if self.is_main:
                print("Compiling model with torch.compile...")
            model_for_forward = torch.compile(model_for_forward)

        if self.distributed:
            self.model = DDP(
                model_for_forward,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
        else:
            self.model = model_for_forward

    def _reduce_stats(self, loss_sum, correct, samples):
        stats = torch.tensor(
            [loss_sum, correct, samples],
            device=self.device,
            dtype=torch.float64,
        )

        if self.distributed:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        return stats.tolist()

    def _train_one_epoch(self, epoch: int):
        self.model.train()

        if self.distributed and hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)

        if hasattr(self.train_loader.dataset, "set_epoch"):
            self.train_loader.dataset.set_epoch(epoch)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        loader = tqdm(
            self.train_loader,
            total=self.steps_per_epoch,
            desc=f"Train [epoch {epoch + 1}/{self.epochs}]",
            file=sys.stderr,
            ncols=120,
            dynamic_ncols=True,
            disable=not self.is_main,
        )

        for step, (images, labels) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optim.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                "cuda",
                enabled=self.use_amp and self.device.type == "cuda",
                dtype=torch.float16,
            ):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

                reg_loss = torch.tensor(0.0, device=self.device)

                for m in self.raw_model.modules():
                    if hasattr(m, "_reg_loss") and m._reg_loss is not None:
                        reg_loss = reg_loss + m._reg_loss

                loss = loss + reg_loss

                if step == 0 and self.is_main:
                    print("regularizer:", reg_loss.item())

            self.scaler.scale(loss).backward()

            if self.grad_clip is not None and self.grad_clip > 0:
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.raw_model.parameters() if p.requires_grad],
                    self.grad_clip,
                )

            self.scaler.step(self.optim)
            self.scaler.update()
            self.lr_sch.step()

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

            if self.is_main and step % 200 == 0:
                with open(os.path.join(self.output_dir, "heartbeat.txt"), "w") as f:
                    f.write(datetime.now().isoformat())

            if self.is_main:
                loader.set_postfix(
                    loss=f"{loss.item():.3f}",
                    lr=f"{self.optim.param_groups[0]['lr']:.2e}",
                )

        total_loss, total_correct, total_samples = self._reduce_stats(
            total_loss,
            total_correct,
            total_samples,
        )

        loss_mean = total_loss / max(1.0, total_samples)
        acc = 100.0 * total_correct / max(1.0, total_samples)

        return loss_mean, acc

    @torch.no_grad()
    def _validate(self, epoch: int):
        self.model.eval()

        if self.distributed and hasattr(self.val_loader.sampler, "set_epoch"):
            self.val_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(
            self.val_loader,
            desc=f"Val   [epoch {epoch + 1}]",
            disable=not self.is_main,
        )

        for step, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.amp.autocast(
                "cuda",
                enabled=self.use_amp and self.device.type == "cuda",
                dtype=torch.float16,
            ):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

        total_loss, total_correct, total_samples = self._reduce_stats(
            total_loss,
            total_correct,
            total_samples,
        )

        loss_mean = total_loss / max(1.0, total_samples)
        acc = 100.0 * total_correct / max(1.0, total_samples)

        return loss_mean, acc

    @torch.no_grad()
    def report_rho_stats(self):
        if not self.is_main:
            return

        print("\n--- Rho Exceed Stats ---")

        for name, m in self.raw_model.named_modules():
            if hasattr(m, "_rho_total_sum") and m._rho_total_sum.item() > 0:
                pct = 100.0 * (m._rho_exceed_sum / m._rho_total_sum)
                print(f"{name} rho_exceed% = {pct.item():.2f}")

                m._rho_exceed_sum.zero_()
                m._rho_total_sum.zero_()

    @torch.no_grad()
    def apply_n_annealing(self, epoch: int):
        if not self.n_anneal_schedule:
            return

        if self._last_applied_anneal_epoch == epoch:
            return

        keys = ["q", "k", "v", "proj", "fc1", "fc2"]

        for sched_epoch, spec in self.n_anneal_schedule:
            if epoch != sched_epoch:
                continue

            if isinstance(spec, (int, float)):
                scale_map = {k: float(spec) for k in keys}
            else:
                scale_map = {k: float(spec.get(k, 1.0)) for k in keys}

            if self.is_main:
                print(f"\nApplying N annealing at epoch {epoch}: {scale_map}")

            layer_types = {k: [] for k in keys}
            num_touched = 0

            for name, m in self.raw_model.named_modules():
                if not hasattr(m, "set_N_factor"):
                    continue

                layer_key = None

                for k in keys:
                    if f".{k}" in name:
                        layer_key = k
                        break

                if layer_key is None:
                    continue

                scale = scale_map[layer_key]

                if scale == 1.0:
                    continue

                new_factor = float(m.base_N_factor) * scale
                m.set_N_factor(new_factor)
                layer_types[layer_key].append(new_factor)
                num_touched += 1

            if self.is_main:
                print(f"Touched {num_touched} layers")

                for k, vals in layer_types.items():
                    if vals:
                        print(f"  {k}: {vals[0]:.3f} (n={len(vals)})")

            self._last_applied_anneal_epoch = epoch
            break

    @torch.no_grad()
    def smooth_resample_G(self, epoch, alpha=0.9):
        if epoch == 0 or epoch % 10 != 0:
            return

        if self.distributed:
            seed_tensor = torch.empty(1, device=self.device, dtype=torch.long)

            if self.is_main:
                seed_tensor.random_(0, 2**31 - 1)

            dist.broadcast(seed_tensor, src=0)
            base_seed = int(seed_tensor.item())
        else:
            base_seed = torch.randint(0, 2**31 - 1, (1,)).item()

        for name, m in self.raw_model.named_modules():
            if hasattr(m, "G"):
                layer_seed = (base_seed + abs(hash(name))) % (2**31 - 1)

                G_new = make_G_from_seed(
                    seed=layer_seed,
                    N=m.G.size(0),
                    d=m.G.size(1),
                    device=m.G.device,
                )

                m.G.mul_(alpha).add_(G_new, alpha=(1 - alpha))

        if self.is_main:
            print(f"Smooth G refresh at epoch {epoch}")

    def fit(self, model_path: str, run_seed: int, start_epoch: int = 0):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        if self.is_main:
            save_run_config(
                path=os.path.join(self.output_dir, "run_config.json"),
                model=self.raw_model,
                trainer=self,
                config=self.config,
                optimizer=self.optim,
                scheduler=self.lr_sch,
                train_loader=self.train_loader,
                seed=run_seed,
            )

        csv_path = os.path.join(self.output_dir, "training_history.txt")
        self._csv_initialized = os.path.exists(csv_path)

        for epoch in range(start_epoch, self.epochs):
            if self.is_main:
                print(f"\n--- Epoch {epoch + 1}/{self.epochs} ---")

            train_loss, train_acc = self._train_one_epoch(epoch)

            if (epoch % self.val_every == 0) or (epoch == self.epochs - 1):
                val_loss, val_acc = self._validate(epoch)
            else:
                val_loss, val_acc = float("nan"), float("nan")

            row = {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "lr": float(self.optim.param_groups[0]["lr"]),
            }

            if self.is_main:
                if not self._csv_initialized:
                    with open(csv_path, "w") as f:
                        f.write("\t".join(row.keys()) + "\n")
                    self._csv_initialized = True

                with open(csv_path, "a") as f:
                    f.write("\t".join(str(v) for v in row.values()) + "\n")

                if hasattr(os, "sync"):
                    os.sync()

                last_path = os.path.join(self.output_dir, "last.pth")

                torch.save(
                    {
                        "model": self.raw_model.state_dict(),
                        "optimizer": self.optim.state_dict(),
                        "scheduler": self.lr_sch.state_dict(),
                        "scaler": self.scaler.state_dict(),
                        "epoch": epoch,
                        "best_val_acc": float(self.best_val_acc),
                    },
                    last_path,
                )

                if (not math.isnan(val_acc)) and (val_acc > self.best_val_acc):
                    self.best_val_acc = float(val_acc)

                    torch.save(
                        {
                            "model": self.raw_model.state_dict(),
                            "optimizer": self.optim.state_dict(),
                            "scheduler": self.lr_sch.state_dict(),
                            "scaler": self.scaler.state_dict(),
                            "epoch": epoch,
                            "best_val_acc": float(self.best_val_acc),
                        },
                        model_path,
                    )

                    print(f"Saved best model: {self.best_val_acc:.2f}% -> {model_path}")

                print(
                    f"lr={self.optim.param_groups[0]['lr']:.3e} | "
                    f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}% | "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%"
                )

            if self.distributed:
                dist.barrier()

        if self.is_main:
            print(f"\nTraining finished. Best val acc: {self.best_val_acc:.2f}%")

        return None

    @staticmethod
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True