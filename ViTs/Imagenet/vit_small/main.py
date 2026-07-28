import os
import torch
import signal
import torch.distributed as dist

from train_utils import DataHandler, Trainer
from model import VisionTransformer

from dataclasses import dataclass


@dataclass
class Config:
    image_size: int = 224
    in_channels: int = 3
    patch_size: int = 16
    d_model: int = 384
    num_heads: int = 6
    transformer_layers: int = 12
    ff_ratio: int = 4
    embed_dropout: float = 0.0
    attn_dropout: float = 0.0
    ff_dropout: float = 0.0

    batch_size: int = 256
    num_classes: int = 1000

    epochs: int = 300
    learning_rate: float = 1e-3
    min_lr: float = 1e-6
    weight_decay: float = 0.1
    warmup_epochs: int = 10
    val_every: int = 5
    label_smoothing: float = 0.1


def force_exit_on_ctrl_c(signum, frame):
    print("\nCtrl+C received. Force stopping now...", flush=True)
    os._exit(130)


def setup_distributed():
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )

        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    return distributed, rank, world_size, local_rank


def cleanup_distributed(distributed):
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def main():
    signal.signal(signal.SIGINT, force_exit_on_ctrl_c)

    distributed, rank, world_size, local_rank = setup_distributed()

    output_dir = "./runs"
    os.makedirs(output_dir, exist_ok=True)

    if is_main_process(rank):
        print("Output directory:", output_dir)
        print(
            f"distributed={distributed}, "
            f"rank={rank}, "
            f"world_size={world_size}, "
            f"local_rank={local_rank}"
        )

    resume_path = os.path.join(output_dir, "last.pth")
    if not os.path.exists(resume_path):
        resume_path = ""

    seed = 42
    use_amp = True
    use_compile = True

    Trainer.set_seed(seed)

    config = Config()

    if distributed:
        if config.batch_size % world_size != 0:
            raise ValueError(
                f"Global batch_size={config.batch_size} must be divisible by world_size={world_size}"
            )
        config.batch_size = config.batch_size // world_size

    data_dir = "./imagenet"

    use_fake_data = False
    stream_imagenet = False

    data = DataHandler(
        config=config,
        data_dir=data_dir,
        use_fake_data=use_fake_data,
        stream_imagenet=stream_imagenet,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )

    train_loader, val_loader = data.get_dataloaders()

    model = VisionTransformer(config)

    if is_main_process(rank):
        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total parameters: {num_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.epochs,
        lr=config.learning_rate,
        min_lr_ratio=config.min_lr / config.learning_rate,
        weight_decay=config.weight_decay,
        output_dir=output_dir,
        warmup_epochs=config.warmup_epochs,
        label_smoothing=config.label_smoothing,
        use_amp=use_amp,
        use_compile=use_compile,
        val_every=config.val_every,
        grad_clip=1.0,
        config=config,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )

    model_path = os.path.join(output_dir, "best.pth")
    start_epoch = 0

    if resume_path:
        if is_main_process(rank):
            print(f"Resuming from {resume_path}")

        ckpt = torch.load(resume_path, map_location=trainer.device)

        trainer.raw_model.load_state_dict(ckpt["model"])
        trainer.optim.load_state_dict(ckpt["optimizer"])
        trainer.lr_sch.load_state_dict(ckpt["scheduler"])
        trainer.scaler.load_state_dict(ckpt["scaler"])

        start_epoch = ckpt["epoch"] + 1
        trainer.best_val_acc = ckpt.get("best_val_acc", 0.0)

        last_lr = trainer.lr_sch.get_last_lr()

        if last_lr:
            for pg in trainer.optim.param_groups:
                pg["lr"] = last_lr[0]

        if is_main_process(rank):
            print(f"Resumed at epoch {start_epoch}, lr={last_lr}")

    try:
        trainer.fit(
            model_path=model_path,
            run_seed=seed,
            start_epoch=start_epoch,
        )
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()