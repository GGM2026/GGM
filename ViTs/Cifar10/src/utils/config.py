import json
from datetime import datetime
from dataclasses import dataclass
@dataclass
class ImageParams:
    width: int
    height: int
    in_channel: int
@dataclass
class ModelParameters:
    patch_size: int
    inner_dim: int
    transformer_layers: int
    num_head: int
    embed_dropout: float
    attn_dropout: float
    mlp_dropout: float
    k_bits_x: int = 2
    k_bits_w: int = 1
    n_factor: int = 2
    rho_cap: float = 0.99

@dataclass
class Hyperparameters:
    batch_size: int
    out_classes: int
    epochs: int
    learning_rate: float
    weight_decay: float


def save_run_config(
    path,
    model,
    hparams,
    optimizer,
    scheduler,
    use_ema,
    ema_decay,
    seed,
):
    cfg = {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,

        "training": {
            "epochs": hparams.epochs,
            "batch_size": hparams.batch_size,
            "learning_rate": hparams.learning_rate,
            "weight_decay": hparams.weight_decay,
        },

        "ema": {
            "enabled": bool(use_ema),
            "decay": float(ema_decay) if use_ema else None,
        },

        "optimizer": {
            "type": optimizer.__class__.__name__,
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

        "ggd_layers": [],
    }

    if scheduler is not None:
        for k, v in scheduler.__dict__.items():
            if isinstance(v, (int, float, str, bool)):
                cfg["scheduler"]["params"][k] = v

    for name, m in model.named_modules():
        if hasattr(m, "k_bits_x") and hasattr(m, "k_bits_w"):
            layer_info = {
                "name": name,
                "type": m.__class__.__name__,
                "k_bits_x": int(m.k_bits_x),
                "k_bits_w": int(m.k_bits_w),
                "N_factor": getattr(m, "base_N_factor", None),
                "N": getattr(m, "base_N", None),
            }
            cfg["ggd_layers"].append(layer_info)

    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)