import os
import torch
import pandas as pd
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from datetime import datetime
from pathlib import Path

from src.utils.config import ImageParams, ModelParameters, Hyperparameters, save_run_config

from src.data.dataset import DataHandler
from src.models.vit_ggd import ViT
from src.training.trainer import Trainer
from src.training.evaluator import Evaluator

# --- Define Paths ---
PROJECT_ROOT = Path(".").resolve()
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_ROOT = "results"
DATASET_NAME = "cifar"

def main(num_runs, master_seed, test_name, image_information, model_parameters, hyperparameters, dataset_name="cifar10"):
    NUM_RUNS = num_runs
    MASTER_SEED = master_seed

    for i in range(NUM_RUNS):
        run_seed = i + MASTER_SEED
        Trainer.set_seed(seed=run_seed)
        print(f"\n--- Starting Run {i+1}/{NUM_RUNS} (Seed: {run_seed}) ---")

        # -------------------------------------------------
        # Run directory with timestamp (NO overwrite)
        # -------------------------------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{test_name}_run{i+1}_{timestamp}"
        run_output_dir = os.path.join(RESULTS_ROOT, dataset_name, run_name)

        os.makedirs(run_output_dir, exist_ok=True)
        os.makedirs(os.path.join(run_output_dir, "evaluation_results"), exist_ok=True)

        local_model_path = os.path.join(run_output_dir, "best_model.pth")
        history_path = os.path.join(run_output_dir, "training_history.csv")

        img_info = image_information
        mparams = model_parameters
        hparams = hyperparameters

        # -------------------------------------------------
        # Data
        # -------------------------------------------------
        data_handler = DataHandler(
            image_information=img_info,
            batch_size=hparams.batch_size,
            data_dir=DATA_DIR,
            dataset_name=dataset_name
        )
        train_loader, val_loader, test_loader = data_handler.get_dataloaders()

        # -------------------------------------------------
        # Model / Optim / Scheduler
        # -------------------------------------------------
        base_model = ViT(mparams=mparams, hparams=hparams, img_info=img_info)

        base_optimizer = optim.AdamW(
            base_model.parameters(),
            lr=hparams.learning_rate,
            weight_decay=hparams.weight_decay,
        )

        base_scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        base_scheduler = OneCycleLR(
            optimizer=base_optimizer,
            max_lr=hparams.learning_rate,
            steps_per_epoch=len(train_loader),
            epochs=hparams.epochs,
            pct_start=0.15,
            div_factor=10,
            final_div_factor=10,
        )

        trainer = Trainer(
            model=base_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=base_optimizer,
            scaler=base_scaler,
            lr_scheduler=base_scheduler,
            epochs=hparams.epochs,
            use_ema=False,
            ema_decay=0.99,
        )

        # -------------------------------------------------
        # 🔹 SAVE RUN CONFIG **BEFORE TRAINING**
        # -------------------------------------------------

        save_run_config(
            path=os.path.join(run_output_dir, "run_config.json"),
            model=base_model,
            hparams=hparams,
            optimizer=base_optimizer,
            scheduler=base_scheduler,
            use_ema=trainer.use_ema,
            ema_decay=trainer.ema_decay,
            seed=run_seed,
        )


        # -------------------------------------------------
        # Train
        # -------------------------------------------------
        training_history = trainer._run_trainer(model_path=local_model_path)
        pd.DataFrame(training_history).to_csv(history_path, index=False)

        print(f"Training history saved to {history_path}")
        print("\n--- Training complete. Starting final evaluation on test set. ---")

        evaluator = Evaluator(
            model=base_model,
            test_loader=test_loader,
            device=trainer.device,
            output_dir=os.path.join(run_output_dir, "evaluation_results"),
        )

        final_metrics = evaluator.evaluate(model_path=local_model_path)




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train GGD-ViT model")
    
    # Run parameters
    parser.add_argument("--num_runs", type=int, default=1, help="Number of runs")
    parser.add_argument("--master_seed", type=int, default=100, help="Master seed")
    parser.add_argument("--test_name", type=str, default="base_testing_F0", help="Test name prefix")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "imagenet", "mnist"], help="Dataset to use")
    
    # Image parameters
    parser.add_argument("--width", type=int, default=32, help="Image width")
    parser.add_argument("--height", type=int, default=32, help="Image height")
    parser.add_argument("--in_channel", type=int, default=3, help="Image input channels")
    
    # Model parameters
    parser.add_argument("--patch_size", type=int, default=4, help="Patch size")
    parser.add_argument("--inner_dim", type=int, default=192, help="Inner dimension")
    parser.add_argument("--transformer_layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--num_head", type=int, default=3, help="Number of heads")
    parser.add_argument("--embed_dropout", type=float, default=0.0, help="Embedding dropout")
    parser.add_argument("--attn_dropout", type=float, default=0.0, help="Attention dropout")
    parser.add_argument("--mlp_dropout", type=float, default=0.0, help="MLP dropout")
    
    # Quantization parameters
    parser.add_argument("--k_bits_x", type=int, default=2, help="Activation quantization bits")
    parser.add_argument("--k_bits_w", type=int, default=1, help="Weight quantization bits")
    parser.add_argument("--n_factor", type=int, default=2, help="N_factor for quantization")
    parser.add_argument("--rho_cap", type=float, default=0.99, help="rho_cap for quantization")
    
    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--out_classes", type=int, default=10, help="Number of output classes")
    parser.add_argument("--epochs", type=int, default=450, help="Training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-4*(512/512), help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.00, help="Weight decay")
    
    args = parser.parse_args()

    # --- base testing ---
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    main(
        num_runs=args.num_runs,
        master_seed=args.master_seed,
        test_name=args.test_name,
        dataset_name=args.dataset,
        image_information=ImageParams(width=args.width, height=args.height, in_channel=args.in_channel),
        model_parameters=ModelParameters(
            patch_size=args.patch_size, 
            inner_dim=args.inner_dim, 
            transformer_layers=args.transformer_layers, 
            num_head=args.num_head, 
            embed_dropout=args.embed_dropout, 
            attn_dropout=args.attn_dropout, 
            mlp_dropout=args.mlp_dropout,
            k_bits_x=args.k_bits_x,
            k_bits_w=args.k_bits_w,
            n_factor=args.n_factor,
            rho_cap=args.rho_cap
        ),
        hyperparameters=Hyperparameters(
            batch_size=args.batch_size, 
            out_classes=args.out_classes, 
            epochs=args.epochs, 
            learning_rate=args.learning_rate, 
            weight_decay=args.weight_decay
        )
    )