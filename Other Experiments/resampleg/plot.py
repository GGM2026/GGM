import os
import csv
import matplotlib.pyplot as plt


def read_training_log(csv_path):
    epochs = []
    train_loss = []
    train_acc = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            train_acc.append(float(row["train_acc"]))

    return epochs, train_loss, train_acc


def main():
    output_dir = "./outputs"
    base_seeds = [67, 68, 69, 70, 71, 72, 73, 74, 75, 76]

    for seed in base_seeds:
        resample_csv = os.path.join(output_dir, f"training_log_resample_seed_{seed}.csv")
        regular_csv = os.path.join(output_dir, f"training_log_regular_seed_{seed}.csv")

        if not os.path.exists(resample_csv):
            print(f"Missing file: {resample_csv}")
            continue
        if not os.path.exists(regular_csv):
            print(f"Missing file: {regular_csv}")
            continue

        epochs_r, train_loss_r, _ = read_training_log(resample_csv)
        epochs_n, train_loss_n, _ = read_training_log(regular_csv)

        plt.figure(figsize=(8, 5))
        plt.plot(epochs_r, train_loss_r, label=f"Resample (seed {seed})")
        plt.plot(epochs_n, train_loss_n, label=f"Regular (seed {seed})")
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
        plt.title(f"Train Loss Comparison for Seed {seed}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"train_loss_compare_seed_{seed}.png")
        plt.savefig(save_path, dpi=200)
        plt.show()
        plt.close()

        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()