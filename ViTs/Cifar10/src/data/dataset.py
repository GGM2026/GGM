import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# Per-dataset normalization stats and metadata
_DATASET_STATS = {
    "cifar10": {
        "mean": [0.4914, 0.4822, 0.4465],
        "std":  [0.2470, 0.2435, 0.2616],
        "num_classes": 10,
        "cls": datasets.CIFAR10,
    },
    "cifar100": {
        "mean": [0.5071, 0.4865, 0.4409],
        "std":  [0.2673, 0.2564, 0.2762],
        "num_classes": 100,
        "cls": datasets.CIFAR100,
    },
    "mnist": {
        "mean": [0.1307],
        "std":  [0.3081],
        "num_classes": 10,
        "cls": datasets.MNIST,
    },
}


class DataHandler:
    def __init__(
        self,
        image_information,
        batch_size,
        data_dir,
        dataset_name="cifar10",
        val_split=0.1,
        num_workers=8,
        split_seed=42,
    ):
        self.img_info = image_information
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.dataset_name = dataset_name.lower()
        self.val_split = val_split
        self.num_workers = num_workers
        self.split_seed = split_seed

        if self.dataset_name not in _DATASET_STATS:
            raise ValueError(
                f"Unsupported dataset '{dataset_name}'. "
                f"Choose from {list(_DATASET_STATS.keys())}."
            )

        stats = _DATASET_STATS[self.dataset_name]
        self.mean = stats["mean"]
        self.std = stats["std"]
        self.num_classes = stats["num_classes"]
        self._dataset_cls = stats["cls"]

    # --------------------------------------------------
    # Train transforms (standard CIFAR augmentation)
    # --------------------------------------------------
    def _train_transform(self):
        return transforms.Compose([
            transforms.RandomCrop(self.img_info.width, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
            transforms.RandomErasing(p=0.25),
        ])

    # --------------------------------------------------
    # Eval transforms (val / test)
    # --------------------------------------------------
    def _eval_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    # --------------------------------------------------
    # Dataloaders: (train, val, test)
    # --------------------------------------------------
    def get_dataloaders(self):
        root = str(self.data_dir)

        # Full training set with two transform "views":
        # the train split gets augmentation, the val split gets eval transforms.
        train_full = self._dataset_cls(
            root=root, train=True, download=True,
            transform=self._train_transform(),
        )
        val_full = self._dataset_cls(
            root=root, train=True, download=True,
            transform=self._eval_transform(),
        )
        test_dataset = self._dataset_cls(
            root=root, train=False, download=True,
            transform=self._eval_transform(),
        )

        n_total = len(train_full)
        n_val = int(round(self.val_split * n_total))
        n_train = n_total - n_val

        # Same generator for both splits so the val indices are exactly the
        # complement of the train indices (no train/val leakage).
        train_subset, _ = random_split(
            train_full, [n_train, n_val],
            generator=torch.Generator().manual_seed(self.split_seed),
        )
        _, val_subset = random_split(
            val_full, [n_train, n_val],
            generator=torch.Generator().manual_seed(self.split_seed),
        )

        loader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

        train_loader = DataLoader(train_subset, shuffle=True, drop_last=True, **loader_kwargs)
        val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

        return train_loader, val_loader, test_loader
