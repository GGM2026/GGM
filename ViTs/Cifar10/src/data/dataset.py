import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class DataHandler:
    def __init__(self, image_information, batch_size, data_dir, **kwargs):
        self.img_info = image_information
        self.batch_size = batch_size
        self.data_dir = data_dir

    def _train_transform(self):
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop((self.img_info.width, self.img_info.height), scale=(0.8, 1.0)),
            # num_ops = number of augmentations to apply
            # magnitude = strength of augmentations
            transforms.RandAugment(num_ops=2, magnitude=9), # can be tweaked after checking how similar the images are
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _val_test_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def get_dataloaders(self):
        train_dataset = datasets.CIFAR10(root=self.data_dir, train=True,
                                         download=True, transform=self._train_transform())

        val_dataset = datasets.CIFAR10(root=self.data_dir, train=True,
                                       download=True, transform=self._val_test_transform())

        # carve validation from train
        num_train = len(train_dataset)
        indices = torch.randperm(num_train).tolist()   # shuffle once before splitting

        split = int(0.9 * num_train)
        train_indices, val_indices = indices[:split], indices[split:]

        # Create subsets based on the indices
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)


        test_dataset = datasets.CIFAR10(root=self.data_dir, train=False,
                                        download=True, transform=self._val_test_transform())

        # # cutmix and mixup need to be used directly on batch, not working with transforms (ofc)
        # mixup_cutmix = [
        #     MixUp(alpha=1.0, num_classes=10),
        #     CutMix(alpha=1.0, num_classes=10)
        # ]
        # combine_fn = lambda batch: mixup_cutmix[torch.randint(0,2,(1,)).item()](*torch.utils.data.default_collate(batch))


        # # dataloaders
        # train_loader = DataLoader(train_subset, batch_size=self.batch_size,
        #                           shuffle=True, num_workers=2, collate_fn=combine_fn)

        # val_loader = DataLoader(val_subset, batch_size=self.batch_size,
        #                         shuffle=False, num_workers=2)

        # test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
        #                          shuffle=False, num_workers=2)

        # --------------------------------------------------
        # DATALOADERS (NO MIXUP / CUTMIX HERE)
        # --------------------------------------------------
        train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        print("DataLoaders created successfully.")
        print(f"Training samples:   {len(train_subset)}")
        print(f"Validation samples: {len(val_subset)}")
        print(f"Test samples:       {len(test_dataset)}")

        return train_loader, val_loader, test_loader
