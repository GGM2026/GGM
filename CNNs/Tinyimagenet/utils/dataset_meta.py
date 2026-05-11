
def get_num_classes(dataset: str) -> int:
    d = dataset.lower()
    if d == "imagenet":
        return 1000
    if d == "cifar10":
        return 10
    if d == "cifar100":
        return 100
    if d == "fashionmnist":
        return 10
    if d == "tinyimagenet": 
        return 200
    raise ValueError(f"Unknown dataset: {dataset}")


def get_in_chans(dataset: str) -> int:
    d = dataset.lower()
    if d == "fashionmnist":
        return 1
    if d in ("imagenet", "cifar10", "cifar100","tinyimagenet"):
        return 3
    raise ValueError(f"Unknown dataset: {dataset}")
