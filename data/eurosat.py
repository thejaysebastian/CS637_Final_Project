# Set transforms and other data below to match the EuroSat paper


from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms


def build_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

class EuroSatTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, batch):
        batch["image"] = [self.transform(img) for img in batch["image"]]
        return batch

def get_dataloaders(config):
    dataset = load_dataset("blanchon/EuroSAT_RGB")

    transform = build_transform(config["image_size"])

    dataset = dataset.with_transform(EuroSatTransform(transform))

    train_loader = DataLoader(
        dataset["train"],
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"]
    )

    val_loader = DataLoader(
        dataset["validation"],
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )

    test_loader = DataLoader(
        dataset["test"],
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )

    class_names = dataset["train"].features["label"].names

    return train_loader, val_loader, test_loader, class_names