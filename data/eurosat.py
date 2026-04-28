# Set transforms and other data below to match the EuroSat paper


from datasets import load_dataset, concatenate_datasets
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

def get_dataloaders_split(config, train_fraction): 
    dataset = load_dataset("blanchon/EuroSAT_RGB")

    # recombining the current Hugging face splits into a single dataset
    full_dataset = concatenate_datasets([
        dataset["train"], 
        dataset["validation"], 
        dataset["test"]
    ])

    # splitting the data into training and testing sets!
    data_split = full_dataset.train_test_split(
        train_size = train_fraction, 
        seed = config.get("seed", 42), 
        stratify_by_column="label"
    )

    train_pool = data_split["train"]
    test_dataset = data_split["test"]

    # splitting the training pool into training and validation!
    validation_fraction = 42
    inner_split = train_pool.train_test_split(
        test_size=validation_fraction, 
        seed = config.get("seed", 42), 
        stratify_by_column="label"
    )

    train_dataset = inner_split["train"]
    val_dataset = inner_split["test"]

    # same preprocessing...
    transform = build_transform(config["image_size"])

    train_dataset = train_dataset.with_transform(EuroSatTransform(transform))
    val_dataset = val_dataset.with_transform(EuroSatTransform(transform))
    test_dataset = test_dataset.with_transform(EuroSatTransform(transform))

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        num_workers=config["num_workers"]
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        num_workers=config["num_workers"]
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        num_workers=config["num_workers"]
    )

    class_names = dataset["train"].features["label"].names
    
    split_info = {
        "train_fraction": train_fraction, 
        "test_fraction": round(1.0 - train_fraction), 
        "train_size": len(train_dataset) + len(val_dataset), 
        "val_fraction_of_train_pool": validation_fraction, 
        "seed": config.get("seed", 42),
    }

    return train_loader, val_loader, test_loader, class_names, split_info
