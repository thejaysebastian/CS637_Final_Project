from data.eurosat import get_dataloaders, get_dataloaders_split
from architectures.model_factory import build_model
from engine.train import train_model
from engine.evaluate import evaluate_model
from utils.config import load_config
import datetime
import os
import json


def main(train_fraction):
    config = load_config("configs/base_splits.yaml", "configs/densenet121_pretrained.yaml")

    split_name = f"{round(train_fraction * 100)}_{round((1.0-train_fraction) * 100)}"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if "experiment_name" not in config: 
        config["experiment_name"] = (
            f"SPLIT_{split_name}_{config['model']}_lr{config['learning_rate']}_bs{config['batch_size']}_{timestamp}"
        )

    # loading the split data 
    train_loader, val_loader, test_loader, class_names, split_info = get_dataloaders_split(config, train_fraction)

    print(f"\nExperiment: {config['experiment_name']}")
    print(f"Model: {config['model']}")
    print(f"Train/Test Split: {split_name}")
    print(f"Split Info: {split_info}")

    model = build_model(
        model_name=config["model"],
        num_classes=len(class_names),
    )
      
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )

    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        class_names=class_names,
        config=config
    )

    results["split_info"] = split_info

    save_dir = f"results/experiments/{config['experiment_name']}"
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    return config["experiment_name"]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-fraction",
        type=float,
        required=True,
        help="Fraction of full dataset used as training pool. Example: 0.8 for 80/20."
    )

    args = parser.parse_args()
    main(args.train_fraction)