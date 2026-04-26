# run_experiment.py

from data.eurosat import get_dataloaders
from architectures.model_factory import build_model
from engine.train import train_model
from engine.evaluate import evaluate_model
from utils.config import load_config
import datetime
import os
import json


def main(config_path):
    config = load_config("configs/base.yaml", config_path)
    if "experiment_name" not in config:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config["experiment_name"] = (
            f"GOLDEN_{config['model']}_lr{config['learning_rate']}_bs{config['batch_size']}_{timestamp}"
            if "golden_run" in config and config["golden_run"]
            else f"TEST_{config['model']}_lr{config['learning_rate']}_bs{config['batch_size']}_{timestamp}"
        )
    
    train_loader, val_loader, test_loader, class_names = get_dataloaders(config)

    print(f"\nExperiment: {config['experiment_name']}")
    print(f"Initiating loading and learning of {config['dataset']} into {config['model']}....")
    
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

    print(results)
    
    save_dir = f"results/experiments/{config['experiment_name']}"
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    main(args.config)