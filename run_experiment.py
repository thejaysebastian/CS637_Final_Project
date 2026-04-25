# run_experiment.py

from data.eurosat import get_dataloaders
from architectures.model_factory import build_model
from engine.train import train_model
from engine.evaluate import evaluate_model
from utils.config import load_config
import datetime


def main(config_path):
    config = load_config("configs/base.yaml", config_path)
    if "experiment_name" not in config:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config["experiment_name"] = (
            f"{config['model']}_lr{config['learning_rate']}_bs{config['batch_size']}_{timestamp}"
        )
    
    train_loader, val_loader, test_loader, class_names = get_dataloaders(config)

    print(config)
    print(class_names)
    
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
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    main(args.config)