# experiments/run_experiment.py

from data.eurosat import get_dataloaders
from models.model_factory import build_model
from engine.train import train_model
from engine.evaluate import evaluate_model

def main(config):
    train_loader, val_loader, test_loader, class_names = get_dataloaders(config)

    model = build_model(
        model_name=config["model"],
        num_classes=len(class_names),
        pretrained=config["pretrained"]
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