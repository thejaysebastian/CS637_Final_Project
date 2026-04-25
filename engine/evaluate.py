import torch

from engine.metrics import compute_metrics


def evaluate_model(model, test_loader, class_names, config):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    results = compute_metrics(
        y_true=all_labels,
        y_pred=all_preds,
        class_names=class_names
    )

    return results