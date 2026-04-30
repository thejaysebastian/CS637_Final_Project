import torch

from engine.metrics import compute_metrics
from utils.device import get_device


def evaluate_model(model, test_loader, class_names, config):
#    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    device = get_device(config.get("device", "auto"))
    print(f"Using device: {device}")

    model = model.to(device)
    print("\n=== Final Test Evaluation ===")
    
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
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision (macro): {results['precision_macro']:.4f}")
    print(f"Recall (macro): {results['recall_macro']:.4f}")
    print(f"F1 (macro): {results['f1_macro']:.4f}")
    
    print("\nPer-class performance:")
    for cls, stats in results["classification_report"].items():
        if isinstance(stats, dict):
            print(f"{cls}: F1={stats['f1-score']:.3f}")

    return results