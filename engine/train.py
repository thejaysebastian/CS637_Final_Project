# Create training logic

# This handles optimization only

import os
import time
import yaml
import copy
import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, val_loader, config):
    save_dir = f"results/experiments/{config.get('experiment_name', 'default')}"
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "final_model.pt")
    log_path = os.path.join(save_dir, "epoch_log.csv")
    
    with open(log_path, "w") as f:
        f.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc, Epoch_Time_Sec, Learning_Rate\n")

    training_start_time = time.time()
    epoch_times = []
    
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_name = config.get("optimizer", "adam").lower()
    lr = config.get("learning_rate", 0.001)
    weight_decay = config.get("weight_decay", 0.0)

    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config.get("momentum", 0.9),
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    scheduler = None
    if config.get("scheduler", "none") == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("step_size", 20),
            gamma=config.get("gamma", 0.1)
        )

    best_val_acc = 0.0
    best_model_state = None

    epochs = config.get("epochs", 50)
    best_epoch = 0
    
    try:
        for epoch in range(epochs):
            epoch_start_time = time.time()
            model.train()

            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, batch in enumerate(train_loader):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()

                outputs = model(images)

                # GoogLeNet can return aux outputs if aux_logits=True
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if batch_idx % 50 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(train_loader)}")

            train_loss = running_loss / total
            train_acc = correct / total

            val_loss, val_acc = validate_model(model, val_loader, criterion, device)
            current_lr = optimizer.param_groups[0]['lr']

            if scheduler is not None:
                scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                
                # save the model if it's the best so far, overwriting the last one
                torch.save(
                    best_model_state,
                    os.path.join(save_dir, "best_model.pt")
                )
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            
            # Print epoch results to console
            print(
                f"[Epoch {epoch+1:03d}/{epochs}] | "
                f"Epoch time: {epoch_time:.1f}s | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} || "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Best: {best_val_acc:.4f}"
            )
            
            # Log epoch results to CSV
            with open(log_path, "a") as f:
                f.write(
                    f"{epoch+1},"
                    f"{train_loss:.4f},"
                    f"{train_acc:.4f},"
                    f"{val_loss:.4f},"
                    f"{val_acc:.4f},"
                    f"{epoch_time:.2f},"
                    f"{current_lr},\n"
                )
                f.flush()  # Ensure data is written to disk
            
            #periodic checkpoints per 5 epochs.
            if (epoch + 1) % 5 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
                )
                        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...") 
        
    finally: 
        # Load best model if available
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # save the model
        torch.save(model.state_dict(), save_path)
        
        total_training_time_sec = time.time() - training_start_time
        
        # save config
        with open(os.path.join(save_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f)
        
        # save summary
        summary = {
            "best_val_acc": round(float(best_val_acc), 4),
            "best_epoch": best_epoch,
            "final_train_acc": round(float(train_acc), 4),
            "final_train_loss": round(float(train_loss), 4),
            "final_val_acc": round(float(val_acc), 4),
            "final_val_loss": round(float(val_loss), 4),
            "total_epochs": epochs,
            "total_training_time_sec": round(total_training_time_sec, 2),
            "total_training_time_min": round(total_training_time_sec / 60, 2),
            "avg_epoch_time_sec": round(sum(epoch_times) / len(epoch_times), 2),
        }
        
        with open(os.path.join(save_dir, "training_summary.yaml"), "w") as f:
            yaml.dump(summary, f)
            
        for file in os.listdir(save_dir):
            if file.startswith("checkpoint_epoch"):
                os.remove(os.path.join(save_dir, file))
    
    return model


def validate_model(model, val_loader, criterion, device):
    print("Running validation...")
    
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total

    return val_loss, val_acc