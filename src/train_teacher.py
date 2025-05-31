import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_cifar10_dataloaders, load_config, set_seed, save_model
from models import get_teacher_model, count_parameters
import os
import mlflow
import mlflow.pytorch
import argparse
import time

def train_teacher(config_path="config.yaml"):
    # Load configuration
    config = load_config(config_path)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get dataloaders
    train_loader, test_loader = get_cifar10_dataloaders(
        batch_size=config["dataset"]["batch_size"],
        data_dir=config["dataset"]["data_dir"]
    )
    
    # Create model
    model = get_teacher_model(
        model_name=config["teacher"]["model"],
        num_classes=10,
        pretrained=config["teacher"]["pretrained"]
    )
    model = model.to(device)
    
    # Print model info
    num_params = count_parameters(model)
    print(f"Teacher model: {config['teacher']['model']}")
    print(f"Number of parameters: {num_params:,}")
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config["teacher"]["learning_rate"],
        momentum=config["teacher"]["momentum"],
        weight_decay=config["teacher"]["weight_decay"]
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["teacher"]["scheduler_step_size"],
        gamma=config["teacher"]["scheduler_gamma"]
    )
    
    # Create model directory
    os.makedirs(os.path.dirname(config["teacher"]["save_path"]), exist_ok=True)
    
    # Start MLflow run
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=f"teacher_{config['teacher']['model']}"):
        # Log parameters
        mlflow.log_param("model", config["teacher"]["model"])
        mlflow.log_param("epochs", config["teacher"]["epochs"])
        mlflow.log_param("batch_size", config["dataset"]["batch_size"])
        mlflow.log_param("learning_rate", config["teacher"]["learning_rate"])
        mlflow.log_param("momentum", config["teacher"]["momentum"])
        mlflow.log_param("weight_decay", config["teacher"]["weight_decay"])
        mlflow.log_param("num_parameters", num_params)
        
        # Training loop
        best_acc = 0.0
        start_time = time.time()
        
        for epoch in range(1, config["teacher"]["epochs"] + 1):
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            train_loss = running_loss / total
            train_acc = correct / total
            
            # Evaluation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_loss /= val_total
            val_acc = val_correct / val_total
            
            # Print progress
            print(
                f"Epoch [{epoch}/{config['teacher']['epochs']}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )
            
            # Log metrics to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                save_model(model, config["teacher"]["save_path"])
                print(f"Saved best model with val acc: {best_acc:.4f}")
                
                if config["mlflow"]["register_model"]:
                    mlflow.pytorch.log_model(model, "teacher_model")
            
            # Update learning rate
            scheduler.step()
        
        # Log training time
        training_time = time.time() - start_time
        mlflow.log_metric("training_time", training_time)
        print(f"Training complete. Best val acc: {best_acc:.4f}")
        print(f"Total training time: {training_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train teacher model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    train_teacher(args.config)
