import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_cifar10_dataloaders, load_config, set_seed, save_model, load_model
from models import get_teacher_model, get_student_model, count_parameters
from distillation_loss import DistillationLoss
import os
import mlflow
import mlflow.pytorch
import argparse
import time

def train_student(config_path="config.yaml"):
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
    
    # Create teacher model and load weights
    teacher_model = get_teacher_model(
        model_name=config["teacher"]["model"],
        num_classes=10,
        pretrained=config["teacher"]["pretrained"]
    )
    teacher_model = load_model(teacher_model, config["teacher"]["save_path"])
    teacher_model = teacher_model.to(device)
    teacher_model.eval()  # Set to evaluation mode
    
    # Create student model
    student_model = get_student_model(
        model_name=config["student"]["model"],
        num_classes=10,
        pretrained=config["student"]["pretrained"]
    )
    student_model = student_model.to(device)
    
    # Print model info
    teacher_params = count_parameters(teacher_model)
    student_params = count_parameters(student_model)
    compression_ratio = teacher_params / student_params
    
    print(f"Teacher model: {config['teacher']['model']} with {teacher_params:,} parameters")
    print(f"Student model: {config['student']['model']} with {student_params:,} parameters")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    # Define loss and optimizer
    distillation_loss = DistillationLoss(
        alpha=config["distillation"]["alpha"],
        beta=config["distillation"]["beta"],
        temperature=config["distillation"]["temperature"]
    )
    
    optimizer = optim.SGD(
        student_model.parameters(),
        lr=config["student"]["learning_rate"],
        momentum=config["student"]["momentum"],
        weight_decay=config["student"]["weight_decay"]
    )
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["student"]["scheduler_step_size"],
        gamma=config["student"]["scheduler_gamma"]
    )
    
    # Create model directory
    os.makedirs(os.path.dirname(config["student"]["save_path"]), exist_ok=True)
    
    # Start MLflow run
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=f"student_{config['student']['model']}"):
        # Log parameters
        mlflow.log_param("teacher_model", config["teacher"]["model"])
        mlflow.log_param("student_model", config["student"]["model"])
        mlflow.log_param("student_pretrained", config["student"]["pretrained"])
        mlflow.log_param("epochs", config["student"]["epochs"])
        mlflow.log_param("batch_size", config["dataset"]["batch_size"])
        mlflow.log_param("learning_rate", config["student"]["learning_rate"])
        mlflow.log_param("temperature", config["distillation"]["temperature"])
        mlflow.log_param("alpha", config["distillation"]["alpha"])
        mlflow.log_param("beta", config["distillation"]["beta"])
        mlflow.log_param("teacher_parameters", teacher_params)
        mlflow.log_param("student_parameters", student_params)
        mlflow.log_param("compression_ratio", compression_ratio)
        
        # Training loop
        best_acc = 0.0
        start_time = time.time()
        
        for epoch in range(1, config["student"]["epochs"] + 1):
            # Training phase
            student_model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass through teacher model (no gradient)
                with torch.no_grad():
                    teacher_logits = teacher_model(images)
                
                # Forward pass through student model
                optimizer.zero_grad()
                student_logits = student_model(images)
                
                # Calculate distillation loss
                loss = distillation_loss(student_logits, teacher_logits, labels)
                
                # Backward and optimize
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                _, predicted = student_logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            train_loss = running_loss / total
            train_acc = correct / total
            
            # Evaluation phase
            student_model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    
                    # Get teacher predictions
                    teacher_logits = teacher_model(images)
                    
                    # Get student predictions
                    student_logits = student_model(images)
                    
                    # Calculate distillation loss
                    loss = distillation_loss(student_logits, teacher_logits, labels)
                    
                    val_loss += loss.item() * images.size(0)
                    _, predicted = student_logits.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_loss /= val_total
            val_acc = val_correct / val_total
            
            # Print progress
            print(
                f"Epoch [{epoch}/{config['student']['epochs']}] "
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
                save_model(student_model, config["student"]["save_path"])
                print(f"Saved best model with val acc: {best_acc:.4f}")
                
                if config["mlflow"]["register_model"]:
                    mlflow.pytorch.log_model(student_model, "student_model")
            
            # Update learning rate
            scheduler.step()
        
        # Log training time and final metrics
        training_time = time.time() - start_time
        mlflow.log_metric("training_time", training_time)
        mlflow.log_metric("best_val_acc", best_acc)
        
        print(f"Training complete. Best val acc: {best_acc:.4f}")
        print(f"Total training time: {training_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train student model with knowledge distillation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    train_student(args.config)
