import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
from utils import get_cifar10_dataloaders, load_config, load_model, set_seed
from models import get_teacher_model, get_student_model, count_parameters
import argparse
import os
import mlflow
from torchvision import transforms
from PIL import Image

def evaluate_model(model, test_loader, device):
    """Evaluate model on test dataset"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0
    
    # For confusion matrix
    all_preds = []
    all_targets = []
    
    # For inference time measurement
    inference_times = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(images)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Save predictions and targets for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    avg_loss = test_loss / total
    avg_inference_time = sum(inference_times) / len(inference_times)
    batch_inference_time = sum(inference_times)
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'avg_inference_time': avg_inference_time,
        'batch_inference_time': batch_inference_time,
        'predictions': np.array(all_preds),
        'targets': np.array(all_targets)
    }

def compare_models(config_path="config.yaml"):
    """Compare teacher and student models"""
    # Load configuration
    config = load_config(config_path)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get dataloaders
    _, test_loader = get_cifar10_dataloaders(
        batch_size=config["dataset"]["batch_size"],
        data_dir=config["dataset"]["data_dir"]
    )
    
    # Create and load teacher model
    teacher_model = get_teacher_model(
        model_name=config["teacher"]["model"],
        num_classes=10,
        pretrained=False
    )
    teacher_model = load_model(teacher_model, config["teacher"]["save_path"])
    teacher_model = teacher_model.to(device)
    
    # Create and load student model
    student_model = get_student_model(
        model_name=config["student"]["model"],
        num_classes=10,
        pretrained=False
    )
    student_model = load_model(student_model, config["student"]["save_path"])
    student_model = student_model.to(device)
    
    # Count parameters
    teacher_params = count_parameters(teacher_model)
    student_params = count_parameters(student_model)
    compression_ratio = teacher_params / student_params
    
    print(f"Teacher model: {config['teacher']['model']} with {teacher_params:,} parameters")
    print(f"Student model: {config['student']['model']} with {student_params:,} parameters")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    # Evaluate models
    print("\nEvaluating teacher model...")
    teacher_results = evaluate_model(teacher_model, test_loader, device)
    
    print("\nEvaluating student model...")
    student_results = evaluate_model(student_model, test_loader, device)
    
    # Print results
    print("\n" + "="*50)
    print("Model Comparison")
    print("="*50)
    print(f"Teacher accuracy: {teacher_results['accuracy']:.4f}")
    print(f"Student accuracy: {student_results['accuracy']:.4f}")
    print(f"Accuracy difference: {teacher_results['accuracy'] - student_results['accuracy']:.4f}")
    print("-"*50)
    print(f"Teacher avg inference time: {teacher_results['avg_inference_time']*1000:.2f} ms")
    print(f"Student avg inference time: {student_results['avg_inference_time']*1000:.2f} ms")
    print(f"Speedup: {teacher_results['avg_inference_time'] / student_results['avg_inference_time']:.2f}x")
    print("-"*50)
    print(f"Teacher parameters: {teacher_params:,}")
    print(f"Student parameters: {student_params:,}")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    # Log to MLflow
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="model_comparison"):
        # Log metrics
        mlflow.log_metric("teacher_accuracy", teacher_results['accuracy'])
        mlflow.log_metric("student_accuracy", student_results['accuracy'])
        mlflow.log_metric("accuracy_difference", teacher_results['accuracy'] - student_results['accuracy'])
        mlflow.log_metric("teacher_inference_time", teacher_results['avg_inference_time'])
        mlflow.log_metric("student_inference_time", student_results['avg_inference_time'])
        mlflow.log_metric("speedup", teacher_results['avg_inference_time'] / student_results['avg_inference_time'])
        mlflow.log_metric("teacher_parameters", teacher_params)
        mlflow.log_metric("student_parameters", student_params)
        mlflow.log_metric("compression_ratio", compression_ratio)
    
    return teacher_results, student_results

def predict_image(image_path, model_path, model_type="student", config_path="config.yaml"):
    """Predict class of a single image"""
    # Load configuration
    config = load_config(config_path)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load image
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Load model
    if model_type == "teacher":
        model = get_teacher_model(
            model_name=config["teacher"]["model"],
            num_classes=10,
            pretrained=False
        )
        model = load_model(model, config["teacher"]["save_path"])
    else:
        model = get_student_model(
            model_name=config["student"]["model"],
            num_classes=10,
            pretrained=False
        )
        model = load_model(model, model_path)
    
    model = model.to(device)
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        start_time = time.time()
        outputs = model(image_tensor)
        inference_time = time.time() - start_time
        
        _, predicted = outputs.max(1)
        probability = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    # CIFAR-10 classes
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    result = {
        'class_id': predicted.item(),
        'class_name': classes[predicted.item()],
        'probability': probability[predicted].item(),
        'inference_time': inference_time * 1000  # Convert to ms
    }
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and compare models")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--image", type=str, help="Path to image for prediction")
    parser.add_argument("--model", type=str, choices=["teacher", "student"], default="student", help="Model to use for prediction")
    
    args = parser.parse_args()
    
    if args.image:
        # Predict single image
        model_path = config["student"]["save_path"] if args.model == "student" else config["teacher"]["save_path"]
        result = predict_image(args.image, model_path, args.model, args.config)
        print(f"Predicted class: {result['class_name']} (ID: {result['class_id']})")
        print(f"Confidence: {result['probability']:.4f}")
        print(f"Inference time: {result['inference_time']:.2f} ms")
    else:
        # Compare models
        compare_models(args.config) 