import torch
import torch.nn as nn
from torchvision import models
from .student_model import SmallCNN, get_resnet18_student

def get_teacher_model(model_name="resnet50", num_classes=10, pretrained=False):
    """
    Create a teacher model based on the model name
    
    Args:
        model_name (str): Name of the model architecture
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        torch.nn.Module: The teacher model
    """
    if model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model

def get_student_model(model_name="resnet18", num_classes=10, pretrained=False):
    """
    Create a student model based on the model name
    
    Args:
        model_name (str): Name of the model architecture
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        torch.nn.Module: The student model
    """
    if model_name == "small_cnn":
        return SmallCNN(num_classes=num_classes)
    elif model_name == "resnet18":
        return get_resnet18_student(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def count_parameters(model):
    """
    Count the number of trainable parameters in the model
    
    Args:
        model (torch.nn.Module): The model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 