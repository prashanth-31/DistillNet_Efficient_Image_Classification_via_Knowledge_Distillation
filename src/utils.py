import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import yaml
import os

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_cifar10_dataloaders(batch_size=128, data_dir='./data'):
    # Define normalization (mean & std from CIFAR-10)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.247, 0.243, 0.261))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.247, 0.243, 0.261))
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, 
                                     download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, 
                                    download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader

def set_seed(seed=42):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def save_model(model, path):
    """Save model to disk"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    
def load_model(model, path):
    """Load model from disk"""
    model.load_state_dict(torch.load(path))
    return model
