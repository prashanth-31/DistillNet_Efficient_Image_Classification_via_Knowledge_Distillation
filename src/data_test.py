from utils import get_cifar10_dataloaders

if __name__ == "__main__":
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=64)
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of testing batches: {len(test_loader)}")

    # Iterate through one batch to check shape
    images, labels = next(iter(train_loader))
    print(f"Images batch shape: {images.shape}")
    print(f"Labels batch shape: {labels.shape}")
