"""
Training script for Pneumonia Detection using ResNet-18.

This script fine-tunes a ResNet-18 model on the chest X-ray dataset
for binary classification (Normal vs. Pneumonia).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import datasets, models, transforms


def get_device() -> torch.device:
    """
    Detect and return the best available device (MPS for Apple Silicon, CUDA, or CPU).
    """
    if torch.backends.mps.is_available():
        print("Using Apple Silicon GPU (MPS)")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")


def get_data_loaders(
    data_dir: Path,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 2,
    use_kfold: bool = True,
    val_split: float = 0.2,
) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    """
    Create DataLoaders for training and validation sets using StratifiedKFold.

    Args:
        data_dir: Root directory containing 'train' and 'val' subdirectories
        device: Device to use (needed to determine pin_memory support)
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        use_kfold: If True, combine train/val and use StratifiedKFold for split
        val_split: Validation split ratio (used if use_kfold=True)

    Returns:
        Tuple of (train_loader, val_loader, class_weights)
    """
    # Aggressive data augmentation for training
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),  # Slightly larger for RandomResizedCrop
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # No augmentation for validation
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if use_kfold:
        # First, load datasets without transforms to get labels for stratification
        train_dataset_no_transform = datasets.ImageFolder(
            root=str(data_dir / "train"), transform=None
        )
        val_dataset_no_transform = datasets.ImageFolder(
            root=str(data_dir / "val"), transform=None
        )
        
        # Get labels for stratification (preserve order)
        train_labels = np.array([train_dataset_no_transform.targets[i] for i in range(len(train_dataset_no_transform))])
        val_labels = np.array([val_dataset_no_transform.targets[i] for i in range(len(val_dataset_no_transform))])
        all_labels = np.concatenate([train_labels, val_labels])
        
        # Use StratifiedKFold to create a proper split (20% validation)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_indices, val_indices = next(iter(skf.split(np.zeros(len(all_labels)), all_labels)))
        
        # Now create datasets with appropriate transforms
        train_dataset_with_transform = datasets.ImageFolder(
            root=str(data_dir / "train"), transform=train_transform
        )
        val_dataset_with_transform = datasets.ImageFolder(
            root=str(data_dir / "val"), transform=train_transform  # Use train_transform for combined
        )
        
        # Combine datasets (order matters: train first, then val)
        combined_train_dataset = ConcatDataset([train_dataset_with_transform, val_dataset_with_transform])
        
        # Create validation dataset with val_transform
        train_dataset_val_transform = datasets.ImageFolder(
            root=str(data_dir / "train"), transform=val_transform
        )
        val_dataset_val_transform = datasets.ImageFolder(
            root=str(data_dir / "val"), transform=val_transform
        )
        combined_val_dataset = ConcatDataset([train_dataset_val_transform, val_dataset_val_transform])
        
        # Create subsets using the stratified indices
        train_dataset = Subset(combined_train_dataset, train_indices)
        val_dataset = Subset(combined_val_dataset, val_indices)
        
        # Calculate class weights from combined dataset
        class_counts = np.bincount(all_labels)
        total_samples = len(all_labels)
        class_weights = torch.tensor(
            [total_samples / (len(class_counts) * count) for count in class_counts],
            dtype=torch.float32
        )
        
        print(f"Combined dataset: {len(all_labels)} total samples")
        print(f"Training samples: {len(train_indices)} ({len(train_indices)/len(all_labels)*100:.1f}%)")
        print(f"Validation samples: {len(val_indices)} ({len(val_indices)/len(all_labels)*100:.1f}%)")
        print(f"Class distribution - Normal: {class_counts[0]}, Pneumonia: {class_counts[1]}")
        print(f"Class weights: {class_weights.tolist()}")
    else:
        # Original behavior: use separate train/val folders
        train_dataset = datasets.ImageFolder(
            root=str(data_dir / "train"), transform=train_transform
        )
        val_dataset = datasets.ImageFolder(
            root=str(data_dir / "val"), transform=val_transform
        )
        
        # Calculate class weights from training set
        train_targets = [train_dataset.targets[i] for i in range(len(train_dataset))]
        class_counts = np.bincount(train_targets)
        total_samples = len(train_targets)
        class_weights = torch.tensor(
            [total_samples / (len(class_counts) * count) for count in class_counts],
            dtype=torch.float32
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Class weights: {class_weights.tolist()}")

    # pin_memory not supported on MPS, so disable it for MPS devices
    use_pin_memory = device.type != "mps"
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader, class_weights


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
    num_epochs: int = 10,
    learning_rate: float = 0.0001,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 3,
) -> nn.Module:
    """
    Train the model with regularization and early stopping.

    Args:
        model: The neural network model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on (MPS/CUDA/CPU)
        class_weights: Class weights for CrossEntropyLoss
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: L2 regularization strength
        early_stopping_patience: Number of epochs to wait before early stopping

    Returns:
        Trained model
    """
    # Move class weights to device
    class_weights = class_weights.to(device)
    
    # Use class weights in loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Add weight decay (L2 regularization) to optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_state = None
    epochs_without_improvement = 0

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        # Training phase
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.float() / total_samples

        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                val_total += inputs.size(0)

        val_loss = val_loss / val_total
        val_acc = val_corrects.float() / val_total

        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Early stopping: check if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            epochs_without_improvement = 0
            print(f"✓ New best validation loss: {best_val_loss:.4f} (acc: {best_val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement ({epochs_without_improvement}/{early_stopping_patience})")
            
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n⚠ Early stopping triggered after {epoch + 1} epochs")
                print(f"Best validation loss: {best_val_loss:.4f}")
                print(f"Best validation accuracy: {best_val_acc:.4f}")
                break

        model.train()

    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✓ Training complete. Best validation loss: {best_val_loss:.4f}, Best accuracy: {best_val_acc:.4f}")

    return model


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Pneumonia Detection Model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root directory containing train/val folders",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="backend/pneumonia_model.pth",
        help="Output path for saved model weights",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)

    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = get_device()

    # Load data with StratifiedKFold split
    print("Loading datasets with StratifiedKFold split...")
    train_loader, val_loader, class_weights = get_data_loaders(
        data_dir, device, batch_size=args.batch_size, num_workers=2, use_kfold=True
    )

    # Create model
    print("\nInitializing ResNet-18 model...")
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        print("Warning: Could not load pretrained weights, using random initialization")
        model = models.resnet18(weights=None)

    # Modify final layer for binary classification with dropout
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 2),
    )

    # Train with regularization and early stopping
    print(f"\nStarting training for {args.epochs} epochs...")
    print("Regularization: Dropout(0.3), Weight Decay(1e-4), Class Weighting, Early Stopping(patience=3)")
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        device,
        class_weights=class_weights,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=1e-4,
        early_stopping_patience=3,
    )

    # Save model
    print(f"\nSaving model to {output_path}...")
    torch.save(trained_model.state_dict(), output_path)
    print("✓ Model saved successfully!")


if __name__ == "__main__":
    main()
