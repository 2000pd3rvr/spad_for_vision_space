#!/usr/bin/env python3
"""
Material Classification Training Script

This script trains a CNN classifier on the 16x16 RGB material images.
Dataset: 12 classes with 28,800 training and 7,200 test images.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from datetime import datetime
import json
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MaterialClassifier(nn.Module):
    """
    CNN architecture for material classification
    Optimized for 16x16 RGB images with 12 classes
    """
    def __init__(self, num_classes=12):
        super(MaterialClassifier, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Global average pooling instead of flattening
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

def get_data_loaders(data_dir, batch_size=64, num_workers=0):
    """Create data loaders for training and validation"""
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # No augmentation for validation
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ImageFolder(
        root=os.path.join(data_dir, 'tr'),
        transform=train_transform
    )
    
    val_dataset = ImageFolder(
        root=os.path.join(data_dir, 'te'),
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.classes

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Create progress bar for training
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs} [Train]', 
                leave=False, ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Update progress bar with current metrics
        current_acc = 100. * correct / total
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{current_acc:.2f}%',
            'Avg_Loss': f'{running_loss/(batch_idx+1):.4f}'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device, epoch, total_epochs):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    # Create progress bar for validation
    pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{total_epochs} [Val]', 
                leave=False, ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            # Update progress bar with current metrics
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%',
                'Avg_Loss': f'{running_loss/(batch_idx+1):.4f}'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_predictions, all_targets

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(train_accs, label='Training Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Configuration
    data_dir = 'data_consolidated3DLED_unequal_samples'
    batch_size = 64  # Reduced for CPU training
    num_epochs = 50  # Reduced for faster completion
    learning_rate = 0.001
    num_classes = 12
    
    # Create output directory
    output_dir = 'training_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Force CPU usage to avoid MPS multiprocessing issues
    device = torch.device('cpu')
    print(f'Using device: {device} (forced CPU)')
    
    # Load data
    print('Loading data...')
    train_loader, val_loader, class_names = get_data_loaders(data_dir, batch_size, num_workers=0)
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Validation samples: {len(val_loader.dataset)}')
    print(f'Classes: {class_names}')
    
    # Create model
    model = MaterialClassifier(num_classes=num_classes).to(device)
    print(f'Model created with {sum(p.numel() for p in model.parameters())} parameters')
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    best_model_state = None
    
    print('Starting training...')
    start_time = datetime.now()
    
    # Main training loop with progress bars
    for epoch in tqdm(range(num_epochs), desc='Training Progress', ncols=100):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs)
        
        # Validate
        val_loss, val_acc, val_preds, val_targets = validate_epoch(model, val_loader, criterion, device, epoch, num_epochs)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Store history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{num_epochs} Summary:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f'  🎯 New best validation accuracy: {best_val_acc:.2f}%')
        
        # Early stopping check
        if epoch > 20 and val_acc < max(val_accs[-20:]) - 5:
            print('  ⏹️  Early stopping triggered')
            break
    
    training_time = datetime.now() - start_time
    print(f'\nTraining completed in {training_time}')
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f'Loaded best model with validation accuracy: {best_val_acc:.2f}%')
    
    # Final evaluation
    print('\nFinal evaluation...')
    val_loss, val_acc, val_preds, val_targets = validate_epoch(model, val_loader, criterion, device, num_epochs-1, num_epochs)
    
    # Generate detailed metrics
    print('\nGenerating detailed performance metrics...')
    
    # Classification report
    report = classification_report(val_targets, val_preds, target_names=class_names, output_dict=True)
    
    # Per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'precision': report[class_name]['precision'],
            'recall': report[class_name]['recall'],
            'f1_score': report[class_name]['f1-score'],
            'support': report[class_name]['support']
        }
    
    # Overall metrics
    overall_metrics = {
        'accuracy': report['accuracy'],
        'macro_avg_precision': report['macro avg']['precision'],
        'macro_avg_recall': report['macro avg']['recall'],
        'macro_avg_f1': report['macro avg']['f1-score'],
        'weighted_avg_precision': report['weighted avg']['precision'],
        'weighted_avg_recall': report['weighted avg']['recall'],
        'weighted_avg_f1': report['weighted avg']['f1-score'],
        'best_val_accuracy': best_val_acc,
        'training_time': str(training_time),
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    # Save metrics
    metrics = {
        'overall_metrics': overall_metrics,
        'per_class_metrics': per_class_metrics,
        'class_names': class_names,
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accs,
            'val_accuracies': val_accs
        }
    }
    
    with open(os.path.join(output_dir, 'detailed_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save model
    model_save_path = os.path.join(output_dir, f'material_classifier_best_{best_val_acc:.2f}%.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'num_classes': num_classes,
        'best_val_accuracy': best_val_acc,
        'metrics': overall_metrics
    }, model_save_path)
    
    # Generate plots
    plot_training_history(train_losses, val_losses, train_accs, val_accs, 
                         os.path.join(output_dir, 'training_history.png'))
    plot_confusion_matrix(val_targets, val_preds, class_names, 
                         os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Print summary
    print('\n' + '='*60)
    print('TRAINING SUMMARY')
    print('='*60)
    print(f'Best Validation Accuracy: {best_val_acc:.2f}%')
    print(f'Final Validation Accuracy: {val_acc:.2f}%')
    print(f'Training Time: {training_time}')
    print(f'Total Parameters: {overall_metrics["total_parameters"]:,}')
    print(f'Model saved to: {model_save_path}')
    print(f'Metrics saved to: {os.path.join(output_dir, "detailed_metrics.json")}')
    print(f'Plots saved to: {output_dir}')
    
    print('\nPer-class Performance:')
    print('-' * 60)
    for class_name, metrics in per_class_metrics.items():
        print(f'{class_name:20s}: Precision={metrics["precision"]:.3f}, '
              f'Recall={metrics["recall"]:.3f}, F1={metrics["f1_score"]:.3f}, '
              f'Support={metrics["support"]}')
    
    print('\nOverall Metrics:')
    print('-' * 60)
    print(f'Accuracy: {overall_metrics["accuracy"]:.3f}')
    print(f'Macro Avg F1: {overall_metrics["macro_avg_f1"]:.3f}')
    print(f'Weighted Avg F1: {overall_metrics["weighted_avg_f1"]:.3f}')

if __name__ == '__main__':
    main()
