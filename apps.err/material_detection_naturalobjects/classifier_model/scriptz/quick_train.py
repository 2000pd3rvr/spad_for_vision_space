#!/usr/bin/env python3
"""
Quick Material Classification Training Script

This script trains a CNN classifier on the 16x16 RGB material images with fewer epochs for quick testing.
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
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from datetime import datetime
import json

class MaterialClassifier(nn.Module):
    """Simplified CNN for material classification"""
    def __init__(self, num_classes=12):
        super(MaterialClassifier, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_data_loaders(data_dir, batch_size=64):
    """Create data loaders"""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ImageFolder(os.path.join(data_dir, 'tr'), transform=train_transform)
    val_dataset = ImageFolder(os.path.join(data_dir, 'te'), transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, train_dataset.classes

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data, target in train_loader:
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
    
    return running_loss / len(train_loader), 100. * correct / total

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    return running_loss / len(val_loader), 100. * correct / total, all_predictions, all_targets

def main():
    # Configuration
    data_dir = 'data_consolidated3DLED_unequal_samples'
    batch_size = 128
    num_epochs = 20  # Reduced for quick testing
    learning_rate = 0.001
    num_classes = 12
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print('Loading data...')
    train_loader, val_loader, class_names = get_data_loaders(data_dir, batch_size)
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Validation samples: {len(val_loader.dataset)}')
    print(f'Classes: {class_names}')
    
    # Create model
    model = MaterialClassifier(num_classes=num_classes).to(device)
    print(f'Model created with {sum(p.numel() for p in model.parameters())} parameters')
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training
    print('Starting training...')
    start_time = datetime.now()
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_targets = validate_epoch(model, val_loader, criterion, device)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f'New best validation accuracy: {best_val_acc:.2f}%')
    
    training_time = datetime.now() - start_time
    print(f'\nTraining completed in {training_time}')
    
    # Load best model for final evaluation
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    print('\nFinal evaluation...')
    val_loss, val_acc, val_preds, val_targets = validate_epoch(model, val_loader, criterion, device)
    
    # Generate metrics
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
        'total_parameters': sum(p.numel() for p in model.parameters())
    }
    
    # Save results
    results = {
        'overall_metrics': overall_metrics,
        'per_class_metrics': per_class_metrics,
        'class_names': class_names
    }
    
    with open('quick_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save model
    model_save_path = f'material_classifier_quick_{best_val_acc:.2f}%.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'num_classes': num_classes,
        'best_val_accuracy': best_val_acc,
        'metrics': overall_metrics
    }, model_save_path)
    
    # Print summary
    print('\n' + '='*60)
    print('TRAINING SUMMARY')
    print('='*60)
    print(f'Best Validation Accuracy: {best_val_acc:.2f}%')
    print(f'Final Validation Accuracy: {val_acc:.2f}%')
    print(f'Training Time: {training_time}')
    print(f'Total Parameters: {overall_metrics["total_parameters"]:,}')
    print(f'Model saved to: {model_save_path}')
    print(f'Results saved to: quick_training_results.json')
    
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
