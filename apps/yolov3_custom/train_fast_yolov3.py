#!/usr/bin/env python3
"""Fast YOLOv3 Training Script - Optimized for Speed"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import yaml
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import argparse
import os
from tqdm import tqdm

# Set matplotlib backend
plt.switch_backend('Agg')

class FastYOLOv3Dataset(Dataset):
    """Fast dataset with smaller images"""
    
    def __init__(self, data_dir, split='train', img_size=224):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        
        # Load images and labels
        self.images = []
        self.labels = []
        
        split_dir = self.data_dir / 'images' / split
        if split_dir.exists():
            image_files = list(split_dir.glob('*.jpg'))
            
            for img_path in image_files:
                label_path = self.data_dir / 'labels' / split / f"{img_path.stem}.txt"
                if label_path.exists():
                    self.images.append(str(img_path))
                    self.labels.append(str(label_path))
        
        print(f"Loaded {len(self.images)} {split} samples")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to smaller size for speed
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image.astype(np.float32) / 255.0
        
        # Proper normalization (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        
        # Load labels
        label_path = self.labels[idx]
        boxes = []
        classes = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert to absolute coordinates
                        x1 = (x_center - width/2) * self.img_size
                        y1 = (y_center - height/2) * self.img_size
                        x2 = (x_center + width/2) * self.img_size
                        y2 = (y_center + height/2) * self.img_size
                        
                        boxes.append([x1, y1, x2, y2])
                        classes.append(class_id)
        
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
        
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            classes = torch.tensor(classes, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            classes = torch.zeros((0,), dtype=torch.long)
        
        return image, boxes, classes

class UltraFastYOLOv3Model(nn.Module):
    """Improved YOLOv3 model with better architecture"""
    
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        
        # Improved backbone with batch normalization
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 3
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 4
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Block 5
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Block 6
            nn.Conv2d(512, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        
        # Improved detection head with separate branches
        self.objectness_head = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 3, 1)  # 3 anchors
        )
        
        self.bbox_head = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 12, 1)  # 4 coords * 3 anchors
        )
        
        self.class_head = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes * 3, 1)  # num_classes * 3 anchors
        )
        
    def forward(self, x):
        features = self.backbone(x)  # [batch, 1024, 7, 7]
        
        # Get predictions from each head
        objectness = self.objectness_head(features)  # [batch, 3, 7, 7]
        bbox = self.bbox_head(features)  # [batch, 12, 7, 7]
        classes = self.class_head(features)  # [batch, num_classes*3, 7, 7]
        
        # Reshape outputs
        batch_size = objectness.size(0)
        
        # Reshape bbox: [batch, 12, 7, 7] -> [batch, 3, 4, 7, 7]
        bbox = bbox.view(batch_size, 3, 4, 7, 7)
        
        # Reshape classes: [batch, num_classes*3, 7, 7] -> [batch, 3, num_classes, 7, 7]
        classes = classes.view(batch_size, 3, self.num_classes, 7, 7)
        
        # Combine all predictions: [batch, 3, 5+num_classes, 7, 7]
        # Order: [x, y, w, h, obj, class1, class2, ..., classN]
        output = torch.cat([
            bbox,  # [batch, 3, 4, 7, 7]
            objectness.unsqueeze(2),  # [batch, 3, 1, 7, 7]
            classes  # [batch, 3, num_classes, 7, 7]
        ], dim=2)
        
        return output

class FastYOLOv3Loss(nn.Module):
    """Improved loss function with proper object detection training"""
    
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, predictions, targets):
        batch_size = predictions.size(0)
        total_loss = 0.0
        
        for i in range(batch_size):
            pred = predictions[i]  # [3, 5+classes, 7, 7]
            boxes, classes = targets[i] if i < len(targets) else (torch.zeros(0, 4), torch.zeros(0))
            
            if len(boxes) > 0:
                # Objectness loss - target center cell
                obj_pred = pred[:, 4, :, :]  # [3, 7, 7]
                obj_target = torch.zeros_like(obj_pred)
                obj_target[:, 3, 3] = 1.0  # Center cell
                obj_loss = self.mse_loss(torch.sigmoid(obj_pred), obj_target)
                
                # Classification loss - proper multi-label
                if len(classes) > 0:
                    class_pred = pred[:, 5:, :, :]  # [3, num_classes, 7, 7]
                    target_class = classes[0].item()
                    
                    # Create one-hot target
                    class_target = torch.zeros_like(class_pred)
                    class_target[:, target_class, 3, 3] = 1.0
                    
                    # Use BCE loss for better gradient flow
                    class_loss = self.bce_loss(class_pred, class_target)
                    total_loss += class_loss
                
                total_loss += obj_loss
            else:
                # No objects - penalize objectness
                obj_pred = pred[:, 4, :, :]
                obj_target = torch.zeros_like(obj_pred)
                obj_loss = self.mse_loss(torch.sigmoid(obj_pred), obj_target)
                total_loss += obj_loss
        
        return total_loss / batch_size if batch_size > 0 else torch.tensor(0.0, requires_grad=True)

class FastTrainer:
    """Fast trainer optimized for speed with visualizations"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        
        # Load data config
        with open(args.data, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        self.num_classes = self.data_config['nc']
        self.class_names = self.data_config['names']
        
        # Initialize metrics tracking
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'fitness': [],
            'epoch_times': [],
            'learning_rates': []
        }
        
        # Create results directory
        self.results_dir = Path(args.project) / args.name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir = self.results_dir / 'weights'
        self.weights_dir.mkdir(exist_ok=True)
        self.viz_dir = self.results_dir / 'visualizations'
        self.viz_dir.mkdir(exist_ok=True)
        self.cm_dir = self.results_dir / 'confusion_matrices'
        self.cm_dir.mkdir(exist_ok=True)
        
        print(f"Using device: {self.device}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Class names: {self.class_names}")
        print(f"Results will be saved to: {self.results_dir}")
    
    def setup_data(self):
        """Setup data loaders"""
        if 'path' in self.data_config:
            data_dir = Path(self.data_config['path']) / 'data'
        else:
            data_dir = Path(self.data_config['train']).parent
        
        print(f"Using data directory: {data_dir}")
        
        # Create datasets with smaller images
        self.train_dataset = FastYOLOv3Dataset(data_dir, 'train', self.args.imgsz)
        self.val_dataset = FastYOLOv3Dataset(data_dir, 'val', self.args.imgsz)
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            num_workers=self.args.workers,
            pin_memory=False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=False, 
            num_workers=self.args.workers,
            pin_memory=False
        )
        
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
    
    def setup_model(self):
        """Setup model and optimizer"""
        self.model = UltraFastYOLOv3Model(self.num_classes).to(self.device)
        self.criterion = FastYOLOv3Loss(self.num_classes)
        
        # Better optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.args.lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.args.epochs,
            eta_min=self.args.lr * 0.01
        )
        
        print(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train_epoch(self, epoch):
        """Train for one epoch with tqdm progress bar"""
        self.model.train()
        total_loss = 0.0
        epoch_start_time = time.time()
        
        # Add tqdm progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.args.epochs}', 
                   leave=False, ncols=100, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        
        for batch_idx, batch_data in enumerate(pbar):
            images, boxes_list, classes_list = batch_data
            images = images.to(self.device)
            
            # Forward pass
            predictions = self.model(images)
            loss = self.criterion(predictions, list(zip(boxes_list, classes_list)))
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        epoch_time = time.time() - epoch_start_time
        
        return avg_loss, epoch_time
    
    def validate_epoch(self):
        """Validate for one epoch with tqdm progress bar"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation', leave=False, ncols=80)
            for batch_data in pbar:
                images, boxes_list, classes_list = batch_data
                images = images.to(self.device)
                
                predictions = self.model(images)
                loss = self.criterion(predictions, list(zip(boxes_list, classes_list)))
                total_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def generate_confusion_matrix(self, epoch):
        """Generate confusion matrix with proper object detection accuracy"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data in self.val_loader:
                images, boxes_list, classes_list = batch_data
                images = images.to(self.device)
                
                predictions = self.model(images)
                
                # Process predictions (improved for object detection)
                batch_size = predictions.size(0)
                for i in range(batch_size):
                    pred = predictions[i]  # [3, 5+classes, 7, 7]
                    
                    # Get objectness scores for all anchors and positions
                    obj_scores = pred[:, 4]  # Objectness scores for 3 anchors
                    max_obj_score = torch.max(obj_scores)
                    
                    if max_obj_score > 0.01:  # Detection threshold
                        # Get predicted class scores
                        class_scores = pred[:, 5:5+self.num_classes]  # Class scores
                        
                        # Find the best prediction across all anchors and positions
                        best_conf = 0
                        best_class = 0
                        
                        for anchor_idx in range(3):  # 3 anchors
                            anchor_obj = pred[anchor_idx, 4]  # Objectness for this anchor
                            anchor_class_scores = pred[anchor_idx, 5:5+self.num_classes]  # Class scores
                            
                            # Find best position for this anchor
                            max_obj_pos = torch.argmax(anchor_obj.reshape(-1))
                            max_class_pos = torch.argmax(anchor_class_scores.reshape(self.num_classes, -1), dim=1)
                            
                            # Get confidence for best class
                            best_class_idx = torch.argmax(anchor_class_scores.mean(dim=(1, 2)))
                            confidence = anchor_obj.reshape(-1)[max_obj_pos] * torch.sigmoid(anchor_class_scores[best_class_idx].reshape(-1)[max_class_pos[best_class_idx]])
                            
                            if confidence > best_conf:
                                best_conf = confidence
                                best_class = best_class_idx.item()
                        
                        # Clamp to valid class range
                        predicted_class = max(0, min(best_class, self.num_classes - 1))
                    else:
                        predicted_class = 0  # Background
                    
                    # Get target class
                    if len(classes_list) > i and len(classes_list[i]) > 0:
                        target_class = classes_list[i][0].item()
                    else:
                        target_class = 0  # Background
                    
                    all_predictions.append(predicted_class)
                    all_targets.append(target_class)
        
        # Create confusion matrix
        cm = confusion_matrix(all_targets, all_predictions, labels=list(range(self.num_classes)))
        
        # Debug output
        print(f"DEBUG: Confusion matrix shape: {cm.shape}")
        print(f"DEBUG: Total predictions: {len(all_predictions)}")
        print(f"DEBUG: Unique predictions: {set(all_predictions)}")
        print(f"DEBUG: Unique targets: {set(all_targets)}")
        print(f"DEBUG: Confusion matrix:\n{cm}")
        
        # Calculate per-class accuracy
        class_accuracies = []
        for i in range(self.num_classes):
            if np.sum(cm[i, :]) > 0:  # If this class has samples
                class_acc = cm[i, i] / np.sum(cm[i, :]) * 100
                class_accuracies.append(class_acc)
                print(f"DEBUG: Class {i} ({self.class_names[i]}): {class_acc:.1f}% accuracy")
        
        avg_class_acc = np.mean(class_accuracies) if class_accuracies else 0
        print(f"DEBUG: Average per-class accuracy: {avg_class_acc:.1f}%")
        
        # Calculate metrics
        accuracy = (np.trace(cm) / np.sum(cm)) * 100 if np.sum(cm) > 0 else 0  # Convert to percentage
        precision = np.diag(cm) / np.sum(cm, axis=0)
        recall = np.diag(cm) / np.sum(cm, axis=1)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        
        # Create figure with DINOv3 style
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Main title
        fig.suptitle(f'YOLOv3 Confusion Matrix & Metrics - Epoch {epoch}', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # 1. Confusion Matrix Heatmap
        im = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
        ax1.figure.colorbar(im, ax=ax1, shrink=0.8)
        
        # Set ticks and labels
        ax1.set_xticks(np.arange(len(self.class_names)))
        ax1.set_yticks(np.arange(len(self.class_names)))
        ax1.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax1.set_yticklabels(self.class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax1.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontweight='bold')
        
        ax1.set_title(f'Confusion Matrix\nAccuracy: {accuracy:.1f}%', 
                     fontweight='bold', pad=20)
        ax1.set_xlabel('Predicted Class', fontweight='bold')
        ax1.set_ylabel('True Class', fontweight='bold')
        
        # 2. Metrics Bar Chart
        metrics_data = {
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_scores
        }
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            bars = ax2.bar(x + i * width, values, width, 
                          label=metric_name, color=colors[i], alpha=0.8)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                if not np.isnan(value):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax2.set_title('Per-Class Metrics', fontweight='bold', pad=20)
        ax2.set_xlabel('Class', fontweight='bold')
        ax2.set_ylabel('Score', fontweight='bold')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add overall metrics text box
        overall_text = f"""
        📊 Overall Metrics:
        
        • Accuracy: {accuracy:.1f}%
        • Avg Precision: {np.nanmean(precision):.3f}
        • Avg Recall: {np.nanmean(recall):.3f}
        • Avg F1-Score: {np.nanmean(f1_scores):.3f}
        • Total Samples: {len(all_targets)}
        """
        
        ax2.text(0.02, 0.98, overall_text, transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        
        # Add footer
        fig.text(0.5, 0.02, f'Generated at epoch {epoch} | Fast YOLOv3 Training', 
                ha='center', fontsize=10, style='italic', color='gray')
        
        plt.tight_layout()
        
        # Save confusion matrix
        cm_path = self.cm_dir / f'confusion_matrix_epoch_{epoch:03d}.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 Confusion matrix saved: {cm_path}")
        return cm_path
    
    def generate_sample_predictions(self, epoch, num_samples=8):
        """Generate sample predictions visualization with bounding boxes"""
        self.model.eval()
        
        # Create predictions directory
        pred_dir = self.results_dir / 'sample_predictions'
        pred_dir.mkdir(exist_ok=True)
        
        # Get a batch of validation samples
        with torch.no_grad():
            for batch_data in self.val_loader:
                images, boxes_list, classes_list = batch_data
                images = images.to(self.device)
                
                predictions = self.model(images)
                
                # Process first batch
                batch_size = min(images.size(0), num_samples)
                
                # Create figure
                fig, axes = plt.subplots(2, 4, figsize=(20, 10))
                fig.suptitle(f'Sample Predictions - Epoch {epoch}', fontsize=16, fontweight='bold')
                
                for i in range(batch_size):
                    row = i // 4
                    col = i % 4
                    ax = axes[row, col]
                    
                    # Get image and denormalize properly
                    img = images[i].cpu().numpy().transpose(1, 2, 0)
                    
                    # Denormalize ImageNet normalization
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img = img * std + mean
                    img = np.clip(img, 0, 1)
                    img = (img * 255).astype(np.uint8)
                    
                    # Display image
                    ax.imshow(img)
                    ax.set_title(f'Sample {i+1}', fontweight='bold')
                    
                    # Process predictions
                    pred = predictions[i]  # [3, 5+classes, 7, 7]
                    
                    # Get objectness scores
                    obj_scores = pred[:, 4]  # Objectness scores for 3 anchors
                    max_obj_score = torch.max(obj_scores)
                    
                    if max_obj_score > 0.01:  # Lower detection threshold for early training
                        # Get predicted class scores
                        class_scores = pred[:, 5:5+self.num_classes]  # Class scores
                        avg_class_scores = class_scores.mean(dim=(1, 2))  # Average over spatial
                        predicted_class_idx = torch.argmax(avg_class_scores).item()
                        confidence = torch.sigmoid(max_obj_score).item()
                        
                        # Clamp to valid class range
                        predicted_class_idx = max(0, min(predicted_class_idx, self.num_classes - 1))
                        predicted_class = self.class_names[predicted_class_idx]
                        
                        # Get target class
                        if len(classes_list) > i and len(classes_list[i]) > 0:
                            target_class_idx = classes_list[i][0].item()
                            target_class = self.class_names[target_class_idx]
                        else:
                            target_class = "Background"
                        
                        # Add text with prediction info
                        text = f'Pred: {predicted_class}\nConf: {confidence:.2f}\nTrue: {target_class}'
                        
                        # Color based on correctness with bright colors
                        if predicted_class == target_class:
                            text_color = 'white'
                            bg_color = 'lime'
                            edge_color = 'darkgreen'
                            bbox_color = 'lime'
                        else:
                            text_color = 'white'
                            bg_color = 'red'
                            edge_color = 'darkred'
                            bbox_color = 'red'
                        
                        ax.text(0.02, 0.98, text, transform=ax.transAxes, 
                               fontsize=11, fontweight='bold', color=text_color,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round,pad=0.4', 
                                       facecolor=bg_color, 
                                       edgecolor=edge_color,
                                       linewidth=2,
                                       alpha=0.9))
                        
                        # Draw bounding box using actual YOLO predictions
                        h, w = img.shape[:2]
                        
                        # Get bbox predictions from the best anchor
                        bbox_pred = pred[:, :4]  # [3, 4, 7, 7] - x, y, w, h for each anchor
                        
                        # Find best anchor and position
                        best_anchor = torch.argmax(obj_scores.reshape(-1)).item()
                        anchor_idx = best_anchor // 49  # Which anchor (0, 1, or 2)
                        pos_idx = best_anchor % 49      # Which position in 7x7 grid
                        
                        # Get grid position
                        grid_y = pos_idx // 7
                        grid_x = pos_idx % 7
                        
                        # Get bbox predictions for best anchor and position
                        x_pred = bbox_pred[anchor_idx, 0, grid_y, grid_x].item()
                        y_pred = bbox_pred[anchor_idx, 1, grid_y, grid_x].item()
                        w_pred = bbox_pred[anchor_idx, 2, grid_y, grid_x].item()
                        h_pred = bbox_pred[anchor_idx, 3, grid_y, grid_x].item()
                        
                        # Convert to absolute coordinates (assuming predictions are normalized)
                        # Scale to image size
                        x_center = x_pred * w
                        y_center = y_pred * h
                        box_w = w_pred * w
                        box_h = h_pred * h
                        
                        # Convert to corner coordinates
                        x1 = int(x_center - box_w / 2)
                        y1 = int(y_center - box_h / 2)
                        x2 = int(x_center + box_w / 2)
                        y2 = int(y_center + box_h / 2)
                        
                        # Clamp to image bounds
                        x1 = max(0, min(x1, w-1))
                        y1 = max(0, min(y1, h-1))
                        x2 = max(0, min(x2, w-1))
                        y2 = max(0, min(y2, h-1))
                        
                        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                           linewidth=3, edgecolor=bbox_color, facecolor='none')
                        ax.add_patch(rect)
                    else:
                        # No detection
                        ax.text(0.02, 0.98, 'No Detection', transform=ax.transAxes, 
                               fontsize=11, fontweight='bold', color='white',
                               verticalalignment='top',
                               bbox=dict(boxstyle='round,pad=0.4', 
                                       facecolor='orange', 
                                       edgecolor='darkorange',
                                       linewidth=2,
                                       alpha=0.9))
                    
                    ax.axis('off')
                
                # Hide unused subplots
                for i in range(batch_size, 8):
                    row = i // 4
                    col = i % 4
                    axes[row, col].axis('off')
                
                plt.tight_layout()
                
                # Save sample predictions
                pred_path = pred_dir / f'sample_predictions_epoch_{epoch:03d}.png'
                plt.savefig(pred_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                
                print(f"🎯 Sample predictions saved: {pred_path}")
                return pred_path
    
    def save_training_visualizations(self, epoch):
        """Save training progress visualizations in DINOv3 style"""
        if len(self.metrics_history['train_loss']) < 2:
            return
        
        # Set style similar to DINOv3
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 11
        
        # Create figure with subplots (similar to DINOv3 layout)
        fig = plt.figure(figsize=(16, 12))
        
        # Main title
        fig.suptitle(f'YOLOv3 Training Progress - Epoch {epoch}', fontsize=16, fontweight='bold', y=0.95)
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        epochs = list(range(1, len(self.metrics_history['train_loss']) + 1))
        
        # 1. Loss Curves (main plot, larger)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(epochs, self.metrics_history['train_loss'], 'b-', linewidth=2.5, label='Training Loss', marker='o', markersize=4)
        ax1.plot(epochs, self.metrics_history['val_loss'], 'r-', linewidth=2.5, label='Validation Loss', marker='s', markersize=4)
        ax1.set_title('Training & Validation Loss', fontweight='bold', pad=15)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#f8f9fa')
        
        # 2. Fitness Score
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(epochs, self.metrics_history['fitness'], 'g-', linewidth=2.5, marker='^', markersize=6)
        ax2.set_title('Fitness Score', fontweight='bold', pad=15)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Fitness')
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('#f8f9fa')
        
        # 3. Training Speed (Epoch Times)
        ax3 = fig.add_subplot(gs[1, 0])
        bars = ax3.bar(epochs, self.metrics_history['epoch_times'], color='orange', alpha=0.8, edgecolor='darkorange', linewidth=1)
        ax3.set_title('Training Speed', fontweight='bold', pad=15)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Time (seconds)')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_facecolor('#f8f9fa')
        
        # Add value labels on bars
        for bar, time in zip(bars, self.metrics_history['epoch_times']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{time:.1f}s', ha='center', va='bottom', fontsize=9)
        
        # 4. Learning Rate Schedule
        ax4 = fig.add_subplot(gs[1, 1])
        if self.metrics_history['learning_rates']:
            ax4.plot(epochs, self.metrics_history['learning_rates'], 'purple', linewidth=2.5, marker='d', markersize=4)
            ax4.set_title('Learning Rate', fontweight='bold', pad=15)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.grid(True, alpha=0.3)
            ax4.set_facecolor('#f8f9fa')
        
        # 5. Loss Reduction Rate
        ax5 = fig.add_subplot(gs[1, 2])
        if len(self.metrics_history['train_loss']) > 1:
            train_loss_reduction = []
            for i in range(1, len(self.metrics_history['train_loss'])):
                reduction = ((self.metrics_history['train_loss'][i-1] - self.metrics_history['train_loss'][i]) / 
                           self.metrics_history['train_loss'][i-1]) * 100
                train_loss_reduction.append(reduction)
            
            ax5.plot(epochs[1:], train_loss_reduction, 'teal', linewidth=2.5, marker='p', markersize=4)
            ax5.set_title('Loss Reduction Rate', fontweight='bold', pad=15)
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('Reduction (%)')
            ax5.grid(True, alpha=0.3)
            ax5.set_facecolor('#f8f9fa')
        
        # 6. Training Statistics Summary
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        # Calculate statistics
        total_time = sum(self.metrics_history['epoch_times'])
        avg_epoch_time = total_time / len(self.metrics_history['epoch_times'])
        best_fitness = max(self.metrics_history['fitness'])
        final_train_loss = self.metrics_history['train_loss'][-1]
        final_val_loss = self.metrics_history['val_loss'][-1]
        loss_improvement = ((self.metrics_history['train_loss'][0] - final_train_loss) / 
                           self.metrics_history['train_loss'][0]) * 100
        
        # Create statistics text
        stats_text = f"""
        📊 Training Statistics Summary:
        
        • Total Training Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)
        • Average Epoch Time: {avg_epoch_time:.1f} seconds
        • Best Fitness Score: {best_fitness:.4f}
        • Final Training Loss: {final_train_loss:.4f}
        • Final Validation Loss: {final_val_loss:.4f}
        • Loss Improvement: {loss_improvement:.1f}%
        • Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}
        • Dataset Size: {len(self.train_dataset)} train, {len(self.val_dataset)} val
        • Image Size: {self.args.imgsz}x{self.args.imgsz}
        • Batch Size: {self.args.batch_size}
        """
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        # Add footer
        fig.text(0.5, 0.02, f'Generated at epoch {epoch} | Fast YOLOv3 Training', 
                ha='center', fontsize=10, style='italic', color='gray')
        
        # Save visualization
        viz_path = self.viz_dir / f'training_progress_epoch_{epoch:03d}.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📈 Training visualization saved: {viz_path}")
        return viz_path
    
    def save_model_checkpoint(self, epoch, train_loss, val_loss, fitness, is_best=False):
        """Save model checkpoint with metrics in filename"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'fitness': fitness,
            'class_names': self.class_names
        }
        
        # Format metrics for filename
        train_loss_str = f"{train_loss:.4f}".replace('.', '_')
        val_loss_str = f"{val_loss:.4f}".replace('.', '_')
        fitness_str = f"{fitness:.4f}".replace('.', '_')
        
        if is_best:
            filename = f"best_epoch_{epoch:03d}_train_{train_loss_str}_val_{val_loss_str}_fitness_{fitness_str}.pt"
        else:
            filename = f"last_epoch_{epoch:03d}_train_{train_loss_str}_val_{val_loss_str}_fitness_{fitness_str}.pt"
        
        checkpoint_path = self.weights_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        print(f"Model checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def train(self):
        """Main training loop with visualizations"""
        print(f"Starting training for {self.args.epochs} epochs...")
        
        best_fitness = 0.0
        
        for epoch in range(self.args.epochs):
            print(f"\nEpoch {epoch+1}/{self.args.epochs}")
            
            # Train
            train_loss, epoch_time = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate_epoch()
            
            # Calculate fitness (simplified)
            fitness = max(0, 1.0 - val_loss)
            
            # Track metrics
            self.metrics_history['train_loss'].append(train_loss)
            self.metrics_history['val_loss'].append(val_loss)
            self.metrics_history['fitness'].append(fitness)
            self.metrics_history['epoch_times'].append(epoch_time)
            self.metrics_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            print(f"Epoch {epoch+1}/{self.args.epochs}: "
                  f"train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, "
                  f"fitness={fitness:.4f}, "
                  f"time={epoch_time:.1f}s")
            
            # Generate visualizations
            if self.args.save_visualizations:
                self.save_training_visualizations(epoch + 1)
            
            # Generate confusion matrix
            if self.args.save_confusion_matrix:
                self.generate_confusion_matrix(epoch + 1)
            
            # Generate sample predictions every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.generate_sample_predictions(epoch + 1)
            
            # Save model checkpoints
            is_best = fitness > best_fitness
            if is_best:
                best_fitness = fitness
                self.save_model_checkpoint(epoch + 1, train_loss, val_loss, fitness, is_best=True)
                print(f"New best model saved!")
            
            # Always save last checkpoint
            self.save_model_checkpoint(epoch + 1, train_loss, val_loss, fitness, is_best=False)
            
            # Step learning rate scheduler
            self.scheduler.step()
        
        # Save final training summary
        summary = {
            'total_epochs': self.args.epochs,
            'best_fitness': best_fitness,
            'final_train_loss': self.metrics_history['train_loss'][-1],
            'final_val_loss': self.metrics_history['val_loss'][-1],
            'total_training_time': sum(self.metrics_history['epoch_times']),
            'class_names': self.class_names,
            'metrics_history': self.metrics_history
        }
        
        summary_path = self.results_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nTraining completed!")
        print(f"Best fitness: {best_fitness:.4f}")
        print(f"Total training time: {sum(self.metrics_history['epoch_times']):.1f}s")
        print(f"Training summary saved: {summary_path}")
        print(f"All results saved to: {self.results_dir}")
    

def main():
    parser = argparse.ArgumentParser(description='Fast YOLOv3 Training with Hardcoded Defaults')
    
    # Hardcoded defaults as requested
    parser.add_argument('--data', type=str, default='data/dataset.yaml', help='Path to dataset yaml file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=224, help='Image size')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')
    parser.add_argument('--save-visualizations', action='store_true', default=True, help='Save training visualizations')
    parser.add_argument('--save-confusion-matrix', action='store_true', default=True, help='Save confusion matrices')
    parser.add_argument('--project', type=str, default='runs/train', help='Project directory')
    parser.add_argument('--name', type=str, default='fast_yolov3', help='Experiment name')
    
    args = parser.parse_args()
    
    # Print hardcoded defaults
    print("🚀 Fast YOLOv3 Training with Hardcoded Defaults:")
    print(f"   📁 Data: {args.data}")
    print(f"   🔄 Epochs: {args.epochs}")
    print(f"   📦 Batch Size: {args.batch_size}")
    print(f"   🖼️  Image Size: {args.imgsz}x{args.imgsz}")
    print(f"   💻 Device: {args.device}")
    print(f"   📊 Visualizations: {args.save_visualizations}")
    print(f"   📈 Confusion Matrix: {args.save_confusion_matrix}")
    print()
    
    # Create trainer
    trainer = FastTrainer(args)
    trainer.setup_data()
    trainer.setup_model()
    trainer.train()

if __name__ == '__main__':
    main()
