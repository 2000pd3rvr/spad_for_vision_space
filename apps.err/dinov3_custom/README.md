# DINOv3 Custom Training

Production-ready DINOv3 training script optimized for CPU with detailed metrics and visualizations.

## 🚀 Features

- **DINOv3 Pretrained Weights**: Uses best performing DINOv3 pretrained weights
- **CPU Optimization**: Optimized for CPU training with efficient data loading
- **Detailed Metrics**: Comprehensive performance tracking and visualization
- **Weight Management**: Smart best/last weight saving with accuracy tracking
- **Confusion Matrices**: Automatic confusion matrix generation for each saved weight
- **Production Ready**: Robust error handling and comprehensive logging

## 📁 Dataset Structure

```
data/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class2/
│       ├── image1.jpg
│       └── image2.jpg
└── val/
    ├── class1/
    │   ├── image1.jpg
    │   └── image2.jpg
    └── class2/
        ├── image1.jpg
        └── image2.jpg
```

## 📈 Output Structure

```
production_results/
└── training_YYYYMMDD_HHMMSS/
    ├── weights/                           # Model weights
    │   ├── bestweight_epoch_X_train_Y_Z_val_A_B_acc_C_D%.pth
    │   ├── lastweight_epoch_X_train_Y_Z_val_A_B_acc_C_D%.pth
    │   └── checkpoint_epoch_X_train_Y_Z_val_A_B_acc_C_D%.pth
    ├── confusion_matrices/                # Confusion matrices
    │   ├── bestweight_epoch_X_train_Y_Z_val_A_B_acc_C_D%.png
    │   ├── lastweight_epoch_X_train_Y_Z_val_A_B_acc_C_D%.png
    │   ├── bestweight_epoch_X_train_Y_Z_val_A_B_acc_C_D%_report.json
    │   └── lastweight_epoch_X_train_Y_Z_val_A_B_acc_C_D%_report.json
    ├── training_metrics_epoch_X.png       # Performance plots
    └── metrics_epoch_X.json               # Detailed metrics
```

### Weight Management
- **Best Weight**: Saved when accuracy improves
- **Last Weight**: Saved after each epoch
- **Filename Format**: `{type}_epoch_{epoch}_train_{train_loss}_val_{val_loss}_acc_{accuracy}%.pth`
- **Confusion Matrix**: Generated for each saved weight
- **Classification Report**: JSON report with detailed metrics

## 🔍 Evaluation Metrics

### Performance Tracking
- **Training Loss**: Cross-entropy loss during training
- **Validation Loss**: Cross-entropy loss during validation
- **Training Accuracy**: Classification accuracy on training set
- **Validation Accuracy**: Classification accuracy on validation set
- **Learning Rate**: Current learning rate (CosineAnnealingLR)
- **Epoch Time**: Time taken per epoch

### Visualizations
- **Loss Curves**: Training and validation loss over time
- **Accuracy Curves**: Training and validation accuracy over time
- **Learning Rate Schedule**: Learning rate changes over epochs
- **Epoch Time**: Training time per epoch
- **Confusion Matrices**: Detailed classification performance

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Basic Training
```bash
python train_dinov3.py --epochs 100 --batch_size 16
```

### Advanced Training
```bash
python train_dinov3.py \
    --epochs 200 \
    --batch_size 32 \
    --lr 0.001 \
    --weight_decay 1e-4 \
    --num_workers 4 \
    --save_interval 10
```

### Custom Dataset
```bash
python train_dinov3.py \
    --data_dir /path/to/your/dataset \
    --epochs 100 \
    --batch_size 16
```

## ⚙️ Arguments

- `--data_dir`: Dataset directory (default: current directory)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size for training (default: 16)
- `--lr`: Learning rate (default: 0.001)
- `--weight_decay`: Weight decay for regularization (default: 1e-4)
- `--num_workers`: Number of data loading workers (default: 4)
- `--save_interval`: Save checkpoint every N epochs (default: 10)
- `--pretrained_path`: Path to DINOv3 pretrained weights (default: dinov3_vitb16_pretrain.pth)

## 🎯 Model Architecture

### DINOv3 Model
- **Backbone**: Vision Transformer (ViT-B/16)
- **Pretrained Weights**: DINOv3 pretrained on ImageNet
- **Head**: Linear classifier for custom number of classes
- **Loss Function**: Cross-entropy loss with self-supervised components

### Data Augmentation
- **Resize**: 224x224 pixels
- **Random Horizontal Flip**: 50% probability
- **Random Rotation**: ±15 degrees
- **Color Jitter**: Brightness, contrast, saturation, hue
- **Normalization**: ImageNet mean/std

## 📊 Performance Features

### Always Verbose Mode
- **Progress Bars**: tqdm progress bars for all training phases
- **Real-time Metrics**: Loss, accuracy, and learning rate displayed
- **Detailed Logging**: Comprehensive epoch summaries
- **Performance Tracking**: Automatic metrics collection

### Weight Management Logic
1. **First Epoch**: Set as both best and last weight
2. **Improvement**: If current accuracy > last accuracy:
   - Promote previous lastweight to bestweight (if better than current best)
   - Update last accuracy
3. **No Improvement**: Keep existing bestweight
4. **Confusion Matrix**: Generated for every saved weight

### Production Features
- **Error Handling**: Robust error handling and recovery
- **Memory Management**: Efficient CPU memory usage
- **Reproducibility**: Fixed random seeds
- **Logging**: Comprehensive training logs
- **Visualization**: Automatic plot generation

## 🔧 Technical Details

### DINOv3 Implementation
- **Teacher-Student**: Knowledge distillation framework
- **Self-Supervised**: Unsupervised pretraining
- **Multi-Crop**: Multiple image crops for consistency
- **Momentum Updates**: Exponential moving average updates

### CPU Optimization
- **Efficient Data Loading**: Optimized DataLoader configuration
- **Memory Management**: Careful memory usage patterns
- **Batch Processing**: Optimized batch sizes for CPU
- **Worker Processes**: Configurable data loading workers

## 📈 Expected Performance

### Training Speed
- **CPU**: ~2-5 minutes per epoch (depending on dataset size)
- **Memory**: ~2-4 GB RAM usage
- **Storage**: ~100-500 MB per training run

### Accuracy
- **Baseline**: 60-80% accuracy (depending on dataset)
- **With DINOv3**: 80-95% accuracy (with proper training)
- **Convergence**: Usually converges within 50-100 epochs

## 🚨 Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch size or number of workers
2. **Slow Training**: Increase number of workers or reduce image size
3. **Poor Accuracy**: Check dataset quality and class balance
4. **Weight Loading**: Ensure pretrained weights are in correct format

### Performance Tips
1. **Batch Size**: Use largest batch size that fits in memory
2. **Workers**: Set num_workers to number of CPU cores
3. **Learning Rate**: Start with 0.001 and adjust based on convergence
4. **Epochs**: Monitor validation accuracy to determine stopping point

## 📝 License

This project is part of the multiwebapp vision system.
