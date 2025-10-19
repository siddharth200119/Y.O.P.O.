# YOLOv8 UI Elements Detection - Optimized Training Pipeline

A comprehensive, notebook-based training pipeline for detecting UI elements using YOLOv8 models with automated comparison and evaluation.

## 📋 Features

- **Automated Dataset Download**: Downloads UI elements dataset from Hugging Face
- **Multi-Model Training**: Train and compare YOLOv8n, YOLOv8s, and YOLOv8m models
- **Performance Metrics**: Comprehensive evaluation with mAP, Precision, Recall
- **Visual Comparison**: Automatic generation of comparison plots and charts
- **Inference Pipeline**: Run predictions on test images with the best model
- **Results Export**: CSV reports, JSON summaries, and visualization plots

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the notebook
# Then run the first cell to install dependencies:
pip install ultralytics datasets tqdm opencv-python-headless pyyaml pandas matplotlib seaborn
```

### 2. Configuration

Edit the configuration cell to customize training:

```python
TRAINING_CONFIG = {
    'epochs': 50,           # Number of training epochs
    'imgsz': 640,          # Input image size
    'batch': 16,           # Batch size
    'device': 'cpu',       # 'cpu' or '0' for GPU
    'patience': 20,        # Early stopping patience
    'workers': 8           # Number of data loading workers
}

# Models to train
MODELS_TO_TRAIN = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
```

### 3. Run the Notebook

Execute cells sequentially:

1. **Setup** - Install dependencies and import libraries
2. **Configuration** - Set training parameters
3. **Dataset Preparation** - Download and structure dataset
4. **Training** - Train multiple models
5. **Validation** - Evaluate model performance
6. **Comparison** - Compare models side-by-side
7. **Inference** - Run predictions on test images
8. **Export** - Generate reports and summaries

## 📁 Directory Structure

```
project/
│
├── optimized_yolo_training.ipynb    # Main training notebook
├── README.md                         # This file
│
├── dataset/                          # Dataset folder (auto-created)
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── data.yaml
│
├── runs/                             # Training runs (auto-created)
│   └── detect/
│       ├── ui-yolov8n/
│       ├── ui-yolov8s/
│       └── ui-yolov8m/
│
└── results/                          # Output results (auto-created)
    ├── model_comparison.csv
    ├── model_comparison_plots.png
    ├── training_summary.json
    └── inference_results/
```

## 🎯 Detected UI Elements

The model detects 15 UI element classes:

1. **button** - Clickable buttons
2. **link** - Hyperlinks
3. **input_field** - Text input boxes
4. **checkbox** - Checkboxes
5. **radio_button** - Radio buttons
6. **dropdown** - Dropdown menus
7. **slider** - Sliders
8. **toggle** - Toggle switches
9. **label** - Text labels
10. **text_block** - Text blocks
11. **icon** - Icons
12. **menu_item** - Menu items
13. **text_area** - Text areas
14. **select_menu** - Select menus
15. **clickable_region** - Generic clickable regions

## 📊 Model Comparison

The notebook automatically compares trained models on:

- **mAP50** - Mean Average Precision at IoU=0.5
- **mAP50-95** - Mean Average Precision across IoU thresholds
- **Precision** - True Positive / (True Positive + False Positive)
- **Recall** - True Positive / (True Positive + False Negative)
- **Model Size** - File size in MB
- **Overall Score** - Weighted performance metric

## 🔧 Hardware Requirements

### CPU Training
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space
- **Time**: ~1-2 hours per model for 50 epochs

### GPU Training
- **GPU**: CUDA-compatible GPU with 4GB+ VRAM
- **RAM**: 8GB minimum
- **Storage**: 5GB free space
- **Time**: ~10-30 minutes per model for 50 epochs

To use GPU, change configuration:
```python
TRAINING_CONFIG = {
    'device': '0',  # or 'cuda:0'
    ...
}
```

## 📈 Expected Results

### YOLOv8n (Nano)
- **Speed**: Fastest inference
- **Size**: ~6MB
- **mAP50-95**: 0.08-0.12
- **Use Case**: Real-time applications, edge devices

### YOLOv8s (Small)
- **Speed**: Fast inference
- **Size**: ~22MB
- **mAP50-95**: 0.10-0.15
- **Use Case**: Balanced performance

### YOLOv8m (Medium)
- **Speed**: Moderate inference
- **Size**: ~52MB
- **mAP50-95**: 0.12-0.18
- **Use Case**: High accuracy requirements

## 🛠️ Troubleshooting

### Out of Memory Error
```python
# Reduce batch size
TRAINING_CONFIG['batch'] = 8  # or 4
```

### Slow Training
```python
# Reduce image size
TRAINING_CONFIG['imgsz'] = 416  # instead of 640

# Reduce epochs
TRAINING_CONFIG['epochs'] = 30
```

### CUDA Not Available
```python
# Use CPU training
TRAINING_CONFIG['device'] = 'cpu'
```

### Dataset Download Issues
```python
# Manual download: Visit Hugging Face dataset page
# Then place in 'dataset' folder manually
```

## 📝 Output Files

### model_comparison.csv
CSV file with detailed metrics for each model:
```csv
Model,mAP50,mAP50-95,Precision,Recall,Size (MB)
yolov8m.pt,0.1230,0.0866,0.7860,0.0923,52.4
yolov8s.pt,0.1180,0.0835,0.7040,0.0872,22.1
yolov8n.pt,0.1150,0.0820,0.6600,0.0840,6.2
```

### model_comparison_plots.png
4-panel visualization showing:
- mAP comparison bar chart
- Precision vs Recall scatter plot
- Model size comparison
- Overall performance scores

### training_summary.json
Complete training configuration and results:
```json
{
  "training_config": {...},
  "models_trained": [...],
  "validation_results": {...},
  "best_model": "yolov8m.pt",
  "dataset_stats": {...}
}
```

### inference_results/
Folder containing predicted images with bounding boxes and labels.

## 🔄 Workflow Steps

1. **Setup Environment** → Install packages
2. **Configure Training** → Set hyperparameters
3. **Download Dataset** → From Hugging Face
4. **Prepare Data** → Normalize YOLO format
5. **Train Models** → Multiple YOLOv8 variants
6. **Validate** → Evaluate on validation set
7. **Compare** → Generate comparison metrics
8. **Inference** → Predict on test images
9. **Export** → Save results and reports

## 📚 References

- **Ultralytics YOLOv8**: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **Dataset**: [Hugging Face - UI Elements Detection](https://huggingface.co/datasets/YashJain/UI-Elements-Detection-Dataset)
- **YOLO Documentation**: [https://docs.ultralytics.com](https://docs.ultralytics.com)

## 🤝 Contributing

Feel free to:
- Report issues
- Suggest improvements
- Share your training results
- Contribute optimizations

## 📄 License

This project follows the Ultralytics YOLO license. Please refer to the official Ultralytics repository for license details.

## ⚠️ Important Notes

- **First Run**: Dataset download may take 5-10 minutes depending on connection
- **Training Time**: CPU training is significantly slower than GPU (10-20x difference)
- **Model Selection**: Start with YOLOv8n for quick experiments, use YOLOv8m for best accuracy
- **Memory**: Monitor RAM/VRAM usage and adjust batch size accordingly
- **Checkpoints**: Models are saved every 10 epochs and at the best performance

## 🎓 Tips for Best Results

1. **Data Quality**: Ensure proper labeling in the dataset
2. **Hyperparameter Tuning**: Experiment with learning rate and batch size
3. **Augmentation**: YOLOv8 includes default augmentations
4. **Early Stopping**: Use patience parameter to prevent overfitting
5. **Model Selection**: Choose based on speed vs accuracy requirements

## 📞 Support

For issues or questions:
- Check the troubleshooting section
- Review Ultralytics documentation
- Examine training logs in `runs/detect/`
- Verify dataset structure matches YOLO format

---

**Happy Training! 🚀**
