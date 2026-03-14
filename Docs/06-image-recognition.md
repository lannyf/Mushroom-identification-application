# Phase 3: Image Recognition Module Development

## Objective

Develop a transfer learning-based image recognition module that uses pretrained CNN models to classify mushroom images. This module will serve as one of three identification methods in the hybrid system.

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│      Image Recognition Module (Phase 3)     │
│                                             │
│  ┌───────────────────────────────────────┐  │
│  │  Image Input (JPEG/PNG)               │  │
│  │  (480x640, 3-channel RGB)             │  │
│  └──────────────┬────────────────────────┘  │
│                 │                           │
│  ┌──────────────▼────────────────────────┐  │
│  │  Image Processor                      │  │
│  │  ├─ Load image                        │  │
│  │  ├─ Resize to 224x224                 │  │
│  │  ├─ Normalize (ImageNet)              │  │
│  │  └─ Augment (if training)             │  │
│  └──────────────┬────────────────────────┘  │
│                 │                           │
│  ┌──────────────▼────────────────────────┐  │
│  │  Base Model (Transfer Learning)       │  │
│  │  ├─ MobileNetV2 (default)             │  │
│  │  ├─ EfficientNetB0 (alternative)      │  │
│  │  └─ ResNet50 (alternative)            │  │
│  │  (Pretrained on ImageNet)             │  │
│  └──────────────┬────────────────────────┘  │
│                 │                           │
│  ┌──────────────▼────────────────────────┐  │
│  │  Custom Classification Head           │  │
│  │  ├─ Global Average Pooling            │  │
│  │  ├─ Dense(256, relu) + Dropout(0.3)   │  │
│  │  ├─ Dense(128, relu) + Dropout(0.2)   │  │
│  │  └─ Dense(20, softmax)                │  │
│  │  (20 = number of mushroom species)    │  │
│  └──────────────┬────────────────────────┘  │
│                 │                           │
│  ┌──────────────▼────────────────────────┐  │
│  │  Output                               │  │
│  │  [probability_class_0, ...,           │  │
│  │   probability_class_19]               │  │
│  │  (Sum = 1.0)                          │  │
│  └───────────────────────────────────────┘  │
│                                             │
└─────────────────────────────────────────────┘
```

## Implementation Components

### 1. Configuration (`config/image_model_config.py`)
Centralized configuration for the entire image recognition module:
- **Model architecture:** Base model selection, input size, number of classes
- **Image preprocessing:** Normalization, augmentation parameters
- **Training hyperparameters:** Learning rate, batch size, epochs, optimizer
- **Transfer learning settings:** Layer freezing, fine-tuning strategy
- **Device and performance:** GPU/CPU selection, mixed precision

**Key Configurations:**
```python
# Model
BASE_MODEL = "mobilenet_v2"          # Lightweight, good accuracy
INPUT_SIZE = (224, 224)              # Standard CNN input
NUM_CLASSES = 20                      # Mushroom species

# Training
BATCH_SIZE = 32
EPOCHS = 50
INITIAL_LEARNING_RATE = 0.001
OPTIMIZER = "adam"

# Preprocessing
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
```

### 2. Image Processor (`models/image_processor.py`)
Handles image loading, resizing, normalization, and augmentation:

**Key Classes:**
- `ImageProcessor` - Load, resize, normalize, augment individual images
- `DataGenerator` - Generate batches for training with on-the-fly augmentation

**Features:**
- Multiple interpolation methods (bilinear, nearest, bicubic)
- Multiple normalization strategies (ImageNet, standard, min-max)
- Data augmentation (rotation, brightness, flip)
- Batch processing with optional augmentation
- Error handling for missing/corrupted images

**Usage:**
```python
processor = ImageProcessor()
image = processor.load_image("path/to/image.jpg")
preprocessed = processor.preprocess(image)  # Resize + normalize
augmented = processor.augment_image(image)  # Create variations
```

### 3. Image Recognition Model (`models/image_recognition.py`)
Core CNN model with transfer learning:

**Key Classes:**
- `ImageRecognitionModel` - Build, compile, train, and save model
- `Predictor` - Make predictions with top-k results

**Supports:**
- TensorFlow/Keras backend
- PyTorch backend
- Multiple pretrained base models (MobileNetV2, EfficientNetB0, ResNet50)
- Fine-tuning with frozen/unfrozen layers
- Custom classification head

**Model Building:**
```python
model = ImageRecognitionModel()
model.build_model(framework='tensorflow')
model.compile(optimizer='adam', learning_rate=0.001)
model.save()
```

### 4. Training Script (`scripts/train_image_model.py`)
End-to-end training pipeline:

**Features:**
- Dummy data generation for testing (without real images)
- Real dataset loading (placeholder for future implementation)
- Data generator creation with augmentation
- Model training with validation
- Model checkpointing and history logging
- Comprehensive logging and progress tracking

**Usage:**
```bash
# Train with dummy data (for testing)
python scripts/train_image_model.py --framework tensorflow --epochs 10

# Train with real data (when available)
python scripts/train_image_model.py --use-real-data --epochs 50 --batch-size 32
```

## Transfer Learning Strategy

### Phase 1: Base Model Training (Frozen)
1. Load pretrained base model (ImageNet weights)
2. Freeze all base model layers
3. Train only custom classification head (100-200 iterations)
4. Goal: Adapt pretrained features to mushroom task quickly

### Phase 2: Fine-tuning (Selective Unfreezing)
1. Unfreeze top layers of base model
2. Train with lower learning rate (0.0001 or lower)
3. Continue for 100-300 iterations
4. Goal: Adapt deeper features to mushroom-specific patterns

### Phase 3: Full Model Fine-tuning (Optional)
1. Unfreeze entire model
2. Very low learning rate (0.00001)
3. Train for remaining epochs
4. Goal: Final optimization

## Why Transfer Learning?

**Advantages:**
- ✅ Reduces training time (100x faster than training from scratch)
- ✅ Requires fewer images (200-400 vs. 100,000+)
- ✅ Better generalization on small datasets
- ✅ Leverages rich ImageNet feature representations
- ✅ Lower computational requirements

**Trade-offs:**
- Domain shift: ImageNet features may not perfectly match mushroom images
- Mitigation: Data augmentation, careful hyperparameter tuning

## Base Model Selection

### MobileNetV2 (Recommended for Production)
- **Size:** ~14 MB
- **Latency:** ~10ms per image (mobile GPU)
- **Accuracy:** 71.8% ImageNet top-1
- **Use case:** Mobile/edge deployment, real-time inference
- **Pros:** Small, fast, good accuracy
- **Cons:** Slightly lower accuracy than larger models

### EfficientNetB0
- **Size:** ~29 MB
- **Latency:** ~20ms per image
- **Accuracy:** 77.1% ImageNet top-1
- **Use case:** Balanced accuracy/speed
- **Pros:** Better accuracy than MobileNet
- **Cons:** Larger model size

### ResNet50
- **Size:** ~98 MB
- **Latency:** ~50ms per image
- **Accuracy:** 76.1% ImageNet top-1
- **Use case:** Server/batch processing
- **Pros:** Good accuracy, well-studied
- **Cons:** Large size, slower inference

**Recommendation:** Start with **MobileNetV2** for ease of deployment and reasonable accuracy.

## Training Workflow

### 1. Data Preparation
```
Raw Images
  ├─ Load from data/raw/images/<species_id>/
  ├─ Verify image quality
  ├─ Apply preprocessing (resize, normalize)
  └─ Create train/validation splits
```

### 2. Model Building
```
Base Model (ImageNet weights)
  ├─ Freeze base layers
  ├─ Add custom head layers
  ├─ Set up optimizer (Adam)
  ├─ Configure loss (categorical crossentropy)
  └─ Compile model
```

### 3. Training Loop
```
For each epoch:
  ├─ Train on training set
  │   ├─ Load batch of images
  │   ├─ Apply data augmentation
  │   ├─ Forward pass through model
  │   ├─ Compute loss
  │   ├─ Backward pass
  │   └─ Update weights
  │
  ├─ Validate on validation set
  │   ├─ Load batch of images (no augmentation)
  │   ├─ Forward pass
  │   ├─ Compute metrics
  │   └─ Log results
  │
  └─ Check early stopping criterion
```

### 4. Evaluation
```
Test Set Evaluation
  ├─ Forward pass on test images
  ├─ Compute metrics
  │   ├─ Accuracy (overall)
  │   ├─ Precision (per-class)
  │   ├─ Recall (per-class)
  │   ├─ F1-score (per-class)
  │   └─ Confusion matrix
  └─ Analyze results
      ├─ Identify challenging species
      ├─ Find common confusions
      └─ Plan improvements
```

## Configuration Files

### `config/image_model_config.py`
- Model architecture parameters
- Training hyperparameters
- Image preprocessing settings
- Data augmentation configuration
- Device and performance settings

**Key Configuration Groups:**
- **PATHS** - Directory locations
- **MODEL_ARCHITECTURE** - Base model, input size, num classes
- **IMAGE_PREPROCESSING** - Normalization, resizing
- **TRAINING_HYPERPARAMETERS** - Learning rate, batch size, epochs
- **TRANSFER_LEARNING** - Freezing strategy, custom head
- **EVALUATION** - Metrics, thresholds
- **MODEL_PERSISTENCE** - File formats, naming

## Input/Output Specifications

### Input
- **Format:** JPEG, PNG, or other image format
- **Size:** Any size (will be resized to 224x224)
- **Channels:** RGB (3-channel, converted if necessary)
- **Range:** 0-255 (will be normalized)

### Output
```python
{
    'species_id': 'CH001',
    'class_index': 0,
    'confidence': 0.92,
}
```

### Top-k Predictions
```python
[
    {'species_id': 'CH001', 'class_index': 0, 'confidence': 0.92},
    {'species_id': 'FALSE_CH', 'class_index': 16, 'confidence': 0.07},
    {'species_id': 'BU001', 'class_index': 1, 'confidence': 0.01},
]
```

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Top-1 Accuracy | >75% | Single best prediction |
| Top-3 Accuracy | >85% | Any of top 3 predictions |
| Top-5 Accuracy | >90% | Any of top 5 predictions |
| Inference Latency | <500ms | GPU inference on mobile |
| Model Size | <50 MB | Deployable on mobile |

## Current Implementation Status

### ✅ Complete
- Configuration system with all hyperparameters
- Image processor with loading, resizing, normalization, augmentation
- Model builder supporting TensorFlow/PyTorch
- Transfer learning architecture with custom head
- Training script with dummy data generation
- Evaluation framework structure

### ⏳ To Be Implemented (When Real Data Available)
- Real image dataset loading from `data/raw/images/`
- Integration with `data/raw/species_images.csv` metadata
- Test set evaluation with metrics computation
- Confusion matrix generation
- Model interpretability (GradCAM, attention visualization)
- Ensemble methods (if multiple models)

## Next Steps for Full Implementation

### 1. Real Image Collection (1-2 weeks)
- Gather 10-20 images per species (200-400 total)
- Organize in `data/raw/images/<species_id>/`
- Update `species_images.csv` with image metadata
- Verify image quality and species identification

### 2. Dataset Integration (1 week)
- Implement real image loading in training script
- Create PyTorch/TensorFlow DataLoader
- Implement data augmentation pipeline
- Set up train/validation/test splits from `dataset_split.csv`

### 3. Model Training (2-4 weeks)
- Run initial training with frozen base model
- Evaluate metrics on validation set
- Fine-tune with unfrozen layers
- Experiment with different base models (MobileNet, EfficientNet, ResNet)
- Hyperparameter optimization

### 4. Evaluation & Analysis (1-2 weeks)
- Comprehensive evaluation on test set
- Error analysis and confusion matrices
- Identify challenging species and confusions
- Create evaluation report

### 5. Optimization (1-2 weeks)
- Model quantization for mobile deployment
- ONNX conversion for cross-platform support
- Performance optimization (inference speed)
- Robustness testing (partial images, poor lighting)

## File Structure

```
models/
├── image_processor.py      # Image loading, preprocessing, augmentation
└── image_recognition.py    # CNN model with transfer learning

scripts/
├── train_image_model.py    # Training pipeline
├── evaluate_image_model.py # Evaluation and metrics (to be created)
└── predict_image_model.py  # Inference script (to be created)

config/
└── image_model_config.py   # All configuration parameters

artifacts/
├── image_recognition_final.pt          # Saved model weights
├── image_recognition_best.h5           # Best checkpoint
├── training_history.json               # Training metrics
└── model_metadata.json                 # Model information
```

## Dependencies

All required packages are listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

**Key dependencies:**
- **Deep Learning:** TensorFlow 2.8+, PyTorch 1.10+ (choose one or both)
- **Computer Vision:** OpenCV, Pillow, scikit-image
- **Data Science:** NumPy, Pandas, scikit-learn
- **Development:** Jupyter, pytest, python-dotenv

## Testing

### Test with Dummy Data
```bash
python scripts/train_image_model.py --epochs 2
```

### Test with Real Data (when available)
```bash
python scripts/train_image_model.py --use-real-data --epochs 10 --batch-size 16
```

## Next Phase

Phase 4 will develop the trait-based classification module, which will use structured mushroom characteristics as input instead of images.

---

**Status:** Phase 3 - In Progress  
**Created:** 2026-03-14  
**Framework Support:** TensorFlow/Keras (primary), PyTorch (secondary)  
**Base Models:** MobileNetV2 (default), EfficientNetB0, ResNet50
