"""
Configuration for image recognition model.

This module contains all configurable parameters for the image recognition
module including model architecture, training hyperparameters, and paths.
"""

from pathlib import Path
from typing import Dict, List, Tuple

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for dir_path in [ARTIFACTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

# Transfer learning base model
BASE_MODEL = "mobilenet_v2"  # Options: mobilenet_v2, efficientnet_b0, resnet50

MODEL_WEIGHTS = "imagenet"  # Use pretrained ImageNet weights

# Input image size (should match model's expected input)
INPUT_SIZE = (224, 224)  # Height, Width

# Number of output classes (mushroom species)
NUM_CLASSES = 20

# Whether to include top classification layer in pretrained model
INCLUDE_TOP = False  # We'll add our own


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB channels
IMAGENET_STD = [0.229, 0.224, 0.225]   # RGB channels

# Image preprocessing parameters
PREPROCESSING = {
    "resize_size": INPUT_SIZE,
    "normalization": "imagenet",
    "mean": IMAGENET_MEAN,
    "std": IMAGENET_STD,
    "interpolation": "bilinear",
}

# Data augmentation parameters
AUGMENTATION = {
    "rotation_range": 20,              # Degrees
    "width_shift_range": 0.2,          # Fraction of width
    "height_shift_range": 0.2,         # Fraction of height
    "horizontal_flip": True,
    "vertical_flip": False,            # Not appropriate for mushrooms
    "zoom_range": 0.2,                 # Fraction of size
    "brightness_range": [0.8, 1.2],
    "fill_mode": "nearest",
}


# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

# Dataset split percentages
TRAIN_SPLIT = 0.70
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# Batch sizes
BATCH_SIZE = 32
VALIDATION_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32

# Learning rate
INITIAL_LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = 0.1
DECAY_STEPS = 10  # Epochs

# Number of training epochs
EPOCHS = 50

# Early stopping
EARLY_STOPPING = {
    "monitor": "val_loss",
    "patience": 10,
    "restore_best_weights": True,
}

# Model checkpoint
CHECKPOINT = {
    "monitor": "val_accuracy",
    "save_best_only": True,
    "mode": "max",
}

# Optimizer
OPTIMIZER = "adam"  # Options: adam, sgd, rmsprop
OPTIMIZER_PARAMS = {
    "adam": {"learning_rate": INITIAL_LEARNING_RATE, "decay": 0.0},
    "sgd": {"learning_rate": INITIAL_LEARNING_RATE, "momentum": 0.9},
}

# Loss function
LOSS = "categorical_crossentropy"

# Metrics
METRICS = ["accuracy", "precision", "recall"]


# ============================================================================
# TRANSFER LEARNING
# ============================================================================

# Fine-tuning strategy
FREEZE_BASE_MODEL = True  # Freeze base model layers during initial training

# Number of base layers to unfreeze (after initial training)
UNFREEZE_LAYERS = 50  # Leave this many layers unfrozen for fine-tuning

# Custom head architecture (layers to add on top of base model)
CUSTOM_HEAD = [
    {"type": "global_average_pooling_2d"},
    {"type": "dense", "units": 256, "activation": "relu"},
    {"type": "dropout", "rate": 0.3},
    {"type": "dense", "units": 128, "activation": "relu"},
    {"type": "dropout", "rate": 0.2},
    {"type": "dense", "units": NUM_CLASSES, "activation": "softmax"},
]


# ============================================================================
# CLASS WEIGHTS
# ============================================================================

# Whether to use class weights for imbalanced datasets
USE_CLASS_WEIGHTS = True

# If True, weights will be computed from training data
# If False, use this dictionary for manual weights
CLASS_WEIGHTS = None  # Will be computed from data


# ============================================================================
# EVALUATION
# ============================================================================

# Confidence threshold for predictions
CONFIDENCE_THRESHOLD = 0.5

# Top-k predictions to return
TOP_K = 5

# Evaluation metrics to compute
EVAL_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "confusion_matrix",
    "roc_auc",
]


# ============================================================================
# MODEL PERSISTENCE
# ============================================================================

# Model file names
MODEL_CHECKPOINT_NAME = "image_recognition_best.h5"
MODEL_FINAL_NAME = "image_recognition_final.pt"
MODEL_ONNX_NAME = "image_recognition.onnx"

# Save model weights in multiple formats
SAVE_FORMATS = ["h5", "pytorch", "onnx"]

# Training history
HISTORY_FILE = "training_history.json"

# Metadata (model info, dataset info, etc.)
METADATA_FILE = "model_metadata.json"


# ============================================================================
# DEVICE
# ============================================================================

# Device to use for training
# Options: "cuda" (GPU), "cpu", "tpu" (if available)
DEVICE = "cuda"  # Will fall back to CPU if CUDA not available

# Number of workers for data loading
NUM_WORKERS = 4

# Mixed precision training (for faster GPU training)
MIXED_PRECISION = False


# ============================================================================
# LOGGING & MONITORING
# ============================================================================

# Logging level
LOG_LEVEL = "INFO"

# TensorBoard logging
TENSORBOARD = {
    "enabled": True,
    "log_dir": LOGS_DIR / "tensorboard",
    "histogram_freq": 1,
}

# Frequency to log metrics
LOG_FREQUENCY = 50  # Batches

# Validation frequency
VALIDATION_FREQUENCY = 1  # Epochs


# ============================================================================
# INFERENCE
# ============================================================================

# Maximum image size for inference
MAX_INFERENCE_SIZE = (1024, 1024)

# Whether to return explanations with predictions
RETURN_EXPLANATIONS = True

# Explanation methods (if supported)
EXPLANATION_METHODS = ["gradcam", "attention"]


# ============================================================================
# TESTING DATA (FOR PROTOTYPE)
# ============================================================================

# When using dummy data for testing
DUMMY_DATA = {
    "num_samples": 100,
    "num_classes": NUM_CLASSES,
    "image_size": INPUT_SIZE,
}


def get_config_dict() -> Dict:
    """Return all configuration as a dictionary."""
    return {
        "base_model": BASE_MODEL,
        "input_size": INPUT_SIZE,
        "num_classes": NUM_CLASSES,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": INITIAL_LEARNING_RATE,
        "optimizer": OPTIMIZER,
        "loss": LOSS,
        "metrics": METRICS,
        "augmentation": AUGMENTATION,
        "early_stopping": EARLY_STOPPING,
        "device": DEVICE,
    }


if __name__ == "__main__":
    # Print configuration
    import json
    
    config = get_config_dict()
    print(json.dumps(config, indent=2))
