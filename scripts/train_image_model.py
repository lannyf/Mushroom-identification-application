"""
Training script for image recognition model.

Trains the image recognition model using transfer learning.
Supports both TensorFlow and PyTorch backends.
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Tuple, List

import numpy as np

from config.image_model_config import (
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    ARTIFACTS_DIR,
    BATCH_SIZE,
    VALIDATION_BATCH_SIZE,
    EPOCHS,
    INITIAL_LEARNING_RATE,
    DEVICE,
)

from models.image_processor import ImageProcessor, DataGenerator
from models.image_recognition import ImageRecognitionModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_dummy_dataset(num_samples: int = 100,
                        num_classes: int = 20,
                        num_train: int = 70,
                        num_val: int = 15) -> Tuple[List, List, List, List]:
    """
    Create dummy dataset for testing (when real images not available).
    
    Args:
        num_samples: Total number of samples
        num_classes: Number of classes
        num_train: Number of training samples
        num_val: Number of validation samples
        
    Returns:
        Tuple of (train_images, train_labels, val_images, val_labels)
    """
    logger.info(f"Creating dummy dataset: {num_samples} samples, {num_classes} classes")
    
    # Create dummy image data (as if loaded from files)
    # In real scenario, these would be actual image arrays
    train_images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) 
                   for _ in range(num_train)]
    train_labels = np.random.randint(0, num_classes, num_train)
    
    val_images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) 
                 for _ in range(num_val)]
    val_labels = np.random.randint(0, num_classes, num_val)
    
    logger.info(f"Dummy dataset created: {len(train_images)} train, {len(val_images)} val")
    
    return train_images, train_labels.tolist(), val_images, val_labels.tolist()


def load_real_dataset() -> Tuple[List, List, List, List]:
    """
    Load real dataset from data directory.
    
    This is a placeholder that will be fully implemented when image data is available.
    
    Returns:
        Tuple of (train_images, train_labels, val_images, val_labels)
    """
    logger.warning("Real dataset loading not yet implemented")
    logger.warning("Using dummy data for now - replace with actual image loading")
    
    # TODO: Implement actual image loading from:
    # - data/raw/species_images.csv (metadata)
    # - data/raw/images/<species_id>/ (image files)
    # - data/raw/dataset_split.csv (train/val/test split)
    
    return create_dummy_dataset()


def train_model(framework: str = 'tensorflow',
               use_dummy_data: bool = True,
               epochs: int = EPOCHS,
               batch_size: int = BATCH_SIZE,
               learning_rate: float = INITIAL_LEARNING_RATE) -> Dict:
    """
    Train image recognition model.
    
    Args:
        framework: Deep learning framework ('tensorflow' or 'pytorch')
        use_dummy_data: Whether to use dummy data for testing
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        
    Returns:
        Dictionary with training results
    """
    logger.info("="*80)
    logger.info("Starting Image Recognition Model Training")
    logger.info("="*80)
    logger.info(f"Framework: {framework}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    
    # Load dataset
    if use_dummy_data:
        train_images, train_labels, val_images, val_labels = create_dummy_dataset(
            num_samples=100,
            num_classes=20,
            num_train=70,
            num_val=15
        )
    else:
        train_images, train_labels, val_images, val_labels = load_real_dataset()
    
    # Create image processor
    image_processor = ImageProcessor()
    
    # Create data generators
    logger.info("Creating data generators...")
    train_generator = DataGenerator(
        image_paths=[f"dummy_{i}" for i in range(len(train_images))],
        labels=train_labels,
        image_processor=image_processor,
        batch_size=batch_size,
        augment=True,
        shuffle=True
    )
    
    val_generator = DataGenerator(
        image_paths=[f"dummy_val_{i}" for i in range(len(val_images))],
        labels=val_labels,
        image_processor=image_processor,
        batch_size=VALIDATION_BATCH_SIZE,
        augment=False,
        shuffle=False
    )
    
    # Build model
    logger.info("Building model...")
    model = ImageRecognitionModel()
    model.build_model(framework=framework)
    model.compile(
        optimizer='adam',
        learning_rate=learning_rate,
        framework=framework
    )
    model.summary()
    
    # Train model
    logger.info("Starting training...")
    if framework == 'tensorflow':
        history = model.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            verbose=1
        )
        training_history = history.history
    else:
        logger.warning("PyTorch training not fully implemented yet")
        training_history = {'loss': [], 'val_loss': []}
    
    # Save model
    logger.info("Saving model...")
    model_path = model.save()
    
    # Save training history
    history_path = ARTIFACTS_DIR / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")
    
    # Summary
    results = {
        'status': 'success',
        'model_path': str(model_path),
        'history_path': str(history_path),
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'framework': framework,
    }
    
    logger.info("="*80)
    logger.info("Training Complete")
    logger.info("="*80)
    logger.info(json.dumps(results, indent=2))
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train image recognition model')
    parser.add_argument('--framework', default='tensorflow',
                       choices=['tensorflow', 'pytorch'],
                       help='Deep learning framework')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=INITIAL_LEARNING_RATE,
                       help='Initial learning rate')
    parser.add_argument('--use-real-data', action='store_true',
                       help='Use real data instead of dummy data')
    parser.add_argument('--device', default=DEVICE,
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Train model
    results = train_model(
        framework=args.framework,
        use_dummy_data=not args.use_real_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    return 0 if results['status'] == 'success' else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
