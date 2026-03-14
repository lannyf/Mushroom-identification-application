"""
Image recognition model for mushroom identification using transfer learning.

Implements a CNN model with transfer learning for classifying mushroom images.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np

from config.image_model_config import (
    BASE_MODEL,
    INPUT_SIZE,
    NUM_CLASSES,
    CUSTOM_HEAD,
    FREEZE_BASE_MODEL,
    ARTIFACTS_DIR,
    MODEL_CHECKPOINT_NAME,
    MODEL_FINAL_NAME,
    METADATA_FILE,
)

logger = logging.getLogger(__name__)


class ImageRecognitionModel:
    """Image recognition model using transfer learning."""
    
    def __init__(self, model_name: str = BASE_MODEL,
                 num_classes: int = NUM_CLASSES,
                 input_size: Tuple[int, int] = INPUT_SIZE,
                 freeze_base: bool = FREEZE_BASE_MODEL):
        """
        Initialize image recognition model.
        
        Args:
            model_name: Base model name (mobilenet_v2, efficientnet_b0, resnet50)
            num_classes: Number of output classes
            input_size: Input image size (height, width)
            freeze_base: Whether to freeze base model layers
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.input_size = input_size
        self.freeze_base = freeze_base
        self.model = None
        self.history = None
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'model_name': model_name,
            'num_classes': num_classes,
            'input_size': input_size,
            'freeze_base': freeze_base,
        }
    
    def build_model(self, framework: str = 'tensorflow'):
        """
        Build the model architecture.
        
        Args:
            framework: Deep learning framework ('tensorflow' or 'pytorch')
        """
        if framework == 'tensorflow':
            self._build_tensorflow_model()
        elif framework == 'pytorch':
            self._build_pytorch_model()
        else:
            raise ValueError(f"Unknown framework: {framework}")
        
        logger.info(f"Model built successfully with {framework}")
    
    def _build_tensorflow_model(self):
        """Build model using TensorFlow/Keras."""
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, Model
            from tensorflow.keras.applications import (
                MobileNetV2, EfficientNetB0, ResNet50
            )
        except ImportError:
            raise ImportError("TensorFlow is required for this model")
        
        # Select base model
        base_models = {
            'mobilenet_v2': (MobileNetV2, (224, 224)),
            'efficientnet_b0': (EfficientNetB0, (224, 224)),
            'resnet50': (ResNet50, (224, 224)),
        }
        
        if self.model_name not in base_models:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        base_model_class, input_shape = base_models[self.model_name]
        
        # Load pretrained base model
        base_model = base_model_class(
            input_shape=(*self.input_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model if requested
        if self.freeze_base:
            base_model.trainable = False
        
        # Build custom head
        inputs = layers.Input(shape=(*self.input_size, 3))
        x = base_model(inputs, training=False)  # training=False for batch norm layers
        
        # Add custom layers
        for layer_config in CUSTOM_HEAD:
            layer_type = layer_config.pop('type')
            
            if layer_type == 'global_average_pooling_2d':
                x = layers.GlobalAveragePooling2D()(x)
            elif layer_type == 'dense':
                x = layers.Dense(**layer_config)(x)
            elif layer_type == 'dropout':
                x = layers.Dropout(**layer_config)(x)
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name='mushroom_classifier')
        
        logger.info(f"TensorFlow model created with {self.model_name}")
    
    def _build_pytorch_model(self):
        """Build model using PyTorch."""
        try:
            import torch
            import torch.nn as nn
            from torchvision import models
        except ImportError:
            raise ImportError("PyTorch is required for this model")
        
        # Select base model
        if self.model_name == 'mobilenet_v2':
            base_model = models.mobilenet_v2(pretrained=True)
            feature_dim = base_model.classifier[1].in_features
        elif self.model_name == 'resnet50':
            base_model = models.resnet50(pretrained=True)
            feature_dim = base_model.fc.in_features
            # Remove classification head
            base_model = nn.Sequential(*list(base_model.children())[:-1])
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        # Freeze base model if requested
        if self.freeze_base:
            for param in base_model.parameters():
                param.requires_grad = False
        
        # Build custom head
        head_layers = []
        prev_dim = feature_dim
        
        for layer_config in CUSTOM_HEAD:
            layer_type = layer_config.get('type')
            
            if layer_type == 'dense':
                units = layer_config['units']
                activation = layer_config.get('activation', 'relu')
                head_layers.append(nn.Linear(prev_dim, units))
                if activation == 'relu':
                    head_layers.append(nn.ReLU())
                prev_dim = units
            
            elif layer_type == 'dropout':
                rate = layer_config['rate']
                head_layers.append(nn.Dropout(rate))
        
        # Output layer
        head_layers.append(nn.Linear(prev_dim, self.num_classes))
        
        # Create full model
        class MushroomClassifier(nn.Module):
            def __init__(self, base_model, head):
                super().__init__()
                self.base = base_model
                self.head = nn.Sequential(*head)
            
            def forward(self, x):
                x = self.base(x)
                x = x.view(x.size(0), -1)  # Flatten
                x = self.head(x)
                return x
        
        self.model = MushroomClassifier(base_model, head_layers)
        logger.info(f"PyTorch model created with {self.model_name}")
    
    def compile(self, optimizer: str = 'adam',
               learning_rate: float = 0.001,
               loss: str = 'categorical_crossentropy',
               metrics: List[str] = None,
               framework: str = 'tensorflow'):
        """
        Compile the model.
        
        Args:
            optimizer: Optimizer name
            learning_rate: Learning rate
            loss: Loss function
            metrics: Metrics to track
            framework: Deep learning framework
        """
        if framework == 'tensorflow':
            self._compile_tensorflow(optimizer, learning_rate, loss, metrics)
        elif framework == 'pytorch':
            self._compile_pytorch(optimizer, learning_rate, loss)
    
    def _compile_tensorflow(self, optimizer: str, learning_rate: float,
                           loss: str, metrics: List[str]):
        """Compile TensorFlow model."""
        try:
            import tensorflow as tf
            from tensorflow.keras import optimizers
        except ImportError:
            raise ImportError("TensorFlow is required")
        
        # Create optimizer
        if optimizer == 'adam':
            opt = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            opt = optimizer
        
        # Compile
        if metrics is None:
            metrics = ['accuracy']
        
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"Model compiled with {optimizer}, lr={learning_rate}")
    
    def _compile_pytorch(self, optimizer: str, learning_rate: float, loss: str):
        """Compile PyTorch model."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required")
        
        # Store optimizer and loss for training
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.loss_name = loss
        
        logger.info(f"PyTorch model configured with {optimizer}, lr={learning_rate}")
    
    def summary(self):
        """Print model summary."""
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")
        
        if hasattr(self.model, 'summary'):
            # TensorFlow model
            self.model.summary()
        else:
            # PyTorch model
            print(self.model)
    
    def save(self, checkpoint_dir: Path = ARTIFACTS_DIR) -> Path:
        """
        Save model and metadata.
        
        Args:
            checkpoint_dir: Directory to save model
            
        Returns:
            Path to saved model
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = checkpoint_dir / MODEL_FINAL_NAME
        metadata_path = checkpoint_dir / METADATA_FILE
        
        # Save model
        if hasattr(self.model, 'save'):
            # TensorFlow model
            self.model.save(str(model_path))
            logger.info(f"Model saved to {model_path}")
        else:
            # PyTorch model
            import torch
            torch.save(self.model.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        return model_path
    
    def load(self, model_path: Path = ARTIFACTS_DIR / MODEL_FINAL_NAME):
        """
        Load saved model.
        
        Args:
            model_path: Path to model file
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Assume TensorFlow for now
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(str(model_path))
            logger.info(f"Model loaded from {model_path}")
        except:
            logger.error(f"Failed to load model from {model_path}")
            raise


class Predictor:
    """Make predictions using the trained model."""
    
    def __init__(self, model: ImageRecognitionModel, 
                 species_ids: List[str]):
        """
        Initialize predictor.
        
        Args:
            model: Trained ImageRecognitionModel
            species_ids: List of species IDs for label mapping
        """
        self.model = model
        self.species_ids = species_ids
        self.id_to_species = {i: sp_id for i, sp_id in enumerate(species_ids)}
    
    def predict(self, image: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Make prediction on an image.
        
        Args:
            image: Preprocessed image array
            top_k: Number of top predictions to return
            
        Returns:
            List of predictions with species_id and confidence
        """
        if self.model.model is None:
            raise RuntimeError("Model not loaded")
        
        # Batch predict
        batch = np.expand_dims(image, axis=0)
        predictions = self.model.model.predict(batch)[0]
        
        # Get top-k
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'species_id': self.id_to_species[idx],
                'class_index': int(idx),
                'confidence': float(predictions[idx]),
            })
        
        return results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    model = ImageRecognitionModel()
    model.build_model(framework='tensorflow')
    model.compile()
    model.summary()
