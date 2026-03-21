"""
Image preprocessing and augmentation for mushroom identification.

Handles loading, resizing, normalizing, and augmenting images for training.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Union
from PIL import Image
import cv2

from config.image_model_config import (
    INPUT_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    AUGMENTATION,
    PREPROCESSING,
)


class ImageProcessor:
    """Process and augment images for model training and inference."""
    
    def __init__(self, input_size: Tuple[int, int] = INPUT_SIZE,
                 normalization: str = "imagenet",
                 mean: List[float] = IMAGENET_MEAN,
                 std: List[float] = IMAGENET_STD):
        """
        Initialize image processor.
        
        Args:
            input_size: Target size (height, width)
            normalization: Normalization method ('imagenet', 'standard', 'minmax')
            mean: Mean values for normalization (per channel)
            std: Standard deviation values for normalization (per channel)
        """
        self.input_size = input_size
        self.normalization = normalization
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load image from file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array (H, W, C) with values in range [0, 255]
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image with PIL
        image = Image.open(image_path).convert('RGB')
        return np.array(image, dtype=np.uint8)
    
    def resize_image(self, image: np.ndarray, 
                    size: Optional[Tuple[int, int]] = None,
                    interpolation: str = "bilinear") -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image (H, W, C)
            size: Target size (height, width). If None, uses self.input_size
            interpolation: Interpolation method ('bilinear', 'nearest', 'bicubic')
            
        Returns:
            Resized image
        """
        if size is None:
            size = self.input_size
        
        # OpenCV uses (width, height) but we use (height, width)
        target_h, target_w = size
        
        # Select interpolation
        interp_map = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4,
        }
        interpolation_flag = interp_map.get(interpolation, cv2.INTER_LINEAR)
        
        # Resize
        resized = cv2.resize(image, (target_w, target_h), 
                            interpolation=interpolation_flag)
        
        return resized
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image using specified method.
        
        Args:
            image: Input image with values in [0, 255]
            
        Returns:
            Normalized image
        """
        image = image.astype(np.float32)
        
        if self.normalization == "imagenet":
            # ImageNet normalization: (x - mean) / std
            image = image / 255.0
            image = (image - self.mean) / self.std
        
        elif self.normalization == "standard":
            # Standard normalization: (x - mean) / std
            image = image / 255.0
            mean = np.mean(image)
            std = np.std(image)
            if std > 0:
                image = (image - mean) / std
        
        elif self.normalization == "minmax":
            # Min-max normalization: (x - min) / (max - min)
            image = image / 255.0
        
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")
        
        return image.astype(np.float32)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for inference (resize + normalize).
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image ready for model input
        """
        image = self.resize_image(image)
        image = self.normalize_image(image)
        return image
    
    def augment_image(self, image: np.ndarray, augmentation_level: str = 'medium') -> List[np.ndarray]:
        """
        Create augmented versions of an image.
        
        Args:
            image: Input image (H, W, C)
            augmentation_level: 'low', 'medium', or 'high'
            
        Returns:
            List of augmented images
        """
        augmented = [image]
        
        aug_config = AUGMENTATION
        
        # Rotation
        if augmentation_level in ['medium', 'high']:
            rotation_range = int(aug_config['rotation_range'])
            for angle in [-rotation_range // 2, rotation_range // 2]:
                if angle != 0:
                    h, w = image.shape[:2]
                    center = (w // 2, h // 2)
                    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(image, matrix, (w, h))
                    augmented.append(rotated)
        
        # Brightness adjustment
        if augmentation_level in ['medium', 'high']:
            brightness_range = aug_config['brightness_range']
            for brightness in [brightness_range[0], brightness_range[1]]:
                if brightness != 1.0:
                    bright = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
                    bright = np.clip(bright, 0, 255).astype(np.uint8)
                    augmented.append(bright)
        
        # Horizontal flip
        if augmentation_level == 'high' and aug_config['horizontal_flip']:
            flipped = cv2.flip(image, 1)
            augmented.append(flipped)
        
        return augmented
    
    def batch_preprocess(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of images
            
        Returns:
            Batch of preprocessed images (N, H, W, C)
        """
        batch = np.array([self.preprocess(img) for img in images])
        return batch


class DataGenerator:
    """Generate batches of images for training."""
    
    def __init__(self, image_paths: List[str],
                 labels: List[int],
                 image_processor: ImageProcessor,
                 batch_size: int = 32,
                 augment: bool = True,
                 shuffle: bool = True):
        """
        Initialize data generator.
        
        Args:
            image_paths: List of paths to image files
            labels: List of class labels
            image_processor: ImageProcessor instance
            batch_size: Number of images per batch
            augment: Whether to apply data augmentation
            shuffle: Whether to shuffle data
        """
        self.image_paths = np.array(image_paths)
        self.labels = np.array(labels)
        self.image_processor = image_processor
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(image_paths))
        
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a batch of images and labels.
        
        Args:
            batch_idx: Batch index
            
        Returns:
            Tuple of (images, labels)
        """
        # Get indices for this batch
        start_idx = batch_idx * self.batch_size
        end_idx = min((batch_idx + 1) * self.batch_size, len(self.indices))
        batch_indices = self.indices[start_idx:end_idx]
        
        # Load and preprocess images
        batch_images = []
        batch_labels = []
        
        for idx in batch_indices:
            try:
                # Load image
                image = self.image_processor.load_image(self.image_paths[idx])
                
                # Augment if training
                if self.augment:
                    augmented = self.image_processor.augment_image(image, 'medium')
                    image = augmented[np.random.randint(0, len(augmented))]
                
                # Preprocess
                image = self.image_processor.preprocess(image)
                batch_images.append(image)
                batch_labels.append(self.labels[idx])
            
            except Exception as e:
                print(f"Error loading image {self.image_paths[idx]}: {e}")
                continue
        
        if not batch_images:
            raise RuntimeError(f"Failed to load any images in batch {batch_idx}")
        
        # Convert to arrays
        X = np.array(batch_images)
        y = np.array(batch_labels)
        
        return X, y
    
    def on_epoch_end(self):
        """Shuffle data at end of epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)


if __name__ == "__main__":
    # Example usage
    processor = ImageProcessor()
    
    # Create dummy test image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Preprocess
    preprocessed = processor.preprocess(dummy_image)
    print(f"Original shape: {dummy_image.shape}")
    print(f"Preprocessed shape: {preprocessed.shape}")
    print(f"Preprocessed dtype: {preprocessed.dtype}")
    print(f"Preprocessed range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
