"""
Data augmentation and preprocessing for mushroom identification.

Handles image resizing, normalization, and augmentation for ML models.
"""

import os
import json
import numpy as np
import argparse
from pathlib import Path
from typing import Tuple, List
from dataset_utils import MushroomDataset

try:
    from PIL import Image
    import cv2
except ImportError:
    print("Warning: PIL/OpenCV not installed. Image processing will be limited.")
    Image = None
    cv2 = None


class ImageProcessor:
    """Process and augment images for ML training."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize image processor.
        
        Args:
            target_size: Target image size (height, width)
        """
        self.target_size = target_size
        self.mean = np.array([0.485, 0.456, 0.406])  # ImageNet normalization
        self.std = np.array([0.229, 0.224, 0.225])
    
    def resize_image(self, image_path: str) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Resized image as numpy array
        """
        if Image is None:
            raise ImportError("PIL is required for image processing")
        
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.target_size)
        return np.array(img)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image using ImageNet statistics.
        
        Args:
            image: Image as numpy array (0-255 range)
            
        Returns:
            Normalized image (0-1 range)
        """
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        return image
    
    def augment_image(self, image: np.ndarray, augmentation_level: str = 'medium') -> List[np.ndarray]:
        """
        Create augmented versions of an image.
        
        Args:
            image: Input image as numpy array
            augmentation_level: 'low', 'medium', or 'high'
            
        Returns:
            List of augmented images
        """
        augmented = [image]
        
        if cv2 is None:
            return augmented
        
        if augmentation_level in ['medium', 'high']:
            # Rotation
            for angle in [-15, 15]:
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(image, matrix, (w, h))
                augmented.append(rotated)
            
            # Brightness
            for alpha in [0.8, 1.2]:
                bright = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
                augmented.append(bright)
        
        if augmentation_level == 'high':
            # Horizontal flip
            flipped = cv2.flip(image, 1)
            augmented.append(flipped)
        
        return augmented


class TraitFeatureEncoder:
    """Encode categorical trait values for ML models."""
    
    def __init__(self, dataset: MushroomDataset):
        """Initialize encoder with dataset."""
        self.dataset = dataset
        self.encoders = {}
        self._build_encoders()
    
    def _build_encoders(self) -> None:
        """Build categorical encoders for each trait."""
        self.dataset.load_all()
        
        for trait_name in self.dataset.traits_df['trait_name'].unique():
            trait_data = self.dataset.traits_df[self.dataset.traits_df['trait_name'] == trait_name]
            values = set()
            
            for val in trait_data['trait_value']:
                if pd.isna(val):
                    continue
                # Handle pipe-separated values
                values.update(str(val).split('|'))
            
            self.encoders[trait_name] = {val.strip(): idx for idx, val in enumerate(sorted(values))}
    
    def encode_trait(self, trait_name: str, trait_value: str) -> int:
        """
        Encode a categorical trait value.
        
        Args:
            trait_name: Name of the trait
            trait_value: Value of the trait
            
        Returns:
            Encoded integer value
        """
        if trait_name not in self.encoders:
            raise ValueError(f"Unknown trait: {trait_name}")
        
        if trait_value not in self.encoders[trait_name]:
            return -1  # Unknown value
        
        return self.encoders[trait_name][trait_value]
    
    def get_encoder_vocab(self, trait_name: str) -> dict:
        """Get the vocabulary for a trait."""
        return self.encoders.get(trait_name, {})


def prepare_training_data(data_dir: str = 'data/raw', 
                         output_dir: str = 'data/processed',
                         augment: bool = False) -> None:
    """
    Prepare training data for ML models.
    
    Args:
        data_dir: Path to raw data
        output_dir: Path to output processed data
        augment: Whether to apply data augmentation
    """
    print(f"Preparing training data from {data_dir}...")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = MushroomDataset(data_dir)
    dataset.load_all()
    
    # Get train/test split
    train_images = dataset.split_df[dataset.split_df['split_set'] == 'TRAIN']
    test_images = dataset.split_df[dataset.split_df['split_set'] == 'TEST']
    
    print(f"  Training images: {len(train_images)}")
    print(f"  Test images: {len(test_images)}")
    
    # Create metadata for training
    metadata = {
        'train': train_images.to_dict('records'),
        'test': test_images.to_dict('records'),
        'species': dataset.species_df.to_dict('records'),
        'image_processor': {
            'target_size': (224, 224),
            'normalization': 'ImageNet'
        },
        'augmentation': 'enabled' if augment else 'disabled'
    }
    
    # Save metadata
    with open(Path(output_dir) / 'training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ Saved training metadata to {output_dir}/training_metadata.json")
    
    # Encode trait features
    print("  Encoding trait features...")
    try:
        import pandas as pd
        encoder = TraitFeatureEncoder(dataset)
        
        # Create feature matrix for traits
        features_list = []
        for species_id in dataset.species_df['species_id']:
            traits = dataset.traits_df[dataset.traits_df['species_id'] == species_id]
            features = {'species_id': species_id}
            for _, trait in traits.iterrows():
                features[f"{trait['trait_category']}_{trait['trait_name']}"] = trait['trait_value']
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        features_df.to_csv(Path(output_dir) / 'trait_features.csv', index=False)
        print(f"  ✓ Saved trait features to {output_dir}/trait_features.csv")
    except ImportError:
        print("  ⚠️  pandas not available, skipping trait feature encoding")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare mushroom identification dataset for ML training"
    )
    parser.add_argument(
        '--data-dir',
        default='data/raw',
        help='Path to raw data directory'
    )
    parser.add_argument(
        '--output-dir',
        default='data/processed',
        help='Path to output processed data'
    )
    parser.add_argument(
        '--augment',
        action='store_true',
        help='Apply data augmentation'
    )
    
    args = parser.parse_args()
    
    try:
        prepare_training_data(args.data_dir, args.output_dir, args.augment)
        print("\n✅ Data preparation complete!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
