"""
Data validation script for mushroom identification dataset.

This script validates the integrity and quality of the dataset.
"""

import sys
import argparse
from pathlib import Path
from dataset_utils import MushroomDataset, DataValidator, DataExporter


def main():
    parser = argparse.ArgumentParser(
        description="Validate mushroom identification dataset"
    )
    parser.add_argument(
        '--data-dir',
        default='data/raw',
        help='Path to raw data directory (default: data/raw)'
    )
    parser.add_argument(
        '--export',
        action='store_true',
        help='Export dataset to processed formats'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Print dataset statistics'
    )
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.data_dir}...")
    dataset = MushroomDataset(args.data_dir)
    try:
        dataset.load_all()
        print("✅ Dataset loaded successfully")
    except FileNotFoundError as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)
    
    # Validate dataset
    print("\nValidating dataset...")
    validator = DataValidator(dataset)
    is_valid, errors, warnings = validator.validate_all()
    
    if errors:
        print("\n❌ Validation Errors:")
        for error in errors:
            print(f"  • {error}")
    
    if warnings:
        print("\n⚠️  Validation Warnings:")
        for warning in warnings:
            print(f"  • {warning}")
    
    if is_valid and not warnings:
        print("\n✅ Dataset validation passed with no issues!")
    
    # Print statistics if requested
    if args.stats:
        print("\n📊 Dataset Statistics:")
        stats = dataset.get_statistics()
        print(f"  Total species: {stats['total_species']}")
        print(f"  Edible species: {stats['edible_species']}")
        print(f"  Toxic species: {stats['toxic_species']}")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Suitable for training: {stats['total_suitable_images']}")
        print(f"  Train set: {stats['train_count']}")
        print(f"  Validation set: {stats['validation_count']}")
        print(f"  Test set: {stats['test_count']}")
        print(f"  Lookalike pairs: {stats['lookalike_pairs']}")
    
    # Export if requested
    if args.export:
        print("\n📤 Exporting dataset...")
        exporter = DataExporter(dataset)
        
        # Create processed directory if it doesn't exist
        Path('data/processed').mkdir(parents=True, exist_ok=True)
        
        # Export to JSON
        exporter.export_to_json('data/processed/dataset.json')
        print("  ✓ Exported to data/processed/dataset.json")
        
        # Export trait features
        exporter.export_trait_features('data/processed/trait_features.csv')
        print("  ✓ Exported to data/processed/trait_features.csv")
    
    return 0 if is_valid else 1


if __name__ == '__main__':
    sys.exit(main())
