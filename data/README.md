# Dataset Directory

This directory contains the mushroom identification dataset and associated data management tools.

## Directory Structure

```
data/
├── raw/                          # Original source data (CSV files)
│   ├── species.csv              # Master species information (20 species)
│   ├── species_traits.csv       # Morphological traits for each species
│   ├── species_images.csv       # Image metadata and sourcing info
│   ├── lookalikes.csv           # Dangerous lookalike relationships
│   ├── dataset_split.csv        # Train/validation/test splits
│   └── images/                  # Mushroom images organized by species
│       ├── CH001/               # Chanterelle images
│       ├── BU001/               # Porcini images
│       ├── MO001/               # Morel images
│       └── ...                  # 17 more species folders
│
├── processed/                    # ML-ready processed data (generated)
│   ├── dataset.json             # Complete dataset in JSON format
│   ├── trait_features.csv       # Wide-format trait feature matrix
│   ├── training_metadata.json   # Image processor settings & augmentation
│   └── ...
│
├── evaluation/                   # Test results and metrics (generated)
│   ├── test_results.json
│   ├── confusion_matrices/
│   └── metrics/
│
├── dataset_utils.py             # Core dataset loading and utilities
├── validate_data.py             # Dataset validation script
├── prepare_data.py              # Data preparation and augmentation
└── README.md                    # This file
```

## Quick Start

### 1. Validate the Dataset

```bash
cd data
python3 validate_data.py --data-dir raw --stats
```

This validates:
- ✓ All CSV files have correct schema
- ✓ Species references are valid
- ✓ Images are properly documented
- ✓ Lookalike relationships are correct
- ✓ Train/test split distribution

### 2. Prepare Data for Training

```bash
python3 prepare_data.py --data-dir raw --output-dir processed
```

This generates:
- `processed/dataset.json` - Complete dataset
- `processed/trait_features.csv` - Encoded trait features
- `processed/training_metadata.json` - Training configuration

### 3. Use Dataset in Python

```python
from dataset_utils import MushroomDataset

# Load dataset
dataset = MushroomDataset('raw')
dataset.load_all()

# Get species information
info = dataset.get_species_info('CH001')

# Get training statistics
stats = dataset.get_statistics()
print(f"Total species: {stats['total_species']}")
print(f"Total images: {stats['total_images']}")

# Get dangerous lookalikes for a species
lookalikes = dataset.get_dangerous_lookalikes('CH001')

# Get images for training
train_species = dataset.get_species_by_split('TRAIN')
train_images = dataset.get_images_by_split('TRAIN')
```

## Data Files

### Core CSV Files

1. **species.csv** (20 species)
   - Master record of all mushroom species
   - Columns: species_id, scientific_name, swedish_name, english_name, edible, toxicity_level, priority_lookalike
   - See: [Data Dictionary](../Docs/05-data-dictionary.md)

2. **species_traits.csv** (100+ traits)
   - Morphological characteristics organized by category
   - Categories: CAP, GILLS, STEM, FLESH, HABITAT, SEASON, GROWTH
   - See: [Data Dictionary](../Docs/05-data-dictionary.md)

3. **species_images.csv** (20+ images)
   - Metadata for all images (path, quality, source, stage, lighting, angle)
   - References: file paths in images/ directory
   - See: [Data Dictionary](../Docs/05-data-dictionary.md)

4. **lookalikes.csv** (8 lookalike pairs)
   - Documents dangerous lookalike relationships
   - Includes identification tips and distinguishing features
   - Confusion likelihood: LOW, MEDIUM, HIGH, CRITICAL
   - See: [Data Dictionary](../Docs/05-data-dictionary.md)

5. **dataset_split.csv** (image assignments)
   - Assigns each image to TRAIN/VALIDATION/TEST
   - Distribution: 70% train, 15% validation, 15% test
   - See: [Data Dictionary](../Docs/05-data-dictionary.md)

## Python API

### MushroomDataset Class

Main class for loading and querying the dataset.

```python
dataset = MushroomDataset('raw')
dataset.load_all()

# Get information
dataset.get_species_info(species_id)           # Full species info with traits & images
dataset.get_species_by_name(name, language)    # Find species by common name
dataset.get_traits_for_species(species_id)     # Get trait DataFrame
dataset.get_images_for_species(species_id)     # Get image DataFrame
dataset.get_dangerous_lookalikes(species_id)   # Get lookalike warnings
dataset.get_species_by_split(split_set)        # Get species in TRAIN/VALIDATION/TEST
dataset.get_images_by_split(split_set)         # Get images in split set

# Statistics
dataset.get_statistics()                        # Dataset summary statistics
```

### DataValidator Class

Validates dataset integrity and quality.

```python
validator = DataValidator(dataset)
is_valid, errors, warnings = validator.validate_all()

if errors:
    print("Validation errors:")
    for error in errors:
        print(f"  - {error}")

if warnings:
    print("Validation warnings:")
    for warning in warnings:
        print(f"  - {warning}")
```

### DataExporter Class

Export dataset to various formats.

```python
exporter = DataExporter(dataset)

# Export to JSON
exporter.export_to_json('processed/dataset.json')

# Export trait features for ML
exporter.export_trait_features('processed/trait_features.csv')
```

## Dataset Statistics

Current dataset contains:

- **20 species total**
  - 12 edible species
  - 8 toxic species (2 extremely toxic)
- **21 sample images** (template - expand as needed)
  - Each species: 1-5 sample images
- **100+ trait records**
  - 5-6 traits per species category
- **8 lookalike pairs**
  - All documented with distinguishing features
- **Split distribution**
  - Train: 14 images (70%)
  - Validation: 3 images (15%)
  - Test: 3 images (15%)

**Note:** The CSV files contain template data with a few complete entries. In production:
- Expand images to 10-20 per species (200+ total)
- Complete trait data for all species
- Document all lookalike relationships
- Balance splits across all species

## Adding New Data

### Adding a New Species

1. Add row to `species.csv` with all required fields
2. Extract and add traits to `species_traits.csv`
3. Create folder: `raw/images/<species_id>/`
4. Add images to folder and document in `species_images.csv`
5. If toxic, document lookalikes in `lookalikes.csv`
6. Assign images to splits in `dataset_split.csv`
7. Run validation: `python3 validate_data.py --data-dir raw`

### Adding Images

1. Save image file to appropriate species folder
2. Use naming convention: `<species_id>_<seq>_<stage>_<lighting>_<angle>.jpg`
3. Add row to `species_images.csv` with metadata
4. Assign to split in `dataset_split.csv`

## Dependencies

### Required Python Libraries

For running the data utilities:
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `PIL` (Pillow) - Image processing (optional, for image augmentation)
- `opencv-python` - Advanced image processing (optional)

Install dependencies:
```bash
pip install pandas numpy pillow opencv-python
```

### Command-line Tools

- `python3` - Python 3.7+ required

## File Sizes (Approximate)

| File | Size | Notes |
|------|------|-------|
| species.csv | ~3 KB | Master data, fixed size |
| species_traits.csv | ~25 KB | Will grow with more detail |
| species_images.csv | ~10 KB | Will grow significantly |
| lookalikes.csv | ~8 KB | Fixed by species count |
| dataset_split.csv | ~2 KB | Will grow with images |
| images/ | ~50 MB | Template only, expand to 200-500 MB |

## Documentation

- **Full Data Dictionary:** See [Docs/05-data-dictionary.md](../Docs/05-data-dictionary.md)
- **Dataset Construction Guide:** See [Docs/03-dataset-construction.md](../Docs/03-dataset-construction.md)
- **System Architecture:** See [Docs/04-system-architecture.md](../Docs/04-system-architecture.md)

## Data Quality

### Current State

- ✅ Schema defined for all CSV files
- ✅ Template data with 5+ complete species
- ✅ Validation scripts functional
- ✅ Data utilities ready for use
- ⚠️ Limited images (template only)
- ⚠️ Partial trait coverage

### Production Readiness Checklist

Before training models:
- [ ] 200+ total images (10+ per species)
- [ ] 95%+ trait data coverage
- [ ] All lookalikes documented
- [ ] Balanced train/validation/test splits
- [ ] Validation script passes with no errors
- [ ] Image files verified (exist, readable, correct format)

## Contact & Questions

For questions about the dataset:
- See [Docs/05-data-dictionary.md](../Docs/05-data-dictionary.md) for detailed column descriptions
- Review [Docs/03-dataset-construction.md](../Docs/03-dataset-construction.md) for collection guidelines
- Check Python docstrings in `dataset_utils.py` for API details

---

**Version:** 1.0  
**Last Updated:** 2026-03-14  
**Status:** Template ready for production data population
