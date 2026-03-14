# Phase 4: Trait-Based Classification Module

## Overview

Implemented a complete trait-based mushroom classification system using decision trees and random forests. This approach provides an identification method that doesn't require images and can be easily explained to users with rule-based reasoning.

## Architecture

### Core Components

#### 1. TraitEncoder (trait_processor.py)
Converts categorical trait values into numerical feature vectors.

**Features:**
- One-hot encoding for categorical traits (e.g., cap color: white, yellow, orange)
- Range encoding for numerical ranges (e.g., cap size: 3-7 cm → [min, max])
- Ordinal encoding for ordered categorical traits
- Fit/transform pattern with serialization support

**Key Methods:**
- `fit()` - Learn encoding scheme from trait data
- `transform()` - Convert trait dict to feature vector
- `save()/load()` - Persist encoder for inference

#### 2. TraitDataset (trait_processor.py)
Loads species trait data from CSV and prepares it for ML training.

**Features:**
- Load species_traits.csv and species.csv
- Build feature matrix from species trait descriptions
- Create train/val/test splits with stratification
- Species ID to label mapping with bilingual names

**Key Methods:**
- `prepare_features()` - Returns (X, y, feature_names)
- `get_species_name()` - Retrieve bilingual species names
- `save_encoder()` - Export encoder for inference

#### 3. TraitClassifier (trait_classifier.py)
Trains and manages classification models with scikit-learn backends.

**Algorithms Supported:**
1. **Decision Tree** - Interpretable, rule-based decisions
   - Max depth: 10 (prevents overfitting)
   - Criterion: Gini impurity
   - Good for understanding feature importance

2. **Random Forest** - High accuracy ensemble method
   - 100 estimators for stability
   - Max depth: 15 (deeper trees in ensemble)
   - N_jobs: -1 (parallel processing)
   - Reduced variance through averaging

**Key Methods:**
- `train()` - Train on features and labels
- `predict()` - Single prediction per sample
- `predict_proba()` - Class probabilities
- `predict_with_confidence()` - Top-k predictions with scores
- `get_feature_importance()` - Feature ranking
- `evaluate()` - Comprehensive metrics (accuracy, precision, recall, F1, ROC AUC)
- `save()/load()` - Model persistence

#### 4. TraitObservation (trait_processor.py)
Converts user observations to feature vectors during inference.

**Supports:**
- Dictionary-based input: `{'CAP.color': 'white', 'CAP.shape': 'round', ...}`
- List-based input: `[('CAP.color', 'white'), ('CAP.shape', 'round'), ...]`

### Training Pipeline (train_trait_model.py)

**Workflow:**
1. Load species trait CSV files
2. Prepare feature matrix with TraitEncoder
3. Split data: train (70%), validation (15%), test (15%)
4. Initialize classifier (decision tree or random forest)
5. Train on training set with validation monitoring
6. Evaluate on test set
7. Save model, scaler, encoder, and metadata
8. Log comprehensive metrics and feature importance

**CLI Options:**
- `--algorithm` - Model type (decision_tree, random_forest)
- `--train-size, --val-size, --test-size` - Data split ratios
- `--artifacts-dir` - Output directory
- `--random-state` - Reproducibility seed

**Output Artifacts:**
- `trait_classifier_{algorithm}.pkl` - Trained sklearn model
- `trait_scaler_{algorithm}.pkl` - Feature scaler
- `trait_metadata_{algorithm}.pkl` - Feature/class names and training history
- `trait_encoder.pkl` - Trait encoder for inference
- `trait_results_{algorithm}.json` - Metrics summary

### Evaluation Pipeline (evaluate_trait_model.py)

**Comprehensive Metrics:**
- Train/val/test accuracy, precision, recall, F1-score
- Per-class metrics with classification report
- Confusion matrix
- ROC AUC score
- Feature importance ranking

**Visualizations Generated:**
1. Confusion matrix heatmap
2. Feature importance bar chart
3. Algorithm comparison chart (if multiple models trained)

**CLI Options:**
- `--algorithm` - Evaluate specific algorithm or 'all'
- `--compare` - Generate algorithm comparison chart
- `--artifacts-dir` - Location of saved models

**Output:**
- `trait_evaluation_{algorithm}.json` - Detailed metrics
- `trait_confusion_matrix_{algorithm}.png` - Confusion matrix heatmap
- `trait_feature_importance_{algorithm}.png` - Feature ranking chart
- `trait_algorithm_comparison.png` - Cross-algorithm comparison

## Data Flow

```
species_traits.csv + species.csv
           ↓
    TraitDataset.prepare_features()
           ↓
    Feature Matrix (n_samples × n_features)
           ↓
    Train/Val/Test Split (70/15/15)
           ↓
    TraitClassifier.train()
           ↓
    [Trained Model] → Save artifacts
           ↓
    TraitClassifier.evaluate()
           ↓
    Metrics & Visualizations
```

## Feature Engineering Details

### Trait Categories Supported
1. **CAP** - shape, color, surface texture, size, margin
2. **GILLS** - attachment, density, color, edge
3. **STEM** - shape, color, surface, hollow/solid, size
4. **FLESH** - color, texture, smell
5. **HABITAT** - substrate type, tree association
6. **SEASON** - fruiting period
7. **GROWTH** - pattern (solitary, scattered, clustered)

### Feature Encoding Examples

**Categorical trait (one-hot):**
```
CAP.color: "yellow" → [1, 0, 0, 0] (if colors are yellow, white, orange, brown)
STEM.color: "white" → [0, 1, 0, 0]
```

**Range trait (min/max):**
```
CAP.size_cm: "3-7" → [3.0, 7.0]
STEM.size_cm: "4-6" → [4.0, 6.0]
```

**Ordinal trait:**
```
Variability: "consistent" → [0]
Variability: "variable" → [1]
Variability: "highly variable" → [2]
```

## Model Characteristics

### Decision Tree Classifier
**Pros:**
- Fully interpretable decision rules
- Shows which features matter most
- Fast inference
- Handles both categorical and numerical features

**Cons:**
- May overfit with deep trees
- Less robust than ensembles
- High variance between different datasets

**Use Case:** When explainability is critical

### Random Forest Classifier
**Pros:**
- High accuracy through ensemble averaging
- Reduces overfitting vs single tree
- Parallel training/inference
- Robust to feature scaling
- Handles imbalanced classes better

**Cons:**
- Less interpretable than single tree
- More memory overhead
- Slower training than single tree

**Use Case:** When accuracy is the priority

## Integration Points

### From Phase 3 (Image Recognition)
- Both models output probability distributions
- Can aggregate confidence scores

### To Phase 5 (LLM-Based)
- TraitObservation can encode LLM-generated trait descriptions
- Confidence scores can be combined with LLM predictions

### To Phase 6 (System Integration)
- Predictor class provides clean inference interface
- Confidence aggregation with image and LLM methods
- Feature importance explains predictions to users

## Usage Examples

### Training
```bash
# Train random forest
python scripts/train_trait_model.py --algorithm random_forest

# Train decision tree
python scripts/train_trait_model.py --algorithm decision_tree

# Custom split ratios
python scripts/train_trait_model.py --algorithm random_forest \
  --train-size 0.8 --val-size 0.1 --test-size 0.1
```

### Evaluation
```bash
# Evaluate single algorithm
python scripts/evaluate_trait_model.py --algorithm random_forest

# Evaluate all and compare
python scripts/evaluate_trait_model.py --algorithm all --compare

# Compare algorithms
python scripts/evaluate_trait_model.py --compare
```

### Inference (Example Code)
```python
from models.trait_processor import TraitEncoder, TraitObservation
from models.trait_classifier import TraitClassifier

# Load trained model
classifier = TraitClassifier(algorithm='random_forest', n_species=20)
classifier.load('artifacts/trait_classifier_random_forest.pkl',
                'artifacts/trait_scaler_random_forest.pkl',
                'artifacts/trait_metadata_random_forest.pkl')

# Encode user observations
encoder = TraitEncoder()
encoder.load('artifacts/trait_encoder.pkl')
observation = TraitObservation(encoder)

# User selects traits
traits = {
    'CAP.color': 'yellow-orange',
    'CAP.shape': 'funnel-shaped',
    'GILLS.color': 'pale-yellow',
    'STEM.color': 'yellow-orange'
}

# Get prediction with confidence
features = observation.from_dict(traits).reshape(1, -1)
predictions = classifier.predict_with_confidence(features, top_k=5)

# Display top predictions
for species, confidence in predictions[0]:
    print(f"{species}: {confidence:.2%}")
```

## Technical Details

### Feature Normalization
- StandardScaler applied to all features
- Fit on training set, applied to val/test
- Prevents feature scale bias in tree models

### Class Balance Handling
- Decision tree and random forest naturally handle imbalanced classes
- Could add class_weight parameter if needed
- Current implementation uses balanced class distribution (20 species)

### Hyperparameter Justification

**Decision Tree:**
- max_depth=10: Prevents deep overfitting while allowing sufficient complexity
- min_samples_split=2: Default, allows fine-grained splits
- min_samples_leaf=1: One sample leaves for specificity

**Random Forest:**
- n_estimators=100: Standard ensemble size for stability
- max_depth=15: Deeper than single tree, variance reduced by averaging
- n_jobs=-1: Parallel on all cores

### Cross-Validation Strategy
- Stratified split maintains class distribution
- Separate train/val/test to detect overfitting
- Val set monitors generalization during training

## Performance Expectations

With 20 species and ~5-6 traits per species:
- **Decision Tree**: 60-75% accuracy expected
- **Random Forest**: 75-85% accuracy expected

Performance will improve significantly with:
1. More detailed trait specifications
2. Finer-grained categorical values
3. Additional distinguishing features
4. Real user observations for calibration

## Files Created

- `models/trait_processor.py` (292 lines)
  - TraitEncoder, TraitDataset, TraitObservation classes

- `models/trait_classifier.py` (397 lines)
  - TraitClassifier, Predictor classes

- `scripts/train_trait_model.py` (315 lines)
  - Training pipeline with CLI

- `scripts/evaluate_trait_model.py` (368 lines)
  - Evaluation framework with visualizations

**Total: 1,372 lines of production code**

## Next Steps

1. **Train on Real Data**
   - Collect more detailed trait descriptions
   - Add rare/exotic species
   - Document user-observed variations

2. **Feature Engineering**
   - Add new trait categories
   - Create interaction features
   - Extract sub-traits from ranges

3. **Hyperparameter Tuning**
   - Grid search or random search
   - Cross-validation for robustness
   - Class weight balancing if needed

4. **Integration Testing**
   - Test with Phase 5 LLM output
   - Combine with Phase 3 image predictions
   - Build Phase 6 aggregation logic

5. **User Interface**
   - Trait selection questionnaire UI
   - Dynamic trait filtering
   - Explanation display

## Related Documentation

- `Docs/02-species-traits.md` - Species trait details
- `Docs/04-system-architecture.md` - System design with trait classifier component
- Phase 3 Summary - Image recognition module (parallel module)
