# Phase 2: Dataset Construction and System Architecture

## Part A: Dataset Construction

### Objective
Create a structured dataset containing 20 mushroom species with trait data, images, and metadata suitable for training ML models and evaluating identification methods.

### Dataset Schema

#### 1. Species Master Data (species.csv)
```
species_id,scientific_name,swedish_name,english_name,edible,toxicity_level,priority_lookalike
CH001,Cantharellus cibarius,Kantarell,Chanterelle,TRUE,SAFE,FALSE_CHANTERELLE
BU001,Boletus edulis,Karljohan,Porcini,TRUE,SAFE,BITTER_BOLETE
...
```

**Columns:**
- `species_id`: Unique species identifier (e.g., CH001 for Chanterelle)
- `scientific_name`: Full scientific name
- `swedish_name`: Swedish common name
- `english_name`: English common name
- `edible`: Boolean (TRUE/FALSE)
- `toxicity_level`: SAFE, TOXIC, EXTREMELY_TOXIC
- `priority_lookalike`: ID of primary dangerous lookalike

#### 2. Traits Data (species_traits.csv)
```
species_id,trait_category,trait_name,trait_value,value_type
CH001,CAP,shape,funnel-shaped,categorical
CH001,CAP,color,yellow-orange,categorical
CH001,CAP,size_cm,3-7,range
CH001,GILLS,attachment,decurrent,categorical
...
```

**Columns:**
- `species_id`: Reference to species.csv
- `trait_category`: CAP, GILLS, STEM, FLESH, HABITAT, SEASON, GROWTH
- `trait_name`: Specific trait name
- `trait_value`: Actual value (can be categorical or numeric)
- `value_type`: categorical, numeric, range, text

**Trait Value Standards:**
- **Categorical traits:** Use standardized values from taxonomy (e.g., "funnel-shaped" not "funnel")
- **Numeric ranges:** Format as "min-max" (e.g., "3-7" cm)
- **Multiple values:** Pipe-separated (e.g., "white|cream|pale yellow")

#### 3. Images Metadata (species_images.csv)
```
image_id,species_id,file_path,image_stage,lighting,angle,source,quality,suitable_for_training
IMG_CH001_001,CH001,images/CH001/CH001_001_young_sunny_top.jpg,young,direct_sunlight,top-down,user_photo,HIGH,TRUE
IMG_CH001_002,CH001,images/CH001/CH001_002_mature_shade_side.jpg,mature,dappled,side,user_photo,HIGH,TRUE
...
```

**Columns:**
- `image_id`: Unique image identifier
- `species_id`: Species reference
- `file_path`: Local path to image file
- `image_stage`: young, developing, mature
- `lighting`: direct_sunlight, dappled, shade, artificial
- `angle`: top-down, side, ground-level, close-up
- `source`: field_guide, user_photo, online_database, mushroom_db
- `quality`: LOW, MEDIUM, HIGH
- `suitable_for_training`: Boolean (exclude poor quality/unclear images)

**Image Requirements:**
- Minimum 10 images per species (target: 15-20)
- Mix of growth stages and conditions
- High quality (clear, in focus, good lighting)
- Diverse angles and perspectives
- Indicate source/attribution for each image

#### 4. Lookalike Relationships (lookalikes.csv)
```
lookalike_id,edible_species_id,toxic_species_id,confusion_likelihood,distinguishing_features
LA001,CH001,FALSE_CHANTERELLE,HIGH,"Thinner ridges; fragile stem; watery flesh"
LA002,BU001,BITTER_BOLETE,HIGH,"Immediate blue bruising; bitter flesh taste"
...
```

**Columns:**
- `lookalike_id`: Unique lookalike pair identifier
- `edible_species_id`: The safe species
- `toxic_species_id`: The dangerous lookalike
- `confusion_likelihood`: LOW, MEDIUM, HIGH, CRITICAL
- `distinguishing_features`: Key differences for identification

#### 5. Training/Test Split (dataset_split.csv)
```
species_id,image_id,split_set,reason
CH001,IMG_CH001_001,TRAIN,balanced_distribution
CH001,IMG_CH001_011,TEST,held_out_for_evaluation
BU001,IMG_BU001_005,VALIDATION,robust_evaluation
...
```

**Columns:**
- `species_id`: Species reference
- `image_id`: Image reference
- `split_set`: TRAIN, VALIDATION, TEST
- `reason`: Why assigned to this set

**Distribution Target:**
- TRAIN: 70% (for model training)
- VALIDATION: 15% (for hyperparameter tuning)
- TEST: 15% (for final evaluation)

### Directory Structure
```
mushroom-identification-system/
├── data/
│   ├── raw/
│   │   ├── species.csv
│   │   ├── species_traits.csv
│   │   ├── species_images.csv
│   │   ├── lookalikes.csv
│   │   ├── dataset_split.csv
│   │   └── images/
│   │       ├── CH001/              (Chanterelle)
│   │       │   ├── CH001_001.jpg
│   │       │   ├── CH001_002.jpg
│   │       │   └── ...
│   │       ├── BU001/              (Porcini)
│   │       │   ├── BU001_001.jpg
│   │       │   └── ...
│   │       └── ... (other species)
│   ├── processed/
│   │   ├── features_image_recognition.pkl
│   │   ├── features_trait_ml.pkl
│   │   ├── training_data_image.npz
│   │   └── training_data_traits.pkl
│   └── evaluation/
│       ├── test_results.json
│       ├── confusion_matrices/
│       └── metrics/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_validation.ipynb
│   └── 03_data_augmentation.ipynb
└── scripts/
    ├── data_validation.py
    ├── data_augmentation.py
    └── dataset_utils.py
```

### Data Collection & Preparation Steps

#### Step 1: Extract Trait Data from Nya Svampboken
- Manual or semi-automated extraction of trait descriptions
- Standardize values using trait taxonomy
- Create species_traits.csv with all traits
- Include variability ranges (e.g., color can vary by season/maturity)

#### Step 2: Source and Organize Images
**Primary sources:**
- Field guide (Nya Svampboken) - digitized photographs
- Publicly available mushroom databases (Arkivoc, MushroomObserver, iNaturalist)
- User-contributed photos from mushroom forums/communities
- Licensed mushroom photography collections

**Organization:**
- Create species folders (CH001/, BU001/, etc.)
- Name images descriptively (species_stage_condition_angle.jpg)
- Verify image quality and suitability
- Create image metadata CSV

#### Step 3: Validate and Clean Data
- Check for missing traits
- Verify trait values are standardized
- Validate image quality
- Ensure balanced distribution across species
- Identify and resolve conflicts/inconsistencies

#### Step 4: Create Data Augmentation Pipeline (for ML)
- Image resizing (e.g., 224x224 for CNN input)
- Image normalization (mean/std per model)
- Rotation, brightness, contrast augmentation for robustness
- Create augmented training dataset

#### Step 5: Train/Test Split
- Stratified split by species (ensure all species in train, validation, test)
- Separate images of same specimen if possible
- Document split strategy in dataset_split.csv

### Data Quality Metrics

**Completeness:**
- Target: 95%+ traits per species
- Missing traits should be documented and justified
- Handle missing values appropriately in models

**Balance:**
- Target: ±20% variation in images per species
- Dangerous species: ensure adequate representation
- Edible/toxic split: roughly balanced

**Image Quality:**
- Resolution: minimum 800x600, target 1200+ pixels
- Clarity: sharp, in-focus, identifiable features
- Diversity: multiple angles, growth stages, seasons

### Data Validation Checklist
- [ ] All 20 species have master data
- [ ] Trait data complete for all species (90%+ coverage per species)
- [ ] Minimum 10 images per species, 200+ total images
- [ ] All images verified for quality and attribution
- [ ] Lookalike relationships documented for all dangerous pairs
- [ ] Train/validation/test splits stratified by species
- [ ] Data values standardized and validated
- [ ] No missing critical information
- [ ] Attribution and licenses documented

---

## Part B: System Architecture Design

### System Overview

The mushroom identification system consists of three independent identification modules that are integrated into a hybrid system:

```
┌─────────────────────────────────────────────────────┐
│          Mushroom Identification System              │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │        User Interface Layer (Mobile/Web)      │   │
│  │  - Image capture / upload                     │   │
│  │  - Trait questionnaire                        │   │
│  │  - Results display                            │   │
│  └──────────────────────────────────────────────┘   │
│                       ↓                              │
│  ┌──────────────────────────────────────────────┐   │
│  │      Hybrid Identification Engine             │   │
│  │                                               │   │
│  │  ┌─────────────────┬──────────────┬────────┐ │   │
│  │  │ Image Recog.    │ Trait-Based  │ LLM    │ │   │
│  │  │ Module          │ Module       │ Module │ │   │
│  │  └─────────────────┴──────────────┴────────┘ │   │
│  │           ↓              ↓           ↓        │   │
│  │  ┌──────────────────────────────────────┐   │   │
│  │  │  Confidence Aggregation Engine       │   │   │
│  │  │  - Combine predictions               │   │   │
│  │  │  - Generate ranked species list      │   │   │
│  │  │  - Identify lookalikes               │   │   │
│  │  └──────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────┘   │
│                       ↓                              │
│  ┌──────────────────────────────────────────────┐   │
│  │      Data & Knowledge Layer                  │   │
│  │  - Mushroom species database                 │   │
│  │  - Trait ontology                            │   │
│  │  - Lookalike relationships                   │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### Module Designs

#### 1. Image Recognition Module (Phase 3)
**Purpose:** Classify mushroom images to species

**Components:**
- Image preprocessing (resizing, normalization)
- Transfer learning CNN (MobileNet/EfficientNet/ResNet)
- Fine-tuned classification layer
- Confidence scoring

**Input:** Mushroom photograph
**Output:** Top-k species predictions with confidence scores

**Example:**
```
Input: mushroom_photo.jpg
Output: [
  {"species_id": "CH001", "name": "Kantarell", "confidence": 0.92},
  {"species_id": "FALSE_CH", "name": "Falsk kantarell", "confidence": 0.07},
  {"species_id": "BU001", "name": "Karljohan", "confidence": 0.01}
]
```

#### 2. Trait-Based Classification Module (Phase 4)
**Purpose:** Identify species based on observable characteristics

**Components:**
- Trait questionnaire (optional fields for incomplete data)
- Feature vectorization (categorical encoding)
- Decision tree / Random forest classifier
- Confidence estimation (probability scores)

**Input:** User-provided traits
```json
{
  "cap_shape": "funnel-shaped",
  "cap_color": "yellow-orange",
  "gills_attachment": "decurrent",
  "habitat_type": "coniferous_forest",
  "season": "summer"
}
```

**Output:** Species predictions with confidence
```
[
  {"species_id": "CH001", "confidence": 0.89},
  {"species_id": "FALSE_CH", "confidence": 0.08}
]
```

#### 3. LLM-Based Classification Module (Phase 5)
**Purpose:** Generate predictions from natural language descriptions

**Components:**
- Observation-to-description conversion
- LLM API integration (GPT-4 / Llama)
- Prompt templates for mushroom classification
- Response parsing

**Input:** Natural language observation
```
"Found in birch forest in July. Bright yellow with funnel-shaped cap. 
Orange-colored gills that run down the stem. Has a pleasant fruity smell."
```

**Output:** Species predictions
```
[
  {"species_id": "CH001", "confidence": 0.85},
  {"species_id": "FALSE_CH", "confidence": 0.10}
]
```

#### 4. Confidence Aggregation Engine (Phase 6)
**Purpose:** Combine predictions from three modules

**Algorithm:**
1. Collect predictions from all three modules
2. Calculate weighted average confidence (tunable weights)
3. Rank species by aggregated confidence
4. Identify lookalike warnings
5. Generate final recommendation

**Weighting strategy:**
```
aggregated_confidence = (
  0.4 * image_confidence +
  0.35 * trait_confidence +
  0.25 * llm_confidence
)
```

**Lookalike detection:**
- If high-confidence species has known dangerous lookalike
- Add warning and alternative identification tips

### Data & Knowledge Layer

#### Species Database Schema
```sql
species (
  species_id PRIMARY KEY,
  scientific_name TEXT,
  swedish_name TEXT,
  english_name TEXT,
  description TEXT,
  edible BOOLEAN,
  toxicity_level ENUM,
  habitat_types TEXT[],
  seasons TEXT[],
  created_at TIMESTAMP
)

species_traits (
  species_id FK,
  trait_category ENUM,
  trait_name TEXT,
  trait_values TEXT[],
  variability TEXT,
  PRIMARY KEY (species_id, trait_category, trait_name)
)

lookalike_relationships (
  lookalike_id PRIMARY KEY,
  safe_species_id FK,
  dangerous_species_id FK,
  confusion_likelihood ENUM,
  distinguishing_features TEXT
)

trained_models (
  model_id PRIMARY KEY,
  module_type ENUM,
  model_path TEXT,
  created_at TIMESTAMP,
  accuracy FLOAT,
  version TEXT
)
```

### API Specification

#### Image Recognition Endpoint
```
POST /api/identify/image
Content-Type: multipart/form-data

Request:
{
  "image": <binary>,
  "language": "sv" | "en",  // Swedish or English
  "top_k": 5  // Number of predictions
}

Response:
{
  "success": true,
  "predictions": [
    {
      "species_id": "CH001",
      "scientific_name": "Cantharellus cibarius",
      "swedish_name": "Kantarell",
      "english_name": "Chanterelle",
      "confidence": 0.92,
      "edible": true
    },
    ...
  ]
}
```

#### Trait-Based Identification Endpoint
```
POST /api/identify/traits
Content-Type: application/json

Request:
{
  "traits": {
    "cap_shape": "funnel-shaped",
    "cap_color": "yellow-orange",
    "gills_attachment": "decurrent",
    ...
  },
  "language": "sv" | "en"
}

Response:
{
  "success": true,
  "predictions": [
    {
      "species_id": "CH001",
      "confidence": 0.89,
      ...
    }
  ]
}
```

#### Hybrid Identification Endpoint
```
POST /api/identify/hybrid
Content-Type: multipart/form-data

Request:
{
  "image": <binary>,
  "traits": {...},
  "observation": "...",
  "language": "sv" | "en"
}

Response:
{
  "success": true,
  "primary_prediction": {
    "species_id": "CH001",
    "confidence": 0.92,
    ...
  },
  "alternatives": [...],
  "lookalikes": [
    {
      "species_id": "FALSE_CH",
      "name": "Falsk kantarell",
      "risk_level": "HIGH",
      "distinguishing_features": [...]
    }
  ],
  "combined_confidence": 0.92
}
```

### UML Diagrams

#### Use Case Diagram
```
                              ┌─────────────┐
                              │    User     │
                              └──────┬──────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
            ┌───────────────┐ ┌──────────────┐ ┌──────────────┐
            │ Capture Image │ │Answer Traits │ │Describe Find │
            └───────┬───────┘ └──────┬───────┘ └──────┬───────┘
                    │                │                │
                    └────────┬───────┴────────┬───────┘
                             │                │
                         ┌───▼──────────────────▼───┐
                         │  Identify Species        │
                         │  (Hybrid Approach)       │
                         └───┬──────────────────┬───┘
                             │                  │
                         ┌───▼──────┐      ┌────▼──────────┐
                         │View Result│      │Check Lookalikes│
                         └──────────┘      └─────────────────┘
```

#### Class Diagram (High-level)
```
┌────────────────────────────────────────────────────┐
│ MushroomIdentificationSystem                        │
├────────────────────────────────────────────────────┤
│ - imageRecognitionModule: ImageRecognizer          │
│ - traitModule: TraitClassifier                     │
│ - llmModule: LLMClassifier                         │
│ - aggregationEngine: ConfidenceAggregator          │
│ - speciesDB: SpeciesDatabase                       │
├────────────────────────────────────────────────────┤
│ + identifyImage(image): Prediction[]               │
│ + identifyTraits(traits): Prediction[]             │
│ + identifyHybrid(...): HybridResult                │
│ + getSpeciesInfo(id): SpeciesInfo                  │
│ + getLookalikes(id): LookalikeInfo[]               │
└────────────────────────────────────────────────────┘
         ▲              ▲              ▲
         │              │              │
    ┌────┴────┐    ┌────┴────┐    ┌───┴────┐
    │ Image   │    │ Trait   │    │  LLM   │
    │Module   │    │ Module  │    │ Module │
    └─────────┘    └─────────┘    └────────┘
```

### Integration Points

1. **User Input Validation**
   - Validate image format and size
   - Validate trait selections
   - Validate text input

2. **Preprocessing**
   - Image resizing and normalization
   - Trait encoding
   - Observation parsing

3. **Inference**
   - Call each module independently
   - Collect confidence scores
   - Handle API errors gracefully

4. **Post-processing**
   - Aggregate predictions
   - Check lookalike database
   - Format response for UI

5. **Error Handling**
   - Retry logic for API failures
   - Graceful degradation (skip failed modules)
   - Clear error messages to user

### Performance Requirements

| Component | Requirement | Notes |
|-----------|-------------|-------|
| Image preprocessing | < 1 second | Local on device if possible |
| Image recognition inference | < 3 seconds | Mobile or cloud |
| Trait-based inference | < 500ms | Local model |
| LLM API call | < 5 seconds | Including network latency |
| Complete hybrid prediction | < 8 seconds | All modules running in parallel |
| UI response time | < 500ms | From end of inference to display |

### Security & Privacy

- No sensitive user data stored
- Images processed locally where possible
- LLM API calls use minimal context
- Database access restricted
- API endpoints require authentication (future)

---

## Next Steps

1. Implement data collection and validation scripts
2. Create initial dataset with 5-10 species for prototype
3. Validate data quality and completeness
4. Design detailed database schema
5. Create Docker containers for reproducibility

---

**Status:** Phase 2 - In Progress  
**Created:** 2026-03-14  
**Last Updated:** 2026-03-14
