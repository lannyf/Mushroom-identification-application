# Mushroom Identification System — Bottom-Up Guide

> **How to read this guide:** Each section builds on the previous one. Start at Part 1 (Data) and work upward. By the time you reach the API layer, you will understand every piece of data every function produces.

---

## Part 0 — Project Overview

### What this system does

This is a **multimodal mushroom identification pipeline**. A user uploads a photo of a mushroom, and the system tries to identify the species by combining four independent signals:

| Signal | Method | Confidence |
|--------|--------|------------|
| Image | EfficientNet-B3 CNN + classical CV fallback | 35 % weight |
| Expert key | Swedish decision tree (`key.xml`) | 45 % weight |
| Trait database | Morphological trait matching | 20 % weight |
| LLM | Ollama llama3.2:3b (optional) | Used in hybrid mode |

The system also warns about toxic lookalikes — a critical safety feature.

### Architecture at a glance

```
┌─────────────────────────────────────────────────────────────────┐
│                         Flutter App                              │
│              (User takes photo, answers questions)               │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP
┌───────────────────────────▼─────────────────────────────────────┐
│                    Java Spring Boot Proxy                        │
│              (Routes to Python API, auth, caching)               │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP
┌───────────────────────────▼─────────────────────────────────────┐
│                  Python FastAPI (api/main.py)                    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
│  │  Step 1     │ │  Step 2     │ │  Step 3     │ │  Step 4   │ │
│  │  CNN/CV     │ │  Tree       │ │  Trait DB   │ │  Fusion   │ │
│  │  + LLM      │ │  (key.xml)  │ │  compare    │ │  aggregate│ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Directory structure

```
project-root/
├── api/                    # FastAPI HTTP layer
│   ├── main.py             # Routes, singletons, orchestration
│   ├── schemas.py          # Pydantic request/response models
│   └── scoring.py          # Pure helper functions (no HTTP)
├── benchmarks/             # Evaluation framework (9 methods, 60 images)
│   ├── runners/            # One wrapper per identification method
│   ├── metrics.py          # Accuracy, coverage, ECE, bootstrap CI
│   ├── reports.py          # JSON / CSV / Markdown output
│   └── visualize.py        # Matplotlib plots
├── config/                 # Central configuration
│   ├── image_model_config.py   # CNN architecture, species list, training params
│   └── segmentation_config.py  # YOLO thresholds
├── data/                   # Data files
│   ├── raw/                # Master data (CSV, XML, evaluation images)
│   └── dataset_utils.py    # DataFrame loaders, validators, exporters
├── models/                 # Core identification algorithms
│   ├── cnn_classifier.py           # EfficientNet-B3
│   ├── visual_trait_extractor.py   # OpenCV colour/shape/texture analysis
│   ├── key_tree_traversal.py       # Swedish XML decision tree
│   ├── trait_database_comparator.py# Trait DB + lookalike checker
│   ├── llm_classifier.py           # Ollama/OpenAI LLM wrapper
│   ├── final_aggregator.py         # Hierarchical 3-step fusion
│   ├── hybrid_classifier.py        # Late fusion (weighted/geometric/voting)
│   └── mushroom_segmenter.py       # YOLOv8-seg instance segmentation
├── java-backend/           # Spring Boot proxy (Maven project)
├── mushroom_id_app/        # Flutter frontend
├── artifacts/              # Trained model weights, plots, samples
└── Makefile                # Build / run commands
```

---

## Part 1 — Data Layer (The Foundation)

Everything in this project flows from five master data files. Understanding them first makes every model easier to follow.

### 1.1 `data/raw/species.csv` — The Species Master List

The canonical registry of all 50 mushroom species the system knows about.

```csv
species_id,scientific_name,swedish_name,english_name,edible,toxicity_level,priority_lookalike
CA.CI,Cantharellus cibarius,Kantarell,Chanterelle,TRUE,SAFE,HY.PS
BO.ED,Boletus edulis,Karljohan,Porcini,TRUE,SAFE,CA.CA
BO.BA,Boletus badius,Brunsopp,Bay Bolete,TRUE,SAFE,CA.CA
AM.MU,Amanita muscaria,Flugsvamp,Fly Agaric,FALSE,PSYCHOACTIVE,AM.VI
```

**Key concept:** `species_id` is the universal foreign key. Every other file references it. The CNN was trained on only 7 of these 50 species (the "in-distribution" set).

### 1.2 `data/raw/species_traits.xml` — Morphological Trait Profiles

1,049 rows of structured morphological data. Each species has traits grouped by category.

```xml
<species id="CA.CI">
  <trait_group category="CAP">
    <trait name="shape" value_type="categorical" variability="consistent">funnel-shaped</trait>
    <trait name="color" value_type="categorical" variability="yellow to deep orange">yellow-orange</trait>
    <trait name="surface_texture" value_type="categorical" variability="smooth with wavy edges">smooth</trait>
  </trait_group>
  <trait_group category="GILLS">
    <trait name="attachment" value_type="categorical" variability="consistent">decurrent (running down stem)</trait>
  </trait_group>
</species>
```

Loaded by `data/dataset_utils.py::load_species_traits_xml()` into a pandas DataFrame.

### 1.3 `data/raw/key.xml` — The Swedish Decision Tree

A hand-crafted expert decision tree in Swedish. Used by Step 2 of the pipeline.

```xml
<key question="Hur ser svampen ut?">
  <condition answer="Undersidan har åsar eller ådror" question="Vilken färg har svampen?">
    <condition answer="Hela svampen är gul">
      <decision namn="Kantarell" ätlighet="*">
        <mixupdecision namn="Narrkantarell (falsk kantarell)" ätlighet="0" skiljetecken="Orangeröd färg..."/>
      </decision>
    </condition>
  </condition>
</key>
```

**Important limitation:** This tree only covers ~20 Swedish species. Fly Agaric is **not in the tree at all**. Some species (False Chanterelle, Amanita virosa) appear only as `mixupdecision` lookalikes, not as traversable decisions.

### 1.4 `data/raw/lookalikes.csv` — Confusion Pairs

Pairs of edible and toxic species that look alike, with confusion likelihood and distinguishing features.

```csv
lookalike_id,edible_species_id,toxic_species_id,confusion_likelihood,distinguishing_features
LA001,CA.CI,HY.PS,HIGH,"True chanterelle: thick meaty ridges false gills..."
```

Used by `TraitDatabaseComparator` to generate safety warnings.

### 1.5 `data/raw/evaluation_images/` — Benchmark Images

60 JPEG images in 12 species folders (5 images each). 35 are "in-distribution" (the 7 CNN training species). 25 are "out-of-distribution" (species the CNN has never seen).

**Data relationships:**

```
species.csv (50 species)
    │ species_id
    ├─► species_traits.xml (1,049 trait rows)
    ├─► lookalikes.csv (confusion pairs)
    └─► evaluation_images/ (60 test images)

key.xml (Swedish tree)
    │ Swedish common names
    ├─► species.csv (via _KEY_XML_ALIASES mapping)
    └─► mixupdecision lookalikes
```

---

## Part 2 — Configuration

### `config/image_model_config.py`

The single source of truth for the CNN. Both **training** (`scripts/train_cnn.py`) and **inference** (`models/cnn_classifier.py`) import from here.

```python
BASE_MODEL = "efficientnet_b3"      # timm model name
INPUT_SIZE = (300, 300)             # after center-crop
RESIZE_SIZE = 320                   # before center-crop
SPECIES = [                         # 7 classes, order = model output order
    "Fly Agaric", "Chanterelle", "False Chanterelle",
    "Porcini", "Other Boletus", "Amanita virosa", "Black Trumpet",
]
NUM_CLASSES = len(SPECIES)          # 7
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
WEIGHTS_PATH = Path("artifacts/cnn_weights.pt")
```

If you change `SPECIES` or `INPUT_SIZE` here, both training and inference stay in sync automatically.

### `config/segmentation_config.py`

Thresholds for deciding whether a YOLO segmentation mask is good enough to replace full-image traits. `USE_MASK_FOR_TRAITS = True` by default, so the visual trait extractor will attempt to segment the mushroom and recompute traits on masked pixels when the mask passes quality checks.

---

## Part 3 — Core Models (Bottom-Up)

Now we walk through every model file. For each one: **what it takes in, what it does, and what it gives back.**

### 3.1 `models/visual_trait_extractor.py` — Step 1: Image Analysis

**Purpose:** Analyse a raw mushroom photo and extract both a species prediction and a structured trait dictionary.

**Public API:**

```python
def extract(image_bytes: bytes) -> dict:
    """Returns {"ml_prediction": {...}, "visible_traits": {...}}"""
```

**What happens inside:**

1. **Load image** → OpenCV BGR array.
2. **Classical CV analysis** (always runs):
   - `analyse_colours()` — KMeans clustering on HSV pixels + heuristic rules → dominant colour, red/orange/brown/white ratios
   - `analyse_shape()` — Otsu thresholding → largest contour → aspect ratio + circularity → cap shape (convex/flat/funnel/wavy/bell)
   - `analyse_texture()` — Canny edge detection + Hough line transform → surface texture (smooth/fibrous/scaly) + `has_ridges` boolean
   - `analyse_brightness()` — mean value channel → dark/medium/bright
3. **Species scoring** — Combines colour/shape/texture signals with weighted heuristics → raw scores for all 7 CNN species.
4. **CNN check** — Tries to load `MushroomCNN` via `get_classifier()`. If trained weights exist, uses CNN softmax probabilities instead of CV scores.
5. **Build `visible_traits`** dict — The 5 key fields every downstream model needs:
   ```python
   visible_traits = {
       "dominant_color": "yellow",
       "secondary_color": "orange",
       "cap_shape": "funnel-shaped",
       "surface_texture": "smooth",
       "has_ridges": True,
       "brightness": "medium",
       "colour_ratios": {"red": 0.02, "orange_yellow": 0.65, ...},
   }
   ```
6. **Optional YOLO masking** — If `USE_MASK_FOR_TRAITS` is True, runs YOLO segmentation and, if the mask passes quality checks, recomputes traits only on masked pixels.

**Key design point:** The `visible_traits` dict is the **universal currency** of the pipeline. Every downstream model (tree, trait DB, LLM) consumes it.

---

### 3.2 `models/cnn_classifier.py` — The Neural Network

**Purpose:** Fine-tuned EfficientNet-B3 for 7-class mushroom classification.

**Public API:**

```python
class MushroomCNN:
    def predict(self, image_bytes: bytes) -> dict[str, float] | None:
        """Returns {"Chanterelle": 0.94, "Porcini": 0.03, ...} or None"""
```

**Key behaviours:**
- **Lazy singleton:** `get_classifier()` creates one instance and reuses it.
- **Graceful degradation:** If `artifacts/cnn_weights.pt` is missing, `is_trained` is `False` and `predict()` returns `None`. The visual trait extractor falls back to classical CV.
- **Image preprocessing:** Resize → CenterCrop → ToTensor → ImageNet normalize.

**Training:** `scripts/train_cnn.py` (not covered here) does transfer learning: freezes backbone for first ⅓ of epochs, then fine-tunes full network at 0.1× learning rate.

---

### 3.3 `models/key_tree_traversal.py` — Step 2: Expert Decision Tree

**Purpose:** Walk the Swedish `key.xml` decision tree. Auto-answer questions from `visible_traits`; ask the user when information is missing.

**Public API:**

```python
engine = KeyTreeEngine("data/raw/key.xml")

# Start a session
result = engine.start_session(session_id=None, visible_traits={...})
# Returns: {"status": "question", "session_id": "...", "question": "...", "options": [...]}
#     OR:  {"status": "conclusion", "species": "Kantarell", ...}

# Answer a question
result = engine.answer(session_id="...", answer="Hela svampen är gul")
```

**Internal architecture:**

1. **Parse XML** at init into `QuestionNode` / `ConditionNode` / `DecisionNode` dataclasses.
2. **Tree compatibility pre-check:** Before traversing, checks if the Step 1 top prediction is:
   - **Exact match** — species has a full traversable path in the tree (e.g. Chanterelle → Kantarell)
   - **Lookalike-only** — species only appears as a `mixupdecision` (e.g. False Chanterelle)
   - **Unsupported** — species not in the tree at all (e.g. Fly Agaric)

   For lookalike-only and unsupported species with high ML confidence (≥0.85), the tree is **skipped entirely** — the engine returns a synthetic conclusion.

3. **Auto-answer loop:** `_try_auto_answer()` maps `visible_traits` to Swedish option strings using heuristics:
   - "åsar eller ådror" (ridges) → `has_ridges=True` + yellow/orange dominant
   - "rör" (pores) → brown dominant + no ridges
   - "skivor" (gills) → red/white Fly Agaric pattern, or white Amanita
   - Colour questions → direct `dominant_color` mapping

4. **Session management:** Each traversal gets a UUID. Sessions live in `_sessions` dict until a conclusion is reached.

**Output shapes:**

```python
# Question
{"status": "question", "session_id": "uuid", "question": "Vilken färg har svampen?",
 "options": ["Hela svampen är gul", "Gråbrun hatt", ...], "auto_answered": [...]}

# Conclusion
{"status": "conclusion", "species": "Kantarell", "edibility": "*",
 "edibility_label": "edible", "url": "...", "lookalikes": [...], "path": [...]}
```

---

### 3.4 `models/trait_database_comparator.py` — Step 3: Trait Validation

**Purpose:** Compare the visually extracted traits against the formal trait profile of a candidate species. Also find dangerous lookalikes.

**Public API:**

```python
comparator = TraitDatabaseComparator("data/raw")
result = comparator.compare("Kantarell", visible_traits={...})
# Returns structured dict with match score, conflicts, and lookalikes
```

**Internal flow:**

1. **Name resolution:** Swedish name → `species_id` via `_KEY_XML_ALIASES` (hard-coded map) or fuzzy token overlap on `species.csv`.
2. **Load traits:** Pull the species' trait profile from `species_traits.xml` via `_load_traits()`.
3. **Compare visible → DB:** For each of 5 comparable traits, compute match quality:
   - **cap_color** — Fuzzy colour matching via synonym groups (orange ≈ yellow-orange ≈ reddish-orange)
   - **cap_shape** — Shape synonym map (funnel-shaped ≈ trumpet ≈ infundibuliform)
   - **cap_texture** — Texture map (smooth ≈ silky ≈ glabrous)
   - **ridges** — Boolean `has_ridges` vs gill attachment description
   - **stem_color** — Secondary colour comparison
4. **Weighted score:** Each trait has a weight (cap_color=0.30, cap_shape=0.25, ridges=0.20, texture=0.15, stem_color=0.10). Quality scores: exact=1.0, partial=0.5, conflict=0.0.
5. **Lookalike lookup:** Scans `lookalikes.csv` for pairs involving this species. For each lookalike, loads both trait profiles, computes differences, and flags `safety_alert=True` if the lookalike is toxic.

**Benchmark addition:** `rank_all_species(visible_traits)` compares against **all 50 species** and returns a sorted list. This is what the `trait_db` benchmark runner uses.

---

### 3.5 `models/llm_classifier.py` — Natural Language Classifier

**Purpose:** Feed a text description of a mushroom to an LLM and get a structured species prediction.

**Public API:**

```python
classifier = LLMClassifier(backend_type="ollama")  # or "openai" or "mock"
result = classifier.classify(observation="Yellow funnel-shaped mushroom with decurrent ridges...")
# Returns PredictionResult with top_species, confidence, reasoning
```

**Architecture:**

1. **`SpeciesDatabase`** — Loads all 50 species from `species.csv` at init. Builds a formatted list for the prompt.
2. **`LLMPromptTemplate`** — Constructs a system prompt with:
   - Safety disclaimer
   - Full species list (English + Swedish + scientific + edibility)
   - JSON response schema instruction
   - Few-shot examples (Chanterelle, Fly Agaric, Porcini)
3. **`LLMBackend` (ABC)** — Three implementations:
   - `MockLLMBackend` — Keyword-based rule fallback (no network)
   - `OpenAIBackend` — GPT-4 via API key
   - `OllamaBackend` — Local llama3.2:3b at `localhost:11434`
4. **`LLMClassifier.classify()`** — Sends prompt, parses JSON response, returns `PredictionResult`.

**Important:** The benchmark uses `llama3.2:3b` on CPU (~25–30 s per sample). With only 5 sparse traits as input, accuracy is very low (~2 %). This is a known limitation of the small model + sparse input, not a bug.

---

### 3.6 `models/final_aggregator.py` — Step 4: Hierarchical Fusion

**Purpose:** Combine the outputs of Steps 1, 2, and 3 into a single final answer with weighted confidence.

**Public API:**

```python
aggregator = FinalAggregator("data/raw/species.csv")
result = aggregator.aggregate(step1_result, step2_result, step3_result)
```

**Confidence formula:**

```
overall = 0.45 × tree_conf + 0.35 × image_conf + 0.20 × trait_conf
if Step 1 and Step 2 agree on species:
    overall += 0.10  (capped at 1.0)
```

**Candidate priority:** Step 2 (tree) → Step 3 DB resolution → Step 1 (CNN fallback). The tree is the primary authority because it follows a validated expert key.

**Output:**

```python
{
    "final_recommendation": {
        "species_id": "CA.CI", "swedish_name": "Kantarell",
        "english_name": "Chanterelle", "overall_confidence": 0.97,
        "confidence_breakdown": {"image_analysis": 0.94, "tree_traversal": 1.0, "trait_match": 0.82},
        "reasoning": "Image analysis: ... Tree traversal concluded 'Kantarell' ... Trait match: 82% ...",
    },
    "ml_alternatives": [...],       # Step 1 top-5
    "exchangeable_species": [...],  # Step 3 lookalikes
    "safety_warnings": [...],
    "verdict": "edible",
    "method_agreement": "full",
}
```

---

### 3.7 `models/hybrid_classifier.py` — Late Fusion Engine

**Purpose:** An alternative fusion approach that combines CNN + trait-based + LLM predictions at the **score level** rather than the pipeline level.

**Public API:**

```python
hybrid = HybridClassifier(aggregation_method=AggregationMethod.WEIGHTED_AVERAGE)
result = hybrid.classify(
    image_prediction=MethodPrediction(method="image", species="Chanterelle", confidence=0.94, ...),
    trait_prediction=MethodPrediction(method="trait", species="Chanterelle", confidence=0.88, ...),
    llm_prediction=MethodPrediction(method="llm", species="Porcini", confidence=0.45, ...),
)
```

**Three strategies:**

| Strategy | How it works | Best for |
|----------|-------------|----------|
| `WEIGHTED_AVERAGE` | Weighted sum of confidences (image=0.4, trait=0.35, llm=0.25) | General use |
| `GEOMETRIC_MEAN` | Multiply confidences, take Nth root | Punishing disagreements |
| `VOTING` | Ranked voting with point assignment | When methods are equally reliable |

**Used by:** `POST /identify` (the one-shot endpoint) and the `multimodal_weighted/geometric/voting` benchmark runners.

---

### 3.8 `models/mushroom_segmenter.py` — YOLO Instance Segmentation

**Purpose:** Find and segment the mushroom region in a photo, isolating it from background clutter.

**Public API:**

```python
segmenter = get_segmenter()  # lazy singleton
result = segmenter.segment(image_bytes)
# Returns {"instances": [...], "selected_index": 0}
```

**What it does:**
1. Runs YOLOv8-seg on the image.
2. For each detected instance, computes quality metrics: area ratio, fragmentation, hole ratio, boundary irregularity, skin-pixel ratio.
3. Filters out bad detections (extreme aspect ratios, too much skin tone = likely hand).
4. Ranks by confidence + centrality bias.

**Integration:** The visual trait extractor optionally uses the best mask to recompute colour/shape/texture only on mushroom pixels. This is controlled by `segmentation_config.USE_MASK_FOR_TRAITS` (enabled by default).

---

## Part 4 — API Layer (Wiring It All Together)

### `api/main.py` — FastAPI Application

At startup, `main.py` creates **singleton instances** of every model:

```python
HYBRID     = HybridClassifier(aggregation_method=AggregationMethod.WEIGHTED_AVERAGE)
KEY_TREE   = KeyTreeEngine("data/raw/key.xml")
COMPARATOR = TraitDatabaseComparator("data/raw")
AGGREGATOR = FinalAggregator("data/raw/species.csv")
LLM        = LLMClassifier(backend_type="ollama")  # or None if Ollama offline
```

### Endpoints

#### `POST /identify` — One-Shot Classification

The simplest endpoint. Upload an image + optional trait JSON. Returns a fused prediction immediately.

**Flow inside:**
1. `image_scores(image_bytes)` → calls `visual_trait_extractor.extract()` → gets CNN/CV scores + visible_traits
2. `trait_scores(trait_data)` → rule-based scoring from questionnaire answers
3. `ollama_scores(LLM, metrics, traits, extraction)` → LLM or rule-based fallback
4. `build_prediction()` wraps each in a `MethodPrediction`
5. `HYBRID.classify(...)` → fuses all three → `HybridResult`
6. `adapt_result()` → converts to API response shape

#### `POST /identify/Species_tree_traversal/start` — Step 2 Begin

Accepts `visible_traits` from Step 1. Returns first question or a conclusion.

#### `POST /identify/Species_tree_traversal/answer` — Step 2 Continue

Accepts `session_id` + `answer`. Returns next question or conclusion.

#### `POST /identify/comparison/compare` — Step 3

Accepts `swedish_name` (from Step 2 conclusion) + `visible_traits`. Returns trait match score + lookalikes.

#### `POST /identify/prediction/finalize` — Step 4

Accepts the full outputs of Steps 1, 2, and 3. Returns the hierarchically aggregated final answer.

### `api/scoring.py` — Pure Business Logic

No HTTP here — just functions that transform data:

- `normalize(scores)` → Softmax-like normalisation (clips negatives, sums to 1)
- `image_scores(image_bytes)` → Runs `extract_visual_traits()`, returns `(scores_dict, metrics_dict, extraction_dict)`
- `trait_scores(traits_dict)` → Rule-based scoring from questionnaire (colour, cap_shape, gill_type, stem_type, habitat, season)
- `llm_scores(metrics, traits)` → Rule-based fallback when Ollama is offline
- `ollama_scores(classifier, metrics, traits, extraction)` → Tries Ollama, falls back to `llm_scores()`
- `adapt_result()` → Converts `HybridResult` into the API response JSON shape

### `api/schemas.py` — Pydantic Models

```python
class Step2StartRequest(BaseModel):
    session_id: Optional[str] = None
    visible_traits: Dict[str, Any]

class Step2AnswerRequest(BaseModel):
    session_id: str
    answer: str

class Step3CompareRequest(BaseModel):
    swedish_name: str
    visible_traits: Dict[str, Any]

class Step4FinalizeRequest(BaseModel):
    trait_extraction_result: Dict[str, Any]
    Species_tree_traversal_result: Dict[str, Any]
    comparison_result: Dict[str, Any]
```

---

## Part 5 — Benchmark Suite

The `benchmarks/` package provides a **unified evaluation framework** that treats all 9 identification methods identically.

### Design Philosophy

Every method must produce a `RunnerResult`:

```python
@dataclass
class RunnerResult:
    method_name: str
    predictions: List[Tuple[str, float]]   # [(species_id, confidence), ...] sorted best-first
    coverage: bool = True                  # Did the method attempt a prediction?
    inference_time_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

Because all methods speak the same output language, the benchmark engine can compare them without knowing which is which.

### The 9 Methods

| Runner | Class | What it does |
|--------|-------|-------------|
| `cnn` | `CNNRunner` | EfficientNet-B3, returns top-5 |
| `tree_auto` | `TreeRunner(mode="auto")` | Auto-answers from traits, stops at first unanswered question |
| `tree_oracle` | `TreeRunner(mode="oracle")` | Feeds pre-recorded correct answers from `oracle_answers.json` |
| `trait_db` | `TraitDBRunner` | Ranks all 50 species by trait match score |
| `llm` | `LLMRunner` | Text-only LLM via 5-trait observation |
| `multimodal_final` | `MultimodalRunner(strategy="final_aggregator")` | Hierarchical 3-step fusion |
| `multimodal_weighted` | `MultimodalRunner(strategy="weighted")` | Late fusion, weighted average |
| `multimodal_geometric` | `MultimodalRunner(strategy="geometric")` | Late fusion, geometric mean |
| `multimodal_voting` | `MultimodalRunner(strategy="voting")` | Late fusion, majority voting |

### Caching Architecture

Two module-level caches prevent redundant work:

1. **`runners/_extract_cache.py`** — Caches `visual_trait_extractor.extract()` by image SHA-256. The tree, trait_db, LLM, and multimodal runners all use the same image — this ensures YOLO/CV analysis runs **once per image**.

2. **`runners/_llm_cache.py`** — Caches Ollama responses by image SHA-256. The standalone LLM runner and the 3 hybrid multimodal runners all need the same LLM prediction — this ensures the slow Ollama call (~30 s) happens **at most once per image**.

### Metrics

- **Top-k accuracy** — Fraction of covered predictions where ground truth is in top-k (k=1 or 3)
- **Coverage** — Fraction of samples where the method produced any prediction
- **Macro F1** — Per-species F1, averaged
- **ECE** — Expected Calibration Error (how well confidences match empirical accuracy)
- **Bootstrap CI** — 95 % confidence intervals for accuracy via 2,000 resamples
- **OOD metrics** — ID vs OOD average confidence gap (CNN only)
- **Pairwise agreement** — Fraction of samples where two methods agree on top-1

### Reports

- `report.json` — Full structured dump with per-image breakdowns
- `report.csv` — Spreadsheet: one row per image, columns per method
- `report.md` — Thesis-ready Markdown table with accuracy + 95% CI
- `accuracy_comparison.png` — Bar chart of Top-1 vs Top-3
- `agreement_heatmap.png` — Pairwise method agreement matrix
- `ood_confidence.png` — Box plot of ID vs OOD confidence
- `confidence_distribution.png` — Histogram of correct vs incorrect confidences

---

## Part 6 — End-to-End Data Flow (Concrete Example)

Let's trace what happens when a user uploads a photo of a **Chanterelle**.

### Step 1 — Visual Trait Extraction

```python
extract(image_bytes) → {
    "ml_prediction": {
        "top_species": "Chanterelle",
        "confidence": 0.94,
        "method": "cnn",
        "top_k": [
            {"species": "Chanterelle", "confidence": 0.94},
            {"species": "False Chanterelle", "confidence": 0.03},
            ...
        ],
        "reasoning": "EfficientNet-B3 CNN prediction — dominant colour 'yellow', cap shape 'funnel-shaped'.",
    },
    "visible_traits": {
        "dominant_color": "yellow",
        "secondary_color": "orange",
        "cap_shape": "funnel-shaped",
        "surface_texture": "smooth",
        "has_ridges": True,
        "brightness": "medium",
        "colour_ratios": {"red": 0.02, "orange_yellow": 0.65, "brown": 0.08, "white": 0.05},
    },
}
```

### Step 2 — Tree Traversal

```python
engine.start_session(visible_traits=visible_traits) →
# _try_auto_answer("Hur ser svampen ut?", ["åsar", "skivor", "rör"], visible_traits)
# has_ridges=True + yellow dominant → returns "åsar eller ådror"
# _try_auto_answer("Vilken färg har svampen?", ["gul", "gråbrun", "svart"], visible_traits)
# dominant_color="yellow" → returns "Hela svampen är gul"
# Reaches decision node → "Kantarell"

{
    "status": "conclusion",
    "species": "Kantarell",
    "edibility": "*",
    "edibility_label": "edible",
    "lookalikes": [
        {"name": "Narrkantarell (falsk kantarell)", "edibility": "0", "distinguishing_feature": "Orangeröd färg..."}
    ],
    "path": ["Undersidan har åsar eller ådror", "Hela svampen är gul"],
    "auto_answered": [
        {"question": "Hur ser svampen ut?", "answer": "Undersidan har åsar eller ådror", "source": "image_analysis"},
        {"question": "Vilken färg har svampen?", "answer": "Hela svampen är gul", "source": "image_analysis"},
    ],
}
```

### Step 3 — Trait Database Comparison

```python
comparator.compare("Kantarell", visible_traits) → {
    "status": "ok",
    "candidate": {"species_id": "CA.CI", "swedish_name": "Kantarell", "english_name": "Chanterelle", ...},
    "name_match_score": 1.0,
    "trait_match": {
        "score": 0.82,
        "matched": [
            {"trait": "cap_color", "visible_value": "yellow", "db_value": "yellow-orange", "quality": "partial", "weight": 0.30},
            {"trait": "cap_shape", "visible_value": "funnel-shaped", "db_value": "funnel-shaped", "quality": "exact", "weight": 0.25},
            {"trait": "ridges", "visible_value": "True", "db_value": "decurrent (running down stem)", "quality": "exact", "weight": 0.20},
        ],
        "conflicts": [],
        "not_comparable": ["cap_texture", "stem_color"],
    },
    "lookalikes": [
        {"species_id": "HY.PS", "swedish_name": "Falsk kantarell", "confusion_likelihood": "HIGH", "safety_alert": True, ...}
    ],
    "safety_alert": False,  # Chanterelle itself is safe, but lookalike is toxic
}
```

### Step 4 — Final Aggregation

```python
aggregator.aggregate(step1, step2, step3) → {
    "final_recommendation": {
        "species_id": "CA.CI",
        "swedish_name": "Kantarell",
        "english_name": "Chanterelle",
        "overall_confidence": 1.0,  # 0.45×1.0 + 0.35×0.94 + 0.20×0.82 = 0.94 + 0.10 bonus
        "confidence_breakdown": {"image_analysis": 0.94, "tree_traversal": 1.0, "trait_match": 0.82},
        "reasoning": "Image analysis: EfficientNet-B3 CNN prediction... Tree traversal concluded 'Kantarell' (2 steps auto-resolved from image, 0 answered by user). Trait database match: 82% (3 traits matched, 0 conflicts).",
    },
    "ml_alternatives": [
        {"species": "Chanterelle", "confidence": 0.94, "swedish_name": "Kantarell", ...},
        {"species": "False Chanterelle", "confidence": 0.03, "swedish_name": "Falsk kantarell", ...},
    ],
    "exchangeable_species": [/* lookalike info from Step 3 */],
    "safety_warnings": [],
    "verdict": "edible",
    "method_agreement": "full",
}
```

---

## Part 7 — Key Design Decisions & Gotchas

### Why the tree speaks Swedish but the CNN speaks English

The decision tree (`key.xml`) is sourced from a Swedish mushroom guide (svampguiden.com). The CNN was trained on an English-labelled dataset. The system bridges them via `species.csv` — every species has both an `english_name` and a `swedish_name`, and mapping tables like `_KEY_XML_ALIASES` and `_STEP1_NAME_TO_ID` handle translation.

### Why Fly Agaric is unsupported in `key.xml`

Fly Agaric (`Flugsvamp`) does not appear anywhere in the Swedish decision tree. This is a **real limitation of the data source**, not a bug. When the CNN predicts Fly Agaric with high confidence (≥0.85), the tree is skipped and a synthetic conclusion is returned with a warning that the tree cannot validate it.

### Why `trait_db` has ~3% top-1 accuracy

The trait database comparator scores **all 50 species** using only **5 visible traits** (dominant colour, cap shape, texture, ridges, brightness). With so little information, many species look identical. It is not designed for standalone identification — its purpose is **validation** of a candidate species produced by the tree.

### Why `tree_auto` can beat `tree_oracle`

This is **survivorship bias**. The auto mode only attempts images where it can auto-answer the first few questions. These are the "easy" images where visual traits are clear. On these 39 easy images, it gets 71.8% correct. The oracle mode is forced to answer on **all** images, including 11 "hard" images where even correct early answers lead to wrong tree conclusions. Overall oracle: 68.0% on 83.3% coverage.

### The name-mapping chain

Different components use different naming conventions. The benchmark has three separate resolution paths:

```
CNN output: "Chanterelle" ──► CNN_NAME_TO_SPECIES_ID ──► "CA.CI"
Tree output: "Kantarell"  ──► _KEY_XML_ALIASES ──► "CA.CI"
LLM output: "Chanterelle (Kantarell)" ──► _resolve_llm_name() ──► "CA.CI"
                              (scans species.csv for English/Swedish match)
```

### Why `multimodal_final` uses `tree_oracle`

The `FinalAggregator` (multimodal_final) requires a Step 2 conclusion. In the benchmark, `tree_oracle` guarantees a conclusion by feeding correct answers. In production, the user would provide those answers interactively.

### The evaluation set includes a "synthetic OOD" species

`FO.BE` (Fomitopsis betulina, birch polypore) is a real mushroom image but was created synthetically and is absent from **all** components (CNN training, tree, trait DB, LLM species list). No method can possibly identify it. This is the strongest OOD test in the suite.

---

## Quick Reference: File → Responsibility

| File | What it does | Key function/class |
|------|-------------|-------------------|
| `models/visual_trait_extractor.py` | Image → traits + ML prediction | `extract(image_bytes)` |
| `models/cnn_classifier.py` | Neural network classifier | `MushroomCNN.predict()` |
| `models/key_tree_traversal.py` | Swedish decision tree | `KeyTreeEngine.start_session()` / `.answer()` |
| `models/trait_database_comparator.py` | Trait validation + lookalikes | `TraitDatabaseComparator.compare()` |
| `models/llm_classifier.py` | LLM natural language ID | `LLMClassifier.classify()` |
| `models/final_aggregator.py` | Hierarchical 3-step fusion | `FinalAggregator.aggregate()` |
| `models/hybrid_classifier.py` | Late fusion of CNN+trait+LLM | `HybridClassifier.classify()` |
| `models/mushroom_segmenter.py` | YOLO instance segmentation | `Segmenter.segment()` |
| `api/main.py` | FastAPI routes | `@app.post("/identify")` etc. |
| `api/scoring.py` | Business logic helpers | `image_scores()`, `trait_scores()` |
| `benchmarks/run_benchmark.py` | Evaluation orchestrator | `run()` |
| `benchmarks/metrics.py` | Metric computations | `compute_top_k_accuracy()`, `bootstrap_ci()` |
