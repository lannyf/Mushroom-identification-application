# Benchmark Suite Implementation Plan

**For:** BSc Thesis — Multimodal Mushroom Identification vs Standalone Components  
**Scope:** Benchmark the actual methods that exist in the repository on 60 evaluation images  
**Output:** `artifacts/benchmarks/` with JSON/CSV/Markdown reports and plots

---

## Thesis Goal

The benchmark should support a thesis comparison between:

1. `CNN` image classification
2. `Species tree traversal` using `key.xml`
3. `Trait database comparison`
4. `LLM classification`
5. `Multimodal aggregation`

The key correction is that the current repository does **not** implement an LLM-driven Step 1 → Step 2 → Step 3 → Step 4 pipeline.

What actually exists is:

- `MushroomCNN` for image classification
- `KeyTreeEngine` for deterministic tree traversal
- `TraitDatabaseComparator` for deterministic trait matching
- `LLMClassifier` as a separate LLM-based method
- `FinalAggregator` and `HybridClassifier` as multimodal combiners

So the benchmark must evaluate the real methods that exist, not an imagined orchestration layer.

---

## Research Questions the Benchmark Should Support

This benchmark should make it possible to answer:

1. How well does the CNN perform alone, especially on out-of-distribution species?
2. How much of tree performance is limited by trait extraction rather than by the tree itself?
3. How competitive is trait-database matching as a standalone method?
4. How does the standalone LLM classifier compare with the non-LLM methods?
5. Do multimodal aggregators outperform the standalone methods?
6. Do different aggregation strategies produce meaningfully different trade-offs?

---

## Non-Negotiable Benchmark Rules

1. Benchmark the real code paths that exist in `models/` and `api/`.
2. Keep oracle answers restricted to evaluating the tree itself, not to boosting unrelated methods.
3. Save per-image outputs for every method so failures can be inspected.
4. Keep name mapping explicit because the repo uses English names, Swedish names, aliases, and `species_id`.
5. Separate in-distribution and out-of-distribution analysis for the CNN.

---

## Quick Reference: What You Are Building

```text
benchmarks/
├── __init__.py
├── config.py
├── dataset.py
├── oracle_answers.json
├── runners/
│   ├── __init__.py
│   ├── base.py
│   ├── cnn_runner.py
│   ├── tree_runner.py
│   ├── trait_db_runner.py
│   ├── llm_runner.py
│   └── multimodal_runner.py
├── metrics.py
├── reports.py
├── visualize.py
└── run_benchmark.py
```

**One likely code change in existing implementation:**

- `models/trait_database_comparator.py`
  Add `rank_all_species()` so the trait DB can be benchmarked as a ranked standalone method.

---

## Phase 0: Confirm the Evaluation Set

The evaluation set currently consists of 12 folders and 60 images total, with 5 images per folder.

| Folder | species_id | English name | CNN trained on it? | key.xml support |
|---|---|---|---|---|
| `AM.MU/` | `AM.MU` | Fly Agaric | yes | unsupported |
| `AM.VI/` | `AM.VI` | Amanita virosa | yes | lookalike only |
| `BO.ED/` | `BO.ED` | Porcini | yes | exact |
| `Brunsopp/` | `BO.BA` | Other Boletus | yes | exact |
| `CA.CI/` | `CA.CI` | Chanterelle | yes | exact |
| `CR.CO/` | `CR.CO` | Black Trumpet | yes | exact |
| `HY.PS/` | `HY.PS` | False Chanterelle | yes | lookalike only |
| `coprinus_comatus/` | `CO.CO` | Shaggy Inkcap | no | exact |
| `fomitopsis_betulina/` | `FO.BE` (synthetic) | Birch Polypore | no | unsupported |
| `lycoperdon_utriforme/` | `LY.PE` | Puffball | no | exact |
| `ramaria_botrytis/` | `RA.BO` | Clustered Coral | no | exact |
| `sparassis_crispa/` | `SP.CR` | Cauliflower Mushroom | no | exact |

This naturally creates:

- `35` in-distribution images for the CNN
- `25` out-of-distribution images for the CNN

That split should be a central part of the benchmark design.

**Note on `fomitopsis_betulina` (`FO.BE`):**
This species is absent from `species.csv`, `key.xml`, and the trait database. It is a "complete unknown" to every system component. For benchmarking, it is assigned the synthetic ID `FO.BE` and included in all 60 images. Every method's prediction on these 5 images will be scored as incorrect, making this the most extreme out-of-distribution test case in the suite.

---

## Phase 1: Dataset Loader

### 1.1 `benchmarks/dataset.py`

Do not hand-maintain benchmark manifests unless there is a clear need. The current dataset is structured well enough to derive records programmatically.

The dataset loader should:

- scan `data/raw/evaluation_images/`
- infer ground truth from the folder name
- map folder names to canonical `species_id`
- load species metadata from `data/raw/species.csv`
- compute whether each sample is in-distribution for the CNN

Suggested dataclass:

```python
@dataclass
class BenchmarkSample:
    image_path: Path
    image_bytes: bytes
    folder_name: str
    species_id: str
    english_name: str
    in_distribution: bool
    key_xml_support: str
```

Suggested loader API:

```python
class GroundTruthDataset:
    def __iter__(self) -> Iterator[BenchmarkSample]: ...
    def __len__(self) -> int: ...
    def in_distribution(self) -> List[BenchmarkSample]: ...
    def out_of_distribution(self) -> List[BenchmarkSample]: ...
    def by_species(self, species_id: str) -> List[BenchmarkSample]: ...
```

### 1.2 `benchmarks/config.py`

Single source of truth. Must contain actual mappings, not just path constants.

```python
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "evaluation_images"
SPECIES_CSV = PROJECT_ROOT / "data" / "raw" / "species.csv"
KEY_XML = PROJECT_ROOT / "data" / "raw" / "key.xml"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "benchmarks"
ORACLE_JSON = PROJECT_ROOT / "benchmarks" / "oracle_answers.json"

FOLDER_TO_SPECIES_ID: Dict[str, str] = {
    "AM.MU": "AM.MU",
    "AM.VI": "AM.VI",
    "BO.ED": "BO.ED",
    "Brunsopp": "BO.BA",
    "CA.CI": "CA.CI",
    "CR.CO": "CR.CO",
    "HY.PS": "HY.PS",
    "coprinus_comatus": "CO.CO",
    "fomitopsis_betulina": "FO.BE",
    "lycoperdon_utriforme": "LY.PE",
    "ramaria_botrytis": "RA.BO",
    "sparassis_crispa": "SP.CR",
}

CNN_NAME_TO_SPECIES_ID: Dict[str, str] = {
    "Fly Agaric": "AM.MU",
    "Chanterelle": "CA.CI",
    "False Chanterelle": "HY.PS",
    "Porcini": "BO.ED",
    "Other Boletus": "BO.BA",
    "Amanita virosa": "AM.VI",
    "Black Trumpet": "CR.CO",
}

IN_DISTRIBUTION_SPECIES: List[str] = [
    "AM.MU", "AM.VI", "BO.ED", "BO.BA", "CA.CI", "CR.CO", "HY.PS"
]
```

### 1.3 `benchmarks/oracle_answers.json`

This is the one benchmark metadata file worth maintaining manually.

It should store the correct answer path for supported `key.xml` species:

```json
{
  "CA.CI": {
    "Hur ser svampen ut?": "Undersidan har åsar eller ådror",
    "Vilken färg har svampen?": "Hela svampen är gul"
  },
  "BO.ED": {
    "Hur ser svampen ut?": "Undersidan har rör",
    "Vilken färg har hatten?": "Brun",
    "Vad stämmer bäst angående utseende?": "Kraftig fot med vitt ådernät"
  }
}
```

Its only purpose is to support `tree_oracle`.

**How to build it:** Open `data/raw/key.xml`, trace each supported species from root to leaf, and record every `(question, answer)` pair. Use the script below to print the tree structure:

```python
from models.key_tree_traversal import parse_key_xml

def print_tree(node, depth=0):
    indent = "  " * depth
    if hasattr(node, 'question'):
        print(f"{indent}Q: {node.question}")
        for child in node.conditions:
            print(f"{indent}  A: {child.answer}")
            for c in child.children:
                print_tree(c, depth + 2)
    elif hasattr(node, 'species'):
        print(f"{indent}→ {node.species}")

tree = parse_key_xml("data/raw/key.xml")
print_tree(tree)
```

---

## Phase 2: Shared Runner API

### 2.1 `benchmarks/runners/base.py`

Use one result format for every method.

```python
@dataclass
class RunnerResult:
    method_name: str
    predictions: List[Tuple[str, float]]
    coverage: bool = True
    inference_time_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

Recommended metadata examples:

- tree path
- stuck question
- auto-answered question count
- extracted traits used
- trait-db score breakdown
- LLM reasoning text
- aggregator agreement information

---

## Phase 3: Standalone Method Runners

### 3.1 `benchmarks/runners/cnn_runner.py`

**Goal:** Benchmark `MushroomCNN` alone.

**Required behavior:**

- call `MushroomCNN.predict(sample.image_bytes)`
- map English names to `species_id`
- sort descending by confidence
- keep top 5 predictions

**Expected output:**

```python
[
    ("CA.CI", 0.72),
    ("HY.PS", 0.18),
    ("CR.CO", 0.05),
    ("BO.ED", 0.03),
    ("AM.VI", 0.02),
]
```

This runner should explicitly preserve the top-5 output because that is part of your thesis intent.

### 3.2 `benchmarks/runners/tree_runner.py`

**Goal:** Benchmark `KeyTreeEngine`.

This runner should support two modes.

#### `tree_auto`

Use only what the current code can infer automatically:

1. Run `visual_trait_extractor.extract(sample.image_bytes)`
2. Pass `visible_traits` to `KeyTreeEngine.start_session(...)`
3. Let the engine auto-answer as far as it can
4. If a conclusion is reached, record the species
5. If it gets stuck at a question, record `coverage=False`

This is the realistic tree benchmark.

#### `tree_oracle`

Use stored perfect answers from `oracle_answers.json`:

1. Start exactly as in `tree_auto`
2. Whenever the engine asks a question, feed the correct stored answer
3. Continue until conclusion or unsupported path

This is not for simulating runtime behavior. It measures the theoretical discriminative power of the species tree.

**Why this matters for the thesis:**

`tree_oracle_accuracy - tree_auto_accuracy`

is a meaningful measure of how much performance is lost because trait extraction cannot answer enough tree questions.

#### Implementation

```python
import json
import time
from typing import Dict, Optional
from pathlib import Path

from benchmarks.config import KEY_XML, ORACLE_JSON
from benchmarks.runners.base import BenchmarkRunner, RunnerResult
from models.key_tree_traversal import KeyTreeEngine
from models.visual_trait_extractor import extract

class TreeRunner(BenchmarkRunner):
    name = "tree"
    
    def __init__(self, mode: str = "auto"):
        self.engine = KeyTreeEngine(str(KEY_XML))
        self.mode = mode
        self.oracle: Dict[str, Dict[str, str]] = {}
        if mode == "oracle" and Path(ORACLE_JSON).exists():
            with open(ORACLE_JSON, encoding="utf-8") as f:
                self.oracle = json.load(f)
    
    def predict(self, sample) -> RunnerResult:
        t0 = time.perf_counter()
        
        # Step 1: extract traits
        step1_result = extract(sample.image_bytes)
        visible_traits = step1_result["visible_traits"]
        
        # Step 2: start tree session
        result = self.engine.start_session(None, visible_traits)
        
        # Step 3: auto mode stops at first question; oracle mode feeds answers
        while result.get("status") == "question":
            if self.mode == "oracle":
                species_answers = self.oracle.get(sample.species_id, {})
                answer = species_answers.get(result["question"])
                if answer is None:
                    break  # no oracle entry for this question
                result = self.engine.answer(result["session_id"], answer)
            else:
                break  # auto mode: stop here
        
        elapsed = (time.perf_counter() - t0) * 1000
        
        if result.get("status") == "conclusion":
            swedish_name = result["species"]
            species_id = self._resolve_swedish_name(swedish_name)
            return RunnerResult(
                method_name=f"tree_{self.mode}",
                predictions=[(species_id, 1.0)],
                coverage=True,
                inference_time_ms=elapsed,
                metadata={
                    "swedish_name": swedish_name,
                    "auto_answered": len(result.get("auto_answered", [])),
                    "path": result.get("path", []),
                }
            )
        
        return RunnerResult(
            method_name=f"tree_{self.mode}",
            predictions=[],
            coverage=False,
            inference_time_ms=elapsed,
            metadata={"stuck_at_question": result.get("question")},
        )
    
    def _resolve_swedish_name(self, swedish_name: str) -> str:
        """Map Swedish tree output to species_id.
        
        Use _KEY_XML_ALIASES from trait_database_comparator or build your own.
        For unsupported species, return the raw string and handle in metrics.
        """
        from models.trait_database_comparator import _KEY_XML_ALIASES
        # _KEY_XML_ALIASES maps lowercase Swedish name -> species_id
        alias = _KEY_XML_ALIASES.get(swedish_name.lower().strip())
        if alias:
            return alias
        # Fallback: search species.csv for fuzzy match
        # ... implement or return swedish_name as-is
        return swedish_name
```

### 3.3 `benchmarks/runners/trait_db_runner.py`

**Goal:** Benchmark `TraitDatabaseComparator` as a standalone ranked method.

Required flow:

1. Run `visual_trait_extractor.extract(sample.image_bytes)`
2. Get `visible_traits`
3. Compare those traits against all species in the database
4. Return a ranked candidate list

You need to add this method to `models/trait_database_comparator.py`:

```python
def rank_all_species(self, visible_traits: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Compare visible traits against ALL species and return ranked list."""
    results = []
    for species in self._species:
        db_traits = _load_traits(species["species_id"], self._trait_rows)
        match = _compare_visible_to_db(visible_traits, db_traits)
        results.append({
            "species_id": species["species_id"],
            "swedish_name": species["swedish_name"],
            "english_name": species["english_name"],
            "score": match["score"],
        })
    return sorted(results, key=lambda x: x["score"], reverse=True)
```

**Note:** `_load_traits()` and `_compare_visible_to_db()` are module-private functions in `trait_database_comparator.py`. They are callable from within the same module because `rank_all_species()` will be added to the `TraitDatabaseComparator` class.

**Runner implementation:**

```python
class TraitDBRunner(BenchmarkRunner):
    name = "trait_db"
    
    def __init__(self):
        from models.trait_database_comparator import TraitDatabaseComparator
        self.comparator = TraitDatabaseComparator(str(DATA_RAW_DIR))
    
    def predict(self, sample) -> RunnerResult:
        from models.visual_trait_extractor import extract
        
        t0 = time.perf_counter()
        visible_traits = extract(sample.image_bytes)["visible_traits"]
        ranked = self.comparator.rank_all_species(visible_traits)
        elapsed = (time.perf_counter() - t0) * 1000
        
        predictions = [(r["species_id"], r["score"]) for r in ranked]
        return RunnerResult(
            method_name="trait_db",
            predictions=predictions,
            coverage=True,
            inference_time_ms=elapsed,
            metadata={"visible_traits": visible_traits},
        )
```

### 3.4 `benchmarks/runners/llm_runner.py`

**Goal:** Benchmark `LLMClassifier` as the fourth standalone method.

This is a separate method, not part of the Step 1 → 2 → 3 pipeline.

**How to call the LLM classifier:**

`LLMClassifier.classify()` expects a **text observation**, not image bytes. The runner must:

1. Extract visual traits via `visual_trait_extractor.extract(image_bytes)`
2. Format traits into a text description
3. Call `LLMClassifier.classify(observation=text, context=...)`
4. Map the returned species name to `species_id`

This text conversion is a **benchmark adapter**, not an existing production interface. Document it clearly in the thesis methodology so it is obvious that:

- the repository's LLM module is text-first
- the benchmark constructs the text observation from extracted visual traits
- LLM results therefore depend partly on the trait-extractor output quality

**Recommended text formatting:**

```python
def _traits_to_observation(visible_traits: dict) -> str:
    return (
        f"Mushroom observation:\n"
        f"- Cap colour: {visible_traits['dominant_color']}"
        f" (secondary: {visible_traits['secondary_color']})\n"
        f"- Cap shape: {visible_traits['cap_shape']}\n"
        f"- Surface texture: {visible_traits['surface_texture']}\n"
        f"- Has ridges: {visible_traits['has_ridges']}\n"
        f"- Brightness: {visible_traits['brightness']}\n"
    )
```

If the chosen backend is unavailable, the runner should:

- fall back cleanly if possible
- otherwise return `coverage=False` with an error message

---

## Phase 4: Multimodal Runners

### 4.1 `benchmarks/runners/multimodal_runner.py`

The repository has two multimodal systems and both should be benchmarked.

#### `multimodal_final`

Wrap `FinalAggregator`.

This is the hierarchical aggregator used in the existing system.

#### `multimodal_weighted`
#### `multimodal_geometric`
#### `multimodal_voting`

Wrap `HybridClassifier` with:

- weighted average
- geometric mean
- voting

### 4.2 Why benchmark both?

This gives you a stronger thesis comparison:

- does hierarchical trust weighting outperform democratic aggregation?
- are there trade-offs in robustness, agreement, or OOD behavior?

### 4.3 Multimodal inputs

The multimodal runner should build its inputs from the actual standalone outputs:

- CNN output
- tree output
- trait DB output
- LLM output where relevant to the target aggregator

Important:

- follow the real expected dictionary / object shapes of `FinalAggregator` and `HybridClassifier`
- do not invent a new multimodal orchestration API

#### Exact dict shapes for `FinalAggregator`

`FinalAggregator.aggregate(step1, step2, step3)` expects:

```python
step1 = {
    "trait_extraction": {
        "ml_prediction": {
            "top_species": "Chanterelle",          # English name
            "confidence": 0.85,
            "top_k": [
                {"species": "Chanterelle", "confidence": 0.85},
                {"species": "False Chanterelle", "confidence": 0.12},
                ...
            ],
            "reasoning": "...",
            "method": "cnn",
        },
        "visible_traits": { ... },                   # full visible_traits dict
    }
}

step2 = {
    "status": "conclusion",                          # or "question"
    "species": "Kantarell",                          # Swedish name
    "edibility": "*",
    "edibility_label": "edible",
    "path": ["Undersidan har åsar eller ådror", "Hela svampen är gul"],
    "auto_answered": [...],
    "tree_compatibility": {...},
}

step3 = {
    "status": "ok",
    "candidate": {
        "species_id": "CA.CI",
        "swedish_name": "Kantarell",
        "english_name": "Chanterelle",
        "scientific_name": "Cantharellus cibarius",
        "edible": "TRUE",
        "toxicity_level": "SAFE",
    },
    "trait_match": {
        "score": 0.92,
        "matched": [...],
        "conflicts": [...],
    },
    "lookalikes": [...],
    "safety_alert": False,
}
```

Build these dicts from the standalone runner outputs before calling `aggregate()`.

**Multimodal runner implementation sketch:**

```python
class MultimodalRunner(BenchmarkRunner):
    name = "multimodal"
    
    def __init__(self, strategy: str = "final_aggregator"):
        self.strategy = strategy
        self.cnn_runner = CNNRunner()
        self.tree_runner = TreeRunner(mode="oracle")
        self.trait_db_runner = TraitDBRunner()
        self.llm_runner = LLMRunner()
        
        if strategy == "final_aggregator":
            from models.final_aggregator import FinalAggregator
            self.aggregator = FinalAggregator(str(SPECIES_CSV))
        else:
            from models.hybrid_classifier import HybridClassifier, AggregationMethod
            method = AggregationMethod(strategy)
            self.aggregator = HybridClassifier(aggregation_method=method)
    
    def predict(self, sample) -> RunnerResult:
        # Run standalone methods
        cnn_res = self.cnn_runner.predict(sample)
        tree_res = self.tree_runner.predict(sample)
        trait_res = self.trait_db_runner.predict(sample)
        llm_res = self.llm_runner.predict(sample)
        
        if self.strategy == "final_aggregator":
            # Build Step 1/2/3 dicts as FinalAggregator expects
            step1 = self._build_step1(cnn_res, sample)
            step2 = self._build_step2(tree_res)
            step3 = self._build_step3(tree_res, trait_res)
            final = self.aggregator.aggregate(step1, step2, step3)
            
            rec = final["final_recommendation"]
            top_id = rec["species_id"]
            conf = rec["overall_confidence"]
            agreement = final.get("method_agreement", "unknown")
        else:
            # HybridClassifier path
            from models.hybrid_classifier import MethodPrediction
            image_prediction = None
            trait_prediction = None
            llm_prediction = None

            if cnn_res.coverage:
                image_prediction = MethodPrediction(
                    method="image",
                    species=cnn_res.top_species,
                    confidence=cnn_res.top_confidence,
                    top_k=cnn_res.predictions[:5],
                )
            if trait_res.coverage:
                trait_prediction = MethodPrediction(
                    method="trait",
                    species=trait_res.top_species,
                    confidence=trait_res.top_confidence,
                    top_k=trait_res.predictions[:5],
                )
            if llm_res.coverage:
                llm_prediction = MethodPrediction(
                    method="llm",
                    species=llm_res.top_species,
                    confidence=llm_res.top_confidence,
                    top_k=llm_res.predictions[:5],
                )
            
            result = self.aggregator.classify(
                image_prediction=image_prediction,
                trait_prediction=trait_prediction,
                llm_prediction=llm_prediction,
            )
            top_id = result.top_species  # may need name→ID mapping
            conf = result.top_confidence
            agreement = "full" if result.consensus_strength == 1.0 else "partial"
        
        return RunnerResult(
            method_name=f"multimodal_{self.strategy}",
            predictions=[(top_id, conf)],
            coverage=True,
            metadata={"method_agreement": agreement},
        )
    
    def _build_step1(self, cnn_res, sample):
        from models.visual_trait_extractor import extract
        step1 = extract(sample.image_bytes)
        # FinalAggregator expects English CNN species names here, not species_id.
        top_species_english = self._species_id_to_cnn_name(cnn_res.top_species)
        step1["trait_extraction"]["ml_prediction"] = {
            "top_species": top_species_english,
            "confidence": cnn_res.top_confidence,
            "top_k": [
                {"species": self._species_id_to_cnn_name(sp), "confidence": conf}
                for sp, conf in cnn_res.predictions[:5]
            ],
            "reasoning": "CNN benchmark prediction",
            "method": "cnn",
        }
        return step1
    
    def _build_step2(self, tree_res):
        if not tree_res.coverage:
            return {"status": "question", "species": ""}
        return {
            "status": "conclusion",
            # FinalAggregator expects the Step 2 tree output species in Swedish.
            "species": tree_res.metadata.get("swedish_name", tree_res.top_species),
            "path": tree_res.metadata.get("path", []),
            "auto_answered": tree_res.metadata.get("auto_answered", []),
        }
    
    def _build_step3(self, tree_res, trait_res):
        # FinalAggregator Step 3 is not "top ranked trait species".
        # It is the database comparison result for the tree candidate.
        if not tree_res.coverage:
            return {"status": "species_not_found"}
        if not trait_res.coverage:
            return {"status": "species_not_found"}
        
        from models.trait_database_comparator import TraitDatabaseComparator
        comparator = TraitDatabaseComparator(str(DATA_RAW_DIR))
        visible_traits = trait_res.metadata["visible_traits"]
        swedish_name = tree_res.metadata.get("swedish_name", tree_res.top_species)
        return comparator.compare(swedish_name, visible_traits)

    def _species_id_to_cnn_name(self, species_id: str) -> str:
        reverse = {
            "AM.MU": "Fly Agaric",
            "CA.CI": "Chanterelle",
            "HY.PS": "False Chanterelle",
            "BO.ED": "Porcini",
            "BO.BA": "Other Boletus",
            "AM.VI": "Amanita virosa",
            "CR.CO": "Black Trumpet",
        }
        return reverse.get(species_id, species_id)
```

#### Exact object wiring for `HybridClassifier`

`HybridClassifier.classify(...)` should be called with the real `MethodPrediction` objects it expects.

Recommended wiring:

```python
from models.hybrid_classifier import MethodPrediction

image_prediction = None
trait_prediction = None
llm_prediction = None

if cnn_res.coverage and cnn_res.predictions:
    image_prediction = MethodPrediction(
        method="image",
        species=cnn_res.top_species,
        confidence=cnn_res.top_confidence,
        top_k=cnn_res.predictions[:5],
    )

if trait_res.coverage and trait_res.predictions:
    trait_prediction = MethodPrediction(
        method="trait",
        species=trait_res.top_species,
        confidence=trait_res.top_confidence,
        top_k=trait_res.predictions[:5],
    )

if llm_res.coverage and llm_res.predictions:
    llm_prediction = MethodPrediction(
        method="llm",
        species=llm_res.top_species,
        confidence=llm_res.top_confidence,
        top_k=llm_res.predictions[:5],
    )

result = hybrid_classifier.classify(
    image_prediction=image_prediction,
    trait_prediction=trait_prediction,
    llm_prediction=llm_prediction,
)
```

Important:

- verify the exact `MethodPrediction` field names against `models/hybrid_classifier.py`
- confirm whether `species` must be English name or canonical `species_id`
- document any name conversion before calling `HybridClassifier`
- do not pass tree results directly unless the current implementation explicitly supports that

This matters because `HybridClassifier` is not just a generic bag-of-methods interface. Its object contract needs to match the current implementation exactly.

---

## Phase 5: Metrics

### 5.1 Shared metrics

Implement:

```python
from typing import Dict, List, Tuple
from benchmarks.runners.base import RunnerResult

def compute_top_k_accuracy(
    results: List[RunnerResult],
    ground_truth: List[str],
    k: int = 1,
) -> float:
    """Fraction where ground truth is in top-k predictions."""
    correct = 0
    total = 0
    for res, gt in zip(results, ground_truth):
        if not res.coverage:
            continue
        total += 1
        top_k = [sp for sp, _ in res.predictions[:k]]
        if gt in top_k:
            correct += 1
    return correct / total if total > 0 else 0.0

def compute_coverage(results: List[RunnerResult]) -> float:
    """Fraction of samples where coverage=True."""
    return sum(1 for r in results if r.coverage) / len(results) if results else 0.0

def compute_macro_f1(
    results: List[RunnerResult],
    ground_truth: List[str],
    all_labels: List[str],
) -> float:
    """Macro-averaged F1 across all species."""
    from sklearn.metrics import f1_score
    y_true = ground_truth
    y_pred = [r.top_species if r.coverage else "UNKNOWN" for r in results]
    return f1_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0)

def compute_ece(
    results: List[RunnerResult],
    ground_truth: List[str],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error. Only meaningful for CNN."""
    confidences = []
    accuracies = []
    for res, gt in zip(results, ground_truth):
        if res.coverage and res.predictions:
            confidences.append(res.top_confidence)
            accuracies.append(1.0 if res.top_species == gt else 0.0)
    
    if not confidences:
        return 0.0
    
    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        bin_accs = [a for c, a in zip(confidences, accuracies) if lo <= c < hi]
        if bin_accs:
            avg_conf = sum(c for c in confidences if lo <= c < hi) / len(bin_accs)
            avg_acc = sum(bin_accs) / len(bin_accs)
            ece += abs(avg_acc - avg_conf) * (len(bin_accs) / len(confidences))
    return ece

def compute_ood_metrics(
    id_results: List[RunnerResult],
    ood_results: List[RunnerResult],
) -> Dict[str, float]:
    """Compare in-distribution vs out-of-distribution behavior."""
    id_conf = [r.top_confidence for r in id_results if r.coverage]
    ood_conf = [r.top_confidence for r in ood_results if r.coverage]
    return {
        "id_avg_confidence": sum(id_conf) / len(id_conf) if id_conf else 0.0,
        "ood_avg_confidence": sum(ood_conf) / len(ood_conf) if ood_conf else 0.0,
        "confidence_gap": (sum(id_conf) / len(id_conf) if id_conf else 0.0)
                        - (sum(ood_conf) / len(ood_conf) if ood_conf else 0.0),
    }

def compute_pairwise_agreement(
    all_results: Dict[str, List[RunnerResult]],
) -> Dict[str, Dict[str, float]]:
    """For each image, check if method A and method B agree on top prediction."""
    methods = list(all_results.keys())
    agreement = {m1: {m2: 0.0 for m2 in methods} for m1 in methods}
    n = len(next(iter(all_results.values())))
    
    for i in range(n):
        for m1 in methods:
            for m2 in methods:
                r1 = all_results[m1][i]
                r2 = all_results[m2][i]
                if r1.coverage and r2.coverage and r1.top_species == r2.top_species:
                    agreement[m1][m2] += 1 / n
    return agreement
```

### 5.2 CNN-specific metrics

For `cnn` measure:

- top-1 accuracy
- top-3 accuracy
- top-5 accuracy
- average top-1 confidence
- ID accuracy
- OOD behavior

### 5.3 Tree-specific metrics

For `tree_auto` and `tree_oracle` measure:

- conclusion rate
- top-1 accuracy
- average traversal depth
- frequency of getting stuck
- oracle vs auto accuracy gap

That oracle-vs-auto gap is one of the strongest thesis results available in this project.

### 5.4 Trait DB metrics

For `trait_db` measure:

- top-1 accuracy
- top-3 accuracy
- top-5 accuracy
- average rank of the true species

### 5.5 LLM metrics

For `llm` measure:

- top-1 accuracy
- top-3 accuracy if available
- coverage
- inference time

### 5.6 Multimodal metrics

For multimodal aggregators measure:

- top-1 accuracy
- top-3 accuracy if available
- agreement with standalone methods
- improvement over best standalone method
- `FinalAggregator` vs `HybridClassifier` comparison under the same dataset split

### 5.7 OOD metrics

Add explicit OOD analysis because the dataset naturally supports it.

Recommended functions:

```python
compute_ood_metrics(in_dist_results, ood_results)
compute_confidence_shift(in_dist_results, ood_results)
```

Recommended OOD outputs:

- average confidence on ID vs OOD
- false positive behavior on OOD
- confidence histogram on OOD

### 5.8 Calibration note

Only use calibration metrics where they make sense.

- CNN probabilities are suitable for ECE-style analysis.
- tree confidence is not naturally probabilistic
- trait-db scores are match scores, not calibrated probabilities
- LLM self-reported confidence should be treated cautiously

So ECE should be:

- primarily reported for CNN
- optional and clearly labeled if explored elsewhere

---

## Phase 6: Reports and Visualization

### 6.1 `benchmarks/reports.py`

Generate:

- full JSON report
- one-row-per-image CSV
- Markdown benchmark summary

Recommended CSV columns:

```text
image_path
ground_truth_species_id
cnn_top1
cnn_top5_json
tree_auto_top1
tree_auto_coverage
tree_oracle_top1
trait_db_top1
trait_db_top5_json
llm_top1
multimodal_final_top1
multimodal_weighted_top1
multimodal_geometric_top1
multimodal_voting_top1
```

### 6.2 `benchmarks/visualize.py`

Recommended plots:

- top-1 / top-3 / top-5 accuracy by method
- CNN ID vs OOD confidence histogram
- tree auto vs oracle accuracy comparison
- per-species grouped accuracy chart
- average true-species rank for trait DB
- pairwise agreement heatmap

These plots are directly relevant to the thesis and should be prioritized over decorative charts.

#### Reports implementation sketch

```python
import json
import csv
from pathlib import Path
from typing import Dict, List

from benchmarks.runners.base import RunnerResult

def generate_json_report(
    metrics: Dict,
    all_results: Dict[str, List[RunnerResult]],
    ground_truth: List[str],
    output_path: Path,
):
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_images": len(ground_truth),
            "methods": list(all_results.keys()),
        },
        "metrics": metrics,
        "per_image": [],
    }
    for i, gt in enumerate(ground_truth):
        entry = {"ground_truth": gt}
        for method, results in all_results.items():
            r = results[i]
            entry[method] = {
                "top1": r.top_species if r.coverage else None,
                "confidence": r.top_confidence if r.coverage else None,
                "correct": r.top_species == gt if r.coverage else False,
                "coverage": r.coverage,
                "time_ms": r.inference_time_ms,
            }
        report["per_image"].append(entry)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

def generate_csv_report(
    all_results: Dict[str, List[RunnerResult]],
    ground_truth: List[str],
    samples,
    output_path: Path,
):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["image_path", "ground_truth"]
        for method in all_results:
            header.extend([f"{method}_top1", f"{method}_correct", f"{method}_coverage"])
        writer.writerow(header)
        
        for i, (sample, gt) in enumerate(zip(samples, ground_truth)):
            row = [str(sample.image_path), gt]
            for method, results in all_results.items():
                r = results[i]
                row.extend([
                    r.top_species if r.coverage else "N/A",
                    "1" if (r.coverage and r.top_species == gt) else "0",
                    "1" if r.coverage else "0",
                ])
            writer.writerow(row)

def generate_markdown_report(metrics: Dict, output_path: Path):
    lines = ["# Benchmark Results\n"]
    
    # Summary table
    lines.append("## Accuracy Summary\n")
    lines.append("| Method | Top-1 | Top-3 | Coverage | Mean Time (ms) |")
    lines.append("|--------|-------|-------|----------|----------------|")
    for method, m in metrics.items():
        if method == "agreement":
            continue
        lines.append(
            f"| {method} | {m.get('top1_accuracy', 0):.3f} | "
            f"{m.get('top3_accuracy', 0):.3f} | "
            f"{m.get('coverage', 0):.3f} | "
            f"{m.get('mean_time_ms', 0):.1f} |"
        )
    
    # OOD section
    if "cnn" in metrics and "ood" in metrics["cnn"]:
        lines.append("\n## OOD Analysis (CNN)\n")
        ood = metrics["cnn"]["ood"]
        lines.append(f"- ID avg confidence: {ood['id_avg_confidence']:.3f}")
        lines.append(f"- OOD avg confidence: {ood['ood_avg_confidence']:.3f}")
        lines.append(f"- Confidence gap: {ood['confidence_gap']:.3f}")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
```

#### Visualization implementation sketch

```python
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.runners.base import RunnerResult

def plot_accuracy_comparison(metrics: Dict, output_path: Path):
    methods = [k for k in metrics if k != "agreement"]
    top1 = [metrics[m].get("top1_accuracy", 0) for m in methods]
    top3 = [metrics[m].get("top3_accuracy", 0) for m in methods]
    
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 0.2, top1, 0.4, label="Top-1")
    ax.bar(x + 0.2, top3, 0.4, label="Top-3")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Method")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_confidence_histogram(
    id_results: List[RunnerResult],
    ood_results: List[RunnerResult],
    output_path: Path,
):
    id_conf = [r.top_confidence for r in id_results if r.coverage]
    ood_conf = [r.top_confidence for r in ood_results if r.coverage]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(id_conf, bins=10, alpha=0.6, label="In-distribution", color="green")
    ax.hist(ood_conf, bins=10, alpha=0.6, label="Out-of-distribution", color="red")
    ax.set_xlabel("Top-1 Confidence")
    ax.set_ylabel("Count")
    ax.set_title("CNN Confidence: ID vs OOD")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_agreement_heatmap(agreement: Dict, output_path: Path):
    methods = list(agreement.keys())
    matrix = [[agreement[m1][m2] for m2 in methods] for m1 in methods]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(methods)))
    ax.set_yticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_yticklabels(methods)
    ax.set_title("Pairwise Method Agreement")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def generate_all_plots(metrics, all_results, ground_truth, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_accuracy_comparison(metrics, output_dir / "accuracy_comparison.png")
    plot_agreement_heatmap(metrics["agreement"], output_dir / "agreement_heatmap.png")
    if "cnn" in all_results:
        id_res = [r for r, gt in zip(all_results["cnn"], ground_truth) 
                  if gt in IN_DISTRIBUTION_SPECIES]
        ood_res = [r for r, gt in zip(all_results["cnn"], ground_truth) 
                   if gt not in IN_DISTRIBUTION_SPECIES]
        plot_confidence_histogram(id_res, ood_res, output_dir / "confidence_histogram.png")
```

---

## Phase 7: CLI Entry Point

### 7.1 `benchmarks/run_benchmark.py`

Methods to support:

```text
cnn
tree_auto
tree_oracle
trait_db
llm
multimodal_final
multimodal_weighted
multimodal_geometric
multimodal_voting
all
```

Suggested CLI:

```bash
python -m benchmarks.run_benchmark --methods all
python -m benchmarks.run_benchmark --methods cnn,tree_auto,trait_db
python -m benchmarks.run_benchmark --methods multimodal_final,multimodal_weighted
```

Useful flags:

- `--methods`
- `--output-dir`
- `--format`
- `--limit`
- `--backend`

#### CLI implementation sketch

```python
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from benchmarks.config import OUTPUT_DIR
from benchmarks.dataset import GroundTruthDataset
from benchmarks.metrics import *
from benchmarks.reports import *
from benchmarks.visualize import *
from benchmarks.runners.cnn_runner import CNNRunner
from benchmarks.runners.tree_runner import TreeRunner
from benchmarks.runners.trait_db_runner import TraitDBRunner
from benchmarks.runners.llm_runner import LLMRunner
from benchmarks.runners.multimodal_runner import MultimodalRunner

ALL_METHODS = [
    "cnn", "tree_auto", "tree_oracle", "trait_db", "llm",
    "multimodal_final", "multimodal_weighted",
    "multimodal_geometric", "multimodal_voting",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="all")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--format", default="json,csv,md")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    
    methods = ALL_METHODS if args.methods == "all" else args.methods.split(",")
    dataset = GroundTruthDataset()
    samples = list(dataset)[:args.limit or None]
    
    runners = {}
    if "cnn" in methods: runners["cnn"] = CNNRunner()
    if "tree_auto" in methods: runners["tree_auto"] = TreeRunner(mode="auto")
    if "tree_oracle" in methods: runners["tree_oracle"] = TreeRunner(mode="oracle")
    if "trait_db" in methods: runners["trait_db"] = TraitDBRunner()
    if "llm" in methods: runners["llm"] = LLMRunner()
    for s in ["final", "weighted", "geometric", "voting"]:
        k = f"multimodal_{s}"
        if k in methods: runners[k] = MultimodalRunner(strategy=s)
    
    all_results = {name: [] for name in runners}
    ground_truth = []
    for sample in samples:
        ground_truth.append(sample.species_id)
        for name, runner in runners.items():
            all_results[name].append(runner.predict(sample))
    
    metrics = {}
    for name, results in all_results.items():
        metrics[name] = {
            "top1_accuracy": compute_top_k_accuracy(results, ground_truth, k=1),
            "top3_accuracy": compute_top_k_accuracy(results, ground_truth, k=3),
            "coverage": compute_coverage(results),
            "macro_f1": compute_macro_f1(results, ground_truth, list(set(ground_truth))),
            "mean_time_ms": compute_mean_inference_time(results),
        }
        if name == "cnn":
            metrics[name]["ece"] = compute_ece(results, ground_truth)
    
    if "cnn" in runners:
        id_res = [r for r, s in zip(all_results["cnn"], samples) if s.in_distribution]
        ood_res = [r for r, s in zip(all_results["cnn"], samples) if not s.in_distribution]
        id_gt = [s.species_id for s in samples if s.in_distribution]
        ood_gt = [s.species_id for s in samples if not s.in_distribution]
        metrics["cnn"]["id_top1"] = compute_top_k_accuracy(id_res, id_gt, k=1)
        metrics["cnn"]["ood_top1"] = compute_top_k_accuracy(ood_res, ood_gt, k=1)
        metrics["cnn"]["ood"] = compute_ood_metrics(id_res, ood_res)
    
    metrics["agreement"] = compute_pairwise_agreement(all_results)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output_dir / ts
    out.mkdir(parents=True, exist_ok=True)
    
    fmt = args.format.split(",")
    if "json" in fmt: generate_json_report(metrics, all_results, ground_truth, out / "results.json")
    if "csv" in fmt: generate_csv_report(all_results, ground_truth, samples, out / "per_image.csv")
    if "md" in fmt: generate_markdown_report(metrics, out / "report.md")
    if "plots" in fmt: generate_all_plots(metrics, all_results, ground_truth, out / "plots")
    
    print(f"Benchmark complete. Results saved to: {out}")

if __name__ == "__main__":
    main()
```

---

## Phase 8: Testing Checklist

Before running the full suite:

- [ ] dataset loader finds exactly 60 images
- [ ] dataset loader splits exactly 35 ID and 25 OOD samples
- [ ] CNN runner returns mapped top-5 predictions
- [ ] tree auto returns conclusions for at least some supported cases
- [ ] tree oracle resolves supported species correctly
- [ ] trait DB returns ranked lists across all species
- [ ] LLM runner works with available backend or fails gracefully
- [ ] multimodal final produces a valid result
- [ ] multimodal weighted / geometric / voting produce valid results
- [ ] reports generate on a 5-image subset
- [ ] plots render on a 5-image subset

---

## Key Implementation Pitfalls

1. **Do not rename deterministic components as LLM components.**  
   `KeyTreeEngine`, `TraitDatabaseComparator`, and `FinalAggregator` are not LLM modules in the current repo.

2. **Do not drop `tree_oracle`.**  
   It is one of the most scientifically valuable comparisons in the thesis.

3. **Do not hand-maintain unnecessary JSON manifests.**  
   The folder structure already contains reliable ground truth.

4. **Keep name resolution centralized.**  
   English CNN names, Swedish tree names, and database aliases all need canonical mapping.

5. **Do not invent new benchmark APIs unless absolutely needed.**  
   Adapt runners to the real model interfaces already in the codebase.

6. **Treat OOD as a first-class result.**  
   The dataset is unusually good for that analysis.

---

## Thesis Claims the Benchmark Should Support

| Claim | Evidence |
|---|---|
| CNN alone is useful but brittle on OOD species | CNN top-k accuracy and OOD confidence behavior |
| The tree has high theoretical discriminative power but limited auto-resolve performance | `tree_oracle` vs `tree_auto` |
| Trait DB adds useful structured evidence beyond image-only classification | trait-db ranking quality and average true-species rank |
| The standalone LLM is competitive but different in behavior from deterministic methods | `llm` vs `cnn` / `tree` / `trait_db` |
| Multimodal aggregation improves over standalone methods | multimodal top-1 compared with best standalone top-1 |
| Aggregation strategy matters | `FinalAggregator` vs `HybridClassifier` variants |

---

## Appendix: Files to Create vs Modify

### Create

- `benchmarks/__init__.py`
- `benchmarks/config.py`
- `benchmarks/dataset.py`
- `benchmarks/oracle_answers.json`
- `benchmarks/runners/__init__.py`
- `benchmarks/runners/base.py`
- `benchmarks/runners/cnn_runner.py`
- `benchmarks/runners/tree_runner.py`
- `benchmarks/runners/trait_db_runner.py`
- `benchmarks/runners/llm_runner.py`
- `benchmarks/runners/multimodal_runner.py`
- `benchmarks/metrics.py`
- `benchmarks/reports.py`
- `benchmarks/visualize.py`
- `benchmarks/run_benchmark.py`

### Modify

- `models/trait_database_comparator.py`
  Add `rank_all_species()`. Note: `_load_traits()` and `_compare_visible_to_db()` are module-private but callable from within the same module.

**Estimated implementation size:** roughly 1,000 lines  
**Estimated implementation time:** about 1 to 2 focused working days
