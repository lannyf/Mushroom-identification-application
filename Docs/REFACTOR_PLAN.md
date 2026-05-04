# Refactor Plan: Pure Extraction + Single Fusion Point

**Goal**: Remove early prediction from Step 1. Make `FinalAggregator` the sole fusion point. Move form traits into the key tree engine. Make LLM a standalone module.

**Date**: 2025-05-04

---

## 1. Problem Statement

The current API has **two independent fusion systems**:

| System | Step | Inputs | Output |
|--------|------|--------|--------|
| `HybridClassifier` | Step 1 (`/identify`) | CNN + form traits + LLM | Prediction A |
| `FinalAggregator` | Step 4 (`/finalize`) | CNN + key tree + trait DB | Prediction B |

Prediction A is returned to the Flutter client at Step 1, then **discarded** when the client later calls Step 4. `FinalAggregator` recomputes its own winner from scratch using a different formula and different inputs.

Additionally:
- Form traits are scored by `trait_scores()` as a separate prediction, rather than feeding the key tree as intended.
- LLM is fused early in Step 1, rather than being an optional standalone consultation.

---

## 2. Target Architecture

```
POST /identify
  Input:  image bytes
  Output: {visible_traits, ml_prediction, image_metrics}
  
POST /identify/Species_tree_traversal/start
  Input:  visible_traits + ml_hint + pre_answers (optional)
  Output: question or conclusion
  
POST /identify/Species_tree_traversal/answer
  Input:  session_id + answer
  Output: next question or conclusion

POST /identify/llm_predict  (NEW)
  Input:  visible_traits
  Output: LLM species prediction + reasoning

POST /identify/comparison/compare
  Input:  swedish_name + visible_traits
  Output: trait_match score + lookalikes + safety_alert

POST /identify/prediction/finalize
  Input:  ml_prediction + tree_traversal + comparison
  Output: FinalAggregator result (CNN + tree + trait DB only)
```

**FinalAggregator fuses exactly 3 signals:**
1. CNN prediction (from Step 1 `ml_prediction`)
2. Key tree traversal conclusion (from Step 2)
3. Trait database comparison (from Step 3)

---

## 3. Module-by-Module Changes

### 3.1 Form Traits (current: `api/scoring.py trait_scores()`)

**Status**: ✅ **DESIGN APPROVED** — Delete `trait_scores()`, use `pre_answers` in `KeyTreeEngine`.

**Rationale**: `trait_scores()` contains 15 hardcoded `if` statements with magic numbers (0.8, 0.45, 0.7) that are not derived from `key.xml` or `species.csv`. This violates the constraint of avoiding hardcoded data that drifts from system data.

**Current behaviour**: Receives Flutter form data (`color`, `cap_shape`, `gill_type`, `stem_type`, `habitat`, `season`), runs 15 hardcoded `if` statements, returns a score dict for the 7 CNN species. This score is fed into `HybridClassifier` as a parallel prediction.

**Target behaviour**: Flutter form answers feed directly into `KeyTreeEngine` as user-supplied answers to the tree's questions. The tree uses these answers to traverse toward a leaf, exactly as a human would use a paper field guide.

**Design decisions**:
1. **Frontend mapping** (chosen): Flutter sends `pre_answers` keyed by exact `key.xml` question text. No backend mapping table. Example:
   ```json
   {"pre_answers": {
     "Vilken färg har svampen?": "Hela svampen är gul",
     "Hur ser svampen ut?": "Undersidan har åsar eller ådror"
   }}
   ```
2. **Precedence**: Auto-answers from `visible_traits` are applied **first**. `pre_answers` are consulted **only** when `visible_traits` cannot provide a conclusive auto-answer (i.e., ambiguous or missing trait). User input is **complementary** — it never overrides image-derived traits. The tree uses image analysis as the primary driver; the form only resolves ambiguity.
3. **Validation**: `KeyTreeEngine` validates that each `pre_answer` is a valid option for its question. Invalid answers are ignored.

**Changes needed**:
- Delete `trait_scores()` from `api/scoring.py`
- Add `pre_answers: Optional[Dict[str, str]]` parameter to `KeyTreeEngine.start_session()`
- Modify `KeyTreeEngine._advance()` to try auto-answer from `visible_traits` first, then fall back to `pre_answers` only when ambiguous or missing
- Update `TraversalSession` dataclass to store `pre_answers`
- Update `api/schemas.py` `Step2StartRequest` to accept `pre_answers`
- Update Flutter client to send form answers to Step 2 instead of Step 1

---

### 3.2 LLM Scoring (current: `api/scoring.py ollama_scores() + llm_scores()`)

**Status**: ✅ **DESIGN APPROVED** — Standalone endpoint, hard-fail on Ollama offline, visible-traits only.

**Current behaviour**: Builds an observation text from `visible_traits` + form data, queries Ollama (or rule-based fallback), converts the result to a score dict, feeds it into `HybridClassifier`.

**Target behaviour**: Standalone module/endpoint. Takes only `visible_traits`, builds observation text, queries Ollama, returns the prediction. Not fused into anything. The Flutter client can call it if the user wants an LLM second opinion.

**Design decisions**:
1. **Hard fail** (chosen): Return `HTTP 503 Service Unavailable` when Ollama is offline. No rule-based fallback. Flutter shows "LLM unavailable."
2. **Visible traits only** (chosen): Observation text uses human-readable labels (`dominant_color`, `cap_shape`, `surface_texture`, `has_ridges`, `brightness`). No raw colour ratios.
3. **No form data**: LLM sees only image-derived traits, not Flutter questionnaire answers. Clean separation.

**Changes needed**:
- Remove `ollama_scores()` and `llm_scores()` from `api/scoring.py`
- Keep `build_observation_text()` but simplify to visible-traits only
- Create new endpoint `POST /identify/llm_predict` in `api/main.py`
- Add `LLMPredictRequest` schema in `api/schemas.py`

---

### 3.3 HybridClassifier (current: `models/hybrid_classifier.py`)

**Status**: ✅ **DESIGN APPROVED** — Delete entirely (Option A).

**Current behaviour**: Fuses `image_prediction` + `trait_prediction` + `llm_prediction` using weighted average / geometric mean / voting. Returns `HybridResult` with top species, confidence breakdown, lookalikes, safety warnings.

**Target behaviour**: **Deleted.** `FinalAggregator` is the sole fusion point. No early prediction in Step 1.

**Rationale**: `HybridClassifier` produces Prediction A in Step 1, which is then discarded when `FinalAggregator` produces Prediction B in Step 4. It serves no purpose in the target architecture and creates confusion about which fusion system is canonical.

**Files to delete**:
- `models/hybrid_classifier.py`
- `tests/test_hybrid_classifier.py`
- `scripts/test_hybrid_system.py`

**Files to clean**:
- `api/main.py` — remove `HYBRID` import and `HYBRID.classify()` call
- `api/scoring.py` — remove `build_prediction()` and `normalize()` (only used by HybridClassifier)
- `benchmarks/runners/multimodal_runner.py` — verify it uses `FinalAggregator`, not `HybridClassifier`

---

### 3.4 Step 1 `/identify` (current: `api/main.py`)

**Current behaviour**: Runs extraction, form scoring, LLM scoring, and hybrid fusion. Returns a full prediction to Flutter.

**Target behaviour**: Pure extraction only. Returns `visible_traits`, `ml_prediction`, and colour-ratio metrics. No predictions.

**Changes needed**:
- Strip `/identify` down to:
  ```python
  image_bytes = await image.read()
  step1 = extract_visual_traits(image_bytes)
  return {"trait_extraction": step1, "image_analysis": metrics}
  ```
- Remove `traits: str = Form("{}")` parameter (form data now goes to Step 2)

---

### 3.5 FinalAggregator (current: `models/final_aggregator.py`)

**Status**: ✅ **DESIGN APPROVED** — Unchanged. Already the sole fusion point.

**Current behaviour**: Fuses CNN (45%) + key tree (35%) + trait DB (20%). Agreement bonus when CNN and tree agree.

**Target behaviour**: **Unchanged functionally.** Already does exactly what we want — it is the sole fusion point for the 3 core signals.

**Design decisions**:
1. **Weights kept as-is** (chosen): Tree 45%, Image 35%, Trait DB 20%. No changes to confidence model.
2. **Agreement bonus kept as-is** (chosen): +10% when CNN and tree agree on the same species.

**Changes needed**:
- Minimal. Verify `s1_ml` extraction still works with the new simplified Step 1 output shape.

---

### 3.6 KeyTreeEngine (current: `models/key_tree_traversal.py`)

**Current behaviour**: Receives `visible_traits` + `ml_hint`. Auto-answers questions from `visible_traits`. Returns first unresolved question or conclusion.

**Target behaviour**: Also accepts `pre_answers` — user-supplied answers from the Flutter form used only when `visible_traits` cannot provide a conclusive auto-answer.

**Changes needed**:
- Add `pre_answers: Optional[Dict[str, str]]` to `start_session()`
- In `_advance()`, try auto-answer from `visible_traits` **first**
- Only if the auto-answer is ambiguous or missing, fall back to `pre_answers` for that question
- Update `TraversalSession` dataclass to store `pre_answers`

---

## 4. API Contract Changes

### 4.1 `POST /identify` — Response simplified

**Before**:
```json
{
  "trait_extraction": {...},
  "top_prediction": "Chanterelle",
  "overall_confidence": 0.87,
  "method_confidences": {...},
  "predictions": [...],
  "lookalikes": [...],
  "safety_warnings": [...],
  "image_analysis": {...}
}
```

**After**:
```json
{
  "trait_extraction": {
    "visible_traits": {...},
    "ml_prediction": {...}
  },
  "image_analysis": {
    "red_ratio": 0.12,
    "orange_red_ratio": 0.03,
    "orange_yellow_ratio": 0.45,
    "brown_ratio": 0.20,
    "white_ratio": 0.08
  }
}
```

### 4.2 `POST /identify/Species_tree_traversal/start` — Accepts pre_answers

**Before**:
```json
{"session_id": null, "visible_traits": {...}, "ml_hint": {...}}
```

**After**:
```json
{
  "session_id": null,
  "visible_traits": {...},
  "ml_hint": {...},
  "pre_answers": {
    "Hur ser svampen ut?": "Undersidan har rör",
    "Vilken färg har svampen?": "Hela svampen är brun"
  }
}
```

### 4.3 New `POST /identify/llm_predict`

**Request**:
```json
{"visible_traits": {...}}
```

**Response**:
```json
{
  "top_species": "Chanterelle",
  "confidence": 0.72,
  "reasoning": "Yellow funnel-shaped mushroom with decurrent ridges...",
  "predictions": [
    {"species": "Chanterelle", "confidence": 0.72},
    {"species": "False Chanterelle", "confidence": 0.15}
  ]
}
```

---

## 5. Test Impact

| Test file | Impact |
|-----------|--------|
| `tests/test_visual_trait_extractor.py` | None — already tests pure extraction |
| `tests/test_key_tree_traversal.py` | Add tests for `pre_answers` |
| `tests/test_trait_regression_real_images.py` | Minor — tree tests may need `pre_answers` |
| `tests/test_scoring.py` | Remove tests for `build_prediction`, `normalize`, `trait_scores`, `llm_scores` |
| `tests/test_hybrid_classifier.py` | Mark as legacy or remove if class deleted |
| `tests/test_final_aggregator.py` | None — already tests the 3-signal fusion |

---

## 6. Benchmark Impact

Benchmark runners currently use `FinalAggregator` (multimodal runner) or individual runners (CNN-only, tree-only, etc.). The benchmark suite should be unaffected because it already follows the target architecture.

If `HybridClassifier` is removed from `models/`, the `multimodal_runner.py` references to `FinalAggregator` are unaffected, but any benchmark using `HybridClassifier` directly would break.

---

## 7. Open Questions

1. **Form trait mapping**: How do Flutter form fields (`color`, `cap_shape`, `gill_type`, ...) map to `key.xml` question text? Is there a lookup table, or does Flutter send the question text directly?

2. **LLM endpoint necessity**: Is `POST /identify/llm_predict` a required endpoint, or should the LLM module simply be callable by the Flutter client without being part of the formal pipeline?

3. **HybridClassifier fate**: Keep in `models/` for benchmark backward-compat, or delete entirely?

4. **Agreement bonus in FinalAggregator**: Currently +10% when CNN and tree agree. With the new architecture, should this bonus remain, or should it check all 3 signals (CNN + tree + trait DB)?

---

## 8. Estimated Effort

| Module | Files touched | Complexity |
|--------|--------------|------------|
| Form traits → tree | `models/key_tree_traversal.py`, `api/schemas.py`, tests | Medium |
| LLM standalone | `api/main.py`, `api/schemas.py`, new module | Low |
| Strip Step 1 | `api/main.py`, `api/scoring.py` | Low |
| Remove HybridClassifier | `api/main.py`, `api/scoring.py`, tests | Low |
| FinalAggregator | Verify only | Minimal |

**Total**: ~1–2 hours of focused coding + test fixes.
