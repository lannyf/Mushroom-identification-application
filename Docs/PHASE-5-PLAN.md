# Phase 5: LLM-Based Classification Module - Initialization

## Overview
Phase 5 will implement the third identification method using large language models (LLMs). This module will:
1. Generate natural language descriptions from user observations
2. Query an LLM (OpenAI GPT or Hugging Face) with mushroom-specific prompts
3. Parse LLM responses to extract species predictions
4. Produce confidence scores compatible with Phase 3 & 4 output

## Why LLM-Based Classification?

**Complementary to Image & Trait Methods:**
- ✓ Works with natural language descriptions
- ✓ Can reason about complex interactions
- ✓ No training data needed (pre-trained models)
- ✓ Flexible input (user can describe in own words)
- ✓ Can handle ambiguous or incomplete information

**Use Cases:**
1. User reads mushroom description from field guide
2. User describes observation in natural language
3. User provides partial observations (only saw cap, not stem)

## Architecture Design

### Input Format
```python
observation = {
    'description': "Yellow mushroom with funnel-shaped cap, gills are pale...",
    'context': {
        'habitat': 'forest floor, mixed trees',
        'season': 'autumn',
        'substrate': 'soil'
    },
    'confidence': 'uncertain'  # User's self-assessed confidence
}
```

### Processing Pipeline
```
User Natural Language Input
        ↓
Prompt Engineering (mushroom-specific template)
        ↓
LLM API Call (GPT-4 / Llama)
        ↓
Response Parsing (extract species, reasoning)
        ↓
Confidence Scoring (derive from LLM confidence indicators)
        ↓
[species, probability] → Phase 6 Aggregation
```

### Output Format
```python
predictions = [
    ('Cantharellus cibarius (Chanterelle)', 0.85, 'yellow funnel, decurrent ridges'),
    ('Craterellus cinereus (Black Trumpet)', 0.65, 'similar but darker color'),
    ('Gomphus clavatus (Pig\'s Ear)', 0.45, 'brown variant possible'),
]
```

## Implementation Components (Planned)

### 1. LLMPromptTemplate
- System prompts for mushroom expertise
- Few-shot examples of observations and predictions
- Safety disclaimers and caveats
- Task-specific instructions

### 2. LLMClassifier
- API client management (OpenAI/Hugging Face)
- Prompt construction from observations
- Response parsing and validation
- Confidence extraction from LLM

### 3. ObservationParser
- Extract structured traits from natural language
- Clean and validate observations
- Handle ambiguity and uncertainty
- Fill in missing information from context

### 4. LLMEvaluator
- Test LLM outputs against known species
- Measure accuracy and confidence calibration
- Generate adversarial test cases
- Benchmark against Phase 3/4 methods

### 5. Scripts
- `train_llm_model.py` - Fine-tune or validate on test set
- `evaluate_llm_model.py` - Compare LLM methods

## API Integration Options

### Option A: OpenAI GPT (Recommended for accuracy)
```python
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": MUSHROOM_SYSTEM_PROMPT},
        {"role": "user", "content": user_observation}
    ],
    temperature=0.3,  # Lower = more deterministic
    max_tokens=500
)
```

**Pros:** Highest accuracy, best at reasoning  
**Cons:** API cost, requires key management  
**Cost:** ~$0.03-0.06 per prediction  

### Option B: Hugging Face Open-Source (Cost-effective)
```python
from transformers import pipeline

classifier = pipeline(
    "text-generation",
    model="meta-llama/Llama-2-7b-hf",
    device=0  # GPU
)

response = classifier(prompt, max_length=300)
```

**Pros:** Free, runs locally, full control  
**Cons:** Lower accuracy, requires GPU  
**Cost:** Free (compute cost only)  

### Option C: Hybrid Approach
- Use local Llama-2 for fast screening
- Use GPT-4 for borderline/complex cases
- Balance cost and accuracy

## Prompt Engineering Strategy

### System Prompt
```
You are an expert mycologist specializing in mushroom identification.
You will analyze descriptions of mushrooms and predict the most likely species.

Safety Note: This system is for educational purposes only. Never use it as the
sole basis for determining if a mushroom is safe to eat.

Available species (20 total):
1. Cantharellus cibarius (Chanterelle)
2. Amanita muscaria (Fly Agaric)
... [full list]

For each description, provide:
1. Top 3 predictions with confidence (0-1 scale)
2. Key distinguishing features observed
3. Similar species that could be confused
4. Safety warnings if toxic
5. Your overall confidence in the identification
```

### Few-Shot Examples
```
Example 1:
  Observation: "Yellow funnel-shaped mushroom, gills are pale, found in forest"
  Response: {"top": "Cantharellus cibarius", "conf": 0.92, "reason": "..."}

Example 2:
  Observation: "Small red cap with white spots, white gills, growing under birch"
  Response: {"top": "Amanita muscaria", "conf": 0.88, "warning": "TOXIC"}
```

## Integration Scenarios

### With Phase 3 (Image Recognition)
```
Scenario: User provides both photo and description

Image classifier: [Cantharellus: 0.75, Craterellus: 0.15, ...]
Text classifier:  [Cantharellus: 0.85, Gomphus: 0.12, ...]
─────────────────────────────────────────────────────────
Aggregated:       [Cantharellus: 0.80, ...]
```

### With Phase 4 (Trait-Based)
```
Scenario: User describes traits but unsure of values

Natural language: "It's yellow, gills are... maybe pale? And funnel-shaped"
     ↓
LLM normalizes: "CAP.color: yellow, GILLS.color: pale, CAP.shape: funnel"
     ↓
Trait classifier: [Cantharellus: 0.78, ...]
```

## Data Requirements

No training data needed! LLMs are pre-trained on general knowledge.

However, for validation:
- Test set: 20-50 mushroom descriptions (known species)
- Edge cases: Ambiguous descriptions, lookalikes, partial info
- Safety cases: Toxic species descriptions

## Success Criteria

✅ Can parse natural language observations  
✅ Returns top-5 predictions with confidence  
✅ Handles ambiguous input gracefully  
✅ Provides explanations for predictions  
✅ Safety warnings for toxic species  
✅ <2 second latency per prediction  
✅ Integrates with Phase 3/4 output format  
✅ Comprehensive error handling  
✅ Clear logging and monitoring  

## Expected Performance

**With GPT-4:**
- Accuracy: 80-90% (on known species)
- Speed: 2-5 seconds (API latency)
- Cost: $0.03-0.06 per prediction
- Reasoning quality: High (explains decisions)

**With Llama-2 (local):**
- Accuracy: 60-75% (lower than GPT-4)
- Speed: 500ms-2s (GPU dependent)
- Cost: Free (compute only)
- Reasoning quality: Moderate

## Risk Mitigation

**Hallucination Risk:** LLM might "invent" facts
- Mitigate: Use system prompt to require species from known list
- Mitigate: Validate predictions against training data
- Mitigate: Add confidence threshold

**Adversarial Input:** User provides misleading descriptions
- Mitigate: Ask clarifying questions
- Mitigate: Flag low-confidence predictions
- Mitigate: Suggest more observations

**API Reliability:** Service downtime
- Mitigate: Cache common predictions
- Mitigate: Fallback to Phase 3/4 methods
- Mitigate: Implement timeout handling

## File Structure (To Create)

```
models/
├── llm_classifier.py           (LLMClassifier, PromptTemplate classes)
├── observation_parser.py       (ObservationParser, TextNormalizer classes)
└── llm_config.py              (Configuration, API settings)

scripts/
├── train_llm_model.py         (Validation and testing)
└── evaluate_llm_model.py      (Comprehensive evaluation)

Docs/
└── 08-llm-classification.md   (Architecture and details)
```

## Next Actions

1. **Choose LLM backend** (GPT-4, Llama-2, or hybrid)
2. **Design system prompt** (mushroom expertise template)
3. **Implement LLMClassifier** (API client)
4. **Build ObservationParser** (description → structured)
5. **Create test suite** (validation data)
6. **Evaluate & tune** (accuracy vs cost vs speed)
7. **Document & integrate** (Phase 6)

## Related Files

- Phase 3: `Docs/06-image-recognition.md`
- Phase 4: `Docs/07-trait-classification.md`
- Implementation Plan: `Docs/implementationplan.md`
- Species Reference: `Docs/02-species-traits.md`

---

**Ready to implement Phase 5 when you give the go-ahead!**
