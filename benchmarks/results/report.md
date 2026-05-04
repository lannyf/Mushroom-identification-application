# Benchmark Results

## Accuracy Summary

| Method | Top-1 | Top-3 | Coverage | Mean Time (ms) |
|--------|-------|-------|----------|----------------|
| cnn | 0.500 | 0.550 | 1.000 | 103.7 |
| tree_auto | 0.718 | 0.718 | 0.650 | 2450.7 |
| tree_oracle | 0.680 | 0.680 | 0.833 | 0.2 |
| trait_db | 0.033 | 0.050 | 1.000 | 3.8 |
| multimodal_final | 0.583 | 0.583 | 1.000 | 116.6 |

## OOD Analysis (CNN)

- ID avg confidence: 0.899
- OOD avg confidence: 0.693
- Confidence gap: 0.206