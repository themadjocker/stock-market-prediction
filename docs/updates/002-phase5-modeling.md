# Update 002: Phase 5 Modeling Infrastructure

Phase 5 turns the leakage-safe validation output into a reusable model-training and comparison layer.

## Included

- Stable model registry and factory for baselines, XGBoost, LSTM, and Transformer.
- Fresh model instance per temporal validation fold.
- Fold-level out-of-sample `ValidationResult` aggregation.
- Deterministic model comparison tables using validation metrics only.
- Global reproducibility controls for Python, NumPy, and PyTorch.
- Binary classification support for LSTM and Transformer using BCE-with-logits training and probability predictions.
- Existing regression targets remain supported for 1-session and 5-session returns.

## Boundary preserved

Phase 5 consumes `ValidatedFoldBundle` from Phase 4. It does not create date splits, fit scalers, or access the quarantined final test partition.

For selection-based models, the Phase 4 trainer creates an inner chronological selection holdout. The outer validation fold remains evaluation-only.

## Validation

The Phase 5 focused suite passes 27 tests in the development environment. The full suite excluding the container's unavailable TA-Lib dependency passes 109 tests. The official project environment should run the complete suite, including `tests/test_technical.py`.
