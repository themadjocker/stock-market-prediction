# Update 003: Phase 6 Optimization & Frozen Experiments

Phase 6 adds a locked hyperparameter-search protocol on top of the Phase 5 model stack. It selects a model family and hyperparameters using only inner temporal folds, writes a frozen manifest, and only then opens Outer Validation for a single Protocol A evaluation.

> **Scope of this update:** the P6 runner currently uses a **synthetic development dataset** (random features, alternating labels). It validates the protocol and the manifests. It does **not** say anything about model quality on real markets.

## Included

- Two inner temporal folds built from the Outer Train partition.
- Target-specific purge and train-only preprocessing per fold.
- One Optuna study per model family (XGBoost, LSTM, Transformer) with a locked budget.
- Strict numerical-validity handling for trials.
- Frozen JSON manifests, written before Outer Validation opens.
- An Outer Validation gate that refuses access until the configuration is frozen.
- Protocol A final evaluation, and a P6 runner for all four active targets.

## Protocol

```text
P4 validated dataset
        ↓
Raw P6 search bundle
        ↓
2 inner temporal folds
  ├─ Fold 1: 60% fit / 20% selection
  └─ Fold 2: 80% fit / 20% selection
        ↓
Target-specific purge → train-only preprocessing per fold
        ↓
30 Optuna trials per family
        ↓
Lowest inner objective wins
        ↓
Frozen manifest written
        ↓
Outer Validation gate: freeze → open
        ↓
Protocol A: fresh model on Fold 2 Inner Fit,
early stopping on Fold 2 Inner Selection,
single evaluation on Outer Validation
```

## Leakage and temporal guarantees

| Topic | Rule |
|:--|:--|
| Chronology | Splits follow session dates. No random splitting. |
| Purge | A training row is dropped when its label resolves on or after the selection (or validation) boundary. Length equals the target horizon: 1 session for `*_1d`, 5 sessions for `*_5d`. |
| Preprocessing | A fresh train-only scaler is fitted for each inner fold on that fold's fit rows only. Refitting is rejected. |
| Final evaluation | Outer Validation is transformed with the fitted Fold 2 preprocessor. |
| Sequence continuity | For LSTM and Transformer, purged rows remain available as **feature-only history**, so windows keep their real chronological order. Their labels never train a model. |
| Lookback alignment | Lookbacks 5 and 10 share the same eligible scoring rows. |

## Search spaces

| Parameter | XGBoost | LSTM | Transformer |
|:--|:-:|:-:|:-:|
| `lookback` | n/a | 5 · 10 | 5 · 10 |
| `max_depth` | 2–8 | n/a | n/a |
| `num_layers` | n/a | 1–2 | 1–2 |
| `hidden_dim` / `d_model` | n/a | 16 · 32 · 64 | 16 · 32 · 64 |
| `nhead` | n/a | n/a | 2 · 4 |
| `learning_rate` / `lr` (log) | 1e-3 – 0.3 | 1e-4 – 1e-2 | 1e-4 – 1e-2 |
| `dropout` | n/a | 0.0 – 0.3 | 0.0 – 0.3 |
| `subsample` · `colsample_bytree` | 0.5 – 1.0 | n/a | n/a |
| Fixed | `n_estimators=1000`, `early_stopping_rounds=30` | `epochs=15` | `epochs=15` |

**Budget (all families):** 30 trials · 10 startup trials · `TPESampler` · seed 42 · `n_jobs=1` · no pruning. The runner validates the 30-trial budget.

## Objectives

| Target | Task | Primary objective |
|:--|:--|:--|
| `target_return_1d` | Regression | Relative RMSE |
| `target_return_5d` | Regression | Relative RMSE |
| `target_up_1d` | Classification | LogLoss |
| `target_up_5d` | Classification | LogLoss |

Secondary metrics are recorded in the manifest and never enter the objective.

## Numerical validity

Non-finite predictions, losses or objective aggregates invalidate the affected trial. They are not dropped or partially averaged.

## Frozen manifests

One manifest per active target, written to the repository root before Outer Validation opens:

```text
p6_frozen_target_return_1d.json
p6_frozen_target_return_5d.json
p6_frozen_target_up_1d.json
p6_frozen_target_up_5d.json
```

Each manifest has 31 top-level fields, including the experiment and target identity, the winning family and hyperparameters, feature schema, preprocessing configuration, inner-fold geometry, purge length, scoring and sequence-alignment rules, early-stopping and Protocol A rules, determinism settings, objective and secondary metrics, baseline definition, Optuna budget, sampler and trial metadata, search-space identity and definition, dataset manifest, library versions and code provenance.

## Outer Validation gate

`OuterValidationGate` wraps the Outer Validation bundle. `get_bundle()` raises until the gate has been frozen and then opened, so the search code cannot read Outer Validation. This is an **in-process** guard: it does not persist across runs.

## Smoke-run result

| Target | Task | Winning family | Purge |
|:--|:--|:--:|:-:|
| `target_return_1d` | Regression | XGBoost | 1 |
| `target_return_5d` | Regression | XGBoost | 5 |
| `target_up_1d` | Classification | LSTM | 1 |
| `target_up_5d` | Classification | XGBoost | 5 |

> These winners come from a synthetic dataset with random features. Treat them as placeholders that prove the pipeline runs end to end.

## Known limitations

- The runner's synthetic features are generated without an explicit seed, so a re-run does not reproduce the same manifests.
- Outer Validation metrics are logged and returned by the runner, but are not written into the manifests.
- The gate has no persistent open-ledger, so it cannot count how many times Outer Validation was opened across runs.
- Family selection takes the lowest inner objective. A baseline guard is not implemented yet.
- No Optuna dashboard or persistent study storage.

## Reproduce

```bash
# from the repository root
uv run --project ml python scripts/run_p6.py --target target_return_5d

# ML tests
cd ml && uv run pytest -q
```

## Next

P7 evaluation and backtesting on real market data, followed by P8 artifacts and versioning.
