from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
import pytest
from scripts.run_p6 import DEFAULT_N_STARTUP_TRIALS, DEFAULT_N_TRIALS, FIXED_SEED

from ml.features.dataset import ModelDataset, build_model_dataset
from ml.models.lstm_model import LSTMModel
from ml.models.sequence import construct_lookback_sequences
from ml.models.target import target_spec_for
from ml.models.xgboost_model import XGBoostModel
from ml.optimization.folds import prepare_inner_folds
from ml.optimization.outer_gate import (
    OuterValidationGate,
    OuterValidationNotFrozenError,
    OuterValidationNotOpenError,
)
from ml.optimization.search import (
    InvalidOptimizationTrial,
    evaluate_model_on_folds,
    finite_mean,
)
from ml.validation.model_selection import make_p6_inner_folds
from ml.validation.split import chronological_split


def make_test_panel_data(n_days: int = 60, seed: int = 42) -> ModelDataset:
    np.random.seed(seed)
    dates = pd.date_range("2026-01-01", periods=n_days, freq="B")
    rows = []
    for ticker in ["AAA", "BBB"]:
        for i, d in enumerate(dates):
            rows.append(
                {
                    "ticker": ticker,
                    "session_date": d,
                    "feat_1": np.sin(i * 0.1) + (10.0 if ticker == "BBB" else 0.0),
                    "feat_2": np.cos(i * 0.1),
                    "target_return_1d": 0.01 * (1.0 if i % 2 == 0 else -1.0),
                    "target_return_5d": 0.02 * (1.0 if i % 2 == 0 else -1.0),
                    "target_up_1d": 1 if i % 2 == 0 else 0,
                    "target_up_5d": 1 if i % 2 == 0 else 0,
                }
            )
    df = pd.DataFrame(rows).sort_values(["session_date", "ticker"]).reset_index(drop=True)
    return build_model_dataset(
        df,
        feature_columns=["feat_1", "feat_2"],
        target_columns=[
            "target_return_1d",
            "target_return_5d",
            "target_up_1d",
            "target_up_5d",
        ],
    )


def _require_frame(value: pd.DataFrame | None) -> pd.DataFrame:
    """Narrow an optional validation bundle frame for test setup."""
    if value is None:
        raise AssertionError("Expected a materialized DataFrame in the test bundle.")
    return value


def test_two_inner_folds_exist_with_correct_chronological_geometry() -> None:
    dataset = make_test_panel_data(n_days=60)
    outer_bundle, _, _ = chronological_split(
        dataset,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        target_horizon=5,
    )
    splits = make_p6_inner_folds(outer_bundle.metadata_train, target_horizon=5)

    assert len(splits) == 2
    f1, f2 = splits[0], splits[1]

    # Fold 1 selection starts before Fold 2 selection
    assert f1.selection_start < f2.selection_start

    # Monotonic chronological order: all fit dates must precede selection start
    meta = outer_bundle.metadata_train
    f1_fit_dates = pd.to_datetime(meta.iloc[f1.fit_indices]["session_date"])
    f2_fit_dates = pd.to_datetime(meta.iloc[f2.fit_indices]["session_date"])

    assert (f1_fit_dates < f1.selection_start).all()
    assert (f2_fit_dates < f2.selection_start).all()


def test_purged_fit_rows_do_not_contribute_labels_to_selection() -> None:
    dataset = make_test_panel_data(n_days=60)
    outer_bundle, _, _ = chronological_split(
        dataset,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        target_horizon=5,
    )
    splits = make_p6_inner_folds(outer_bundle.metadata_train, target_horizon=5)

    for split in splits:
        # Zero intersection between fit indices and selection indices
        intersection = set(split.fit_indices.tolist()) & set(split.selection_indices.tolist())
        assert len(intersection) == 0

        # Purged count must be greater than zero when horizon > 0
        assert split.purged_fit_count > 0


def test_purged_historical_features_remain_available_as_sequence_context() -> None:
    dataset = make_test_panel_data(n_days=60)
    outer_bundle, _, _ = chronological_split(
        dataset,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        target_horizon=5,
    )
    splits = make_p6_inner_folds(outer_bundle.metadata_train, target_horizon=5)
    inner_folds = prepare_inner_folds(
        X_raw=_require_frame(outer_bundle.X_train_raw),
        y_raw=outer_bundle.y_train[["target_return_5d"]],
        metadata=outer_bundle.metadata_train,
        splits=splits,
    )

    meta = outer_bundle.metadata_train.reset_index(drop=True)
    for split, fold in zip(splits, inner_folds):
        raw_fit = set(split.fit_indices.tolist())
        purged = (
            set(np.flatnonzero((pd.to_datetime(meta["session_date"]) < split.selection_start).to_numpy()).tolist())
            - raw_fit
        )
        context_ids = set(
            zip(
                fold.metadata_context["ticker"].astype(str),
                pd.to_datetime(fold.metadata_context["session_date"]),
            )
        )
        purged_ids = set(
            zip(
                meta.iloc[sorted(purged)]["ticker"].astype(str),
                pd.to_datetime(meta.iloc[sorted(purged)]["session_date"]),
            )
        )
        selection_ids = set(
            zip(
                fold.metadata_selection["ticker"].astype(str),
                pd.to_datetime(fold.metadata_selection["session_date"]),
            )
        )

        # Exact historical context is retained, including every purge-gap feature row.
        assert purged_ids
        assert purged_ids.issubset(context_ids)
        assert selection_ids.isdisjoint(context_ids)

        # Labels are supplied only from Inner Fit. No purge-gap label frame exists in the fold.
        fit_ids = set(
            zip(
                fold.metadata_fit["ticker"].astype(str),
                pd.to_datetime(fold.metadata_fit["session_date"]),
            )
        )
        assert len(fit_ids) == len(fold.y_fit)
        assert purged_ids.isdisjoint(fit_ids)


def test_purge_gap_is_used_as_sequence_context_without_purged_labels() -> None:
    dataset = make_test_panel_data(n_days=60)
    outer_bundle, _, _ = chronological_split(dataset, target_horizon=5)
    splits = make_p6_inner_folds(outer_bundle.metadata_train, target_horizon=5)
    inner_folds = prepare_inner_folds(
        X_raw=_require_frame(outer_bundle.X_train_raw),
        y_raw=outer_bundle.y_train[["target_return_5d"]],
        metadata=outer_bundle.metadata_train,
        splits=splits,
    )
    fold = inner_folds[0]
    split = splits[0]
    meta = outer_bundle.metadata_train.reset_index(drop=True)

    raw_fit_indices = set(
        np.flatnonzero((pd.to_datetime(meta["session_date"]) < split.selection_start).to_numpy()).tolist()
    )
    purged_indices = raw_fit_indices - set(split.fit_indices.tolist())
    assert purged_indices

    # Every purged feature row is physically present in history, while selection rows are not.
    context_positions = set(range(len(fold.metadata_context)))
    assert context_positions
    context_dates = pd.to_datetime(fold.metadata_context["session_date"])
    assert (context_dates < split.selection_start).all()

    # Sequence endpoints at the beginning of Selection must therefore be able to use
    # the immediately preceding historical context instead of a concatenated gap.
    selection_first = fold.metadata_selection["session_date"].min()
    preceding = fold.metadata_context[pd.to_datetime(fold.metadata_context["session_date"]) < selection_first]
    assert not preceding.empty
    assert pd.to_datetime(preceding["session_date"]).is_monotonic_increasing


def test_lookback_5_and_10_have_identical_scoring_ids() -> None:
    dataset = make_test_panel_data(n_days=60)
    outer_bundle, _, _ = chronological_split(dataset, target_horizon=5)
    splits = make_p6_inner_folds(outer_bundle.metadata_train, target_horizon=5)
    fold = splits[0]
    X = _require_frame(outer_bundle.X_train_raw)
    meta = outer_bundle.metadata_train

    # Score only the actual inner-selection endpoints. The shared max-lookback rule
    # must make eligibility identical for lookback 5 and 10.
    endpoints = fold.selection_indices
    _, valid_lb5 = construct_lookback_sequences(
        X, metadata=meta, lookback=5, endpoint_indices=endpoints, max_lookback=10
    )
    _, valid_lb10 = construct_lookback_sequences(
        X, metadata=meta, lookback=10, endpoint_indices=endpoints, max_lookback=10
    )

    def scoring_ids(indices: np.ndarray) -> set[tuple[str, pd.Timestamp]]:
        return set(
            zip(
                meta.iloc[indices]["ticker"].astype(str),
                pd.to_datetime(meta.iloc[indices]["session_date"]),
            )
        )

    assert scoring_ids(valid_lb5) == scoring_ids(valid_lb10)
    assert scoring_ids(valid_lb5) == scoring_ids(endpoints)


def test_sequence_rows_remain_chronologically_contiguous_through_purge_gap() -> None:
    dataset = make_test_panel_data(n_days=40)
    outer_bundle, _, _ = chronological_split(
        dataset,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        target_horizon=5,
    )
    splits = make_p6_inner_folds(outer_bundle.metadata_train, target_horizon=5)
    inner_folds = prepare_inner_folds(
        X_raw=_require_frame(outer_bundle.X_train_raw),
        y_raw=outer_bundle.y_train[["target_return_5d"]],
        metadata=outer_bundle.metadata_train,
        splits=splits,
    )

    for split, fold in zip(splits, inner_folds):
        context = fold.metadata_context.reset_index(drop=True)
        for ticker in context["ticker"].unique():
            ticker_context = context[context["ticker"] == ticker].reset_index(drop=True)
            dates = pd.to_datetime(ticker_context["session_date"])
            assert dates.is_monotonic_increasing

            # The context must equal the original contiguous prefix before Selection,
            # not Fit rows with the purge gap physically removed.
            expected = outer_bundle.metadata_train[
                pd.to_datetime(outer_bundle.metadata_train["session_date"]) < split.selection_start
            ]
            expected_ticker = expected[expected["ticker"] == ticker].reset_index(drop=True)
            pd.testing.assert_series_equal(
                dates.reset_index(drop=True),
                pd.to_datetime(expected_ticker["session_date"]).reset_index(drop=True),
                check_names=False,
            )


def test_outer_validation_gate_is_closed_until_configuration_is_frozen() -> None:
    dataset = make_test_panel_data(n_days=40)
    outer_bundle, _, _ = chronological_split(dataset, target_horizon=5)
    gate = OuterValidationGate(outer_bundle)

    with pytest.raises(OuterValidationNotOpenError):
        gate.get_bundle()

    with pytest.raises(OuterValidationNotFrozenError):
        gate.open()

    frozen = gate.freeze()
    with pytest.raises(OuterValidationNotOpenError):
        frozen.get_bundle()

    opened = frozen.open()
    assert opened.get_bundle() is outer_bundle


def test_outer_validation_values_cannot_change_inner_selection() -> None:
    dataset = make_test_panel_data(n_days=60)
    bundle_a, _, _ = chronological_split(dataset, target_horizon=5)

    # The inner objective receives only outer-train raw data. Build two equivalent
    # train bundles and alter only their Outer Validation payloads.
    from dataclasses import replace

    bundle_b = replace(
        bundle_a,
        X_val_raw=_require_frame(bundle_a.X_val_raw) * 100000.0,
        y_val=bundle_a.y_val.copy(),
    )
    bundle_b.y_val.iloc[:, 0] = bundle_b.y_val.iloc[:, 0] + 999.0

    splits_a = make_p6_inner_folds(bundle_a.metadata_train, target_horizon=5)
    splits_b = make_p6_inner_folds(bundle_b.metadata_train, target_horizon=5)
    x_train_raw_a = _require_frame(bundle_a.X_train_raw)
    x_train_raw_b = _require_frame(bundle_b.X_train_raw)

    folds_a = prepare_inner_folds(
        X_raw=x_train_raw_a,
        y_raw=bundle_a.y_train[["target_return_5d"]],
        metadata=bundle_a.metadata_train,
        splits=splits_a,
    )
    folds_b = prepare_inner_folds(
        X_raw=x_train_raw_b,
        y_raw=bundle_b.y_train[["target_return_5d"]],
        metadata=bundle_b.metadata_train,
        splits=splits_b,
    )

    score_a = evaluate_model_on_folds(XGBoostModel, {"n_estimators": 5, "max_depth": 2}, folds_a, "regression")
    score_b = evaluate_model_on_folds(XGBoostModel, {"n_estimators": 5, "max_depth": 2}, folds_b, "regression")
    np.testing.assert_allclose(score_a, score_b, rtol=0.0, atol=0.0)


def test_classification_predict_and_predict_proba() -> None:
    dataset = make_test_panel_data(n_days=30)
    outer_bundle, _, _ = chronological_split(
        dataset,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        target_horizon=1,
    )

    # 1. XGBoost
    xgb = XGBoostModel(n_estimators=10, max_depth=2, random_state=42)
    xgb.fit(
        X_train=outer_bundle.X_train,
        y_train=outer_bundle.y_train[["target_up_1d"]],
        X_val=outer_bundle.X_val,
        y_val=outer_bundle.y_val[["target_up_1d"]],
    )
    preds = xgb.predict(outer_bundle.X_val)
    probas = xgb.predict_proba(outer_bundle.X_val)
    assert set(np.unique(preds)).issubset({0.0, 1.0})
    assert probas.shape == (len(outer_bundle.X_val), 2)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)

    # 2. LSTM
    lstm = LSTMModel(lookback=5, max_lookback=10, hidden_dim=8, epochs=2, batch_size=16, seed=42)
    lstm.fit(
        X_train=outer_bundle.X_train,
        y_train=outer_bundle.y_train[["target_up_1d"]],
        metadata_train=outer_bundle.metadata_train,
    )
    l_preds = lstm.predict(
        outer_bundle.X_val,
        metadata=outer_bundle.metadata_val,
        history_X=outer_bundle.X_train,
        history_metadata=outer_bundle.metadata_train,
    )
    l_probas = lstm.predict_proba(
        outer_bundle.X_val,
        metadata=outer_bundle.metadata_val,
        history_X=outer_bundle.X_train,
        history_metadata=outer_bundle.metadata_train,
    )
    assert set(np.unique(l_preds)).issubset({0.0, 1.0})
    assert l_probas.shape == (len(outer_bundle.X_val), 2)
    np.testing.assert_allclose(l_probas.sum(axis=1), 1.0, atol=1e-5)


def test_binary_neural_output_sigmoid_math() -> None:
    raw = np.array([-2.0, 0.0, 1.5, 3.0])
    p1 = 1.0 / (1.0 + np.exp(-raw))
    p0 = 1.0 - p1
    np.testing.assert_allclose(p0 + p1, 1.0, atol=1e-7)
    assert (p1 > 0.0).all() and (p1 < 1.0).all()


def _make_inner_folds_for_phase4(target: str = "target_return_5d"):
    dataset = make_test_panel_data(n_days=40)
    horizon = 5 if "5d" in target else 1
    outer_bundle, _, _ = chronological_split(dataset, target_horizon=horizon)
    splits = make_p6_inner_folds(outer_bundle.metadata_train, target_horizon=horizon)
    return prepare_inner_folds(
        X_raw=_require_frame(outer_bundle.X_train_raw),
        y_raw=outer_bundle.y_train[[target]],
        metadata=outer_bundle.metadata_train,
        splits=splits,
    )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_predictions_invalidate_trial(bad_value: float) -> None:
    inner_folds = _make_inner_folds_for_phase4()

    class NonFinitePredictionModel(XGBoostModel):
        def predict(
            self,
            X: pd.DataFrame,
            metadata: pd.DataFrame | None = None,
            history_X: pd.DataFrame | None = None,
            history_metadata: pd.DataFrame | None = None,
        ) -> np.ndarray:
            result = super().predict(
                X,
                metadata=metadata,
                history_X=history_X,
                history_metadata=history_metadata,
            )
            result[0] = bad_value
            return result

    with pytest.raises(InvalidOptimizationTrial, match="non-finite predictions"):
        evaluate_model_on_folds(
            NonFinitePredictionModel,
            {"n_estimators": 5, "max_depth": 2},
            inner_folds,
            task_type="regression",
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_training_loss_invalidates_trial(bad_value: float) -> None:
    inner_folds = _make_inner_folds_for_phase4()

    class NonFiniteTrainingLossModel(XGBoostModel):
        def fit(
            self,
            X_train: pd.DataFrame,
            y_train: pd.DataFrame,
            X_val: pd.DataFrame | None = None,
            y_val: pd.DataFrame | None = None,
            metadata_train: pd.DataFrame | None = None,
            metadata_val: pd.DataFrame | None = None,
        ) -> XGBoostModel:
            raise ValueError(f"Non-finite training loss in test model: {bad_value}")

    with pytest.raises(InvalidOptimizationTrial, match="Non-finite training loss"):
        evaluate_model_on_folds(
            NonFiniteTrainingLossModel,
            {"n_estimators": 5, "max_depth": 2},
            inner_folds,
            task_type="regression",
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_selection_loss_invalidates_trial(bad_value: float) -> None:
    inner_folds = _make_inner_folds_for_phase4()

    class NonFiniteSelectionLossModel(XGBoostModel):
        def fit(
            self,
            X_train: pd.DataFrame,
            y_train: pd.DataFrame,
            X_val: pd.DataFrame | None = None,
            y_val: pd.DataFrame | None = None,
            metadata_train: pd.DataFrame | None = None,
            metadata_val: pd.DataFrame | None = None,
        ) -> XGBoostModel:
            raise ValueError(f"Non-finite selection loss in test model: {bad_value}")

    with pytest.raises(InvalidOptimizationTrial, match="Non-finite selection loss"):
        evaluate_model_on_folds(
            NonFiniteSelectionLossModel,
            {"n_estimators": 5, "max_depth": 2},
            inner_folds,
            task_type="regression",
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_ticker_aggregation_invalidates_trial(monkeypatch: pytest.MonkeyPatch, bad_value: float) -> None:
    inner_folds = _make_inner_folds_for_phase4()

    def bad_metrics(y_true, y_pred):
        return {"relative_rmse": bad_value}

    monkeypatch.setattr("ml.optimization.search.calculate_metrics", bad_metrics)

    with pytest.raises(InvalidOptimizationTrial, match="Non-finite metric"):
        evaluate_model_on_folds(
            XGBoostModel,
            {"n_estimators": 5, "max_depth": 2},
            inner_folds,
            task_type="regression",
        )


@pytest.mark.parametrize("bad_values", [[np.nan], [np.inf], [-np.inf]])
def test_non_finite_aggregate_mean_is_never_silently_dropped(bad_values: list[float]) -> None:

    with pytest.raises(InvalidOptimizationTrial, match="Non-finite value"):
        finite_mean([0.5, *bad_values, 0.7], label="test aggregation")


def test_finite_aggregation_remains_mean_of_all_values() -> None:

    assert finite_mean([0.5, 0.7, 0.9], label="test aggregation") == pytest.approx(0.7)


def test_regression_objective_relative_rmse() -> None:
    from ml.models.metrics import calculate_metrics

    y_true = np.array([0.02, -0.01, 0.03, -0.02])
    y_pred = np.array([0.01, -0.01, 0.02, -0.01])

    res = calculate_metrics(y_true, y_pred)
    zero_rmse = np.sqrt(np.mean(y_true**2))
    expected_rel_rmse = res["rmse"] / zero_rmse
    np.testing.assert_allclose(res["relative_rmse"], expected_rel_rmse, atol=1e-6)


def test_aggregation_mean_across_tickers_then_folds() -> None:
    dataset = make_test_panel_data(n_days=40)
    outer_bundle, _, _ = chronological_split(dataset, target_horizon=5)
    splits = make_p6_inner_folds(outer_bundle.metadata_train, target_horizon=5)
    inner_folds = prepare_inner_folds(
        X_raw=_require_frame(outer_bundle.X_train_raw),
        y_raw=outer_bundle.y_train[["target_return_5d"]],
        metadata=outer_bundle.metadata_train,
        splits=splits,
    )

    study = optuna.create_study()
    trial = study.ask()
    score = evaluate_model_on_folds(
        XGBoostModel,
        {"n_estimators": 10, "max_depth": 2},
        inner_folds,
        task_type="regression",
        trial=trial,
    )
    assert np.isfinite(score)
    fold_scores = trial.user_attrs["fold_scores"]
    assert len(fold_scores) == 2
    np.testing.assert_allclose(score, np.mean(fold_scores), atol=1e-6)


def test_classification_primary_objective_logloss() -> None:
    dataset = make_test_panel_data(n_days=40)
    outer_bundle, _, _ = chronological_split(dataset, target_horizon=1)
    splits = make_p6_inner_folds(outer_bundle.metadata_train, target_horizon=1)
    inner_folds = prepare_inner_folds(
        X_raw=_require_frame(outer_bundle.X_train_raw),
        y_raw=outer_bundle.y_train[["target_up_1d"]],
        metadata=outer_bundle.metadata_train,
        splits=splits,
    )

    score = evaluate_model_on_folds(
        XGBoostModel,
        {"n_estimators": 10, "max_depth": 2},
        inner_folds,
        task_type="classification",
    )
    assert np.isfinite(score)
    assert score > 0.0  # LogLoss is strictly positive


def test_optuna_default_configuration() -> None:
    assert DEFAULT_N_TRIALS == 30
    assert DEFAULT_N_STARTUP_TRIALS == 10
    assert FIXED_SEED == 42


def test_xgboost_settings_and_protocol_a() -> None:
    model = XGBoostModel(n_estimators=1000, early_stopping_rounds=30)
    assert model.n_estimators == 1000
    assert model.early_stopping_rounds == 30


def test_trial_metadata_retention() -> None:
    dataset = make_test_panel_data(n_days=40)
    outer_bundle, _, _ = chronological_split(dataset, target_horizon=5)
    splits = make_p6_inner_folds(outer_bundle.metadata_train, target_horizon=5)
    inner_folds = prepare_inner_folds(
        X_raw=_require_frame(outer_bundle.X_train_raw),
        y_raw=outer_bundle.y_train[["target_return_5d"]],
        metadata=outer_bundle.metadata_train,
        splits=splits,
    )

    study = optuna.create_study()
    trial = study.ask()
    score = evaluate_model_on_folds(
        XGBoostModel,
        {"n_estimators": 10, "max_depth": 2},
        inner_folds,
        task_type="regression",
        trial=trial,
        model_family="XGBoost",
        target_name="target_return_5d",
    )

    attrs = trial.user_attrs
    assert attrs["model_family"] == "XGBoost"
    assert attrs["target"] == "target_return_5d"
    assert attrs["task_type"] == "regression"
    assert "fold_scores" in attrs
    assert "fold_ticker_metrics" in attrs
    assert "fold_best_iteration_or_epoch" in attrs
    assert attrs["objective_score"] == score


def test_target_horizon_resolved_from_spec() -> None:
    assert target_spec_for("target_return_1d").horizon == 1
    assert target_spec_for("target_return_5d").horizon == 5
    assert target_spec_for("target_up_1d").horizon == 1
    assert target_spec_for("target_up_5d").horizon == 5
