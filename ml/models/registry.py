from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ml.models.base import BaseModel
from ml.models.baseline import NaiveBaselineModel
from ml.models.lstm_model import LSTMModel
from ml.models.transformer_model import TransformerModel
from ml.models.xgboost_model import XGBoostModel

ModelFactory = Callable[..., BaseModel]

MODEL_REGISTRY: dict[str, ModelFactory] = {
    "baseline_zero": lambda **kwargs: NaiveBaselineModel(strategy="zero", **kwargs),
    "baseline_mean": lambda **kwargs: NaiveBaselineModel(strategy="mean", **kwargs),
    "baseline_majority": lambda **kwargs: NaiveBaselineModel(strategy="majority", **kwargs),
    "xgboost": XGBoostModel,
    "lstm": LSTMModel,
    "transformer": TransformerModel,
}


def create_model(name: str, **kwargs: Any) -> BaseModel:
    """Create one configured P5 model by stable registry name."""
    try:
        factory = MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available models: {available}") from exc
    return factory(**kwargs)


def available_models() -> tuple[str, ...]:
    return tuple(sorted(MODEL_REGISTRY))
