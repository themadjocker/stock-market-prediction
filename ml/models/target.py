from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

TaskType = Literal["regression", "classification"]

ACTIVE_TARGET_HORIZONS: tuple[int, ...] = (1, 5)
_TARGET_RE = re.compile(r"^target_(return|up)_(\d+)d$")


@dataclass(frozen=True)
class TargetSpec:
    """Explicit contract for one model-training target."""

    name: str
    task_type: TaskType
    horizon: int

    def __post_init__(self) -> None:
        if self.task_type not in {"regression", "classification"}:
            raise ValueError(f"Unsupported task_type: {self.task_type}")
        if self.horizon <= 0:
            raise ValueError("Target horizon must be positive.")


def target_spec_for(name: str) -> TargetSpec:
    """Resolve an active target column into its explicit task contract."""
    match = _TARGET_RE.fullmatch(name)
    if match is None:
        raise ValueError(
            f"Unsupported target '{name}'. Active targets are target_return_1d, "
            "target_return_5d, target_up_1d, and target_up_5d."
        )

    kind, horizon_text = match.groups()
    horizon = int(horizon_text)
    if horizon not in ACTIVE_TARGET_HORIZONS:
        raise ValueError(
            f"Target '{name}' uses inactive horizon {horizon}d. Active horizons are {ACTIVE_TARGET_HORIZONS}."
        )

    return TargetSpec(
        name=name,
        task_type="regression" if kind == "return" else "classification",
        horizon=horizon,
    )


def validate_target_frame(
    y: pd.DataFrame,
    target_spec: TargetSpec | None = None,
) -> TargetSpec:
    """Require exactly one explicit active target and return its resolved spec."""
    if y.empty:
        raise ValueError("Target data cannot be empty.")
    if len(y.columns) != 1:
        raise ValueError(
            f"P5 requires exactly one target column per training run; received {len(y.columns)}: {list(y.columns)}"
        )

    actual = target_spec_for(str(y.columns[0]))
    if target_spec is not None and actual != target_spec:
        raise ValueError(f"Target mismatch: data contains '{actual.name}' but model expects '{target_spec.name}'.")

    values = pd.to_numeric(y.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Target '{actual.name}' contains non-finite values.")

    if actual.task_type == "classification":
        unique = set(np.unique(values).tolist())
        if not unique.issubset({0.0, 1.0}):
            raise ValueError(f"Classification target '{actual.name}' must contain only 0/1 values.")

    return actual
