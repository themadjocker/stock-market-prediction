from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.models.base import BaseModel
from ml.models.reproducibility import set_global_seed
from ml.models.sequence import construct_lookback_sequences
from ml.models.target import TargetSpec


class _LSTMNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, F]
        out, _ = self.lstm(x)
        # Take the representation at the final time-step L
        last_step = out[:, -1, :]
        return self.fc(last_step).squeeze(-1)


class LSTMModel(BaseModel):
    """
    Deterministic PyTorch LSTM model for stock return forecasting.

    Constructs strictly backward-looking [B, L, F] sequence tensors.
    """

    def __init__(
        self,
        lookback: int = 10,
        max_lookback: int | None = None,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        lr: float = 0.005,
        epochs: int = 15,
        batch_size: int = 32,
        seed: int = 42,
    ):
        self.lookback = lookback
        self.max_lookback = max_lookback if max_lookback is not None else lookback
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

        self._net: _LSTMNet | None = None
        self._is_fitted: bool = False
        self._feature_dim: int = 0
        self._target_spec: TargetSpec | None = None
        self._history_X: pd.DataFrame | None = None
        self._history_metadata: pd.DataFrame | None = None
        self._requires_prediction_metadata: bool = False
        self._selection_best_epoch: int | None = None

    @property
    def requires_selection_holdout(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return f"LSTM(L={self.lookback}, H={self.hidden_dim})"

    def _set_seed(self) -> None:
        set_global_seed(self.seed)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame | None = None,
        y_val: pd.DataFrame | None = None,
        metadata_train: pd.DataFrame | None = None,
        metadata_val: pd.DataFrame | None = None,
        val_history_X: pd.DataFrame | None = None,
        val_history_metadata: pd.DataFrame | None = None,
    ) -> LSTMModel:
        self._set_seed()

        if X_train.empty or y_train.empty:
            raise ValueError("Training data cannot be empty.")

        spec = self._validate_and_set_target(y_train)

        self._feature_dim = X_train.shape[1]
        self._history_X = X_train.reset_index(drop=True).copy()
        self._history_metadata = metadata_train.reset_index(drop=True).copy() if metadata_train is not None else None
        self._requires_prediction_metadata = (
            bool(metadata_train is not None and metadata_train["ticker"].nunique() > 1)
            if metadata_train is not None and "ticker" in metadata_train.columns
            else False
        )
        X_seq, valid_idx = construct_lookback_sequences(
            X_train,
            metadata=metadata_train,
            lookback=self.lookback,
            max_lookback=self.max_lookback,
        )

        if len(X_seq) == 0:
            raise ValueError(f"Insufficient training observations ({len(X_train)}) for lookback {self.lookback}.")

        y_targets = y_train.iloc[:, 0].to_numpy(dtype=float)[valid_idx]

        train_dataset = TensorDataset(
            torch.tensor(X_seq, dtype=torch.float32),
            torch.tensor(y_targets, dtype=torch.float32),
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)

        val_loader = None
        val_sample_count = 0
        if X_val is not None and y_val is not None and not X_val.empty and not y_val.empty:
            if val_history_X is not None:
                context_X = pd.concat([val_history_X, X_val], ignore_index=True)
                if val_history_metadata is not None and metadata_val is not None:
                    context_metadata = pd.concat([val_history_metadata, metadata_val], ignore_index=True)
                else:
                    context_metadata = None
                validation_endpoints = np.arange(len(val_history_X), len(context_X), dtype=int)
                val_offset = len(val_history_X)
            else:
                context_X = pd.concat([X_train, X_val], ignore_index=True)
                if metadata_train is not None and metadata_val is not None:
                    context_metadata = pd.concat([metadata_train, metadata_val], ignore_index=True)
                else:
                    context_metadata = None
                validation_endpoints = np.arange(len(X_train), len(context_X), dtype=int)
                val_offset = len(X_train)

            X_val_seq, val_global_idx = construct_lookback_sequences(
                context_X,
                metadata=context_metadata,
                lookback=self.lookback,
                endpoint_indices=validation_endpoints,
                max_lookback=self.max_lookback,
            )
            val_valid_idx = val_global_idx - val_offset
            val_sample_count = len(X_val_seq)
            if val_sample_count > 0:
                y_val_targets = y_val.iloc[:, 0].to_numpy(dtype=float)[val_valid_idx]
                val_dataset = TensorDataset(
                    torch.tensor(X_val_seq, dtype=torch.float32),
                    torch.tensor(y_val_targets, dtype=torch.float32),
                )
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        self._net = _LSTMNet(
            input_dim=self._feature_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        criterion = nn.MSELoss() if spec.task_type == "regression" else nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(self.epochs):
            self._net.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                pred = self._net(batch_x)
                loss = criterion(pred, batch_y)
                if not torch.isfinite(loss):
                    raise ValueError(f"Non-finite training loss in LSTM: {loss.item()}")
                loss.backward()
                optimizer.step()

            if val_loader is not None:
                self._net.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vpred = self._net(vx)
                        v_loss = criterion(vpred, vy).item()
                        if not np.isfinite(v_loss):
                            raise ValueError(f"Non-finite batch validation loss in LSTM: {v_loss}")
                        val_loss += v_loss * len(vy)
                val_loss /= val_sample_count
                if not np.isfinite(val_loss):
                    raise ValueError(f"Non-finite epoch validation loss in LSTM: {val_loss}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}
                    self._selection_best_epoch = epoch + 1

        if best_state is not None and self._net is not None:
            self._net.load_state_dict(best_state)

        self._is_fitted = True
        return self

    def predict(
        self,
        X: pd.DataFrame,
        metadata: pd.DataFrame | None = None,
        history_X: pd.DataFrame | None = None,
        history_metadata: pd.DataFrame | None = None,
    ) -> np.ndarray:
        if not self._is_fitted or self._net is None:
            raise RuntimeError("Model must be fitted before predict.")
        if len(X) == 0:
            return np.array([], dtype=float)

        if history_X is None:
            history_X = self._history_X
            history_metadata = self._history_metadata

        if history_X is not None:
            if metadata is None and self._requires_prediction_metadata:
                raise ValueError("metadata is required for multi-ticker prediction.")
            if history_metadata is not None and metadata is None:
                # Single-ticker history can still use its stored metadata if the
                # caller omits prediction metadata only when no new ticker identity
                # information is needed. For multi-ticker panels we require it above.
                raise ValueError("metadata is required when stored history contains ticker metadata.")
            context_X = pd.concat([history_X, X], ignore_index=True)
            if history_metadata is not None and metadata is not None:
                context_metadata = pd.concat([history_metadata, metadata], ignore_index=True)
            else:
                context_metadata = None
            endpoints = np.arange(len(history_X), len(context_X), dtype=int)
            X_seq, global_idx = construct_lookback_sequences(
                context_X,
                metadata=context_metadata,
                lookback=self.lookback,
                endpoint_indices=endpoints,
                max_lookback=self.max_lookback,
            )
            valid_idx = global_idx - len(history_X)
        else:
            X_seq, valid_idx = construct_lookback_sequences(
                X,
                metadata=metadata,
                lookback=self.lookback,
                max_lookback=self.max_lookback,
            )

        if len(X_seq) != len(X):
            missing = sorted(set(range(len(X))) - set(valid_idx.tolist()))
            raise ValueError(
                f"Insufficient same-ticker historical context for prediction rows {missing}; lookback={self.lookback}."
            )

        self._net.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X_seq, dtype=torch.float32)
            raw_preds = self._net(x_tensor).numpy().ravel()

        if self.target_spec is not None and self.target_spec.task_type == "classification":
            probs = 1.0 / (1.0 + np.exp(-raw_preds))
            preds = (probs >= 0.5).astype(float)
        else:
            preds = raw_preds

        predictions = np.empty(len(X), dtype=float)
        predictions[valid_idx] = preds
        return predictions

    def predict_proba(
        self,
        X: pd.DataFrame,
        metadata: pd.DataFrame | None = None,
        history_X: pd.DataFrame | None = None,
        history_metadata: pd.DataFrame | None = None,
    ) -> np.ndarray:
        if not self._is_fitted or self._net is None:
            raise RuntimeError("Model must be fitted before predict_proba.")
        if len(X) == 0:
            return np.empty((0, 2), dtype=float)
        if self.target_spec is None or self.target_spec.task_type != "classification":
            raise ValueError("predict_proba is only available for classification tasks.")

        if history_X is None:
            history_X = self._history_X
            history_metadata = self._history_metadata

        if history_X is not None:
            if metadata is None and self._requires_prediction_metadata:
                raise ValueError("metadata is required for multi-ticker prediction.")
            if history_metadata is not None and metadata is None:
                raise ValueError("metadata is required when stored history contains ticker metadata.")
            context_X = pd.concat([history_X, X], ignore_index=True)
            if history_metadata is not None and metadata is not None:
                context_metadata = pd.concat([history_metadata, metadata], ignore_index=True)
            else:
                context_metadata = None
            endpoints = np.arange(len(history_X), len(context_X), dtype=int)
            X_seq, global_idx = construct_lookback_sequences(
                context_X,
                metadata=context_metadata,
                lookback=self.lookback,
                endpoint_indices=endpoints,
                max_lookback=self.max_lookback,
            )
            valid_idx = global_idx - len(history_X)
        else:
            X_seq, valid_idx = construct_lookback_sequences(
                X,
                metadata=metadata,
                lookback=self.lookback,
                max_lookback=self.max_lookback,
            )

        if len(X_seq) != len(X):
            missing = sorted(set(range(len(X))) - set(valid_idx.tolist()))
            raise ValueError(
                f"Insufficient same-ticker historical context for prediction rows {missing}; lookback={self.lookback}."
            )

        self._net.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X_seq, dtype=torch.float32)
            raw_preds = self._net(x_tensor).numpy().ravel()

        probs1 = 1.0 / (1.0 + np.exp(-raw_preds))
        probs0 = 1.0 - probs1

        predictions = np.empty((len(X), 2), dtype=float)
        predictions[valid_idx, 0] = probs0
        predictions[valid_idx, 1] = probs1
        return predictions

    def get_params(self) -> dict[str, Any]:
        return {
            "lookback": self.lookback,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "selection_best_epoch": self._selection_best_epoch,
            "task_type": self.target_spec.task_type if self.target_spec is not None else None,
            "target_name": self.target_spec.name if self.target_spec is not None else None,
        }
