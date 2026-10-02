"""GPU-capable tabular Transformer classifier for flow-level NIDS data.

The public classifier API follows ``ARCHITECTURE.md``. It accepts preprocessed
numeric flow features from ``src.data`` and returns the original integer class
IDs supplied by the data pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class ModelConfig:
    """Configuration shared by local smoke tests and GPU experiments.

    ``device='auto'`` selects CUDA when available and otherwise CPU. Mixed
    precision is enabled only on CUDA, so the same configuration stays safe on
    a local CPU-only machine.
    """

    num_features: int
    num_classes: int
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    mlp_dim: int = 256
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 20
    num_workers: int = 0
    device: str = "auto"
    mixed_precision: bool = True
    seed: int = 42
    early_stopping_patience: int | None = 5

    def __post_init__(self) -> None:
        if self.num_features < 1:
            raise ValueError("num_features must be at least 1")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if self.num_heads < 1 or self.num_layers < 1 or self.mlp_dim < 1:
            raise ValueError("num_heads, num_layers, and mlp_dim must be positive")
        if self.hidden_dim < 1 or self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.batch_size < 1 or self.epochs < 1 or self.num_workers < 0:
            raise ValueError("batch_size and epochs must be positive; num_workers non-negative")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive or None")


@dataclass
class TrainingHistory:
    """Training-monitoring values, not final experiment evaluation results."""

    train_loss: list[float]
    validation_loss: list[float]
    validation_accuracy: list[float]
    epochs_completed: int
    best_epoch: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NumericFeatureTokenizer(nn.Module):
    """Maps every numeric feature to one learned Transformer token."""

    def __init__(self, num_features: int, hidden_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_features, hidden_dim))
        self.bias = nn.Parameter(torch.empty(num_features, hidden_dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, features: Tensor) -> Tensor:
        return features.unsqueeze(-1) * self.weight + self.bias


class TabularTransformer(nn.Module):
    """A compact feature-token Transformer for numeric network-flow features."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.tokenizer = NumericFeatureTokenizer(config.num_features, config.hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.num_features + 1, config.hidden_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def encode_features(self, features: Tensor) -> Tensor:
        """Return the CLS embedding for a batch of numeric feature rows."""

        tokens = self.tokenizer(features)
        cls_tokens = self.cls_token.expand(features.shape[0], -1, -1)
        sequence = torch.cat((cls_tokens, tokens), dim=1)
        encoded = self.encoder(sequence + self.position_embedding)
        return encoded[:, 0]

    def forward(self, features: Tensor) -> Tensor:
        return self.classifier(self.encode_features(features))


class TabularTransformerClassifier:
    """Train, predict, checkpoint, and expand a tabular Transformer classifier.

    Labels passed to this class are external integer IDs from the data pipeline.
    Internally they are mapped to contiguous output columns, and ``predict``
    maps predictions back to the original IDs. New labels can be added before a
    later continual-learning task through :meth:`add_classes`.
    """

    def __init__(
        self,
        config: ModelConfig,
        class_ids: Sequence[int] | None = None,
    ) -> None:
        self.config = config
        initial_class_ids = tuple(range(config.num_classes)) if class_ids is None else class_ids
        self.class_ids = self._validate_class_ids(initial_class_ids)
        if len(self.class_ids) != config.num_classes:
            raise ValueError("class_ids length must match config.num_classes")

        self._set_seed(config.seed)
        self._device = self._resolve_device(config.device)
        self.network = TabularTransformer(config).to(self._device)

    @property
    def device(self) -> torch.device:
        return self._device

    @staticmethod
    def _validate_class_ids(class_ids: Sequence[int]) -> tuple[int, ...]:
        normalized = tuple(int(class_id) for class_id in class_ids)
        if len(set(normalized)) != len(normalized):
            raise ValueError("class_ids must be unique")
        return normalized

    @staticmethod
    def _set_seed(seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _resolve_device(requested_device: str) -> torch.device:
        if requested_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        device = torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return device

    def add_classes(self, new_class_ids: Sequence[int]) -> None:
        """Expand the classifier head while preserving existing output weights."""

        unseen_ids: list[int] = []
        for class_id in new_class_ids:
            normalized_id = int(class_id)
            if normalized_id not in self.class_ids and normalized_id not in unseen_ids:
                unseen_ids.append(normalized_id)
        if not unseen_ids:
            return

        old_head = self.network.classifier[-1]
        if not isinstance(old_head, nn.Linear):
            raise RuntimeError("classifier head must end in nn.Linear")

        expanded_head = nn.Linear(old_head.in_features, old_head.out_features + len(unseen_ids))
        expanded_head = expanded_head.to(self.device)
        nn.init.xavier_uniform_(expanded_head.weight)
        nn.init.zeros_(expanded_head.bias)
        with torch.no_grad():
            expanded_head.weight[: old_head.out_features].copy_(old_head.weight)
            expanded_head.bias[: old_head.out_features].copy_(old_head.bias)

        self.network.classifier[-1] = expanded_head
        self.class_ids = (*self.class_ids, *unseen_ids)
        self.config = replace(self.config, num_classes=len(self.class_ids))

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        replay: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> TrainingHistory:
        """Train on one task, optionally mixing a replay batch into the task data."""

        train_features = self._validate_features(X_train)
        train_labels = self._validate_labels(y_train, expected_rows=train_features.shape[0])

        if replay is not None:
            replay_features = self._validate_features(replay[0])
            replay_labels = self._validate_labels(replay[1], expected_rows=replay_features.shape[0])
            train_features = np.concatenate((train_features, replay_features), axis=0)
            train_labels = np.concatenate((train_labels, replay_labels), axis=0)

        self.add_classes(np.unique(train_labels).tolist())
        encoded_train_labels = self._encode_labels(train_labels)
        train_loader = self._make_loader(train_features, encoded_train_labels, shuffle=True)

        validation_data: tuple[Tensor, Tensor] | None = None
        if X_val is not None or y_val is not None:
            if X_val is None or y_val is None:
                raise ValueError("X_val and y_val must be supplied together")
            validation_features = self._validate_features(X_val)
            validation_labels = self._validate_labels(y_val, expected_rows=validation_features.shape[0])
            self.add_classes(np.unique(validation_labels).tolist())
            validation_data = (
                torch.as_tensor(validation_features, dtype=torch.float32),
                torch.as_tensor(self._encode_labels(validation_labels), dtype=torch.long),
            )

        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        amp_enabled = self.config.mixed_precision and self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        history = TrainingHistory([], [], [], epochs_completed=0, best_epoch=None)
        best_loss = float("inf")
        best_state: dict[str, Tensor] | None = None
        stale_epochs = 0

        for epoch in range(self.config.epochs):
            self.network.train()
            total_loss = 0.0
            total_rows = 0
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device, non_blocking=True)
                batch_labels = batch_labels.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    logits = self.network(batch_features)
                    loss = criterion(logits, batch_labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.detach().item() * batch_features.shape[0]
                total_rows += batch_features.shape[0]

            history.train_loss.append(total_loss / max(total_rows, 1))
            history.epochs_completed = epoch + 1

            if validation_data is None:
                continue

            validation_loss, validation_accuracy = self._validation_loss_and_accuracy(
                *validation_data,
                criterion=criterion,
            )
            history.validation_loss.append(validation_loss)
            history.validation_accuracy.append(validation_accuracy)

            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.network.state_dict().items()
                }
                history.best_epoch = epoch + 1
                stale_epochs = 0
            else:
                stale_epochs += 1
                if (
                    self.config.early_stopping_patience is not None
                    and stale_epochs >= self.config.early_stopping_patience
                ):
                    break

        if best_state is not None:
            self.network.load_state_dict(best_state)
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        class_indices = probabilities.argmax(axis=1)
        return np.asarray(self.class_ids, dtype=np.int64)[class_indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        features = self._validate_features(X)
        self.network.eval()
        batches: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, features.shape[0], self.config.batch_size):
                batch = torch.as_tensor(
                    features[start : start + self.config.batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                probabilities = torch.softmax(self.network(batch), dim=1)
                batches.append(probabilities.cpu().numpy())

        return np.concatenate(batches, axis=0) if batches else np.empty((0, len(self.class_ids)))

    def save(self, path: str | Path) -> None:
        """Save model weights, configuration, and output-column class mapping."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "class_ids": list(self.class_ids),
                "state_dict": self.network.state_dict(),
            },
            output_path,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str = "auto",
    ) -> "TabularTransformerClassifier":
        """Load a checkpoint produced by :meth:`save`."""

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")

        config = ModelConfig(**checkpoint["config"])
        if device != "auto":
            config = replace(config, device=device)
        classifier = cls(config, class_ids=checkpoint["class_ids"])
        classifier.network.load_state_dict(checkpoint["state_dict"])
        classifier.network.to(classifier.device)
        return classifier

    def _make_loader(self, features: np.ndarray, labels: np.ndarray, shuffle: bool) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(self.config.seed)
        dataset = TensorDataset(
            torch.as_tensor(features, dtype=torch.float32),
            torch.as_tensor(labels, dtype=torch.long),
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            generator=generator,
        )

    def _validation_loss_and_accuracy(
        self,
        features: Tensor,
        labels: Tensor,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        self.network.eval()
        total_loss = 0.0
        correct = 0
        total_rows = 0
        with torch.no_grad():
            for start in range(0, features.shape[0], self.config.batch_size):
                batch_features = features[start : start + self.config.batch_size].to(self.device)
                batch_labels = labels[start : start + self.config.batch_size].to(self.device)
                logits = self.network(batch_features)
                total_loss += criterion(logits, batch_labels).item() * batch_features.shape[0]
                correct += (logits.argmax(dim=1) == batch_labels).sum().item()
                total_rows += batch_features.shape[0]
        return total_loss / max(total_rows, 1), correct / max(total_rows, 1)

    def _encode_labels(self, labels: np.ndarray) -> np.ndarray:
        index_by_class_id = {class_id: index for index, class_id in enumerate(self.class_ids)}
        try:
            return np.asarray([index_by_class_id[int(label)] for label in labels], dtype=np.int64)
        except KeyError as error:
            raise ValueError(f"Label {error.args[0]} is not in class_ids") from error

    def _validate_features(self, features: np.ndarray) -> np.ndarray:
        array = np.asarray(features, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("features must have shape (n_samples, n_features)")
        if array.shape[1] != self.config.num_features:
            raise ValueError(
                f"Expected {self.config.num_features} features, received {array.shape[1]}"
            )
        if array.shape[0] == 0:
            raise ValueError("features must contain at least one row")
        if not np.isfinite(array).all():
            raise ValueError("features must not contain NaN or infinite values")
        return np.ascontiguousarray(array)

    @staticmethod
    def _validate_labels(labels: np.ndarray, expected_rows: int) -> np.ndarray:
        array = np.asarray(labels)
        if array.ndim != 1 or array.shape[0] != expected_rows:
            raise ValueError("labels must be one-dimensional and match feature rows")
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError("labels must use integer class IDs")
        return np.ascontiguousarray(array.astype(np.int64, copy=False))
