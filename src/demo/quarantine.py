"""Persistent local review queue and deterministic replay-gate demo."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Iterator

import numpy as np

from src.mitigation import apply_label_consistency_gate
from src.poisoning import apply_label_flip


FINAL_DECISIONS = ("rejected", "released")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class QuarantineStore:
    """SQLite queue with immutable decision-history rows."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS review_items (
                    id INTEGER PRIMARY KEY,
                    event_key TEXT NOT NULL UNIQUE,
                    original_id TEXT NOT NULL,
                    supplied_label INTEGER NOT NULL,
                    score REAL NOT NULL CHECK(score >= 0 AND score <= 1),
                    threshold REAL NOT NULL CHECK(threshold >= 0 AND threshold <= 1),
                    model_identity TEXT NOT NULL,
                    row_payload_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK(status IN ('pending', 'rejected', 'released')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS decision_history (
                    id INTEGER PRIMARY KEY,
                    item_id INTEGER NOT NULL REFERENCES review_items(id),
                    decision TEXT NOT NULL
                        CHECK(decision IN ('quarantined', 'rejected', 'released')),
                    reviewer TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    decided_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _event_key(record: dict[str, Any]) -> str:
        identity = "|".join(
            (
                str(record["original_id"]),
                str(record["supplied_label"]),
                format(float(record["score"]), ".17g"),
                format(float(record["threshold"]), ".17g"),
                str(record["model_identity"]),
            )
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def enqueue(self, records: Iterable[dict[str, Any]]) -> int:
        """Insert new suspicious rows idempotently and audit initial quarantine."""

        inserted = 0
        now = _utc_now()
        with self._connect() as connection:
            for record in records:
                score = float(record["score"])
                threshold = float(record["threshold"])
                if not 0 <= score <= 1 or not 0 <= threshold <= 1:
                    raise ValueError("score and threshold must be in [0, 1]")
                event_key = self._event_key(record)
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO review_items
                    (event_key, original_id, supplied_label, score, threshold,
                     model_identity, row_payload_json, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                    """,
                    (
                        event_key,
                        str(record["original_id"]),
                        int(record["supplied_label"]),
                        score,
                        threshold,
                        str(record["model_identity"]),
                        json.dumps(record.get("row_payload", {}), sort_keys=True),
                        now,
                        now,
                    ),
                )
                if cursor.rowcount:
                    item_id = int(cursor.lastrowid)
                    connection.execute(
                        """
                        INSERT INTO decision_history
                        (item_id, decision, reviewer, reason, decided_at)
                        VALUES (?, 'quarantined', 'system-gate',
                                'score exceeded the configured gate threshold', ?)
                        """,
                        (item_id, now),
                    )
                    inserted += 1
        return inserted

    def decide(self, item_id: int, decision: str, reviewer: str, reason: str) -> None:
        """Finalize one pending item; release is review only and never trains."""

        if decision not in FINAL_DECISIONS:
            raise ValueError("decision must be rejected or released")
        reviewer = reviewer.strip()
        reason = reason.strip()
        if not reviewer or not reason:
            raise ValueError("reviewer and reason are required")
        now = _utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE review_items SET status = ?, updated_at = ?
                WHERE id = ? AND status = 'pending'
                """,
                (decision, now, int(item_id)),
            )
            if cursor.rowcount != 1:
                raise ValueError("item is absent or already finalized")
            connection.execute(
                """
                INSERT INTO decision_history
                (item_id, decision, reviewer, reason, decided_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (int(item_id), decision, reviewer, reason, now),
            )

    def items(self, *, status: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM review_items"
        arguments: tuple[Any, ...] = ()
        if status is not None:
            if status not in ("pending", *FINAL_DECISIONS):
                raise ValueError("unknown queue status")
            query += " WHERE status = ?"
            arguments = (status,)
        query += " ORDER BY id DESC"
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(query, arguments)]

    def history(self, item_id: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM decision_history WHERE item_id = ? ORDER BY id",
                    (int(item_id),),
                )
            ]


@dataclass(frozen=True)
class ReplayBuffer:
    features: np.ndarray
    labels: np.ndarray
    original_ids: np.ndarray


@dataclass(frozen=True)
class GateDemoResult:
    poisoned_labels: np.ndarray
    changed_indices: np.ndarray
    retained_mask: np.ndarray
    scores: np.ndarray
    threshold: float
    audit: dict[str, Any]

    def queue_records(self, replay: ReplayBuffer, model_identity: str) -> list[dict[str, Any]]:
        """Create operational queue rows without simulator ground-truth flags."""

        return [
            {
                "original_id": int(replay.original_ids[index]),
                "supplied_label": int(self.poisoned_labels[index]),
                "score": float(self.scores[index]),
                "threshold": self.threshold,
                "model_identity": model_identity,
                "row_payload": {"features": replay.features[index].astype(float).tolist()},
            }
            for index in np.flatnonzero(~self.retained_mask)
        ]


def load_replay_buffer(path: str | Path) -> ReplayBuffer:
    """Load only numeric arrays from the supplied replay NPZ."""

    with np.load(Path(path), allow_pickle=False) as arrays:
        required = {"features", "labels", "original_row_ids"}
        if set(arrays.files) != required:
            raise ValueError(f"Replay buffer arrays must be exactly {sorted(required)}")
        features = np.asarray(arrays["features"], dtype=np.float32)
        labels = np.asarray(arrays["labels"])
        original_ids = np.asarray(arrays["original_row_ids"])
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("Replay features must be a finite matrix")
    if labels.shape != (len(features),) or original_ids.shape != (len(features),):
        raise ValueError("Replay labels and IDs must align with features")
    if not np.issubdtype(labels.dtype, np.integer) or not np.issubdtype(
        original_ids.dtype, np.integer
    ):
        raise ValueError("Replay labels and IDs must be integer arrays")
    if np.unique(original_ids).size != original_ids.size:
        raise ValueError("Replay original IDs must be unique")
    return ReplayBuffer(features.copy(), labels.astype(np.int64), original_ids.astype(np.int64))


def evaluate_targeted_gate(
    replay: ReplayBuffer,
    teacher_probabilities: np.ndarray,
    *,
    threshold: float,
    seed: int = 42,
    rate: float = 0.2,
) -> GateDemoResult:
    """Reproduce targeted attack and gate with truth reserved for audit output."""

    attack = apply_label_flip(
        replay.labels,
        rate=rate,
        seed=seed,
        mode="targeted",
        source_class_ids=(1, 2, 3, 4),
        target_class_id=0,
        allowed_class_ids=(0, 1, 2, 3, 4),
        original_row_indices=replay.original_ids,
    )
    gate = apply_label_consistency_gate(
        replay.features,
        attack.poisoned_labels,
        teacher_probabilities,
        class_ids=(0, 1, 2, 3, 4),
        threshold=threshold,
    )
    changed = np.zeros(len(replay.labels), dtype=bool)
    changed[attack.changed_indices] = True
    rejected = ~gate.retained_mask
    audit = {
        "experiment_audit_only": True,
        "total_candidates": len(replay.labels),
        "poisoned_candidates": int(changed.sum()),
        "poison_rejected": int(np.count_nonzero(changed & rejected)),
        "clean_candidates": int((~changed).sum()),
        "clean_false_rejected": int(np.count_nonzero((~changed) & rejected)),
        "retained_total": int(gate.retained_mask.sum()),
        "changed_original_ids": replay.original_ids[changed].astype(int).tolist(),
    }
    return GateDemoResult(
        attack.poisoned_labels.copy(),
        attack.changed_indices.copy(),
        gate.retained_mask.copy(),
        gate.scores.copy(),
        float(threshold),
        audit,
    )
