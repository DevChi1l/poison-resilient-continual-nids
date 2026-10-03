"""Plot saved classification counts without model inference or retraining."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def confusion_plot_data(record: dict) -> dict:
    """Validate saved counts and derive row-normalized confusion and class scores."""

    metrics = record.get("metrics", record)
    ids = np.asarray(metrics["class_ids"])
    counts = np.asarray(metrics["confusion_matrix"])
    if (ids.ndim != 1 or ids.size == 0 or
            not np.issubdtype(ids.dtype, np.integer) or
            np.unique(ids).size != ids.size or
            counts.shape != (ids.size, ids.size) or
            not np.issubdtype(counts.dtype, np.integer) or
            np.any(counts < 0)):
        raise ValueError("Invalid class IDs or nonnegative integer confusion counts")
    support = counts.sum(axis=1)
    predicted = counts.sum(axis=0)
    correct = np.diag(counts)
    normalized = np.divide(counts, support[:, None],
                           out=np.zeros_like(counts, dtype=np.float64),
                           where=support[:, None] != 0)
    precision = np.divide(correct, predicted, out=np.zeros(ids.size), where=predicted != 0)
    recall = np.divide(correct, support, out=np.zeros(ids.size), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(ids.size), where=precision + recall != 0)
    names = [metrics.get("per_class", {}).get(str(int(class_id)), {}).get("name", str(class_id))
             for class_id in ids]
    return {"class_ids": ids.tolist(), "class_names": names, "counts": counts,
            "normalized": normalized, "support": support.tolist(),
            "recall": recall, "f1": f1}


def save_result_plots(metrics_path: str | Path, output_dir: str | Path, *,
                      prefix: str = "test") -> dict[str, str]:
    """Save separate raw-count, annotated row-normalized, and recall/F1 PNGs.

    The supplied JSON is read-only. Existing files are not overwritten.
    This utility never loads a checkpoint or opens a test dataset.
    """

    import matplotlib.pyplot as plt

    source = Path(metrics_path)
    data = confusion_plot_data(json.loads(source.read_text(encoding="utf-8")))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    names = data["class_names"]
    count = len(names)
    if not prefix or not prefix.replace("_", "").replace("-", "").isalnum():
        raise ValueError("prefix must be a nonempty simple name")
    outputs = {
        "raw_counts": destination / f"{prefix}_confusion_counts.png",
        "row_normalized": destination / f"{prefix}_confusion_row_normalized.png",
        "per_class": destination / f"{prefix}_per_class_recall_f1.png",
    }
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("Plot output already exists; use a new folder or prefix")

    for key, values, title, colorbar_label in (
        ("raw_counts", data["counts"], "Raw confusion counts", "Rows"),
        ("row_normalized", data["normalized"], "Row-normalized confusion", "Fraction of true class"),
    ):
        fig, ax = plt.subplots(figsize=(max(7, count), max(6, count * 0.8)))
        image = ax.imshow(values, cmap="Blues", vmin=0,
                          vmax=1 if key == "row_normalized" else None)
        ax.set_xticks(range(count), names, rotation=45, ha="right")
        ax.set_yticks(range(count), names)
        ax.set(xlabel="Predicted class", ylabel="True class", title=title)
        fig.colorbar(image, ax=ax, label=colorbar_label)
        if key == "row_normalized":
            for row in range(count):
                for column in range(count):
                    value = values[row, column]
                    ax.text(column, row, f"{value:.1%}", ha="center", va="center",
                            color="white" if value > 0.55 else "black", fontsize=8)
        fig.tight_layout()
        fig.savefig(outputs[key], dpi=160)
        plt.close(fig)

    positions = np.arange(count)
    fig, ax = plt.subplots(figsize=(max(8, count * 1.1), 4.5))
    ax.bar(positions - 0.18, data["recall"], width=0.36, label="Recall")
    ax.bar(positions + 0.18, data["f1"], width=0.36, label="F1")
    ax.set_xticks(positions, names, rotation=45, ha="right")
    ax.set(xlabel="True class", ylabel="Score", ylim=(0, 1.05),
           title="Per-class recall and F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outputs["per_class"], dpi=160)
    plt.close(fig)
    return {name: str(path) for name, path in outputs.items()}
