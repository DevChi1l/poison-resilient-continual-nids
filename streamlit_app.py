"""Artifact-backed local Review-2 demonstration.

Launch from the repository root with:
    streamlit run streamlit_app.py -- --config configs/demo_artifacts.example.json
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.demo.artifacts import (
    ArtifactConfigurationError,
    DemoArtifactConfig,
    ModelArtifact,
    file_sha256,
    load_demo_config,
    load_fitted_preprocessor,
    load_task1_novelty_threshold,
)
from src.demo.inference import (
    UploadSchemaError,
    load_verified_model,
    prepare_upload,
    read_uploaded_table,
    run_inference,
)
from src.demo.quarantine import (
    QuarantineStore,
    evaluate_targeted_gate,
    load_replay_buffer,
)
from src.demo.results import (
    ARM_LABELS,
    ARM_ORDER,
    arm_summary_rows,
    load_task2_playback,
    normalized_confusion,
)


CLASS_NAMES = (
    "Benign",
    "DDoS",
    "DoS",
    "Botnet",
    "Bruteforce",
    "Infiltration",
    "Webattack",
    "Portscan",
)


def _config_argument() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config")
    arguments, _ = parser.parse_known_args()
    return arguments.config or os.environ.get(
        "NIDS_DEMO_CONFIG", "configs/demo_artifacts.example.json"
    )


@st.cache_resource(show_spinner=False)
def _cached_model(
    key: str,
    checkpoint: str,
    checkpoint_hash: str,
    preprocessor: str,
    preprocessor_hash: str,
    scope: str,
    identity: str,
    display_name: str,
    class_names: tuple[str, ...],
):
    spec = ModelArtifact(
        key=key,
        display_name=display_name,
        identity=identity,
        checkpoint=Path(checkpoint),
        checkpoint_sha256=checkpoint_hash,
        preprocessor=Path(preprocessor),
        preprocessor_sha256=preprocessor_hash,
        preprocessing_scope=scope,
        class_names={index: name for index, name in enumerate(class_names)},
    )
    return load_verified_model(spec)


def _load_model(spec: ModelArtifact):
    return _cached_model(
        spec.key,
        str(spec.checkpoint),
        spec.checkpoint_sha256,
        str(spec.preprocessor),
        spec.preprocessor_sha256,
        spec.preprocessing_scope,
        spec.identity,
        spec.display_name,
        tuple(spec.class_names.values()),
    )


@st.cache_data(show_spinner=False)
def _cached_playback(path: str, expected_hash: str):
    return load_task2_playback(path, expected_hash)


def _playback(config: DemoArtifactConfig):
    return _cached_playback(str(config.task2_results_zip), config.task2_results_sha256)


def _metric_cards(metrics: dict[str, Any]) -> None:
    columns = st.columns(3)
    columns[0].metric("Accuracy", f"{metrics['accuracy']:.4f}")
    columns[1].metric("Macro-F1", f"{metrics['macro_f1']:.4f}")
    fpr = metrics.get("benign_false_positive", {}).get("rate")
    columns[2].metric("Benign FPR", "n/a" if fpr is None else f"{fpr:.4f}")


def render_flow_prediction(config: DemoArtifactConfig) -> None:
    st.header("Flow prediction")
    st.write(
        "CPU-bounded inference using a configured checkpoint, its matching fit state, "
        "and its recorded class mapping. Nothing is retrained or uploaded elsewhere."
    )
    model_keys = list(config.models)
    default_index = model_keys.index(config.default_model)
    selected_key = st.selectbox(
        "Configured model",
        model_keys,
        index=default_index,
        format_func=lambda key: config.models[key].display_name,
    )
    spec = config.models[selected_key]
    st.info(
        f"**Model identity:** `{spec.identity}`  \n"
        f"**Preprocessing scope:** `{spec.preprocessing_scope}`  \n"
        f"**Class mapping:** "
        + ", ".join(f"{class_id}={name}" for class_id, name in spec.class_names.items())
    )
    if len(spec.class_ids) == 8:
        st.caption(
            "The Task 1 novelty threshold is not valid for this eight-class model and is not applied."
        )
    else:
        st.warning(
            "Optional UNKNOWN flags use the saved weak experimental max-confidence "
            "baseline. Its recorded held-out unknown recall was low; this is not a robust detector."
        )

    uploaded = st.file_uploader("Flow table", type=("csv", "parquet", "pq"))
    if uploaded is None:
        st.caption(
            f"Expected: exactly 54 raw numeric feature columns in recorded order, plus an "
            f"optional broad-label column; maximum {config.max_inference_rows:,} rows."
        )
        return
    try:
        frame = read_uploaded_table(
            uploaded.getvalue(), uploaded.name, max_rows=config.max_inference_rows
        )
    except UploadSchemaError as error:
        st.error(str(error))
        return

    choices: list[str | None] = [None, *map(str, frame.columns)]
    preferred = "ClassLabel" if "ClassLabel" in frame.columns else None
    label_column = st.selectbox(
        "Optional broad-label column",
        choices,
        index=choices.index(preferred),
        format_func=lambda value: "No labels" if value is None else value,
    )
    if not st.button("Run bounded CPU inference", type="primary"):
        return
    try:
        with st.spinner("Validating schema and loading the verified checkpoint on CPU..."):
            preprocessor = load_fitted_preprocessor(spec)
            prepared = prepare_upload(
                frame,
                spec,
                preprocessor,
                max_rows=config.max_inference_rows,
                label_column=label_column,
            )
            classifier, _ = _load_model(spec)
            result = run_inference(classifier, prepared, spec)
    except (ArtifactConfigurationError, UploadSchemaError, ImportError, RuntimeError) as error:
        st.error(str(error))
        return

    output = pd.DataFrame(
        {
            "row": np.arange(len(result.predictions)),
            "predicted_id": result.predictions,
            "predicted_class": [spec.class_names[int(value)] for value in result.predictions],
            "confidence": result.probabilities.max(axis=1),
        }
    )
    for column, name in enumerate(spec.class_names.values()):
        output[f"p({name})"] = result.probabilities[:, column]
    if len(spec.class_ids) == 5:
        try:
            threshold = load_task1_novelty_threshold(spec)
            output["UNKNOWN (weak baseline)"] = output["confidence"] < threshold
            st.caption(f"Saved Task 1 max-confidence cutoff: {threshold:.10f}.")
        except ArtifactConfigurationError as error:
            st.warning(str(error))
    st.success(f"Predicted {len(output):,} rows with `{spec.identity}` on CPU.")
    st.dataframe(output, width="stretch", hide_index=True)
    if result.metrics is not None:
        st.subheader("Optional labelled metrics")
        _metric_cards(result.metrics)
        st.dataframe(
            pd.DataFrame(result.metrics["per_class"]).T,
            width="stretch",
        )


def _queue_table(items: list[dict[str, Any]]) -> pd.DataFrame:
    fields = (
        "id",
        "original_id",
        "supplied_label",
        "score",
        "threshold",
        "model_identity",
        "status",
        "created_at",
        "updated_at",
    )
    return pd.DataFrame([{field: item[field] for field in fields} for item in items])


def render_poisoning_quarantine(config: DemoArtifactConfig) -> None:
    st.header("Poisoning and quarantine review")
    st.write(
        "The live demo deterministically changes 20% of eligible old-attack replay labels "
        "to Benign (seed 42), scores supplied labels with the real frozen Task 1 teacher, "
        "and sends rows above the saved gate threshold to a local SQLite review queue."
    )
    teacher_spec = config.models[config.task1_teacher]
    st.info(
        f"**Teacher:** `{teacher_spec.identity}`  \n"
        f"**Queue:** `{config.quarantine_db}`  \n"
        "A reviewer release only changes queue status. It does **not** start or authorize training."
    )
    try:
        playback = _playback(config)
        threshold = float(playback.calibration["threshold"])
        gate_saved = playback.gate_audit["audit"]
        st.caption(
            f"Saved label-inconsistency gate threshold: {threshold:.12f}; calibrated on clean "
            "Task 1 validation. This is separate from the novelty cutoff."
        )
    except (ArtifactConfigurationError, KeyError, TypeError, ValueError) as error:
        st.error(f"Saved gate evidence unavailable: {error}")
        playback = None
        threshold = None
        gate_saved = None

    try:
        store = QuarantineStore(config.quarantine_db)
    except (OSError, sqlite3.Error) as error:
        st.error(f"Could not open local review queue: {error}")
        return

    if st.button("Run deterministic teacher/gate demo", type="primary", disabled=threshold is None):
        try:
            with st.spinner("Running Task 1 teacher inference over the 500-row replay buffer..."):
                teacher, _ = _load_model(teacher_spec)
                replay = load_replay_buffer(config.replay_buffer)
                probabilities = teacher.predict_proba(replay.features)
                result = evaluate_targeted_gate(
                    replay,
                    probabilities,
                    threshold=float(threshold),
                    seed=42,
                    rate=0.2,
                )
                identity = (
                    f"{teacher_spec.identity}:{file_sha256(teacher_spec.checkpoint)[:12]}"
                )
                inserted = store.enqueue(result.queue_records(replay, identity))
            st.success(
                f"Gate completed. Added {inserted} new suspicious rows; repeated runs are idempotent."
            )
            with st.expander("Experiment audit only: simulator ground-truth mask", expanded=True):
                st.json(result.audit)
        except (ArtifactConfigurationError, ImportError, RuntimeError, ValueError) as error:
            st.error(str(error))

    status_filter = st.selectbox("Queue status", ("pending", "rejected", "released", "all"))
    items = store.items(status=None if status_filter == "all" else status_filter)
    st.dataframe(_queue_table(items), width="stretch", hide_index=True)
    if items:
        item_by_id = {int(item["id"]): item for item in items}
        item_id = st.selectbox("Review item", tuple(item_by_id))
        reviewer = st.text_input("Reviewer name")
        reason = st.text_input("Decision reason")
        left, right = st.columns(2)
        if left.button("Reject suspicious row"):
            try:
                store.decide(item_id, "rejected", reviewer, reason)
                st.success("Decision recorded as rejected.")
                st.rerun()
            except ValueError as error:
                st.error(str(error))
        if right.button("Release for external handling (no training)"):
            try:
                store.decide(item_id, "released", reviewer, reason)
                st.success("Decision recorded as released. No training was triggered.")
                st.rerun()
            except ValueError as error:
                st.error(str(error))
        st.subheader("Decision history")
        st.dataframe(pd.DataFrame(store.history(item_id)), width="stretch")

    if gate_saved is not None:
        with st.expander("Saved Task 2 experiment gate audit", expanded=False):
            st.warning(
                "Ground-truth poison masks are shown only in this saved experiment audit. "
                "They are not stored in or exposed to the operational review queue."
            )
            st.json(gate_saved)


def render_continual_comparison(config: DemoArtifactConfig) -> None:
    st.header("Continual comparison — saved experiment playback")
    st.warning(
        "This view reads verified artifacts from the supplied Task 2 ZIP. The UI did not "
        "perform, repeat, or select any training run."
    )
    try:
        playback = _playback(config)
    except ArtifactConfigurationError as error:
        st.error(str(error))
        return
    comparison = playback.comparison
    before = comparison["task1_before"]
    st.caption(
        f"Archive `{config.task2_results_zip.name}`; SHA-256 "
        f"`{config.task2_results_sha256}`; run `{playback.run_prefix.rstrip('/')}`."
    )
    st.subheader("Task 1 before Task 2")
    columns = st.columns(4)
    columns[0].metric("Accuracy", f"{before['accuracy']:.6f}")
    columns[1].metric("Macro-F1", f"{before['macro_f1']:.6f}")
    columns[2].metric("Benign FPR", f"{before['benign_fpr']:.6f}")
    columns[3].metric(
        "Old attack→Benign", f"{before['old_attack_to_benign']['rate']:.6f}"
    )

    st.subheader("Three-arm Task 2 comparison")
    summary = pd.DataFrame(arm_summary_rows(playback))
    st.dataframe(summary, width="stretch", hide_index=True)
    st.caption(
        "All arms completed 6 epochs, selected epoch 1, ran 1,656 optimizer steps, and "
        f"started from the same expanded weights `{comparison['same_expanded_initial_sha256']}`."
    )

    selected = st.selectbox(
        "Inspect saved arm",
        ARM_ORDER,
        format_func=lambda arm: ARM_LABELS[arm],
    )
    metrics = playback.combined_metrics[selected]
    _metric_cards(metrics)
    per_class = pd.DataFrame(metrics["per_class"]).T
    per_class.index.name = "class_id"
    st.subheader("Saved per-class metrics")
    st.dataframe(per_class, width="stretch")
    st.subheader("Saved combined-test confusion matrix (row-normalized)")
    labels = [metrics["per_class"][str(index)]["name"] for index in metrics["class_ids"]]
    normalized = pd.DataFrame(
        normalized_confusion(metrics),
        index=pd.Index(labels, name="actual"),
        columns=pd.Index(labels, name="predicted"),
    )
    st.dataframe(normalized.style.format("{:.3f}"), width="stretch")

    st.subheader("Gate trade-off")
    audit = playback.gate_audit["audit"]
    columns = st.columns(4)
    columns[0].metric("Poison rejected", f"{audit['poison_rejected']}/{audit['poisoned_candidates']}")
    columns[1].metric("Clean rejected", f"{audit['clean_false_rejected']}/{audit['clean_candidates']}")
    columns[2].metric("Rows retained", str(audit["retained_total"]))
    columns[3].metric("Clean false-reject rate", f"{audit['clean_false_rejection_rate']:.1%}")
    recall_rows = []
    for arm in ARM_ORDER:
        arm_metrics = playback.combined_metrics[arm]
        details = comparison["arms"][arm]["metrics"]
        recall_rows.append(
            {
                "Arm": ARM_LABELS[arm],
                "DoS recall": arm_metrics["per_class"]["2"]["recall"],
                "Old attack→Benign count": details["old_attack_to_benign"]["count"],
                "Old attack→Benign rate": details["old_attack_to_benign"]["rate"],
            }
        )
    st.dataframe(pd.DataFrame(recall_rows), width="stretch", hide_index=True)
    st.error(
        "Observed limitation: filtering improved combined accuracy and Benign FPR over the "
        "poisoned arm, but harmed DoS recall and increased old-attack→Benign errors. This "
        "single-seed result does not establish broad poisoning robustness."
    )


def main() -> None:
    st.set_page_config(page_title="Poison-Resilient Continual NIDS", layout="wide")
    st.title("Poison-Resilient / Novelty-Aware Continual NIDS")
    st.caption("Local Review-2 artifact demonstration — research prototype, not a production IDS")
    try:
        config = load_demo_config(_config_argument())
    except ArtifactConfigurationError as error:
        st.error(str(error))
        st.stop()
    view = st.sidebar.radio(
        "View",
        ("Flow prediction", "Poisoning / quarantine", "Continual comparison"),
    )
    st.sidebar.caption(f"Config: `{config.source_path}`")
    if view == "Flow prediction":
        render_flow_prediction(config)
    elif view == "Poisoning / quarantine":
        render_poisoning_quarantine(config)
    else:
        render_continual_comparison(config)


if __name__ == "__main__":
    main()
