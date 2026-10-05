# Local artifact demonstration

The Streamlit interface is a bounded local demonstration over existing private
artifacts. It does not train, mutate checkpoints, rewrite replay data, or claim
that playback metrics were produced by the UI.

## Set up

Use Python 3.13 for the simplest PyTorch compatibility and to match the
recorded Kaggle environment:

```bash
python3.13 -m venv .venv-demo
source .venv-demo/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-demo.txt
python -m pip install 'torch==2.11.0+cpu' \
  --index-url https://download.pytorch.org/whl/cpu
```

Copy and edit `configs/demo_artifacts.example.json` only if your private
artifact locations differ. Paths are relative to its `base_dir`; absolute
paths are also accepted. Keep correct SHA-256 values so a mismatched
checkpoint, preprocessor, or result ZIP fails closed.

Launch from the repository root:

```bash
.venv-demo/bin/streamlit run streamlit_app.py -- \
  --config configs/demo_artifacts.example.json
```

## Required private artifacts

The example configuration expects:

- static flow inference: clean eight-class `best.pt` plus the matching
  `preprocessing_static.json`;
- poisoning/quarantine: five-class Task 1 `best.pt`, its matching
  `preprocessing_continual.json`, `replay_buffer.npz`, and saved Task 1
  `novelty.json`;
- playback and gate calibration: the original `data/task2/results.zip` with
  SHA-256
  `0c73a44cca06cb92f7c21c98652808f739df67236d58d790f5cf4b93027d832e`.

The checkpoint, replay buffer, uploaded flow file, result ZIP, and generated
SQLite queue remain outside Git. The default queue is
`.local/nids-demo/quarantine.sqlite3`, an ignored path.

## Flow input contract

Upload CSV or Parquet containing at most 2,000 rows. It must contain exactly
the configured 54 raw numeric feature columns in the recorded order. An
optional broad-label column may be selected for metrics and is excluded before
schema validation. Inputs with missing, extra, duplicate, reordered, or
non-numeric feature columns fail before inference.

The default is the clean eight-class static model and static preprocessing.
The Task 1 model always uses continual preprocessing. Its weak saved
max-confidence novelty cutoff is available only for the five-class teacher;
the application never applies it to the eight-class static model.

## Teacher and quarantine demonstration

The same deterministic demo can be run without the UI:

```bash
.venv-demo/bin/python scripts/demo_teacher_quarantine.py \
  --config configs/demo_artifacts.example.json
```

This performs CPU inference over the 500 supplied replay rows, changes 80 of
400 eligible old-attack labels to Benign with seed 42, applies the saved
Task-1-teacher label-consistency gate, and queues rejected rows. It prints an
experiment-only ground-truth audit. The persistent queue stores original ID,
supplied label, score, threshold, model identity, row payload, status, and
decision history—but never the simulator poison mask. Reviewer release is an
audited queue decision only and cannot trigger training.

## Saved comparison playback

The continual view reads only named JSON evidence from the ZIP, after checking
its hash and rejecting unsafe member paths, symlinks, encrypted members, or
oversized JSON. Bundled source and checkpoints are not imported or executed.
The compact evidence index is
`docs/evidence/task2_targeted_summary.json`; the original ZIP remains the
authoritative artifact and is not modified.
