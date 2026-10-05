# Five-minute faculty demonstration

This is artifact-backed playback and bounded CPU inference. It does not train
or update any model. Before the live session, choose a new ignored queue path;
do not delete an older queue merely to make its count return to zero.

## Before the timer

From the repository root:

```bash
.venv-demo/bin/python scripts/demo_preflight.py \
  --config configs/demo_artifacts.example.json \
  --database .local/nids-demo-live/quarantine.sqlite3

.venv-demo/bin/python scripts/export_demo_sample.py \
  --config configs/demo_artifacts.example.json \
  --output-dir uploads/live-demo \
  --rows-per-class 5 --seed 42

.venv-demo/bin/streamlit run streamlit_app.py -- \
  --config configs/demo_artifacts.example.json \
  --database .local/nids-demo-live/quarantine.sqlite3
```

Preflight should end with `PREFLIGHT PASS`. On a fresh output directory the
export reports 40 rows, five from every broad class. If either path already
exists, choose a new ignored path instead of erasing audit history. Open the
printed local URL.

## 0:00–1:20 — real flow prediction

1. Keep **Clean static Transformer (8 classes)** selected.
2. Upload `uploads/live-demo/heldout_flows.csv`.
3. Leave **ClassLabel** as the optional label column.
4. Click **Run bounded CPU inference**.

Expected: `Predicted 40 rows ... on CPU`, probability columns, and labelled
metrics. The 2026-10-05 rehearsal of the exact hash-recorded sample showed
accuracy `0.7250`, macro-F1 `0.71395`, and Benign FPR `0.0000`.

Say: “These are real predictions from the saved clean checkpoint and its
matching static preprocessor. The sample was selected only by saved test-split
membership, class, and seed—not by prediction success. Five rows per class is
a UI rehearsal, so its metrics are not our full-test result.”

Also point out that the eight-class model never receives the Task 1 novelty
threshold. That threshold belongs only to the five-class teacher and was weak:
its saved held-out unknown recall was about `4.6%`.

## 1:20–2:50 — targeted corruption and quarantine

1. Open **Poisoning / quarantine**.
2. Click **Run deterministic teacher/gate demo**.
3. Enter reviewer `faculty-demo` and a short reason; reject one row.
4. Review a second pending row and release it with reason `reviewed for
   external handling only`; emphasize that release does not start training.

Expected on a fresh queue: 122 suspicious rows added. The experiment-only
audit shows 80/80 poisoned rows rejected, but also 42/420 clean rows rejected,
with 378 retained.

Say: “The gate catches all simulated targeted flips in this replay buffer, but
the 42 clean rejections are false alarms—a 10% clean false-reject rate. Ground-
truth poison masks are available only in the audit panel, not to the operational
queue. Human decisions are immutable audit events.”

## 2:50–4:30 — saved continual-learning comparison

1. Open **Continual comparison**.
2. Point to the Task 1-before metrics and the three-arm table.
3. Select **Targeted poison**, then **Targeted poison + gate**.
4. Show one per-class table and row-normalized confusion matrix.

State the saved clean/poisoned/filtered values:

- accuracy: `0.60100 / 0.44914 / 0.60772`;
- macro-F1: `0.56646 / 0.53793 / 0.55785`;
- Benign FPR: `0.49891 / 0.68860 / 0.46663`.

Say: “All three arms completed six epochs, selected epoch one, used 1,656
steps, and started from identical expanded weights. Task 1 accuracy before
Task 2 was `0.99257`; accuracy forgetting was `0.39227`, `0.54735`, and
`0.38549` for clean, poisoned, and filtered arms. The gate improves aggregate
accuracy over poisoning, but harms DoS recall (`0.96059` to `0.60729`) and
increases old-attack-to-Benign errors (3,641 to 6,856).”

## 4:30–5:00 — honest conclusion and persistence

Say: “This is single-seed evidence for one targeted replay-label attack. It
shows a mixed mitigation result, not broad poison robustness. The confidence-
threshold novelty detector is an explicitly weak baseline. The next review
needs stronger novelty and mitigation experiments, more seeds, and provenance-
aware evaluation.”

If asked to prove persistence, stop Streamlit with `Ctrl+C`, rerun the same
launch command, open **Poisoning / quarantine**, and select `rejected` and
`released`. The decisions and reasons remain because they are stored in the
separate SQLite queue.
