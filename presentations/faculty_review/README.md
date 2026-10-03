# Faculty review presentation

`faculty_review.pptx` is the editable 16:9 faculty deck (15 slides).
`preview.png` is the slide overview. `generate.mjs` owns the layout;
`content.json` owns slide copy, displayed results, evidence notes and all
future figure/metrics paths. No training is invoked by this directory.

## Update future results

Keep raw run JSONs/figures in `inputs/` here (ignored by Git), or use absolute
paths outside the repository. Keep datasets and checkpoints outside Git.
Only fill slots after inspecting the actual completed run and its manifest.
Do not substitute epoch validation accuracy for final test performance.

Edit `result_slots` in `content.json`:

| Slot key | Slide | JSON path | Figure path |
| --- | --- | --- | --- |
| `learning_curves` | 12 | `inputs/full_clean/history.json` | `inputs/full_clean/training_curves.png` |
| `confusion_matrix` | 12 | `inputs/full_clean/test_metrics.json` | `inputs/full_clean/confusion_matrix.png` |
| `per_class_metrics` | 13 | `inputs/full_clean/test_metrics.json` | Not needed: native table |
| `continual_forgetting` | 14 | `inputs/continual/summary.json` | `inputs/continual/forgetting.png` |
| `poisoning_comparisons` | 14 | `inputs/poisoning/comparison.json` | `inputs/poisoning/comparison.png` |

For each ready slot set `status` to `VERIFIED`, `metrics_json_path` to the
JSON path, `image_path` to the PNG path (except the per-class table), and edit
`caption` to include the correct units and experiment scope. Relative paths
resolve beside `content.json`. Leave unrun slots `PENDING` with null paths.
The image slots embed supplied figures; they do not invent plots from metrics.
Their JSON paths are provenance pointers. Labels in supplied figures must be
large enough to read in a half-slide area; use true rows/predicted columns
and identify counts versus normalized percentages for the confusion matrix.

For `per_class_metrics`, the generator reads the existing `test_metrics.json`
schema: `per_class["0".."7"]` with `name`, `precision`, `recall`, `f1`, `support`.
It populates all eight rows automatically (P/R/F1 are fractions, 3 decimals).
Alternatively set its `table.headers` and `table.rows` directly in content.json
and leave `metrics_json_path` null. Do not use missing/null values as zeros.

Also edit the corresponding slide's `status`, `callout`, `notes` and `sources`
in the same content file: slide indices 11–13 are zero-based. Replace the
slide-13 callout with macro-F1, balanced accuracy and Benign FPR from the
completed clean test JSON. Update the cover and slide 11 progress statement
when training finishes. Never relabel the bounded-subset findings as full-data
results. Continual/poisoning slots require separate future experiments.

## Reproduce

Requires the supplied Codex primary runtime, `@oai/artifact-tool` 2.8.77,
Aptos fonts, and the Presentations skill finalizer. This generator is source
reproducible in that runtime; it does not claim byte-identical ZIP metadata
or universal portability to a plain Node installation.

Run from the repository root with a NEW output filename:

```bash
export RUNTIME_NODE="$CODEX_PRIMARY_RUNTIME_NODE"
export RUNTIME_NODE_MODULES="$CODEX_PRIMARY_RUNTIME_NODE_MODULES"
export RUNTIME_BIN_DIR="$CODEX_PRIMARY_RUNTIME/dependencies/bin/override"
export RUNTIME_PYTHON="$CODEX_PRIMARY_RUNTIME_PYTHON"
"$RUNTIME_NODE" presentations/faculty_review/generate.mjs \
  presentations/faculty_review/content.json \
  presentations/faculty_review/faculty_review_updated.pptx
```

The default build directory is `presentations/faculty_review/.build/` (ignored).
Optional `PRESENTATIONS_SKILL_DIR` points to the installed presentation skill.
The finalizer refuses to overwrite an existing output. It validates package
structure, slide geometry, required native tables, font use and re-import.
PNG previews and the receipt go in `.build/`; inspect them after every update.
The PPTX uses editable text, six native tables and editable process diagrams.
Future supplied PNGs remain embedded images; keep their original source plots.

## Evidence and validation

Evidence base: `3cd90ac217c0842766b5849e2fc343a9a3d2c71f` on
`feature/unified-novelty`, verified against the remote on 2026-10-03.
Directly inspected supplied JSONs: clean metrics, static poisoning comparison,
replay mitigation comparison, replay test summary, strategy selection, novelty,
split manifest and benchmark results. Historical continual execution details
come from the repository walkthrough. Sources and caveats are in every slide's
speaker notes. No full-data final test artifact was available.

All 15 slides rendered and visually reviewed. The model diagram overlap was
fixed. Structural/layout/font/native-table checks passed. Native Microsoft
PowerPoint execution was not available; this is rendered-file QA, not a claim
of having opened the deck in PowerPoint. No model suite or training was run.
