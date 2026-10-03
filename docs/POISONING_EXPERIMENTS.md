# Static training-label poisoning comparison

The unified A+B+C team owns this experiment. The Kaggle notebook compares a
newly trained 0% control with random and targeted label corruption at 5%, 10%,
and 20%. It uses the reviewed combined flow collection (per-row source
provenance unverified), the same seeded 2,000-per-broad-class sample, 70/15/15
split, train-only preprocessing, model configuration, and initialization seed.
Only training labels change; features and clean validation/test labels do not.

## Attack mechanics and budgets

`apply_label_flip` is NumPy-only. It copies the input labels and selects rows
without replacement using seed 42. The count is half-up rounded as
`floor(rate * eligible_count + 0.5)`. Random mode's eligible set is **all
training rows**, and each selected row receives a uniformly sampled different
ID from `allowed_class_ids`. Targeted mode's eligible set is rows whose labels
are in `source_class_ids`, excluding rows already labeled `target_class_id`.
Selected rows receive the explicit target, initially Benign. Targeted rate is
a fraction of that eligible set, not all training rows. For 11,200 balanced
training rows with 9,800 attack-source rows, 5% random changes 560 rows while
5% targeted changes 490. The notebook records both eligible and total-training
fractions; equal nominal percentages must not be called equal budgets.

The result includes changed positions in the training array and original
dataset row indices for audit, plus copied original/poisoned labels, requested
rate, achieved rates, counts, mode, classes, and seed. No data is altered in
place. Zero rate returns a copied unchanged array. Invalid rates, labels,
class choices, and row-index mappings raise errors.

## Controls, evaluation, and interpretation

Each condition creates a fresh Transformer with the same seed/configuration.
Initial weights are hashed and checked equal. Model selection minimizes loss
on the **clean validation set**; this explicitly assumes trusted clean
validation data and is not a defense. Test is inspected only after each
model's checkpoint is fixed. Each condition saves its own checkpoint,
history, attack audit, test metrics, timings, and reload verification.

The shared evaluator reports accuracy, macro-F1, per-class recall/F1,
confusion counts, benign false-positive rate (true Benign predicted as an
attack), and source-to-Benign misclassification (true attack-source rows
predicted as Benign). Differences are against the new 0% control, not an old
run. A clean model can also make source-to-Benign errors; a count or delta is
not proof that every individual error was caused by poisoning. All-known
confidence flags are diagnostics, not unknown-attack detection performance.

This is **static training-label corruption**, not replay poisoning or a
continual-learning task. Timing backdoors and replay poisoning are planned
extensions only. A feature-trigger attack must wait until feature semantics
and safe bounds are inspected. The separate clean baseline is not modified.

## Observed single-seed Kaggle comparison

The supplied `data/poison/manifest.json`, `comparison.json`, and executed
notebook were reviewed locally and kept out of Git. Seven initial-weight
hashes matched and all seven checkpoint reloads reproduced predictions. On
the balanced subset, the 0% control accuracy/macro-F1 were
`0.8833333333333333`/`0.878016638838182` and Benign FPR `0.56`. Random
20% changed 2,240/11,200 rows: accuracy `0.87`, macro-F1
`0.8590190796024948`, Benign FPR `0.67`. Targeted 20% changed 1,960/9,800
eligible rows, or 17.5% of all training rows: accuracy
`0.8533333333333334`, macro-F1 `0.8514854740614037`, and attack-to-Benign
rate `0.06571428571428571` versus clean `0.025238095238095237`. Random
10% accuracy and targeted 10% macro-F1 slightly exceeded control; a
single-seed comparison does not show that poisoning always worsens scores.

## Prepared replay-label corruption and consistency-gate study

This is a distinct threat model from static training-label corruption. After
training Task 1 cleanly, an attacker changes labels on the *stored old-class
replay buffer* before Task 2 reuses it. The 4,200 new-class Task 2 training
rows, clean seen-class validation, replay features, and Task 1 checkpoint
remain untouched. One seed-42 balanced buffer contains 500 unique Task 1
training rows, 100 per global old-class ID 0–4. Random 20% selects 100/500
replay rows and replaces each label with a different old ID. Targeted 20%
selects 80/400 eligible old-attack rows (IDs 1–4) and changes their supplied
label to Benign (ID 0): that is 16% of the full 500-row buffer. These are
not equal full-buffer budgets.

The six default conditions pair each buffer with no gate and the same buffer
with the gate: clean, random 20%, targeted 20%. Every Task 2 model starts
from an independent reload of one newly trained clean Task 1 checkpoint,
expands the same output classes, and uses identical seeds/hyperparameters.
The attached prior clean run did not include its Task 1 checkpoint and ran
under a different Kaggle Python/PyTorch stack, so the notebook regenerates
Task 1 and uses its own within-run controls.

The frozen Task 1 teacher scores each replay candidate as
`1 - P_teacher(supplied old label | features)`, mapping the supplied external
label through `teacher.class_ids`. A threshold is fixed at the 95th
percentile of scores on *clean Task 1 validation* before attacked conditions.
Only scores above the threshold are quarantined; equal scores stay. The
gate receives replay features, supplied labels, teacher probabilities and
class mapping, and this frozen threshold. It does **not** receive the
simulator's clean labels or changed-index mask, and it never restores a
label. An independent audit later uses those hidden records to report poison
rejection, clean false rejection, retained poison fraction, and retained
counts by supplied class. Empty retained replay is explicitly represented as
no replay to the model. The gate is never applied to genuinely new Task 2
classes, which the old teacher was not trained to recognize.

The notebook compares each filtered attack with its identical unfiltered
attack and with clean-filtered replay; clean-unfiltered is also reported.
It records old/new/combined metrics, signed forgetting, Benign FPR,
attack-to-Benign errors, epochs, optimizer steps, timings, and checkpoint
reloads. Filtering can remove clean examples, retain poison, or change
training work; improvement is not assumed. The teacher saw the clean Task 1
distribution, so this is a narrow label-consistency baseline, not evidence
against backdoors, adaptive attackers, arbitrary new-class poisoning, or
production deployment. No replay-poisoning or gate outcome has been observed
yet at the time that notebook was prepared; subsequent bounded Kaggle
mitigation results are recorded in `PROJECT_STATUS.md`.

## Prepared full-row Task 2 targeted replay comparison

The new full-row notebook is narrower than the earlier six-arm bounded
study: **only** clean replay, targeted20 unfiltered, and the identical
targeted20 buffer after filtering. It reuses the completed five-class Task 1
checkpoint and its 500 clean old-class training exemplars; Task 1 is never
retrained. All 70,076 new-class training rows remain clean and available.
Seed 42 flips 80 of 400 eligible old attack replay labels (1–4) to Benign
(0): 20% of eligible rows but only 16% of the 500-row buffer. The changed
original IDs and both denominators are saved for audit only.

The old teacher calibrates a 95th-percentile label-inconsistency threshold
on clean Task 1 validation. It sees only candidate features, supplied
labels, probabilities/class mapping, and that frozen threshold when gating
replay. Simulator originals and changed masks are never used to decide or
restore a label. This is experimental exclusion, not a persistent human
review/release workflow or a backdoor defense. Every arm uses the same
class-balanced sampling protocol and 70,576 draws per epoch, although
early stopping can produce unequal total optimizer steps. Validation
selects each model; test results cannot tune the gate or training settings.
No full-row Task 2 result exists until Kaggle executes the notebook.
