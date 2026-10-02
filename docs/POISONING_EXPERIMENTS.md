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
