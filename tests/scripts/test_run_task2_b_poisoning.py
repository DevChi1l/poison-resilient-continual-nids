import hashlib
import json
from pathlib import Path
import stat
import unittest
import zipfile

import numpy as np

from scripts.run_task2_b_poisoning import (
    fixed_b_probabilities,
    read_verified_json_archive,
    require_supplied_class_buckets,
)
from src.poisoning import apply_label_flip


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FrozenBProtocolTests(unittest.TestCase):
    def test_named_evidence_reader_rejects_unsafe_members(self):
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            safe = root / "safe.zip"
            with zipfile.ZipFile(safe, "w") as archive:
                archive.writestr("evidence.json", json.dumps({"ok": True}))
            records = read_verified_json_archive(
                safe, expected_sha256=_sha256(safe), members=("evidence.json",)
            )
            self.assertEqual(records, {"evidence.json": {"ok": True}})

            unsafe = root / "unsafe.zip"
            with zipfile.ZipFile(unsafe, "w") as archive:
                archive.writestr("../escape.json", "{}")
            with self.assertRaisesRegex(ValueError, "Unsafe ZIP member"):
                read_verified_json_archive(
                    unsafe, expected_sha256=_sha256(unsafe),
                    members=("../escape.json",),
                )

            symlink = root / "symlink.zip"
            with zipfile.ZipFile(symlink, "w") as archive:
                member = zipfile.ZipInfo("link")
                member.create_system = 3
                member.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(member, "target")
            with self.assertRaisesRegex(ValueError, "Unsafe ZIP member"):
                read_verified_json_archive(
                    symlink, expected_sha256=_sha256(symlink), members=("link",)
                )

    def test_exact_targeted_budget_and_required_filtered_buckets(self):
        labels = np.repeat(np.arange(5, dtype=np.int64), 5_000)
        row_ids = np.arange(25_000, dtype=np.int64)
        attacked = apply_label_flip(
            labels,
            0.20,
            42,
            mode="targeted",
            source_class_ids=(1, 2, 3, 4),
            target_class_id=0,
            allowed_class_ids=tuple(range(5)),
            original_row_indices=row_ids,
        )
        self.assertEqual(attacked.eligible_count, 20_000)
        self.assertEqual(attacked.changed_count, 4_000)
        self.assertEqual(attacked.achieved_total_fraction, 0.16)
        self.assertTrue(np.all(attacked.poisoned_labels[attacked.changed_indices] == 0))
        self.assertEqual(require_supplied_class_buckets(attacked.poisoned_labels), {
            "0": 9_000, "1": 4_016, "2": 3_979, "3": 4_018, "4": 3_987,
        })
        with self.assertRaisesRegex(RuntimeError, "emptied required"):
            require_supplied_class_buckets(np.array([0, 1, 2, 3], dtype=np.int64))

    def test_frozen_b_probabilities_match_saved_quota_basis(self):
        names = ("Benign", "DDoS", "DoS", "Botnet", "Bruteforce", "X", "Y", "Z")
        old_counts = (5_029_455, 864_052, 278_187, 102_211, 72_309)
        manifest = {
            "counts": {
                name: {"train": count} for name, count in zip(names[:5], old_counts)
            }
        }
        probabilities = fixed_b_probabilities(manifest, names)
        self.assertAlmostEqual(sum(probabilities[class_id] for class_id in range(5)), 0.625)
        self.assertEqual(probabilities[5], probabilities[6])
        self.assertEqual(probabilities[6], probabilities[7])
        self.assertEqual(probabilities[7], 0.125)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
