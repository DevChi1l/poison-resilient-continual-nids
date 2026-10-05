import hashlib
import json
from pathlib import Path
import stat
import zipfile

import numpy as np
import pytest

from scripts.run_task2_b_poisoning import (
    fixed_b_probabilities,
    read_verified_json_archive,
    require_supplied_class_buckets,
)
from src.poisoning import apply_label_flip


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_named_evidence_reader_rejects_unsafe_members(tmp_path):
    safe = tmp_path / "safe.zip"
    with zipfile.ZipFile(safe, "w") as archive:
        archive.writestr("evidence.json", json.dumps({"ok": True}))
    records = read_verified_json_archive(
        safe, expected_sha256=_sha256(safe), members=("evidence.json",)
    )
    assert records == {"evidence.json": {"ok": True}}

    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as archive:
        archive.writestr("../escape.json", "{}")
    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        read_verified_json_archive(
            unsafe, expected_sha256=_sha256(unsafe), members=("../escape.json",)
        )

    symlink = tmp_path / "symlink.zip"
    with zipfile.ZipFile(symlink, "w") as archive:
        member = zipfile.ZipInfo("link")
        member.create_system = 3
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(member, "target")
    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        read_verified_json_archive(
            symlink, expected_sha256=_sha256(symlink), members=("link",)
        )


def test_exact_targeted_budget_and_required_filtered_buckets():
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
    assert attacked.eligible_count == 20_000
    assert attacked.changed_count == 4_000
    assert attacked.achieved_total_fraction == 0.16
    assert np.all(attacked.poisoned_labels[attacked.changed_indices] == 0)
    assert require_supplied_class_buckets(attacked.poisoned_labels) == {
        "0": 9_000, "1": 4_016, "2": 3_979, "3": 4_018, "4": 3_987,
    }
    with pytest.raises(RuntimeError, match="emptied required"):
        require_supplied_class_buckets(np.array([0, 1, 2, 3], dtype=np.int64))


def test_frozen_b_probabilities_match_saved_quota_basis():
    names = ("Benign", "DDoS", "DoS", "Botnet", "Bruteforce", "X", "Y", "Z")
    old_counts = (5_029_455, 864_052, 278_187, 102_211, 72_309)
    manifest = {
        "counts": {
            name: {"train": count} for name, count in zip(names[:5], old_counts)
        }
    }
    probabilities = fixed_b_probabilities(manifest, names)
    assert sum(probabilities[class_id] for class_id in range(5)) == pytest.approx(0.625)
    assert probabilities[5] == probabilities[6] == probabilities[7] == 0.125
    assert sum(probabilities.values()) == pytest.approx(1.0)
