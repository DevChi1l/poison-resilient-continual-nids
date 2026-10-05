from pathlib import Path
import json
import unittest

from src.demo.results import load_task2_playback


ARCHIVE = Path("data/task2/results.zip")
SUMMARY = Path("docs/evidence/task2_targeted_summary.json")
ARCHIVE_SHA256 = "0c73a44cca06cb92f7c21c98652808f739df67236d58d790f5cf4b93027d832e"


@unittest.skipUnless(ARCHIVE.is_file(), "private Task 2 result archive is absent")
class RealTask2PlaybackTests(unittest.TestCase):
    def test_recorded_three_arm_values_match_original_evidence(self) -> None:
        playback = load_task2_playback(ARCHIVE, ARCHIVE_SHA256)
        summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
        self.assertEqual(summary["evidence"]["archive_sha256"], ARCHIVE_SHA256)
        expected = {
            "clean_replay": (0.6010047020872148, 0.5664584687018178, 0.4989118011592611),
            "targeted20_unfiltered": (0.44913586603196487, 0.5379282266745713, 0.6885997951625712),
            "targeted20_filtered": (0.6077218656509111, 0.5578505675817005, 0.4666276411782605),
        }
        for arm, values in expected.items():
            metrics = playback.combined_metrics[arm]
            self.assertEqual(metrics["accuracy"], values[0])
            self.assertEqual(metrics["macro_f1"], values[1])
            self.assertEqual(metrics["benign_false_positive"]["rate"], values[2])
            self.assertEqual(summary["arms"][arm]["combined_accuracy"], values[0])
            self.assertEqual(summary["arms"][arm]["combined_macro_f1"], values[1])
            self.assertEqual(summary["arms"][arm]["benign_fpr"], values[2])
            training = playback.comparison["arms"][arm]["training"]
            self.assertEqual(training["completed_epoch"], 6)
            self.assertEqual(training["best_epoch"], 1)
            self.assertEqual(training["optimizer_steps"], 1656)
        gate = playback.gate_audit["audit"]
        self.assertEqual(gate["poison_rejected"], 80)
        self.assertEqual(gate["clean_false_rejected"], 42)
        self.assertEqual(gate["retained_total"], 378)


if __name__ == "__main__":
    unittest.main()
