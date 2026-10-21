"""Check validation-only selection and test evaluation of frozen model weights."""

from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import finalize
import model
import test_model


class FinalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_model.ModelTests()
        self.fixture.setUp()
        self.root = self.fixture.root
        self.current, self.validation = self.fixture.train_and_evaluate("random_forest")
        self.comparison = self.root / "comparison.json"
        self.validation_path = self.root / "validation.json"
        self.selected = self.root / "selected"
        model.write_json(self.validation_path, self.validation)
        self.report = self.make_comparison()
        model.write_json(self.comparison, self.report)

    def tearDown(self):
        self.fixture.tearDown()

    def make_comparison(self):
        pairs = []
        for name in ("random_forest", "logreg", "mlp"):
            original = {
                "experiment": f"original-{name}", "revision": "fixture-revision",
                "model": name, "dataset": "prepared", "seed": 42,
                "parameters": deepcopy(self.fixture.config[name]),
                "metrics": deepcopy(self.validation),
            }
            if name != "random_forest":
                original["metrics"]["macro_f1"] *= 0.5
                original["metrics"]["accuracy"] = 1.0
            augmented = deepcopy(original)
            augmented.update(experiment=f"augmented-{name}", dataset="augmented")
            augmented["metrics"]["macro_f1"] *= 0.5
            pairs.append({"model": name, "original": original, "augmented": augmented})
        return {
            "evaluation_split": "validation", "selection_metric": "macro_f1",
            "best_validation_experiment": "original-random_forest",
            "validation_sha256": self.fixture.manifest["splits"]["validation"]["sha256"],
            "comparisons": pairs,
        }

    def freeze(self):
        return finalize.freeze(self.current, self.validation_path, self.comparison,
                               self.fixture.prepared, self.selected)

    def test_selection_uses_validation_macro_f1(self):
        self.assertEqual(finalize.choose(self.report)["experiment"], "original-random_forest")
        for field, value in (("selection_metric", "accuracy"), ("evaluation_split", "test"),
                             ("best_validation_experiment", "original-logreg")):
            report = {**self.report, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                finalize.choose(report)

    def test_freeze_rejects_model_that_differs_from_selection(self):
        self.report["comparisons"][0]["original"]["parameters"]["n_estimators"] += 1
        model.write_json(self.comparison, self.report)
        with self.assertRaisesRegex(ValueError, "differs"):
            self.freeze()
        self.assertFalse(self.selected.exists())

    def test_freeze_and_test_leave_model_and_validation_unchanged(self):
        selection = self.freeze()
        watched = [
            self.current / "model.joblib", self.current / "training.json",
            self.selected / "model.joblib", self.selected / "training.json",
            self.validation_path, self.fixture.prepared / "validation.npz",
        ]
        before = {str(path): model.digest(path) for path in watched}
        with contextlib.redirect_stderr(io.StringIO()):
            summary = finalize.test_selected(
                self.selected, self.fixture.prepared, self.root / "test-metrics.json",
                self.root / "reports", self.root / "final.json",
            )
        self.assertTrue(selection["decision_made_before_test"])
        self.assertFalse(summary["test_used_for_selection"])
        self.assertEqual(summary["metrics"]["test_samples"], 20)
        self.assertEqual(summary["metrics"]["training_samples"], 100)
        self.assertEqual(summary["source_id_overlap"], 0)
        self.assertEqual(before, {str(path): model.digest(path) for path in watched})
        with np.load(self.root / "reports/test_predictions.npz") as predictions:
            accuracy = np.mean(predictions["y_true"] == predictions["y_pred"])
            self.assertEqual(accuracy, summary["metrics"]["accuracy"])
        details = json.loads((self.root / "reports/test_details.json").read_text())
        self.assertEqual(np.asarray(details["confusion_matrix"]).sum(), 20)

    def test_selected_model_change_is_rejected(self):
        self.freeze()
        path = self.selected / "model.joblib"
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            finalize.test_selected(self.selected, self.fixture.prepared,
                                   self.root / "metrics.json", self.root / "reports",
                                   self.root / "final.json")

    def test_test_source_overlap_is_rejected(self):
        self.freeze()
        path = self.fixture.prepared / "test.npz"
        with np.load(path) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["source_ids"][0] = 0
        np.savez_compressed(path, **arrays)
        manifest = json.loads((self.fixture.prepared / "summary.json").read_text())
        manifest["splits"]["test"]["sha256"] = model.digest(path)
        model.write_json(self.fixture.prepared / "summary.json", manifest)
        with self.assertRaisesRegex(ValueError, "overlap"):
            finalize.test_selected(self.selected, self.fixture.prepared,
                                   self.root / "metrics.json", self.root / "reports",
                                   self.root / "final.json")


if __name__ == "__main__":
    unittest.main()
