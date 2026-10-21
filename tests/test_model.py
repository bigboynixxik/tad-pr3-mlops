"""Check model input validation, isolation of validation data and saved artifacts."""

from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import model
import augment_data as augmentation


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.config = model.read_config(model.ROOT / "params.yaml")
        self.config["train"].update(dataset="prepared", seed=42, limit_per_class=0)
        self.config["logreg"]["max_iter"] = 80
        self.config["random_forest"].update(n_estimators=3, max_depth=4)
        self.config["mlp"].update(hidden_layer_sizes=[16], max_iter=5, batch_size=32)
        self.temporary = tempfile.TemporaryDirectory(prefix="tad-pr3-model-test-")
        self.root = Path(self.temporary.name)
        self.prepared = self.root / "prepared"
        self.prepared.mkdir()
        self.manifest = {"dataset": "Fashion-MNIST", "class_names": list(model.CLASSES), "splits": {}}
        self.write_split("train", 10, 0)
        self.write_split("validation", 3, 1000)
        self.write_split("test", 2, 2000)

    def tearDown(self):
        self.temporary.cleanup()

    def write_split(self, name, per_class, first_id):
        labels = np.repeat(np.arange(10, dtype=np.uint8), per_class)
        images = np.random.default_rng(7).integers(0, 25, size=(len(labels), 28, 28), dtype=np.uint8)
        for index, label in enumerate(labels):
            images[index, int(label) * 2:int(label) * 2 + 2, :] = 255
        np.savez_compressed(
            self.prepared / f"{name}.npz", images=images, labels=labels,
            source_ids=np.arange(first_id, first_id + len(labels), dtype=np.int32),
        )
        self.refresh_manifest(name, len(labels), [per_class] * 10)

    def refresh_manifest(self, name, samples, counts):
        self.manifest["splits"][name] = {
            "samples": samples, "class_counts": counts,
            "sha256": model.digest(self.prepared / f"{name}.npz"),
        }
        model.write_json(self.prepared / "summary.json", self.manifest)

    def train_and_evaluate(self, name):
        config = deepcopy(self.config)
        config["train"]["model"] = name
        directory = self.root / name
        with contextlib.redirect_stderr(io.StringIO()):
            model.train(config, self.prepared, directory)
            metrics = model.evaluate(self.prepared, directory, directory / "metrics.json",
                                     directory / "reports")
        return directory, metrics

    def make_augmented_config(self):
        config = deepcopy(self.config)
        config["train"].update(dataset="augmented", limit_per_class=0)
        config["augmentation"] = {
            "seed": 42, "copies_per_image": 1, "rotation_degrees": 10.0,
            "max_shift_pixels": 2, "horizontal_flip_probability": 0.5,
        }
        with contextlib.redirect_stderr(io.StringIO()):
            augmentation.build(self.prepared, self.root / "augmented", config["augmentation"])
        return config

    def test_augmented_training_all_models_preserves_validation(self):
        config = self.make_augmented_config()
        validation_hash = model.digest(self.prepared / "validation.npz")
        for name in model.MODEL_NAMES:
            with self.subTest(model=name), contextlib.redirect_stderr(io.StringIO()):
                config["train"]["model"] = name
                directory = self.root / f"augmented-{name}"
                metadata = model.train(config, self.prepared, directory)
                metrics = model.evaluate(self.prepared, directory, directory / "metrics.json",
                                         directory / "reports")
                self.assertEqual(metadata["dataset"], "augmented")
                self.assertEqual(metadata["training_original_samples"], 100)
                self.assertEqual(metadata["training_augmented_samples"], 100)
                self.assertEqual(metadata["unique_training_source_records"], 100)
                self.assertEqual(metrics["training_samples"], 200)
                self.assertEqual(metrics["validation_samples"], 30)
                artifact = joblib.load(directory / "model.joblib")
                self.assertEqual(len(artifact["training_source_ids"]), 200)
                self.assertEqual(np.unique(artifact["training_source_ids"]).size, 100)
                self.assertEqual(model.digest(self.prepared / "validation.npz"), validation_hash)

    def test_augmented_subset_keeps_all_variants_of_selected_originals(self):
        config = self.make_augmented_config()
        part, _ = augmentation.load_augmented(self.prepared, self.root / "augmented",
                                              config["augmentation"])
        part["sha256"] = "fixture"
        selected = model.subset(part, 3, 42)
        self.assertEqual(len(selected["labels"]), 60)
        pairs = np.column_stack((selected["source_ids"], selected["variant_ids"]))
        self.assertEqual(np.unique(pairs, axis=0).shape[0], 60)
        ids, counts = np.unique(selected["source_ids"], return_counts=True)
        self.assertEqual(len(ids), 30)
        np.testing.assert_array_equal(counts, np.full(30, 2))
        original = model.subset(model.load_split(self.prepared, "train"), 3, 42)
        np.testing.assert_array_equal(ids, original["source_ids"])
        np.testing.assert_array_equal(np.bincount(selected["labels"]), [6] * 10)

    def test_training_rejects_changed_augmentation_parameters(self):
        config = self.make_augmented_config()
        config["augmentation"]["seed"] = 43
        with self.assertRaisesRegex(ValueError, "parameters"):
            model.train(config, self.prepared, self.root / "invalid")
        self.assertFalse((self.root / "invalid/model.joblib").exists())

    def test_legacy_artifact_can_still_be_evaluated(self):
        directory, before = self.train_and_evaluate("random_forest")
        path = directory / "model.joblib"
        artifact = joblib.load(path)
        artifact["metadata"]["train"].pop("dataset")
        for key in ("dataset", "training_original_samples", "training_augmented_samples",
                    "unique_training_source_records", "augmentation"):
            artifact["metadata"].pop(key)
        joblib.dump(artifact, path)
        model.write_json(directory / "training.json", {
            **artifact["metadata"], "model_sha256": model.digest(path),
        })
        with contextlib.redirect_stderr(io.StringIO()):
            after = model.evaluate(self.prepared, directory, directory / "metrics.json",
                                   directory / "reports")
        self.assertEqual(before, after)

    def test_pixels_are_flattened_and_normalized_without_mutating_input(self):
        images = np.array([np.full((28, 28), 255, np.uint8), np.zeros((28, 28), np.uint8)])
        transformed = model.features(images)
        self.assertEqual(transformed.shape, (2, 784))
        self.assertEqual(transformed.dtype, np.float32)
        np.testing.assert_array_equal(transformed[0], np.ones(784))
        np.testing.assert_array_equal(transformed[1], np.zeros(784))
        self.assertEqual(images[0, 0, 0], 255)
        for bad in (images.astype(float), images[:, :27], images.reshape(2, 784)):
            with self.assertRaises(ValueError):
                model.features(bad)

    def test_invalid_configuration_is_rejected(self):
        changes = (
            ("train", "model", "unknown"), ("train", "dataset", "unknown"), ("train", "seed", True),
            ("train", "seed", -1), ("train", "seed", 2**32),
            ("train", "limit_per_class", -1), ("logreg", "C", float("nan")),
            ("logreg", "max_iter", 0), ("logreg", "tol", float("inf")),
            ("random_forest", "n_estimators", 0), ("random_forest", "max_depth", 0),
            ("random_forest", "min_samples_leaf", 1.5), ("mlp", "hidden_layer_sizes", []),
            ("mlp", "hidden_layer_sizes", [True]), ("mlp", "batch_size", 0),
            ("mlp", "learning_rate_init", 0), ("mlp", "alpha", -0.1),
        )
        for group, key, value in changes:
            config = deepcopy(self.config)
            config[group][key] = value
            with self.subTest(group=group, key=key, value=value), self.assertRaises(ValueError):
                model.validate_config(config)
        config = deepcopy(self.config)
        config["logreg"]["typo"] = 1
        with self.assertRaises(ValueError):
            model.validate_config(config)

    def test_subset_is_balanced_repeatable_and_keeps_source_ids(self):
        part = model.load_split(self.prepared, "train")
        first = model.subset(part, 3, 42)
        np.testing.assert_array_equal(np.bincount(first["labels"]), [3] * 10)
        np.testing.assert_array_equal(first["source_ids"], model.subset(part, 3, 42)["source_ids"])
        self.assertFalse(np.array_equal(first["source_ids"], model.subset(part, 3, 43)["source_ids"]))
        np.testing.assert_array_equal(first["images"], part["images"][first["source_ids"]])
        self.assertIs(model.subset(part, 0, 42), part)
        with self.assertRaises(ValueError):
            model.subset(part, 11, 42)

    def test_loader_rejects_checksum_and_structural_errors(self):
        path = self.prepared / "train.npz"
        original = path.read_bytes()
        path.write_bytes(original + b"corruption")
        with self.assertRaisesRegex(ValueError, "Checksum"):
            model.load_split(self.prepared, "train")
        path.write_bytes(original)
        with np.load(path) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["source_ids"][1] = arrays["source_ids"][0]
        np.savez_compressed(path, **arrays)
        self.refresh_manifest("train", 100, [10] * 10)
        with self.assertRaisesRegex(ValueError, "repeated"):
            model.load_split(self.prepared, "train")
        arrays["source_ids"] = np.arange(100, dtype=np.int32)
        arrays["images"] = arrays["images"].astype(np.float32)
        np.savez_compressed(path, **arrays)
        self.refresh_manifest("train", 100, [10] * 10)
        with self.assertRaisesRegex(ValueError, "dtype"):
            model.load_split(self.prepared, "train")

    def test_all_models_reload_and_metrics_match_saved_predictions(self):
        for name in model.MODEL_NAMES:
            with self.subTest(model=name):
                directory, metrics = self.train_and_evaluate(name)
                self.assertEqual(metrics["training_samples"], 100)
                self.assertEqual(metrics["validation_samples"], 30)
                self.assertGreaterEqual(metrics["accuracy"], 0)
                self.assertLessEqual(metrics["accuracy"], 1)
                self.assertGreaterEqual(metrics["macro_f1"], 0)
                self.assertLessEqual(metrics["macro_f1"], 1)
                artifact = joblib.load(directory / "model.joblib")
                part = model.load_split(self.prepared, "validation")
                with model.threadpool_limits(limits=1):
                    reloaded_predictions = artifact["pipeline"].predict(model.features(part["images"]))
                with np.load(directory / "reports" / "validation_predictions.npz") as archive:
                    np.testing.assert_array_equal(reloaded_predictions, archive["y_pred"])
                    np.testing.assert_array_equal(part["source_ids"], archive["source_ids"])
                    expected_accuracy = np.count_nonzero(archive["y_true"] == archive["y_pred"]) / 30
                self.assertEqual(metrics["accuracy"], expected_accuracy)
                details = json.loads((directory / "reports" / "validation_details.json").read_text())
                matrix = np.array(details["confusion_matrix"])
                self.assertEqual(matrix.shape, (10, 10))
                self.assertEqual(matrix.sum(), 30)
                self.assertEqual(matrix.trace() / 30, metrics["accuracy"])
                if name != "random_forest":
                    scaler = artifact["pipeline"].named_steps["scale"]
                    training = model.load_split(self.prepared, "train")
                    np.testing.assert_allclose(
                        scaler.mean_, model.features(training["images"]).mean(axis=0, dtype=np.float64),
                    )
                    self.assertEqual(scaler.n_samples_seen_, 100)

    def test_validation_overlap_is_rejected(self):
        directory, _ = self.train_and_evaluate("random_forest")
        self.write_split("validation", 3, 0)
        with self.assertRaisesRegex(ValueError, "overlap"):
            model.evaluate(self.prepared, directory, directory / "metrics.json", directory / "reports")

    def test_model_corruption_and_version_mismatch_are_rejected(self):
        directory, _ = self.train_and_evaluate("random_forest")
        path = directory / "model.joblib"
        original = path.read_bytes()
        path.write_bytes(original + b"corruption")
        with self.assertRaisesRegex(ValueError, "checksum"):
            model.evaluate(self.prepared, directory, directory / "metrics.json", directory / "reports")
        path.write_bytes(original)
        artifact = joblib.load(path)
        artifact["metadata"]["versions"]["scikit_learn"] = "different"
        joblib.dump(artifact, path)
        model.write_json(directory / "training.json", {
            **artifact["metadata"], "model_sha256": model.digest(path),
        })
        with self.assertRaisesRegex(ValueError, "versions"):
            model.evaluate(self.prepared, directory, directory / "metrics.json", directory / "reports")


if __name__ == "__main__":
    unittest.main()
