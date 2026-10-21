"""Test reproducibility, original records and held-out isolation of augmentation."""

from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import augment_data as augmentation


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "seed": 42, "copies_per_image": 1, "rotation_degrees": 10.0,
            "max_shift_pixels": 2, "horizontal_flip_probability": 0.5,
        }
        self.temporary = tempfile.TemporaryDirectory(prefix="tad-pr3-augmentation-test-")
        self.root = Path(self.temporary.name)
        self.prepared = self.root / "prepared"
        self.prepared.mkdir()
        manifest = {"dataset": "Fashion-MNIST", "class_names": list(augmentation.CLASSES), "splits": {}}
        for name, per_class, first_id in (("train", 4, 0), ("validation", 1, 100), ("test", 1, 200)):
            labels = np.repeat(np.arange(10, dtype=np.uint8), per_class)
            images = np.random.default_rng(7).integers(0, 256, (len(labels), 28, 28), dtype=np.uint8)
            path = self.prepared / f"{name}.npz"
            np.savez_compressed(path, images=images, labels=labels,
                                source_ids=np.arange(first_id, first_id + len(labels), dtype=np.int32))
            manifest["splits"][name] = {
                "samples": len(labels), "class_counts": [per_class] * 10,
                "sha256": augmentation.digest(path),
            }
        augmentation.write_json(self.prepared / "summary.json", manifest)
        self.source, _, self.held_out = augmentation.load_base(self.prepared)
        self.output = self.root / "augmented"

    def tearDown(self):
        self.temporary.cleanup()

    def build(self):
        with contextlib.redirect_stderr(io.StringIO()):
            return augmentation.build(self.prepared, self.output, self.config)

    def test_invalid_parameters_are_rejected(self):
        for key, value in (
            ("seed", True), ("seed", -1), ("seed", 2**32),
            ("copies_per_image", 0), ("copies_per_image", 11),
            ("rotation_degrees", -1), ("rotation_degrees", float("nan")),
            ("max_shift_pixels", 1.5), ("max_shift_pixels", 8),
            ("horizontal_flip_probability", 1.1), ("horizontal_flip_probability", float("inf")),
        ):
            config = {**self.config, key: value}
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                augmentation.validate_config(config)
        with self.assertRaises(ValueError):
            augmentation.validate_config({**self.config, "typo": 1})
        with self.assertRaises(ValueError):
            augmentation.validate_config({
                **self.config, "rotation_degrees": 0, "max_shift_pixels": 0,
                "horizontal_flip_probability": 0,
            })

    def test_originals_labels_and_lineage_are_preserved(self):
        self.config["copies_per_image"] = 2
        part = augmentation.generate(self.source, self.config)
        augmentation.validate_combined(part, self.source, self.config, self.held_out)
        self.assertEqual(part["images"].shape, (120, 28, 28))
        self.assertEqual(part["images"].dtype, np.uint8)
        np.testing.assert_array_equal(part["images"][:40], self.source["images"])
        np.testing.assert_array_equal(np.bincount(part["labels"]), [12] * 10)
        pairs = np.column_stack((part["source_ids"], part["variant_ids"]))
        self.assertEqual(np.unique(pairs, axis=0).shape[0], 120)
        self.assertEqual(np.unique(part["source_ids"]).size, 40)

    def test_each_transformation_and_record_seed(self):
        image = self.source["images"][0]
        flip_only = {**self.config, "rotation_degrees": 0, "max_shift_pixels": 0,
                     "horizontal_flip_probability": 1}
        np.testing.assert_array_equal(
            augmentation.transform(image, 0, 1, flip_only), np.fliplr(image),
        )
        for settings in (
            {"rotation_degrees": 10.0, "max_shift_pixels": 0, "horizontal_flip_probability": 0},
            {"rotation_degrees": 0, "max_shift_pixels": 2, "horizontal_flip_probability": 0},
        ):
            config = {**self.config, **settings}
            self.assertFalse(np.array_equal(image, augmentation.transform(image, 0, 1, config)))
        first = augmentation.generate(self.source, self.config)
        reordered = {key: array[::-1] for key, array in self.source.items()}
        second = augmentation.generate(reordered, self.config)
        np.testing.assert_array_equal(first["images"][40:], second["images"][40:][::-1])
        changed_seed = augmentation.generate(self.source, {**self.config, "seed": 43})
        self.assertFalse(np.array_equal(first["images"][40:], changed_seed["images"][40:]))

    def test_held_out_overlap_and_wrong_lineage_are_rejected(self):
        part = augmentation.generate(self.source, self.config)
        with self.assertRaisesRegex(ValueError, "overlap"):
            augmentation.validate_combined(part, self.source, self.config, np.array([0]))
        part["source_ids"][40] = 999
        with self.assertRaisesRegex(ValueError, "source_ids"):
            augmentation.validate_combined(part, self.source, self.config, self.held_out)

    def test_build_is_repeatable_and_does_not_change_source_files(self):
        before = {name: augmentation.digest(self.prepared / f"{name}.npz")
                  for name in augmentation.SPLITS}
        self.build()
        first = {path.name: augmentation.digest(path) for path in self.output.iterdir()}
        report = augmentation.check(self.prepared, self.output, self.config)
        self.assertEqual(report["train"]["samples"], 80)
        self.assertEqual(report["source_id_overlap"], 0)
        self.assertTrue(report["held_out_splits_unchanged"])
        self.build()
        self.assertEqual(first, {path.name: augmentation.digest(path) for path in self.output.iterdir()})
        self.assertEqual(before, {name: augmentation.digest(self.prepared / f"{name}.npz")
                                  for name in augmentation.SPLITS})

    def test_corrupt_pixels_are_rejected_even_with_updated_file_checksum(self):
        self.build()
        path = self.output / "train.npz"
        original = path.read_bytes()
        path.write_bytes(original + b"corruption")
        with self.assertRaisesRegex(ValueError, "checksum"):
            augmentation.check(self.prepared, self.output, self.config)
        path.write_bytes(original)
        with np.load(path) as archive:
            part = {key: archive[key] for key in archive.files}
        part["images"][40, 10, 10] ^= np.uint8(255)
        np.savez_compressed(path, **part)
        report = json.loads((self.output / "summary.json").read_text())
        report["train"]["sha256"] = augmentation.digest(path)
        augmentation.write_json(self.output / "summary.json", report)
        with self.assertRaisesRegex(ValueError, "pixels"):
            augmentation.check(self.prepared, self.output, self.config)

    def test_changed_parameters_or_source_dataset_are_rejected(self):
        self.build()
        with self.assertRaisesRegex(ValueError, "parameters"):
            augmentation.check(self.prepared, self.output, {**self.config, "seed": 43})
        path = self.prepared / "validation.npz"
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "Checksum"):
            augmentation.check(self.prepared, self.output, self.config)

    def test_inconsistent_summary_is_rejected(self):
        self.build()
        path = self.output / "summary.json"
        report = json.loads(path.read_text())
        report["train"]["original_samples"] = 0
        augmentation.write_json(path, report)
        with self.assertRaisesRegex(ValueError, "counts"):
            augmentation.check(self.prepared, self.output, self.config)

    def test_original_split_overlap_is_rejected(self):
        path = self.prepared / "test.npz"
        with np.load(path) as archive:
            part = {key: archive[key] for key in archive.files}
        part["source_ids"][0] = 0
        np.savez_compressed(path, **part)
        manifest = json.loads((self.prepared / "summary.json").read_text())
        manifest["splits"]["test"]["sha256"] = augmentation.digest(path)
        augmentation.write_json(self.prepared / "summary.json", manifest)
        with self.assertRaisesRegex(ValueError, "overlap"):
            augmentation.load_base(self.prepared)


if __name__ == "__main__":
    unittest.main()
