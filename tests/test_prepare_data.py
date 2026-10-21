"""Check input validation, stratified splitting and interrupted downloads."""

import contextlib
import gzip
import hashlib
import importlib.util
import io
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

import numpy as np


spec = importlib.util.spec_from_file_location(
    "prepare_data", Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"
)
data = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = data
spec.loader.exec_module(data)


class DataPreparationTests(unittest.TestCase):
    def test_stratified_split_is_complete_disjoint_and_repeatable(self):
        labels = np.repeat(np.arange(10, dtype=np.uint8), 20)
        train, validation = data.stratified_indices(labels, 42, 5)
        self.assertEqual(len(train), 150)
        self.assertEqual(len(validation), 50)
        self.assertEqual(len(set(train) & set(validation)), 0)
        np.testing.assert_array_equal(np.sort(np.concatenate((train, validation))), np.arange(200))
        np.testing.assert_array_equal(np.bincount(labels[validation]), [5] * 10)
        np.testing.assert_array_equal(validation, data.stratified_indices(labels, 42, 5)[1])
        self.assertFalse(np.array_equal(validation, data.stratified_indices(labels, 43, 5)[1]))

    def test_invalid_parameters_are_rejected(self):
        for seed, size in ((True, 1), (-1, 1), (2**32, 1), (42, 0), (42, 1.5)):
            with self.subTest(seed=seed, size=size), self.assertRaises(ValueError):
                data.validate_parameters(seed, size)

    def test_missing_classes_and_excessive_validation_are_rejected(self):
        with self.assertRaises(ValueError):
            data.stratified_indices(np.array([0, 1]), 42, 1)
        with self.assertRaises(ValueError):
            data.stratified_indices(np.repeat(np.arange(10), 2), 42, 2)

    def test_idx_images_reject_bad_magic_dimensions_and_truncation(self):
        with tempfile.TemporaryDirectory(prefix="tad-pr3-test-") as folder:
            path = Path(folder) / "images.gz"
            valid = struct.pack(">IIII", 2051, 2, 28, 28) + bytes(2 * 784)
            path.write_bytes(gzip.compress(valid))
            self.assertEqual(data.read_images(path, 2).shape, (2, 28, 28))
            for invalid in (
                struct.pack(">IIII", 2050, 2, 28, 28) + bytes(2 * 784),
                struct.pack(">IIII", 2051, 2, 14, 56) + bytes(2 * 784),
                valid[:-1], valid + b"x", b"short",
            ):
                path.write_bytes(gzip.compress(invalid))
                with self.assertRaises(ValueError):
                    data.read_images(path, 2)

    def test_idx_labels_reject_unknown_classes_and_wrong_lengths(self):
        with tempfile.TemporaryDirectory(prefix="tad-pr3-test-") as folder:
            path = Path(folder) / "labels.gz"
            path.write_bytes(gzip.compress(struct.pack(">II", 2049, 2) + bytes([0, 9])))
            np.testing.assert_array_equal(data.read_labels(path, 2), [0, 9])
            for payload in (
                struct.pack(">II", 2049, 2) + bytes([0, 10]),
                struct.pack(">II", 2049, 3) + bytes([0, 9]),
                struct.pack(">II", 2049, 2) + bytes([0]),
            ):
                path.write_bytes(gzip.compress(payload))
                with self.assertRaises(ValueError):
                    data.read_labels(path, 2)

    def test_download_retry_and_cache_validation(self):
        payload = b"verified dataset bytes"
        source = data.SourceFile("sample.gz", hashlib.md5(payload).hexdigest())
        with tempfile.TemporaryDirectory(prefix="tad-pr3-test-") as folder:
            directory = Path(folder)
            with contextlib.redirect_stderr(io.StringIO()), patch.object(
                data, "urlopen", side_effect=[URLError("interrupted"), io.BytesIO(payload)]
            ) as request, patch.object(data.time, "sleep"):
                data.download_file(source, directory)
                self.assertEqual(request.call_count, 2)
            self.assertEqual((directory / source.name).read_bytes(), payload)
            self.assertFalse(list(directory.glob("*.part")))
            with contextlib.redirect_stderr(io.StringIO()), patch.object(data, "urlopen") as request:
                data.download_file(source, directory)
                request.assert_not_called()
            (directory / source.name).write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                data.download_file(source, directory)

    def test_invalid_download_is_not_saved(self):
        source = data.SourceFile("sample.gz", hashlib.md5(b"correct").hexdigest())
        with tempfile.TemporaryDirectory(prefix="tad-pr3-test-") as folder:
            with contextlib.redirect_stderr(io.StringIO()), patch.object(
                data, "urlopen", side_effect=lambda *args, **kwargs: io.BytesIO(b"wrong")
            ), patch.object(data.time, "sleep"):
                with self.assertRaises(OSError):
                    data.download_file(source, Path(folder))
            self.assertEqual(list(Path(folder).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
