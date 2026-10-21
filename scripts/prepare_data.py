"""Download, split and verify Fashion-MNIST without changing its official test set."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import random
import struct
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen
import zipfile

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PREPARED = ROOT / "data" / "prepared"
PARAMS = ROOT / "params.yaml"
BASE_URLS = (
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
    "https://fashion-mnist.s3.eu-central-1.amazonaws.com/",
)
CLASSES = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)


@dataclass(frozen=True)
class SourceFile:
    name: str
    md5: str


FILES = (
    SourceFile("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
    SourceFile("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
    SourceFile("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
    SourceFile("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
)


def digest(path: Path, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm, usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def verify_source(path: Path, source: SourceFile) -> None:
    if digest(path, "md5") != source.md5:
        raise ValueError(f"Checksum mismatch: {path}. Restore this file with DVC.")


def download_file(source: SourceFile, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / source.name
    if target.exists():
        verify_source(target, source)
        print(f"Verified existing {source.name}", file=sys.stderr, flush=True)
        return
    last_error: Exception | None = None
    for base in BASE_URLS:
        for attempt in range(2):
            temporary: Path | None = None
            try:
                request = Request(base + source.name, headers={"User-Agent": "tad-pr3-mlops"})
                print(f"Downloading {source.name} (attempt {attempt + 1})",
                      file=sys.stderr, flush=True)
                with urlopen(request, timeout=30) as response:
                    with tempfile.NamedTemporaryFile(
                        dir=directory, prefix=".", suffix=".part", delete=False
                    ) as stream:
                        temporary = Path(stream.name)
                        received = 0
                        next_progress = 2 * 1024 * 1024
                        while block := response.read(1024 * 1024):
                            received += len(block)
                            if received > 64 * 1024 * 1024:
                                raise ValueError("Downloaded file exceeds the expected size limit.")
                            stream.write(block)
                            if received >= next_progress:
                                print(f"  {source.name}: {received / 1024**2:.1f} MiB",
                                      file=sys.stderr, flush=True)
                                next_progress += 2 * 1024 * 1024
                verify_source(temporary, source)
                temporary.replace(target)
                print(f"Verified {source.name}", file=sys.stderr, flush=True)
                return
            except (OSError, URLError, ValueError) as exc:
                last_error = exc
                print(f"  Download failed: {exc}", file=sys.stderr, flush=True)
                if attempt == 0:
                    time.sleep(1)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    raise OSError(f"Could not download {source.name}: {last_error}")


def read_images(path: Path, expected_count: int) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        payload = stream.read(expected_count * 784 + 17)
    if len(payload) < 16:
        raise ValueError(f"Truncated image header: {path}")
    magic, count, rows, columns = struct.unpack(">IIII", payload[:16])
    if (magic, count, rows, columns) != (2051, expected_count, 28, 28):
        raise ValueError(f"Invalid image header: {path}")
    if len(payload) != 16 + count * rows * columns:
        raise ValueError(f"Invalid image payload size: {path}")
    return np.frombuffer(payload, dtype=np.uint8, offset=16).reshape(count, 28, 28).copy()


def read_labels(path: Path, expected_count: int) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        payload = stream.read(expected_count + 9)
    if len(payload) < 8:
        raise ValueError(f"Truncated label header: {path}")
    magic, count = struct.unpack(">II", payload[:8])
    if magic != 2049 or count != expected_count or len(payload) != 8 + count:
        raise ValueError(f"Invalid labels: {path}")
    labels = np.frombuffer(payload, dtype=np.uint8, offset=8).copy()
    if np.any(labels > 9):
        raise ValueError(f"Unknown class label: {path}")
    return labels


def load_raw() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    for source in FILES:
        verify_source(RAW / source.name, source)
    return (
        read_images(RAW / FILES[0].name, 60000),
        read_labels(RAW / FILES[1].name, 60000),
        read_images(RAW / FILES[2].name, 10000),
        read_labels(RAW / FILES[3].name, 10000),
    )


def validate_parameters(seed: object, validation_per_class: object) -> tuple[int, int]:
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("data.seed must be an integer from 0 to 4294967295.")
    if type(validation_per_class) is not int or validation_per_class <= 0:
        raise ValueError("data.validation_per_class must be a positive integer.")
    return seed, validation_per_class


def parameters() -> tuple[int, int]:
    config = yaml.safe_load(PARAMS.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("data"), dict):
        raise ValueError("params.yaml must contain a data mapping.")
    return validate_parameters(
        config["data"].get("seed"), config["data"].get("validation_per_class")
    )


def stratified_indices(labels: np.ndarray, seed: int, per_class: int) -> tuple[np.ndarray, np.ndarray]:
    validate_parameters(seed, per_class)
    if labels.ndim != 1 or not np.array_equal(np.unique(labels), np.arange(10)):
        raise ValueError("Training data must contain exactly classes 0 through 9.")
    rng = random.Random(seed)
    train, validation = [], []
    for label in range(10):
        indices = np.flatnonzero(labels == label).tolist()
        if len(indices) <= per_class:
            raise ValueError("Validation size must leave training examples in every class.")
        rng.shuffle(indices)
        validation.extend(indices[:per_class])
        train.extend(indices[per_class:])
    return np.array(sorted(train), dtype=np.int32), np.array(sorted(validation), dtype=np.int32)


def atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".", suffix=".part", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer(stream)
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        if not path.exists() or digest(path) != digest(temporary):
            temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_parts(parts: dict, raw: tuple, per_class: int) -> None:
    train_images, train_labels, test_images, test_labels = raw
    all_ids = []
    expected_counts = {
        "train": len(train_labels) - 10 * per_class,
        "validation": 10 * per_class,
        "test": len(test_labels),
    }
    for name, part in parts.items():
        images, labels, ids = part["images"], part["labels"], part["source_ids"]
        count = expected_counts[name]
        if images.shape != (count, 28, 28) or labels.shape != (count,) or ids.shape != (count,):
            raise ValueError(f"Unexpected split shape: {name}")
        if images.dtype != np.uint8 or labels.dtype != np.uint8 or ids.dtype != np.int32:
            raise ValueError(f"Unexpected array type: {name}")
        if name == "test":
            expected_ids = np.arange(len(test_labels), dtype=np.int32) + len(train_labels)
            if not np.array_equal(ids, expected_ids):
                raise ValueError("The official test order was changed.")
            original_images, original_labels = test_images, test_labels
            local_ids = ids - len(train_labels)
        else:
            if np.any(ids < 0) or np.any(ids >= len(train_labels)):
                raise ValueError(f"Invalid original training IDs: {name}")
            original_images, original_labels, local_ids = train_images, train_labels, ids
        if not np.array_equal(images, original_images[local_ids]):
            raise ValueError(f"Images differ from their original records: {name}")
        if not np.array_equal(labels, original_labels[local_ids]):
            raise ValueError(f"Labels differ from their original records: {name}")
        all_ids.append(ids)
    combined = np.concatenate(all_ids)
    if not np.array_equal(np.sort(combined), np.arange(len(train_labels) + len(test_labels))):
        raise ValueError("Splits overlap or omit source records.")
    validation_counts = np.bincount(parts["validation"]["labels"], minlength=10)
    if not np.all(validation_counts == per_class):
        raise ValueError("Validation split is not balanced.")


def summary(parts: dict, seed: int, per_class: int) -> dict:
    return {
        "dataset": "Fashion-MNIST",
        "seed": seed,
        "validation_per_class": per_class,
        "class_names": list(CLASSES),
        "source_urls": list(BASE_URLS),
        "sources": {
            source.name: {"official_md5": source.md5, "sha256": digest(RAW / source.name)}
            for source in FILES
        },
        "splits": {
            name: {
                "samples": len(part["labels"]),
                "class_counts": np.bincount(part["labels"], minlength=10).tolist(),
                "sha256": digest(PREPARED / f"{name}.npz"),
            }
            for name, part in parts.items()
        },
        "source_id_overlap": 0,
        "official_test_unchanged": True,
        "images_dtype": "uint8",
        "images_shape": [28, 28],
    }


def prepare() -> dict:
    seed, per_class = parameters()
    raw = load_raw()
    train_ids, validation_ids = stratified_indices(raw[1], seed, per_class)
    parts = {}
    for name, ids in (("train", train_ids), ("validation", validation_ids)):
        parts[name] = {"images": raw[0][ids], "labels": raw[1][ids], "source_ids": ids}
    parts["test"] = {
        "images": raw[2], "labels": raw[3],
        "source_ids": np.arange(len(raw[3]), dtype=np.int32) + len(raw[1]),
    }
    validate_parts(parts, raw, per_class)
    for name, part in parts.items():
        atomic_write(
            PREPARED / f"{name}.npz",
            lambda stream, arrays=part: np.savez_compressed(stream, **arrays),
        )
    result = summary(parts, seed, per_class)
    content = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write(PREPARED / "summary.json", lambda stream: stream.write(content))
    return result


def check() -> dict:
    seed, per_class = parameters()
    raw = load_raw()
    parts = {}
    for name in ("train", "validation", "test"):
        with np.load(PREPARED / f"{name}.npz", allow_pickle=False) as archive:
            if set(archive.files) != {"images", "labels", "source_ids"}:
                raise ValueError(f"Unexpected arrays in {name}.npz")
            parts[name] = {key: archive[key] for key in archive.files}
    validate_parts(parts, raw, per_class)
    train_ids, validation_ids = stratified_indices(raw[1], seed, per_class)
    if not np.array_equal(parts["train"]["source_ids"], train_ids):
        raise ValueError("Training split does not match the configured seed.")
    if not np.array_equal(parts["validation"]["source_ids"], validation_ids):
        raise ValueError("Validation split does not match the configured seed.")
    result = summary(parts, seed, per_class)
    saved = json.loads((PREPARED / "summary.json").read_text(encoding="utf-8"))
    if saved != result:
        raise ValueError("Saved data summary does not match the current files.")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "prepare", "check"))
    args = parser.parse_args()
    try:
        if args.command == "download":
            for source in FILES:
                download_file(source, RAW)
            result = {"status": "ok", "files_verified": len(FILES), "directory": "data/raw"}
        else:
            report = prepare() if args.command == "prepare" else check()
            result = {
                "status": "ok",
                "samples": {name: part["samples"] for name, part in report["splits"].items()},
                "seed": report["seed"],
                "source_id_overlap": report["source_id_overlap"],
                "official_test_unchanged": report["official_test_unchanged"],
                "summary": "data/prepared/summary.json",
            }
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, EOFError, zipfile.BadZipFile, yaml.YAMLError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
