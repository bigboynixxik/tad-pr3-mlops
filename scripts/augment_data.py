"""Create and verify a reproducible augmented Fashion-MNIST training dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw
import scipy
from scipy.ndimage import rotate, shift
import yaml

from prepare_data import CLASSES, atomic_write, digest


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")
ARRAYS = {"images", "labels", "source_ids"}


def validate_config(config: object) -> dict:
    keys = {"seed", "copies_per_image", "rotation_degrees", "max_shift_pixels",
            "horizontal_flip_probability"}
    if not isinstance(config, dict) or set(config) != keys:
        raise ValueError(f"augmentation must contain exactly {sorted(keys)}.")
    for key, minimum, maximum in (
        ("seed", 0, 2**32 - 1), ("copies_per_image", 1, 10), ("max_shift_pixels", 0, 7),
    ):
        value = config[key]
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"augmentation.{key} must be an integer in [{minimum}, {maximum}].")
    for key, maximum in (("rotation_degrees", 45), ("horizontal_flip_probability", 1)):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError(f"augmentation.{key} must be a finite number in [0, {maximum}].")
    if not any(config[key] for key in ("rotation_degrees", "max_shift_pixels",
                                      "horizontal_flip_probability")):
        raise ValueError("Enable at least one augmentation transformation.")
    return dict(config)


def read_config() -> dict:
    config = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("params.yaml must contain a mapping.")
    return validate_config(config.get("augmentation"))


def write_json(path: Path, value: dict) -> None:
    content = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    atomic_write(path, lambda stream: stream.write(content))


def load_base(prepared: Path) -> tuple[dict, dict, np.ndarray]:
    manifest = json.loads((prepared / "summary.json").read_text(encoding="utf-8"))
    if manifest.get("dataset") != "Fashion-MNIST" or manifest.get("class_names") != list(CLASSES):
        raise ValueError("Unexpected dataset or class names.")
    parts, descriptions = {}, {}
    for name in SPLITS:
        path = prepared / f"{name}.npz"
        description = manifest.get("splits", {}).get(name)
        if not isinstance(description, dict) or description.get("sha256") != digest(path):
            raise ValueError(f"Checksum mismatch for original {name}.npz.")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != ARRAYS:
                raise ValueError(f"Unexpected arrays in original {name}.npz.")
            keys = ARRAYS if name == "train" else {"labels", "source_ids"}
            part = {key: archive[key] for key in keys}
        count = description.get("samples")
        if type(count) is not int or count <= 0:
            raise ValueError(f"Invalid sample count for {name}.")
        for key, dtype in (("labels", np.uint8), ("source_ids", np.int32)):
            if part[key].shape != (count,) or part[key].dtype != dtype:
                raise ValueError(f"Invalid {name}.{key} shape or dtype.")
        if name == "train" and (part["images"].shape != (count, 28, 28) or
                                 part["images"].dtype != np.uint8):
            raise ValueError("Invalid training images.")
        if not np.array_equal(np.unique(part["labels"]), np.arange(10)):
            raise ValueError(f"{name} must contain all ten classes.")
        if np.bincount(part["labels"], minlength=10).tolist() != description.get("class_counts"):
            raise ValueError(f"Invalid class counts for {name}.")
        if np.any(part["source_ids"] < 0) or np.unique(part["source_ids"]).size != count:
            raise ValueError(f"Invalid or repeated original source IDs in {name}.")
        parts[name] = part
        descriptions[name] = {"samples": count, "sha256": description["sha256"]}
    combined_ids = np.concatenate([parts[name]["source_ids"] for name in SPLITS])
    if np.unique(combined_ids).size != combined_ids.size:
        raise ValueError("Original training, validation and test source IDs overlap.")
    held_out = np.concatenate([parts[name]["source_ids"] for name in ("validation", "test")])
    return parts["train"], descriptions, held_out


def select_source(part: dict, per_class: int, seed: int) -> dict:
    if type(per_class) is not int or per_class < 0:
        raise ValueError("selection_per_class must be a nonnegative integer.")
    if per_class == 0:
        return part
    rng = np.random.default_rng(seed)
    selected = []
    for label in range(10):
        indices = np.flatnonzero(part["labels"] == label)
        if len(indices) < per_class:
            raise ValueError(f"Not enough training examples of class {label}.")
        selected.extend(rng.choice(indices, per_class, replace=False).tolist())
    indices = np.sort(selected)
    return {key: value[indices] for key, value in part.items()}


def transform(image: np.ndarray, source_id: int, variant: int, config: dict) -> np.ndarray:
    # A record has the same transformation regardless of processing order or subset size.
    rng = np.random.default_rng(np.random.SeedSequence([config["seed"], int(source_id), variant]))
    transformed = np.fliplr(image) if rng.random() < config["horizontal_flip_probability"] else image
    angle = rng.uniform(-config["rotation_degrees"], config["rotation_degrees"])
    if angle:
        transformed = rotate(transformed, angle, reshape=False, order=1,
                             mode="constant", cval=0, prefilter=False)
    displacement = rng.integers(-config["max_shift_pixels"], config["max_shift_pixels"] + 1, size=2)
    if np.any(displacement):
        transformed = shift(transformed, displacement, order=1, mode="constant", cval=0,
                            prefilter=False)
    return np.ascontiguousarray(transformed, dtype=np.uint8)


def generate(source: dict, config: dict) -> dict:
    config = validate_config(config)
    count = len(source["labels"])
    variants = config["copies_per_image"] + 1
    images = np.empty((count * variants, 28, 28), dtype=np.uint8)
    images[:count] = source["images"]
    for variant in range(1, variants):
        for index, image in enumerate(source["images"]):
            images[variant * count + index] = transform(
                image, int(source["source_ids"][index]), variant, config,
            )
            if (index + 1) % 5000 == 0:
                print(f"Augmentation copy {variant}: {index + 1}/{count}",
                      file=sys.stderr, flush=True)
    return {
        "images": images,
        "labels": np.tile(source["labels"], variants),
        "source_ids": np.tile(source["source_ids"], variants),
        "variant_ids": np.repeat(np.arange(variants, dtype=np.uint8), count),
    }


def validate_combined(part: dict, source: dict, config: dict, held_out: np.ndarray) -> None:
    count = len(source["labels"])
    variants = config["copies_per_image"] + 1
    if set(part) != ARRAYS | {"variant_ids"}:
        raise ValueError("Unexpected arrays in augmented train.npz.")
    for key, shape, dtype in (
        ("images", (count * variants, 28, 28), np.uint8),
        ("labels", (count * variants,), np.uint8),
        ("source_ids", (count * variants,), np.int32),
        ("variant_ids", (count * variants,), np.uint8),
    ):
        if part[key].shape != shape or part[key].dtype != dtype:
            raise ValueError(f"Invalid augmented {key} shape or dtype.")
    if not np.array_equal(part["images"][:count], source["images"]):
        raise ValueError("Original training images were changed.")
    for key in ("labels", "source_ids"):
        if not np.array_equal(part[key], np.tile(source[key], variants)):
            raise ValueError(f"Augmented {key} do not match their original records.")
    if not np.array_equal(part["variant_ids"], np.repeat(np.arange(variants), count)):
        raise ValueError("Invalid original/augmented variant IDs.")
    if np.intersect1d(part["source_ids"], held_out).size:
        raise ValueError("Augmented data overlap validation or test records.")


def save_preview(path: Path, part: dict, original_count: int) -> None:
    canvas = Image.new("RGB", (1040, 250), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "Fashion-MNIST: training originals and augmented copies", fill="black")
    for row, title in enumerate(("Original", "Augmented")):
        draw.text((8, 80 + row * 105), title, fill="black")
    for label, title in enumerate(CLASSES):
        index = int(np.flatnonzero(part["labels"][:original_count] == label)[0])
        x = 125 + label * 91
        draw.text((x, 30), title, fill="black")
        for row, offset in enumerate((0, original_count)):
            image = Image.fromarray(part["images"][index + offset]).resize(
                (84, 84), resample=Image.Resampling.NEAREST,
            )
            canvas.paste(image, (x, 50 + row * 105))
    atomic_write(path, lambda stream: canvas.save(stream, format="PNG"))


def build(prepared: Path, output: Path, config: dict, per_class: int = 0) -> dict:
    config = validate_config(config)
    original, descriptions, held_out = load_base(prepared)
    source = select_source(original, per_class, config["seed"])
    print(f"Augmenting {len(source['labels'])} original training images",
          file=sys.stderr, flush=True)
    combined = generate(source, config)
    validate_combined(combined, source, config, held_out)
    atomic_write(output / "train.npz", lambda stream: np.savez_compressed(stream, **combined))
    save_preview(output / "preview.png", combined, len(source["labels"]))
    for name in SPLITS:
        if digest(prepared / f"{name}.npz") != descriptions[name]["sha256"]:
            raise ValueError(f"Original {name} changed during augmentation.")
    report = {
        "format_version": 1,
        "dataset": "Fashion-MNIST",
        "class_names": list(CLASSES),
        "purpose": "functional_smoke_check" if per_class else "augmented_training",
        "selection_per_class": per_class,
        "augmentation": config,
        "source_files": descriptions,
        "train": {
            "original_samples": len(source["labels"]),
            "augmented_samples": len(combined["labels"]) - len(source["labels"]),
            "samples": len(combined["labels"]),
            "class_counts": np.bincount(combined["labels"], minlength=10).tolist(),
            "unique_source_records": np.unique(combined["source_ids"]).size,
            "sha256": digest(output / "train.npz"),
        },
        "source_id_overlap": 0,
        "held_out_splits_unchanged": True,
        "interpolation": "linear_constant_zero",
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "preview_sha256": digest(output / "preview.png"),
    }
    # NumPy scalar counts must be plain Python integers in the JSON manifest.
    report["train"]["unique_source_records"] = int(report["train"]["unique_source_records"])
    write_json(output / "summary.json", report)
    return report


def load_augmented(prepared: Path, output: Path, config: dict,
                   verify_pixels: bool = False) -> tuple[dict, dict]:
    config = validate_config(config)
    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    original, descriptions, held_out = load_base(prepared)
    if saved.get("augmentation") != config or saved.get("source_files") != descriptions:
        raise ValueError("Augmentation parameters or source files changed; reproduce augment.")
    if saved.get("versions") != {"numpy": np.__version__, "scipy": scipy.__version__}:
        raise ValueError("Augmentation library versions differ; use the locked environment.")
    if saved.get("dataset") != "Fashion-MNIST" or saved.get("format_version") != 1:
        raise ValueError("Unsupported augmented dataset format.")
    if digest(output / "train.npz") != saved.get("train", {}).get("sha256"):
        raise ValueError("Augmented training checksum mismatch.")
    if digest(output / "preview.png") != saved.get("preview_sha256"):
        raise ValueError("Augmentation preview checksum mismatch.")
    source = select_source(original, saved["selection_per_class"], config["seed"])
    with np.load(output / "train.npz", allow_pickle=False) as archive:
        part = {key: archive[key] for key in archive.files}
    validate_combined(part, source, config, held_out)
    count = len(source["labels"])
    actual_counts = {
        "original_samples": count,
        "augmented_samples": len(part["labels"]) - count,
        "samples": len(part["labels"]),
        "class_counts": np.bincount(part["labels"], minlength=10).tolist(),
        "unique_source_records": int(np.unique(part["source_ids"]).size),
        "sha256": digest(output / "train.npz"),
    }
    if saved["train"] != actual_counts:
        raise ValueError("Augmented sample counts do not match summary.json.")
    purpose = "functional_smoke_check" if saved["selection_per_class"] else "augmented_training"
    if (saved.get("purpose") != purpose or saved.get("class_names") != list(CLASSES)
            or saved.get("source_id_overlap") != 0
            or saved.get("held_out_splits_unchanged") is not True
            or saved.get("interpolation") != "linear_constant_zero"):
        raise ValueError("Augmentation summary metadata is inconsistent.")
    if verify_pixels:
        for variant in range(1, config["copies_per_image"] + 1):
            for index, image in enumerate(source["images"]):
                expected = transform(image, int(source["source_ids"][index]), variant, config)
                if not np.array_equal(part["images"][variant * count + index], expected):
                    raise ValueError("Augmented pixels do not match the recorded seed and parameters.")
                if (index + 1) % 5000 == 0:
                    print(f"Verifying copy {variant}: {index + 1}/{count}", file=sys.stderr, flush=True)
    return part, saved


def check(prepared: Path, output: Path, config: dict) -> dict:
    return load_augmented(prepared, output, config, verify_pixels=True)[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "smoke"))
    args = parser.parse_args()
    try:
        config = read_config()
        prepared = ROOT / "data" / "prepared"
        output = ROOT / "data" / "augmented"
        if args.command == "smoke":
            output = ROOT / "results" / "augmentation-smoke"
            build(prepared, output, config, per_class=50)
            report = check(prepared, output, config)
        elif args.command == "prepare":
            report = build(prepared, output, config)
        else:
            report = check(prepared, output, config)
        print(json.dumps({
            "status": "ok", "purpose": report["purpose"], "train": report["train"],
            "validation_samples": report["source_files"]["validation"]["samples"],
            "test_samples": report["source_files"]["test"]["samples"],
            "source_id_overlap": report["source_id_overlap"],
            "held_out_splits_unchanged": report["held_out_splits_unchanged"],
            "output": str(output.relative_to(ROOT)).replace("\\", "/"),
        }, indent=2))
        return 0
    except (OSError, ValueError, EOFError, KeyError, TypeError, zipfile.BadZipFile, yaml.YAMLError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
