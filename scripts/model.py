"""Train and validate three Fashion-MNIST classifiers with DVC."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time
import warnings
import zipfile

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import yaml

from prepare_data import CLASSES, atomic_write, digest
from augment_data import load_augmented, validate_config as validate_augmentation


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ("logreg", "random_forest", "mlp")
PREPROCESSING = "uint8_flatten_div255_v1"


def integer(value: object, name: str, minimum: int = 1, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer >= {minimum}" +
                         (f" and <= {maximum}." if maximum is not None else "."))
    return value


def positive(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number.")
    return float(value)


def validate_config(config: object) -> dict:
    if not isinstance(config, dict):
        raise ValueError("params.yaml must contain a mapping.")
    result = {}
    for name in ("train", *MODEL_NAMES):
        if not isinstance(config.get(name), dict):
            raise ValueError(f"params.yaml must contain a {name} mapping.")
        result[name] = deepcopy(config[name])
    train = result["train"]
    train.setdefault("dataset", "prepared")
    if train["dataset"] not in ("prepared", "augmented"):
        raise ValueError("train.dataset must be prepared or augmented.")
    if train["dataset"] == "augmented":
        result["augmentation"] = validate_augmentation(config.get("augmentation"))
    if train.get("model") not in MODEL_NAMES:
        raise ValueError(f"train.model must be one of {MODEL_NAMES}.")
    integer(train.get("seed"), "train.seed", 0, 2**32 - 1)
    integer(train.get("limit_per_class"), "train.limit_per_class", 0)
    for name in ("logreg", "mlp"):
        integer(result[name].get("max_iter"), f"{name}.max_iter")
        positive(result[name].get("tol"), f"{name}.tol")
    positive(result["logreg"].get("C"), "logreg.C")
    forest = result["random_forest"]
    integer(forest.get("n_estimators"), "random_forest.n_estimators")
    integer(forest.get("min_samples_leaf"), "random_forest.min_samples_leaf")
    if forest.get("max_depth") is not None:
        integer(forest["max_depth"], "random_forest.max_depth")
    mlp = result["mlp"]
    layers = mlp.get("hidden_layer_sizes")
    if not isinstance(layers, list) or not layers:
        raise ValueError("mlp.hidden_layer_sizes must be a nonempty list.")
    for size in layers:
        integer(size, "mlp.hidden_layer_sizes entry")
    integer(mlp.get("batch_size"), "mlp.batch_size")
    for name in ("alpha", "learning_rate_init"):
        positive(mlp.get(name), f"mlp.{name}")
    # Reject misspelled or unsupported parameters instead of silently ignoring them.
    expected = {
        "train": {"model", "seed", "limit_per_class", "dataset"},
        "logreg": {"C", "max_iter", "tol"},
        "random_forest": {"n_estimators", "max_depth", "min_samples_leaf"},
        "mlp": {"hidden_layer_sizes", "max_iter", "batch_size", "learning_rate_init", "alpha", "tol"},
    }
    for name, keys in expected.items():
        if set(result[name]) != keys:
            raise ValueError(f"Unexpected or missing parameters in {name}; expected {sorted(keys)}.")
    return result


def read_config(path: Path) -> dict:
    return validate_config(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_split(prepared: Path, name: str) -> dict:
    manifest = json.loads((prepared / "summary.json").read_text(encoding="utf-8"))
    if manifest.get("dataset") != "Fashion-MNIST" or manifest.get("class_names") != list(CLASSES):
        raise ValueError("Unexpected dataset or class names in summary.json.")
    description = manifest.get("splits", {}).get(name)
    path = prepared / f"{name}.npz"
    if not isinstance(description, dict) or description.get("sha256") != digest(path):
        raise ValueError(f"Checksum mismatch for {name}.npz.")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"images", "labels", "source_ids"}:
            raise ValueError(f"Unexpected arrays in {name}.npz.")
        part = {key: archive[key] for key in archive.files}
    count = description.get("samples")
    if type(count) is not int or count <= 0:
        raise ValueError(f"Invalid sample count for {name}.")
    for key, shape, dtype in (
        ("images", (count, 28, 28), np.uint8),
        ("labels", (count,), np.uint8),
        ("source_ids", (count,), np.int32),
    ):
        if part[key].shape != shape or part[key].dtype != dtype:
            raise ValueError(f"Invalid shape or dtype for {name}.{key}.")
    if np.any(part["labels"] > 9) or not np.array_equal(np.unique(part["labels"]), np.arange(10)):
        raise ValueError(f"{name} must contain exactly classes 0 through 9.")
    if np.unique(part["source_ids"]).size != count or np.any(part["source_ids"] < 0):
        raise ValueError(f"Invalid or repeated source IDs in {name}.")
    if np.bincount(part["labels"], minlength=10).tolist() != description.get("class_counts"):
        raise ValueError(f"Class counts do not match summary.json for {name}.")
    part["sha256"] = description["sha256"]
    return part


def subset(part: dict, per_class: int, seed: int) -> dict:
    integer(per_class, "limit_per_class", 0)
    integer(seed, "seed", 0, 2**32 - 1)
    if per_class == 0:
        return part
    rng = np.random.default_rng(seed)
    selected = []
    originals = part.get("variant_ids", np.zeros(len(part["labels"]), dtype=np.uint8)) == 0
    for label in range(10):
        indices = np.flatnonzero((part["labels"] == label) & originals)
        if len(indices) < per_class:
            raise ValueError(f"limit_per_class exceeds available original samples for class {label}.")
        selected.extend(rng.choice(indices, size=per_class, replace=False).tolist())
    selected = np.sort(selected)
    if "variant_ids" in part:
        # Keep every copy of a selected original record together.
        selected = np.flatnonzero(np.isin(part["source_ids"], part["source_ids"][selected]))
    return {key: value[selected] if isinstance(value, np.ndarray) else value
            for key, value in part.items()}


def features(images: np.ndarray) -> np.ndarray:
    if images.ndim != 3 or images.shape[1:] != (28, 28) or images.dtype != np.uint8:
        raise ValueError("Expected uint8 images of shape (N, 28, 28).")
    return images.reshape(len(images), 784).astype(np.float32) / np.float32(255)


def estimator(config: dict) -> Pipeline:
    name = config["train"]["model"]
    seed = config["train"]["seed"]
    parameters = deepcopy(config[name])
    if name == "logreg":
        classifier = LogisticRegression(solver="lbfgs", random_state=seed, **parameters)
    elif name == "random_forest":
        classifier = RandomForestClassifier(n_jobs=1, random_state=seed, **parameters)
    else:
        parameters["hidden_layer_sizes"] = tuple(parameters["hidden_layer_sizes"])
        classifier = MLPClassifier(early_stopping=False, random_state=seed, **parameters)
    steps = [] if name == "random_forest" else [("scale", StandardScaler())]
    return Pipeline([*steps, ("classifier", classifier)])


@contextmanager
def progress(message: str):
    stop = threading.Event()
    started = time.perf_counter()

    def heartbeat():
        while not stop.wait(15):
            print(f"{message}: still running ({time.perf_counter() - started:.0f} s)",
                  file=sys.stderr, flush=True)

    print(message, file=sys.stderr, flush=True)
    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join()


def write_json(path: Path, value: dict) -> None:
    content = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    atomic_write(path, lambda stream: stream.write(content))


def train(config: dict, prepared: Path, directory: Path, *,
          fit_barrier=None, fit_trace: dict | None = None) -> dict:
    config = validate_config(config)
    name = config["train"]["model"]
    if config["train"]["dataset"] == "augmented":
        loaded, report = load_augmented(prepared, prepared.parent / "augmented", config["augmentation"])
        if report["purpose"] != "augmented_training":
            raise ValueError("Full training requires the full augmented dataset, not smoke outputs.")
        loaded["sha256"] = report["train"]["sha256"]
    else:
        loaded = load_split(prepared, "train")
    part = subset(loaded, config["train"]["limit_per_class"], config["train"]["seed"])
    pipeline = estimator(config)
    inputs = features(part["images"])
    # A parallel group releases all three workers only after their inputs are ready.
    if fit_barrier is not None:
        fit_barrier.wait(timeout=180)
    started = time.perf_counter()
    with progress(f"Training {name} on {len(inputs)} images"), threadpool_limits(limits=1):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            if fit_trace is not None:
                fit_trace["fit_started_unix"] = time.time()
            try:
                pipeline.fit(inputs, part["labels"])
            finally:
                if fit_trace is not None:
                    fit_trace["fit_finished_unix"] = time.time()
    warning_messages = sorted({str(item.message) for item in caught
                               if issubclass(item.category, ConvergenceWarning)})
    for item in caught:
        if not issubclass(item.category, ConvergenceWarning):
            warnings.warn_explicit(item.message, item.category, item.filename, item.lineno)
    for message in warning_messages:
        print(f"ConvergenceWarning: {message}", file=sys.stderr, flush=True)
    metadata = {
        "format_version": 1,
        "preprocessing": PREPROCESSING,
        "model": name,
        "parameters": config[name],
        "train": config["train"],
        "training_samples": len(inputs),
        "dataset": config["train"]["dataset"],
        "training_original_samples": int(np.count_nonzero(part["variant_ids"] == 0))
            if "variant_ids" in part else len(inputs),
        "training_augmented_samples": int(np.count_nonzero(part["variant_ids"] != 0))
            if "variant_ids" in part else 0,
        "unique_training_source_records": int(np.unique(part["source_ids"]).size),
        "augmentation": config.get("augmentation"),
        "training_class_counts": np.bincount(part["labels"], minlength=10).tolist(),
        "training_file_sha256": part["sha256"],
        "training_source_ids_sha256": hashlib.sha256(
            part["source_ids"].astype("<i4").tobytes()).hexdigest(),
        "fit_seconds": time.perf_counter() - started,
        "converged": not warning_messages,
        "convergence_warnings": warning_messages,
        "versions": {"numpy": np.__version__, "scikit_learn": sklearn.__version__},
    }
    artifact = {"metadata": metadata, "pipeline": pipeline, "training_source_ids": part["source_ids"]}
    atomic_write(directory / "model.joblib", lambda stream: joblib.dump(artifact, stream, compress=3))
    write_json(directory / "training.json", {
        **metadata, "model_sha256": digest(directory / "model.joblib"),
    })
    return metadata


def evaluate(prepared: Path, directory: Path, metrics_path: Path,
             report_directory: Path, limit_per_class: int = 0) -> dict:
    saved = json.loads((directory / "training.json").read_text(encoding="utf-8"))
    if saved.get("model_sha256") != digest(directory / "model.joblib"):
        raise ValueError("Model checksum does not match training.json.")
    # joblib is used only for artifacts produced locally by this project's training stage.
    artifact = joblib.load(directory / "model.joblib")
    metadata = artifact["metadata"]
    if saved != {**metadata, "model_sha256": saved["model_sha256"]}:
        raise ValueError("Model metadata does not match training.json.")
    if metadata.get("format_version") != 1 or metadata.get("preprocessing") != PREPROCESSING:
        raise ValueError("Unsupported model artifact format.")
    if metadata["versions"] != {"numpy": np.__version__, "scikit_learn": sklearn.__version__}:
        raise ValueError("Model library versions differ; restore the locked environment.")
    part = subset(load_split(prepared, "validation"), limit_per_class, metadata["train"]["seed"])
    if np.intersect1d(artifact["training_source_ids"], part["source_ids"]).size:
        raise ValueError("Validation records overlap with training records.")
    with progress(f"Validating {metadata['model']} on {len(part['labels'])} images"), \
            threadpool_limits(limits=1):
        predicted = artifact["pipeline"].predict(features(part["images"]))
    metrics = {
        "accuracy": float(accuracy_score(part["labels"], predicted)),
        "macro_f1": float(f1_score(part["labels"], predicted, labels=np.arange(10),
                                  average="macro", zero_division=0)),
        "training_samples": metadata["training_samples"],
        "validation_samples": len(predicted),
        "fit_seconds": metadata["fit_seconds"],
        "converged": int(metadata["converged"]),
    }
    details = {
        "model": metadata["model"],
        "evaluation_split": "validation",
        "validation_file_sha256": part["sha256"],
        "class_names": list(CLASSES),
        "confusion_matrix": confusion_matrix(part["labels"], predicted, labels=np.arange(10)).tolist(),
        "classification_report": classification_report(
            part["labels"], predicted, labels=np.arange(10), target_names=CLASSES,
            output_dict=True, zero_division=0,
        ),
        "training": metadata,
    }
    atomic_write(report_directory / "validation_predictions.npz", lambda stream: np.savez_compressed(
        stream, source_ids=part["source_ids"], y_true=part["labels"], y_pred=predicted,
    ))
    write_json(report_directory / "validation_details.json", details)
    write_json(metrics_path, metrics)
    return metrics


def smoke(config: dict) -> dict:
    output = ROOT / "results" / "smoke"
    result = {}
    for name in MODEL_NAMES:
        small = deepcopy(config)
        small["train"].update(model=name, limit_per_class=20)
        small["logreg"].update(max_iter=100)
        small["random_forest"].update(n_estimators=5, max_depth=6)
        small["mlp"].update(hidden_layer_sizes=[32], max_iter=5, batch_size=64)
        directory = output / name
        train(small, ROOT / "data" / "prepared", directory / "model")
        result[name] = evaluate(ROOT / "data" / "prepared", directory / "model",
                                directory / "metrics.json", directory / "reports", 10)
    write_json(output / "summary.json", {"purpose": "functional_smoke_check", "models": result})
    return {"status": "ok", "purpose": "functional_smoke_check", "models": result,
            "output": "results/smoke"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate", "smoke"))
    args = parser.parse_args()
    try:
        if args.command == "evaluate":
            result = evaluate(ROOT / "data" / "prepared", ROOT / "models" / "current",
                              ROOT / "metrics" / "validation.json", ROOT / "reports")
        else:
            config = read_config(ROOT / "params.yaml")
            result = smoke(config) if args.command == "smoke" else train(
                config, ROOT / "data" / "prepared", ROOT / "models" / "current")
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, EOFError, KeyError, TypeError, zipfile.BadZipFile, yaml.YAMLError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
