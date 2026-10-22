"""Freeze the validation-selected model, evaluate it on test data and render figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import zipfile

import joblib
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from threadpoolctl import threadpool_limits

from model import PREPROCESSING, features, load_split, progress, write_json
from prepare_data import CLASSES, atomic_write, digest


ROOT = Path(__file__).resolve().parents[1]
LABELS_RU = (
    "Футболка/топ", "Брюки", "Пуловер", "Платье", "Пальто",
    "Сандалии", "Рубашка", "Кроссовки", "Сумка", "Ботильоны",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def choose(comparison: dict) -> dict:
    if comparison.get("evaluation_split") != "validation" or comparison.get("selection_metric") != "macro_f1":
        raise ValueError("Model selection must use validation macro-F1.")
    candidates = [pair[kind] for pair in comparison["comparisons"] for kind in ("original", "augmented")]
    if not candidates:
        raise ValueError("No validation experiments to select.")
    for candidate in candidates:
        score = candidate["metrics"]["macro_f1"]
        if type(score) not in (int, float) or not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid validation macro-F1.")
        if candidate["metrics"]["converged"] != 1:
            raise ValueError("Final comparison contains an unconverged model.")
    selected = max(candidates, key=lambda item: item["metrics"]["macro_f1"])
    if "best_validation_candidate" in comparison:
        matches_selection = selected.get("candidate_id") == comparison["best_validation_candidate"]
    else:
        matches_selection = selected["experiment"] == comparison.get("best_validation_experiment")
    if not matches_selection:
        raise ValueError("Declared selection differs from validation results.")
    return selected


def freeze(current: Path, validation_metrics: Path, comparison_path: Path,
           prepared: Path, output: Path) -> dict:
    comparison = read_json(comparison_path)
    selected = choose(comparison)
    metadata = read_json(current / "training.json")
    metrics = read_json(validation_metrics)
    source = read_json(prepared / "summary.json")
    if metadata.get("model_sha256") != digest(current / "model.joblib"):
        raise ValueError("Current model checksum mismatch.")
    expected = selected["metrics"]
    if (metadata["model"] != selected["model"]
            or metadata["parameters"] != selected["parameters"]
            or metadata["train"]["seed"] != selected["seed"]
            or metadata["train"].get("dataset", "prepared") != selected["dataset"]
            or metadata["train"]["limit_per_class"] != 0
            or metadata["training_samples"] != expected["training_samples"]):
        raise ValueError("Current model differs from the validation-selected experiment; restore it first.")
    for key in ("accuracy", "macro_f1", "training_samples", "validation_samples", "converged"):
        if metrics.get(key) != expected[key]:
            raise ValueError(f"Current validation {key} differs from the selected experiment.")
    if comparison["validation_sha256"] != source["splits"]["validation"]["sha256"]:
        raise ValueError("Validation dataset differs from the selection dataset.")
    if selected["dataset"] == "prepared" and metadata["training_file_sha256"] != source["splits"]["train"]["sha256"]:
        raise ValueError("Training data differ from the selected experiment.")
    selection = {
        "experiment": selected["experiment"],
        "experiment_revision": selected["revision"],
        "selection_metric": "macro_f1",
        "selection_split": "validation",
        "validation_accuracy": expected["accuracy"],
        "validation_macro_f1": expected["macro_f1"],
        "validation_sha256": comparison["validation_sha256"],
        "model_sha256": metadata["model_sha256"],
        "training_metadata_sha256": digest(current / "training.json"),
        "decision_made_before_test": True,
    }
    if "candidate_id" in selected:
        selection["candidate_id"] = selected["candidate_id"]
        selection["execution_mode"] = comparison["execution_mode"]
    for name in ("model.joblib", "training.json"):
        with (current / name).open("rb") as stream:
            atomic_write(output / name, lambda target: shutil.copyfileobj(stream, target))
    write_json(output / "selection.json", selection)
    return selection


def test_selected(selected: Path, prepared: Path, metrics_path: Path,
                  report_directory: Path, summary_path: Path) -> dict:
    selection = read_json(selected / "selection.json")
    saved = read_json(selected / "training.json")
    if (selection.get("decision_made_before_test") is not True
            or selection.get("selection_split") != "validation"
            or selection.get("selection_metric") != "macro_f1"):
        raise ValueError("The model must be selected on validation before test evaluation.")
    if (selection["model_sha256"] != digest(selected / "model.joblib")
            or selection["model_sha256"] != saved["model_sha256"]
            or selection["training_metadata_sha256"] != digest(selected / "training.json")):
        raise ValueError("The selected model or its training metadata changed.")
    artifact = joblib.load(selected / "model.joblib")
    metadata = artifact["metadata"]
    if saved != {**metadata, "model_sha256": selection["model_sha256"]}:
        raise ValueError("Selected model metadata mismatch.")
    if metadata.get("preprocessing") != PREPROCESSING or metadata.get("format_version") != 1:
        raise ValueError("Unsupported model format.")
    if metadata["versions"] != {"numpy": np.__version__, "scikit_learn": sklearn.__version__}:
        raise ValueError("Use the locked model library versions.")
    part = load_split(prepared, "test")
    if np.intersect1d(artifact["training_source_ids"], part["source_ids"]).size:
        raise ValueError("Test records overlap training records.")
    with progress(f"Testing selected {metadata['model']} on {len(part['labels'])} images"), threadpool_limits(limits=1):
        predicted = artifact["pipeline"].predict(features(part["images"]))
    metrics = {
        "accuracy": float(accuracy_score(part["labels"], predicted)),
        "macro_f1": float(f1_score(part["labels"], predicted, labels=np.arange(10),
                                  average="macro", zero_division=0)),
        "training_samples": metadata["training_samples"],
        "test_samples": len(predicted),
    }
    details = {
        "model": metadata["model"],
        "evaluation_split": "test",
        "class_names": list(CLASSES),
        "confusion_matrix": confusion_matrix(part["labels"], predicted, labels=np.arange(10)).tolist(),
        "classification_report": classification_report(
            part["labels"], predicted, labels=np.arange(10), target_names=CLASSES,
            output_dict=True, zero_division=0,
        ),
        "test_sha256": part["sha256"],
        "selection": selection,
    }
    summary = {
        "evaluation_split": "test",
        "test_used_for_selection": False,
        "test_sha256": part["sha256"],
        "source_id_overlap": 0,
        "selection": selection,
        "metrics": metrics,
        "model": metadata["model"],
        "dataset": metadata["train"].get("dataset", "prepared"),
        "parameters": metadata["parameters"],
        "seed": metadata["train"]["seed"],
        "epochs_completed": int(artifact["pipeline"].named_steps["classifier"].n_iter_)
            if metadata["model"] == "mlp" else None,
    }
    atomic_write(report_directory / "test_predictions.npz", lambda stream: np.savez_compressed(
        stream, source_ids=part["source_ids"], y_true=part["labels"], y_pred=predicted,
    ))
    write_json(report_directory / "test_details.json", details)
    write_json(metrics_path, metrics)
    write_json(summary_path, summary)
    return summary


def plots(comparison_path: Path, details_path: Path, output: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    comparison = read_json(comparison_path)
    details = read_json(details_path)
    if details.get("evaluation_split") != "test":
        raise ValueError("Expected final test details.")
    pairs = comparison["comparisons"]
    model_title = {"mlp": "MLP", "random_forest": "случайный лес",
                   "logreg": "логистическая регрессия"}[details["model"]]
    titles = {"logreg": "Логистическая\nрегрессия", "random_forest": "Случайный лес", "mlp": "MLP"}
    names = [titles[pair["model"]] for pair in pairs]
    colors = ("#2463A6", "#DE8F33")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "tad-pr3-results"})
    output.mkdir(parents=True, exist_ok=True)

    def save(figure, name):
        for suffix in ("png", "svg"):
            path = output / f"{name}.{suffix}"
            metadata = {"Software": "tad-pr3-mlops"} if suffix == "png" else {"Date": None}
            atomic_write(path, lambda stream: figure.savefig(
                stream, format=suffix, dpi=300, bbox_inches="tight", metadata=metadata,
            ))
        plt.close(figure)

    x = np.arange(len(pairs))
    width = 0.34
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for axis, key, title, multiplier, limit in (
        (axes[0], "accuracy", "Accuracy, %", 100, 100),
        (axes[1], "macro_f1", "Macro-F1", 1, 1),
    ):
        for offset, kind, label, color in (
            (-width/2, "original", "Исходные данные", colors[0]),
            (width/2, "augmented", "С аугментацией", colors[1]),
        ):
            values = [pair[kind]["metrics"][key] * multiplier for pair in pairs]
            bars = axis.bar(x + offset, values, width, label=label, color=color)
            axis.bar_label(bars, fmt="%.2f" if key == "accuracy" else "%.3f", padding=3, fontsize=9)
        axis.set_xticks(x, names)
        axis.set_ylim(0, limit)
        axis.set_ylabel(title)
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(loc="lower left")
    figure.suptitle("Качество на одной валидационной выборке: 10 000 изображений")
    save(figure, "validation_quality")

    figure, axis = plt.subplots(figsize=(8, 4.7), layout="constrained")
    for offset, kind, label, color in (
        (-width/2, "original", "50 000 изображений", colors[0]),
        (width/2, "augmented", "100 000 изображений", colors[1]),
    ):
        values = [pair[kind]["metrics"]["fit_seconds"] for pair in pairs]
        bars = axis.bar(x + offset, values, width, label=label, color=color)
        axis.bar_label(bars, fmt="%.1f", padding=3)
    axis.set_xticks(x, names)
    axis.set_ylim(0, max(pair["augmented"]["metrics"]["fit_seconds"] for pair in pairs) * 1.22)
    axis.set_ylabel("Время обучения, с")
    axis.set_title("Наблюдаемое время обучения в выполненных запусках")
    axis.legend(loc="upper left")
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=0.2)
    save(figure, "training_time")

    matrix = np.asarray(details["confusion_matrix"])
    if matrix.shape != (10, 10) or np.any(matrix < 0):
        raise ValueError("Expected a nonnegative 10 x 10 confusion matrix.")
    figure, axis = plt.subplots(figsize=(10, 8), layout="constrained")
    heatmap = axis.imshow(matrix, cmap="Blues", vmin=0)
    for row in range(10):
        for column in range(10):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center",
                      fontsize=9, color="white" if matrix[row, column] > matrix.max()/2 else "#162C46")
    axis.set_xticks(np.arange(10), LABELS_RU, rotation=45, ha="right")
    axis.set_yticks(np.arange(10), LABELS_RU)
    axis.set_xlabel("Предсказанный класс")
    axis.set_ylabel("Правильный класс")
    axis.set_title(f"Матрица ошибок: {model_title}, тестовая выборка")
    figure.colorbar(heatmap, ax=axis, label="Число изображений", shrink=0.8)
    save(figure, "test_confusion_matrix")

    values = [details["classification_report"][name]["f1-score"] for name in CLASSES]
    figure, axis = plt.subplots(figsize=(8, 5), layout="constrained")
    bars = axis.barh(np.arange(10), values, color=colors[0])
    axis.bar_label(bars, fmt="%.3f", padding=4)
    axis.set_yticks(np.arange(10), LABELS_RU)
    axis.invert_yaxis()
    axis.set_xlim(0, 1.08)
    axis.set_xlabel("F1")
    axis.set_title(f"F1 по классам: {model_title}, тестовая выборка")
    axis.xaxis.set_major_locator(MaxNLocator(6))
    axis.set_axisbelow(True)
    axis.grid(axis="x", alpha=0.2)
    save(figure, "test_class_f1")
    return {"status": "ok", "figures": 4, "formats": ["png", "svg"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "test", "plots"))
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            result = freeze(ROOT / "models/current", ROOT / "metrics/validation.json",
                            ROOT / "results/validation_comparison.json", ROOT / "data/prepared",
                            ROOT / "models/selected")
        elif args.command == "test":
            result = test_selected(ROOT / "models/selected", ROOT / "data/prepared",
                                   ROOT / "metrics/test.json", ROOT / "reports",
                                   ROOT / "results/final_test.json")
        else:
            result = plots(ROOT / "results/validation_comparison.json",
                           ROOT / "reports/test_details.json", ROOT / "results/figures")
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (OSError, ValueError, EOFError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
