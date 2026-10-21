"""Check pinned packages, scientific imports and the mounted Git/DVC project."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys


LOCK_FILE = Path(__file__).resolve().parents[1] / "requirements.lock"
IMPORTS = {
    "dvc": "dvc",
    "numpy": "numpy",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
    "Pillow": "PIL.Image",
    "PyYAML": "yaml",
}


def command(*args: str) -> str:
    result = subprocess.run(
        args, capture_output=True, text=True, encoding="utf-8", timeout=30
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def locked_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for line in LOCK_FILE.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or line[0].isspace():
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;\\]+)(?:\s*\\)?", line)
        if not match:
            raise RuntimeError(f"Unsupported lock-file entry: {line}")
        packages[match.group(1)] = match.group(2)
    if not packages:
        raise RuntimeError("The dependency lock file is empty.")
    return packages


def main() -> int:
    try:
        if sys.version_info[:2] != (3, 13):
            raise RuntimeError(f"Python 3.13 is required; found {platform.python_version()}.")
        expected_python = os.environ.get("EXPECTED_PYTHON_VERSION")
        if expected_python and platform.python_version() != expected_python:
            raise RuntimeError(f"Expected Python {expected_python}.")

        workspace = Path.cwd().resolve()
        workspace_lock = workspace / "requirements.lock"
        if workspace_lock.read_bytes() != LOCK_FILE.read_bytes():
            raise RuntimeError("requirements.lock changed; rebuild the Docker image.")

        expected = locked_packages()
        installed: dict[str, str] = {}
        for package, wanted in expected.items():
            found = importlib.metadata.version(package)
            if found != wanted:
                raise RuntimeError(f"{package}: expected {wanted}, found {found}.")
            installed[package.lower()] = found

        for module in IMPORTS.values():
            importlib.import_module(module)
        for module in (
            "sklearn.linear_model",
            "sklearn.ensemble",
            "sklearn.neural_network",
            "matplotlib.pyplot",
        ):
            importlib.import_module(module)

        git_version = command("git", "--version")
        git_root = Path(command("git", "rev-parse", "--show-toplevel")).resolve()
        if git_root != workspace:
            raise RuntimeError("Run from the mounted project root.")
        dvc_root = Path(command(sys.executable, "-m", "dvc", "root")).resolve()
        if dvc_root != workspace:
            raise RuntimeError("DVC must be initialized in the project root.")

        print(json.dumps({
            "status": "ok",
            "python": platform.python_version(),
            "platform": platform.system(),
            "git": git_version,
            "repository": str(workspace),
            "locked_packages_checked": len(expected),
            "packages": {name: installed[name.lower()] for name in IMPORTS},
        }, indent=2))
        return 0
    except (
        OSError,
        RuntimeError,
        ImportError,
        importlib.metadata.PackageNotFoundError,
        subprocess.TimeoutExpired,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
