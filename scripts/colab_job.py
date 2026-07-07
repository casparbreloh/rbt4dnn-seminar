from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


REPO_URL = "https://github.com/casparbreloh/rbt4dnn-seminar.git"
REMOTE_ROOT = Path("/content/rbt4dnn-seminar")
ZIP_PATH = Path("/content/rbt4dnn-seminar-results.zip")


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def prepare_repo(branch: str) -> None:
    if REMOTE_ROOT.exists():
        shutil.rmtree(REMOTE_ROOT)
    run(["git", "clone", "--depth", "1", "--branch", branch, REPO_URL, str(REMOTE_ROOT)])
    commit = subprocess.check_output(
        ["git", "-C", str(REMOTE_ROOT), "log", "-1", "--oneline"],
        text=True,
    ).strip()
    print(commit, flush=True)


def install_runtime_deps() -> None:
    run([sys.executable, "-m", "pip", "install", "-q", "transformers>=4.45", "pillow>=10.0"])


def run_experiments(epochs: int, samples: int) -> None:
    run(
        [
            sys.executable,
            "scripts/run_experiments.py",
            "--train-shared-generator",
            "--shared-generator-epochs",
            str(epochs),
            "--shared-generator-samples",
            str(samples),
        ],
        cwd=REMOTE_ROOT,
    )


def zip_results() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for pattern in ["results.csv", "summary.md", "training-log.csv", "samples.png"]:
            for path in sorted((REMOTE_ROOT / "experiments").glob(f"*/{pattern}")):
                archive.write(path, path.relative_to(REMOTE_ROOT))
        for path in [
            REMOTE_ROOT / "experiments" / "cost-analysis" / "assumptions.csv",
            REMOTE_ROOT / "experiments" / "cost-analysis" / "valid-failures.csv",
        ]:
            if path.exists():
                archive.write(path, path.relative_to(REMOTE_ROOT))
    print(ZIP_PATH, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default="main")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()

    prepare_repo(args.branch)
    install_runtime_deps()
    run_experiments(args.epochs, args.samples)
    zip_results()


if __name__ == "__main__":
    main()
