from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


REPO_URL = "https://github.com/casparbreloh/rbt4dnn-seminar.git"
REMOTE_ROOT = Path("/content/rbt4dnn-seminar")
ZIP_PATH = Path("/content/rbt4dnn-celeba-results.zip")


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    if REMOTE_ROOT.exists():
        shutil.rmtree(REMOTE_ROOT)
    run(["git", "clone", "--depth", "1", "--branch", "main", REPO_URL, str(REMOTE_ROOT)])
    commit = subprocess.check_output(
        ["git", "-C", str(REMOTE_ROOT), "log", "-1", "--oneline"],
        text=True,
    ).strip()
    print(commit, flush=True)

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "transformers>=4.45",
            "pillow>=10.0",
        ]
    )
    run(
        [
            sys.executable,
            "scripts/run_experiments.py",
            "--only",
            "replication",
            "--train-celeba-generator",
            "--celeba-generator-epochs",
            "120",
            "--celeba-generator-samples",
            "48",
            "--celeba-generator-seeds",
            "7,13,29",
        ],
        cwd=REMOTE_ROOT,
    )

    out_dir = REMOTE_ROOT / "experiments" / "celeba-shared-generator"
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in [
            "results.csv",
            "seed-results.csv",
            "summary.md",
            "training-log.csv",
            "samples.png",
        ]:
            path = out_dir / name
            archive.write(path, path.relative_to(REMOTE_ROOT))
    print(ZIP_PATH, flush=True)


if __name__ == "__main__":
    main()
