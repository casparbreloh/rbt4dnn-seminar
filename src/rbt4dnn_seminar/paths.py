from __future__ import annotations

import csv
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path

REPO_URL = "https://github.com/casparbreloh/rbt4dnn-seminar.git"
REPO_NAME = "rbt4dnn-seminar"
ROOT_MARKER = Path("data/requirements.csv")

CsvRow = dict[str, str]


def find_repo_root(start: Path | None = None) -> Path:
    base = (start or Path.cwd()).resolve()
    for candidate in [base, *base.parents]:
        if (candidate / ROOT_MARKER).exists():
            return candidate
    raise FileNotFoundError(f"Could not find {ROOT_MARKER} from {base}")


def ensure_repo_root(start: Path | None = None) -> Path:
    try:
        return find_repo_root(start)
    except FileNotFoundError:
        content_root = Path("/content")
        if not content_root.exists():
            raise
        repo = content_root / REPO_NAME
        if not repo.exists():
            subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(repo)], check=True)
        return find_repo_root(repo)


def requirements_csv(root: Path | None = None) -> Path:
    return find_repo_root(root) / ROOT_MARKER


def read_csv_rows(path: Path) -> list[CsvRow]:
    with path.open(newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
