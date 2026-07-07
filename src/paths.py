from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

ROOT_MARKER = Path("data/requirements.csv")

CsvRow = dict[str, str]


def find_repo_root(start: Path | None = None) -> Path:
    base = (start or Path.cwd()).resolve()
    for candidate in [base, *base.parents]:
        if (candidate / ROOT_MARKER).exists():
            return candidate
    raise FileNotFoundError(f"Could not find {ROOT_MARKER} from {base}")


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
