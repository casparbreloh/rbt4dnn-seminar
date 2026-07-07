from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

import lance

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
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def open_lance(root: Path | None = None) -> lance.LanceDataset:
    root = find_repo_root(root)
    return lance.dataset(root / "data" / "rbt4dnn.lance")


def requirement_rows(root: Path | None = None) -> list[CsvRow]:
    ds = open_lance(root)
    columns = [
        "dataset",
        "requirement",
        "requirement_type",
        "requirement_text",
        "method",
        "method_label",
        "reported_n_images",
        "pass_rate_mean",
        "precondition_match_mean",
        "local_replication_pass_rate_n100",
        "paper_reference_pass_rate",
        "notes",
    ]
    table = ds.to_table(columns=columns)
    grouped: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in table.to_pylist():
        key = (row["dataset"], row["requirement"], row["method"])
        if key not in grouped:
            grouped[key] = {name: row.get(name) for name in columns}
            grouped[key]["available_images"] = 0
        grouped[key]["available_images"] = int(str(grouped[key]["available_images"])) + 1

    out: list[CsvRow] = []
    for row in grouped.values():
        out.append({key: "" if value is None else str(value) for key, value in row.items()})
    return sorted(out, key=lambda row: (row["dataset"], row["requirement"], row["method"]))
