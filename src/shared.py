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
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def requirement_rows(root: Path | None = None) -> list[CsvRow]:
    root = find_repo_root(root)
    rows = []
    for row in read_csv_rows(root / ROOT_MARKER):
        method = row["variant"]
        rows.append(
            {
                "dataset": image_dataset_name(row["dataset"]),
                "requirement": row["requirement"],
                "requirement_type": row["requirement_type"],
                "requirement_text": row["requirement_text"],
                "method": method,
                "method_label": row["variant_label"],
                "reported_n_images": row["n_images"],
                "pass_rate_mean": row["pass_rate_mean"],
                "precondition_match_mean": row["precondition_match_mean"],
                "local_replication_pass_rate_n100": row["local_replication_pass_rate_n100"],
                "paper_reference_pass_rate": row["paper_reference_pass_rate"],
                "notes": row["notes"],
                "available_images": str(
                    count_available_images(root, row["dataset"], row["requirement"], method)
                ),
            }
        )
    return rows


def image_dataset_name(dataset: str) -> str:
    return "celeba-hq" if dataset == "celeba" else dataset


def image_folder(dataset: str, requirement: str, method: str) -> str:
    if dataset == "mnist" and method == "allreq":
        return f"Allreq_{requirement}"
    if dataset == "mnist" and method == "alldata":
        return f"Alldata_{requirement}"
    return requirement


def count_available_images(root: Path, dataset: str, requirement: str, method: str) -> int:
    folder = (
        root
        / "data"
        / "images"
        / image_dataset_name(dataset)
        / image_folder(dataset, requirement, method)
    )
    return len(list(folder.glob("*.png")))
