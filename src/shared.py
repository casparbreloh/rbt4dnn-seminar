from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

ROOT_MARKER = Path("data/requirements.csv")

CsvRow = dict[str, str]


class DataContractError(RuntimeError):
    """The restored upstream image corpus does not match its frozen manifest."""


def load_data_manifest(root: Path | None = None) -> dict[str, Any]:
    root = find_repo_root(root)
    path = root / "data" / "manifest.json"
    try:
        manifest = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise DataContractError(f"Missing data manifest: {path}") from error
    sidecar = root / "data" / manifest["checksum_sidecar"]
    try:
        lines = sidecar.read_text().splitlines()
    except FileNotFoundError as error:
        raise DataContractError(f"Missing checksum sidecar: {sidecar}") from error
    files = []
    for line_number, line in enumerate(lines, start=1):
        parts = line.split("\t")
        if len(parts) != 5:
            raise DataContractError(f"Malformed checksum sidecar line {line_number}")
        sha256, size, dataset, local_path, source_path = parts
        files.append(
            {
                "sha256": sha256,
                "size": int(size),
                "dataset": dataset,
                "local_path": local_path,
                "source_path": source_path,
            }
        )
    manifest["files"] = files
    if len(files) != manifest["file_count"]:
        raise DataContractError(
            f"Manifest expects {manifest['file_count']} files but sidecar has {len(files)}"
        )
    return manifest


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_entries(
    root: Path | None = None, datasets: Iterable[str] | None = None
) -> list[dict[str, Any]]:
    manifest = load_data_manifest(root)
    selected = set(datasets or manifest["datasets"])
    unknown = selected.difference(manifest["datasets"])
    if unknown:
        raise DataContractError(f"Unknown dataset selection: {', '.join(sorted(unknown))}")
    return [entry for entry in manifest["files"] if entry["dataset"] in selected]


def validate_image_corpus(
    root: Path | None = None,
    datasets: Iterable[str] | None = None,
    *,
    verify_hashes: bool = True,
) -> None:
    """Fail clearly if selected corpus files are missing, altered, or unsafe."""
    root = find_repo_root(root)
    missing: list[str] = []
    invalid: list[str] = []
    entries = data_entries(root, datasets)
    for entry in entries:
        relative = Path(entry["local_path"])
        path = root / relative
        if path.is_symlink():
            invalid.append(f"{relative} (symlink)")
        elif not path.is_file():
            missing.append(str(relative))
        elif path.stat().st_size != entry["size"]:
            invalid.append(f"{relative} (size)")
        elif verify_hashes and file_sha256(path) != entry["sha256"]:
            invalid.append(f"{relative} (sha256)")
    if missing or invalid:
        details = []
        if missing:
            details.append(f"missing {len(missing)} (first: {missing[0]})")
        if invalid:
            details.append(f"invalid {len(invalid)} (first: {invalid[0]})")
        selection = ", ".join(sorted(set(entry["dataset"] for entry in entries)))
        raise DataContractError(
            f"Incomplete RBT4DNN image corpus for {selection}: {'; '.join(details)}. "
            "Restore it with `uv run python scripts/fetch_data.py`."
        )


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
    # This field describes the pinned review corpus, not whatever ignored files happen to
    # exist in a particular checkout.  Deriving it from the manifest keeps metadata-only
    # reproduction deterministic in a fresh clone while still exposing the smaller sample
    # size used by this artifact.
    manifest_counts = Counter(
        str(Path(entry["local_path"]).parent) for entry in load_data_manifest(root)["files"]
    )
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
                    manifest_counts[
                        str(
                            Path("data")
                            / "images"
                            / image_dataset_name(row["dataset"])
                            / image_folder(row["dataset"], row["requirement"], method)
                        )
                    ]
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
