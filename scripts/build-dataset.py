from __future__ import annotations

import argparse
import csv
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

import lance
import pyarrow as pa
from lance import Blob, blob_array, blob_field


ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = ROOT / "data" / "images"
REQUIREMENTS = ROOT / "data" / "requirements.csv"
DATASET = ROOT / "data" / "rbt4dnn.lance"


@contextmanager
def repo_cwd():
    cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        yield
    finally:
        os.chdir(cwd)


def parse_folder(dataset: str, folder: str) -> tuple[str, str]:
    if dataset == "mnist" and folder.startswith("Allreq_"):
        return folder.removeprefix("Allreq_"), "allreq"
    if dataset == "mnist" and folder.startswith("Alldata_"):
        return folder.removeprefix("Alldata_"), "alldata"
    return folder, "lr"


def input_key(dataset: str, requirement: str, method: str) -> tuple[str, str, str]:
    dataset = "celeba" if dataset == "celeba-hq" else dataset
    return dataset, requirement, method


def load_input_rows() -> dict[tuple[str, str, str], dict[str, str]]:
    with REQUIREMENTS.open(newline="") as f:
        rows = csv.DictReader(f)
        return {(row["dataset"], row["requirement"], row["variant"]): row for row in rows}


def optional_float(value: str) -> float | None:
    return float(value) if value else None


def optional_int(value: str) -> int | None:
    return int(value) if value else None


def collect_rows() -> dict[str, list]:
    input_rows = load_input_rows()
    rows = {
        "row_id": [],
        "dataset": [],
        "requirement": [],
        "requirement_type": [],
        "requirement_text": [],
        "method": [],
        "method_label": [],
        "source_folder": [],
        "image_id": [],
        "image_path": [],
        "byte_size": [],
        "reported_n_images": [],
        "pass_rate_mean": [],
        "precondition_match_mean": [],
        "local_replication_pass_rate_n100": [],
        "paper_reference_pass_rate": [],
        "notes": [],
        "image": [],
    }

    for path in sorted(IMAGE_ROOT.glob("*/*/*.png")):
        dataset = path.relative_to(IMAGE_ROOT).parts[0]
        source_folder = path.parent.name
        requirement, method = parse_folder(dataset, source_folder)
        input_row = input_rows.get(input_key(dataset, requirement, method), {})
        rel_path = path.relative_to(ROOT).as_posix()
        image_id = int(path.stem) if path.stem.isdigit() else None

        rows["row_id"].append(f"{dataset}/{source_folder}/{path.name}")
        rows["dataset"].append(dataset)
        rows["requirement"].append(requirement)
        rows["requirement_type"].append(input_row.get("requirement_type"))
        rows["requirement_text"].append(input_row.get("requirement_text"))
        rows["method"].append(method)
        rows["method_label"].append(input_row.get("variant_label"))
        rows["source_folder"].append(source_folder)
        rows["image_id"].append(image_id)
        rows["image_path"].append(rel_path)
        rows["byte_size"].append(path.stat().st_size)
        rows["reported_n_images"].append(optional_int(input_row.get("n_images", "")))
        rows["pass_rate_mean"].append(optional_float(input_row.get("pass_rate_mean", "")))
        rows["precondition_match_mean"].append(
            optional_float(input_row.get("precondition_match_mean", ""))
        )
        rows["local_replication_pass_rate_n100"].append(
            optional_float(input_row.get("local_replication_pass_rate_n100", ""))
        )
        rows["paper_reference_pass_rate"].append(
            optional_float(input_row.get("paper_reference_pass_rate", ""))
        )
        rows["notes"].append(input_row.get("notes"))
        rows["image"].append(Blob.from_uri(rel_path))

    return rows


def build_table(rows: dict[str, list]) -> pa.Table:
    schema = pa.schema(
        [
            pa.field("row_id", pa.string()),
            pa.field("dataset", pa.string()),
            pa.field("requirement", pa.string()),
            pa.field("requirement_type", pa.string()),
            pa.field("requirement_text", pa.string()),
            pa.field("method", pa.string()),
            pa.field("method_label", pa.string()),
            pa.field("source_folder", pa.string()),
            pa.field("image_id", pa.int64()),
            pa.field("image_path", pa.string()),
            pa.field("byte_size", pa.int64()),
            pa.field("reported_n_images", pa.int64()),
            pa.field("pass_rate_mean", pa.float64()),
            pa.field("precondition_match_mean", pa.float64()),
            pa.field("local_replication_pass_rate_n100", pa.float64()),
            pa.field("paper_reference_pass_rate", pa.float64()),
            pa.field("notes", pa.string()),
            blob_field("image"),
        ]
    )
    return pa.table(
        {
            **{name: rows[name] for name in rows if name != "image"},
            "image": blob_array(rows["image"]),
        },
        schema=schema,
    )


def write_dataset(table: pa.Table) -> lance.LanceDataset:
    if DATASET.exists():
        shutil.rmtree(DATASET)
    with repo_cwd():
        return lance.write_dataset(
            table,
            DATASET,
            mode="create",
            data_storage_version="2.2",
            external_blob_mode="reference",
            initial_bases=[
                lance.DatasetBasePath("data/images", name="images", is_dataset_root=False)
            ],
        )


def validate(ds: lance.LanceDataset) -> None:
    count = ds.count_rows()
    if count != 14_500:
        raise RuntimeError(f"expected 14500 rows, got {count}")
    with repo_cwd():
        sample = ds.read_blobs("image", indices=[0])[0][1]
    if not sample:
        raise RuntimeError("sample blob read returned no bytes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    rows = collect_rows()
    table = build_table(rows)
    ds = write_dataset(table)
    if not args.no_validate:
        validate(ds)
    print(f"wrote {ds.count_rows()} rows to {DATASET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
