from __future__ import annotations

import argparse
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

import lance
import pyarrow as pa
from lance import Blob, blob_array, blob_field


ROOT = Path(__file__).resolve().parents[2]
IMAGE_ROOT = ROOT / "data" / "images"
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


def collect_rows() -> dict[str, list]:
    rows = {
        "row_id": [],
        "dataset": [],
        "requirement": [],
        "method": [],
        "source_folder": [],
        "image_id": [],
        "image_path": [],
        "byte_size": [],
        "image": [],
    }

    for path in sorted(IMAGE_ROOT.glob("*/*/*.png")):
        dataset = path.relative_to(IMAGE_ROOT).parts[0]
        source_folder = path.parent.name
        requirement, method = parse_folder(dataset, source_folder)
        rel_path = path.relative_to(ROOT).as_posix()
        image_id = int(path.stem) if path.stem.isdigit() else None

        rows["row_id"].append(f"{dataset}/{source_folder}/{path.name}")
        rows["dataset"].append(dataset)
        rows["requirement"].append(requirement)
        rows["method"].append(method)
        rows["source_folder"].append(source_folder)
        rows["image_id"].append(image_id)
        rows["image_path"].append(rel_path)
        rows["byte_size"].append(path.stat().st_size)
        rows["image"].append(Blob.from_uri(rel_path))

    return rows


def build_table(rows: dict[str, list]) -> pa.Table:
    schema = pa.schema(
        [
            pa.field("row_id", pa.string()),
            pa.field("dataset", pa.string()),
            pa.field("requirement", pa.string()),
            pa.field("method", pa.string()),
            pa.field("source_folder", pa.string()),
            pa.field("image_id", pa.int64()),
            pa.field("image_path", pa.string()),
            pa.field("byte_size", pa.int64()),
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
