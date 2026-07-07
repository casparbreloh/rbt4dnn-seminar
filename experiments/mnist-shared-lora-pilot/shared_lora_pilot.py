from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from mnist import evaluate_mnist_folders
from shared import CsvRow, find_repo_root, requirement_rows, write_csv, write_text

REQUIREMENTS = ["M1", "M2", "M3", "M4", "M5", "M6"]
RESULT_FIELDS = [
    "requirement",
    "method",
    "n_images",
    "pass_rate",
    "paper_lr_pass_rate",
    "delta_vs_paper_lr",
    "failure_count",
]


def prepare_training_data(root: Path | None = None, max_per_requirement: int = 100) -> Path:
    root = find_repo_root(root)
    out_dir = root / "outputs" / "mnist-shared-lora-pilot" / "train"
    image_dir = out_dir / "images"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    image_dir.mkdir(parents=True)

    metadata: list[dict[str, str]] = []
    row_by_requirement = {
        row["requirement"]: row
        for row in requirement_rows(root)
        if row["dataset"] == "mnist" and row["method"] == "lr"
    }
    for requirement in REQUIREMENTS:
        source = root / "data" / "images" / "mnist" / requirement
        prompt = row_by_requirement[requirement]["requirement_text"]
        for index, path in enumerate(sorted(source.glob("*.png"))[:max_per_requirement]):
            name = f"{requirement}_{index:04d}.png"
            shutil.copy2(path, image_dir / name)
            metadata.append(
                {"file_name": f"images/{name}", "text": prompt, "requirement": requirement}
            )

    with (out_dir / "metadata.jsonl").open("w") as file:
        for row in metadata:
            file.write(json.dumps(row) + "\n")
    with (out_dir / "metadata.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["file_name", "text", "requirement"])
        writer.writeheader()
        writer.writerows(metadata)
    return out_dir


def generated_folders(root: Path | None = None) -> dict[str, Path]:
    root = find_repo_root(root)
    base = root / "outputs" / "mnist-shared-lora-pilot" / "generated"
    return {requirement: base / requirement for requirement in REQUIREMENTS}


def evaluate_generated(root: Path | None = None, batch_size: int = 100) -> list[CsvRow]:
    root = find_repo_root(root)
    rows = []
    paper_pass = {
        row["requirement"]: float(row["paper_reference_pass_rate"])
        for row in requirement_rows(root)
        if row["dataset"] == "mnist" and row["method"] == "lr" and row["paper_reference_pass_rate"]
    }
    for row in evaluate_mnist_folders(generated_folders(root), batch_size=batch_size):
        requirement = row["requirement"]
        paper = paper_pass[requirement]
        rows.append(
            {
                "requirement": requirement,
                "method": "shared-lora-pilot",
                "n_images": str(row["n_images"]),
                "pass_rate": f"{row['pass_rate']:.6f}",
                "paper_lr_pass_rate": f"{paper:.6f}",
                "delta_vs_paper_lr": f"{row['pass_rate'] - paper:+.6f}",
                "failure_count": str(len(row["failures"])),
            }
        )
    return rows


def write_results(root: Path | None = None) -> tuple[Path, Path]:
    root = find_repo_root(root)
    out_dir = root / "experiments" / "mnist-shared-lora-pilot"
    folders = generated_folders(root)
    rows = (
        []
        if any(not list(folder.glob("*.png")) for folder in folders.values())
        else evaluate_generated(root)
    )
    write_csv(out_dir / "results.csv", RESULT_FIELDS, rows)
    write_text(out_dir / "summary.md", summary(rows))
    return out_dir / "results.csv", out_dir / "summary.md"


def write_template(root: Path | None = None) -> Path:
    root = find_repo_root(root)
    path = root / "experiments" / "mnist-shared-lora-pilot" / "results-template.csv"
    write_csv(path, RESULT_FIELDS, [])
    return path


def summary(rows: list[CsvRow]) -> str:
    lines = [
        "# MNIST Shared LoRA Pilot Summary",
        "",
        "This file is filled after generated images exist under `outputs/mnist-shared-lora-pilot/generated/M*`.",
        "",
    ]
    if not rows:
        lines.append("No generated shared-LoRA images were found yet.")
    else:
        for row in rows:
            lines.append(
                f"- {row['requirement']}: pass {row['pass_rate']} "
                f"(delta vs paper LR {row['delta_vs_paper_lr']}, failures {row['failure_count']})"
            )
    return "\n".join(lines) + "\n"
