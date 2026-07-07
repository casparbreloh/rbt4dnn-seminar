from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from shared import CsvRow, find_repo_root, read_csv_rows, requirements_csv, write_csv

TARGET = {
    "M1": 2,
    "M2": 3,
    "M3": 7,
    "M4": 9,
    "M5": 6,
    "M6": 0,
    "M7": 8,
}
PAPER = {
    "M1": 0.999,
    "M2": 0.977,
    "M3": 0.724,
    "M4": 0.982,
    "M5": 0.994,
    "M6": 0.976,
    "M7": 0.981,
}


class EvaluationRow(TypedDict):
    requirement: str
    n_images: int
    pass_rate: float
    paper_pass_rate: float | None
    failures: list[tuple[str, int]]


def summarize_replication(rows: list[CsvRow]) -> list[CsvRow]:
    out_rows: list[CsvRow] = []
    for row in rows:
        if (
            row["dataset"] == "mnist"
            and row["variant"] == "lr"
            and row["local_replication_pass_rate_n100"]
        ):
            local = float(row["local_replication_pass_rate_n100"])
            paper = float(row["paper_reference_pass_rate"])
            out_rows.append(
                {
                    "requirement": row["requirement"],
                    "local_pass_rate_n100": f"{local:.3f}",
                    "paper_pass_rate": f"{paper:.3f}",
                    "delta": f"{local - paper:+.3f}",
                }
            )
    return out_rows


def write_replication_summary(root: Path | None = None) -> Path:
    root = find_repo_root(root)
    out = root / "experiments" / "replication" / "results.csv"
    write_csv(
        out,
        ["requirement", "local_pass_rate_n100", "paper_pass_rate", "delta"],
        summarize_replication(read_csv_rows(requirements_csv(root))),
    )
    return out


def evaluate_mnist_images(
    root: Path | None = None,
    batch_size: int = 25,
    variants: list[str] | None = None,
) -> list[EvaluationRow]:
    root = find_repo_root(root)
    base = root / "data" / "images" / "mnist"
    selected_variants = variants or ["", "Allreq_", "Alldata_"]
    folders = {
        f"{variant}{requirement}": base / f"{variant}{requirement}"
        for variant in selected_variants
        for requirement in TARGET
    }
    return evaluate_mnist_folders(folders, batch_size=batch_size)


def evaluate_mnist_folders(
    folders: dict[str, Path],
    batch_size: int = 25,
) -> list[EvaluationRow]:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained("farleyknight-org-username/vit-base-mnist")
    model = (
        AutoModelForImageClassification.from_pretrained("farleyknight-org-username/vit-base-mnist")
        .to(device)
        .eval()
    )

    rows: list[EvaluationRow] = []
    for label, folder in folders.items():
        requirement = label.removeprefix("Allreq_").removeprefix("Alldata_")
        if requirement not in TARGET:
            continue
        target = TARGET[requirement]
        files = sorted(folder.glob("*.png"))
        if not files:
            continue

        predictions: list[int] = []
        for start in range(0, len(files), batch_size):
            images = [Image.open(path).convert("RGB") for path in files[start : start + batch_size]]
            with torch.no_grad():
                inputs = processor(images, return_tensors="pt").to(device)
                predictions.extend(model(**inputs).logits.argmax(-1).tolist())

        failures = [
            (files[index].name, prediction)
            for index, prediction in enumerate(predictions)
            if prediction != target
        ]
        rows.append(
            {
                "requirement": label,
                "n_images": len(files),
                "pass_rate": (len(files) - len(failures)) / len(files),
                "paper_pass_rate": PAPER[requirement] if label == requirement else None,
                "failures": failures,
            }
        )
    return rows
