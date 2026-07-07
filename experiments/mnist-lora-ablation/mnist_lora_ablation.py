from __future__ import annotations

from pathlib import Path

from shared import CsvRow, find_repo_root, requirement_rows, write_csv, write_text

FIELDS = [
    "requirement",
    "method",
    "reported_n_images",
    "available_images",
    "pass_rate",
    "precondition_match",
    "valid_failures",
    "valid_failures_per_1000",
    "relative_to_per_requirement_lora",
]


def metric(row: CsvRow) -> float:
    return float(row["precondition_match_mean"]) * (1 - float(row["pass_rate_mean"]))


def build_rows(rows: list[CsvRow]) -> list[CsvRow]:
    mnist_rows = [
        row
        for row in rows
        if row["dataset"] == "mnist"
        and row["requirement"] != "M7"
        and row["method"] in {"lr", "allreq", "alldata"}
        and row["reported_n_images"]
        and row["pass_rate_mean"]
        and row["precondition_match_mean"]
    ]
    baseline = {row["requirement"]: metric(row) for row in mnist_rows if row["method"] == "lr"}

    out: list[CsvRow] = []
    for row in sorted(mnist_rows, key=lambda item: (item["requirement"], item["method"])):
        valid_failure_rate = metric(row)
        n_images = int(row["reported_n_images"])
        base = baseline[row["requirement"]]
        out.append(
            {
                "requirement": row["requirement"],
                "method": row["method"],
                "reported_n_images": str(n_images),
                "available_images": row["available_images"],
                "pass_rate": f"{float(row['pass_rate_mean']):.6f}",
                "precondition_match": f"{float(row['precondition_match_mean']):.6f}",
                "valid_failures": f"{n_images * valid_failure_rate:.3f}",
                "valid_failures_per_1000": f"{1000 * valid_failure_rate:.3f}",
                "relative_to_per_requirement_lora": f"{valid_failure_rate / base:.3f}",
            }
        )
    return out


def write_results(root: Path | None = None) -> tuple[Path, Path]:
    root = find_repo_root(root)
    out_dir = root / "experiments" / "mnist-lora-ablation"
    rows = build_rows(requirement_rows(root))
    write_csv(out_dir / "results.csv", FIELDS, rows)
    write_text(out_dir / "summary.md", summary(rows))
    return out_dir / "results.csv", out_dir / "summary.md"


def summary(rows: list[CsvRow]) -> str:
    by_requirement: dict[str, list[CsvRow]] = {}
    for row in rows:
        by_requirement.setdefault(row["requirement"], []).append(row)

    lines = [
        "# MNIST LoRA Ablation Summary",
        "",
        "Compares per-requirement LoRA (`lr`) with the paper's shared MNIST `allreq` and `alldata` ablations.",
        "",
    ]
    for requirement, req_rows in sorted(by_requirement.items()):
        best = max(req_rows, key=lambda item: float(item["valid_failures_per_1000"]))
        lr = next(row for row in req_rows if row["method"] == "lr")
        lines.append(
            f"- {requirement}: best valid-failure efficiency is `{best['method']}` "
            f"({best['valid_failures_per_1000']} per 1000). "
            f"`allreq/alldata` ratios are measured against `lr={lr['valid_failures_per_1000']}`."
        )
    return "\n".join(lines) + "\n"
