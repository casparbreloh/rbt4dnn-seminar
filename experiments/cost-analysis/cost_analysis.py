from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from shared import CsvRow, find_repo_root, requirement_rows, write_csv, write_text


@dataclass(frozen=True)
class CostAssumption:
    parameter: str
    value: float
    unit: str


ASSUMPTIONS = (
    CostAssumption("gpu_hour_cost_usd", 3.00, "USD/hour"),
    CostAssumption("per_requirement_train_hours", 1.00, "hours"),
    CostAssumption("shared_train_hours", 1.50, "hours"),
    CostAssumption("generation_hours_per_1000", 0.10, "hours/1000 images"),
    CostAssumption("human_validation_cost_per_image_usd", 0.03, "USD/image"),
    CostAssumption("engineer_hours_per_requirement", 2.00, "hours"),
    CostAssumption("engineer_hourly_rate_usd", 25.00, "USD/hour"),
)

RESULT_FIELDS = [
    "dataset",
    "requirement",
    "method",
    "reported_n_images",
    "pass_rate",
    "precondition_match",
    "estimated_valid_failures",
    "estimated_cost_usd",
    "cost_per_estimated_valid_failure_usd",
]
VALID_FIELDS = [
    "dataset",
    "requirement",
    "method",
    "reported_n_images",
    "pass_rate",
    "precondition_match",
    "estimated_valid_failures",
    "estimated_invalid_failures",
    "estimated_valid_failure_rate",
]


def values() -> dict[str, float]:
    return {item.parameter: item.value for item in ASSUMPTIONS}


def estimated_cost(row: CsvRow) -> float:
    v = values()
    n_images = int(row["reported_n_images"])
    if row["method"] == "lr":
        train_hours = v["per_requirement_train_hours"]
        engineer_hours = v["engineer_hours_per_requirement"]
    else:
        train_hours = v["shared_train_hours"] / 6
        engineer_hours = v["engineer_hours_per_requirement"] / 2
    return (
        v["gpu_hour_cost_usd"] * (train_hours + v["generation_hours_per_1000"] * n_images / 1000)
        + v["human_validation_cost_per_image_usd"] * n_images
        + engineer_hours * v["engineer_hourly_rate_usd"]
    )


def build_rows(rows: list[CsvRow]) -> tuple[list[CsvRow], list[CsvRow]]:
    valid_rows: list[CsvRow] = []
    cost_rows: list[CsvRow] = []
    for row in rows:
        if (
            not row["reported_n_images"]
            or not row["pass_rate_mean"]
            or not row["precondition_match_mean"]
        ):
            continue

        n_images = int(row["reported_n_images"])
        pass_rate = float(row["pass_rate_mean"])
        precondition_match = float(row["precondition_match_mean"])
        failures = n_images * (1 - pass_rate)
        estimated_valid_failures = failures * precondition_match
        estimated_invalid_failures = failures * (1 - precondition_match)
        cost = estimated_cost(row)

        valid_rows.append(
            {
                "dataset": row["dataset"],
                "requirement": row["requirement"],
                "method": row["method"],
                "reported_n_images": str(n_images),
                "pass_rate": f"{pass_rate:.6f}",
                "precondition_match": f"{precondition_match:.6f}",
                "estimated_valid_failures": f"{estimated_valid_failures:.3f}",
                "estimated_invalid_failures": f"{estimated_invalid_failures:.3f}",
                "estimated_valid_failure_rate": f"{estimated_valid_failures / n_images:.6f}",
            }
        )
        cost_rows.append(
            {
                "dataset": row["dataset"],
                "requirement": row["requirement"],
                "method": row["method"],
                "reported_n_images": str(n_images),
                "pass_rate": f"{pass_rate:.6f}",
                "precondition_match": f"{precondition_match:.6f}",
                "estimated_valid_failures": f"{estimated_valid_failures:.3f}",
                "estimated_cost_usd": f"{cost:.2f}",
                "cost_per_estimated_valid_failure_usd": ""
                if estimated_valid_failures <= 0
                else f"{cost / estimated_valid_failures:.2f}",
            }
        )

    valid_rows.sort(key=lambda item: float(item["estimated_valid_failures"]), reverse=True)
    cost_rows.sort(key=lambda item: float(item["cost_per_estimated_valid_failure_usd"] or "inf"))
    return valid_rows, cost_rows


def write_results(root: Path | None = None) -> tuple[Path, Path, Path, Path]:
    root = find_repo_root(root)
    out_dir = root / "experiments" / "cost-analysis"
    valid_rows, cost_rows = build_rows(requirement_rows(root))

    assumptions = out_dir / "assumptions.csv"
    valid_failures = out_dir / "valid-failures.csv"
    results = out_dir / "results.csv"
    summary_path = out_dir / "summary.md"
    write_csv(
        assumptions,
        ["parameter", "value", "unit"],
        [
            {"parameter": item.parameter, "value": f"{item.value:.2f}", "unit": item.unit}
            for item in ASSUMPTIONS
        ],
    )
    write_csv(valid_failures, VALID_FIELDS, valid_rows)
    write_csv(results, RESULT_FIELDS, cost_rows)
    write_text(summary_path, summary(valid_rows, cost_rows))
    return assumptions, valid_failures, results, summary_path


def summary(valid_rows: list[CsvRow], cost_rows: list[CsvRow]) -> str:
    top_yield = valid_rows[:5]
    top_cost = cost_rows[:5]
    lines = [
        "# Cost Analysis Summary",
        "",
        "**Question:** If only valid failures matter, which requirement-testing targets look "
        "most cost-effective?",
        "",
        "**Method:** Estimate valid failures from pass and precondition rates, then divide "
        "illustrative compute, generation, validation, and engineering costs by that yield.",
        "",
        "**Result:** SGSM S2, MNIST M3, and SGSM S1 are the cheapest estimated valid-failure "
        "targets under the committed assumptions.",
        "",
        "**Limitation:** Dollar values are illustrative assumptions, not measured invoices. "
        "Estimated failure counts use the paper's reported generated-test counts, not the "
        "smaller copied image samples in this repo.",
        "",
        "## Largest Estimated Valid-Failure Yields",
        "",
    ]
    lines += [
        f"- {row['dataset']} {row['requirement']} {row['method']}: "
        f"{row['estimated_valid_failures']} estimated valid failures"
        for row in top_yield
    ]
    lines += ["", "## Cheapest Estimated Valid Failures", ""]
    lines += [
        f"- {row['dataset']} {row['requirement']} {row['method']}: "
        f"${row['cost_per_estimated_valid_failure_usd']} per estimated valid failure"
        for row in top_cost
    ]
    return "\n".join(lines) + "\n"
