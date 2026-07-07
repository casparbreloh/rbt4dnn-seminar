from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from paths import CsvRow, find_repo_root, read_csv_rows, requirements_csv, write_csv


@dataclass(frozen=True)
class CostAssumption:
    parameter: str
    value: float
    unit: str


DEFAULT_ASSUMPTIONS = (
    CostAssumption("gpu_hour_cost_usd", 3.00, "USD/hour"),
    CostAssumption("per_requirement_train_hours", 1.00, "hours"),
    CostAssumption("shared_train_hours", 1.50, "hours"),
    CostAssumption("generation_hours_per_1000", 0.10, "hours/1000 images"),
    CostAssumption("human_validation_cost_per_image_usd", 0.03, "USD/image"),
    CostAssumption("engineer_hours_per_requirement", 2.00, "hours"),
    CostAssumption("engineer_hourly_rate_usd", 25.00, "USD/hour"),
)

VALID_FAILURE_FIELDS = [
    "dataset",
    "requirement",
    "variant",
    "n_images",
    "pass_rate",
    "precondition_match",
    "valid_failures",
    "nonmatching_failure_share",
]
COST_FIELDS = [
    "dataset",
    "requirement",
    "variant",
    "n_images",
    "pass_rate",
    "precondition_match",
    "valid_failures",
    "estimated_cost_usd",
    "cost_per_valid_failure_usd",
]
CostTableRow = dict[str, str]


def assumption_values(
    assumptions: tuple[CostAssumption, ...] = DEFAULT_ASSUMPTIONS,
) -> dict[str, float]:
    return {assumption.parameter: assumption.value for assumption in assumptions}


def optional_float(value: str) -> float | None:
    return float(value) if value else None


def estimated_cost(
    row: CsvRow, assumptions: tuple[CostAssumption, ...] = DEFAULT_ASSUMPTIONS
) -> float:
    values = assumption_values(assumptions)
    n_images = int(row["n_images"])
    if row["variant"] == "lr":
        train_hours = values["per_requirement_train_hours"]
        engineer_hours = values["engineer_hours_per_requirement"]
    else:
        train_hours = values["shared_train_hours"] / 6
        engineer_hours = values["engineer_hours_per_requirement"] / 2

    return (
        values["gpu_hour_cost_usd"]
        * (train_hours + values["generation_hours_per_1000"] * n_images / 1000)
        + values["human_validation_cost_per_image_usd"] * n_images
        + engineer_hours * values["engineer_hourly_rate_usd"]
    )


def build_cost_rows(
    rows: list[CsvRow],
    assumptions: tuple[CostAssumption, ...] = DEFAULT_ASSUMPTIONS,
) -> tuple[list[CostTableRow], list[CostTableRow]]:
    valid_rows: list[CostTableRow] = []
    cost_rows: list[CostTableRow] = []
    for row in rows:
        pass_rate = optional_float(row["pass_rate_mean"])
        precondition_match = optional_float(row["precondition_match_mean"])
        n_images = int(row["n_images"]) if row["n_images"] else None
        if pass_rate is None or precondition_match is None or n_images is None:
            continue

        valid_failures = n_images * precondition_match * (1 - pass_rate)
        valid_rows.append(
            {
                "dataset": row["dataset"],
                "requirement": row["requirement"],
                "variant": row["variant"],
                "n_images": str(n_images),
                "pass_rate": f"{pass_rate:.6f}",
                "precondition_match": f"{precondition_match:.6f}",
                "valid_failures": f"{valid_failures:.3f}",
                "nonmatching_failure_share": f"{(1 - precondition_match) * (1 - pass_rate):.6f}",
            }
        )

        cost = estimated_cost(row, assumptions)
        cost_rows.append(
            {
                "dataset": row["dataset"],
                "requirement": row["requirement"],
                "variant": row["variant"],
                "n_images": str(n_images),
                "pass_rate": f"{pass_rate:.6f}",
                "precondition_match": f"{precondition_match:.6f}",
                "valid_failures": f"{valid_failures:.3f}",
                "estimated_cost_usd": f"{cost:.2f}",
                "cost_per_valid_failure_usd": ""
                if valid_failures <= 0
                else f"{cost / valid_failures:.2f}",
            }
        )

    valid_rows.sort(key=lambda row: float(row["valid_failures"]), reverse=True)
    cost_rows.sort(key=cost_per_failure_sort_key)
    return valid_rows, cost_rows


def cost_per_failure_sort_key(row: CostTableRow) -> float:
    value = row["cost_per_valid_failure_usd"]
    return float(value) if value != "" else float("inf")


def write_cost_analysis(
    root: Path | None = None,
    assumptions: tuple[CostAssumption, ...] = DEFAULT_ASSUMPTIONS,
) -> Path:
    root = find_repo_root(root)
    out_dir = root / "experiments" / "cost-analysis"
    rows = read_csv_rows(requirements_csv(root))
    valid_rows, cost_rows = build_cost_rows(rows, assumptions)

    write_csv(
        out_dir / "assumptions.csv",
        ["parameter", "value", "unit"],
        [
            {"parameter": item.parameter, "value": f"{item.value:.2f}", "unit": item.unit}
            for item in assumptions
        ],
    )
    write_csv(out_dir / "valid-failures.csv", VALID_FAILURE_FIELDS, valid_rows)
    write_csv(out_dir / "results.csv", COST_FIELDS, cost_rows)
    return out_dir / "results.csv"
