from __future__ import annotations

from pathlib import Path

from shared import CsvRow, find_repo_root, requirement_rows, write_csv, write_text

FIELDS = [
    "dataset",
    "requirement",
    "method",
    "available_images",
    "reported_n_images",
    "pass_rate",
    "precondition_match",
    "raw_failure_rate",
    "valid_failure_rate",
    "valid_failures",
    "invalid_failures",
    "validity_gap",
    "notes",
]


def build_rows(rows: list[CsvRow]) -> list[CsvRow]:
    out: list[CsvRow] = []
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
        raw_failure_rate = 1 - pass_rate
        valid_failure_rate = precondition_match * raw_failure_rate
        invalid_failures = n_images * (1 - precondition_match) * raw_failure_rate

        out.append(
            {
                "dataset": row["dataset"],
                "requirement": row["requirement"],
                "method": row["method"],
                "available_images": row["available_images"],
                "reported_n_images": str(n_images),
                "pass_rate": f"{pass_rate:.6f}",
                "precondition_match": f"{precondition_match:.6f}",
                "raw_failure_rate": f"{raw_failure_rate:.6f}",
                "valid_failure_rate": f"{valid_failure_rate:.6f}",
                "valid_failures": f"{n_images * valid_failure_rate:.3f}",
                "invalid_failures": f"{invalid_failures:.3f}",
                "validity_gap": f"{1 - precondition_match:.6f}",
                "notes": row["notes"],
            }
        )

    out.sort(key=lambda item: float(item["valid_failure_rate"]), reverse=True)
    return out


def write_results(root: Path | None = None) -> tuple[Path, Path]:
    root = find_repo_root(root)
    out_dir = root / "experiments" / "precondition-validity"
    rows = build_rows(requirement_rows(root))
    write_csv(out_dir / "results.csv", FIELDS, rows)
    write_text(out_dir / "summary.md", summary(root, rows))
    return out_dir / "results.csv", out_dir / "summary.md"


def summary(root: Path, rows: list[CsvRow]) -> str:
    low_precondition = sorted(rows, key=lambda item: float(item["precondition_match"]))[:5]
    highest_valid_failure = rows[:5]
    misleading_pass = sorted(
        [row for row in rows if float(row["pass_rate"]) >= 0.95],
        key=lambda item: float(item["precondition_match"]),
    )[:5]

    lines = [
        "# Precondition Validity Summary",
        "",
        f"Requirement/method rows with pass and precondition metrics: {len(rows)}",
        "",
        "## Highest Valid-Failure Rates",
        "",
    ]
    lines += [
        f"- {row['dataset']} {row['requirement']} {row['method']}: "
        f"valid failure rate {row['valid_failure_rate']} "
        f"(pass {row['pass_rate']}, precondition {row['precondition_match']})"
        for row in highest_valid_failure
    ]
    lines += ["", "## Lowest Precondition Match", ""]
    lines += [
        f"- {row['dataset']} {row['requirement']} {row['method']}: "
        f"precondition {row['precondition_match']}"
        for row in low_precondition
    ]
    lines += ["", "## High Pass Rate But Weak Precondition Match", ""]
    lines += [
        f"- {row['dataset']} {row['requirement']} {row['method']}: "
        f"pass {row['pass_rate']}, precondition {row['precondition_match']}"
        for row in misleading_pass
    ]
    return "\n".join(lines) + "\n"
