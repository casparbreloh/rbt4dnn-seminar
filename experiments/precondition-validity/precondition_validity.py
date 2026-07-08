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
    "estimated_valid_failure_rate",
    "estimated_valid_failures",
    "estimated_invalid_failures",
    "validity_gap",
    "notes",
]


def has_metrics(row: CsvRow) -> bool:
    return bool(
        row["reported_n_images"] and row["pass_rate_mean"] and row["precondition_match_mean"]
    )


def skipped_rows(rows: list[CsvRow]) -> list[CsvRow]:
    return [row for row in rows if not has_metrics(row)]


def build_rows(rows: list[CsvRow]) -> list[CsvRow]:
    out: list[CsvRow] = []
    for row in rows:
        if not has_metrics(row):
            continue

        n_images = int(row["reported_n_images"])
        pass_rate = float(row["pass_rate_mean"])
        precondition_match = float(row["precondition_match_mean"])
        raw_failure_rate = 1 - pass_rate
        estimated_valid_failure_rate = precondition_match * raw_failure_rate
        estimated_invalid_failures = n_images * (1 - precondition_match) * raw_failure_rate

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
                "estimated_valid_failure_rate": f"{estimated_valid_failure_rate:.6f}",
                "estimated_valid_failures": f"{n_images * estimated_valid_failure_rate:.3f}",
                "estimated_invalid_failures": f"{estimated_invalid_failures:.3f}",
                "validity_gap": f"{1 - precondition_match:.6f}",
                "notes": row["notes"],
            }
        )

    out.sort(key=lambda item: float(item["estimated_valid_failure_rate"]), reverse=True)
    return out


def write_results(root: Path | None = None) -> tuple[Path, Path]:
    root = find_repo_root(root)
    out_dir = root / "experiments" / "precondition-validity"
    rows = build_rows(requirement_rows(root))
    write_csv(out_dir / "results.csv", FIELDS, rows)
    write_text(out_dir / "summary.md", summary(root, rows))
    return out_dir / "results.csv", out_dir / "summary.md"


def summary(root: Path, rows: list[CsvRow]) -> str:
    skipped = skipped_rows(requirement_rows(root))
    main_rows = [row for row in rows if row["method"] == "lr"]
    low_precondition = sorted(rows, key=lambda item: float(item["precondition_match"]))[:5]
    highest_valid_failure = sorted(
        main_rows,
        key=lambda item: float(item["estimated_valid_failure_rate"]),
        reverse=True,
    )[:5]
    misleading_pass = sorted(
        [row for row in rows if float(row["pass_rate"]) >= 0.95],
        key=lambda item: float(item["precondition_match"]),
    )[:5]

    lines = [
        "# Precondition Validity Summary",
        "",
        "**Question:** Which generated-test failures are most likely to be valid failures, not "
        "artifacts from images that miss the natural-language precondition?",
        "",
        "**Method:** Combine each paper pass rate with its precondition-match rate as "
        "`precondition_match * (1 - pass_rate)`.",
        "",
        "**Result:** The strongest main LoRA targets are SGSM S2, MNIST M3, and SGSM S1 by "
        "estimated valid-failure rate.",
        "",
        "**Limitation:** These are aggregate estimates, not observed joint labels for "
        "individual generated images. Estimated counts use the paper's reported generated-test "
        "counts; `available_images` is the smaller copied image sample in this repo.",
        "",
        f"Requirement/method rows with pass and precondition metrics: {len(rows)}",
    ]
    if skipped:
        lines += [
            "",
            "Skipped rows with missing metrics: "
            + ", ".join(
                f"{row['dataset']} {row['requirement']} {row['method']}" for row in skipped
            ),
        ]
    lines += [
        "",
        "## Highest Estimated Valid-Failure Rates (Main LoRA Rows)",
        "",
    ]
    lines += [
        f"- {row['dataset']} {row['requirement']} {row['method']}: "
        f"estimated valid-failure rate {row['estimated_valid_failure_rate']} "
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
