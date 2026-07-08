from __future__ import annotations

import base64
import json
import os
import random
import time
from urllib.error import HTTPError
import urllib.request
from pathlib import Path
from statistics import mean

from shared import CsvRow, find_repo_root, image_folder, requirement_rows, write_csv, write_text

DATASETS = ["mnist", "celeba-hq", "sgsm"]
DEFAULT_MODEL = "google/gemini-3-flash-preview"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_REFERER = "https://github.com/casparbreloh/rbt4dnn-seminar"
MANIFEST_FIELDS = [
    "sample_id",
    "dataset",
    "requirement",
    "requirement_text",
    "image_path",
    "paper_pass_rate",
    "paper_precondition_match",
]
RESULT_FIELDS = [
    *MANIFEST_FIELDS,
    "model",
    "valid",
    "confidence",
    "reason",
    "visible_evidence",
    "raw_response",
]


def output_dir(root: Path) -> Path:
    return root / "experiments" / "llm-validity-audit"


def image_dir(root: Path, row: CsvRow) -> Path:
    return (
        root
        / "data"
        / "images"
        / row["dataset"]
        / image_folder(row["dataset"], row["requirement"], row["method"])
    )


def select_samples(
    root: Path,
    samples_per_requirement: int = 2,
    seed: int = 7,
    datasets: list[str] | None = None,
) -> list[CsvRow]:
    root = find_repo_root(root)
    selected_datasets = set(datasets or DATASETS)
    rows: list[CsvRow] = []
    for row in requirement_rows(root):
        if row["dataset"] not in selected_datasets or row["method"] != "lr":
            continue
        files = sorted(image_dir(root, row).glob("*.png"))
        if not files:
            continue
        rng = random.Random(f"{seed}:{row['dataset']}:{row['requirement']}")
        picks = rng.sample(files, min(samples_per_requirement, len(files)))
        for index, path in enumerate(sorted(picks)):
            rows.append(
                {
                    "sample_id": f"{row['dataset']}-{row['requirement']}-{index}",
                    "dataset": row["dataset"],
                    "requirement": row["requirement"],
                    "requirement_text": row["requirement_text"],
                    "image_path": str(path.relative_to(root)),
                    "paper_pass_rate": row["pass_rate_mean"],
                    "paper_precondition_match": row["precondition_match_mean"],
                }
            )
    return rows


def prompt(row: CsvRow) -> str:
    return (
        "You are auditing one generated test image for a neural-network testing paper.\n"
        "Judge only what is visible in the image. Do not assume hidden labels.\n\n"
        f"Dataset: {row['dataset']}\n"
        f"Requirement: {row['requirement_text']}\n\n"
        "Does the image clearly satisfy the requirement?\n"
        "Return only JSON with exactly these keys:\n"
        '{"valid":"yes|no|unclear","confidence":0.0,'
        '"reason":"short reason","visible_evidence":"short evidence"}'
    )


def read_api_key() -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY to run the LLM validity audit.")
    return api_key


def openrouter_request(image_path: Path, row: CsvRow, model: str, api_key: str) -> str:
    image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt(row)},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"},
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 250,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        OPENROUTER_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": OPENROUTER_REFERER,
            "X-Title": "rbt4dnn-seminar",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def request_with_retry(image_path: Path, row: CsvRow, model: str, api_key: str) -> str:
    wait = 30
    while True:
        try:
            return openrouter_request(image_path, row, model=model, api_key=api_key)
        except HTTPError as error:
            if error.code != 429:
                raise
            message = http_error_message(error)
            if "credit" in message.lower() and "deplet" in message.lower():
                raise RuntimeError(
                    "OpenRouter credits are depleted. Add credits or use another API key, then rerun."
                ) from error
            print(f"rate limited on {row['sample_id']}; waiting {wait}s", flush=True)
            time.sleep(wait)
            wait = min(wait * 2, 3600)


def http_error_message(error: HTTPError) -> str:
    body = error.read().decode("utf-8", errors="replace")
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return body
    return str(data.get("error", {}).get("message", body))


def parse_response(raw: str) -> CsvRow:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    data = json.loads(text)
    valid = str(data.get("valid", "unclear")).lower()
    if valid not in {"yes", "no", "unclear"}:
        valid = "unclear"
    return {
        "valid": valid,
        "confidence": str(data.get("confidence", "")),
        "reason": str(data.get("reason", "")),
        "visible_evidence": str(data.get("visible_evidence", "")),
    }


def write_manifest(
    root: Path | None = None,
    samples_per_requirement: int = 2,
    seed: int = 7,
) -> list[Path]:
    root = find_repo_root(root)
    rows = select_samples(root, samples_per_requirement=samples_per_requirement, seed=seed)
    manifest = output_dir(root) / "sample-manifest.csv"
    write_csv(manifest, MANIFEST_FIELDS, rows)
    return [manifest]


def run_audit(
    root: Path | None = None,
    samples_per_requirement: int = 2,
    seed: int = 7,
    model: str = DEFAULT_MODEL,
) -> list[Path]:
    root = find_repo_root(root)
    rows = select_samples(root, samples_per_requirement=samples_per_requirement, seed=seed)
    out_dir = output_dir(root)
    manifest = out_dir / "sample-manifest.csv"
    results = out_dir / "results.csv"
    summary_path = out_dir / "summary.md"
    existing = {row["sample_id"]: row for row in read_existing_results(results)}
    api_key: str | None = None
    result_rows: list[CsvRow] = []
    for row in rows:
        if row["sample_id"] in existing:
            result_rows.append(existing[row["sample_id"]])
            print(row["sample_id"], "cached", flush=True)
            continue
        if api_key is None:
            api_key = read_api_key()
        raw = request_with_retry(root / row["image_path"], row, model=model, api_key=api_key)
        parsed = parse_response(raw)
        result_rows.append({**row, "model": model, **parsed, "raw_response": raw})
        write_csv(results, RESULT_FIELDS, result_rows)
        write_text(summary_path, summary(result_rows, model, samples_per_requirement, len(rows)))
        print(row["sample_id"], parsed["valid"], parsed["confidence"], flush=True)
        time.sleep(8)

    write_csv(manifest, MANIFEST_FIELDS, rows)
    write_csv(results, RESULT_FIELDS, result_rows)
    write_text(summary_path, summary(result_rows, model, samples_per_requirement, len(rows)))
    return [manifest, results, summary_path]


def read_existing_results(path: Path) -> list[CsvRow]:
    if not path.exists():
        return []
    from shared import read_csv_rows

    return read_csv_rows(path)


def summary(
    rows: list[CsvRow],
    model: str,
    samples_per_requirement: int,
    total_samples: int | None = None,
) -> str:
    valid_rates = []
    by_dataset: dict[str, list[CsvRow]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)
        valid_rates.append(1.0 if row["valid"] == "yes" else 0.0)

    lines = [
        "# LLM Validity Audit Summary",
        "",
        f"Model: `{model}`.",
        f"Samples per requirement: `{samples_per_requirement}`.",
        f"Completed samples: `{len(rows)}/{total_samples or len(rows)}`.",
        "",
        "OpenRouter judges whether generated images visibly satisfy the natural-language "
        "requirement with Gemini Flash. This is an external audit, not ground truth.",
        "If quota stops a run, rerun the same command later; cached rows are skipped.",
        "",
        f"Completed-sample valid rate: {mean(valid_rates):.3f}.",
        "",
    ]
    for dataset, dataset_rows in sorted(by_dataset.items()):
        rate = mean(1.0 if row["valid"] == "yes" else 0.0 for row in dataset_rows)
        unclear = sum(1 for row in dataset_rows if row["valid"] == "unclear")
        lines.append(f"- {dataset}: valid rate {rate:.3f}, unclear {unclear}/{len(dataset_rows)}")
    return "\n".join(lines) + "\n"
