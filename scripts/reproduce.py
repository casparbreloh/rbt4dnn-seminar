from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path


Experiment = Callable[[Path], list[Path]]


def find_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "data" / "requirements.csv").exists():
            return candidate
    raise FileNotFoundError("Run this from inside the rbt4dnn-seminar repo.")


def add_paths(root: Path) -> None:
    sys.path.insert(0, str(root / "src"))
    for path in sorted((root / "experiments").iterdir()):
        if path.is_dir():
            sys.path.append(str(path))


def replication(root: Path) -> list[Path]:
    from mnist import write_replication_outputs

    return list(write_replication_outputs(root))


def precondition_validity(root: Path) -> list[Path]:
    from precondition_validity import write_results

    return list(write_results(root))


def mnist_lora_ablation(root: Path) -> list[Path]:
    from mnist_lora_ablation import write_results

    return list(write_results(root))


def cost_analysis(root: Path) -> list[Path]:
    from cost_analysis import write_results

    return list(write_results(root))


EXPERIMENTS: dict[str, Experiment] = {
    "replication": replication,
    "precondition-validity": precondition_validity,
    "mnist-lora-ablation": mnist_lora_ablation,
    "cost-analysis": cost_analysis,
}


def train_shared_generator(root: Path) -> list[Path]:
    from mnist_shared_generator import TrainConfig, train_and_evaluate

    return train_and_evaluate(root, TrainConfig(), seeds=[7, 13, 29])


def train_celeba_generator(root: Path) -> list[Path]:
    from celeba_shared_generator import TrainConfig, train_and_evaluate

    return train_and_evaluate(root, TrainConfig(), seeds=[7])


def llm_validity_audit(root: Path, samples_per_requirement: int) -> list[Path]:
    from llm_validity_audit import run_audit

    return run_audit(root, samples_per_requirement=samples_per_requirement)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training",
        action="store_true",
        help="Also rerun generator training.",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Also rerun the cached/external LLM audit.",
    )
    args = parser.parse_args()

    root = find_root()
    add_paths(root)

    written: list[Path] = []
    for name in EXPERIMENTS:
        written.extend(EXPERIMENTS[name](root))
    if args.training:
        written.extend(train_shared_generator(root))
        written.extend(train_celeba_generator(root))
    if args.llm:
        written.extend(llm_validity_audit(root, samples_per_requirement=2))

    for path in written:
        print(path.relative_to(root) if path.is_relative_to(root) else path)


if __name__ == "__main__":
    main()
