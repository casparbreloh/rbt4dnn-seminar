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
    from mnist import write_replication_summary

    return [write_replication_summary(root)]


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


def parse_seeds(value: str) -> list[int]:
    return [int(seed.strip()) for seed in value.split(",") if seed.strip()]


def train_shared_generator(
    root: Path,
    epochs: int,
    samples_per_requirement: int,
    seeds: list[int],
) -> list[Path]:
    from mnist_shared_generator import TrainConfig, train_and_evaluate

    config = TrainConfig(epochs=epochs, samples_per_requirement=samples_per_requirement)
    return train_and_evaluate(root, config, seeds=seeds)


def train_celeba_generator(
    root: Path,
    epochs: int,
    samples_per_requirement: int,
    seeds: list[int],
) -> list[Path]:
    from celeba_shared_generator import TrainConfig, train_and_evaluate

    config = TrainConfig(epochs=epochs, samples_per_requirement=samples_per_requirement)
    return train_and_evaluate(root, config, seeds=seeds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=EXPERIMENTS, action="append")
    parser.add_argument("--train-shared-generator", action="store_true")
    parser.add_argument("--shared-generator-epochs", type=int, default=100)
    parser.add_argument("--shared-generator-samples", type=int, default=100)
    parser.add_argument("--shared-generator-seeds", default="7,13,29")
    parser.add_argument("--train-celeba-generator", action="store_true")
    parser.add_argument("--celeba-generator-epochs", type=int, default=40)
    parser.add_argument("--celeba-generator-samples", type=int, default=24)
    parser.add_argument("--celeba-generator-seeds", default="7")
    args = parser.parse_args()

    root = find_root()
    add_paths(root)

    names = args.only or list(EXPERIMENTS)
    written: list[Path] = []
    for name in names:
        written.extend(EXPERIMENTS[name](root))
    if args.train_shared_generator:
        written.extend(
            train_shared_generator(
                root,
                epochs=args.shared_generator_epochs,
                samples_per_requirement=args.shared_generator_samples,
                seeds=parse_seeds(args.shared_generator_seeds),
            )
        )
    if args.train_celeba_generator:
        written.extend(
            train_celeba_generator(
                root,
                epochs=args.celeba_generator_epochs,
                samples_per_requirement=args.celeba_generator_samples,
                seeds=parse_seeds(args.celeba_generator_seeds),
            )
        )

    for path in written:
        print(path.relative_to(root) if path.is_relative_to(root) else path)


if __name__ == "__main__":
    main()
