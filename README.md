# rbt4dnn-seminar

Seminar replication and extension work for **RBT4DNN: Requirements-based Testing of Neural Networks**.

This repository builds on the original RBT4DNN paper artifact:

- Paper artifact: https://github.com/less-lab-uva/RBT4DNN
- Paper: https://arxiv.org/abs/2504.02737

## Layout

```text
data/
  results.csv              compact table shared by the notebooks
  rbt4dnn-artifact/        selected upstream artifact files
docs/
  artifact.md              what was copied and what stays external
experiments/
  replication-mnist/       MNIST replication notebook and runner
  valid-failure/           pass rate plus precondition-match analysis
  cost-analysis/           cost per valid requirement-matching failure
packages/
  <future package>/        optional uv workspace packages later
```

New extension experiments should get their own folder under `experiments/`.
Small shared tables belong in `data/`; large generated outputs belong in ignored
`outputs/` or outside the repo.

## Environment

This repo is managed with [`uv`](https://docs.astral.sh/uv/). The root project is also a uv workspace, so later experiments can add packages under `packages/*` without changing the repository layout.

```bash
uv sync
```

## Open in Colab

- [MNIST replication](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/replication-mnist/replication_mnist.ipynb)
- [Valid failure overview](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/valid-failure/valid_failure_overview.ipynb)
- [Cost per valid failure](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/cost-analysis/cost_valid_failure.ipynb)

The notebooks clone this repository automatically in Colab if `data/results.csv`
is not already available.

## Main extension idea

Raw pass rate is not enough for requirements-based testing. A generator can make a model look good if it mostly misses the requirement precondition.

The more useful metric is:

```text
cost per valid requirement-matching failure
```

For MNIST M3 (`very thick 7`), the per-requirement LoRA finds many valid failures, while the all-data LoRA has a high raw pass rate because it rarely generates valid M3 tests.

## Reproduce the MNIST rerun

```bash
uv run python experiments/replication-mnist/rerun_mnist_model.py
```

This uses the copied MNIST generated images under `data/rbt4dnn-artifact/mnist-images/`.

## Scope

This is not a full fork of the original artifact. It keeps the seminar experiments small and focused while preserving the upstream files needed for the reported tables. Full datasets, model checkpoints, and non-MNIST generated images remain external; see `docs/artifact.md`.
