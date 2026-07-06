# rbt4dnn-seminar

Seminar replication and extension work for **RBT4DNN: Requirements-based
Testing of Neural Networks**.

This repository builds on the original RBT4DNN paper/code release:

- Original code/data release: https://github.com/less-lab-uva/RBT4DNN
- Paper: https://arxiv.org/abs/2504.02737

## Layout

```text
reference/
  generated-images/
    mnist/                 M1-M7, Allreq_M1-M6, Alldata_M1-M6
    celeba-hq/             C1-C7
    sgsm/                  S1-S7
    imagenet/              I1-I4
  reported-results/        paper result files copied for comparison
  scripts/                 reference scripts copied from RBT4DNN
experiments/
  results.csv              compact table shared by the notebooks
  replication-mnist/       MNIST replication notebook and runner
  valid-failure/           pass rate plus precondition-match analysis
  cost-analysis/           cost per valid requirement-matching failure
packages/
  <future package>/        optional uv workspace packages later
```

Think of `reference/` as the exact paper material this repo depends on.
Think of `experiments/` as our seminar work on top.

This is not a mirror of the original repository. It keeps the generated images,
reported result files, and reference scripts needed for reproducibility and
comparison. It does not keep the original README, original requirements file,
training datasets, or model checkpoints.

The generated images are exact copies from the local RBT4DNN release we have:
14,500 PNGs across MNIST, CelebA-HQ, SGSM, and ImageNet.

New extension experiments should get their own folder under `experiments/`.
Large new outputs should go into ignored `outputs/` unless they are small,
final results worth committing.

## Environment

This repo is managed with [`uv`](https://docs.astral.sh/uv/). The root project is also a uv workspace, so later experiments can add packages under `packages/*` without changing the repository layout.

```bash
uv sync
```

## Open in Colab

- [MNIST replication](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/replication-mnist/replication_mnist.ipynb)
- [Valid failure overview](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/valid-failure/valid_failure_overview.ipynb)
- [Cost per valid failure](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/cost-analysis/cost_valid_failure.ipynb)

The notebooks clone this repository automatically in Colab if
`experiments/results.csv` is not already available.

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

This uses the copied MNIST generated images under
`reference/generated-images/mnist/`.

## Scope

This repo is our seminar repo. Copied paper material is kept only where it
supports exact reproduction, comparison, or auditability. Full datasets and
model checkpoints remain external.
