# rbt4dnn-seminar

Seminar replication and extension work for **RBT4DNN: Requirements-based
Testing of Neural Networks**.

This repository builds on the original RBT4DNN paper artifact:

- Paper artifact: https://github.com/less-lab-uva/RBT4DNN
- Paper: https://arxiv.org/abs/2504.02737

## Layout

```text
artifact/
  generated-images/
    mnist/                 M1-M7, Allreq_M1-M6, Alldata_M1-M6
    celeba-hq/             C1-C7
    sgsm/                  S1-S7
    imagenet/              I1-I4
  results/                 upstream result files
  scripts/                 upstream scripts and notebooks
  upstream-readme.md       original artifact README
  upstream-requirements.txt
experiments/
  results.csv              compact table shared by the notebooks
  replication-mnist/       MNIST replication notebook and runner
  valid-failure/           pass rate plus precondition-match analysis
  cost-analysis/           cost per valid requirement-matching failure
packages/
  <future package>/        optional uv workspace packages later
```

Think of `artifact/` as the original RBT4DNN artifact, but with clearer
subfolders. Think of `experiments/` as the seminar layer on top.

This repo includes all generated-image PNGs available in the local artifact
copy: 14,500 images across MNIST, CelebA-HQ, SGSM, and ImageNet. It does not
include original training datasets or model checkpoints, because those are not
part of the local artifact copy and should stay external.

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
`artifact/generated-images/mnist/`.

## Scope

This is not a full fork of every external asset used by the paper. It keeps the
local artifact contents that matter for the seminar work: scripts, upstream
results, and all locally available generated images. Full datasets and model
checkpoints remain external.

The upstream repository did not include a clear license file in the local
artifact snapshot. Copied upstream files are therefore treated as attributed
research artifact excerpts, not as relicensed material.
