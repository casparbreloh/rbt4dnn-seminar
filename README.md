# rbt4dnn-seminar

Seminar replication and extension work for **RBT4DNN: Requirements-based Testing of Neural Networks**.

This repository builds on the original RBT4DNN paper artifact:

- Paper artifact: https://github.com/less-lab-uva/RBT4DNN
- Paper: https://arxiv.org/abs/2504.02737

## What is here

- `results.csv`  
  Compact table used by the notebooks. It combines artifact pass rates, precondition-match rates, local MNIST replication numbers, and notes about missing artifact fields.
- `replication_mnist.ipynb`  
  Minimal MNIST replication summary.
- `valid_failure_overview.ipynb`  
  Cross-dataset view over MNIST, CelebA-HQ, and SGSM: pass rate plus precondition match.
- `cost_valid_failure.ipynb`  
  Cost-per-valid-failure analysis.
- `rerun_mnist_model.py`  
  Optional script to rerun the MNIST model on the copied generated images.
- `rbt4dnn-artifact/`  
  Selected upstream artifact files needed for this seminar work.

## Open in Colab

- [MNIST replication](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/replication_mnist.ipynb)
- [Valid failure overview](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/valid_failure_overview.ipynb)
- [Cost per valid failure](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/cost_valid_failure.ipynb)

The notebooks clone this repository automatically in Colab if `results.csv` is not already mounted.

## Main extension idea

Raw pass rate is not enough for requirements-based testing. A generator can make a model look good if it mostly misses the requirement precondition.

The more useful metric is:

```text
cost per valid requirement-matching failure
```

For MNIST M3 (`very thick 7`), the per-requirement LoRA finds many valid failures, while the all-data LoRA has a high raw pass rate because it rarely generates valid M3 tests.

## Reproduce the MNIST rerun

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python rerun_mnist_model.py
```

This uses the copied MNIST generated images under `rbt4dnn-artifact/mnist-images/`.

## Scope

This is not a full fork of the original artifact. It keeps the seminar experiments small and focused while preserving the upstream files needed for the reported tables. Full datasets, model checkpoints, and non-MNIST generated images remain external; see `artifact.md`.
