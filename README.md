# rbt4dnn-seminar

Seminar work around **RBT4DNN: Requirements-based Testing of Neural Networks**.

- Original code/data release: https://github.com/less-lab-uva/RBT4DNN
- Paper: https://arxiv.org/abs/2504.02737

This repo is notebook-first. The notebooks are the experiment entry points; the
small shared helpers live directly in `src/`.

## Notebooks

- [Replication](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/replication/notebook.ipynb)
- [Precondition validity](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/precondition-validity/notebook.ipynb)
- [MNIST LoRA ablation](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/mnist-lora-ablation/notebook.ipynb)
- [Cost analysis](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/cost-analysis/notebook.ipynb)
- [MNIST shared LoRA pilot](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/mnist-shared-lora-pilot/notebook.ipynb)

The notebooks clone this repo automatically in Colab if the data is not already
available. Finished notebooks write `results.csv` and `summary.md` into their
experiment folder so later Codex runs can inspect the actual findings.

## Data

The repo includes the copied generated images, copied paper result files, a
compact `data/requirements.csv`, and `data/rbt4dnn.lance`. The Lance dataset
stores external references to `data/images/...`, not another copy of the PNG
bytes.

Rebuild the Lance dataset with:

```bash
uv run python src/dataset.py
```

The original upstream scripts are preserved as `data/original-rbt4dnn-code.tar.gz`
for provenance, but they are not maintained as this repo's code.

## Setup

```bash
uv sync
```

Quality checks:

```bash
uv run ruff format --check
uv run ruff check
uv run ty check
```

## Scope

The replication is intentionally conservative: it uses the copied RBT4DNN
generated images and reported result files. Extension work should add new
notebooks under `experiments/` and reusable code under `src/` only when it is
shared by more than one experiment.
