# rbt4dnn-seminar

Seminar work around **RBT4DNN: Requirements-based Testing of Neural Networks**.

- Original code/data release: https://github.com/less-lab-uva/RBT4DNN
- Paper: https://arxiv.org/abs/2504.02737

This repo is notebook-first. The notebooks are the experiment entry points; the
small shared helpers live directly in `src/`.

The repo is meant to be the public review artifact. Private notes, slides, and
paper drafts can live in Google Drive, but the experiments and inspectable
results should not depend on private Drive links.

## Notebooks

- [Replication](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/replication/notebook.ipynb)
- [Precondition validity](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/precondition-validity/notebook.ipynb)
- [MNIST LoRA ablation](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/mnist-lora-ablation/notebook.ipynb)
- [Cost analysis](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/cost-analysis/notebook.ipynb)
- [MNIST shared generator](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/mnist-shared-generator/notebook.ipynb)

The notebooks clone this repo automatically in Colab if the data is not already
available. Finished notebooks write `results.csv` and `summary.md` into their
experiment folder so later Codex runs can inspect the actual findings.

In Colab, always run the first cell before any experiment cell. If
`/content/rbt4dnn-seminar` already exists, that cell fetches `origin/main`,
resets the clone to the latest GitHub commit, and clears stale Python imports.
The first cell prints the active commit so stale notebooks are easier to catch.

To regenerate the current CSV/Markdown results without opening notebooks:

```bash
uv run python scripts/run_experiments.py
```

To also train the shared MNIST generator extension:

```bash
uv run python scripts/run_experiments.py --train-shared-generator
```

Colab CLI execution uses the same code through `scripts/colab_job.py`.

## Data

The repo includes copied generated images, copied paper result files, and a
compact `data/requirements.csv` table used by the notebooks. ImageNet examples
are kept for completeness, while the requirement table covers the MNIST,
CelebA-HQ, and SGSM rows used by the current analyses.

The original upstream scripts are preserved as `data/original-rbt4dnn-code.tar.gz`
for provenance, but they are not maintained as this repo's code. The public
upstream artifact includes sampling/evaluation scripts, not a self-contained
FLUX LoRA training recipe.

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
