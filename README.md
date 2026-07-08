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
- [Gemini validity audit](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/gemini-validity-audit/notebook.ipynb)
- [MNIST shared generator](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/mnist-shared-generator/notebook.ipynb)
- [CelebA-HQ shared generator](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/celeba-shared-generator/notebook.ipynb)

## Results

- MNIST replication: copied generated images reproduce the paper pass rates closely
  (`M1-M7` deltas between `-0.012` and `+0.013`).
- Precondition validity: pass rate alone can be misleading; the strongest estimated
  valid-failure yields are `SGSM S2`, `MNIST M3`, and `SGSM S1`.
- Cost analysis: under the stated assumptions, the cheapest estimated valid
  failures are `SGSM S2` at `$0.05`, `MNIST M3` at `$0.12`, and `SGSM S1`
  at `$0.14` per estimated valid failure.
- Gemini validity audit: the API key quota stopped the first run at `16/42`
  images. The partial audit gives `0.812` Gemini-valid rate overall, with
  weaker validity on the first CelebA-HQ samples (`0.500`) than on MNIST
  (`0.857`). Rerunning resumes from the saved CSV.
- Shared-generator extension: one small conditional MNIST generator is evaluated
  over three seeds and reaches mean pass `0.941` versus `0.942` for the paper's
  per-requirement LoRA reference, with `0` exact generated/training image
  matches.
- CelebA-HQ shared-generator extension: tests whether the shared-generator idea
  still works on a harder face dataset. In the 64x64 Colab stress test, a
  small copied-image classifier reaches `0.557` validation accuracy, while
  generated images reach only `0.488` mean requirement alignment. This is a
  weak sanity check, not the paper's attribute-classifier pass rate, but it
  suggests the MNIST shared-generator result does not transfer cleanly to
  faces.

The notebooks clone this repo automatically in Colab if the data is not already
available. Finished notebooks write `results.csv` and `summary.md` into their
experiment folder so later Codex runs can inspect the actual findings.

In Colab, always run the first cell before any experiment cell. If
`/content/rbt4dnn-seminar` already exists, that cell fetches `origin/main`,
resets the clone to the latest GitHub commit, and clears stale Python imports.
The first cell prints the active commit so stale notebooks are easier to catch.

To regenerate the non-training CSV/Markdown results:

```bash
uv run python scripts/run_experiments.py
```

To regenerate everything, including the MNIST and CelebA-HQ generator runs:

```bash
uv run python scripts/run_experiments.py --all
```

For Colab CLI, use the same arguments through the wrapper:

```bash
uvx --from google-colab-cli colab run --gpu T4 scripts/colab_job.py --all
```

The training results use fixed seeds, but GPU kernels can still produce tiny
numeric drift across machines.

The Gemini validity audit is intentionally not part of `--all`, because it calls
an external model. To run it, set `GEMINI_API_KEY` or `GOOGLE_API_KEY`, then run:

```bash
uv run python scripts/run_experiments.py --run-gemini-audit
```

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
