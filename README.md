# Requirements-based Testing of Neural Networks

This repository contains an independent seminar review, artifact check, and set of
exploratory extensions for **rbt4dnn: Requirements-based Testing of Neural Networks** by
Nusrat Jahan Mozumder, Felipe Toledo, Swaroopa Dola, and Matthew B. Dwyer.

- [Original paper (arXiv:2504.02737)](https://arxiv.org/abs/2504.02737)
- [Original RBT4DNN artifact](https://github.com/less-lab-uva/RBT4DNN)
- [Seminar project source](https://github.com/casparbreloh/rbt4dnn-seminar)
- [HTML presentation](https://casparbreloh.github.io/rbt4dnn-seminar/)

This is not an official RBT4DNN repository and is not affiliated with or endorsed by the
original authors. Their paper and artifact remain the authoritative sources for RBT4DNN.

## What this project contributes

The repository deliberately separates recorded upstream evidence from work performed for this
seminar:

| Area | Relationship to the original study |
| --- | --- |
| Artifact comparison | Compares recorded, precomputed MNIST pass-rate fields with values reported by the paper. It does not retrain the paper's LoRAs or freshly classify the full image corpus. |
| Precondition and cost analyses | Original secondary analyses over aggregate values reported or released by the RBT4DNN authors, with explicit assumptions and limitations. |
| MNIST LoRA ablation | Original comparison logic applied to the paper's aggregate ablation measurements; no new LoRAs are trained. |
| LLM validity audit | An original, small external semantic audit of a fixed sample of upstream-generated images. It is not ground truth. |
| Shared-generator experiments | Original exploratory VAE baselines trained on upstream-generated samples for MNIST and CelebA-HQ. They are not replications of the paper's FLUX/LoRA method. |
| Review, paper, and presentation | Independent authored synthesis and critique for the seminar. The HTML slide source is included; manuscript source is added when reviewed. |

The main observations are therefore not all replication results. The artifact comparison found a
largest MNIST pass-rate difference of `0.013` between its recorded fields and the paper reference.
The secondary analyses identify SGSM S2, MNIST M3, and SGSM S1 as high estimated valid-failure
targets under their assumptions. The fixed 42-image LLM audit produced a `0.738` valid rate. The
small MNIST shared VAE reached mean pass `0.945`; the harder CelebA-HQ experiment remained
exploratory, with classifier validation `0.636` and generated-image alignment `0.613`. Read the
individual summaries before interpreting these figures.

## Repository map

- [`experiments/`](experiments/) — notebooks, committed small results, and a summary for each
  artifact comparison, secondary analysis, or extension.
- [`src/`](src/) — shared Python implementation.
- [`scripts/`](scripts/) — root-relative data retrieval and reproduction commands.
- [`tests/`](tests/) — lightweight artifact and data-contract tests.
- [`data/`](data/) — the authored aggregate table plus provenance and checksum manifests; the
  upstream image corpus is restored locally and remains ignored.
- [`references/`](references/) — stable citations and provenance notes instead of copied papers.
- [`slides/`](slides/) — the single-file HTML presentation, its reusable
  [design-system primitives](slides/DESIGN_SYSTEM.md), and independently generated assets; the
  live deck is <https://casparbreloh.github.io/rbt4dnn-seminar/>.
- `paper/` — reviewed manuscript source once published.

Private working notes, local reference PDFs, downloaded images, environments, caches, and
generated previews are intentionally not part of the public repository.

## Reproduce the lightweight analyses

[Install `uv`](https://docs.astral.sh/uv/getting-started/installation/), then run from the
repository root:

```bash
uv sync
uv run python scripts/reproduce.py
```

The default command is metadata-only. It rebuilds summaries from committed CSV data, including
aggregate fields derived from the original artifact. The replication table copies precomputed
local pass-rate fields from `data/requirements.csv`; it is a comparison of recorded results, not
a fresh image re-evaluation.

The experiment notebooks can also be opened directly in Colab:

- [Artifact comparison](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/replication/notebook.ipynb)
- [Precondition validity](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/precondition-validity/notebook.ipynb)
- [MNIST LoRA ablation](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/mnist-lora-ablation/notebook.ipynb)
- [Cost analysis](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/cost-analysis/notebook.ipynb)
- [LLM validity audit](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/llm-validity-audit/notebook.ipynb)
- [MNIST shared generator](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/mnist-shared-generator/notebook.ipynb)
- [CelebA-HQ shared generator](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/celeba-shared-generator/notebook.ipynb)

## Restore image-dependent inputs

The image audit, training baselines, and direct MNIST evaluation require the exact generated-image
subset from the original artifact. The repository records the upstream commit, 14,500 expected
paths, sizes, and SHA-256 hashes in [`data/manifest.json`](data/manifest.json) and
[`data/manifest-files.tsv`](data/manifest-files.tsv).

```bash
# Verify an existing local corpus, or restore it directly from the pinned upstream revision.
uv run python scripts/fetch_data.py --verify
uv run python scripts/fetch_data.py

# Run the corpus-dependent training experiments.
uv run python scripts/reproduce.py --training

# Run the external LLM audit (may incur API cost).
OPENROUTER_API_KEY=... uv run python scripts/reproduce.py --llm
```

The fetcher downloads directly from the original authors' repository, extracts only allowlisted
paths, and verifies every file. No license was found in that upstream repository during the
publication audit, so this project does not repackage or grant rights to its code, results, or
images. See [`data/README.md`](data/README.md) and [`NOTICE.md`](NOTICE.md) before using them.

## Checks

```bash
uv run ruff format --check
uv run ruff check
uv run ty check
uv run python -m unittest discover -s tests
uv run python scripts/check_publication.py
```

## Citation and licensing

Use [`CITATION.cff`](CITATION.cff) to cite this independent project, and cite the original paper
for RBT4DNN itself. Original project code is available under the [MIT License](LICENSE). Original
authored documentation is available under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Upstream-derived data, measurements,
images, papers, names, and other third-party material are excluded from those grants; their
provenance and applicable terms are described in [`NOTICE.md`](NOTICE.md).
