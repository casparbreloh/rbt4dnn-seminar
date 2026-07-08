# rbt4dnn-seminar

Seminar replication and extension work for **RBT4DNN: Requirements-based Testing of Neural Networks**.

- Paper: https://arxiv.org/abs/2504.02737
- Original artifact: https://github.com/less-lab-uva/RBT4DNN

This repository is the public experiment artifact. It is notebook-first, with small Python helpers kept next to each experiment or in `src/`.

## Experiments

- [Replication](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/replication/notebook.ipynb)
- [Precondition validity](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/precondition-validity/notebook.ipynb)
- [MNIST LoRA ablation](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/mnist-lora-ablation/notebook.ipynb)
- [Cost analysis](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/cost-analysis/notebook.ipynb)
- [LLM validity audit](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/llm-validity-audit/notebook.ipynb)
- [MNIST shared generator](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/mnist-shared-generator/notebook.ipynb)
- [CelebA-HQ shared generator](https://colab.research.google.com/github/casparbreloh/rbt4dnn-seminar/blob/main/experiments/celeba-shared-generator/notebook.ipynb)

## Main Results

- **Artifact check:** copied MNIST generated images match the paper reference closely; largest pass-rate delta is `0.013`.
- **Validity matters:** the strongest aggregate estimated valid-failure targets are `SGSM S2`, `MNIST M3`, and `SGSM S1`.
- **Cost signal:** under illustrative assumptions, cheapest estimated valid failures are `SGSM S2` (`$0.05`), `MNIST M3` (`$0.12`), and `SGSM S1` (`$0.14`).
- **LLM audit:** Gemini Flash via OpenRouter marks `0.738` of a fixed `42` image sample valid; CelebA-HQ is strongest (`1.000`), SGSM weakest (`0.500`).
- **Shared generator:** a small MNIST shared VAE reaches mean pass `0.945` vs `0.942` for the paper's per-requirement LoRA reference, with `0` exact training-image matches.
- **Harder dataset check:** the CelebA-HQ shared generator is only exploratory: classifier validation is `0.636`, generated-image alignment is `0.613`.

## Reproduce

```bash
uv sync
uv run python scripts/run_experiments.py
```

Run the training experiments too:

```bash
uv run python scripts/run_experiments.py --all
```

Run the external LLM audit:

```bash
OPENROUTER_API_KEY=... uv run python scripts/run_experiments.py --run-llm-audit
```

Colab CLI uses the same runner:

```bash
uvx --from google-colab-cli colab run --gpu T4 scripts/colab_job.py --all
```

Quality checks:

```bash
uv run ruff format --check
uv run ruff check
uv run ty check
```

## Scope

The replication is intentionally conservative: it uses copied paper-release outputs and result files, not fresh FLUX LoRA training. Cost and valid-failure numbers are aggregate estimates. The LLM audit is an external semantic check, not ground truth. Training outputs use fixed seeds, but GPU kernels can still cause small drift; the committed CSVs are the reference run.
