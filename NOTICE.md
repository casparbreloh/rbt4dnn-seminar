# Copyright, licensing, and provenance notice

This is an independent seminar study of **rbt4dnn: Requirements-based Testing of Neural
Networks**. It is not the original RBT4DNN artifact and is not affiliated with or endorsed by its
authors, the University of Virginia, or the LESS Lab.

## Material authored for this project

- Original software in `src/`, `scripts/`, `tests/`, and the experiment implementations is
  licensed under the repository's [MIT License](LICENSE).
- Original prose in `README.md`, `references/`, experiment summaries, future `paper/` sources,
  and future `slides/` sources is licensed under
  [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
  (SPDX: `CC-BY-4.0`).
- Original tabular results, plots, and images produced by this project's extension experiments
  are available under CC BY 4.0 unless a file or its accompanying summary states otherwise.

These grants apply only to contributions for which Caspar Breloh holds the relevant rights. They
do not change the terms of embedded, quoted, measured, or derived third-party material.

## Upstream RBT4DNN material

The original [RBT4DNN repository](https://github.com/less-lab-uva/RBT4DNN) did not contain a
license when audited at commit `c3deff871ed41043d83a05289faab84c41706586`. In the absence of a
license grant, its code, result files, generated images, model artifacts, and other content are
**not** covered by this repository's MIT or CC BY 4.0 licenses.

Accordingly:

- copied upstream source archives and result-file mirrors are not distributed in the current
  tree;
- `scripts/fetch_data.py` retrieves the required image subset directly from the pinned upstream
  repository and verifies it locally;
- `data/manifest.json` and `data/manifest-files.tsv` describe provenance and integrity but do not
  grant permission to copy or redistribute upstream files; and
- columns in `data/requirements.csv` that transcribe or derive from upstream measurements remain
  subject to any rights in their source data. The project's selection, annotations, analysis
  logic, and independently generated measurements do not imply ownership of those source values.

The generated-image corpus may also inherit restrictions from underlying datasets, including the
CelebA/CelebAMask-HQ family. Users are responsible for checking upstream and dataset terms for
their intended use. This notice is a provenance record, not legal advice.

## Papers, course material, services, and models

Third-party papers and course slides are cited in [`references/README.md`](references/README.md)
instead of being redistributed. Their copyrights and terms remain with their respective owners.

Names and marks such as GitHub, Google Colab, OpenRouter, Gemini, FLUX, CelebA-HQ, MNIST, and
ImageNet belong to their respective owners. Dependencies, hosted services, model weights, and
datasets are governed by their own licenses and terms. Their mention does not imply endorsement.
