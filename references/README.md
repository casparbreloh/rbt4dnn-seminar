# References and source provenance

This project links to canonical public records instead of committing local copies of papers,
course slides, or upstream artifacts. Dates and revisions below make clear which material the
seminar work used.

## Primary study

- Nusrat Jahan Mozumder, Felipe Toledo, Swaroopa Dola, and Matthew B. Dwyer.
  **“rbt4dnn: Requirements-based Testing of Neural Networks.”** arXiv:2504.02737, version 3,
  3 September 2025. [Abstract and versioned files](https://arxiv.org/abs/2504.02737).
- **Original RBT4DNN artifact:**
  [less-lab-uva/RBT4DNN](https://github.com/less-lab-uva/RBT4DNN). The data contract in this
  repository pins commit `c3deff871ed41043d83a05289faab84c41706586`; use the
  [permanent commit view](https://github.com/less-lab-uva/RBT4DNN/tree/c3deff871ed41043d83a05289faab84c41706586)
  when auditing the exact source. No repository license was present at that revision.
- **Author-hosted trained-model records:**
  [Zenodo record 14051679](https://doi.org/10.5281/zenodo.14051679). Consult the individual
  record/version metadata and file terms before downloading or redistributing model artifacts.

## Closely related work

- Oliver Weißl, Vincenzo Riccio, Severin Kacianka, and Andrea Stocco.
  **“HyperNet-Adaptation for Diffusion-Based Test Case Generation.”** arXiv:2601.15041, 2026.
  [Abstract and versioned files](https://arxiv.org/abs/2601.15041).

## Local-only teaching material

The seminar introduction deck, used only to understand the course framing and deliverables, is
kept as an ignored local reference copy. It is not part of this public artifact because no public
redistribution grant or stable canonical course URL has been established. No experimental value
is sourced from that deck.

## Data and measurement boundary

The exact generated-image subset is described by [`../data/manifest.json`](../data/manifest.json)
and its checksum sidecar. It is fetched from the pinned original-artifact commit rather than
redistributed here. Aggregate paper/artifact values used by this study are identified in
`data/requirements.csv` and in the limitation section of each experiment summary. Independently
generated extension results live beside their experiment code.

See [`../NOTICE.md`](../NOTICE.md) for the licensing boundary between this project's original
contributions and third-party material.
