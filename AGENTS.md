# RBT4DNN Seminar Repository

This folder is the single Git working tree for the seminar study and reproducible
experiment artifact.

- `experiments/`: replication and extension analyses, notebooks, results, and summaries.
- `src/`: shared Python implementation used by the analyses.
- `scripts/`: root-relative reproduction and data-retrieval commands.
- `tests/`: lightweight artifact and data-contract tests.
- `data/manifest.json` and `data/manifest-files.tsv`: pinned provenance and checksums for
  the ignored image corpus restored under `data/images/`.
- `notizen/`: private local working notes; keep ignored and do not edit unless explicitly
  requested.
- `.context/`: local plans and migration records; keep ignored.
- `outputs/`: generated presentation and preview material; keep ignored.
- Root-level PDFs and `rbt4dnn-paper-fulltext.txt`: local reference copies; keep ignored.

Run project commands from this repository root. Do not recreate a nested `github/`
working tree or commit private notes, downloaded images, environments, caches, generated
outputs, or local reference copies.
