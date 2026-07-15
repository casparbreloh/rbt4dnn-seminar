# Data provenance

This directory contains metadata and small, reviewer-facing study inputs. It does not distribute
the original RBT4DNN code archive, the authors' raw result files, or their generated-image corpus.

## Tracked files

- `requirements.csv` is this project's consolidated analysis table. It combines transcribed or
  derived RBT4DNN measurements with annotations and fields produced by the seminar analyses. The
  original source values are not relicensed; see `NOTICE.md` and each experiment summary.
- `manifest.json` pins the original artifact repository and commit, records dataset counts, and
  classifies each execution path as metadata-only or corpus-required.
- `manifest-files.tsv` maps 14,500 allowlisted upstream image paths to ignored local paths and
  records size and SHA-256 for exact verification.

## Restored local files

Run the following from the repository root:

```bash
uv run python scripts/fetch_data.py
uv run python scripts/fetch_data.py --verify
```

The fetcher downloads from the original authors' pinned GitHub revision. It checks archive paths,
rejects links and traversal entries, avoids overwriting unexpected local files, and verifies each
file. Restored images are placed under `data/images/` and are ignored by Git.

The original repository had no license when audited at the pinned commit. Fetching a public file
does not itself grant redistribution rights. Users must review the upstream repository and the
terms of underlying datasets—including possible CelebA/CelebAMask-HQ restrictions—before reuse.
