# Artifact Scope

This repository includes selected files from the RBT4DNN artifact:

```text
data/rbt4dnn-artifact/
  README.md
  requirements.txt
  results/
  scripts/
  mnist-images/
```

Included:

- all upstream `results/` files available locally,
- upstream scripts from the artifact root,
- MNIST generated images used by the replication and ablation work:
  - `M1` through `M7`,
  - `Allreq_M1` through `Allreq_M6`,
  - `Alldata_M1` through `Alldata_M6`.

Not included:

- full original datasets,
- trained model checkpoints,
- full CelebA-HQ, SGSM, and ImageNet generated image folders,
- the full 1 GB local artifact copy.

Reason:

- The selected files are enough to reproduce the seminar tables and rerun the included MNIST script.
- The omitted files are large and are better obtained from the original artifact and model archives.

Upstream source:

- https://github.com/less-lab-uva/RBT4DNN
- Zenodo checkpoints referenced by the upstream README.

Note: the upstream repository did not include a clear license file in the local artifact snapshot. This repository therefore treats copied upstream files as research artifact excerpts with attribution, not as relicensed material.
