# CelebA-HQ Shared Generator Summary

A single conditional VAE was trained on copied CelebA-HQ RBT4DNN LoRA images.

Evaluation uses a small requirement classifier trained on the copied paper images plus nearest-train image checks. This is not a replacement for the paper's attribute-classifier pass rate.

Classifier validation accuracy on copied paper images: 0.557.
Mean generated-image classifier top-1 alignment: 0.488.
Exact generated/training image matches: 0.
Hardest requirement by classifier top-1: C6 (0.041667).
Sample grid: `experiments/celeba-shared-generator/samples.png`.

- C1: classifier top-1 0.833333 (std 0.000000, margin 0.419887)
- C2: classifier top-1 0.625000 (std 0.000000, margin 0.103098)
- C3: classifier top-1 0.125000 (std 0.000000, margin -0.108602)
- C4: classifier top-1 0.333333 (std 0.000000, margin -0.083782)
- C5: classifier top-1 0.875000 (std 0.000000, margin 0.257278)
- C6: classifier top-1 0.041667 (std 0.000000, margin -0.276569)
- C7: classifier top-1 0.583333 (std 0.000000, margin 0.131986)
