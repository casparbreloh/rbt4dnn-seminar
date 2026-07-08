# CelebA-HQ Shared Generator Summary

A single conditional VAE was trained on copied CelebA-HQ RBT4DNN LoRA images.

Evaluation uses a small requirement classifier trained on the copied paper images plus nearest-train image checks. This is not a replacement for the paper's attribute-classifier pass rate.

Classifier validation accuracy on copied paper images: 0.636.
Mean generated-image classifier top-1 alignment: 0.613.
Exact generated/training image matches: 0.
Hardest requirement by classifier top-1: C4 (0.333333).
Sample grid: `experiments/celeba-shared-generator/samples.png`.

- C1: classifier top-1 0.875000 (std 0.000000, margin 0.380670)
- C2: classifier top-1 0.416667 (std 0.000000, margin -0.076625)
- C3: classifier top-1 0.458333 (std 0.000000, margin 0.020027)
- C4: classifier top-1 0.333333 (std 0.000000, margin -0.100723)
- C5: classifier top-1 1.000000 (std 0.000000, margin 0.520457)
- C6: classifier top-1 0.583333 (std 0.000000, margin 0.147371)
- C7: classifier top-1 0.625000 (std 0.000000, margin 0.156258)
