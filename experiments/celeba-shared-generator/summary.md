# CelebA-HQ Shared Generator Summary

**Question:** Does the cheap shared-generator idea still look plausible on a harder natural-image dataset?

**Method:** Train one conditional VAE on copied CelebA-HQ RBT4DNN LoRA images and evaluate generated images with a small requirement classifier plus nearest-train image checks.

**Result:** Classifier validation accuracy is 0.636; generated-image classifier top-1 alignment is 0.613.

**Limitation:** This is exploratory and not a replacement for the paper's attribute-classifier pass rate. The classifier itself is weak, so the result is a caution signal rather than a final benchmark.

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
