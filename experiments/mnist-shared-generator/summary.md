# MNIST Shared Generator Summary

**Question:** Can a small shared generator approximate the paper's per-requirement LoRA behavior on MNIST?

**Method:** Train one conditional VAE on copied RBT4DNN MNIST LoRA images for M1-M6, using 3 seeds and 100 generated images per requirement per seed.

**Result:** Mean pass rate is 0.945 versus 0.942 for the paper's per-requirement LoRA reference, with no exact training-image matches.

**Limitation:** This is a cheap shared-generator baseline, not a FLUX LoRA reproduction, and MNIST is much easier than natural-image datasets.

Exact generated/training image matches: 0. Mean nearest-train MSE: 0.0020.
Worst requirement: M3 at mean pass 0.743333 (25.667 mean failures).
Sample grid: `experiments/mnist-shared-generator/samples.png`.

- M1: mean pass 1.000000 (std 0.000000, delta +0.001000)
- M2: mean pass 0.956667 (std 0.035119, delta -0.020333)
- M3: mean pass 0.743333 (std 0.116762, delta +0.019333)
- M4: mean pass 0.973333 (std 0.015275, delta -0.008667)
- M5: mean pass 1.000000 (std 0.000000, delta +0.006000)
- M6: mean pass 0.996667 (std 0.005774, delta +0.020667)
