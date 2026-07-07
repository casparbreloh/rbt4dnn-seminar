# MNIST Shared Generator Summary

A single conditional VAE was trained on the copied RBT4DNN MNIST LoRA images for M1-M6, then generated images by resampling the learned latent space.

It generated 100 images per requirement. This is a cheap shared-generator baseline, not a FLUX LoRA reproduction.

Mean pass rate: 0.940 versus 0.942 for the paper's per-requirement LoRA reference.
Worst requirement: M3 at pass 0.710000 (29 failures).
Sample grid: `experiments/mnist-shared-generator/samples.png`.

- M1: pass 1.000000 (delta +0.001000)
- M2: pass 0.950000 (delta -0.027000)
- M3: pass 0.710000 (delta -0.014000)
- M4: pass 0.990000 (delta +0.008000)
- M5: pass 1.000000 (delta +0.006000)
- M6: pass 0.990000 (delta +0.014000)
