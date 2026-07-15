# Replication Summary

**Question:** How close are the copied, precomputed MNIST pass-rate values to the paper's reported classifier pass rates?

**Method:** Copy the precomputed local pass-rate fields from `data/requirements.csv` and compare them with the paper reference fields. This command does not classify images.

**Result:** Rows checked: `7`. Largest absolute delta versus the paper reference: `0.013`.

**Limitation:** This is an artifact-level check, not a fresh reproduction of LoRA training or FLUX sampling.

## Details

- M1: local `1.000`, paper `0.999`, delta `+0.001`
- M2: local `0.990`, paper `0.977`, delta `+0.013`
- M3: local `0.730`, paper `0.724`, delta `+0.006`
- M4: local `0.970`, paper `0.982`, delta `-0.012`
- M5: local `1.000`, paper `0.994`, delta `+0.006`
- M6: local `0.980`, paper `0.976`, delta `+0.004`
- M7: local `0.980`, paper `0.981`, delta `-0.001`
