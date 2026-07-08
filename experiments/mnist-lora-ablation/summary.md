# MNIST LoRA Ablation Summary

**Question:** Does MNIST really need one LoRA per requirement, or can the paper's shared ablations produce useful valid failures too?

**Method:** Compare per-requirement LoRA (`lr`) with the paper's shared MNIST `allreq` and `alldata` ablations using estimated valid failures per 1000 images.

**Result:** Shared `allreq` is best for M1, M4, M5, and M6; per-requirement LoRA is best for M2 and M3.

**Limitation:** This reuses the paper's aggregate pass and precondition rates; it does not train new LoRAs.

## Details

- M1: best estimated valid-failure efficiency is `allreq` (13.598 per 1000). `allreq/alldata` ratios are measured against `lr=1.065`.
- M2: best estimated valid-failure efficiency is `lr` (27.048 per 1000). `allreq/alldata` ratios are measured against `lr=27.048`.
- M3: best estimated valid-failure efficiency is `lr` (293.052 per 1000). `allreq/alldata` ratios are measured against `lr=293.052`.
- M4: best estimated valid-failure efficiency is `allreq` (31.020 per 1000). `allreq/alldata` ratios are measured against `lr=11.655`.
- M5: best estimated valid-failure efficiency is `allreq` (13.740 per 1000). `allreq/alldata` ratios are measured against `lr=2.520`.
- M6: best estimated valid-failure efficiency is `allreq` (49.036 per 1000). `allreq/alldata` ratios are measured against `lr=21.724`.
