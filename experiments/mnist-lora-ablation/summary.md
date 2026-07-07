# MNIST LoRA Ablation Summary

Compares per-requirement LoRA (`lr`) with the paper's shared MNIST `allreq` and `alldata` ablations. Valid-failure values are aggregate estimates from pass and precondition rates.

- M1: best estimated valid-failure efficiency is `allreq` (13.598 per 1000). `allreq/alldata` ratios are measured against `lr=1.065`.
- M2: best estimated valid-failure efficiency is `lr` (27.048 per 1000). `allreq/alldata` ratios are measured against `lr=27.048`.
- M3: best estimated valid-failure efficiency is `lr` (293.052 per 1000). `allreq/alldata` ratios are measured against `lr=293.052`.
- M4: best estimated valid-failure efficiency is `allreq` (31.020 per 1000). `allreq/alldata` ratios are measured against `lr=11.655`.
- M5: best estimated valid-failure efficiency is `allreq` (13.740 per 1000). `allreq/alldata` ratios are measured against `lr=2.520`.
- M6: best estimated valid-failure efficiency is `allreq` (49.036 per 1000). `allreq/alldata` ratios are measured against `lr=21.724`.
