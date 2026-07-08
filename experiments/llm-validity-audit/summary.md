# LLM Validity Audit Summary

**Question:** Do generated images visibly satisfy the natural-language requirement when judged semantically rather than only by classifier pass rate?

**Method:** Sample two main LoRA images per requirement for MNIST, CelebA-HQ, and SGSM, then ask Gemini Flash via OpenRouter for a JSON validity judgment.

**Result:** Completed-sample valid rate: 0.738.

**Limitation:** This is an external LLM audit, not ground truth. The sample is small and cached rows are reused when present.

Model: `google/gemini-3-flash-preview`.
Samples per requirement: `2`.
Completed samples: `42/42`.
If quota stops a run, rerun the same command later; cached rows are skipped.

- celeba-hq: valid rate 1.000, unclear 0/14
- mnist: valid rate 0.714, unclear 0/14
- sgsm: valid rate 0.500, unclear 0/14
