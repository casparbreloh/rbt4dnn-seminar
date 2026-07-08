# Cost Analysis Summary

**Question:** If only valid failures matter, which requirement-testing targets look most cost-effective?

**Method:** Estimate valid failures from pass and precondition rates, then divide illustrative compute, generation, validation, and engineering costs by that yield.

**Result:** SGSM S2, MNIST M3, and SGSM S1 are the cheapest estimated valid-failure targets under the committed assumptions.

**Limitation:** Dollar values are illustrative assumptions, not measured invoices. Estimated failure counts use the paper's reported generated-test counts, not the smaller copied image samples in this repo.

## Largest Estimated Valid-Failure Yields

- sgsm S2 lr: 7364.400 estimated valid failures
- mnist M3 lr: 2930.522 estimated valid failures
- sgsm S1 lr: 2633.400 estimated valid failures
- celeba-hq C6 lr: 2004.183 estimated valid failures
- celeba-hq C2 lr: 1170.296 estimated valid failures

## Cheapest Estimated Valid Failures

- sgsm S2 lr: $0.05 per estimated valid failure
- mnist M3 lr: $0.12 per estimated valid failure
- sgsm S1 lr: $0.14 per estimated valid failure
- celeba-hq C6 lr: $0.18 per estimated valid failure
- celeba-hq C2 lr: $0.30 per estimated valid failure
