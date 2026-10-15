# Reproduce Stage0 evidence

Run in the registered repository on the original host after obtaining the binary artifacts using [68_stage0_artifact_retrieval.md](../../68_stage0_artifact_retrieval.md). Git includes raw textual outcomes, rollouts, generation calls, audits and bindings; learner/adapter binaries remain server assets with recorded hashes. A Git clone alone is not a complete model checkpoint package. Keep historical outputs and use a new report output directory.

The original environment is `.venv-rl`; model/cache and environment versions are bound by the registered execution/job artifacts and original run receipts. No paid API, new generation or optimization is required for this CPU replay. The report re-audits all51 training branches and104 evaluation panels, checks the sealed models/standardization/point predictions against the exact frozen fit, and scores the original heldout forecasts.

```bash
CUDA_VISIBLE_DEVICES= .venv-rl/bin/python pipeline/report_bfcl_transfer_prediction_eight.py \
  --fit-manifest rl/bfcl_source_stage0_fit_jobs_20261013/manifest.json \
  --fit-manifest-sha256 107af98a69bc9c9ce1f166634901d6c005ddda34d4a4465bc97fb87388b90b65 \
  --heldout-manifest rl/bfcl_source_stage0_heldout_jobs_20261015/manifest.json \
  --heldout-manifest-sha256 73cdc552862072169c6bfa5ca83a2dfd5d2b9178a5154bc0f35a0874b5e63b04 \
  --prediction-receipt rl/bfcl_source_stage0_prediction_seal_20261015/receipt.json \
  --prediction-receipt-sha256 c19ddc54e26aff0bc7579d0e185cd2bafad075ae928c0e58a6fef5f094cc2c15 \
  --output rl/bfcl_source_stage0_transfer_prediction_reproduction
python3 pipeline/describe_bfcl_stage0_complete.py \
  --output rl/bfcl_source_stage0_transfer_prediction_reproduction_descriptive.json
```

The descriptive supplement reads the published canonical full104 report (SHA c26424eee244968470b478a32ad6bf4b96d5563197972bc04331d6bc753bb992), verifies the audited per-task files and computes descriptive repeats, task/stratum cells and error contributions. It does not refit/select a model or create a new primary test. Its timestamp/output metadata may differ.

The original seals are E238/E239; execution-chain reviews are E145/E237/E275. The generated report's `implementation_sha256` and this directory's evidence manifest bind the exact analysis code. Preserve prediction publication and actual child-start timing evidence; do not regenerate sealed forecasts to repair a publication problem.
