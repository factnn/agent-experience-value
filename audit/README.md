# audit/ — data-pool reconnaissance

Two kinds of files live here. They are not the same thing, and the difference matters
when you are judging how much of this directory is our own work.

## Ours

| File | What it does |
|---|---|
| `audit_pool.py` | Reads four public dataset cards at pinned SHAs plus Hugging Face Dataset Viewer windows (400 rows total) and records schema, field coverage, distributions and content hashes. Needs network; never executes trajectory content. |
| `probe_outcomes.py` | Public-only search for trial-level verifier artifacts across 11 candidate repos. Records 401/unavailable responses without assuming they are private. Never uses credentials. |
| `summaries.json` | Output of `audit_pool.py`. |
| `parquet_files.json` | Pinned file list, sizes and LFS SHA256 for the SFT-100K shards. |
| `tokenizer_model_metadata.json` | Pinned revision of the Qwen3-4B tokenizer used for all token counts. |
| `outcome_probes/` | One JSON per probed repo, plus the probe index and saved upstream dataset cards. |
| `<dataset>/summary.json`, `metadata.json`, `README.upstream.md` | Per-dataset audit summary and the pinned upstream card. |

## Not ours — upstream snapshots kept for provenance

These three files were downloaded from the public OpenThoughts-Agent repository
(Apache-2.0, https://github.com/open-thoughts/OpenThoughts-Agent) and are kept only so the
label-recovery claim in `05_data_pool_audit.md` can be checked against the actual source.
They are third-party code, not part of this project's pipeline.

| File | Why it is here |
|---|---|
| `analyze_sft_export_gaps.py` | Shows that upstream success labels are read from `verifier_result.rewards` / `verifier_result.reward` in trial artifacts — i.e. the labels exist upstream but are absent from the released parquet. |
| `run_and_export_traces.py` | The export path that produced these datasets, including the success/failure filter. |
| `make_and_upload_trace_dataset.py` | The upload path, for the same reason. |

The corresponding upstream revision is recorded in `05_data_pool_audit.md`
(Git SHA `3bd1917e62c9d03d73063b433f5c442c279c0563`).

## Not committed

Raw `rows_*.json` / `records.json` / `recovery_candidates.json` dumps (~20 MB of public
dataset rows) are excluded by `.gitignore`; regenerate them with `python audit/audit_pool.py`.
