# prepared/ — what is committed and what is not

## Committed

| File | Size | Contents |
|---|---:|---|
| `summary.json` | 5 KB | Full-pool scan: 94,334 rows, branch/result/source/teacher counts, operation distribution, parse rate, unique hashes. |
| `manifest_summary.json` | 1 KB | Provisional grouped train/dev split counts and exclusion reasons. |
| `token_summary.json` | 165 B | Token-length summary for the 200 random main trajectories. |
| `token_lengths.json` | 47 KB | Per-row chat/official-template/assistant token counts for the 299 annotation rows. |
| `full_metadata.json` | 1 KB | Row counts per pinned shard and empty-`trace_source` count. |
| `artifact_hashes.json` | 512 B | SHA256 of the key prepared artifacts. |
| `reviewed_events.json` | 4 KB | Five event-level recovery annotations used to calibrate the rubric in `07_recovery_annotation_guide.md`. Assistant review, not human gold. |
| `review/index.html` + 299 pages | 18 MB | Offline, escaped HTML reader for the 299 sampled trajectories. Text only: no trajectory command is ever executed. |

## Not committed — regenerate locally

| File | Size | Regenerate with |
|---|---:|---|
| `features.jsonl` | 108 MB | `python pipeline/prepare_pool.py` |
| `candidate_manifest.jsonl` | 20 MB | `python pipeline/build_review.py` |
| `annotation_random200.jsonl` | 11 MB | `python pipeline/prepare_pool.py` |
| `annotation_error100.jsonl` | 5.7 MB | `python pipeline/prepare_pool.py` |
| `parse_failures20.jsonl` | 2.3 MB | `python pipeline/prepare_pool.py` |

Both scripts are CPU-only and need the pinned parquet shards in `data/`
(`python pipeline/download_pool.py`, ~1.7 GB, verified against HF LFS SHA256).

## Status of the labels

Every record in the candidate manifest carries `outcome=unknown` and `training_ready=false`.
No success label is available in the released data; see `05_data_pool_audit.md` and
`08_full_pool_preparation.md`. Error-keyword matches are candidates, not verified recoveries.
