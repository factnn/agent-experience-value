"""Review newly completed original native8B transfer panels without generation.

Reuse the frozen raw auditor. Preserve original saved audits and large raw files.
Only create new execution-review receipts; never start or retry science.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from bfcl_source_eval_contract import pinned
from report_bfcl_qwen8_source_transfer import audit_job
from supervise_bfcl_qwen8_source_transfer import load_manifest

ROOT = Path(__file__).resolve().parents[1]
DISPATCH = {"path": "rl/qwen3_8b_stage0_forward32_binding_20261017/target_controller_dispatch.json",
            "sha256": "6d56d873badf1726c3e4f7713deead43ac48ecf2a839f7c604236684e0ac8786"}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def reference(root, path):
    return {"path": str(path.relative_to(root)), "sha256": digest(path), "bytes": path.stat().st_size}


def exclusive(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def process_state(pid):
    try:
        return Path("/proc/" + str(pid) + "/stat").read_text().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        return None


def review(root):
    dispatch = pinned(root, DISPATCH)
    manifest, master, binding, jobs = load_manifest(root, dispatch["manifest"])
    authority = pinned(root, dispatch["authority"])
    control = root / authority["control_root"]
    launch = read(control / "launch.json")
    if (launch["supervisor_pid"] != dispatch["pid"] or launch["manifest"] != dispatch["manifest"]
            or launch["authority"] != dispatch["authority"] or launch["publication"] != dispatch["publication"]):
        raise ValueError("Original controller identity changed")
    queue = read(control / "queue.json")
    entries = {entry["name"]: (entry, job) for entry, job in zip(manifest["jobs"], jobs)}
    reviewed, newly_reviewed, publication_paths = [], [], []
    for entry in manifest["jobs"]:
        name = entry["name"]
        if name not in queue["complete"]:
            continue
        _, job = entries[name]
        output = root / job["output"]
        marker = output / "execution_agent_review.json"
        if marker.exists():
            old = read(marker)
            if (old["original_job"] != entry or old["manifest"] != dispatch["manifest"]
                    or old["fresh_full_raw_audit_equals_saved"] is not True):
                raise ValueError("Different existing execution review")
            for item in old["original_raw_files"].values():
                if digest(root / item["path"]) != item["sha256"]:
                    raise ValueError("Previously reviewed original bytes changed")
            reviewed.append(old)
            continue
        started = time.monotonic()
        fresh = audit_job(root, root / entry["path"])
        saved = read(output / "audit.json")
        if fresh != saved or fresh["status"] != "passed":
            raise ValueError("Fresh full raw audit differs from original saved audit: " + name)
        run = read(output / "run.json")
        summary = read(output / "summary.json")
        terminal = read(output / "supervisor.json")
        process_path = control / (name + ".console.process.json")
        process = read(process_path)
        state = process_state(process["pid"])
        command = process["command"]
        if (state not in (None, "Z", "X") or process["pid"] != run["pid"]
                or terminal["returncode"] != 0 or terminal["hard_timeout"]
                or terminal["wall_seconds"] > manifest["hard_seconds_per_panel"] + 30
                or process["hard_seconds"] != manifest["hard_seconds_per_panel"]
                or command[command.index("--job") + 1] != str(root / entry["path"])
                or command[command.index("--job-sha256") + 1] != entry["sha256"]
                or run["current_eval_authority"] != dispatch["authority"]
                or run["execution_publication"] != dispatch["publication"]
                or summary["status"] != "complete" or summary["completed_tasks"] != 32
                or summary["optimizer_updates"] != 0 or not summary["weights_unchanged"]):
            raise ValueError("Original scientific job or clean termination not established")
        audit_console = control / (name + ".audit.console.log")
        original_audit_messages = []
        for line in audit_console.read_text().splitlines():
            if line.lstrip().startswith("{"):
                try:
                    original_audit_messages.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if saved not in original_audit_messages:
            raise ValueError("Original CPU audit segment log differs from saved result")
        raw = {}
        for filename, expected in fresh["source_files"].items():
            path = output / filename
            item = reference(root, path)
            if item["sha256"] != expected:
                raise ValueError("Original audited raw bytes changed")
            item["retrieve"] = str(path)
            item["publication"] = ("on_server_original_with_git_sha_and_retrieval"
                                   if filename in ("generation_calls.jsonl", "rollouts.jsonl") else "git_evidence")
            raw[filename] = item
        controls = {path.name: reference(root, path)
                    for path in sorted(control.glob(name + ".*")) if path.is_file()}
        value = {
            "kind": "complete_original_native8b_transfer_panel_execution_agent_review",
            "status": "reviewed", "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "reviewer": "Current SSH execution agent; not independent human review.",
            "review_implementation": reference(root, root / "pipeline/review_bfcl_qwen8_completed_target_panels.py"),
            "original_job": entry, "manifest": dispatch["manifest"], "authority": dispatch["authority"],
            "actual_pre_target_publication": dispatch["publication"],
            "fresh_full_raw_audit_equals_saved": True, "fresh_raw_audit": fresh,
            "raw_audit_implementation": reference(root, root / "pipeline/report_bfcl_qwen8_source_transfer.py"),
            "original_process": process, "original_process_state_at_review": state,
            "original_scientific_process_terminal": True, "original_scientific_terminal": terminal,
            "original_cpu_audit_console_matches_saved": True,
            "original_cpu_audit_pid_recorded": False,
            "cpu_audit_segment_evidence_limit": "Existing supervisor retained its result and console but no separate auditor PID receipt. Do not invent one.",
            "original_control_files": controls, "original_raw_files": raw,
            "actual_cost": fresh["cost"], "actual_supervised_worker_wall_seconds": terminal["wall_seconds"],
            "fresh_replay_wall_seconds": time.monotonic() - started,
            "scientific_reruns": 0, "new_scientific_tokens_from_review": 0,
            "scope": "One complete original32-task panel; full paired G/V and frozen prediction errors require all32 registered panels.",
            "reserved21_accessed": False, "goal_complete": False
        }
        exclusive(marker, value)
        reviewed.append(value)
        newly_reviewed.append(name)
        publication_paths.extend([str(marker.relative_to(root)), str(audit_console.relative_to(root)),
                                  str(process_path.relative_to(root))])
        preflight = control / (name + ".preflight.json")
        if preflight.exists():
            publication_paths.append(str(preflight.relative_to(root)))
        publication_paths.extend(item["path"] for item in raw.values() if item["publication"] == "git_evidence")
        print(json.dumps({"reviewed_original_panel": name, "new_model_tokens": fresh["cost"]["new_model_tokens"],
                          "fresh_raw_replay_matches": True, "original_pid": process["pid"],
                          "worker_wall_seconds": terminal["wall_seconds"]}), flush=True)
    aliases = defaultdict(list)
    for value in reviewed:
        audit = value["fresh_raw_audit"]
        aliases[(audit["policy_fingerprint"], audit["evaluation_repeat"])].append(value)
    alias_checks = []
    for (fingerprint, repeat), rows in aliases.items():
        if len(rows) < 2:
            continue
        alias_checks.append({
            "policy_fingerprint": fingerprint, "evaluation_repeat": repeat,
            "original_panels": [value["original_job"]["name"] for value in rows],
            "unique_generation_journal_sha256": len({value["original_raw_files"]["generation_calls.jsonl"]["sha256"] for value in rows}),
            "unique_outcomes_sha256": len({value["original_raw_files"]["outcomes.jsonl"]["sha256"] for value in rows}),
            "unique_rollout_journal_sha256": len({value["original_raw_files"]["rollouts.jsonl"]["sha256"] for value in rows}),
            "actual_inference_performed_for_every_original_panel": True,
            "independent_policy_conditions": False
        })
    current_queue = read(control / "queue.json")
    return {
        "kind": "completed_original_native8b_target_panel_execution_review_batch",
        "reviewed_at": datetime.now(timezone.utc).isoformat(), "manifest": dispatch["manifest"],
        "authority": dispatch["authority"], "publication": dispatch["publication"],
        "controller_pid": dispatch["pid"], "controller_state": process_state(dispatch["pid"]),
        "eligible_completed_panels_at_start": len(queue["complete"]), "newly_reviewed": newly_reviewed,
        "all_reviewed_original_panels": [value["original_job"]["name"] for value in reviewed],
        "panel_review_refs": [reference(root, root / entries[value["original_job"]["name"]][1]["output"] / "execution_agent_review.json") for value in reviewed],
        "completed_panel_cost_charged_once": {
            "new_model_tokens": sum(value["actual_cost"]["new_model_tokens"] for value in reviewed),
            "input_token_positions_excluding_padding": sum(value["actual_cost"]["input_token_positions_excluding_padding"] for value in reviewed),
            "external_tokens": sum(value["actual_cost"]["external_tokens"] for value in reviewed),
            "rollouts": sum(value["actual_cost"]["rollouts"] for value in reviewed),
            "supervised_worker_wall_seconds": sum(value["actual_supervised_worker_wall_seconds"] for value in reviewed)
        },
        "same_weight_same_seed_original_panel_checks": alias_checks,
        "actual_queue_counts": {key: len(value) for key, value in current_queue.items()},
        "publication_paths": list(dict.fromkeys(publication_paths)),
        "large_original_raw_retention": "Original generation_calls and rollouts retained on server; every completed panel review publishes exact byteSHA, size and absolute retrieval path. No inference reruns for publication.",
        "scientific_reruns": 0, "new_scientific_tokens_from_review": 0,
        "complete_G_or_V_available": False, "frozen_forecasts_modified": False,
        "reserved21_accessed": False, "goal_complete": False
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior batch receipt; inspect and use the next separately named batch")
    value = review(ROOT)
    exclusive(args.output, value)
    print(json.dumps({"output": str(args.output), "newly_reviewed": value["newly_reviewed"],
                      "cost": value["completed_panel_cost_charged_once"],
                      "same_weight_same_seed_checks": value["same_weight_same_seed_original_panel_checks"],
                      "actual_queue_counts": value["actual_queue_counts"]}), flush=True)


if __name__ == "__main__":
    main()
