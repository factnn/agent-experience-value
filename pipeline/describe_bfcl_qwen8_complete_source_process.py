"""Describe the completed native8B source cohort using original small training logs.

This is a CPU-only descriptive analysis, with no new sampling or fitting.
It neither reads target outcomes nor changes any frozen scientific input.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CERT = {"path": "rl/qwen3_8b_stage0_forward15_train_20261017/audit_status.json",
        "sha256": "ccd71adb544ddf6b78179bde87ff3946233dbd7328b4959727ecf32b1f470e25"}
EXECUTION = {"path": "rl/QWEN3_8B_STAGE0_FORWARD15_EXECUTION_20261017.json",
             "sha256": "e9626233b91e82a0fc8455cd2e745fec567e5356d086439a5f26d16286a613ee"}
COMMON_FP = "b6f2e35a120c675cb06286c597493035bd4f18aec3b3481113c4380b655828a8"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(root, name):
    path = root / name
    return {"path": name, "sha256": digest(path), "bytes": path.stat().st_size}


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(row) for row in path.read_text().splitlines() if row.strip()]


def describe(root):
    for required in (CERT, EXECUTION):
        if digest(root / required["path"]) != required["sha256"]:
            raise ValueError("Original completed cohort identity changed: " + required["path"])
    cert = read(root / CERT["path"])
    if cert["status"] != "passed" or cert["complete_jobs"] != 15:
        raise ValueError("This description requires the original complete source cohort")
    branches, all_steps, groups = [], [], {}
    for audit in cert["jobs"]:
        case = audit["case_id"]
        job_name = "rl/qwen3_8b_stage0_forward15_jobs_20261017/" + case + ".json"
        job = read(root / job_name)
        if digest(root / job_name) != audit["job_sha256"] or job["case_id"] != case:
            raise ValueError("Original source job substituted")
        base = root / job["output"]
        saved = read(base / "audit.json")
        review = read(base / "execution_agent_review.json")
        if (saved != audit or review["fresh_full_raw_replay_equals_saved_audit"] is not True
                or review["actual_cost"] != audit["cost"]):
            raise ValueError("Original completed audit or prior full raw review changed")
        reward_groups = lines(base / "groups.jsonl")
        updates = lines(base / "updates.jsonl")
        metrics = [row for row in lines(base / "metrics.jsonl") if "grad_norm" in row]
        if not len(reward_groups) == len(updates) == len(metrics) == audit["optimizer_steps"]:
            raise ValueError("Incomplete per-update logs; terminal runtime metric is not an update")
        rows, fresh, changed = [], [], []
        previous_fp = COMMON_FP
        for step, (group, update, metric) in enumerate(zip(reward_groups, updates, metrics), 1):
            rewards = group["rewards"]
            if (update["step"] != step or metric["step"] != step or group["policy_step"] != step - 1
                    or len(rewards) != 4 or len(set(group["task_ids"])) != 1
                    or any(value not in (0.0, 1.0) for value in rewards)
                    or update["before_sha256"] != previous_fp
                    or update["parameters_changed"] != (update["before_sha256"] != update["after_sha256"])):
                raise ValueError("Original per-step reward or parameter chain is inconsistent")
            previous_fp = update["after_sha256"]
            kind = ("all_success" if all(value == 1 for value in rewards) else
                    "all_zero_reward" if all(value == 0 for value in rewards) else "mixed_rewards")
            norm = metric["grad_norm"]
            if not math.isfinite(norm) or norm < 0:
                raise ValueError("Nonfinite or invalid recorded gradient")
            nonzero, prior_fresh = norm > 0, bool(fresh)
            if nonzero:
                fresh.append(step)
            if update["parameters_changed"]:
                changed.append(step)
            row = {"step": step, "task_id": group["task_ids"][0], "rewards": rewards,
                   "advantages": group["advantages"], "reward_kind": kind,
                   "fresh_nonzero_gradient": nonzero, "grad_norm": norm,
                   "parameters_changed": update["parameters_changed"],
                   "earlier_fresh_gradient_in_same_branch": prior_fresh,
                   "before_sha256": update["before_sha256"], "after_sha256": update["after_sha256"],
                   "model_tokens_in_loss": group["model_tokens_in_loss"],
                   "clipped_completion_ratio": metric["completions/clipped_ratio"]}
            rows.append(row)
            all_steps.append({"case": case, **row})
        if (fresh != audit["fresh_nonzero_gradient_steps"] or changed != audit["changed_parameter_steps"]
                or previous_fp != review["final_fingerprint"]
                or review["same_weights_as_native_common"] != (previous_fp == COMMON_FP)
                or sum(row["model_tokens_in_loss"] for row in rows) != audit["cost"]["new_model_tokens"]):
            raise ValueError("Description differs from completed raw audit or final endpoint")
        counts = dict(Counter(row["reward_kind"] for row in rows))
        label = job["source_group"] if job["role"] == "source_group" else "Uniform"
        branch = {"case": case, "source_group": label, "training_repeat": job["training_repeat"],
                  "original_job": ref(root, job_name), "prior_complete_audit": ref(root, job["output"] + "/audit.json"),
                  "prior_full_raw_review": ref(root, job["output"] + "/execution_agent_review.json"),
                  "raw_small_inputs": {name: ref(root, job["output"] + "/" + name)
                                       for name in ("groups.jsonl", "updates.jsonl", "metrics.jsonl")},
                  "new_model_tokens": audit["cost"]["new_model_tokens"], "optimizer_steps": len(rows),
                  "reward_group_counts": counts, "fresh_nonzero_gradient_steps": fresh,
                  "changed_parameter_steps": changed, "zero_fresh_gradient_branch": not fresh,
                  "same_weights_as_common": previous_fp == COMMON_FP, "final_fingerprint": previous_fp,
                  "steps": rows}
        branches.append(branch)
        aggregate = groups.setdefault(label, {"branches": 0, "reward_group_counts": Counter(),
                                             "optimizer_steps": 0, "fresh_nonzero_gradient_steps": 0,
                                             "parameter_changed_steps": 0, "new_model_tokens": 0,
                                             "zero_fresh_gradient_branches": 0})
        aggregate["branches"] += 1
        aggregate["reward_group_counts"].update(counts)
        aggregate["optimizer_steps"] += len(rows)
        aggregate["fresh_nonzero_gradient_steps"] += len(fresh)
        aggregate["parameter_changed_steps"] += len(changed)
        aggregate["new_model_tokens"] += branch["new_model_tokens"]
        aggregate["zero_fresh_gradient_branches"] += not fresh
    reward_kinds, transitions = {}, {}
    mismatches = []
    for row in all_steps:
        entry = reward_kinds.setdefault(row["reward_kind"], {"groups": 0, "new_model_tokens": 0})
        entry["groups"] += 1
        entry["new_model_tokens"] += row["model_tokens_in_loss"]
        key = ("fresh" if row["fresh_nonzero_gradient"] else "zero_fresh") + (
            "_changed" if row["parameters_changed"] else "_unchanged")
        entry = transitions.setdefault(key, {"steps": 0, "new_model_tokens": 0})
        entry["steps"] += 1
        entry["new_model_tokens"] += row["model_tokens_in_loss"]
        if (row["reward_kind"] == "mixed_rewards") != row["fresh_nonzero_gradient"]:
            mismatches.append({"case": row["case"], "step": row["step"]})
    for value in groups.values():
        value["reward_group_counts"] = dict(value["reward_group_counts"])
    changed_without_fresh = [row for row in all_steps
                             if row["parameters_changed"] and not row["fresh_nonzero_gradient"]]
    return {
        "kind": "complete_native8b_source_reward_gradient_parameter_description",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis_class": "Post-training descriptive process analysis; not a new predictor or confirmatory rule.",
        "implementation": ref(root, "pipeline/describe_bfcl_qwen8_complete_source_process.py"),
        "complete_training_audit": CERT, "execution_protocol": EXECUTION,
        "complete_original_branches": len(branches), "optimizer_steps": len(all_steps),
        "new_model_tokens": sum(branch["new_model_tokens"] for branch in branches),
        "cost_scope": "Original15 source training only, already charged in E388. No ancestor/probe/evaluation cost added or recharged.",
        "reward_groups": reward_kinds, "gradient_parameter_transitions": transitions,
        "changed_without_fresh_after_earlier_fresh": sum(
            row["earlier_fresh_gradient_in_same_branch"] for row in changed_without_fresh),
        "mixed_reward_nonzero_gradient_mismatches": mismatches,
        "groups_with_nonzero_recorded_completion_clipping_ratio": sum(
            row["clipped_completion_ratio"] > 0 for row in all_steps),
        "zero_fresh_gradient_branches": [branch["case"] for branch in branches
                                         if branch["zero_fresh_gradient_branch"]],
        "zero_fresh_gradient_branch_model_tokens": sum(
            branch["new_model_tokens"] for branch in branches if branch["zero_fresh_gradient_branch"]),
        "distinct_training_weight_endpoints": len({branch["final_fingerprint"] for branch in branches}),
        "by_source_group": groups, "branches": branches,
        "interpretation_limits": [
            "Observed mixed-reward groups and nonzero fresh gradients coincide in this frozen cohort. This is a trace description, not a universal equivalence or a new pre-acquisition signal.",
            "All20 parameter changes without a fresh gradient occur after an earlier fresh gradient in the same initially cold branch. This is consistent with within-branch AdamW state, without establishing a numerical counterfactual cause.",
            "Five branches have no fresh gradient and retain common weights. Four Math branches have all-success groups; Uniform repeat1 also contains two all-zero-reward groups. Zero gradient alone therefore does not establish an ability ceiling.",
            "No recorded completion-clipping ratio is nonzero. A zero reward is still not by itself a proof of the reason for failure or a general ability limit.",
            "113 sequential groups are dependent within15 branches, three repeats, source/tool identities and a shared ancestor. Do not use113 as independent evidence for a transfer relationship.",
            "Realized source-process statistics must not revise the176 frozen prospective forecasts. Complete original32 target panels remain necessary for G, V and prediction errors.",
            "Reward, update opportunity, parameter changes and transfer benefit are distinct. No benefit-per-token correction or positive source-value claim is inferred here."
        ],
        "target_outcomes_read": False, "new_scientific_tokens_from_analysis": 0,
        "new_sampling_or_refitting": False, "frozen_runtime_modified": False,
        "reserved21_accessed": False, "goal_complete": False
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = describe(ROOT)
    with args.output.open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output),
                      "branches": value["complete_original_branches"], "steps": value["optimizer_steps"],
                      "new_model_tokens": value["new_model_tokens"], "reward_groups": value["reward_groups"],
                      "gradient_parameter_transitions": value["gradient_parameter_transitions"],
                      "changed_without_fresh_after_earlier_fresh": value["changed_without_fresh_after_earlier_fresh"],
                      "mixed_reward_gradient_mismatches": value["mixed_reward_nonzero_gradient_mismatches"]}), flush=True)


if __name__ == "__main__":
    main()
