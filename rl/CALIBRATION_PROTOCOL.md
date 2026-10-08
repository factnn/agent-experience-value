# Development calibration, 2026-10-08

Written before sampling. This is harness/task-pool calibration, not an allocation-policy experiment.

Fixed untrained Qwen3-4B; no SFT adapter and no optimizer steps. Same TRL environment, binary state reward, thinking enabled, temperature 1, top-p 1, 2,048 retained completion-token limit (including feedback), eight tool rounds and 12 executable tool calls per episode. Record all generated tokens, including discarded suffixes.

Six development tasks: seeds 91010/91011 × send/new_contact/replace. Four trajectories per task. These instances differ from the previous engineering seeds 91000–91003 but are still development data, not held-out transfer tests. Both conditions use identical underlying tasks and per-task random seeds.

- `original`: previous system prompt.
- `protocol`: same prompt plus explicit instructions that names are not IDs, dependent calls must wait for returned values, guessed IDs/placeholders must not be used, successful sends should not repeat, and reasoning should stay brief. Exact text is in `pipeline/calibrate_rl.py`. It provides tool-use rules, not task-specific IDs or answers.

Use two idle GPUs, one per condition; 30-minute process timeout per condition. Each condition has 24 episodes. Save/push partial results at completed-group milestones and final audited summaries. No extra model/data download or paid service.

Screen for a usable initial pool: at most 6/24 trajectories reaching the retained-token cap; at least 3/6 tasks with both success and failure in their four samples, spanning at least two task types. These small-sample thresholds guide the next engineering step, not statistical evidence of learning or generalization. Do not start or claim a meaningful allocation comparison merely because this screen passes; training/held-out splits and the budget contract must still be frozen. If all tasks become easy or remain impossible, expand/calibrate task difficulty rather than manufacture rewards.

Readouts: task rewards, group variance, cap incidence, generated and discarded tokens, environment-error results, wall time, and a check that parameters stayed unchanged and retained trajectories match raw generations. A failed or timed-out condition remains visible; do not silently rerun/relabel it as successful.

## Adaptive horizon diagnostic (declared after the first two 2K groups)

The first 8 original-prompt episodes all reached the retained cap; the first 8 explicit-protocol episodes had 6 cap hits. Add a bounded **two-task** diagnostic using the protocol condition with a 4,096-token cap: send/new_contact, seed 91010, same per-task seeds and base policy. No parameter updates; no third full calibration arm is claimed. Use a third idle GPU with a 30-minute timeout. The full six-task readiness screen is **not applicable** to this subset (reported as null). Keep both planned 2K conditions unchanged and retain their failures. This adaptive diagnostic tests horizon plausibility and cannot establish transfer or allocation effectiveness.

## Complete the 4K development panel (adaptive extension)

After the full protocol/2K condition completed with 10/24 successes, 4 mixed groups and 14 cap hits, and the first protocol/4K task showed 0/4 cap hits with 2/4 successes, complete the remaining four tasks of the same six-task manifest at 4K. The two-task probe remains intact. Launch `--task-offset 2 --task-limit 4` on the now-free GPU 5 with a 30-minute timeout, while the probe finishes on GPU 6. At most three project GPUs remain active. Sampling seeds are based on the **canonical task index**, including the offset; initial model/adapter seed and all other settings are identical. The combined six-task 4K panel is an adaptive development result, not an originally prespecified third arm or held-out confirmation. Apply the same six-task readiness screen only after both parts finish and disjoint coverage/configuration are checked.
