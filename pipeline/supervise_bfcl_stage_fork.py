"""Bounded, single-attempt execution of the frozen stage-fork acceptance.

Run this supervisor detached only after the dose batch has been reviewed.
It records abnormal exits even when the GPU runner cannot write its own failure.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'rl/BFCL_STAGE_FORK_ACCEPTANCE_PROTOCOL_20261011.json'
FROZEN_SHA = '5f60c862c82a1571ee2e95c11b84f03ec00fd11dea99da910c84201de8c97233'


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def process_has_not_exited(pid, proc_root=Path('/proc')):
    try:
        state = (proc_root / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[0]
    except FileNotFoundError:
        return False
    # A zombie has exited but its parent has not reaped it. It cannot execute work.
    return state not in {'Z', 'X'}


def supervise(command, out, log, hard_seconds, env=None, grace_seconds=30):
    started = time.monotonic()
    timed_out = False
    with log.open('x') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        write(log.with_suffix('.process.json'), {'pid': child.pid, 'pgid': child.pid,
              'started_unix': time.time(), 'hard_seconds': hard_seconds, 'command': command})
        try:
            code = child.wait(timeout=hard_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            # Kill the entire owned process group, including any surviving children.
            try:
                child.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            code = child.wait()
    try:
        summary = read(out / 'summary.json')
    except (OSError, ValueError):
        summary = {}
    complete = not timed_out and code == 0 and summary.get('status') == 'complete'
    receipt = {'returncode': code, 'hard_timeout': timed_out,
               'wall_seconds': time.monotonic() - started,
               'status': 'runner_complete_pending_replay_audit' if complete else 'failed_or_incomplete',
               'partial_evidence_retained': True, 'automatic_retry': False}
    if not complete:
        # A killed in-flight generation may have consumed tokens before journaling.
        # Preserve a lower bound, never present the journal as an exact total.
        tokens = rows = malformed = 0
        calls = out / 'generation_calls.jsonl'
        if calls.exists():
            for line in calls.read_text().splitlines():
                try:
                    row = json.loads(line)
                    tokens += len(row['generated_ids'])
                    rows += 1
                except (ValueError, KeyError, TypeError):
                    malformed += 1
        receipt.update(recorded_model_tokens_lower_bound=tokens, recorded_generation_calls=rows,
                       malformed_journal_rows=malformed, unrecorded_inflight_tokens_possible=True,
                       runner_summary_status=summary.get('status'))
        write(out / 'scheduler_failure.json', receipt)
    write(out / 'supervisor.json', receipt)
    return receipt


def preflight(gpu, acceptance):
    from codex_goal_status import goal_status
    if goal_status(os.environ['CODEX_HOME'], os.environ['CODEX_THREAD_ID']) != 'active':
        raise RuntimeError('Goal must be active')
    state = read(ROOT / 'rl/STATUS.json')['bfcl_intervention']
    if state['phase'] != 'audited_complete' or state['active'] or state['pending'] or state['failures']:
        raise RuntimeError('Dose repeats are not audited complete')
    if process_has_not_exited(state['scheduler_pid']):
        raise RuntimeError('Dose scheduler must have exited')
    source = state['path'] + '/audit_status.json'
    if read(ROOT / source).get('status') != 'passed':
        raise RuntimeError('Dose audit did not pass')
    events = [e for e in read(ROOT / 'experiments/index.json') if e['source'] == source]
    if not events or read(ROOT / 'experiments/reviews' / (events[-1]['id'] + '.json')).get('status') != 'reviewed':
        raise RuntimeError('Final dose audit must be manually reviewed first')
    if gpu not in acceptance['allowed_gpus']:
        raise ValueError('GPU outside frozen allowed set')
    info = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
           '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits'], text=True).strip().split(',')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                   '--format=csv,noheader'], text=True).splitlines()
    if int(info[1].strip()) > 64 or info[0].strip() in apps:
        raise RuntimeError('Selected GPU is occupied')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != FROZEN_SHA:
        raise RuntimeError('Frozen acceptance protocol changed')
    acceptance = read(PROTOCOL)
    out = ROOT / acceptance['output']
    if out.exists():
        raise FileExistsError('Never rerun or overwrite an observed attempt')
    preflight(args.gpu, acceptance)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive launch receipt also prevents two supervisors from launching together.
    with (out.parent / 'supervisor_launch.json').open('x') as stream:
        json.dump({'supervisor_pid': os.getpid(), 'gpu': args.gpu,
                   'started_unix': time.time(), 'protocol_sha256': FROZEN_SHA}, stream, indent=2)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    command = [str(ROOT / '.venv-rl/bin/python'), '-u',
               str(ROOT / 'pipeline/accept_bfcl_stage_fork.py'), '--protocol', str(PROTOCOL)]
    result = supervise(command, out, out.parent / 'replay.console.log', acceptance['hard_seconds'], env)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result['status'] == 'runner_complete_pending_replay_audit' else 1)


if __name__ == '__main__':
    main()
