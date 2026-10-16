"""Single-attempt capacity measurement using the existing owned-process supervisor."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from supervise_bfcl_stage_fork import ROOT, read, write, supervise


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True); parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--publication', type=Path, required=True); parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args(); assert digest(args.protocol) == args.protocol_sha256
    p = read(args.protocol); control = ROOT / p['control_root']; out = ROOT / p['output']
    assert p['automatic_retry'] is False and p['max_simultaneous_gpus'] == 1 and args.gpu in p['allowed_gpus']
    for name in [p['stop_file'], p['global_stop_file']]:
        assert not (ROOT / name).exists(), 'Current user stop marker: ' + name
    for name, sha in p['input_sha256'].items(): assert digest(ROOT / name) == sha, name
    publication = read(args.publication)
    assert publication['scientific_child_started'] is False and publication['commit'] == publication['remote_main']
    assert publication['files'][str(args.protocol.relative_to(ROOT))] == args.protocol_sha256
    for name, sha in publication['files'].items(): assert digest(ROOT / name) == sha, name
    cap = read(ROOT / p['resource_authorization']['path'])
    assert digest(ROOT / p['resource_authorization']['path']) == p['resource_authorization']['sha256']
    assert cap['max_simultaneous_gpus_for_subsequent_batches'] == 8
    info = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu), '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits'], text=True, timeout=15).strip().split(',')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True, timeout=15).splitlines()
    assert int(info[1].strip()) <= 64 and info[0].strip() not in apps, 'GPU occupied; no scientific child launched'
    assert not out.exists() and not (control / 'launch.json').exists(), 'Never repeat a previous capacity attempt'
    write(control / 'launch.json', {'supervisor_pid': os.getpid(), 'started_unix': time.time(), 'gpu': args.gpu,
        'protocol': {'path': str(args.protocol.relative_to(ROOT)), 'sha256': args.protocol_sha256},
        'publication': {'path': str(args.publication.relative_to(ROOT)), 'sha256': digest(args.publication)},
        'preflight': {'memory_mib': int(info[1].strip()), 'no_compute_process': True}, 'hard_seconds': p['hard_seconds'],
        'scope': 'One engineering group/update only; no target evaluation or automatic downstream launch'})
    command = [str(ROOT / '.venv-rl/bin/python'), '-u', str(ROOT / 'pipeline/run_qwen8_capacity_check.py'), '--protocol', str(args.protocol), '--protocol-sha256', args.protocol_sha256]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    result = supervise(command, out, control / 'capacity.console.log', p['hard_seconds'], env)
    write(control / 'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
