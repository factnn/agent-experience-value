"""One frozen target panel with strict source and pre-prediction publication gates."""
import argparse
import os
from pathlib import Path
import time
import run_bfcl_intervention as r
from bfcl_source_eval_contract import load_job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--job-sha256', required=True)
    args = parser.parse_args()
    job, master, protocol, policy, seal = load_job(r.ROOT, args.job, args.job_sha256)
    out = r.ROOT / job['output']
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        assert r.torch.cuda.is_available() and r.torch.cuda.device_count() == 1
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
        r.torch.set_num_threads(4)
        r.torch.use_deterministic_algorithms(True, warn_only=False)
        r.torch.backends.cudnn.deterministic = True
        r.torch.backends.cudnn.benchmark = False
        r.set_seed(protocol['seeds']['initialization'])
        start_unix = time.time()
        assert seal is None or seal['sealed_unix'] <= start_unix
        r.write(out / 'run.json', {'job_sha256': args.job_sha256, 'policy_case': job['policy_case'],
            'role': job['role'], 'evaluation_repeat': job['evaluation_repeat'], 'pid': os.getpid(),
            'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'evaluation_started_unix': start_unix,
            'policy_fingerprint': job['policy_fingerprint'], 'prediction_seal': job.get('prediction_seal'),
            'optimizer_updates': 0, 'heldout_gate': 'verified' if seal else 'not_applicable'})
        tokenizer = r.AutoTokenizer.from_pretrained(r.ROOT / 'smoke/model', local_files_only=True)
        tokenizer.padding_side = 'left'
        model = r.AutoModelForCausalLM.from_pretrained(r.ROOT / 'smoke/model', dtype=r.torch.bfloat16,
                                                      attn_implementation='sdpa', local_files_only=True)
        model = r.PeftModel.from_pretrained(model, policy / 'adapter', is_trainable=True).to('cuda:0').eval()
        initial = r.fingerprint(model)
        assert initial == job['policy_fingerprint']
        from transformers import GenerationConfig
        a = master['evaluation']
        for key in ['temperature', 'top_p', 'top_k', 'thinking', 'whole_completion_budget', 'segment_cap', 'max_calls', 'max_segments']:
            assert a[key] == protocol['algorithm'][key]
        config = GenerationConfig(do_sample=True, temperature=a['temperature'], top_p=a['top_p'], top_k=a['top_k'],
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            bos_token_id=None, use_cache=True, max_new_tokens=a['segment_cap'])
        total, seconds, outcomes = 0, 0.0, []
        pairs = r.load_tasks(protocol, job['task_ids'], 'composition_transfer_eval')
        with r.torch.inference_mode():
            for index, (task, seed) in enumerate(zip(job['task_ids'], job['task_seeds'])):
                r.set_seed(seed)
                episodes = [r.episode(tokenizer, pairs[task], protocol)]
                def record(call):
                    r.append(out / 'generation_calls.jsonl', call | {'group': index, 'phase': 'evaluation', 'seed': seed})
                stats = r.sample_episodes(model, tokenizer, episodes, config, record, segment_cap=a['segment_cap'])
                total += stats['sampled_tokens']
                seconds += stats['generation_seconds']
                e = episodes[0]
                r.append(out / 'rollouts.jsonl', e.evidence() | {'group': index, 'rollout_index': 0,
                         'phase': 'evaluation', 'seed': seed, 'policy_fingerprint': initial})
                row = {'task_id': task, 'split': 'composition_transfer_eval', 'stratum': a['registry'][task],
                       'seed': seed, 'rewards': [e.conversation.reward()], 'sampled_tokens': stats['sampled_tokens'],
                       'stop_reasons': [e.stop_reason]}
                outcomes.append(row)
                r.append(out / 'outcomes.jsonl', row)
                r.published_progress(out, 'source_transfer_evaluation', completed_tasks=len(outcomes),
                                     planned_tasks=len(job['task_ids']), sampled_tokens=total)
        assert r.fingerprint(model) == initial and all(p.grad is None for p in model.parameters())
        assert len(outcomes) == 32 and total <= 32 * a['whole_completion_budget']
        r.write(out / 'summary.json', {'status': 'complete', 'policy_case': job['policy_case'], 'role': job['role'],
            'evaluation_repeat': job['evaluation_repeat'], 'policy_fingerprint': initial, 'sampled_tokens': total,
            'completed_tasks': len(outcomes), 'generation_seconds': seconds, 'wall_seconds': time.perf_counter()-started,
            'optimizer_updates': 0, 'weights_unchanged': True, 'scope': 'One registered transfer panel; G/V require the paired baseline and repeat aggregation.'})
    except Exception as error:
        r.write(out / 'failure.json', {'type': type(error).__name__, 'message': str(error),
            'wall_seconds': time.perf_counter()-started, 'partial_evidence_retained': True, 'automatic_retry': False})
        raise


if __name__ == '__main__':
    main()
