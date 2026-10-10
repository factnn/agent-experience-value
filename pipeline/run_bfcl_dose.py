"""One frozen Uniform trajectory with development-only dose measurements."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from datetime import datetime, timezone
import run_bfcl_intervention as r
from bfcl_dose_checkpoint import checked_snapshot, crossed_thresholds

PROTOCOL = r.ROOT/'rl/BFCL_DOSE_PROTOCOL_20261010.json'


class DoseCheckpoint(r.TrainerCallback):
    def __init__(self, trainer):
        self.t = trainer
        self.previous_tokens = 0
        self.labels = []

    def on_step_end(self, args, state, control, **kwargs):
        t = self.t
        crossed = crossed_thresholds(t.protocol['dose_thresholds'],
            {x['threshold'] for x in self.labels}, self.previous_tokens, t.sampled_tokens)
        self.previous_tokens = t.sampled_tokens
        if not crossed: return control
        assert t._step % 4 == 0 and len(t.allocation_records) == state.global_step
        assert t.current_fingerprint == r.fingerprint(t.model)
        out = t.evidence_dir/'doses'/f'step_{state.global_step:03d}'
        out.mkdir(parents=True, exist_ok=False)
        progress = {'global_step':state.global_step, 'branch_local_optimizer_steps':state.global_step,
            'cumulative_learner_optimizer_steps':state.global_step,
            'micro_step':t._step, 'at_optimizer_boundary':True,
            'branch_sampled_tokens':t.sampled_tokens,
            'inherited_probe_tokens':t.inherited_tokens,
            'fresh_rollout_on_next_group':True, 'trainer_resume_supported':False}
        manifest, verification = checked_snapshot(out/'components', t.model,
            t.optimizer.optimizer, t.lr_scheduler, t.allocator, progress, r.provenance(t.protocol),
            save_adapter=lambda:t.model.save_pretrained(out/'adapter'))
        summary = {'status':'complete', 'policy_fingerprint':t.current_fingerprint,
            'optimizer_steps':state.global_step, 'branch_sampled_tokens':t.sampled_tokens,
            'thresholds':crossed, 'manifest':manifest, 'verification':verification,
            'optimizer_backend':type(t.optimizer.optimizer).__qualname__,
            'optimizer_wrapper':type(t.optimizer).__qualname__}
        r.write(out/'summary.json', summary)
        for threshold in crossed:
            self.labels.append({'threshold':threshold, 'actual_tokens':t.sampled_tokens,
                'overshoot':t.sampled_tokens-threshold, 'optimizer_steps':state.global_step,
                'policy':str(out.relative_to(r.ROOT)), 'policy_fingerprint':t.current_fingerprint,
                'same_step_labels':crossed, 'component_roundtrip':'passed'})
        r.write(t.evidence_dir/'dose_checkpoints.json', self.labels)
        r.published_progress(t.evidence_dir, 'dose_checkpoint',
            optimizer_steps=state.global_step, branch_sampled_tokens=t.sampled_tokens,
            checkpoint_labels=self.labels, stop_reason=t.stop_reason)
        return control


def train(p, out, tokenizer, model, args, started):
    t = r.make_trainer(p, out, tokenizer, model)
    t.condition = 'AllTasks_dose'; t.rule = 'uniform'; t.condition_index = 0; t.started = started
    t.add_callback(r.BranchState(t, args.common))
    t.add_callback(r.UpdateEvidence(t))
    checkpoint = DoseCheckpoint(t); t.add_callback(checkpoint)
    r.write(out/'dose_checkpoints.json', [])
    t.train()
    assert t._step % 4 == 0 and all(p.grad is None for p in t.model.parameters())
    r.write(out/'summary.json', {'status':'complete', 'stop_reason':t.stop_reason,
        'optimizer_steps':t.state.global_step, 'branch_sampled_tokens':t.sampled_tokens,
        'inherited_probe_tokens':t.inherited_tokens, 'dose_checkpoints':checkpoint.labels,
        'all_thresholds_reached':len(checkpoint.labels)==len(p['dose_thresholds']),
        'generation_seconds':t.generation_seconds,
        'non_generation_training_seconds':t.train_compute_seconds,
        'checkpoint_verification_seconds':sum(json.loads((r.ROOT/x/'summary.json').read_text())[
            'verification']['wall_seconds'] for x in {d['policy'] for d in checkpoint.labels}),
        'wall_seconds':time.perf_counter()-started, 'final_fingerprint':r.fingerprint(t.model)})
    r.published_progress(out, 'dose_training_complete', optimizer_steps=t.state.global_step,
        branch_sampled_tokens=t.sampled_tokens, checkpoint_labels=checkpoint.labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['train','eval'], required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--protocol', type=Path, default=PROTOCOL)
    ap.add_argument('--common', type=Path); ap.add_argument('--policy', type=Path)
    args = ap.parse_args(); r.PROTOCOL = args.protocol.resolve()
    started = time.perf_counter(); p = json.loads(r.PROTOCOL.read_text())
    out = r.ROOT/args.out; out.mkdir(parents=True, exist_ok=False)
    assert r.digest(r.ROOT/'rl/bfcl_research_split_candidate/manifest.json')==p['manifest_sha256']
    assert r.digest(r.PKG/'bfcl_eval/data/BFCL_v4_multi_turn_base.json')==p['data_sha256']
    assert all(x['split']=='development' for x in p['eval_panel'])
    assert r.torch.cuda.is_available() and r.torch.cuda.device_count()==1
    assert args.common if args.mode=='train' else args.policy
    names = ['run_bfcl_dose.py','bfcl_dose_checkpoint.py','run_bfcl_intervention.py',
        'bfcl_allocation.py','rl_learning_state.py','accept_bfcl_grpo.py','bfcl_sampling.py',
        'bfcl_token_rollout.py','bfcl_conversation.py','bfcl_safe_runtime.py','train_rl_smoke.py']
    r.write(out/'run.json', {'mode':args.mode,'pid':os.getpid(),
        'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'started_at_utc':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':r.digest(r.PROTOCOL),'provenance':r.provenance(p),
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r.ROOT,text=True).strip(),
        'source_sha256':{n:r.digest(r.ROOT/'pipeline'/n) for n in names}})
    r.set_seed(p['seeds']['initialization']); r.torch.set_num_threads(4)
    tokenizer = r.AutoTokenizer.from_pretrained(r.ROOT/'smoke/model',local_files_only=True)
    tokenizer.padding_side = 'left'
    model = r.AutoModelForCausalLM.from_pretrained(r.ROOT/'smoke/model',dtype=r.torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True)
    try:
        if args.mode=='train': train(p,out,tokenizer,model,args,started)
        else: r.evaluate(p,out,tokenizer,model,args,started)
    except Exception as error:
        r.write(out/'failure.json',{'type':type(error).__name__,'message':str(error),
            'wall_seconds':time.perf_counter()-started,'partial_generation_evidence_retained':True})
        raise


if __name__=='__main__': main()
