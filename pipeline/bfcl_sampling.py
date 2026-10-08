"""Local Qwen sampling shared by BFCL GRPO and fixed-policy calibration."""
import copy
import time
import torch
from bfcl_token_rollout import generation_buckets


def sample_episodes(model,tokenizer,episodes,generation_config,record,
                    segment_cap=1024,should_stop=None):
    device=next(model.parameters()).device
    generation_seconds=0.;sampled_tokens=0
    while any(e.active for e in episodes):
        for allowance,active in generation_buckets(episodes,segment_cap):
            reason=should_stop() if should_stop else None
            if reason:
                for episode in episodes:
                    if episode.active:episode.stop_reason=reason
                return {'sampled_tokens':sampled_tokens,'generation_seconds':generation_seconds,'censored':True}
            sequences=[e.input_ids for e in active];width=max(map(len,sequences))
            ids=torch.tensor([[tokenizer.pad_token_id]*(width-len(s))+s for s in sequences],device=device)
            attention=torch.tensor([[0]*(width-len(s))+[1]*len(s) for s in sequences],device=device)
            config=copy.deepcopy(generation_config);config.max_new_tokens=allowance
            if device.type=='cuda':torch.cuda.synchronize(device)
            before=time.perf_counter()
            output=model.generate(input_ids=ids,attention_mask=attention,
                generation_config=config,disable_compile=True)[:,width:].tolist()
            if device.type=='cuda':torch.cuda.synchronize(device)
            generation_seconds+=time.perf_counter()-before
            for episode,generated in zip(active,output):
                if tokenizer.eos_token_id in generated:generated=generated[:generated.index(tokenizer.eos_token_id)+1]
                sampled_tokens+=len(generated)
                record({'rollout_index':next(i for i,e in enumerate(episodes) if e is episode),
                    'task_id':episode.conversation.task['id'],'input_ids':episode.input_ids,
                    'generated_ids':generated,'generation_allowance':allowance,
                    'remaining_completion_budget':episode.budget-len(episode.completion_ids)})
                episode.accept(generated)
    return {'sampled_tokens':sampled_tokens,'generation_seconds':generation_seconds,'censored':False}
