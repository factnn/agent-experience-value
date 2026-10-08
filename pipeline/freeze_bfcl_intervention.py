"""Freeze the first finite common-base intervention matrix using metadata only."""
import hashlib
import json
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def order(row):return hashlib.sha256(('bfcl-value-wave1-20261008:'+row['id']).encode()).hexdigest()
def main():
    path=ROOT/'rl/BFCL_INTERVENTION_PROTOCOL_20261008.json'
    if path.exists():raise FileExistsError('Never overwrite a frozen protocol')
    manifest_path=ROOT/'rl/bfcl_research_split_candidate/manifest.json'
    m=json.loads(manifest_path.read_text());primary=['GorillaFileSystem','TradingBot']
    pools={c:[r['id'] for r in m['train'] if c in r['available_tool_classes']] for c in primary}
    combos=defaultdict(list)
    for r in m['train']:
        if any(r['id'] in pool for pool in pools.values()):combos['+'.join(r['available_tool_classes'])].append(r)
    probes=[min(rows,key=order)['id'] for _,rows in sorted(combos.items())]
    evaluation=[]
    for c in ['GorillaFileSystem','TradingBot','VehicleControlAPI','TravelAPI']:
        rows=sorted([r for r in m['id_eval'] if c in r['available_tool_classes']],key=order)
        evaluation.extend({'task_id':r['id'],'split':'id_eval','stratum':c} for r in rows[:2])
    transfer=defaultdict(list)
    for r in m['composition_transfer_eval']:transfer['+'.join(r['available_tool_classes'])].append(r)
    for c,rows in sorted(transfer.items()):
        evaluation.extend({'task_id':r['id'],'split':'composition_transfer_eval','stratum':c}
            for r in sorted(rows,key=order)[:2])
    assert len(evaluation)==16 and len({r['task_id'] for r in evaluation})==16
    protocol={'version':1,'purpose':'First exploratory common-base short RL/value intervention wave; broad scope retained',
        'manifest_sha256':digest(manifest_path),
        'data_sha256':digest(ROOT/'.third_party/bfcl_eval_pkg/bfcl_eval/data/BFCL_v4_multi_turn_base.json'),
        'base_revision':'1cfa9a7208912126459214e8b04321603b3df60c',
        'selection':'all training tasks containing the declared primary class; probes first task by fixed salted ID hash per combination; eval first two by same hash per declared stratum',
        'conditions':pools,'rules':['uniform','frontier','coverage'],
        'training_registry':{r['id']:'+'.join(r['available_tool_classes']) for r in m['train']},
        'record_sha256':{r['id']:r['record_sha256'] for s in ['train','id_eval','composition_transfer_eval'] for r in m[s]},
        'common_probe_task_ids':probes,'common_probe_updates':0,
        'common_state':'base-equivalent zero-initialized LoRA, initialized empty AdamW/constant scheduler, complete RNG and all charged probe history; optimizer global_step 0',
        'common_probe_soft_wall_seconds':3600,'common_probe_raw_token_ceiling':len(probes)*4*8192,
        'branch_raw_token_budget':32768,'branch_max_updates':12,'branch_soft_wall_seconds':3600,
        'branch_stop':'after complete optimizer update when new acquisition tokens reach budget; no within-group time/token censoring; max one group overshoot <=32768; wall/max-update stops separately reported',
        'eval_panel':evaluation,'eval_rollouts_per_task':1,'eval_soft_wall_seconds':3600,
        'eval_stop':'between tasks; incomplete panel never scored as full G/V; no censored trajectories counted as failures',
        'seeds':{'initialization':20261018,'common_probe':20261018,'branch':20261118,'eval':20261218},
        'sampling_seed_contract':'common seed+100*probe index; branch seed+10000*condition index+100*optimizer step, shared across rules; eval seed+100*panel index, shared across policies',
        'algorithm':{'algorithm':'official TRL 0.29 GRPO','thinking':True,'whole_completion_budget':8192,
            'segment_cap':4096,'max_calls':32,'max_segments':32,'num_generations':4,
            'temperature':1.0,'top_p':1.0,'top_k':0,'learning_rate':1e-5,'beta':0.0,
            'num_iterations':1,'scale_rewards':'group','loss_type':'grpo','lora_r':8,'lora_alpha':16,
            'lora_dropout':0,'lora_targets':['q_proj','v_proj'],'scheduler':'constant','optim':'adamw_torch'},
        'signals':{'unit':'visible available-tool-class combination','reward_window':16,'exploration':0.2,
            'uniform':'uniform tasks within the declared condition',
            'frontier':'p=(successes+1)/(observed episodes+2); task score=4p(1-p); 0.2 uniform-task exploration +0.8 normalized task scores',
            'coverage':'combo deficit=1/(1+acquired groups); 0.2 uniform-task exploration +0.8 normalized combo deficit divided uniformly among tasks in that combo',
            'unobserved':'Beta(1,1) prior and zero acquired groups, never free success-rate probes',
            'history':'only prior training interactions, including paid common probes; no gold paths or eval scores'},
        'metrics':{'G':'branch terminal-success fraction minus common-state fraction on identical frozen panel',
            'V':'branch fraction minus same-condition Uniform fraction on identical panel',
            'primary':'separate all-panel ID and withheld-combination macro means; equal-sized strata',
            'secondary':'primary-class-matched strata, task-level paired outcomes, exposure/cost, fresh gradients and parameter changes'},
        'cost_accounts':['shared common probe generation','branch new generation including all failed/capped output','evaluation generation','input token positions','generation seconds','non-generation training seconds','whole-group overshoot'],
        'repeat_count':1,'stage':'base only; later learner stages and other primary classes remain subsequent scope',
        'resource_contract':'common state one idle GPU; branches/evaluation at most three idle GPUs, total <=4',
        'inference_limits':['exploratory one training seed, sixteen sampled evaluation episodes per policy',
            'not official BFCL leaderboard; prior model exposure unknown; ID can share subtask/templates',
            'withheld visible-tool combinations only, not unseen tools or real environments',
            'two conditions x three rules cannot establish a universal signal-value predictor',
            'thinking chosen as a common operational config from paired development successes, not proved cost-optimal',
            'zero gradient or identical exposure is a measured null/weak intervention, never a reason for post-hoc task replacement']}
    path.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'protocol':str(path.relative_to(ROOT)),'sha256':digest(path),'pool_sizes':{c:len(x) for c,x in pools.items()},'probe_tasks':probes,'eval_panel':evaluation},ensure_ascii=False))
if __name__=='__main__':main()
