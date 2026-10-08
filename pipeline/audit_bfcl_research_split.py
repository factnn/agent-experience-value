"""Metadata-only BFCL research split; no model calls or answer inspection.

Candidate composition split, not an official BFCL train/test split. The gold
`path` field is never used as an allocator feature or partition criterion.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'.third_party/bfcl_eval_pkg/bfcl_eval/data'
HOLDOUT={tuple(sorted(p)) for p in [
    ['GorillaFileSystem','MathAPI'],['VehicleControlAPI','TwitterAPI'],
    ['TradingBot','MessageAPI'],['TravelAPI','TicketAPI']]}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def used_indices():
    seen=defaultdict(set)
    # Model result IDs from all historical local evaluation logs. Augmented
    # variants share a base index, so quarantine that index across categories.
    for path in (ROOT/'smoke/eval_bfcl').glob('*.jsonl'):
        for line in path.read_text().splitlines():
            row=json.loads(line);key=row.get('id','')
            match=re.fullmatch(r'multi_turn_(?:base|long_context|miss_func|miss_param)_(\d+)',key)
            if match:seen[int(match.group(1))].add(str(path.relative_to(ROOT)))
    return seen


def main():
    source=DATA/'BFCL_v4_multi_turn_base.json'
    data=[json.loads(line) for line in source.read_text().splitlines()]
    historical=used_indices();groups=defaultdict(list);quarantine=[];transfer=[]
    for row in data:
        index=int(row['id'].rsplit('_',1)[1]);tools=tuple(sorted(row['involved_classes']))
        record={'id':row['id'],'available_tool_classes':list(tools),
                'question_sha256':digest(row['question']),
                'initial_state_sha256':digest(row['initial_config']),
                'question_and_state_sha256':digest([row['question'],row['initial_config']]),
                'user_turns':len(row['question']),
                'record_sha256':digest(row)}
        if index in historical:
            record['historical_result_files']=sorted(historical[index]);quarantine.append(record)
        elif tools in HOLDOUT:transfer.append(record)
        else:groups[tools].append(record)
    partitions={'train':[],'development':[],'id_eval':[],'composition_transfer_eval':transfer,
                'historical_quarantine':quarantine}
    for tools,records in sorted(groups.items()):
        records.sort(key=lambda r:hashlib.sha256(('bfcl-research-20261008:'+r['id']).encode()).hexdigest())
        n=len(records);n_eval=max(1,n//5) if n>=3 else 0
        partitions['development'].extend(records[:n_eval])
        partitions['id_eval'].extend(records[n_eval:2*n_eval])
        partitions['train'].extend(records[2*n_eval:])
    all_records=[r for group in partitions.values() for r in group]
    assert len(all_records)==len(data) and len({r['id'] for r in all_records})==len(data)
    # Exact duplicate questions must never span partitions. Fail visibly instead
    # of treating task ID disjointness as sufficient; near duplicates remain unaudited.
    for field in ['question_sha256','question_and_state_sha256']:
        ownership=defaultdict(set)
        for name,records in partitions.items():
            for r in records:ownership[r[field]].add(name)
        assert all(len(v)==1 for v in ownership.values()), f'cross-partition exact duplicate: {field}'
    train_tools={t for r in partitions['train'] for t in r['available_tool_classes']}
    train_combos={tuple(r['available_tool_classes']) for r in partitions['train']}
    assert all(set(r['available_tool_classes'])<=train_tools for r in transfer)
    assert not {tuple(r['available_tool_classes']) for r in transfer}&train_combos
    out=ROOT/'rl/bfcl_research_split_candidate';out.mkdir(parents=True,exist_ok=True)
    (out/'manifest.json').write_text(json.dumps(partitions,indent=2)+'\n')
    result={'status':'metadata_audit_passed_runtime_not_yet_accepted','benchmark_package':'bfcl_eval==2026.3.23',
        'source_file':str(source.relative_to(ROOT)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'license_in_local_data_card':'apache-2.0','official_split':False,
        'split_design':'candidate research split with four withheld visible tool-class combinations',
        'heldout_combinations':[list(p) for p in sorted(HOLDOUT)],
        'partition_counts':{k:len(v) for k,v in partitions.items()},
        'training_tool_classes':sorted(train_tools),
        'tool_combination_counts':{name:{' + '.join(k):v for k,v in Counter(tuple(r['available_tool_classes']) for r in records).items()} for name,records in partitions.items()},
        'exact_question_overlap_check':'passed','exact_question_and_state_overlap_check':'passed',
        'semantic_template_overlap':'not yet audited',
        'model_prior_exposure':'unknown; fixed initial model comparison cannot certify pretraining exclusion',
        'gold_paths_or_answers_used_as_features':False,'model_sampling':False,
        'acceptance_remaining':['semantic overlap audit','reset/isolation and official reward oracle checks',
                                'multi-user-turn RL integration','checkpoint/optimizer/history fork snapshots',
                                'freeze branch budgets/repeats before sampling'],
        'interpretation':'Tests held-out combinations of already-seen simulated tool classes only. Not an official leaderboard score, unseen individual tools, or independent real-world environment transfer.'}
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
