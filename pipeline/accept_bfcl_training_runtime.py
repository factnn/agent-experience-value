"""Training-pool oracle self-consistency; no model, ranking or held-out answers."""
import hashlib
import json
import time
from bfcl_safe_runtime import ROOT,PKG
from bfcl_conversation import BFCLConversation


def main():
    start=time.perf_counter();manifest_path=ROOT/'rl/bfcl_research_split_candidate/manifest.json'
    manifest=json.loads(manifest_path.read_text());ids={r['id'] for r in manifest['train']}
    data_path=PKG/'bfcl_eval/data/BFCL_v4_multi_turn_base.json'
    answers_path=PKG/'bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_base.json'
    tasks={r['id']:r for r in map(json.loads,data_path.read_text().splitlines()) if r['id'] in ids}
    answers={r['id']:r['ground_truth'] for r in map(json.loads,answers_path.read_text().splitlines()) if r['id'] in ids}
    results=[]
    for key in sorted(ids):
        try:
            c=BFCLConversation(tasks[key],answers[key])
            for calls in answers[key]:
                for call in calls:c.tool_call(call)
                c.finish_user_turn({'role':'assistant','content':'CPU reference-action acceptance.'})
            results.append({'task_id':key,'passed':c.reward()==1,'user_turns':len(c.grades)})
        except Exception as error:results.append({'task_id':key,'passed':False,'error':f'{type(error).__name__}: {error}'})
    report={'scope':'87 training cases only; no model generation or ID/transfer answer checks',
        'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'passed':all(r['passed'] for r in results),'cases':results,
        'user_turns':sum(r.get('user_turns',0) for r in results),
        'wall_seconds':time.perf_counter()-start,
        'oracle_features_for_allocator':False,
        'limitation':'Reference self-consistency verifies runtime support, not task-semantic validity or model learning. No successful-only task selection, split or reward changes.'}
    out=ROOT/'rl/bfcl_research_split_candidate/training_runtime_acceptance.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}))
    assert report['passed']


if __name__=='__main__':main()
