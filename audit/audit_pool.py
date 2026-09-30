"""Read-only public HF audit. Cached viewer windows are exploratory, not IID samples.
Run: python audit_pool.py ; outputs live beside this script. Never executes traces.
"""
import collections, concurrent.futures, datetime, hashlib, json, pathlib, re
import requests
ROOT = pathlib.Path(__file__).resolve().parent
DATASETS = ['OpenThoughts-Agent-SFT-100K', 'AgentTrove', 'OpenThoughts-Agent-v1-SFT', 'OpenThoughts-Agent-SFT-ColdStartForRL-10K']

def get(url, params=None):
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r

def missing(x):
    return x is None or (isinstance(x, str) and x.strip().lower() in ['', 'none', 'null', 'nan'])

def run(name):
    d = ROOT / name
    d.mkdir(exist_ok=True)
    repo = 'open-thoughts/' + name
    meta = get('https://huggingface.co/api/datasets/' + repo).json()
    (d/'metadata.json').write_text(json.dumps(meta, indent=2))
    (d/'README.upstream.md').write_text(get('https://huggingface.co/datasets/'+repo+'/raw/'+meta['sha']+'/README.md').text)
    params = {'dataset':repo, 'config':'default', 'split':'train'}
    first = get('https://datasets-server.huggingface.co/rows', {**params, 'offset':0, 'length':1}).json()
    total = first['num_rows_total']
    offsets = sorted(set(round((total-10)*i/9) for i in range(10)))
    rows, windows = [], []
    for offset in offsets:
        path = d/f'rows_{offset}.json'
        if path.exists():
            payload = json.loads(path.read_text())
        else:
            r = get('https://datasets-server.huggingface.co/rows', {**params, 'offset':offset, 'length':10})
            payload = r.json(); path.write_text(r.text)
        windows.append({'offset':offset, 'returned':len(payload.get('rows',[])), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        rows.extend(payload.get('rows',[]))
    cols = [f['name'] for f in first['features']]
    values = {k: collections.Counter() for k in ['original_source','original_teacher','trace_source','model','agent','result','verifier_output','judgment','reward']}
    nonnull = collections.Counter(); roles = collections.Counter(); horizons=[]; lengths=[]; fingerprints=collections.Counter(); examples=[]; records=[]
    for wrapper in rows:
        row = wrapper['row']; messages = row.get('conversations', row.get('messages', [])) or []
        for k,v in row.items():
            if not missing(v): nonnull[k]+=1
        for k in values:
            if not missing(row.get(k)): values[k][str(row[k])[:300]]+=1
        roles.update(m.get('role','?') for m in messages)
        horizon=sum(m.get('role')=='assistant' for m in messages); horizons.append(horizon)
        lengths.append(sum(len(m.get('content') or '') for m in messages))
        # Feedback-only lexical candidates: never label them verified recovery.
        error_indices=[i for i,m in enumerate(messages) if i>1 and m.get('role') in ['user','tool'] and re.search(r'Traceback \(most recent call last\)|command not found|No such file or directory|\b[1-9]\d* failed\b|SyntaxError:|ModuleNotFoundError:',m.get('content') or '')]
        later_action=any(any(m.get('role')=='assistant' for m in messages[i+1:]) for i in error_indices)
        text=json.dumps(messages,sort_keys=True); fp=hashlib.sha256(text.encode()).hexdigest(); fingerprints[fp]+=1
        rec={'row_idx':wrapper['row_idx'],'task':row.get('task'),'trial_name':row.get('trial_name'),'horizon':horizon,'characters':lengths[-1],'error_feedback_candidate':bool(error_indices),'later_assistant':later_action,'truncated_cells':wrapper.get('truncated_cells',[]),'sha256':fp}
        records.append(rec)
        if error_indices and len(examples)<5:
            i=error_indices[0]; examples.append({'row_idx':wrapper['row_idx'],'task':row.get('task'),'result':row.get('result'),'feedback':messages[i]['content'][:1800],'next_turn':messages[i+1]['content'][:1800] if i+1<len(messages) else None})
    def dist(a):
        a=sorted(a);return {'min':a[0], 'median':a[len(a)//2], 'max':a[-1]}
    summary={'dataset':repo,'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'repo_sha':meta['sha'],'viewer_revision_caveat':'Viewer rows use default branch; metadata SHA recorded, no guarantee viewer cache matches it. Pin raw parquet before training.','total_rows_viewer':total,'sample_n':len(rows),'design':'10 evenly spaced windows x 10 consecutive rows; exploratory, not IID','windows':windows,'columns':cols,'nonnull':dict(nonnull),'values':{k:dict(v) for k,v in values.items()},'roles':dict(roles),'assistant_turns':dist(horizons),'characters_not_tokens':dist(lengths),'error_feedback_candidates':sum(r['error_feedback_candidate'] for r in records),'error_followed_by_assistant':sum(r['later_assistant'] for r in records),'duplicate_conversations_in_sample':sum(v-1 for v in fingerprints.values()),'truncated_rows':sum(bool(r['truncated_cells']) for r in records)}
    (d/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (d/'records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    (d/'recovery_candidates.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2))
    return summary

if __name__ == '__main__':
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(run,n):n for n in DATASETS}
        for f in concurrent.futures.as_completed(futures):
            try:
                s=f.result();results.append(s)
                print(json.dumps({k:s[k] for k in ['dataset','total_rows_viewer','sample_n','nonnull','values','assistant_turns','error_feedback_candidates','truncated_rows']},ensure_ascii=False),flush=True)
            except Exception as e: print('FAILED',futures[f],repr(e),flush=True)
    (ROOT/'summaries.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    if len(results)!=len(DATASETS):raise SystemExit(1)
