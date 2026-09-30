"""CPU-only pinned parquet audit and reproducible annotation samples. No trace execution."""
import collections,hashlib,json,pathlib,random,re,shlex
import pyarrow.parquet as pq
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'prepared';OUT.mkdir(exist_ok=True)
ERROR=re.compile(r'Traceback \(most recent call last\)|command not found|No such file or directory|\b[1-9]\d* failed\b|SyntaxError:|ModuleNotFoundError:')
DEC=json.JSONDecoder()
def sha(x):return hashlib.sha256(x.encode()).hexdigest()
def parse_action(content):
    # Ignore thought prose; accept a top-level commands object only. Never eval.
    if '</think>' in content:content=content.rsplit('</think>',1)[1]
    for hit in re.finditer(r'\{',content):
        try:obj,_=DEC.raw_decode(content[hit.start():])
        except ValueError:continue
        if isinstance(obj,dict) and isinstance(obj.get('commands'),list):
            commands=obj['commands']
            if all(isinstance(c,dict) and isinstance(c.get('keystrokes'),str) for c in commands):return obj
    return None

def operation(command):
    # Conservative first-command classification, not full shell AST or API identity.
    if not command.strip():return 'wait'
    try:t=shlex.split(command,comments=True)
    except ValueError:return 'unparsed_shell'
    if not t:return 'wait'
    while t and ('=' in t[0] and not t[0].startswith(('/', './'))):t=t[1:]
    if t and t[0]=='sudo':t=t[1:]
    if not t:return 'other'
    p=pathlib.PurePosixPath(t[0]).name
    if p in ['bash','sh','zsh'] and len(t)>=3 and t[1] in ['-c','-lc']:return operation(t[2])
    if p in ['pytest','unittest','ctest'] or (p.startswith('python') and t[1:3]==['-m','pytest']):return 'test'
    if p in ['pip','pip3','apt','apt-get','yum','conda','uv'] and 'install' in t:return 'install'
    if p in ['grep','rg','find','locate']:return 'search'
    if p in ['cat','head','tail','less','more'] and not re.search(r'(?<![<])>(?![>])|>>',command):return 'read'
    if p in ['sed','perl'] and any(s.startswith('-i') for s in t[1:]):return 'edit'
    if p in ['sed','awk']:return 'read'
    if p in ['cat','tee','touch','cp','mv','rm','mkdir','rmdir','patch','apply_patch']:return 'edit'
    if p in ['ls','pwd','cd','stat','file']:return 'inspect'
    if p=='git':return 'version_control'
    if p in ['curl','wget']:return 'network'
    if p.startswith('python') or p in ['node','java','ruby','go','cargo','make','gcc','g++','javac']:return 'execute'
    return 'other'

def extract_instruction(messages):
    s=next((m.get('content','') for m in messages if m.get('role')=='user'),'')
    if 'Task Description:\n' not in s:return None
    return s.split('Task Description:\n',1)[1].split('\nCurrent terminal state:',1)[0].strip()

def source_hint(task):
    for prefix in ['swesmith','superuser','tezos','issue']:
        if task.startswith(prefix+'-'):return prefix
    return 'unknown'

def feature(row,idx,shard):
    messages=row.get('conversations') or []
    task=row.get('task') or ''; family=re.sub(r'_copy\d+$','',task)
    ins=extract_instruction(messages)
    ops=[];failed=[];assistant=0;valid=0;self_complete=False
    for i,m in enumerate(messages):
        if m.get('role')!='assistant':continue
        assistant+=1;a=parse_action(m.get('content') or '')
        if a is None:failed.append(i);continue
        valid+=1;self_complete|=a.get('task_complete') is True
        ops.extend(operation(c['keystrokes']) for c in a['commands'])
    errors=[i for i,m in enumerate(messages) if i>1 and m.get('role') in ['user','tool'] and ERROR.search(m.get('content') or '')]
    return {'row_id':idx,'shard':shard,'task':task,'task_family_hint':family,'source_hint':source_hint(task),'teacher_raw':row.get('model'),'trial_name':row.get('trial_name'),'run_id':row.get('run_id'),'branch':row.get('trace_source'),'runtime_result':row.get('result'),'terminal_outcome':'unknown','instruction_sha256':sha(ins) if ins else None,'trajectory_sha256':sha(json.dumps(messages,sort_keys=True)),'assistant_turns':assistant,'parsed_action_turns':valid,'unparsed_turn_indices':failed,'operation_sequence':ops,'motifs':sorted(set(a+'>'+b for a,b in zip(ops,ops[1:]))),'error_candidate_turns':errors,'self_reported_complete':self_complete,'characters':sum(len(m.get('content') or '') for m in messages),'has_conversations':bool(messages)}

def reservoir_add(pool,item,count,rng,size):
    if len(pool)<size:pool.append(item)
    else:
        j=rng.randrange(count)
        if j<size:pool[j]=item

def main():
    files=sorted((ROOT/'data/SFT-100K').glob('*.parquet'))
    assert len(files)==10,'Wait for all pinned shards to finish downloading'
    counters={k:collections.Counter() for k in ['branch','runtime_result','source_hint','teacher_raw','operation','family','instruction_hash','trajectory_hash']}
    random_pool=[];error_pool=[];rng=random.Random(20260929);erng=random.Random(20260930)
    n=main_n=error_n=eligible_n=assistant=parsed=0;invalid=[];schema=[];seen_eligible=set()
    with (OUT/'features.jsonl').open('w') as out:
        for file in files:
            pf=pq.ParquetFile(file);schema.append({'file':file.name,'rows':pf.metadata.num_rows,'columns':pf.schema_arrow.names})
            for batch in pf.iter_batches(batch_size=64,use_threads=False):
                for row in batch.to_pylist():
                    f=feature(row,n,file.name);n+=1;assistant+=f['assistant_turns'];parsed+=f['parsed_action_turns']
                    for k in ['branch','runtime_result','source_hint','teacher_raw']:counters[k][str(f[k])]+=1
                    counters['operation'].update(f['operation_sequence'])
                    for dest,src in [('family','task_family_hint'),('instruction_hash','instruction_sha256'),('trajectory_hash','trajectory_sha256')]:
                        if f[src]:counters[dest][f[src]]+=1
                    out.write(json.dumps(f,ensure_ascii=False)+'\n')
                    if f['unparsed_turn_indices'] and len(invalid)<20:invalid.append({'features':f,'conversations':row['conversations']})
                    if f['branch']=='main' and f['has_conversations']:
                        main_n+=1
                        if f['trajectory_sha256'] in seen_eligible:continue
                        seen_eligible.add(f['trajectory_sha256']);eligible_n+=1
                        item={'features':f,'conversations':row['conversations'],'annotation':{'annotator':None,'error_origin':'unreviewed','recovery_status':'unreviewed','terminal_outcome':'unknown','evidence_turns':[]}}
                        reservoir_add(random_pool,item,eligible_n,rng,200)
                        if f['error_candidate_turns']:
                            error_n+=1;reservoir_add(error_pool,item,error_n,erng,100)
            print('scanned',file.name,'rows',n,flush=True)
    for name,items in [('annotation_random200',random_pool),('annotation_error100',error_pool),('parse_failures20',invalid)]:
        with (OUT/(name+'.jsonl')).open('w') as out:
            for item in items:out.write(json.dumps(item,ensure_ascii=False)+'\n')
    summary={'rows':n,'shards':schema,'counts':{k:dict(v) for k,v in counters.items() if k not in ['family','instruction_hash','trajectory_hash']},'unique':{k:len(counters[k]) for k in ['family','instruction_hash','trajectory_hash']},'duplicate_excess':{k:sum(v-1 for v in counters[k].values()) for k in ['family','instruction_hash','trajectory_hash']},'assistant_turns':assistant,'parsed_action_turns':parsed,'action_parse_rate':parsed/assistant,'main_rows':main_n,'unique_main_conversations':eligible_n,'error_candidate_unique_main':error_n,'annotation_random_n':len(random_pool),'annotation_error_n':len(error_pool),'annotation_overlap':len({x['features']['row_id'] for x in random_pool}&{x['features']['row_id'] for x in error_pool}),'sampling':'Uniform reservoir over unique main conversation hashes; error sample conditional on lexical candidates. Not family-uniform.','labels':'All terminal outcomes unknown; error matches are candidates, not verified recovery.','source_hint_caveat':'Task-name prefix only, not verified source lineage.'}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['shards','counts']},indent=2),flush=True)
if __name__=='__main__':main()
