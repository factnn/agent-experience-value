"""Build offline, escaped HTML reader and group-disjoint provisional manifests."""
import collections,hashlib,html,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1];P=ROOT/'prepared'
def main():
    rows=[json.loads(l) for l in (P/'features.jsonl').open()]
    parent=list(range(len(rows)))
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    seen={}
    for r in rows:
        i=r['row_id']
        for key in ['task_family_hint','instruction_sha256','trajectory_sha256']:
            value=r[key]
            if not value:continue
            k=(key,value)
            if k in seen:union(i,seen[k])
            else:seen[k]=i
        if r['run_id'] and r['trial_name']:
            k=('trial',r['run_id'],r['trial_name'])
            if k in seen:union(i,seen[k])
            else:seen[k]=i
    groups=collections.defaultdict(list)
    for r in rows:groups[find(r['row_id'])].append(r['row_id'])
    assignment={};sizes=collections.Counter();kept_hash=set();exclusions=collections.Counter();counts=collections.Counter()
    for root,ids in groups.items():
        identities=sorted(rows[i]['trajectory_sha256'] for i in ids)
        digest=hashlib.sha256('\n'.join(identities).encode()).hexdigest()
        split='dev' if int(digest[:8],16)%10==0 else 'train'
        for i in ids:assignment[i]=(digest,split)
        sizes[len(ids)]+=1
    with (P/'candidate_manifest.jsonl').open('w') as out:
        for r in rows:
            reason=None
            if r['branch']!='main':reason='non_main'
            elif not r['has_conversations']:reason='empty'
            elif r['unparsed_turn_indices']:reason='unparsed_action'
            elif r['trajectory_sha256'] in kept_hash:reason='exact_duplicate'
            if reason:exclusions[reason]+=1;continue
            kept_hash.add(r['trajectory_sha256']);group,split=assignment[r['row_id']];counts[split]+=1
            out.write(json.dumps({'row_id':r['row_id'],'group_sha256':group,'provisional_split':split,'source_hint':r['source_hint'],'trajectory_sha256':r['trajectory_sha256'],'outcome':'unknown','training_ready':False})+'\n')
    (P/'manifest_summary.json').write_text(json.dumps({'eligible':dict(counts),'excluded':dict(exclusions),'connected_components':len(groups),'max_component_rows':max(map(len,groups.values())),'group_size_histogram':dict(sizes),'note':'Provisional 90/10 grouped split. Connected by task-family hints, exact instructions/conversations, or run+trial. No near-duplicate/repository contamination clearance. No success labels or token matching. NOT a training-ready dataset.'},indent=2))
    directory=P/'review';directory.mkdir(exist_ok=True);items={};membership=collections.defaultdict(list)
    for name in ['annotation_random200','annotation_error100']:
        for l in (P/(name+'.jsonl')).open():
            x=json.loads(l);idx=x['features']['row_id'];items[idx]=x;membership[idx].append(name)
    links=[]
    style='<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f5f7;padding:16px}summary{cursor:pointer;padding:12px;border-top:1px solid #ccc}input{padding:10px;width:80%}a{color:#2256ab}</style>'
    for idx,x in sorted(items.items()):
        f=x['features'];parts=['<!doctype html><meta charset="utf-8">'+style+'<a href="index.html">目录</a>',f'<h1>Row {idx}: {html.escape(f["task"])}</h1>','<p>未标注；错误关键词只是候选。请分别记录事件起因、纠正动作、后续验证与终局结果。</p>','<pre>'+html.escape(json.dumps(f,ensure_ascii=False,indent=2))+'</pre>']
        for i,m in enumerate(x['conversations']):
            parts.append(f'<details id="turn-{i}"><summary>Turn {i} · {html.escape(m["role"])} · {len(m["content"])} chars</summary><pre>{html.escape(m["content"])}</pre></details>')
        (directory/f'{idx}.html').write_text('\n'.join(parts))
        links.append(f'<li><a href="{idx}.html">{idx} — {html.escape(f["task"])}</a> · {", ".join(membership[idx])} · {f["assistant_turns"]} turns</li>')
    page='<!doctype html><meta charset="utf-8">'+style+'<h1>经验轨迹审阅</h1><p>200 条随机样本 + 100 条错误候选；去重后 '+str(len(items))+' 条。全部尚未人工标注。点击展开完整 turn；轨迹命令不会执行。</p><input id="q" placeholder="搜索 task / row / 分组"><ul>'+''.join(links)+'</ul><script>document.getElementById("q").oninput=e=>document.querySelectorAll("li").forEach(x=>x.hidden=!x.textContent.toLowerCase().includes(e.target.value.toLowerCase()))</script>'
    (directory/'index.html').write_text(page)
    print('candidate counts',dict(counts),'review pages',len(items))
if __name__=='__main__':main()
