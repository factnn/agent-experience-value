"""Public-only outcome artifact discovery; no authentication or private DB access."""
import concurrent.futures,json,pathlib,requests,datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
ROOT=pathlib.Path(__file__).resolve().parents[1]
REPOS=['DCAgent/exp_tas_baseline_traces','DCAgent/swesmith-sandboxes-with_tests-gpt-5-mini-passed_glm_4.7_traces','DCAgent/exp_rpt_stack-pytest-v2','DCAgent/nl2bash-GLM-4.6-traces','mlfoundations-dev/inferredbugs-sandboxes-traces-terminus-2','mlfoundations-dev/codeforces-sandboxes-traces-terminus-2','DCAgent/exp_rpt_methods2test-v2','penfever/glm46-swesmith-maxeps-131k','penfever/Kimi-2.5-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-32k-reward1','open-thoughts/OpenThoughts-Agent-RL-5K','open-thoughts/TaskTrove']
OUT=ROOT/'audit/outcome_probes';OUT.mkdir(exist_ok=True)
def probe(repo):
 session=requests.Session();session.mount('https://',HTTPAdapter(max_retries=Retry(total=2,backoff_factor=1,status_forcelist=[429,500,502,503,504])))
 result={'repo':repo,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 try:
  r=session.get('https://huggingface.co/api/datasets/'+repo,timeout=30);result['metadata_status']=r.status_code
  if r.ok:
   m=r.json();repo=m.get('id',repo);result['resolved_repo']=repo;result['sha']=m.get('sha');result['files']=[x['rfilename'] for x in m.get('siblings',[])];
   card=session.get(f'https://huggingface.co/datasets/{repo}/raw/{m["sha"]}/README.md',timeout=30)
   if card.ok:(OUT/(repo.replace('/','__')+'.md')).write_text(card.text)
  r=session.get('https://datasets-server.huggingface.co/rows',params={'dataset':repo,'config':'default','split':'train','offset':0,'length':3},timeout=45)
  result['viewer_status']=r.status_code;j=r.json();result['schema']=j.get('features',[]);result['num_rows']=j.get('num_rows_total');result['sample']=j.get('rows',[])
 except Exception as e:result['error']=repr(e)
 (OUT/(repo.replace('/','__')+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2))
 return {'repo':repo,'metadata_status':result.get('metadata_status'),'viewer_status':result.get('viewer_status'),'outcome_columns':[f['name'] for f in result.get('schema',[]) if any(x in f['name'] for x in ['reward','result','verif','judg'])],'artifact_paths':[s for s in result.get('files',[]) if any(x in s.lower() for x in ['result.json','reward.json','reward.txt'])],'error':result.get('error')}
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(3) as pool:
  records=list(pool.map(probe,REPOS))
 (OUT/'index.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
