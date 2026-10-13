"""Authorized CPU-only, immutable ModelScope downloads; no scientific evaluation."""
import os
from pathlib import Path
ROOT=Path('/share/project/zpy/RL/agent_experience_project_20260928')
BASE=ROOT/'models_downloads'; EVIDENCE=ROOT/'rl/model_downloads_20261013'
os.environ.update(CUDA_VISIBLE_DEVICES='',MODELSCOPE_CACHE=str(BASE/'cache'),MODELSCOPE_HOME=str(BASE/'sdk_home'),MODELSCOPE_ENDPOINT='https://modelscope.cn',MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS='1',MODELSCOPE_DOWNLOAD_MAX_RETRIES='3',MODELSCOPE_API_MAX_RETRIES='2',TQDM_DISABLE='1')
for name in ['MODELSCOPE_API_TOKEN','MODELSCOPE_TOKEN']:os.environ.pop(name,None)
import sys,json,time,hashlib,re,urllib.request,urllib.parse,struct,math,datetime,importlib.metadata,subprocess
class SafeStream:
 def __init__(self,out):self.out=out
 def write(self,s):return self.out.write(re.sub(r'https?://[^\s\"\']+','[URL redacted]',s))
 def flush(self):self.out.flush()
 def isatty(self):return False
sys.stdout=SafeStream(sys.stdout);sys.stderr=SafeStream(sys.stderr)
from modelscope_hub import HubApi
api=HubApi(token='',endpoint='https://modelscope.cn')
MODELS=[('LLM-Research/Meta-Llama-3.1-8B-Instruct','359efdbb8af05b788a4ad4185215c6b8caa9052c','llama')]
EVIDENCE.mkdir(parents=True,exist_ok=True)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def event(phase,**kw):
 row={'utc':now(),'phase':phase,**kw}
 with (EVIDENCE/'events_resume_02.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 write(EVIDENCE/'progress_resume_02.json',row);print(json.dumps(row),flush=True)
def raw_listing(mid,rev):
 url='https://modelscope.cn/api/v1/models/'+mid+'/repo/files?'+urllib.parse.urlencode({'Revision':rev,'Recursive':'true'})
 with urllib.request.urlopen(url,timeout=60) as f:body=json.load(f)
 assert body['Code']==200 and body['Success']
 return body
def content_identity(files):
 entries={x['Path']:(x['Size'],x['Sha256']) for x in files}
 assert len(entries)==len(files), 'Duplicate file paths in listing'
 return entries
versions={'python':sys.version,'modelscope-hub':importlib.metadata.version('modelscope-hub')}
write(EVIDENCE/'environment_resume_02.json',{'versions':versions,'created_at':now(),'cuda_visible_devices':'','max_file_workers':4,'max_range_workers':1,'model_order':[x[0] for x in MODELS], 'sdk_official_source':'https://github.com/modelscope/modelscope_hub','source_script_sha256':digest(Path(__file__))})
try:
 for mid,rev,model_type in MODELS:
  slug=mid.split('/')[-1];dest=BASE/slug;dest.mkdir(exist_ok=True)
  manifest_path=EVIDENCE/(slug+'_source_manifest.json')
  body=raw_listing(mid,rev)
  rows=[x for x in body['Data']['Files'] if x['Type']=='blob' and '/' not in x['Path'] and not x['Path'].endswith(('.pth','.bin','.gguf'))]
  manifest={'repo_id':mid,'immutable_commit':rev,'branch_resolved':'master','source_page':'https://modelscope.cn/models/'+mid,'retrieved_at':now(),'destination':str(dest),'files':[{k:x[k] for k in ['Path','Size','Sha256','Revision','IsLFS']} for x in rows],'excluded_paths':[x['Path'] for x in body['Data']['Files'] if x['Type']=='blob' and x not in rows]}
  if manifest_path.exists():
   old=json.loads(manifest_path.read_text());assert old['immutable_commit']==rev and content_identity(old['files'])==content_identity(manifest['files']);manifest=old
  else:
   with manifest_path.open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
  assert sum(x['Size'] for x in rows)<21*10**9
  assert all(re.fullmatch('[0-9a-f]{64}',x['Sha256']) for x in rows)
  event('metadata_pinned',model=mid,commit=rev,files=len(rows),bytes=sum(x['Size'] for x in rows))
  # Confirm full precision architecture and complete shard set before weights.
  for filename in ['config.json','model.safetensors.index.json']:
   api.download_file(mid,'model',filename,revision=rev,local_dir=str(dest),cache_dir=str(BASE/'cache'))
  config=json.loads((dest/'config.json').read_text());index=json.loads((dest/'model.safetensors.index.json').read_text())
  assert config['model_type']==model_type and 'quantization_config' not in config
  shards=set(index['weight_map'].values());assert shards=={x['Path'] for x in rows if x['Path'].endswith('.safetensors')}
  event('download_started',model=mid,commit=rev,shards=len(shards),bytes=sum(x['Size'] for x in rows))
  api.download_repo(mid,'model',revision=rev,local_dir=str(dest),cache_dir=str(BASE/'cache'),allow_patterns=[x['Path'] for x in rows],max_workers=4)
  actual=[];tensors={};dtypes=set();parameter_count=0
  event('validating',model=mid)
  for x in rows:
   p=dest/x['Path'];assert p.is_file() and p.stat().st_size==x['Size'],x['Path'];sha=digest(p);assert sha==x['Sha256'],x['Path']
   actual.append({'path':x['Path'],'bytes':p.stat().st_size,'sha256':sha})
   if p.suffix=='.safetensors':
    with p.open('rb') as f:
     headlen=struct.unpack('<Q',f.read(8))[0];assert headlen<10000000;header=json.loads(f.read(headlen))
    for key,value in header.items():
     if key=='__metadata__':continue
     assert key not in tensors and index['weight_map'][key]==x['Path'];tensors[key]=x['Path'];dtypes.add(value['dtype']);parameter_count+=math.prod(value['shape'])
     start,end=value['data_offsets'];assert 0<=start<=end<=p.stat().st_size-8-headlen
  assert tensors==index['weight_map'];assert dtypes.issubset({'BF16','F16','F32'})
  assert 7*10**9<parameter_count<9*10**9
  assert (dest/'LICENSE').exists() and (dest/'README.md').exists() and (dest/'tokenizer.json').exists() and (dest/'tokenizer_config.json').exists()
  json.loads((dest/'tokenizer.json').read_text());json.loads((dest/'tokenizer_config.json').read_text())
  result={'status':'verified','verified_at':now(),'repo_id':mid,'immutable_commit':rev,'destination':str(dest),'files':actual,'model_type':model_type,'quantization_config_present':False,'weight_index_complete':True,'tensor_count':len(tensors),'parameter_count':parameter_count,'dtypes':sorted(dtypes),'license_and_readme_preserved':True,'scope':'File identity, complete full-precision safetensor index/header integrity and parseable config/tokenizer; no model inference, training, or target evaluation.'}
  target=EVIDENCE/(slug+'_verification.json')
  if target.exists():assert json.loads(target.read_text())['files']==actual
  else:
   with target.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
  event('model_verified',model=mid,bytes=sum(x['bytes'] for x in actual),parameter_count=parameter_count)
 event('complete',models=1,retained_verified_model='Qwen/Qwen3-8B')
except Exception as e:
 event('failed',error_type=type(e).__name__,message=re.sub(r'https?://[^\s]+','[URL redacted]',str(e))[:1500]);raise
