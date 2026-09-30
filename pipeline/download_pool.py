"""Download pinned public parquet; verify against HF LFS SHA256. Two workers."""
import concurrent.futures,hashlib,json,pathlib,requests
ROOT=pathlib.Path(__file__).resolve().parents[1]
FILES=json.loads((ROOT/'audit/parquet_files.json').read_text())
SHA=json.loads((ROOT/'audit/OpenThoughts-Agent-SFT-100K/metadata.json').read_text())['sha']
def fetch(f):
 p=ROOT/'data/SFT-100K'/pathlib.Path(f['path']).name;p.parent.mkdir(parents=True,exist_ok=True)
 def digest(path):
  h=hashlib.sha256()
  with path.open('rb') as s:
   for b in iter(lambda:s.read(1024*1024),b''):h.update(b)
  return h.hexdigest()
 expected=f['lfs']['oid']
 if p.exists() and p.stat().st_size==f['size'] and digest(p)==expected:return str(p)
 tmp=p.with_suffix('.partial')
 with requests.get(f'https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-SFT-100K/resolve/{SHA}/{f["path"]}',stream=True,timeout=(30,120)) as r:
  r.raise_for_status()
  with tmp.open('wb') as out:
   for b in r.iter_content(1024*1024):out.write(b)
 assert tmp.stat().st_size==f['size'],str(p)
 assert digest(tmp)==expected,str(p)
 tmp.replace(p);print('verified',p.name,p.stat().st_size,flush=True);return str(p)
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(2) as pool:list(pool.map(fetch,FILES))
 print('ALL VERIFIED',SHA,flush=True)
