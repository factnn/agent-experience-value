"""Question-only surface/template screening; no semantic-exclusion guarantee."""
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


def text(question,first=False):
    return '\nUSER_TURN\n'.join('\n'.join(m['content'] for m in turn if m['role']=='user') for turn in (question[:1] if first else question))


def normalize(value):
    value=value.lower()
    value=re.sub(r'"[^"\n]*"|\u201c[^\u201d\n]*\u201d',' QUOTED_VALUE ',value)
    value=re.sub(r'\b\d+(?:\.\d+)?\b',' NUMBER ',value)
    return ' '.join(re.findall(r'\w+',value))


def vectors(documents):
    bags={}
    for key,value in documents.items():
        words=value.split();grams=words+[' '.join(words[i:i+2]) for i in range(len(words)-1)]
        bags[key]=Counter(grams)
    frequency=Counter(g for bag in bags.values() for g in bag)
    result={}
    for key,bag in bags.items():
        weights={g:(1+math.log(n))*(1+math.log((1+len(bags))/(1+frequency[g]))) for g,n in bag.items()}
        norm=math.sqrt(sum(w*w for w in weights.values()))
        result[key]={g:w/norm for g,w in weights.items()} if norm else {}
    return result


def cosine(a,b):return min(1.,sum(w*b.get(g,0.) for g,w in a.items()))


def main():
    manifest_path=ROOT/'rl/bfcl_research_split_candidate/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    source=ROOT/'.third_party/bfcl_eval_pkg/bfcl_eval/data/BFCL_v4_multi_turn_base.json'
    data={r['id']:r['question'] for r in map(json.loads,source.read_text().splitlines())}
    partition={r['id']:name for name,rs in manifest.items() for r in rs if name!='historical_quarantine'}
    training=[key for key,name in partition.items() if name=='train']
    documents={key:normalize(text(data[key])) for key in partition}
    first={key:normalize(text(data[key],first=True)) for key in partition}
    full_vectors=vectors(documents);first_vectors=vectors(first)
    nearest=[];flags=[]
    for key,name in partition.items():
        if name=='train':continue
        a=max(training,key=lambda k:cosine(full_vectors[key],full_vectors[k]))
        b=max(training,key=lambda k:cosine(first_vectors[key],first_vectors[k]))
        full=cosine(full_vectors[key],full_vectors[a]);initial=cosine(first_vectors[key],first_vectors[b])
        exact=documents[key]==documents[a]
        row={'candidate_id':key,'partition':name,'nearest_train_full_question':a,
            'full_normalized_tfidf_cosine':full,'normalized_full_template_exact':exact,
            'nearest_train_initial_request':b,'initial_request_tfidf_cosine':initial}
        assert 0<=full<=1 and 0<=initial<=1
        nearest.append(row)
        if exact or full>=0.80 or initial>=0.90:flags.append(row)
    result={'status':'surface_screen_complete_semantic_review_remaining',
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'inputs':'User question text only; no gold answers, gold path, model performance or allocator signal',
        'method':'Lowercase, normalize numbers/double-quoted spans; TF-IDF word unigrams+bigrams; nearest train cosine, full dialogue and first request separately',
        'flag_rule':'Normalized full template exact OR full cosine>=0.80 OR initial-request cosine>=0.90',
        'flagged_by_partition':dict(Counter(r['partition'] for r in flags)),
        'full_template_exact_pairs':sum(r['normalized_full_template_exact'] for r in nearest),
        'full_dialogue_high_similarity':sum(r['full_normalized_tfidf_cosine']>=0.80 for r in nearest),
        'initial_request_high_similarity':sum(r['initial_request_tfidf_cosine']>=0.90 for r in nearest),
        'flagged_candidates':flags,'all_nearest_training_pairs':nearest,
        'limitations':'Surface screen is not semantic equivalence certification. A shared initial subtask may be intended in composition transfer. Do not drop cases based on model success or treat a low score as proof of independence. Human review and pretraining-exposure caveats remain; no split changed.'}
    path=ROOT/'rl/bfcl_research_split_candidate/text_overlap_screen.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'flagged_candidates','all_nearest_training_pairs'}}))


if __name__=='__main__':main()
