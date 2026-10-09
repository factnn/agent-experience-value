"""CPU-only endpoint diagnostics from checksum-verified, locally created states."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from rl_learning_state import tensor_hash

def load_components(directory):
    directory=Path(directory)/'components'
    manifest=json.loads((directory/'manifest.json').read_text())
    payload=directory/'learning_state.pt'
    assert hashlib.sha256(payload.read_bytes()).hexdigest()==manifest['payload_sha256']
    # These are this project's own local snapshots, containing NumPy RNG/optimizer.
    state=torch.load(payload,map_location='cpu',weights_only=False)
    assert tensor_hash(state['trainable'].items())==manifest['trainable_sha256']
    assert state['frozen_hash']==manifest['frozen_base_sha256']
    return state,manifest

def audit(common,branch,scaling):
    torch.set_num_threads(min(torch.get_num_threads(),4))
    base,base_manifest=load_components(common)
    end,end_manifest=load_components(branch)
    assert base['model_signature']==end['model_signature']
    assert base['frozen_hash']==end['frozen_hash']
    assert base_manifest['provenance']['algorithm_config_sha256']==end_manifest['provenance']['algorithm_config_sha256']
    before=base['trainable'];after=end['trainable'];assert before.keys()==after.keys()
    squared=0.;initial_squared=0.;parameters=0;modules=[]
    for name,value in after.items():
        delta=value.double()-before[name].double()
        squared+=delta.square().sum().item();initial_squared+=before[name].double().square().sum().item()
        parameters+=value.numel()
        if '.lora_A.' not in name:continue
        bname=name.replace('.lora_A.','.lora_B.')
        assert bname in before and torch.count_nonzero(before[bname]).item()==0
        a=value.double();b=after[bname].double()
        # ||B A||_F^2 = tr((B^T B)(A A^T)); avoid materializing dense deltas.
        norm_squared=(b.T@b * (a@a.T).T).sum().item()*scaling**2
        assert norm_squared>=-1e-12
        modules.append({'module':name.split('.lora_A.')[0],
            'effective_weight_delta_frobenius':max(0.,norm_squared)**.5})
    assert len(modules)*2==len(after)
    return {'status':'checksum_and_parameter_displacement_audited',
        'common_payload_sha256':base_manifest['payload_sha256'],
        'branch_payload_sha256':end_manifest['payload_sha256'],
        'trainable_parameter_count':parameters,'trainable_parameter_delta_l2':squared**.5,
        'trainable_parameter_delta_rms':(squared/parameters)**.5,
        'delta_l2_relative_to_initial_trainable_l2':(squared/initial_squared)**.5,
        'lora_scaling':scaling,'effective_weight_delta_frobenius':sum(m['effective_weight_delta_frobenius']**2 for m in modules)**.5,
        'modules':modules,'new_model_calls':0,
        'interpretation':'Endpoint update magnitude within the fixed LoRA subspace. Neither parameter displacement nor effective weight delta is evidence of ability gain. Initial LoRA B is verified zero; relative norm includes randomly initialized A.'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--common',type=Path,required=True)
    ap.add_argument('--branch',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--lora-scaling',type=float,required=True);args=ap.parse_args()
    torch.set_num_threads(4)
    result=audit(args.common,args.branch,args.lora_scaling)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='modules'}))
if __name__=='__main__':main()
