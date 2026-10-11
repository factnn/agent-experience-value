"""Supplementary optimizer and update-scale audit; original criteria unchanged."""
import argparse
import hashlib
from pathlib import Path
import torch
from report_bfcl_backward import ROOT, read, write, tensors, digest
from bfcl_fixed_rollout_inputs import tensor_distance


def fingerprint(weights):
    h=hashlib.sha256()
    for name,tensor in weights.items():
        h.update(name.encode());h.update(tensor.float().numpy().tobytes())
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4);plan=read(args.protocol);root=ROOT/plan['output_root']
    assert read(root/'audit_status.json')['status']=='diagnostic_complete'
    summaries={};optimizers={};deltas={}
    for case in plan['cases']:
        name=case['name'];folder=root/name;run=read(folder/'run.json');s=read(folder/'summary.json')
        start,final=tensors(folder,'start_parameters'),tensors(folder,'final_parameters')
        assert fingerprint(start)==run['start_fingerprint']==plan['zero_gradient_reconstruction'][-1]['fingerprint']
        assert fingerprint(final)==s['final_fingerprint']
        assert run['deterministic_algorithms']==case['deterministic']
        assert run['cublas_workspace_config']==(':4096:8' if case['deterministic'] else None)
        deltas[name]=tensor_distance(start,final)
        path=folder/'final_optimizer/learning_state.pt';assert digest(path)==s['final_optimizer_sha256']
        optimizers[name]=torch.load(path,map_location='cpu',weights_only=False)
        summaries[name]=s
    pairs=[]
    for left,right in plan['comparisons']:
        a,b=optimizers[left],optimizers[right]
        assert a['param_groups']==b['param_groups'] and a['state'].keys()==b['state'].keys()
        flat=[]
        for state in [a,b]:
            flat.append({str(i)+':'+key:value for i,values in state['state'].items() for key,value in values.items()})
        assert all(torch.is_tensor(v) for mapping in flat for v in mapping.values())
        distances=tensor_distance(tensors(root/left,'final_parameters'),tensors(root/right,'final_parameters'))
        pairs.append({'left':left,'right':right,'optimizer_exact':tensor_distance(*flat)['exact'],
            'optimizer_distance_by_field':{field:tensor_distance(
                {k:v for k,v in flat[0].items() if k.endswith(':'+field)},
                {k:v for k,v in flat[1].items() if k.endswith(':'+field)})
                for field in sorted({k.split(':',1)[1] for k in flat[0]})},
            'parameter_difference_relative_to_left_update':distances['l2_difference']/deltas[left]['l2_difference'],
            'parameter_difference_relative_to_right_update':distances['l2_difference']/deltas[right]['l2_difference']})
    result={'status':'passed','case_update_distances':deltas,'pairs':pairs,'new_sampled_tokens':0,
        'scope':'Post-run descriptive audit of saved tensors; no new acceptance tolerance, efficacy claim, or significance test.'}
    write(root/'tensor_detail_audit.json',result)
    print(result)


if __name__=='__main__':main()
