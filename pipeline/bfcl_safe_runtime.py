"""Episode-local simulator dispatch with literal-only calls and official checks.

No model-provided Python expression is evaluated. This is CPU runtime acceptance,
not yet the multi-user-turn TRL environment.
"""
import ast
import copy
import importlib
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
PKG=ROOT/'.third_party/bfcl_eval_pkg'
sys.path.insert(0,str(PKG))
from bfcl_eval.constants.executable_backend_config import (CLASS_FILE_PATH_MAPPING,
    MULTI_TURN_FUNC_DOC_FILE_MAPPING, STATELESS_CLASSES)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import state_checker,response_checker
from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils

ALLOWED_CLASSES={'GorillaFileSystem','MathAPI','MessageAPI','TwitterAPI',
                 'TicketAPI','TradingBot','TravelAPI','VehicleControlAPI'}


class SimulatorEpisode:
    def __init__(self,task):
        self.task=copy.deepcopy(task);self.instances={};self.methods={};self.responses=[]
        for name in task['involved_classes']:
            if name not in ALLOWED_CLASSES:raise ValueError('unsupported simulator class')
            module=importlib.import_module(CLASS_FILE_PATH_MAPPING[name])
            instance=getattr(module,name)()
            if name not in STATELESS_CLASSES:
                instance._load_scenario(copy.deepcopy(task['initial_config'].get(name,{})))
            self.instances[name]=instance
            path=PKG/'bfcl_eval/data/multi_turn_func_doc'/MULTI_TURN_FUNC_DOC_FILE_MAPPING[name]
            for line in path.read_text().splitlines():
                schema=json.loads(line);method=schema['name']
                if method in task.get('excluded_function',[]):continue
                if method.startswith('_'):raise ValueError('private documented method')
                self.methods[(name,method)]=getattr(instance,method)

    def call(self,text):
        tree=ast.parse(text,mode='eval').body
        if not isinstance(tree,ast.Call):raise ValueError('expected one documented function call')
        if isinstance(tree.func,ast.Name):
            matches=[key for key in self.methods if key[1]==tree.func.id]
            if len(matches)!=1:raise ValueError('unknown or ambiguous tool name')
            key=matches[0]
        elif isinstance(tree.func,ast.Attribute) and isinstance(tree.func.value,ast.Name):
            key=(tree.func.value.id,tree.func.attr)
            if key not in self.methods:raise ValueError('unknown documented method')
        else:raise ValueError('nested or indirect callable rejected')
        if any(isinstance(a,ast.Starred) for a in tree.args) or any(k.arg is None for k in tree.keywords):
            raise ValueError('argument unpacking rejected')
        if len({k.arg for k in tree.keywords})!=len(tree.keywords):raise ValueError('duplicate argument')
        try:
            args=[ast.literal_eval(a) for a in tree.args]
            kwargs={k.arg:ast.literal_eval(k.value) for k in tree.keywords}
        except (ValueError,TypeError):raise ValueError('only literal arguments accepted')
        result=self.methods[key](*args,**kwargs)
        if isinstance(result,str):formatted=result
        elif isinstance(result,dict):
            try:formatted=json.dumps(result)
            except TypeError:formatted=str(result)
        else:formatted=str(result)
        self.responses.append(formatted);return formatted

    def matches(self,reference,turn):
        state=state_checker(self.instances,reference.instances)
        if not state['valid']:return state
        return response_checker(self.responses,reference.responses[-reference.turn_response_count:],turn) if reference.turn_response_count else {'valid':True}


def acceptance():
    manifest=json.loads((ROOT/'rl/bfcl_research_split_candidate/manifest.json').read_text())
    targets={r['id'] for r in manifest['development']}
    data={r['id']:r for r in map(json.loads,(PKG/'bfcl_eval/data/BFCL_v4_multi_turn_base.json').read_text().splitlines()) if r['id'] in targets}
    answers={r['id']:r['ground_truth'] for r in map(json.loads,(PKG/'bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_base.json').read_text().splitlines()) if r['id'] in targets}
    results=[]
    for key in sorted(targets):
        task=data[key];reference=SimulatorEpisode(task);replica=SimulatorEpisode(task)
        untouched=SimulatorEpisode(task);turn_results=[];no_op_fails=False;rejected=[];parity=[]
        official_name=f'dev_parity_{os.getpid()}'
        try:
            for turn,calls in enumerate(answers[key]):
                reference.turn_response_count=len(calls)
                for call in calls:reference.call(call);replica.call(call)
                result=replica.matches(reference,turn);turn_results.append(bool(result['valid']))
                # Only trusted, already literal-validated benchmark oracle calls
                # reach the legacy executor; never use this path for model calls.
                official_responses,official_instances=multi_turn_utils.execute_multi_turn_func_call(
                    calls,task['initial_config'],task['involved_classes'],official_name,key)
                parity.append(bool(state_checker(reference.instances,official_instances)['valid'])
                    and response_checker(reference.responses,official_responses,turn)['valid'])
                if calls and not untouched.matches(reference,turn)['valid']:no_op_fails=True
            reset=SimulatorEpisode(task)
            assert state_checker(reset.instances,untouched.instances)['valid']
            for bad in ['__import__("os").getcwd()', 'eval("1")',
                        'ls(directory_path=__import__("os").getcwd())',
                        'ls(**{})','GorillaFileSystem.__dict__()']:
                try:reset.call(bad)
                except (ValueError,SyntaxError):rejected.append(bad)
            results.append({'id':key,'oracle_turn_checks':turn_results,
                'oracle_passed':bool(turn_results) and all(turn_results),'noop_rejected':no_op_fails,
                'official_executor_parity':all(parity),
                'reset_and_instance_isolation':True,'nonliteral_or_undocumented_calls_rejected':len(rejected)==5})
        except Exception as error:
            results.append({'id':key,'oracle_passed':False,'error':f'{type(error).__name__}: {error}'})
        finally:
            for name in task['involved_classes']:
                variable=re.sub(r'[-./:]','_',f'{official_name}_{key}_{name}_instance')
                vars(multi_turn_utils).pop(variable,None)
    result={'scope':'22 development tasks only; no ID/transfer scoring or model generation',
        'tasks':results,'passed':all(r.get('oracle_passed') and r.get('noop_rejected')
            and r.get('official_executor_parity')
            and r.get('reset_and_instance_isolation') and r.get('nonliteral_or_undocumented_calls_rejected') for r in results),
        'official_checkers':'state_checker + cumulative response_checker at each user turn',
        'runtime':'fresh simulator instances; documented method allowlist; AST call + literal_eval arguments',
        'limitation':'Oracle self-consistency and negative controls; not exhaustive reward validity, semantic isolation, or RL multi-turn integration.'}
    (ROOT/'rl/bfcl_research_split_candidate/runtime_acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':acceptance()
