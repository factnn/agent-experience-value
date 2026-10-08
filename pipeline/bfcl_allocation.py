"""Visible-combination allocation with finite, explicitly charged probe history."""
from collections import deque
import copy
import random


class CombinationAllocator:
    def __init__(self, registry, rule='uniform', seed=0, pool=None, exploration=.2, window=16):
        self.registry=copy.deepcopy(registry)
        self.task_ids=list(registry)
        self.pool=list(pool if pool is not None else registry)
        if not self.pool or len(set(self.pool))!=len(self.pool) or not set(self.pool)<=registry.keys():
            raise ValueError('invalid training pool')
        if rule not in ['uniform','frontier','coverage'] or not 0<exploration<=1 or window<1:
            raise ValueError('invalid allocation configuration')
        self.rule=rule;self.exploration=exploration;self.window=window
        self.rng=random.Random(seed)
        self.history={c:deque(maxlen=window) for c in sorted(set(registry.values()))}
        self.counts={c:0 for c in self.history}
        self.completed_batches=[];self.total_sampled_tokens=0;self.total_episodes=0
        self.selection_count=0;self.pending=None

    def signals(self):
        return {c:{'observed_episodes':len(h),'successes_in_window':sum(h),
            'posterior_success':(sum(h)+1)/(len(h)+2),
            'frontier':4*((sum(h)+1)/(len(h)+2))*(1-(sum(h)+1)/(len(h)+2)),
            'coverage_deficit':1/(1+self.counts[c]),'acquired_groups':self.counts[c]}
            for c,h in self.history.items()}

    def probabilities(self):
        n=len(self.pool);signals=self.signals()
        if self.rule=='uniform':return {k:1/n for k in self.pool}
        if self.rule=='frontier':
            scores={k:signals[self.registry[k]]['frontier'] for k in self.pool}
            total=sum(scores.values())
            return {k:self.exploration/n+(1-self.exploration)*scores[k]/total for k in self.pool}
        # Coverage balances acquisition GROUP counts across visible combinations;
        # after selecting a combination, its tasks are uniform. No oracle paths.
        sizes={c:sum(self.registry[k]==c for k in self.pool) for c in signals}
        total=sum(signals[c]['coverage_deficit'] for c in sizes if sizes[c])
        return {k:self.exploration/n+(1-self.exploration)*signals[self.registry[k]]['coverage_deficit']/
            total/sizes[self.registry[k]] for k in self.pool}

    def choose(self, forced=None):
        if self.pending is not None:raise RuntimeError('unobserved group')
        probabilities=self.probabilities()
        key=forced if forced is not None else self.rng.choices(self.pool,weights=list(probabilities.values()))[0]
        if key not in self.pool:raise ValueError('task outside pool')
        self.pending=key
        return {'task_id':key,'combination':self.registry[key],
            'phase':'common_probe' if forced is not None else 'allocation',
            'probabilities':{k:float(k==key) for k in self.pool} if forced is not None else probabilities,
            'signals_before':self.signals(),'cost_before_choice':self.total_sampled_tokens}

    def observe(self, task_id, rewards, sampled_tokens, batch_id):
        if task_id!=self.pending or batch_id in self.completed_batches:raise ValueError('invalid observation')
        if not rewards or any(r not in [0,1] for r in rewards):raise ValueError('binary rewards required')
        if type(sampled_tokens) is not int or sampled_tokens<1:raise ValueError('charge all generated tokens')
        c=self.registry[task_id];self.history[c].extend(rewards);self.counts[c]+=1
        self.total_sampled_tokens+=sampled_tokens;self.total_episodes+=len(rewards)
        self.selection_count+=1;self.completed_batches.append(batch_id);self.pending=None

    def state_dict(self):
        if self.pending is not None:raise RuntimeError('checkpoint at complete group boundary')
        return {'kind':'bfcl_combination','version':1,'registry':self.registry,'pool':self.pool,
            'rule':self.rule,'exploration':self.exploration,'window':self.window,
            'history':{c:list(h) for c,h in self.history.items()},'counts':self.counts,
            'rng_state':self.rng.getstate(),'completed_batches':self.completed_batches,
            'total_sampled_tokens':self.total_sampled_tokens,'total_episodes':self.total_episodes,
            'selection_count':self.selection_count}

    @classmethod
    def from_state_dict(cls,state,rule=None,pool=None):
        if state['kind']!='bfcl_combination' or state['version']!=1:raise ValueError('unsupported state')
        result=cls(state['registry'],state['rule'] if rule is None else rule,
            pool=state['pool'] if pool is None else pool,exploration=state['exploration'],window=state['window'])
        if set(state['history'])!=set(result.history) or set(state['counts'])!=set(result.counts):
            raise ValueError('combination registry mismatch')
        for c,h in state['history'].items():
            if len(h)>result.window or any(r not in [0,1] for r in h):raise ValueError('invalid history')
            result.history[c].extend(h)
        if state['selection_count']!=len(state['completed_batches']) or len(set(state['completed_batches']))!=state['selection_count']:
            raise ValueError('invalid group accounting')
        if sum(state['counts'].values())!=state['selection_count'] or any(type(v) is not int or v<0 for v in state['counts'].values()):
            raise ValueError('invalid coverage counts')
        if state['total_sampled_tokens']<state['selection_count'] or state['total_episodes']<sum(map(len,state['history'].values())):
            raise ValueError('invalid cost accounting')
        def tuples(value):return tuple(map(tuples,value)) if isinstance(value,(list,tuple)) else value
        result.rng.setstate(tuples(state['rng_state']))
        for key in ['counts','completed_batches','total_sampled_tokens','total_episodes','selection_count']:
            setattr(result,key,copy.deepcopy(state[key]))
        return result
