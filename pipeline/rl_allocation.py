"""Candidate pre-rollout allocation rules; not yet a frozen experiment recipe.

Uses only previously observed training-task rewards. Cost accounting includes
warmup, failures and discarded samples supplied by the rollout collector.
"""
from collections import deque
import math
import random


class TaskAllocator:
    def __init__(self, task_ids, rule='uniform', seed=0, exploration=0.2, window=16):
        self.task_ids = list(task_ids)
        if not self.task_ids or len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError('task IDs must be nonempty and unique')
        if rule not in ['uniform', 'frontier'] or not 0 < exploration <= 1 or window < 1:
            raise ValueError('invalid allocation configuration')
        self.rule, self.exploration, self.window = rule, exploration, window
        self.rng = random.Random(seed)
        self.history = {key: deque(maxlen=window) for key in self.task_ids}
        self.warmup = list(self.task_ids)
        self.rng.shuffle(self.warmup)
        self.completed_batches = set()
        self.pending = None
        self.total_sampled_tokens = 0
        self.total_episodes = 0
        self.selection_count = 0

    def probabilities(self):
        n = len(self.task_ids)
        if self.rule == 'uniform':
            return {key: 1/n for key in self.task_ids}
        # Beta(1,1) smoothing, applied only to the rolling history window.
        p = {key: (sum(h)+1)/(len(h)+2) for key,h in self.history.items()}
        scores = {key: 4*x*(1-x) for key,x in p.items()}
        total = sum(scores.values())
        return {key: self.exploration/n + (1-self.exploration)*scores[key]/total for key in self.task_ids}

    def choose(self):
        if self.pending is not None:
            raise RuntimeError('observe the pending rollout group before choosing another task')
        if self.selection_count < len(self.warmup):
            key = self.warmup[self.selection_count]
            probabilities = {k: float(k == key) for k in self.task_ids}
            phase = 'common_warmup'
        else:
            probabilities = self.probabilities()
            key = self.rng.choices(self.task_ids, weights=[probabilities[k] for k in self.task_ids])[0]
            phase = 'allocation'
        self.pending = key
        self.selection_count += 1
        return {'task_id': key, 'phase': phase, 'probabilities': probabilities,
                'cost_before_choice': self.total_sampled_tokens}

    def observe(self, task_id, rewards, sampled_tokens, batch_id):
        if task_id != self.pending or task_id not in self.history:
            raise ValueError('observation does not match the pending training task')
        if batch_id in self.completed_batches:
            raise ValueError('duplicate rollout batch')
        rewards = list(rewards)
        if not rewards or any(not math.isfinite(r) or r not in [0, 1] for r in rewards):
            raise ValueError('expected nonempty binary oracle rewards')
        if type(sampled_tokens) is not int or sampled_tokens < 1:
            raise ValueError('charge every sampled token, not only retained/successful samples')
        self.history[task_id].extend(rewards)
        self.total_sampled_tokens += sampled_tokens
        self.total_episodes += len(rewards)
        self.completed_batches.add(batch_id)
        self.pending = None
