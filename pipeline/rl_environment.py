"""Independent smoke tasks on the pinned BFCL MessageAPI simulator.

No benchmark questions/answers are loaded. These are engineering tasks, not an
OOD benchmark. Rewards inspect simulator state, never the assistant's claims.
"""
import copy
import hashlib
import importlib.util
import json
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '.third_party/bfcl_eval_pkg/bfcl_eval/eval_checker/multi_turn_eval/func_source_code/message_api.py'
spec = importlib.util.spec_from_file_location('rl_bfcl_message_api', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
MessageAPI = module.MessageAPI


def make_task(seed, family='send', split='engineering'):
    rng = random.Random(seed)
    names = [f'Person_{seed}_{c}' for c in 'ABC']
    ids = rng.sample(range(100, 999), 3)
    users = dict(zip(names, [f'USR{x:03}' for x in ids]))
    sender, receiver, other = names
    text = f'Package {seed} is ready.'
    inbox = [{users[other]: f'Keep record {seed}.'}]
    if family == 'replace':
        inbox.append({users[receiver]: f'Obsolete record {seed}.'})
    instruction = f'Log in as {sender}, then send exactly "{text}" to {receiver} exactly once. Preserve all existing contacts and other messages.'
    if family == 'replace':
        instruction += f' Before sending, delete the latest old message to {receiver}.'
    elif family == 'new_contact':
        del users[receiver]
        instruction += f' {receiver} is not a contact yet; add this contact first.'
    elif family != 'send':
        raise ValueError(family)
    return {'task_id': f'{split}:{family}:{seed}', 'seed': seed, 'family': family,
            'sender': sender, 'receiver': receiver, 'text': text,
            'scenario': {'random_seed': seed, 'user_map': users, 'user_count': 3,
                         'inbox': inbox, 'message_count': len(inbox), 'current_user': None},
            'prompt': [{'role': 'system', 'content': 'Complete the request using the available tools. Look up IDs rather than guessing. Tool calls change a simulated workspace. Once finished, give a brief confirmation.'},
                       {'role': 'user', 'content': instruction}]}


class MessageEnv:
    def reset(self, task_json: str, **kwargs) -> None:
        self.task = json.loads(task_json)
        self.api = MessageAPI()
        self.api._load_scenario(copy.deepcopy(self.task['scenario']))
        self.events = []
        self.env_seconds = 0.0
        self.limit_exceeded = False

    def _call(self, method, **kwargs):
        start = time.perf_counter()
        before = self._state()
        if len(self.events) >= 12:
            self.limit_exceeded = True
            result = {'error': 'Episode tool-call limit reached. Stop.'}
        else:
            result = getattr(self.api, method)(**kwargs)
        elapsed = time.perf_counter() - start
        self.env_seconds += elapsed
        self.events.append({'method': method, 'arguments': kwargs, 'result': copy.deepcopy(result),
                            'before': before, 'after': self._state(), 'seconds': elapsed})
        return result

    def lookup(self, name: str) -> dict:
        """Look up a contact's ID by exact name.

        Args:
            name: Exact contact name.
        """
        return self._call('get_user_id', user=name)

    def login(self, user_id: str) -> dict:
        """Log in as an existing contact before sending or deleting messages.

        Args:
            user_id: Contact ID returned by lookup.
        """
        return self._call('message_login', user_id=user_id)

    def send(self, receiver_id: str, message: str) -> dict:
        """Send one message to an existing contact; requires login.

        Args:
            receiver_id: Recipient contact ID returned by lookup or add_contact.
            message: Exact message text to send.
        """
        return self._call('send_message', receiver_id=receiver_id, message=message)

    def add_contact(self, name: str) -> dict:
        """Add a new contact and return its ID.

        Args:
            name: New contact's exact name.
        """
        return self._call('add_contact', user_name=name)

    def delete_latest(self, receiver_id: str) -> dict:
        """Delete the latest message to a contact; requires login.

        Args:
            receiver_id: Recipient ID whose latest message should be deleted.
        """
        return self._call('delete_message', receiver_id=receiver_id)

    def _state(self):
        return copy.deepcopy({'user_map': self.api.user_map, 'inbox': self.api.inbox,
                              'current_user': self.api.current_user, 'message_count': self.api.message_count,
                              'user_count': self.api.user_count, 'generated_ids': sorted(self.api.generated_ids)})

    def _reward(self):
        t = self.task
        users = copy.deepcopy(t['scenario']['user_map'])
        if t['family'] == 'new_contact':
            users[t['receiver']] = 'USR004'
        inbox = copy.deepcopy(t['scenario']['inbox'])
        if t['family'] == 'replace':
            inbox.pop()
        inbox.append({users[t['receiver']]: t['text']})
        return float(not self.limit_exceeded and self.api.user_map == users and self.api.inbox == inbox
                     and self.api.current_user == users[t['sender']]
                     and self.api.message_count == t['scenario']['message_count'] + 1
                     and self.api.user_count == t['scenario']['user_count'] + (t['family'] == 'new_contact')
                     and len(self.api.generated_ids) == 1)


def oracle(env):
    t = env.task
    env.login(env.lookup(t['sender'])['user_id'])
    if t['family'] == 'new_contact':
        rid = env.add_contact(t['receiver'])['user_id']
    else:
        rid = env.lookup(t['receiver'])['user_id']
    if t['family'] == 'replace':
        env.delete_latest(rid)
    env.send(rid, t['text'])


def dataset_rows():
    tasks = [make_task(91000 + i, family) for i in range(4)
             for family in ['send', 'new_contact', 'replace']]
    return [{'prompt': t['prompt'], 'task_json': json.dumps(t), 'task_id': t['task_id']} for t in tasks]


def acceptance():
    rows = []
    for family in ['send', 'new_contact', 'replace']:
        for seed in [91000, 91001, 91002, 91003]:
            task = make_task(seed, family)
            env, independent = MessageEnv(), MessageEnv()
            env.reset(json.dumps(task)); independent.reset(json.dumps(task))
            initial = env._state()
            assert env._reward() == 0
            oracle(env)
            assert env._reward() == 1
            assert independent._state() == initial
            env.send(env.api.user_map[task['receiver']], task['text'])
            assert env._reward() == 0, 'duplicate message must fail'
            env.reset(json.dumps(task))
            assert env._state() == initial and not env.events
            # A plausible final claim or an invalid tool call cannot earn reward.
            env.send('missing', task['text'])
            assert env._reward() == 0
            env.reset(json.dumps(task)); oracle(env)
            env.api.current_user = 'wrong'
            assert env._reward() == 0
            env.reset(json.dumps(task)); oracle(env)
            env.api.inbox.pop(0)
            assert env._reward() == 0, 'collateral deletion must fail'
            env.reset(json.dumps(task))
            for _ in range(13):
                env.lookup(task['sender'])
            assert env.limit_exceeded and env._reward() == 0
            rows.append({'task_id': task['task_id'], 'oracle': 1, 'negative_reset_isolation_checks': 'passed'})
    return {'status': 'passed', 'cases': rows, 'simulator_version': 'bfcl_eval==2026.3.23',
            'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'scope': 'engineering only; shared BFCL simulator, independent synthetic tasks; no transfer claim'}


if __name__ == '__main__':
    out = ROOT / 'rl/environment_acceptance.json'
    out.write_text(json.dumps(acceptance(), indent=2) + '\n')
    print(out)
