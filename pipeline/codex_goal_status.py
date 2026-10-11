"""Read the owning daemon's Goal state through its generated JSON-RPC contract.

Never fall back to a potentially stale SQLite/WAL side view. Errors fail closed.
Uses the already installed websockets client; no model calls, new sessions or keys.
"""
import asyncio
import json
from pathlib import Path


async def query_goal(home, thread):
    import websockets
    path = Path(home) / 'app-server-control/app-server-control.sock'
    async with websockets.unix_connect(str(path), uri='ws://localhost',
                                       open_timeout=3, close_timeout=1) as client:
        async def request(identity, method, params):
            await client.send(json.dumps({'id': identity, 'method': method, 'params': params}))
            for _ in range(25):
                response = json.loads(await asyncio.wait_for(client.recv(), timeout=3))
                if response.get('id') == identity:
                    if 'error' in response:
                        raise RuntimeError('Owning app-server Goal query failed')
                    return response['result']
            raise RuntimeError('No matching Goal response')

        await request(1, 'initialize', {'clientInfo': {'name': 'bfcl-goal-observer', 'version': '2.0'},
                                         'capabilities': {'experimentalApi': True}})
        await client.send(json.dumps({'method': 'initialized'}))
        result = await request(2, 'thread/goal/get', {'threadId': thread})
        goal = result.get('goal')
        if goal is None:
            return 'missing'
        if goal.get('threadId') != thread:
            return 'unknown'
        status = goal.get('status')
        normalized = {'usageLimited': 'usage_limited', 'budgetLimited': 'budget_limited'}.get(status, status)
        return normalized if normalized in {'active', 'paused', 'blocked', 'complete',
                                             'usage_limited', 'budget_limited'} else 'unknown'


def goal_status(home, thread):
    try:
        return asyncio.run(asyncio.wait_for(query_goal(home, thread), timeout=10))
    except Exception:
        return 'unknown'
