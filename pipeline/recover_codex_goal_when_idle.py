"""One bounded daemon recovery for SQLite 522, only after every session is idle.

No Goal mutations, database writes, new model sessions, or GPU process signals.
Run detached so the current conversation can finish before an eligible restart.
"""
import asyncio
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from codex_goal_status import goal_status
from watch_research_events import write

ROOT = Path(__file__).resolve().parents[1]


async def snapshot(home, thread):
    import websockets
    async with websockets.unix_connect(str(home / 'app-server-control/app-server-control.sock'),
                                       uri='ws://localhost', open_timeout=3, close_timeout=1) as ws:
        async def rpc(identity, method, params):
            await ws.send(json.dumps({'id': identity, 'method': method, 'params': params}))
            for _ in range(100):
                response = json.loads(await asyncio.wait_for(ws.recv(), 3))
                if response.get('id') == identity:
                    return response
            raise RuntimeError('No matching daemon response')
        init = await rpc(1, 'initialize', {'clientInfo': {'name': 'bfcl-control-recovery', 'version': '1.0'},
                                          'capabilities': {'experimentalApi': True}})
        if 'error' in init:
            raise RuntimeError('Cannot initialize control client')
        await ws.send(json.dumps({'method': 'initialized'}))
        goal = await rpc(2, 'thread/goal/get', {'threadId': thread})
        error = goal.get('error', {}).get('message', '')
        if 'error' not in goal:
            state = goal.get('result', {}).get('goal')
            return {'goal_query_ok': True, 'goal_status': state['status'] if state else 'missing'}
        loaded = await rpc(3, 'thread/loaded/list', {'limit': 100})
        page = loaded.get('result')
        if page is None or page.get('nextCursor'):
            return {'goal_query_ok': False, 'sqlite_522': '(code: 522)' in error,
                    'all_idle': False, 'reason': 'incomplete_session_status'}
        states = []
        for index, tid in enumerate(page['data']):
            result = await rpc(index + 4, 'thread/read', {'threadId': tid, 'includeTurns': False})
            states.append(result.get('result', {}).get('thread', {}).get('status', {}).get('type', 'unknown'))
        return {'goal_query_ok': False, 'sqlite_522': '(code: 522)' in error and 'disk I/O' in error,
                'all_idle': bool(states) and all(status == 'idle' for status in states),
                'loaded_sessions': len(states), 'active_sessions': states.count('active'),
                'unknown_sessions': states.count('unknown')}


def action(state, consecutive_idle, attempted):
    if state.get('goal_query_ok'):
        return 'healthy'
    if attempted:
        return 'wait_after_single_attempt'
    if state.get('sqlite_522') and state.get('all_idle') and consecutive_idle >= 2:
        return 'restart_once'
    return 'wait'


def detached_workers():
    status = json.loads((ROOT / 'rl/STATUS.json').read_text())['bfcl_intervention']
    pids = [status['scheduler_pid']] + [job['pid'] for job in status['active']]
    for pid in pids:
        try:
            if os.getsid(pid) != pid or os.getpgid(pid) != pid:
                return False
        except ProcessLookupError:
            pass
    return True


def persist_receipt(record):
    path = 'experiments/control_recovery_receipt.json'
    write(ROOT / path, record)
    commands = [['git', 'add', '--', path],
                ['git', 'commit', '--only', '-m', 'Record bounded idle-only Goal control recovery', '--', path],
                ['git', 'push']]
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=60)
        if result.returncode:
            return False
    return True


def main():
    home = Path(os.environ['CODEX_HOME'])
    thread = os.environ['CODEX_THREAD_ID']
    runtime = ROOT / 'experiments/.runtime'
    with (runtime / 'control_recovery.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = runtime / 'control_recovery.json'
        record = json.loads(state_path.read_text()) if state_path.exists() else {
            'started_at_epoch': time.time(), 'attempted_restart': False}
        idle = 0
        while time.time() - record['started_at_epoch'] < 21600:
            try:
                current = asyncio.run(asyncio.wait_for(snapshot(home, thread), 15))
                idle = idle + 1 if current.get('all_idle') else 0
                decision = action(current, idle, record['attempted_restart'])
                record.update(heartbeat=datetime.now(timezone.utc).isoformat(), last_snapshot=current,
                              decision=decision, consecutive_idle=idle, pid=os.getpid())
                write(state_path, record)
                if decision == 'healthy':
                    record['status'] = 'service_goal_query_recovered'
                    record['publication_ok'] = persist_receipt(record)
                    # No resume/set operation: notify only an already active Goal.
                    if goal_status(home, thread) == 'active':
                        message = ('【自动控制恢复通知，非用户新指令】Goal 服务查询已恢复；'
                                   '先 get_goal，非 active 不推进。读取 experiments/control_recovery_receipt.json '
                                   '并补审阅已有完整事件，检查四卡队列。恢复器没有修改 Goal 或数据库。')
                        result = subprocess.run(['codex', 'queue', '--thread', thread, '--message', message],
                                                capture_output=True, timeout=30)
                        record['notification_queued'] = result.returncode == 0
                    write(state_path, record)
                    return
                if decision == 'restart_once':
                    # Revalidate immediately, including our own thread. No active session is exempted.
                    fresh = asyncio.run(asyncio.wait_for(snapshot(home, thread), 15))
                    if action(fresh, idle, False) != 'restart_once' or not detached_workers():
                        idle = 0
                        continue
                    record.update(attempted_restart=True, before_restart=fresh,
                                  restart_at=datetime.now(timezone.utc).isoformat())
                    write(state_path, record)  # Persist before command: never retry a timeout/restart.
                    result = subprocess.run(['codex', 'app-server', 'daemon', 'restart'],
                                            capture_output=True, timeout=60)
                    record['restart_returncode'] = result.returncode
                    write(state_path, record)
            except Exception as error:
                idle = 0
                record.update(last_error=type(error).__name__, heartbeat=datetime.now(timezone.utc).isoformat())
                write(state_path, record)
            time.sleep(30)
        record['status'] = 'bounded_wait_expired_no_further_restart'
        write(state_path, record)
        persist_receipt(record)


if __name__ == '__main__':
    main()
