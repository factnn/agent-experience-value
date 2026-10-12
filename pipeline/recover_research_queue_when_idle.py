"""Bounded queue-storage recovery; never resume Goal or access its database."""
import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from recover_codex_goal_when_idle import snapshot, incident_paths, persist_receipt
from research_queue import enqueue
from watch_research_events import ROOT, now, read, write


def eligible(state, delivery, consecutive_idle, attempted):
    return (state.get('goal_query_ok') and state.get('goal_status') == 'active'
            and state.get('all_idle') and consecutive_idle >= 2 and not attempted
            and delivery.get('error') == 'queue_storage_sqlite_522')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--incident', required=True)
    args = parser.parse_args()
    incident_paths(args.incident)  # Validate explicit incident name; preserve older attempts.
    runtime = ROOT / 'experiments/.runtime'
    path = runtime / ('queue_recovery_' + args.incident + '.json')
    receipt = 'experiments/queue_recovery_' + args.incident + '_receipt.json'
    home, thread = Path(os.environ['CODEX_HOME']), os.environ['CODEX_THREAD_ID']
    record = read(path, {'started_epoch': time.time(), 'attempted_restart': False, 'incident': args.incident})
    message = ('【自动控制恢复通知，非用户新指令】监控消息队列已可投递。先get_goal，非active停止；'
               '读取experiments/NEXT.md与' + receipt + '，按原Goal继续来源组RL研究。'
               '本通知不恢复Goal、不增加授权、不重跑已完成探测。')
    with (runtime / 'control_recovery.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        idle = 0
        while time.time() - record['started_epoch'] < 21600:
            try:
                current = asyncio.run(asyncio.wait_for(snapshot(home, thread, include_sessions=True), 15))
                idle = idle + 1 if current.get('all_idle') else 0
                record.update(heartbeat=now(), pid=os.getpid(), last_snapshot=current, consecutive_idle=idle)
                if current.get('goal_query_ok') and current.get('goal_status') != 'active':
                    record['status'] = 'stopped_goal_not_active'
                    break
                # Do not queue into an active turn, or restart any active/unknown session.
                if current.get('goal_status') == 'active' and current.get('all_idle'):
                    delivery = enqueue(thread, message)
                    record['last_delivery'] = delivery
                    if delivery['queued']:
                        record['status'] = 'queue_delivery_recovered'
                        break
                    if delivery.get('error') == 'timeout_delivery_unknown':
                        record['status'] = 'delivery_unknown_no_duplicate_probe'
                        break
                    if eligible(current, delivery, idle, record['attempted_restart']):
                        fresh = asyncio.run(asyncio.wait_for(snapshot(home, thread, include_sessions=True), 15))
                        if eligible(fresh, delivery, idle, record['attempted_restart']):
                            record.update(attempted_restart=True, before_restart=fresh, restart_at=now())
                            write(path, record)  # Persist before command; never retry restart.
                            result = subprocess.run(['codex', 'app-server', 'daemon', 'restart'],
                                                    capture_output=True, timeout=60)
                            record['restart_returncode'] = result.returncode
                write(path, record)
            except Exception as error:
                idle = 0
                record.update(last_error=type(error).__name__, heartbeat=now())
                write(path, record)
            time.sleep(30)
        else:
            record['status'] = 'bounded_wait_expired'
        write(path, record)
        persist_receipt(record, receipt)


if __name__ == '__main__':
    main()
