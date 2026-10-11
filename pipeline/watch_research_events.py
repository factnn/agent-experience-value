"""Record terminal research events and wake the owning Codex Goal; stdlib only.

This observer never launches experiments or changes their frozen protocols.
The Goal database is read-only; lifecycle changes belong to Codex/user controls.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    value = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, indent=2) + '\n'
    if path.exists() and path.read_text() == value:
        return
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(value)
    tmp.replace(path)


def goal_status(home, thread):
    try:
        with sqlite3.connect(f'file:{home}/goals_1.sqlite?mode=ro', uri=True, timeout=2) as db:
            row = db.execute('SELECT status FROM thread_goals WHERE thread_id=?', (thread,)).fetchone()
            return row[0] if row else 'missing'
    except sqlite3.Error:
        return 'unknown'


def discover(root, batches):
    for batch in batches:
        folder = root / batch
        paths = set(folder.glob('*/summary.json'))
        paths.update(folder.glob('*/failure.json'))
        paths.update(folder.glob('*/scheduler_failure.json'))
        paths.update(folder / name for name in ('audit_status.json', 'failures.json'))
        for path in sorted(paths):
            data = read(path)
            if data is None:
                continue
            if path.name == 'summary.json' and (not isinstance(data, dict) or data.get('status') != 'complete'):
                continue
            if path.name == 'failures.json' and not data:
                continue
            kind = 'complete' if path.name == 'summary.json' else path.stem
            canonical = json.dumps(data, sort_keys=True).encode()
            yield {'source': str(path.relative_to(root)), 'kind': kind,
                   'sha256': hashlib.sha256(canonical).hexdigest(), 'facts': data,
                   'batch': batch}


def review(root, eid):
    return read(root / 'experiments/reviews' / (eid + '.json'), {})


def record(root, batches):
    directory = root / 'experiments'
    index = directory / 'index.json'
    registry = json.loads(index.read_text()) if index.exists() else []
    if not isinstance(registry, list):
        raise ValueError('Invalid event registry; refuse to renumber history')
    known = {(r['source'], r['sha256']) for r in registry}
    for event in discover(root, batches):
        key = (event['source'], event['sha256'])
        if key in known:
            continue
        event.update(id=f'E{len(registry)+1:03d}', observed_at=now())
        registry.append(event)
        known.add(key)
    # Persist identity before deriving documents, so restart preserves numbering.
    write(directory / 'index.json', registry)
    pending = []
    rows = ['# 实验事件索引', '',
            '编号按首次登记顺序；一条记录是一个作业或批次事件，不是一个独立科学重复。',
            '已有完成作业为追溯登记；发现源文件修订会另登记，旧事实保留。', '',
            '| 记录 | 事件 | 来源 | 分析状态 |', '|---|---|---|---|']
    paths = ['experiments/index.json', 'experiments/README.md', 'experiments/events.jsonl']
    for event in registry:
        eid = event['id']
        path = directory / (eid + '.md')
        if not path.exists():
            facts = json.dumps(event['facts'], ensure_ascii=False, indent=2)
            design = ('当前批次检验剂量曲线的训练/评测方差。遵循 [31](../31_dose_repeats_design.md)；'
                      '三条新轨迹共享有偿祖先，旧发现轨迹单列。'
                      if event['batch'] == 'rl/bfcl_intervention_dose_repeats_20261011'
                      else '读取此批次冻结协议与 NEXT，审阅时补充问题、设计及注册时间。')
            write(path, f'# {eid}: {Path(event["source"]).parent.name} / {event["kind"]}\n\n'
                  f'首次登记：{event["observed_at"]}。\n\n'
                  f'原始证据：[{event["source"]}](../{event["source"]})。\n'
                  f'事实快照 SHA256（排序 JSON）：`{event["sha256"]}`。\n\n'
                  '## 问题与设计\n\n'
                  f'{design}此事件不能独立回答经验价值的迁移预测问题。\n\n'
                  f'## 机器事实\n\n```json\n{facts}\n```\n\n'
                  '## 分析与下一决策\n\n'
                  f'等待 Goal agent 读取原始证据；审阅写入 [reviews/{eid}.json](reviews/{eid}.json)。'
                  'complete 只表示作业声明完成，不代表全量审计通过或科学结论成立。'
                  '固定对照继续执行，不按部分分数删臂。\n')
        checked = review(root, eid).get('status') == 'reviewed'
        if not checked:
            pending.append(eid)
        rows.append(f'| [{eid}]({eid}.md) | {event["kind"]} | `{event["source"]}` | '
                    + ('已分析' if checked else '待分析') + ' |')
        paths.append(str(path.relative_to(root)))
    write(directory / 'README.md', '\n'.join(rows) + '\n')
    write(directory / 'events.jsonl', ''.join(json.dumps(
        {k: event[k] for k in ('id', 'source', 'kind', 'sha256', 'observed_at')},
        ensure_ascii=False) + '\n' for event in registry))
    return registry, pending, paths


def git(root, *args):
    return subprocess.run(['git', *args], cwd=root, capture_output=True, timeout=60)


def publish(root, paths):
    # --only prevents staging/committing concurrent scheduler evidence.
    if git(root, 'add', '--', *paths).returncode:
        return False
    changes = git(root, 'diff', 'HEAD', '--quiet', '--', *paths)
    if changes.returncode == 1:
        if git(root, 'commit', '--only', '-m', 'Record completed research events for adaptive Goal review', '--', *paths).returncode:
            return False
    elif changes.returncode:
        return False
    return git(root, 'push').returncode == 0


def notify(thread, ids):
    message = ('【自动实验监控事件，非用户新的指令】完成/失败事件已写入 '
               '/share/project/zpy/RL/agent_experience_project_20260928/experiments/：'
               + ', '.join(ids) + '。先调用 get_goal；若非 active，停止本次自动推进，不能恢复 Goal。'
               '若 active，读取 RESEARCH_LOOP.md、NEXT.md、对应 E 文档及原始证据；'
               '更新分析/下一决策、reviews 与 git push。维持已冻结对照，准备或执行有证据支持的下一实验；'
               '不要把单个作业/一批队列结束当作 Goal 完成。此消息仅通知，不增加用户授权。')
    return subprocess.run(['codex', 'queue', '--thread', thread, '--message', message],
                          capture_output=True, timeout=30).returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--no-notify', action='store_true')
    parser.add_argument('--no-publish', action='store_true')
    args = parser.parse_args()
    home = Path(os.environ['CODEX_HOME'])
    thread = os.environ['CODEX_THREAD_ID']
    runtime = ROOT / 'experiments/.runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    lock = (runtime / 'watch.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = read(runtime / 'state.json', {'notified': [], 'publication_pending': False})
    while True:
        status = goal_status(home, thread)
        try:
            if status == 'active':
                batches = read(ROOT / 'experiments/watch_roots.json')
                if not isinstance(batches, list):
                    raise ValueError('Missing watch roots')
                registry, pending, paths = record(ROOT, batches)
                signature = hashlib.sha256((ROOT / 'experiments/README.md').read_bytes()).hexdigest()
                if state.get('signature') != signature:
                    state.update(signature=signature, publication_pending=True)
                if state['publication_pending'] and not args.no_publish:
                    state['publication_pending'] = not publish(ROOT, paths)
                unseen = [eid for eid in pending if eid not in state['notified']]
                # Recheck immediately before queueing, including user pause during publish.
                if unseen and not args.no_notify and goal_status(home, thread) == 'active':
                    if notify(thread, unseen):
                        state['notified'].extend(unseen)
                        state['last_queued_at'] = now()
                state.update(events=len(registry), pending_review=pending, last_error=None)
        except Exception as error:
            # Preserve retry state; never log raw command output/credentials.
            state['last_error'] = type(error).__name__
        state.update(heartbeat=now(), pid=os.getpid(), goal_status=status)
        write(runtime / 'state.json', state)
        if args.once:
            break
        time.sleep(30)
    lock.close()


if __name__ == '__main__':
    main()
