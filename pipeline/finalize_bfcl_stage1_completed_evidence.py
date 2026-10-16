"""Publish reviews of completed original Stage1 evidence; never run science."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess

from watch_research_events import record

ROOT = Path(__file__).resolve().parents[1]
REPORT = 'rl/bfcl_source_stage1_forward_report_20261016'
REPORT_SHA = 'fb10d82ad47845bee0bc0dc4788c3eb95ba8606b283326cdc55b8b0294ba510b'
ERROR_SHA = 'be2431be86ca250ffce700b239800b6f6819f513b6986c9b6649526898c3be3b'
POINTER = 'experiments/.runtime/ssh_stage1_forward32_eval_launch_cont1_20261016.json'
SELF = 'pipeline/finalize_bfcl_stage1_completed_evidence.py'


def read(p): return json.loads(p.read_text())
def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()
def ref(p): return {'path': str(p.relative_to(ROOT)), 'sha256': digest(p)}
def pinned(value):
    p = ROOT / value['path']; assert digest(p) == value['sha256'], value['path']; return read(p)
def exclusive(p, value):
    with p.open('x') as handle: json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write('\n')


def review_chain():
    d = read(ROOT / REPORT / 'audit_status.json')
    assert digest(ROOT / REPORT / 'audit_status.json') == REPORT_SHA
    assert digest(ROOT / REPORT / 'error_description.json') == ERROR_SHA
    assert d['raw_replay'] == {'training_jobs': 15, 'target_panels': 32, 'saved_audits_equal_fresh': True}
    assert d['reserved21_accessed'] is False and d['within_late_fitting_performed'] is False
    rt = read(ROOT / POINTER); manifest = pinned(rt['manifest']); authority = pinned(rt['authority'])
    control = ROOT / rt['control_root']; queue = read(control / 'queue.json'); launch = read(control / 'launch.json')
    assert rt['manifest'] == d['evidence']['manifest'] and launch['supervisor_pid'] == rt['pid']
    assert authority['control_root'] == rt['control_root'] and launch['authority'] == rt['authority']
    assert launch['scientific_panels'] == 32 and launch['max_simultaneous_gpus'] == 8
    for path, sha in authority['control_input_sha256'].items(): assert digest(ROOT / path) == sha, path
    assert set(queue['complete']) == {e['name'] for e in manifest['jobs']}
    assert len(queue['complete']) == 32 and all(not queue[k] for k in ['active', 'pending', 'failures'])
    supervisor = read(control / 'supervisor.json'); result = read(control / 'target_control_result.json')
    assert supervisor['status'] == 'audited_complete' and supervisor['complete_cases'] == 32
    assert result['complete_panels'] == 32 and not result['pending_panels'] and not result['classified_failures']
    assert supervisor['automatic_retry'] is False and result['automatic_scientific_retry'] is False
    binding = read(control / 'target_audit_binding.json')
    assert binding['complete_audit'] == d['evidence']['complete_target_audit']
    cert = pinned(binding['complete_audit']); assert cert['complete_panels'] == 32 and cert['status'] == 'passed'
    assert cert['rows'] == d['costs']['evaluation_panels']
    publication = pinned(launch['publication'])
    assert publication['commit'] == rt['publication_commit'] and publication['remote_main'] == publication['commit']
    assert publication['verified_unix'] < launch['started_unix']
    for gpu, preflight in launch['preflights'].items():
        assert preflight['gpu'] == int(gpu) and preflight['no_compute_process'] is True
    idx = read(ROOT / 'experiments/index.json'); receipts = []; pids = []; raw_count = 0; raw_bytes = 0
    for entry, saved in zip(manifest['jobs'], cert['rows']):
        job = pinned(entry); out = ROOT / job['output']; proof = read(out / 'evidence_manifest.json')
        assert proof['original_job'] == entry and proof['original_manifest'] == rt['manifest']
        assert proof['original_control_root'] == rt['control_root'] and proof['fresh_full_raw_audit_matches_saved'] is True
        assert proof['fresh_raw_audit'] == read(out / 'audit.json') == saved
        assert proof['scientific_reruns'] == 0 and proof['new_scientific_tokens_from_review'] == 0
        for name, value in proof['files'].items():
            p = out / name; assert not p.is_symlink() and p.stat().st_size == value['bytes'] and digest(p) == value['sha256'], str(p)
            raw_count += 1; raw_bytes += value['bytes']
        for name, value in proof['controls'].items():
            p = ROOT / name; assert not p.is_symlink() and p.stat().st_size == value['bytes'] and digest(p) == value['sha256'], name
        process = read(control / (entry['name'] + '.console.process.json'))
        terminal = read(out / 'supervisor.json'); run = read(out / 'run.json'); summary = read(out / 'summary.json')
        assert process == proof['original_owned_process'] and terminal == proof['terminal']
        assert run['pid'] == process['pid'] and run['current_eval_authority'] == rt['authority']
        assert terminal['returncode'] == 0 and terminal['hard_timeout'] is False
        assert process['hard_seconds'] == manifest['hard_seconds_per_panel']
        assert terminal['wall_seconds'] <= manifest['hard_seconds_per_panel'] + 30
        assert summary['status'] == 'complete' and summary['completed_tasks'] == 32 and summary['weights_unchanged'] is True
        assert run['optimizer_updates'] == 0
        command = process['command']; assert command[command.index('--job') + 1] == str(ROOT / entry['path'])
        assert command[command.index('--job-sha256') + 1] == entry['sha256']
        matching = [x for x in idx if x['source'] == job['output'] + '/summary.json']; assert len(matching) == 1
        review = read(ROOT / 'experiments/reviews' / (matching[0]['id'] + '.json')); assert review['status'] == 'reviewed'
        receipts.append({'job': entry, 'evidence_manifest': ref(out / 'evidence_manifest.json'), 'review_event': matching[0]['id'], 'pid': process['pid']})
        pids.append(process['pid'])
    assert len(set(pids)) == 32
    prior = []
    for folder in authority.get('prior_admission_control_roots', []):
        path = ROOT / folder; children = list(path.glob('*.console.process.json')); assert not children
        failures = read(path / 'failures.json')
        assert all(x['classification'] == 'publication_dependency_query' and x['scientific_children_launched'] == 0 for x in failures)
        prior.append({'control_root': folder, 'failures': ref(path / 'failures.json'), 'scientific_children': 0, 'scientific_tokens': 0})
    return d, {'status': 'reviewed_complete_original32_execution', 'reviewed_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'reviewer': 'Current SSH execution agent; not independent human review', 'manifest': rt['manifest'], 'authority': rt['authority'],
        'control_root': rt['control_root'], 'controller_pid': rt['pid'], 'controller_terminal': supervisor,
        'queue_counts': {k: len(v) for k, v in queue.items()}, 'complete_target_audit': binding['complete_audit'],
        'primary_science_report': ref(ROOT / REPORT / 'audit_status.json'), 'full_raw_replay': d['raw_replay'],
        'reverified_panel_raw_files': raw_count, 'reverified_panel_raw_bytes': raw_bytes,
        'original32_process_and_review_receipts': receipts, 'prior_admission_only_failures': prior,
        'original_scientific_children': 32, 'scientific_child_failures': 0, 'scientific_reruns': 0,
        'review_new_scientific_tokens': 0, 'actual_evaluation_new_tokens': cert['new_sampled_tokens'],
        'reserved21_accessed': False, 'goal_complete': False}, control


def register(source, kind, facts, conclusion, next_action, cost_description):
    idx = read(ROOT / 'experiments/index.json'); eid = 'E' + str(len(idx) + 1).zfill(3)
    assert not any(x['source'] == source for x in idx), source
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    idx.append({'id': eid, 'source': source, 'kind': kind,
                'sha256': hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest(),
                'facts': facts, 'batch': REPORT, 'observed_at': now})
    exclusive(ROOT / 'experiments/reviews' / (eid + '.json'), {'status': 'reviewed', 'reviewed_at': now,
        'reviewer': 'Current primary SSH execution agent; not independent human review', 'evidence': [source],
        'conclusion': conclusion, 'next_action': next_action, 'new_sampled_tokens': 0,
        'cost_accounting': cost_description, 'goal_complete': False})
    with (ROOT / 'experiments' / (eid + '.md')).open('x') as handle:
        handle.write('# ' + eid + ': ' + kind + '\n\n' + now + '\n\n' + conclusion + '\n\n' + next_action + '\n\n```json\n' + json.dumps(facts, ensure_ascii=False, indent=2) + '\n```\n')
    (ROOT / 'experiments/index.json').write_text(json.dumps(idx, ensure_ascii=False, indent=2) + '\n'); record(ROOT, [])
    return eid


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--publication-recoveries', required=True); args = parser.parse_args()
    assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).strip()
    assert not subprocess.check_output(['git', 'status', '--short', '-uno'], cwd=ROOT, text=True).strip()
    idx = read(ROOT / 'experiments/index.json'); assert idx[-1]['id'] == 'E357'
    execution_path = ROOT / 'rl/bfcl_stage1_forward32_complete_execution_review_20261017.json'
    ledger_path = ROOT / REPORT / 'failure_and_cost_ledger.json'
    assert not execution_path.exists() and not ledger_path.exists()
    d, execution, control = review_chain()
    supplemental = read(ROOT / REPORT / 'post_batch_description.json')
    assert supplemental['source_report']['sha256'] == REPORT_SHA and supplemental['original_primary_results_unchanged'] is True
    render_path = ROOT / REPORT / 'figures/render_metadata.json'; render = read(render_path)
    assert digest(ROOT / REPORT / 'figures/plot_input.json') == render['input_sha256']
    assert read(ROOT / REPORT / 'figures/plot_input.json') == read(ROOT / REPORT / 'plot_data.json')
    assert render['source_report']['sha256'] == REPORT_SHA and render['source_error_description']['sha256'] == ERROR_SHA
    for name, value in render['files'].items():
        p = render_path.parent / name; assert digest(p) == value['sha256'] and p.stat().st_size == value['bytes']
    recoveries_path = ROOT / args.publication_recoveries; recoveries = read(recoveries_path)
    assert recoveries['provenance'] == 'Operator transcription of observed tool results; not retroactive machine receipts'
    ledger = {'scope': 'Stage1 completed 16-source probe + original15 training + original32 evaluation only',
        'costs_by_phase': supplemental['costs_by_phase'], 'actual_new_model_tokens_once': d['costs']['late_total_new_model_tokens_once'],
        'training_nominal_tokens': d['costs']['training_nominal_tokens'], 'training_complete_group_overshoot_tokens': d['costs']['training_complete_group_overshoot_tokens'],
        'shared_probe_charge_count': 1, 'inherited_parent_new_charge': 0,
        'scientific_training_failures': 0, 'scientific_evaluation_failures': 0, 'scientific_retries': 0,
        'admission_dependency_failure': {'event': 'E323', 'scientific_children': 0, 'new_model_tokens': 0,
            'evidence': execution['prior_admission_only_failures'], 'classification': 'pre_inference_publication_dependency_query'},
        'review_failure': {'event': 'E326', 'classification': 'evidence_postprocessing_false_positive', 'new_model_tokens': 0,
            'evidence': ref(ROOT / 'experiments/E326.md'), 'explanation': 'Secret scanner matched its own marker literal; raw original panel evidence retained, only unfrozen review utility repaired, no scientific rerun.'},
        'publication_recoveries': ref(recoveries_path),
        'routine_observer_transport_events': {'path': 'experiments/.runtime/ssh_operational_events.jsonl', 'publication': 'Ignored machine operational log; selected consequential publication failures transcribed separately', 'scientific_restarts': 0},
        'limitations': ['Worker wall includes loading and CPU environment work; not GPU-active time or FLOPs.', 'Nominal token equality is not actual equal computation.', 'Input positions include repeated context; external tokens are not model-generated tokens.', 'Equal policies and seeds produced byte-identical behavioral evidence but both inference runs actually incurred their recorded cost.', 'CPU reports and plotting add zero model sampling; no claim of zero engineering wall time.']}
    exclusive(ledger_path, ledger); execution['classified_failure_and_cost_ledger'] = ref(ledger_path); exclusive(execution_path, execution)
    e1 = register(str(execution_path.relative_to(ROOT)), '完整原32前向评测执行链审阅', execution,
        '原32面板全部单次RC0退出，队列32完成/0活跃/0待执行/0科学故障。逐面板原job、实际归属、原始文件SHA/大小、保存审计、完整重放报告及E325–E357核对一致。E323首次准入故障、E326后处理误报及Git/SSH出版故障单列，0科学重试。',
        '发布完整15/32科学结果与gpt21误差分解；不把审计通过或单批结束当作研究完成。', 'Original32 inference counted in the existing per-panel ledger; this review adds zero model sampling.')
    assert e1 == 'E358'
    report_files = {str(p.relative_to(ROOT)): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted((ROOT / REPORT).rglob('*')) if p.is_file()}
    facts = {'status': 'reviewed_complete_forward15_science', 'primary_report': ref(ROOT / REPORT / 'audit_status.json'),
        'execution_review': ref(execution_path), 'gpt21_error_description': ref(ROOT / REPORT / 'error_description.json'),
        'descriptive_summary': ref(ROOT / REPORT / 'post_batch_description.json'), 'failure_and_cost_ledger': ref(ledger_path),
        'artifacts': report_files, 'facts': supplemental['facts'], 'original_five_MSE': d['prediction_results']['original_five']['MSE'],
        'separate_H1_MSE': d['prediction_results']['separate_H1']['MSE'], 'new_scientific_tokens_from_report': 0,
        'model_refitted': False, 'reserved21_accessed': False, 'original_full_late51_104_complete': False, 'goal_complete': False}
    conclusion = '原15训练/32面板证据链完成：四来源平均G +1.953125pp、V +3.515625pp；Uniform平均G −1.5625pp。冻结任务级Beta点MSE较冻结均值低7.2047%、较组合级低0.1792%，2/4来源改善；原Stage0任务级未胜均值结果保留。晚任务级MSE的91.33%是整体偏差平方，相对均值改善的79.46%来自该分量差，未校正截距或重拟合。两重复政策和同seed行为字节一致，15终点仅14种权重；零新梯度正G不能归因于新经验奖励梯度。实际Stage1生成6,419,587 token，训练完整组超额10.17%。所有模型、不利结果和重复依赖均保留，尚不能建立稳定高分采样规则。'
    next_action = '按完整结果选择下一笔有限科学投入，优先针对完整学习状态/新经验归因、不同机制或关键模型外推缺口；不自动补回晚51/104，不改原确认性结果，21保留题须另行协议与事前预测。原始ID、机制广度、规模/家族/环境及21题目标仍未完成。'
    e2 = register(REPORT + '/audit_status.json', '缩减Stage1完整科学报告与跨状态预测解释', facts, conclusion, next_action,
        'Shared probe + original15 training + original32 evaluation counted once in failure_and_cost_ledger; reporting and plotting add zero sampling.')
    assert e2 == 'E359'
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    next_path = ROOT / 'experiments/NEXT.md'; text = next_path.read_text()
    status = now + '：原Stage0 51/104（E276）及缩减Stage1 15/32（E358/E359）均已完成原始审计和整批分析；当前该批没有运行或待执行科学子进程。Stage1原封存预测、G/V、重复、描述误差分解及真实成本见[完整报告](../' + REPORT + '/README.md)。晚全51/104仍延后，21题关闭，整个Goal未完成。'
    text, count = re.subn(r'(## 当前状态\n\n)[^\n]+', lambda m: m.group(1) + status, text, count=1); assert count == 1
    text = re.sub(r'^当前训练控制为E304[^\n]+', '原15训练控制E304已终结并由E321审阅；原32评测使用E323登记的当前SSH接续归属，已由E358完整审阅。原20＋4预测在新训练/目标前封存并核验发布。所有控制与队列现在作为该批完成证据保留。', text, count=1, flags=re.M)
    next_path.write_text(text + '\n' + now + ' E358/E359：' + conclusion + ' ' + next_action + '\n')
    with (ROOT / 'experiments/SCOPE_REVIEW.md').open('a') as handle:
        handle.write('\n### ' + now + ' E358/E359：完成固定早期映射跨状态前向检验\n\n' + conclusion + '\n\n当前仍直接回答经验干预后的迁移收益及采样前信号信息；成功率、局部机会、真实G/V和预测误差分别报告。原五模型不足以覆盖不同价值机制；本批不是完整阶段机制识别。' + next_action + '\n')
    readme = ROOT / 'README.md'
    readme.write_text('**最新完成结果（' + now + '）：** 原Stage0 51训练/104面板已完成（E276）；缩减Stage1 15训练/32面板亦已完成并审阅（E358/E359），当前该批无运行/排队任务。见[完整科学报告](' + REPORT + '/README.md)、[原始证据与检查点取得](docs/stage1_forward15_reproduction.md)及[NEXT](experiments/NEXT.md)。21保留题关闭，晚全51/104延后，整个研究尚未完成。\n\n以下是注明日期的历史快照，不作当前运行状态。\n\n' + readme.read_text())
    paths = [SELF, 'pipeline/describe_bfcl_stage1_completed_evidence.py', 'pipeline/plot_bfcl_stage1_completed_evidence.py',
             str(execution_path.relative_to(ROOT)), args.publication_recoveries, *report_files,
             d['evidence']['complete_target_audit']['path'],
             *[str((control / name).relative_to(ROOT)) for name in ['launch.json', 'publication_verified.json', 'startup_verified.json', 'supervisor.json', 'queue.json', 'target_control_result.json', 'target_audit_binding.json']],
             'README.md', 'experiments/E358.md', 'experiments/E359.md', 'experiments/reviews/E358.json', 'experiments/reviews/E359.json',
             'experiments/index.json', 'experiments/README.md', 'experiments/events.jsonl', 'experiments/NEXT.md', 'experiments/SCOPE_REVIEW.md']
    paths = list(dict.fromkeys(paths))
    secret = rb'gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|sk-[A-Za-z0-9_-]{30,}'
    for name in paths:
        p = ROOT / name; assert not p.is_symlink() and p.suffix not in ['.pt', '.bin', '.safetensors']
        if p.suffix != '.png': assert not re.search(secret, p.read_bytes()), name
        assert subprocess.run(['git', 'check-ignore', '-q', name], cwd=ROOT).returncode != 0, name
    assert digest(ROOT / REPORT / 'audit_status.json') == REPORT_SHA and digest(ROOT / REPORT / 'error_description.json') == ERROR_SHA
    subprocess.run(['git', 'add', '--', *paths], cwd=ROOT, check=True)
    # csv.writer uses valid CRLF; Matplotlib SVG paths contain harmless trailing spaces.
    # Preserve generated bytes and their recorded SHA values rather than reformat them.
    subprocess.run(['git', '-c', 'core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol',
                    'diff', '--cached', '--check', '--', '.', ':(exclude)' + REPORT + '/figures/*.svg'], cwd=ROOT, check=True)
    subprocess.run(['git', 'commit', '-m', 'Complete Stage1 15-training 32-panel forward evidence with frozen forecast errors and honest dependence accounting'], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    print(json.dumps({'events': [e1, e2], 'complete_panels': 32, 'new_scientific_tokens': 0, 'primary_report_sha256': REPORT_SHA,
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'goal_complete': False}), flush=True)


if __name__ == '__main__': main()
