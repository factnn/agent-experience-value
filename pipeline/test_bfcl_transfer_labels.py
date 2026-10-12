import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from bfcl_source_eval_contract import digest, verify_seal
from bfcl_transfer_labels import source_registry, aggregate_labels
from seal_bfcl_transfer_predictions import fit_artifacts, seal, record_publication
from prepare_bfcl_source_evaluation import phase_cases, prepare, EVAL_CODE
from supervise_bfcl_source_evaluation import validate_manifest

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')
    return {'path': str(path), 'sha256': digest(path)}


class TransferLabelsTests(unittest.TestCase):
    def fixture(self):
        master = json.loads((ROOT/'rl/BFCL_SOURCE_STAGE0_EXECUTION_PROTOCOL_20261012.json').read_text())
        analysis = json.loads((ROOT/master['analysis_contract']['path']).read_text())
        features = json.loads((ROOT/master['features']['path']).read_text())
        jobs = [json.loads((ROOT/e['path']).read_text()) for e in master['jobs']]
        registry = source_registry(master, jobs, analysis)
        panels = []
        for case, row in registry.items():
            if row['role'] == 'heldout':
                continue
            for e in range(2):
                base = [.25, .5][e]
                r = row['training_repeat']
                # Control always loses .125; source repeat G is -.125, 0, +.125.
                score = base if row['role'] == 'baseline' else base + (-.125 if row['role'] == 'control' else (r-1)*.125)
                panels.append({'policy_case': case, 'evaluation_repeat': e, 'role': row['role'],
                               'status': 'passed', 'macro_success': score,
                               'stratum_success': {s: score for s in set(analysis['target_panel']['registry'].values())}})
        return master, analysis, features, jobs, registry, panels

    def test_pairing_and_complete_cells(self):
        master, analysis, features, jobs, registry, panels = self.fixture()
        labels = aggregate_labels(panels, registry, analysis, 'fit')
        self.assertEqual(labels['panel_count'], 80)
        for row in labels['groups'].values():
            self.assertEqual(row['G'], 0)
            self.assertEqual(row['V'], .125)
            self.assertEqual([r['G'] for r in row['training_repeats']], [-.125, 0, .125])
        models, prediction = fit_artifacts(master, analysis, features, labels)
        self.assertEqual(set(prediction['point_predictions']), set(analysis['source_split']['heldout']))
        self.assertFalse(prediction['heldout_labels_used'])
        for values in prediction['point_predictions'].values():
            self.assertTrue(all(x == 0 for x in values.values()))
        for altered in [panels[:-1], panels+[panels[0]], panels+[panels[0] | {'policy_case': 'unregistered'}]]:
            with self.assertRaises(ValueError):
                aggregate_labels(altered, registry, analysis, 'fit')
        bad = copy.deepcopy(panels)
        bad[0]['stratum_success'].pop(next(iter(bad[0]['stratum_success'])))
        with self.assertRaises(ValueError):
            aggregate_labels(bad, registry, analysis, 'fit')
        with self.assertRaises(ValueError):
            source_registry(master, jobs[:-1]+[jobs[0]], analysis)
        bad_jobs = copy.deepcopy(jobs)
        bad_jobs[1]['training_repeat'] = 1
        with self.assertRaises(ValueError):
            source_registry(master, bad_jobs, analysis)

    def test_prepare_requires_review_and_binds_all_fit_policies(self):
        master, analysis, _, training, _, _ = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def put(relative, value):
                ref = write(root/relative, value)
                ref['path'] = relative
                return ref
            master['analysis_contract'] = put('analysis.json', analysis)
            algorithm = put('algorithm.json', {'synthetic_fixture': True})
            master['input_sha256'] = {}
            master['required_review_events'] = []
            master['jobs'] = []
            for j in training:
                j['input_sha256'] = {}
                j['algorithm_protocol'] = algorithm
                entry = put('training/'+j['case_id']+'.json', j) | {'name': j['case_id']}
                master['jobs'].append(entry)
                put(j['output']+'/summary.json', {'status': 'complete', 'final_fingerprint': 'synthetic',
                    'final_manifest': {'payload_sha256': 'synthetic'}})
                put(j['output']+'/audit.json', {'status': 'passed', 'job_sha256': entry['sha256'],
                    'final_payload_sha256': 'synthetic'})
                put(j['output']+'/adapter/adapter_config.json', {'synthetic_fixture': True})
                put(j['output']+'/adapter/adapter_model.safetensors', {'synthetic_fixture_not_model': True})
            for name in ['summary.json', 'adapter/adapter_config.json', 'adapter/adapter_model.safetensors']:
                put(master['source_stage0']['path']+'/'+name, {'synthetic_fixture': True})
            for name in EVAL_CODE:
                (root/'pipeline').mkdir(exist_ok=True)
                (root/'pipeline'/name).write_bytes((ROOT/'pipeline'/name).read_bytes())
            put(master['output_root']+'/audit_status.json', {'status': 'passed', 'complete_jobs': 51})
            master_ref = put('master.json', master)
            put('experiments/index.json', [{'source': master['output_root']+'/audit_status.json', 'id': 'Efixture'}])
            put('experiments/reviews/Efixture.json', {'status': 'pending'})
            with self.assertRaises(ValueError):
                prepare(root, master_ref, root/'registered', root/'evaluations', 'fit')
            self.assertFalse((root/'registered').exists())
            put('experiments/reviews/Efixture.json', {'status': 'reviewed'})
            plan = prepare(root, master_ref, root/'registered', root/'evaluations', 'fit')
            self.assertEqual(len(plan['jobs']), 80)
            self.assertEqual(plan['required_review_events'], ['Efixture'])
            self.assertTrue(all(len(lane) == 20 for lane in plan['gpu_lanes'].values()))
            self.assertFalse((root/'evaluations').exists())
            validate_manifest(root, plan)
            with self.assertRaises(FileExistsError):
                prepare(root, master_ref, root/'registered', root/'evaluations', 'fit')

    def test_evaluation_manifest_covers_exact_phase_and_unique_gpu_lanes(self):
        master, analysis, _, training, registry, _ = self.fixture()
        self.assertEqual(len(phase_cases(registry, 'fit')), 40)
        self.assertEqual(len(phase_cases(registry, 'heldout')), 12)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def put(relative, value):
                ref = write(root/relative, value)
                ref['path'] = relative
                return ref
            master['analysis_contract'] = put('analysis.json', analysis)
            master['jobs'] = [put('training/'+j['case_id']+'.json', j) | {'name': j['case_id']} for j in training]
            master_ref = put('master.json', master)
            for phase in ['fit', 'heldout']:
                entries = []
                seal_ref = None if phase == 'fit' else {'path': 'synthetic-receipt', 'sha256': 'not-used-by-structural-check'}
                for case in phase_cases(registry, phase):
                    for e in range(2):
                        name = case+'_e'+str(e)
                        job = {'policy_case': case, 'role': registry[case]['role'], 'evaluation_repeat': e,
                               'execution_protocol': master_ref, 'output': phase+'/'+name, 'prediction_seal': seal_ref}
                        entries.append(put('jobs/'+name+'.json', job) | {'name': name})
                plan = {'phase': phase, 'execution_protocol': master_ref, 'jobs': entries,
                        'output_root': phase, 'prediction_seal': seal_ref, 'automatic_retry': False,
                        'max_simultaneous_gpus': 4, 'allowed_gpus': [4, 5, 6, 7],
                        'hard_seconds_per_panel': master['evaluation']['hard_seconds_per_panel'],
                        'gpu_lanes': {str(g): [e['name'] for i, e in enumerate(entries) if i % 4 == g-4] for g in [4, 5, 6, 7]}}
                validate_manifest(root, plan)
                bad = copy.deepcopy(plan)
                bad['jobs'].pop()
                with self.assertRaises(ValueError):
                    validate_manifest(root, bad)
                bad = copy.deepcopy(plan)
                bad['gpu_lanes']['4'].append(bad['gpu_lanes']['5'][0])
                with self.assertRaises(ValueError):
                    validate_manifest(root, bad)
                bad = copy.deepcopy(plan)
                bad['automatic_retry'] = True
                with self.assertRaises(ValueError):
                    validate_manifest(root, bad)
                bad = copy.deepcopy(plan)
                job = json.loads((root/entries[0]['path']).read_text())
                job['role'] = 'heldout' if phase == 'fit' else 'fit'
                bad['jobs'][0] = put('substituted.json', job) | {'name': entries[0]['name']}
                with self.assertRaises(ValueError):
                    validate_manifest(root, bad)

    def test_manifest_audit_seal_and_real_git_publication(self):
        master, analysis, features, training, registry, panels = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'repo'
            root.mkdir()
            def put(relative, value):
                ref = write(root/relative, value)
                ref['path'] = relative
                return ref
            master['analysis_contract'] = put('analysis.json', analysis)
            features['analysis_contract_sha256'] = master['analysis_contract']['sha256']
            master['features'] = put('features.json', features)
            master['jobs'] = [put('training/'+j['case_id']+'.json', j) | {'name': j['case_id']} for j in training]
            master_ref = put('master.json', master)
            entries, audited = [], {}
            for i, row in enumerate(panels):
                job = {'policy_case': row['policy_case'], 'role': row['role'], 'evaluation_repeat': row['evaluation_repeat'],
                       'execution_protocol': master_ref, 'output': 'outputs/'+str(i)}
                entry = put('jobs/'+str(i)+'.json', job)
                audit = row | {'source_files': {'synthetic': 'fixture-only'}, 'cost': {'new_model_tokens': 0}}
                put(job['output']+'/audit.json', audit)
                entries.append(entry)
                audited[root/entry['path']] = audit
            manifest = {'phase': 'fit', 'execution_protocol': master_ref, 'jobs': entries}
            manifest_ref = put('manifest.json', manifest)
            for name in ['bfcl_transfer_labels.py', 'bfcl_transfer_predictor.py', 'seal_bfcl_transfer_predictions.py']:
                (root/'pipeline').mkdir(exist_ok=True)
                (root/'pipeline'/name).write_bytes((ROOT/'pipeline'/name).read_bytes())
            out = root/'sealed'
            # This integration fixture substitutes synthetic audit results only; no real model result is claimed.
            with patch('seal_bfcl_transfer_predictions.audit_job', side_effect=lambda r, p: audited[p]) as auditor:
                receipt = seal(root, root/'manifest.json', manifest_ref['sha256'], out)
                self.assertEqual(auditor.call_count, 80)
            self.assertEqual(len(receipt['heldout_groups']), 4)
            reference = {'path': 'sealed/receipt.json', 'sha256': digest(out/'receipt.json')}
            def git(*args):
                p = subprocess.run(['git', *args], cwd=root, text=True, capture_output=True)
                self.assertEqual(p.returncode, 0, p.stderr)
                return p.stdout.strip()
            git('init', '-b', 'main')
            git('config', 'user.name', 'CPU Fixture')
            git('config', 'user.email', 'fixture@example.invalid')
            git('add', 'sealed')
            git('commit', '-m', 'Synthetic prediction seal')
            commit = git('rev-parse', 'HEAD')
            bare = Path(tmp)/'remote.git'
            p = subprocess.run(['git', 'init', '--bare', str(bare)], capture_output=True)
            self.assertEqual(p.returncode, 0)
            git('remote', 'add', 'origin', str(bare))
            with self.assertRaises(RuntimeError):
                record_publication(root, out/'receipt.json', reference['sha256'], commit)
            self.assertFalse((out/'publication.json').exists())
            git('push', 'origin', 'main')
            publication = record_publication(root, out/'receipt.json', reference['sha256'], commit)
            self.assertEqual(publication['remote_main'], commit)
            verify_seal(root, reference, master_ref['sha256'], master['analysis_contract']['sha256'], receipt['heldout_groups'])
            with self.assertRaises(ValueError):
                record_publication(root, out/'receipt.json', reference['sha256'], commit)
            bad = copy.deepcopy(manifest)
            heldout_case = next(c for c, row in registry.items() if row['role'] == 'heldout')
            bad['jobs'][0] = put('bad_job.json', {'policy_case': heldout_case, 'role': 'fit', 'evaluation_repeat': 0,
                                                'execution_protocol': master_ref})
            badref = put('bad_manifest.json', bad)
            with patch('seal_bfcl_transfer_predictions.audit_job') as auditor:
                with self.assertRaises(ValueError):
                    seal(root, root/'bad_manifest.json', badref['sha256'], root/'bad_sealed')
                auditor.assert_not_called()
            self.assertFalse((root/'bad_sealed').exists())


if __name__ == '__main__':
    unittest.main()
