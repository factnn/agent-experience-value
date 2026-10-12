import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from bfcl_source_eval_contract import policy_role, validate_panel, verify_seal
from report_bfcl_source_eval import audit_panel


class SourceEvalTests(unittest.TestCase):
    def test_source_group_binding_and_panel_cannot_be_relabeled(self):
        analysis = {'source_split': {'fit': ['fit'], 'heldout': ['test']}}
        job = {'case_id': 'case', 'role': 'source_group', 'source_group': 'test', 'prediction_split': 'fit'}
        self.assertEqual(policy_role('case', job, analysis), 'heldout')
        with self.assertRaises(ValueError):
            policy_role('other', job, analysis)
        panel = {'task_ids': ['a', 'b'], 'task_seed_vectors': [[1, 2], [3, 4]], 'rollouts_per_task': 1, 'reserved21_allowed': False}
        evaluation = {'evaluation_repeat': 0, 'task_ids': ['a', 'b'], 'task_seeds': [1, 2]}
        validate_panel(evaluation, panel)
        for change in [{'task_ids': ['a', 'reserved']}, {'task_seeds': [3, 4]}, {'evaluation_repeat': 2}]:
            with self.assertRaises(ValueError):
                validate_panel(evaluation | change, panel)

    def test_actual_git_seal_rejects_changed_predictions_and_early_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(['git', 'init', '-q'], cwd=root, check=True)
            folder = root/'seal';folder.mkdir()
            files = {}
            for name in ['models.json', 'predictions.json', 'fit_labels.json']:
                path = folder/name;path.write_text('{}\n')
                files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            receipt = {'execution_protocol_sha256': 'master', 'analysis_contract_sha256': 'analysis',
                       'heldout_groups': ['test'], 'stage': 0, 'sealed_unix': 10, 'files': files}
            path = folder/'receipt.json';path.write_text(json.dumps(receipt))
            reference = {'path': 'seal/receipt.json', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            subprocess.run(['git', 'add', 'seal'], cwd=root, check=True)
            subprocess.run(['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                            '-c', 'commit.gpgsign=false', 'commit', '-qm', 'Synthetic prediction seal'], cwd=root, check=True)
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
            publication = {'receipt_sha256': reference['sha256'], 'verified_remote_at_unix': 11, 'commit': commit}
            (folder/'publication.json').write_text(json.dumps(publication))
            self.assertEqual(verify_seal(root, reference, 'master', 'analysis', ['test'], check_remote=False), receipt)
            (folder/'predictions.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                verify_seal(root, reference, 'master', 'analysis', ['test'], check_remote=False)
            (folder/'predictions.json').write_text('{}\n')
            (folder/'publication.json').write_text(json.dumps(publication | {'verified_remote_at_unix': 9}))
            with self.assertRaises(ValueError):
                verify_seal(root, reference, 'master', 'analysis', ['test'], check_remote=False)

    def test_panel_raw_reward_seed_and_cost_corruption_rejected(self):
        job = {'policy_case': 'case', 'role': 'fit', 'evaluation_repeat': 0, 'policy_fingerprint': 'fp',
               'task_ids': ['a'], 'task_seeds': [1]}
        evaluation = {'registry': {'a': 'target'}, 'whole_completion_budget': 8192}
        summary = {'status': 'complete', 'weights_unchanged': True, 'optimizer_updates': 0,
                   **{k: job[k] for k in ['policy_case', 'role', 'evaluation_repeat', 'policy_fingerprint']},
                   'completed_tasks': 1, 'sampled_tokens': 2}
        run = summary | {'prediction_seal': None}
        outcomes = [{'task_id': 'a', 'split': 'composition_transfer_eval', 'stratum': 'target', 'seed': 1,
                     'rewards': [1], 'stop_reasons': ['completed'], 'sampled_tokens': 2}]
        rollouts = [{'group': 0, 'rollout_index': 0, 'task_id': 'a', 'seed': 1, 'phase': 'evaluation',
                     'policy_fingerprint': 'fp', 'reward': 1, 'stop_reason': 'completed'}]
        calls = [{'group': 0, 'rollout_index': 0, 'phase': 'evaluation', 'seed': 1, 'generated_ids': [1, 2]}]
        cost = {'rollouts': 1, 'new_model_tokens': 2}
        fixture = job, evaluation, summary, run, outcomes, rollouts, calls, cost
        self.assertEqual(audit_panel(*fixture)['macro_success'], 1)
        for index, key, value in [(4, 'seed', 2), (5, 'reward', 0), (6, 'generated_ids', [1])]:
            altered = copy.deepcopy(fixture);altered[index][0][key] = value
            with self.assertRaises(AssertionError):
                audit_panel(*altered)


if __name__ == '__main__':
    unittest.main()
