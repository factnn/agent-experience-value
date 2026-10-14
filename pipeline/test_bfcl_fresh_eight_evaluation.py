"""CPU-only fresh phase binding tests; synthetic policies, no scientific labels."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import prepare_bfcl_source_evaluation as original
import prepare_bfcl_source_evaluation_eight as current
from supervise_bfcl_source_evaluation_fresh_eight import validate_manifest
from test_bfcl_ssh_evaluation_control import put


class FreshEightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = r = Path(self.tmp.name)
        self.registry = {'common': {'role': 'baseline'}}
        entries = []
        for i in range(51):
            case = 'case_'+str(i)
            self.registry[case] = {'role': 'fit' if i < 39 else 'heldout'}
            entries.append(put(r, 'jobs/'+case+'.json', {'case_id': case, 'output': 'policies/'+case, 'input_sha256': {}}))
        for case in self.registry:
            folder = 'policies/'+case
            put(r, folder+'/summary.json', {'status': 'complete', 'final_fingerprint': case})
            put(r, folder+'/audit.json', {'status': 'passed'})
            put(r, folder+'/adapter/adapter_config.json', {})
            put(r, folder+'/adapter/adapter_model.safetensors', {})
        for name in current.EVAL_CODE:
            put(r, 'pipeline/'+name, {'synthetic': True})
        put(r, 'training/audit_status.json', {'status': 'passed', 'complete_jobs': 51})
        put(r, 'experiments/index.json', [{'id': 'Eall', 'source': 'training/audit_status.json'}])
        put(r, 'experiments/reviews/Eall.json', {'status': 'reviewed'})
        self.master = put(r, 'master.json', {'jobs': entries, 'output_root': 'training', 'allowed_gpus': [4,5,6,7],
            'required_review_events': [], 'input_sha256': {},
            'analysis_contract': put(r, 'analysis.json', {'source_split': {'heldout': ['synthetic']}}),
            'source_stage0': {'path': 'policies/common', 'policy_fingerprint': 'common'},
            'evaluation': {'task_ids': ['a','b'], 'task_seed_vectors': [[1,2],[3,4]], 'hard_seconds_per_panel': 77400}})
        self.cap = put(r, 'experiments/SSH_FUTURE_GPU_CAP_V3_20261014.json', {'max_simultaneous_gpus_for_subsequent_batches': 8})

    def test_missing_seal_blocks_before_registration_or_output(self):
        r = self.root
        with patch.object(current, 'source_registry', return_value=self.registry):
            with self.assertRaisesRegex(ValueError, 'published actual predictions'):
                current.prepare(r, self.master, r/'reg', r/'out', 'heldout', resource_authorization=self.cap)
        self.assertFalse((r/'reg').exists())
        self.assertFalse((r/'out').exists())

    def test_rejected_seal_cannot_create_registration(self):
        r = self.root
        with patch.object(current, 'source_registry', return_value=self.registry), patch.object(current, 'verify_seal', side_effect=ValueError('unpublished seal')) as guard:
            with self.assertRaisesRegex(ValueError, 'unpublished seal'):
                current.prepare(r, self.master, r/'reg', r/'out', 'heldout', {'path':'not-a-real-seal','sha256':'synthetic'}, self.cap)
        guard.assert_called_once()
        self.assertFalse((r/'reg').exists())
        self.assertFalse((r/'out').exists())

    def test_24_synthetic_panels_keep_science_under_eight_lanes(self):
        r = self.root
        seal = {'path': 'synthetic-only', 'sha256': 'synthetic'}
        # Mock publication only to compare administrative binding. The other
        # tests exercise rejection; this is not an actual published seal test.
        plans = []
        for module, name, cap in [(original, 'old', None), (current, 'new', self.cap)]:
            with patch.object(module, 'source_registry', return_value=self.registry), patch.object(module, 'load_job'), patch.object(module, 'verify_seal'):
                plans.append(module.prepare(r, self.master, r/(name+'reg'), r/(name+'out'), 'heldout', seal, cap))
        before, after = plans
        self.assertEqual(len(after['jobs']), 24)
        self.assertEqual(after['allowed_gpus'], list(range(8)))
        self.assertEqual(after['max_simultaneous_gpus'], 8)
        self.assertEqual(after['hard_seconds_per_panel'], before['hard_seconds_per_panel'])
        self.assertEqual([len(x) for x in after['gpu_lanes'].values()], [3]*8)
        for left, right in zip(before['jobs'], after['jobs']):
            a = json.loads((r/left['path']).read_text()); b = json.loads((r/right['path']).read_text())
            a.pop('output'); b.pop('output')
            old_code = a.pop('input_sha256'); new_code = b.pop('input_sha256')
            self.assertEqual(a, b)
            self.assertTrue(all(new_code[k] == v for k,v in old_code.items()))
            self.assertEqual(len(new_code)-len(old_code), 3)
        with patch('supervise_bfcl_source_evaluation_fresh_eight.source_registry', return_value=self.registry):
            validate_manifest(r, after)
            after['gpu_lanes']['0'].append(after['gpu_lanes']['1'][0])
            with self.assertRaisesRegex(ValueError, 'Lane omission or duplication'):
                validate_manifest(r, after)

    def test_explicit_current_cap_required(self):
        r = self.root
        with self.assertRaisesRegex(ValueError, 'explicit resource authority'):
            current.prepare(r, self.master, r/'reg', r/'out', 'heldout')
        six = put(r, 'experiments/SSH_FUTURE_GPU_CAP_V2_20261013.json', {'max_simultaneous_gpus_for_subsequent_batches': 6})
        with self.assertRaisesRegex(ValueError, 'Unknown evaluation resource amendment'):
            current.prepare(r, self.master, r/'reg', r/'out', 'heldout', resource_authorization=six)


if __name__ == '__main__':
    unittest.main(verbosity=2)
