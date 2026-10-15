"""Targeted CPU checks of the new forward integration; no science samples or refits."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import bfcl_stage1_forward_bundle as bundle


DEFINITION = 'rl/BFCL_STAGE1_FORWARD_BINDING_20261015.json'


class ForwardIntegration(unittest.TestCase):
    def actual_definition(self):
        reference = bundle.reference(bundle.ROOT, bundle.ROOT/DEFINITION)
        return reference, bundle.load_definition(bundle.ROOT, reference)

    def test_actual_early_frozen_inputs_and_feature_only_H1(self):
        ref, (definition, master, probe, parent, early, analysis) = self.actual_definition()
        five, h1_receipt, h1 = bundle.early_inputs(bundle.ROOT, definition, master, probe, early, analysis)
        self.assertEqual(parent['parent_case'], 's0_uniform_r0')
        self.assertEqual(parent['source_step'], 7)
        self.assertEqual(len(five['heldout_groups']), 4)
        self.assertLess(h1['model']['slope'], 0)
        # In-memory fixture only: every late x is1. The expected forecast still
        # uses the ACTUAL early mean/std, not this constant late fixture mean.
        features = {'stage': 1, 'validation_labels_used': False, 'late_target_labels_used': False,
                    'rows': [{'source_group': group, 'source_success_rate': 1.0} for group in analysis['source_split']['all_groups']]}
        prediction = bundle.h1_predictions(h1, features, analysis)
        model = h1['model']
        expected = model['intercept']+model['slope']*(1-model['early_fit_mean'])/model['early_fit_population_std']
        self.assertEqual(set(prediction['point_predictions']), set(analysis['source_split']['heldout']))
        for value in prediction['point_predictions'].values(): self.assertAlmostEqual(value, expected, places=14)
        self.assertNotAlmostEqual(expected, model['intercept'], places=6)
        self.assertFalse(prediction['renormalized'])
        bad = copy.deepcopy(features); bad['validation_labels_used'] = True
        with self.assertRaises(ValueError): bundle.h1_predictions(h1, bad, analysis)
        bad = copy.deepcopy(features); bad['rows'] = bad['rows'][:-1]
        with self.assertRaises(ValueError): bundle.h1_predictions(h1, bad, analysis)

    def test_actual_incomplete_probe_cannot_create_forecast_output(self):
        ref, (definition, master, probe, parent, early, analysis) = self.actual_definition()
        status = bundle.ROOT/probe['output_root']/'audit_status.json'
        if status.exists() and bundle.read(status).get('complete_cases') == 16:
            self.skipTest('Actual batch now complete; real reviewed seal command must handle it, never this negative test.')
        output = bundle.ROOT/definition['output_root']
        self.assertFalse(output.exists())
        with self.assertRaisesRegex(ValueError, 'All16 late probe'):
            bundle.seal(bundle.ROOT, ref, None, output)
        self.assertFalse(output.exists())

    def test_publication_chronology_and_missing_payload_rejected(self):
        receipt_ref = {'path': 'fixture/receipt.json', 'sha256': 'fixture_sha'}
        receipt = {'sealed_unix': 100}
        files = {'fixture/model.json': 'fixture_model_sha'}
        proof = {'kind': 'stage1_forward_bundle_actual_GitHub_publication', 'receipt_sha256': 'fixture_sha',
                 'files': files, 'verified_remote_at_unix': 120, 'verified_unix': 120}
        bundle.validate_publication(proof, receipt_ref, receipt, files, 130)
        with self.assertRaises(ValueError): bundle.validate_publication(proof, receipt_ref, receipt, files, 119)
        with self.assertRaises(ValueError): bundle.validate_publication(proof, receipt_ref, {'sealed_unix': 121}, files, 130)
        bad = copy.deepcopy(proof); bad['files'] = {}
        with self.assertRaises(ValueError): bundle.validate_publication(bad, receipt_ref, receipt, files, 130)
        bad = copy.deepcopy(proof); bad['verified_unix'] = float('nan')
        with self.assertRaises(ValueError): bundle.validate_publication(bad, receipt_ref, receipt, files, 130)

    def test_every_target_role_routes_through_forward_gate_and_current_stop(self):
        # Routing fixture: publication mathematics/bytes are tested separately.
        # This does not claim an actual late publication or an eligible target run.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'experiments/.runtime').mkdir(parents=True)
            master = {'kind': 'source_stage1_training_and_transfer_execution', 'stage': 1}
            definition = {'global_stop_file': 'experiments/.runtime/global',
                          'evaluation_stop_file': 'experiments/.runtime/eval'}
            for name, value in [('master.json', master), ('definition.json', definition)]:
                (root/name).write_bytes(bundle.encode(value))
            definition_ref = bundle.reference(root, root/'definition.json')
            (root/'receipt.json').write_bytes(bundle.encode({'definition': definition_ref}))
            forward_ref = bundle.reference(root, root/'receipt.json')
            protocol_ref = bundle.reference(root, root/'master.json')
            with patch.object(bundle, 'verify_bundle', return_value={'status': 'routing_fixture_only'}) as verify:
                for role in ('baseline', 'control', 'fit', 'heldout'):
                    job = {'role': role, 'execution_protocol': protocol_ref}
                    with self.assertRaisesRegex(ValueError, 'Every late target panel'):
                        bundle.verify_forward_target_gate(root, job, master, 130)
                    job['forward_bundle'] = forward_ref
                    result = bundle.verify_forward_target_gate(root, job, master, 130)
                    self.assertTrue(result['all_roles_checked'])
                    verify.assert_called_with(root, forward_ref, protocol_ref, 130)
                    stop = root/definition['evaluation_stop_file']; stop.touch()
                    with self.assertRaisesRegex(RuntimeError, 'stop blocks stage1 evaluation'):
                        bundle.verify_forward_target_gate(root, job, master, 130)
                    stop.unlink()


if __name__ == '__main__': unittest.main(verbosity=2)
