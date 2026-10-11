import unittest
import json
from pathlib import Path
import tempfile
from bfcl_stage_fork import validate_boundary, continuation_seed, learner_position


class StageForkTests(unittest.TestCase):
    def test_existing_step6_next_three_source_seeds_and_positions(self):
        progress = dict(global_step=6, branch_local_optimizer_steps=6,
                        cumulative_learner_optimizer_steps=6, micro_step=24,
                        at_optimizer_boundary=True, fresh_rollout_on_next_group=True)
        step = validate_boundary(progress, dict(num_generations=4, num_iterations=1))
        self.assertEqual([continuation_seed(20266210, step, n) for n in range(3)],
                         [20266810, 20266910, 20267010])
        self.assertEqual(learner_position(6, 3), dict(branch_local_optimizer_steps=3,
                         cumulative_learner_optimizer_steps=9, micro_step=36))
        for changes in ({'micro_step': 25}, {'fresh_rollout_on_next_group': False},
                        {'at_optimizer_boundary': False}, {'global_step': 0}):
            with self.assertRaises(ValueError):
                validate_boundary(progress | changes, dict(num_generations=4, num_iterations=1))
        with self.assertRaises(ValueError):
            validate_boundary(progress, dict(num_generations=4, num_iterations=2))

    def test_historical_report_fixture_and_corruption(self):
        # A report test made from existing evidence, never represented as new GPU replay.
        import report_bfcl_stage_fork as r
        root = Path(__file__).resolve().parents[1]
        a = json.loads((root / 'rl/BFCL_STAGE_FORK_ACCEPTANCE_PROTOCOL_20261011.json').read_text())
        source = root / a['reference_training']
        if not (source / 'rollouts.jsonl').exists():
            self.skipTest('Archived dose evidence unavailable')
        start, count = a['source_step'], a['replay_updates']
        manifest = r.read(root / a['source_checkpoint'] / 'components/manifest.json')
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            protocol = a | {'output': directory}
            total = 0
            for name in ['allocation', 'groups', 'rollouts', 'generation_calls', 'updates', 'metrics']:
                index = 'step' if name in ['updates', 'metrics'] else 'policy_step'
                shift = 1 if index == 'step' else 0
                selected = [row.copy() for row in r.lines(source / (name + '.jsonl'))
                            if start + shift <= row[index] < start + count + shift]
                for row in selected:
                    row[index] -= start
                    if name == 'allocation':
                        row['cumulative_policy_step'] = row[index] + start
                        total += row['sampled_tokens']
                        row['branch_cost_after_group'] = total
                (out / (name + '.jsonl')).write_text(''.join(json.dumps(x) + '\n' for x in selected))
            restored = {'source_step': start, 'local_step': 0, 'source_payload_sha256': a['source_payload_sha256'],
                        'optimizer_nonzero_moments': True, 'fresh_rollout_buffer': True,
                        'allocator_reset': False, 'inherited_tokens': manifest['cumulative_acquisition_tokens']}
            summary = {'status': 'complete', **learner_position(start, count),
                       'inherited_tokens': manifest['cumulative_acquisition_tokens'],
                       'new_sampled_tokens': total,
                       'final_fingerprint': r.lines(source / 'updates.jsonl')[start + count - 1]['after_sha256']}
            for name, data in [('restored', restored), ('summary', summary)]:
                (out / (name + '.json')).write_text(json.dumps(data))
            self.assertEqual(r.audit(protocol)['status'], 'passed')
            rows = r.lines(out / 'allocation.jsonl')
            rows[0]['seed'] += 1
            (out / 'allocation.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
            result = r.audit(protocol)
            self.assertEqual(result['status'], 'mismatch_requires_review')
            self.assertTrue(any('seed' in x.get('fields', []) for x in result['differences']))
            summary['inherited_tokens'] = 0
            (out / 'summary.json').write_text(json.dumps(summary))
            with self.assertRaises(AssertionError):
                r.audit(protocol)


if __name__ == '__main__':
    unittest.main()
