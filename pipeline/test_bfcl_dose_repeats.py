"""Queue acceptance: no duplicate sampling, incomplete state or seed mixing."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import watch_bfcl_dose_repeats as scheduler


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.root_patch=patch.object(scheduler.w,'ROOT',self.root);self.root_patch.start()
        self.out_patch=patch.object(scheduler.w,'OUT',self.root/'new');self.out_patch.start()
        old=self.root/'old/train';old.mkdir(parents=True)
        (old/'dose_checkpoints.json').write_text('[]')
        self.p={'source_common_path':'source','previous_wave':{'path':'old'},
            'training_repeats':[{'name':f'repeat_{i}','protocol':{'path':f'train_{i}','sha256':str(i)}} for i in range(1,4)],
            'evaluation_repeats':[{'name':'e0','protocol':{'path':'e0','sha256':'0'}},
                                  {'name':'e1','protocol':{'path':'e1','sha256':'1'}}]}
    def tearDown(self):
        self.out_patch.stop();self.root_patch.stop();self.temp.cleanup()
    def test_four_initial_jobs_and_completed_filter(self):
        jobs=scheduler.plan(self.p)
        self.assertEqual([j['mode'] for j in jobs],['train','train','train','eval'])
        remaining=scheduler.available_jobs(self.p,{'train_repeat_1'},[{'name':'train_repeat_2'}])
        self.assertEqual([j['name'] for j in remaining],['train_repeat_3','eval_common_e1'])
    def test_multi_label_snapshot_evaluated_once_per_seed(self):
        train=self.root/'new/train_repeat_1';train.mkdir(parents=True)
        policy=self.root/'new/train_repeat_1/doses/step_002';policy.mkdir(parents=True)
        (policy/'summary.json').write_text('{"status":"complete"}')
        labels=[{'policy':str(policy.relative_to(self.root)),'optimizer_steps':2,'threshold':v} for v in [32768,65536]]
        (train/'dose_checkpoints.json').write_text(json.dumps(labels))
        jobs=scheduler.plan(self.p)[4:]
        self.assertEqual(len(jobs),2)
        self.assertEqual([j['protocol']['path'] for j in jobs],['e0','e1'])
        (policy/'summary.json').write_text('{"status":"incomplete"}')
        with self.assertRaises(AssertionError):scheduler.plan(self.p)


if __name__=='__main__':unittest.main(verbosity=2)
