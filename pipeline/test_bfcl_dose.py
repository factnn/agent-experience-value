"""Dose aliases and transparent snapshot continuation with nonzero Adam moments."""
from pathlib import Path
import tempfile
import unittest
import torch
from bfcl_dose_checkpoint import checked_snapshot, crossed_thresholds, assert_equal, rng_state
from test_rl_learning_state import components, update, PROVENANCE
from rl_learning_state import restore_learning_state


class DoseTests(unittest.TestCase):
    def test_whole_group_aliases_and_no_resave(self):
        thresholds=[10,20,40]
        self.assertEqual(crossed_thresholds(thresholds,set(),0,25),[10,20])
        self.assertEqual(crossed_thresholds(thresholds,{10,20},25,39),[])
        self.assertEqual(crossed_thresholds(thresholds,{10,20},39,45),[40])
        with self.assertRaises(ValueError): crossed_thresholds(thresholds,set(),25,20)

    def test_snapshot_has_no_effect_on_stochastic_nonzero_adam_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'components'
            parts=components()
            for _ in range(2): update(*parts)
            before_rng=rng_state()
            manifest,verification=checked_snapshot(path,*parts,
                {'global_step':2,'at_optimizer_boundary':True},PROVENANCE)
            self.assertEqual(verification['status'],'passed')
            assert_equal(before_rng,rng_state())
            expected=update(*parts)
            model,opt,scheduler,_=components()
            allocator,progress=restore_learning_state(path,model,opt,scheduler,PROVENANCE)
            actual=update(model,opt,scheduler,allocator)
            assert_equal(expected,actual)
            assert_equal(parts[1].state_dict(),opt.state_dict())
            self.assertEqual(progress['global_step'],2)
            self.assertTrue(manifest['trainable_sha256'])


if __name__=='__main__': unittest.main(verbosity=2)
