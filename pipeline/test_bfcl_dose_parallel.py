"""Resource choices must preserve running training, busy GPUs and four-card bound."""
import unittest
from watch_bfcl_dose_parallel import choose_gpu


class ResourceTests(unittest.TestCase):
    def test_busy_cards_and_training_reservation(self):
        allowed=['4','5','6','7'];evaluation={'mode':'eval'}
        self.assertEqual(choose_gpu(evaluation,[{'gpu':'4'},{'gpu':'5'}],set(allowed),set(),allowed),'6')
        self.assertIsNone(choose_gpu(evaluation,[],{'4'},set(),allowed))
        self.assertEqual(choose_gpu(evaluation,[],{'4'},{'train'},allowed),'4')

    def test_no_foreign_gpu_or_training_migration(self):
        self.assertIsNone(choose_gpu({'mode':'train'},[],{'0','5','6','7'},set(),['4','5','6','7']))
        self.assertIsNone(choose_gpu({'mode':'eval'},[],{'0','1','2','3'},set(),['4','5','6','7']))


if __name__=='__main__':unittest.main(verbosity=2)
