import copy
import unittest
from bfcl_structural_value_features import normalized_edit_distance,recovery_details,explicit_error,source_features,SIGNALS


def episode(index=0):
    return {'task_id':'train_task','phase':'feature','rollout_index':index,'events':[],
        'grades':[],'total_user_turns':1,'stop_reason':'completed'}


class StructureTests(unittest.TestCase):
    def test_order_and_empty_sequences(self):
        self.assertEqual(normalized_edit_distance([],[]),0)
        self.assertEqual(normalized_edit_distance([],['A.f']),1)
        self.assertEqual(normalized_edit_distance(['A.f','A.g'],['A.g','A.f']),1)

    def test_recovery_respects_order_and_latched_turn(self):
        row=episode();row['events']=[{'turn':0,'call':'A.f()','response':'{"error":"missing"}'},
            {'turn':0,'call':'A.g()','response':'{"result":"ok"}'}]
        row['grades']=[{'turn':0,'valid':True}]
        self.assertEqual(recovery_details(row)['fraction'],1)
        row['events'].reverse()
        self.assertEqual(recovery_details(row)['fraction'],0)
        row['events'].reverse();row['total_user_turns']=2;row['events'][1]['turn']=1
        row['grades']=[{'turn':0,'valid':False},{'turn':1,'valid':True}]
        self.assertEqual(recovery_details(row)['fraction'],0)

    def test_error_marker_is_explicit(self):
        self.assertFalse(explicit_error('{"message":"no error"}'))
        self.assertFalse(explicit_error('{"error":null}'))
        self.assertTrue(explicit_error('Error during execution: invalid'))

    def test_validation_and_missing_draws(self):
        rows=[episode(i) for i in range(4)]
        before=source_features(rows,['train_task'])
        validation=copy.deepcopy(rows[0]);validation['phase']='validation';validation['reward']=10**9
        validation['events']=[{'bad':'validation fields must never be interpreted'}]
        after=source_features(rows+[validation],['train_task'])
        self.assertEqual([before[k] for k in SIGNALS],[after[k] for k in SIGNALS])
        with self.assertRaises(AssertionError):source_features(rows[:3],['train_task'])


if __name__=='__main__':unittest.main()
