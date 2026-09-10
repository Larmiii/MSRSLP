"""CPU checks for the preserved reference length and token-boundary rules."""
import unittest
from reference_common import *

class Tests(unittest.TestCase):
    def test_native_residual_sentinel(self):
        self.assertEqual(validate_momask_code_groups([[1,-1,2,3,4,5]])['native_zero_residual_count'],1)
        for row in [[-1,1,2,3,4,5],[1,-2,2,3,4,5],[1,True,2,3,4,5]]:
            with self.assertRaises(ValueError):validate_momask_code_groups([row])
    def test_nonempty_complete_group_eos_and_cap(self):
        base=10;codes=[base+i*512 for i in range(6)]
        self.assertNotIn(1,allowed_mt5_tokens([0],base))
        self.assertIn(1,allowed_mt5_tokens([0]+codes,base))
        arr,reason=decode_mt5_ids([0]+codes+[1],base)
        self.assertEqual(arr.shape,(1,6));self.assertEqual(reason,'natural_eos')
        self.assertEqual(allowed_mt5_tokens([0]+codes*80,base),[1])
        with self.assertRaises(ValueError):decode_mt5_ids([0]+codes[:-1]+[1],base)
    def test_fixed_train_length_model(self):
        p=fit_length([{'text':'a b','frames':14},{'text':'a b c d','frames':18}],'phix')
        self.assertEqual(predict_length(p,'a b c d e f','phix'),24)
        p['fit_split']='test'
        with self.assertRaises(ValueError):predict_length(p,'a','phix')

if __name__=='__main__':unittest.main()
