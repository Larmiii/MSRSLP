"""CPU objective/grammar checks without trained weights."""
import json,unittest
from pathlib import Path
import numpy as np
import torch
import grouped_mask as s

class Tests(unittest.TestCase):
    def test_group_legality_and_complete_length(self):
        p=json.loads(Path(__file__).with_name('config.json').read_text())
        good=np.cumsum([0]+p['codebook_sizes'][:-1]);s.validate_tokens(good,p)
        with self.assertRaises(ValueError):s.validate_tokens(good[:-1],p)
        bad=good.copy();bad[1]=0
        with self.assertRaises(ValueError):s.validate_tokens(bad,p)
        self.assertEqual(s.fixed_length(-100,p)[0],6)
        self.assertEqual(s.fixed_length(701,p)[0],480)
        with self.assertRaises(ValueError):s.fixed_length(float('nan'),p)

    def test_truncated_length_and_uncorrupted_tokens_are_masked(self):
        pred=torch.tensor([0.,100.],requires_grad=True)
        loss=s.length_target_loss(pred,torch.tensor([6,480]),torch.tensor([True,False]));loss.backward()
        self.assertEqual(float(pred.grad[1]),0.)
        logits=torch.tensor([[[1.,2.],[100.,-100.]]],requires_grad=True)
        loss=s.masked_ce(logits,torch.tensor([[0,1]]),torch.tensor([[True,False]]));loss.backward()
        self.assertTrue(torch.equal(logits.grad[:,1],torch.zeros_like(logits.grad[:,1])))

if __name__=='__main__':unittest.main()
