"""MotionGPT(mT5) generation on sign data -> SLRTP pickle.

    python gen_motiongpt_sign.py --dataset phix --vq_name momask_vq_phix --name mgpt_phix --splits dev,test
"""
import argparse, gzip, pickle, os
from os.path import join as pjoin
from types import SimpleNamespace
import numpy as np
import torch
import sys
sys.path.insert(0, '.')
from models.vq.model import RVQVAE

NQ, NB = 6, 512
SRC = {
    'phix': (r'D:\Graduation\MSRSLP\data\phix', 'phix_lift3d.{split}.pt', '{split}.pickle'),
    'csl':  (r'D:\Graduation\MSRSLP\data\csl', 'csl_daily_lift3d.{split}.pt', 'csl_daily.{split}'),
}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['phix', 'csl'])
    ap.add_argument('--vq_name', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--splits', default='dev,test')
    ap.add_argument('--max_motion_length', type=int, default=196)
    ap.add_argument('--batch_size', type=int, default=16)
    args = ap.parse_args()

    device = torch.device('cuda')
    data_dir, tmpl, pkl_tmpl = SRC[args.dataset]
    droot = pjoin('.', 'dataset', f'{args.dataset}_sign')
    best_dir = pjoin('.', 'sign_ckpt', f'{args.dataset}_sign', args.name, 'best')
    out_dir = pjoin('.', 'sign_results', f'{args.dataset}_mgpt_e2e')
    os.makedirs(out_dir, exist_ok=True)
    max_tok = args.max_motion_length // 4

    mean = np.load(pjoin(droot, 'Mean.npy')); std = np.load(pjoin(droot, 'Std.npy'))
    mean_t = torch.from_numpy(mean).float().to(device); std_t = torch.from_numpy(std).float().to(device)

    va = SimpleNamespace(num_quantizers=NQ, shared_codebook=False, quantize_dropout_prob=0.2,
                         quantize_dropout_cutoff_index=0, mu=0.99, nb_code=NB, code_dim=512)
    vq = RVQVAE(va, 534, NB, 512, 512, 2, 2, 512, 3, 3, 'relu', None).to(device)
    vq.load_state_dict(torch.load(pjoin('.', 'sign_ckpt', f'{args.dataset}_sign', args.vq_name, 'net_best.tar'),
                                  map_location='cpu')['vq_model'])
    vq.eval()

    from transformers import AutoTokenizer, MT5ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(best_dir)
    model = MT5ForConditionalGeneration.from_pretrained(best_dir).to(device).eval()
    base_id = len(tok) - NQ * NB
    motion_ids = list(range(base_id, base_id + NQ * NB))
    allowed = set(motion_ids + [tok.eos_token_id, tok.pad_token_id])
    allow_list = list(allowed)

    def prefix_fn(batch_id, input_ids):
        return allow_list

    for split in args.splits.split(','):
        d = torch.load(pjoin(data_dir, tmpl.format(split=split)), map_location='cpu', weights_only=False)
        sids = list(d.keys())
        out_list = []
        for i in range(0, len(sids), args.batch_size):
            chunk = sids[i:i + args.batch_size]
            caps = [(d[s].get('text', '') or '').strip() for s in chunk]
            enc = tok(caps, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
            gen = model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                                 max_length=max_tok * NQ + 2, num_beams=1, do_sample=False,
                                 prefix_allowed_tokens_fn=prefix_fn)
            for k, s in enumerate(chunk):
                ids = [int(x) - base_id for x in gen[k].tolist() if base_id <= int(x) < base_id + NQ * NB]
                Tp = len(ids) // NQ
                if Tp == 0:
                    ids = [0] * NQ; Tp = 1
                ids = ids[:Tp * NQ]
                code = torch.tensor(ids, device=device).view(Tp, NQ) % NB        # (T',6) codes
                pred = vq.forward_decoder(code.unsqueeze(0))                      # (1,T,534) or (1,534,T)
                if pred.shape[-1] != 534:
                    pred = pred.permute(0, 2, 1)
                sign = (pred[0] * std_t + mean_t).cpu().float()
                out_list.append({'name': s, 'signer': '', 'gloss': d[s].get('gloss', ''),
                                 'text': d[s].get('text', ''), 'sign': sign})
        out_path = pjoin(out_dir, pkl_tmpl.format(split=split))
        with gzip.open(out_path, 'wb') as f:
            pickle.dump(out_list, f, protocol=4)
        Ts = [s['sign'].shape[0] for s in out_list]
        print(f'[OK] {split}: {len(out_list)} samples, frames {min(Ts)}/{np.mean(Ts):.1f}/{max(Ts)} -> {out_path}')


if __name__ == '__main__':
    main()
