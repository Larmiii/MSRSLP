"""MoMask tokenizer VQ-ceiling: GT pose -> RVQVAE encode/decode -> SLRTP pickle.

Mirrors MSRSLP's compute_vq_ceiling_bleu.py output format so the existing
slrtp_eval_phix.py can score it. Measures MoMask's residual-VQ tokenizer
representation upper bound (independent of any text->token generator).

    python compute_momask_ceiling.py --dataset phix --name momask_vq_phix --num_quantizers 6 --splits dev,test
"""
import argparse, gzip, pickle, os
from os.path import join as pjoin
from types import SimpleNamespace
import numpy as np
import torch
import sys
sys.path.insert(0, '.')
from models.vq.model import RVQVAE

SRC = {
    'phix': (r'D:\Graduation\MSRSLP\data\phix', 'phix_lift3d.{split}.pt', '{split}.pickle'),
    'csl':  (r'D:\Graduation\MSRSLP\data\csl', 'csl_daily_lift3d.{split}.pt', 'csl_daily.{split}'),
}

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['phix', 'csl'])
    ap.add_argument('--name', required=True)
    ap.add_argument('--splits', default='dev,test')
    ap.add_argument('--num_quantizers', type=int, default=6)
    ap.add_argument('--nb_code', type=int, default=512)
    ap.add_argument('--code_dim', type=int, default=512)
    ap.add_argument('--down_t', type=int, default=2)
    ap.add_argument('--stride_t', type=int, default=2)
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--dilation_growth_rate', type=int, default=3)
    args = ap.parse_args()

    device = torch.device('cuda')
    data_dir, tmpl, pkl_tmpl = SRC[args.dataset]
    droot = pjoin('.', 'dataset', f'{args.dataset}_sign')
    _cdir = pjoin('.', 'sign_ckpt', f'{args.dataset}_sign', args.name)
    ckpt = pjoin(_cdir, 'net_best.tar')
    if not os.path.exists(ckpt):
        ckpt = pjoin(_cdir, 'net_last.tar')
    out_dir = pjoin('.', 'sign_results', f'{args.dataset}_momask_ceil_{args.name}')
    os.makedirs(out_dir, exist_ok=True)

    mean = np.load(pjoin(droot, 'Mean.npy')); std = np.load(pjoin(droot, 'Std.npy'))
    mean_t = torch.from_numpy(mean).float().to(device)
    std_t = torch.from_numpy(std).float().to(device)

    vq_args = SimpleNamespace(num_quantizers=args.num_quantizers, shared_codebook=False,
                              quantize_dropout_prob=0.2, quantize_dropout_cutoff_index=0,
                              mu=0.99, nb_code=args.nb_code, code_dim=args.code_dim)
    net = RVQVAE(vq_args, 534, args.nb_code, args.code_dim, args.code_dim, args.down_t,
                 args.stride_t, args.width, args.depth, args.dilation_growth_rate, 'relu', None).to(device)
    sd = torch.load(ckpt, map_location='cpu')
    net.load_state_dict(sd['vq_model']); net.eval()
    print(f'[*] loaded {ckpt} (iter {sd.get("iter")})')

    for split in args.splits.split(','):
        d = torch.load(pjoin(data_dir, tmpl.format(split=split)), map_location='cpu', weights_only=False)
        out_list = []
        for sid, v in d.items():
            pose = v['poses_3d']
            pose = pose.cpu().numpy() if torch.is_tensor(pose) else np.asarray(pose)
            T = pose.shape[0]
            x = torch.from_numpy(pose.reshape(T, -1).astype(np.float32)).to(device)  # (T,534)
            xn = ((x - mean_t) / std_t).unsqueeze(0)                                  # (1,T,534)
            pred = net(xn)[0].squeeze(0)                                              # (T',534)
            rec = (pred * std_t + mean_t).cpu()                                       # denorm
            out_list.append({'name': sid, 'signer': '', 'gloss': v.get('gloss', ''),
                             'text': v.get('text', ''), 'sign': rec.float()})
        out_path = pjoin(out_dir, pkl_tmpl.format(split=split))
        with gzip.open(out_path, 'wb') as f:
            pickle.dump(out_list, f, protocol=4)
        Ts = [s['sign'].shape[0] for s in out_list]
        print(f'[OK] {split}: {len(out_list)} samples, frames {min(Ts)}/{np.mean(Ts):.1f}/{max(Ts)} -> {out_path}')

if __name__ == '__main__':
    main()
