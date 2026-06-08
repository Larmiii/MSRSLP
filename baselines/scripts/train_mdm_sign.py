"""MDM (Motion Diffusion Model) reproduction on sign data, under SLRTP-canonical.

Faithful to MDM's core: a Transformer that denoises the RAW motion sequence
(no VQ), predicting x0, conditioned on a pooled text embedding with
classifier-free guidance. Text encoder = mBART (de_DE / zh_CN) to match MSR's
text representation (fair). Capacity kept comparable to MSR (~20M, frozen mBART
not counted). HumanML3D-specific machinery not used.

    python train_mdm_sign.py --dataset phix --name mdm_phix --text mbart:de_DE --total_iter 80000
"""
import os, time, math, argparse
from os.path import join as pjoin
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, '.')
from train_t2m_sign import SignT2MDataset, cycle
from sign_text_encoder import build_sign_text_encoder


def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)
    ac = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return torch.clamp(betas, 1e-8, 0.999)


class PositionalEncoding(nn.Module):
    def __init__(self, d, maxlen=400):
        super().__init__()
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(0, maxlen).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MDM(nn.Module):
    def __init__(self, njoints=534, latent=512, ff=1024, layers=8, heads=8, dropout=0.1, text_dim=1024):
        super().__init__()
        self.latent = latent
        self.input_process = nn.Linear(njoints, latent)
        self.pos = PositionalEncoding(latent)
        self.time_mlp = nn.Sequential(nn.Linear(latent, latent), nn.SiLU(), nn.Linear(latent, latent))
        self.embed_text = nn.Linear(text_dim, latent)
        self.null_text = nn.Parameter(torch.zeros(text_dim))
        enc = nn.TransformerEncoderLayer(latent, heads, ff, dropout, activation='gelu', batch_first=True)
        self.seqTransEncoder = nn.TransformerEncoder(enc, layers)
        self.output_process = nn.Linear(latent, njoints)

    def timestep_embed(self, t):
        half = self.latent // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        a = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(a), torch.sin(a)], dim=-1)
        return self.time_mlp(emb)

    def forward(self, x_t, t, text_feat, pad_mask):
        # x_t (B,T,534), t (B,), text_feat (B,text_dim), pad_mask (B,T) True=pad
        cond = self.timestep_embed(t) + self.embed_text(text_feat)        # (B, latent)
        h = self.input_process(x_t)                                      # (B,T,latent)
        h = torch.cat([cond.unsqueeze(1), h], dim=1)                     # prepend cond token
        h = self.pos(h)
        m = torch.cat([torch.zeros(x_t.size(0), 1, dtype=torch.bool, device=x_t.device), pad_mask], dim=1)
        h = self.seqTransEncoder(h, src_key_padding_mask=m)
        return self.output_process(h[:, 1:])                            # x0_hat (B,T,534)


@torch.no_grad()
def validate(model, txt, vq_none, loader, diff, device, cond_drop=0.0):
    model.eval(); tot, n = 0.0, 0
    acp = diff
    for cap, motion, mlen in loader:
        motion = motion.float().to(device); mlen = mlen.long().to(device)
        B, T, _ = motion.shape
        pad = torch.arange(T, device=device)[None] >= mlen[:, None]
        fmask = (~pad).float().unsqueeze(-1)
        t = torch.randint(0, acp.shape[0], (B,), device=device)
        noise = torch.randn_like(motion)
        a = acp[t][:, None, None]
        xt = a.sqrt() * motion + (1 - a).sqrt() * noise
        tf = txt(list(cap))
        x0 = model(xt, t, tf, pad)
        loss = (((x0 - motion) ** 2) * fmask).sum() / (fmask.sum() * x0.size(-1)).clamp(min=1)
        tot += loss.item() * B; n += B
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['phix', 'csl'])
    ap.add_argument('--name', required=True)
    ap.add_argument('--text', default='mbart:de_DE')
    ap.add_argument('--latent', type=int, default=512)
    ap.add_argument('--ff', type=int, default=1024)
    ap.add_argument('--layers', type=int, default=8)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--T_diff', type=int, default=1000)
    ap.add_argument('--max_motion_length', type=int, default=196)
    ap.add_argument('--cond_drop', type=float, default=0.1)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--total_iter', type=int, default=80000)
    ap.add_argument('--warm_up_iter', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--num_workers', type=int, default=6)
    ap.add_argument('--log_every', type=int, default=500)
    ap.add_argument('--val_every', type=int, default=1000)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--min_delta', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=3407)
    opt = ap.parse_args()

    torch.manual_seed(opt.seed); np.random.seed(opt.seed)
    device = torch.device('cuda')
    droot = pjoin('.', 'dataset', f'{opt.dataset}_sign')
    save_dir = pjoin('.', 'sign_ckpt', f'{opt.dataset}_sign', opt.name)
    os.makedirs(save_dir, exist_ok=True)
    mean = np.load(pjoin(droot, 'Mean.npy')); std = np.load(pjoin(droot, 'Std.npy'))

    src_lang = opt.text.split(':')[1] if ':' in opt.text else 'de_DE'
    txt, tdim = build_sign_text_encoder('mbart', device, src_lang=src_lang)

    model = MDM(534, opt.latent, opt.ff, opt.layers, opt.heads, text_dim=tdim).to(device)
    print('MDM trainable params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1e6))

    acp = torch.cumprod(1 - cosine_beta_schedule(opt.T_diff), dim=0).to(device)

    tr = SignT2MDataset(droot, 'train', mean, std, opt.max_motion_length)
    va = SignT2MDataset(droot, 'val', mean, std, opt.max_motion_length)
    loader = cycle(DataLoader(tr, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                              drop_last=True, pin_memory=True, persistent_workers=True))
    val_loader = DataLoader(va, batch_size=opt.batch_size, shuffle=False, num_workers=4, drop_last=False)

    optim = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(0.9, 0.99), weight_decay=1e-5)
    best, best_it, bad = float('inf'), 0, 0
    model.train(); t0 = time.time(); run = 0.0
    for it in range(1, opt.total_iter + 1):
        if it <= opt.warm_up_iter:
            for g in optim.param_groups:
                g['lr'] = opt.lr * it / opt.warm_up_iter
        cap, motion, mlen = next(loader)
        motion = motion.float().to(device); mlen = mlen.long().to(device)
        B, T, _ = motion.shape
        pad = torch.arange(T, device=device)[None] >= mlen[:, None]
        fmask = (~pad).float().unsqueeze(-1)
        t = torch.randint(0, opt.T_diff, (B,), device=device)
        noise = torch.randn_like(motion)
        a = acp[t][:, None, None]
        xt = a.sqrt() * motion + (1 - a).sqrt() * noise
        tf = txt(list(cap))
        drop = torch.rand(B, device=device) < opt.cond_drop
        tf = torch.where(drop[:, None], model.null_text[None], tf)
        x0 = model(xt, t, tf, pad)
        loss = (((x0 - motion) ** 2) * fmask).sum() / (fmask.sum() * x0.size(-1)).clamp(min=1)
        optim.zero_grad(); loss.backward(); optim.step()
        run += loss.item()
        if it % opt.log_every == 0:
            print('iter %6d | loss %.4f | lr %.2e | %.0fs'
                  % (it, run / opt.log_every, optim.param_groups[0]['lr'], time.time() - t0), flush=True)
            run = 0.0
        if it % opt.val_every == 0:
            v = validate(model, txt, None, val_loader, acp, device)
            torch.save({'model': model.state_dict(), 'iter': it, 'opt': vars(opt)}, pjoin(save_dir, 'net_last.tar'))
            if v < best - opt.min_delta:
                best, best_it, bad = v, it, 0
                torch.save({'model': model.state_dict(), 'iter': it, 'val': v, 'opt': vars(opt)},
                           pjoin(save_dir, 'net_best.tar'))
                tag = 'IMPROVED -> net_best'
            else:
                bad += 1; tag = f'no-improve {bad}/{opt.patience} (best {best:.4f}@{best_it})'
            print('  [val] iter %d val_mse %.4f | %s' % (it, v, tag), flush=True)
            if bad >= opt.patience:
                print(f'[EARLY STOP] best val_mse {best:.4f} @ {best_it}', flush=True)
                break
    print(f'[DONE] best val_mse {best:.4f} @ {best_it} -> {save_dir}')


if __name__ == '__main__':
    main()
