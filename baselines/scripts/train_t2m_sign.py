"""Train MoMask's masked transformer (text -> base tokens) on sign data.

Faithful to MoMask: MaskTransformer + its internal masked-CE objective.
Differences: HumanML3D FID/R-precision eval removed; text encoder is mBART
(German) instead of English CLIP (fair vs MSRSLP); val-CE early stopping.

    python train_t2m_sign.py --dataset phix --vq_name momask_vq_phix \
        --name momask_mtrans_phix --text mbart:de_DE --total_iter 60000
"""
import os, time, argparse
from os.path import join as pjoin
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, '.')
from models.vq.model import RVQVAE
from models.mask_transformer.transformer import MaskTransformer


class SignT2MDataset(Dataset):
    def __init__(self, data_root, split, mean, std, max_len, unit=4):
        self.mean, self.std, self.max_len, self.unit = mean, std, max_len, unit
        self.motion_dir = pjoin(data_root, 'new_joint_vecs')
        self.text_dir = pjoin(data_root, 'texts')
        ids = [l.strip() for l in open(pjoin(data_root, f'{split}.txt'), encoding='utf-8') if l.strip()]
        self.items = []
        for sid in ids:
            p = pjoin(self.motion_dir, sid + '.npy')
            try:
                L = np.load(p, mmap_mode='r').shape[0]
            except Exception:
                continue
            if L < unit:
                continue
            cap = open(pjoin(self.text_dir, sid + '.txt'), encoding='utf-8').readline().strip()
            self.items.append((p, cap))
        print(f'[{split}] {len(self.items)} / {len(ids)} samples')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, cap = self.items[i]
        m = np.load(p)                              # lazy load (avoids per-worker RAM copy)
        T = m.shape[0]
        if T > self.max_len:
            s = np.random.randint(0, T - self.max_len + 1)
            m = m[s:s + self.max_len]; T = self.max_len
        T = (T // self.unit) * self.unit          # multiple of unit so m_len//4 matches tokens
        m = m[:T]
        m = (m - self.mean) / self.std
        pad = np.zeros((self.max_len - T, m.shape[1]), dtype=np.float32)
        m = np.concatenate([m.astype(np.float32), pad], axis=0)
        return cap, torch.from_numpy(m).float(), T


def cycle(dl):
    while True:
        for b in dl:
            yield b


@torch.no_grad()
def validate(trans, vq, loader, device):
    trans.eval()
    tot, n = 0.0, 0
    for cap, motion, mlen in loader:
        motion = motion.float().to(device); mlen = mlen.long().to(device)
        code_idx, _ = vq.encode(motion)
        loss = trans(code_idx[..., 0], list(cap), mlen // 4)[0]
        tot += loss.item() * len(cap); n += len(cap)
    trans.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['phix', 'csl'])
    ap.add_argument('--vq_name', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--text', default='mbart:de_DE')
    ap.add_argument('--num_quantizers', type=int, default=6)
    ap.add_argument('--nb_code', type=int, default=512)
    ap.add_argument('--code_dim', type=int, default=512)
    ap.add_argument('--latent_dim', type=int, default=384)
    ap.add_argument('--ff_size', type=int, default=1024)
    ap.add_argument('--n_layers', type=int, default=8)
    ap.add_argument('--n_heads', type=int, default=6)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--cond_drop_prob', type=float, default=0.1)
    ap.add_argument('--max_motion_length', type=int, default=196)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--total_iter', type=int, default=60000)
    ap.add_argument('--warm_up_iter', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--milestones', type=int, nargs='+', default=[50000])
    ap.add_argument('--gamma', type=float, default=0.1)
    ap.add_argument('--num_workers', type=int, default=6)
    ap.add_argument('--log_every', type=int, default=500)
    ap.add_argument('--val_every', type=int, default=1000)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--min_delta', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=3407)
    opt = ap.parse_args()

    torch.manual_seed(opt.seed); np.random.seed(opt.seed)
    device = torch.device('cuda')
    data_root = pjoin('.', 'dataset', f'{opt.dataset}_sign')
    save_dir = pjoin('.', 'sign_ckpt', f'{opt.dataset}_sign', opt.name)
    os.makedirs(save_dir, exist_ok=True)
    mean = np.load(pjoin(data_root, 'Mean.npy')); std = np.load(pjoin(data_root, 'Std.npy'))

    # frozen RVQ tokenizer
    vq_args = SimpleNamespace(num_quantizers=opt.num_quantizers, shared_codebook=False,
                              quantize_dropout_prob=0.2, quantize_dropout_cutoff_index=0,
                              mu=0.99, nb_code=opt.nb_code, code_dim=opt.code_dim)
    vq = RVQVAE(vq_args, 534, opt.nb_code, opt.code_dim, opt.code_dim, 2, 2, 512, 3, 3, 'relu', None).to(device)
    vqc = pjoin('.', 'sign_ckpt', f'{opt.dataset}_sign', opt.vq_name, 'net_best.tar')
    vq.load_state_dict(torch.load(vqc, map_location='cpu')['vq_model']); vq.eval()
    for p in vq.parameters():
        p.requires_grad_(False)
    print('[*] loaded VQ', vqc)

    topt = SimpleNamespace(num_tokens=opt.nb_code, device=device)
    trans = MaskTransformer(code_dim=opt.code_dim, cond_mode='text', latent_dim=opt.latent_dim,
                            ff_size=opt.ff_size, num_layers=opt.n_layers, num_heads=opt.n_heads,
                            dropout=opt.dropout, clip_dim=1024, cond_drop_prob=opt.cond_drop_prob,
                            clip_version=opt.text, opt=topt).to(device)
    print('mask-transformer trainable params: %.2fM' %
          (sum(p.numel() for p in trans.parameters_wo_clip()) / 1e6))

    tr = SignT2MDataset(data_root, 'train', mean, std, opt.max_motion_length)
    va = SignT2MDataset(data_root, 'val', mean, std, opt.max_motion_length)
    loader = cycle(DataLoader(tr, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                              drop_last=True, pin_memory=True, persistent_workers=True))
    val_loader = DataLoader(va, batch_size=opt.batch_size, shuffle=False, num_workers=4, drop_last=False)

    optim = torch.optim.AdamW(trans.parameters_wo_clip(), lr=opt.lr, betas=(0.9, 0.99), weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=opt.milestones, gamma=opt.gamma)

    best, best_it, bad = float('inf'), 0, 0
    trans.train()
    t0 = time.time(); run = {'loss': 0, 'acc': 0}
    for it in range(1, opt.total_iter + 1):
        if it <= opt.warm_up_iter:
            for g in optim.param_groups:
                g['lr'] = opt.lr * it / opt.warm_up_iter
        cap, motion, mlen = next(loader)
        motion = motion.float().to(device); mlen = mlen.long().to(device)
        with torch.no_grad():
            code_idx, _ = vq.encode(motion)
        loss, _pred_ids, acc = trans(code_idx[..., 0], list(cap), mlen // 4)
        optim.zero_grad(); loss.backward(); optim.step()
        if it > opt.warm_up_iter:
            sched.step()
        run['loss'] += loss.item(); run['acc'] += float(acc)
        if it % opt.log_every == 0:
            n = opt.log_every
            print('iter %6d | loss %.4f acc %.4f | lr %.2e | %.0fs'
                  % (it, run['loss']/n, run['acc']/n, optim.param_groups[0]['lr'], time.time()-t0), flush=True)
            run = {k: 0 for k in run}
        if it % opt.val_every == 0:
            v = validate(trans, vq, val_loader, device)
            torch.save({'trans': trans.state_dict(), 'iter': it, 'opt': vars(opt)}, pjoin(save_dir, 'net_last.tar'))
            if v < best - opt.min_delta:
                best, best_it, bad = v, it, 0
                torch.save({'trans': trans.state_dict(), 'iter': it, 'val': v, 'opt': vars(opt)},
                           pjoin(save_dir, 'net_best.tar'))
                tag = 'IMPROVED -> net_best'
            else:
                bad += 1; tag = f'no-improve {bad}/{opt.patience} (best {best:.4f}@{best_it})'
            print('  [val] iter %d val_ce %.4f | %s' % (it, v, tag), flush=True)
            if bad >= opt.patience:
                print(f'[EARLY STOP] best val_ce {best:.4f} @ {best_it}', flush=True)
                break
    print(f'[DONE] best val_ce {best:.4f} @ {best_it} -> {save_dir}')


if __name__ == '__main__':
    main()
