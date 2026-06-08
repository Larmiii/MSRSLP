"""Train MoMask's RVQ-VAE tokenizer on sign data (178-kpt -> 534-dim).

Faithful to MoMask: uses MoMask's RVQVAE model + its exact VQ loss
(l1_smooth recon + loss_vel * explicit + commit * commit_loss) + AdamW +
warmup + MultiStepLR. Differences vs official train_vq.py: HumanML3D
FID/R-precision eval scaffolding removed (no sign evaluator); 'explicit'
loss over the full 534-d vector; validation-recon early stopping added.

    python train_vq_sign.py --dataset phix --name momask_vq_phix --num_quantizers 6 \
        --batch_size 1024 --num_workers 8 --val_every 1000 --patience 12
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


class SignMotionDataset(Dataset):
    def __init__(self, data_root, split, mean, std, window_size):
        self.window = window_size
        self.mean, self.std = mean, std
        self.motion_dir = pjoin(data_root, 'new_joint_vecs')
        ids = [l.strip() for l in open(pjoin(data_root, f'{split}.txt'), encoding='utf-8') if l.strip()]
        self.paths = []
        for sid in ids:
            p = pjoin(self.motion_dir, sid + '.npy')
            try:
                L = np.load(p, mmap_mode='r').shape[0]
            except Exception:
                continue
            if L >= window_size:
                self.paths.append(p)
        print(f'[{split}] usable motions: {len(self.paths)} / {len(ids)}')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        m = np.load(self.paths[i])                 # lazy load (avoids per-worker RAM copy)
        s = np.random.randint(0, m.shape[0] - self.window + 1)
        clip = (m[s:s + self.window] - self.mean) / self.std
        return torch.from_numpy(clip).float()


def cycle(dl):
    while True:
        for x in dl:
            yield x


@torch.no_grad()
def validate(net, val_loader, crit, device):
    net.eval()
    tot, n = 0.0, 0
    for x in val_loader:
        x = x.to(device)
        pred = net(x)[0]
        tot += crit(pred, x).item() * x.size(0); n += x.size(0)
    net.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['phix', 'csl'])
    ap.add_argument('--name', required=True)
    ap.add_argument('--num_quantizers', type=int, default=6)
    ap.add_argument('--nb_code', type=int, default=512)
    ap.add_argument('--code_dim', type=int, default=512)
    ap.add_argument('--down_t', type=int, default=2)
    ap.add_argument('--stride_t', type=int, default=2)
    ap.add_argument('--width', type=int, default=512)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--dilation_growth_rate', type=int, default=3)
    ap.add_argument('--window_size', type=int, default=64)
    ap.add_argument('--batch_size', type=int, default=1024)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--total_iter', type=int, default=50000)
    ap.add_argument('--warm_up_iter', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--milestones', type=int, nargs='+', default=[30000, 40000])
    ap.add_argument('--gamma', type=float, default=0.1)
    ap.add_argument('--commit', type=float, default=0.02)
    ap.add_argument('--loss_vel', type=float, default=0.5)
    ap.add_argument('--mu', type=float, default=0.99)
    ap.add_argument('--quantize_dropout_prob', type=float, default=0.2)
    ap.add_argument('--log_every', type=int, default=500)
    # early stopping
    ap.add_argument('--val_every', type=int, default=1000)
    ap.add_argument('--patience', type=int, default=3, help='stop after N evals w/o val improvement')
    ap.add_argument('--min_delta', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=3407)
    opt = ap.parse_args()

    torch.manual_seed(opt.seed); np.random.seed(opt.seed)
    device = torch.device('cuda')
    data_root = pjoin('.', 'dataset', f'{opt.dataset}_sign')
    save_dir = pjoin('.', 'sign_ckpt', f'{opt.dataset}_sign', opt.name)
    os.makedirs(save_dir, exist_ok=True)
    dim_pose = 534

    mean = np.load(pjoin(data_root, 'Mean.npy'))
    std = np.load(pjoin(data_root, 'Std.npy'))

    vq_args = SimpleNamespace(num_quantizers=opt.num_quantizers, shared_codebook=False,
                              quantize_dropout_prob=opt.quantize_dropout_prob,
                              quantize_dropout_cutoff_index=0, mu=opt.mu,
                              nb_code=opt.nb_code, code_dim=opt.code_dim)
    net = RVQVAE(vq_args, dim_pose, opt.nb_code, opt.code_dim, opt.code_dim,
                 opt.down_t, opt.stride_t, opt.width, opt.depth,
                 opt.dilation_growth_rate, 'relu', None).to(device)
    print('RVQVAE params: %.2fM | num_quantizers=%d nb_code=%d batch=%d workers=%d' %
          (sum(p.numel() for p in net.parameters()) / 1e6, opt.num_quantizers,
           opt.nb_code, opt.batch_size, opt.num_workers))

    train_ds = SignMotionDataset(data_root, 'train', mean, std, opt.window_size)
    val_ds = SignMotionDataset(data_root, 'val', mean, std, opt.window_size)
    loader = cycle(DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True,
                              num_workers=opt.num_workers, drop_last=True, pin_memory=True,
                              persistent_workers=True))
    val_loader = DataLoader(val_ds, batch_size=opt.batch_size, shuffle=False,
                            num_workers=4, drop_last=False, pin_memory=True)

    opt_vq = torch.optim.AdamW(net.parameters(), lr=opt.lr, betas=(0.9, 0.99), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt_vq, milestones=opt.milestones, gamma=opt.gamma)
    crit = torch.nn.SmoothL1Loss()

    best_val, best_iter, bad = float('inf'), 0, 0
    net.train()
    t0 = time.time()
    run = {'loss': 0, 'rec': 0, 'commit': 0, 'ppl': 0}
    for it in range(1, opt.total_iter + 1):
        if it <= opt.warm_up_iter:
            for g in opt_vq.param_groups:
                g['lr'] = opt.lr * it / opt.warm_up_iter
        x = next(loader).to(device)
        pred, loss_commit, ppl = net(x)
        loss_rec = crit(pred, x)
        loss = loss_rec + opt.loss_vel * crit(pred, x) + opt.commit * loss_commit
        opt_vq.zero_grad(); loss.backward(); opt_vq.step()
        if it > opt.warm_up_iter:
            sched.step()

        run['loss'] += loss.item(); run['rec'] += loss_rec.item()
        run['commit'] += loss_commit.item(); run['ppl'] += ppl.item()
        if it % opt.log_every == 0:
            n = opt.log_every
            print('iter %6d | loss %.4f rec %.4f commit %.4f ppl %.1f | lr %.2e | %.0fs'
                  % (it, run['loss']/n, run['rec']/n, run['commit']/n, run['ppl']/n,
                     opt_vq.param_groups[0]['lr'], time.time()-t0), flush=True)
            run = {k: 0 for k in run}

        if it % opt.val_every == 0:
            vrec = validate(net, val_loader, crit, device)
            improved = vrec < best_val - opt.min_delta
            torch.save({'vq_model': net.state_dict(), 'iter': it, 'opt': vars(opt)},
                       pjoin(save_dir, 'net_last.tar'))
            if improved:
                best_val, best_iter, bad = vrec, it, 0
                torch.save({'vq_model': net.state_dict(), 'iter': it, 'val_rec': vrec,
                            'opt': vars(opt)}, pjoin(save_dir, 'net_best.tar'))
                tag = 'IMPROVED -> saved net_best'
            else:
                bad += 1
                tag = f'no-improve {bad}/{opt.patience} (best {best_val:.4f}@{best_iter})'
            print('  [val] iter %d val_rec %.4f | %s' % (it, vrec, tag), flush=True)
            if bad >= opt.patience:
                print(f'[EARLY STOP] no val improvement for {opt.patience} evals. '
                      f'best val_rec {best_val:.4f} @ iter {best_iter}.', flush=True)
                break

    print(f'[DONE] best val_rec {best_val:.4f} @ iter {best_iter}. '
          f'net_best.tar + net_last.tar in {save_dir}')


if __name__ == '__main__':
    main()
