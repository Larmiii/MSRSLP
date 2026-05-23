"""Train Residual VQ-VAE on sign-language data (Module 2)."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils.losses as losses_lib
from models.vqvae_residual import ResidualVQVAE
from dataset.dataset_VQ_sign import VQSignDataset


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataname', default='phix', choices=['phix', 'csl', 'phix14t', 'csl_lift3d', 'phix_lift3d'])
    p.add_argument('--exp-name', default='vq_rvq_debug')
    p.add_argument('--out-dir', default='output_sign/')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--window-size', type=int, default=64)
    p.add_argument('--total-iter', type=int, default=100000)
    p.add_argument('--warm-up-iter', type=int, default=1000)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--lr-scheduler', type=int, nargs='+', default=[50000, 80000])
    p.add_argument('--gamma', type=float, default=0.05)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--commit', type=float, default=0.02)
    p.add_argument('--loss-vel', type=float, default=0.1)
    p.add_argument('--recons-loss', default='l2', choices=['l1', 'l1_smooth', 'l2'])
    p.add_argument('--nb-code', type=int, default=512)
    p.add_argument('--nb-code-residual', type=int, default=512)
    p.add_argument('--code-dim', type=int, default=512)
    p.add_argument('--output-emb-width', type=int, default=512)
    p.add_argument('--down-t', type=int, default=2)
    p.add_argument('--stride-t', type=int, default=2)
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--depth', type=int, default=3)
    p.add_argument('--dilation-growth-rate', type=int, default=3)
    p.add_argument('--vq-act', default='relu')
    p.add_argument('--vq-norm', default=None)
    p.add_argument('--quantizer', default='ema_reset',
                    choices=['ema', 'orig', 'ema_reset', 'reset'])
    p.add_argument('--mu', type=float, default=0.99)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--print-iter', type=int, default=500)
    p.add_argument('--eval-iter', type=int, default=2000)
    p.add_argument('--save-iter', type=int, default=20000)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--early-stop-patience', type=int, default=4)
    p.add_argument('--early-stop-min-delta', type=float, default=0.001)
    p.add_argument('--min-iter-before-early-stop', type=int, default=5000)
    return p.parse_args()


def warmup_lr(opt, it, warm_up, max_lr):
    lr = max_lr * (it + 1) / (warm_up + 1)
    for g in opt.param_groups: g['lr'] = lr
    return lr


@torch.no_grad()
def eval_dev_recon(model, loader, device):
    model.eval()
    total = 0.0; total_vq = 0.0; n = 0
    for batch in loader:
        batch = batch.to(device)
        out, vq_loss, _ = model(batch)
        bs = batch.size(0); n += bs
        total += (out - batch).pow(2).mean().item() * bs
        total_vq += vq_loss.item() * bs
    model.train()
    return total / max(n, 1), total_vq / max(n, 1)


def main():
    args = get_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(args.out_dir) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / 'train.log', 'w', encoding='utf-8')
    def log(m): print(m); log_f.write(m + '\n'); log_f.flush()
    log(f'[*] args: {vars(args)}')

    train_set = VQSignDataset(args.dataname, split='train',
                                window_size=args.window_size)
    dev_set = VQSignDataset(args.dataname, split='dev',
                              window_size=args.window_size,
                              mean=train_set.mean, std=train_set.std)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, drop_last=True, pin_memory=True)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    log(f'[*] train: {len(train_set)}, dev: {len(dev_set)}')
    np.save(out_dir / 'mean.npy', train_set.mean)
    np.save(out_dir / 'std.npy', train_set.std)

    model = ResidualVQVAE(
        args, nb_code=args.nb_code, code_dim=args.code_dim,
        output_emb_width=args.output_emb_width,
        down_t=args.down_t, stride_t=args.stride_t,
        width=args.width, depth=args.depth,
        dilation_growth_rate=args.dilation_growth_rate,
        activation=args.vq_act, norm=args.vq_norm,
        nb_code_residual=args.nb_code_residual,
    ).to(args.device)
    log(f'[*] R-VQ-VAE: input_dim={model.input_dim}, '
        f'params={sum(p.numel() for p in model.parameters())/1e6:.2f}M, '
        f'nb_code_base={args.nb_code}, nb_code_residual={args.nb_code_residual}')

    opt = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99),
                       weight_decay=args.weight_decay)
    sch = optim.lr_scheduler.MultiStepLR(opt, milestones=args.lr_scheduler, gamma=args.gamma)
    K = model.input_dim // 3
    Loss = losses_lib.ReConsLoss(args.recons_loss, K)

    it = 0; t0 = time.time()
    best_dev = float('inf'); best_iter = 0; no_improve = 0
    train_iter = iter(train_loader)
    while it < args.total_iter:
        try: batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader); batch = next(train_iter)
        batch = batch.to(args.device)
        cur_lr = warmup_lr(opt, it, args.warm_up_iter, args.lr) \
                  if it < args.warm_up_iter else opt.param_groups[0]['lr']
        out, vq_loss, (ppl_b, ppl_r) = model(batch)
        recon_loss = Loss(out, batch)
        vel_loss = Loss.forward_vel(out, batch)
        loss = recon_loss + args.commit * vq_loss + args.loss_vel * vel_loss
        opt.zero_grad(); loss.backward(); opt.step()
        if it >= args.warm_up_iter: sch.step()
        if it % args.print_iter == 0:
            log(f'iter {it:>6d} | lr {cur_lr:.5f} | recon {recon_loss.item():.4f} | '
                f'vel {vel_loss.item():.4f} | vq {vq_loss.item():.4f} | '
                f'ppl base:{ppl_b.item():.0f} res:{ppl_r.item():.0f} | '
                f'elapsed {time.time()-t0:.0f}s')
        if it > 0 and it % args.eval_iter == 0:
            dev_recon, dev_vq = eval_dev_recon(model, dev_loader, args.device)
            log(f'    >> DEV iter {it} recon={dev_recon:.4f} vq={dev_vq:.4f}')
            improved = dev_recon < best_dev - args.early_stop_min_delta
            if improved:
                best_dev = dev_recon; best_iter = it; no_improve = 0
                torch.save({'model': model.state_dict(), 'iter': it,
                              'args': vars(args), 'dev_recon': dev_recon},
                             out_dir / 'best.pt')
                log(f'    >> SAVED best @ dev recon {best_dev:.4f} (iter {best_iter})')
            else:
                no_improve += 1
                log(f'    >> no improvement ({no_improve}/{args.early_stop_patience}), '
                    f'best still {best_dev:.4f} @ iter {best_iter}')
                if (it >= args.min_iter_before_early_stop and
                        no_improve >= args.early_stop_patience):
                    log(f'    >> EARLY STOP at iter {it}'); break
        if it > 0 and it % args.save_iter == 0:
            torch.save({'model': model.state_dict(), 'iter': it, 'args': vars(args)},
                        out_dir / f'iter{it}.pt')
        it += 1
    torch.save({'model': model.state_dict(), 'iter': it, 'args': vars(args)},
                out_dir / 'final.pt')
    log(f'[OK] done in {(time.time()-t0)/60:.1f} min, best dev recon {best_dev:.4f}')
    log_f.close()


if __name__ == '__main__':
    main()
