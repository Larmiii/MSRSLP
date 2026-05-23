"""Train Multi-Stream VQ-VAE on sign-language data (Module 1).

Same losses as train_vq_sign.py (recon + vq + velocity), but uses
MultiStreamVQVAE (body/hand/face independent encoders + codebooks).

Usage:
    python train_vq_sign_ms.py --dataname phix --exp-name vq_phix_ms \
        --batch-size 128 --nb-code 256 --code-dim 256 --width 256 --total-iter 100000
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils.losses as losses_lib
from models.vqvae_multistream import MultiStreamVQVAE
from dataset.dataset_VQ_sign import VQSignDataset


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataname', type=str, default='phix', choices=['phix', 'csl', 'phix14t', 'csl_lift3d', 'phix_lift3d'])
    p.add_argument('--exp-name', type=str, default='vq_ms_debug')
    p.add_argument('--out-dir', type=str, default='output_sign/')

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
    p.add_argument('--recons-loss', type=str, default='l2', choices=['l1', 'l1_smooth', 'l2'])
    p.add_argument('--hand-weight', type=float, default=2.0,
                    help='hand stream recon loss multiplier (GAR-BT convention)')

    # VQ-VAE (per-stream)
    p.add_argument('--nb-code', type=int, default=256, help='uniform per-stream when *_body/hand/face not set')
    p.add_argument('--nb-code-body', type=int, default=None)
    p.add_argument('--nb-code-hand', type=int, default=None)
    p.add_argument('--nb-code-face', type=int, default=None)
    p.add_argument('--code-dim', type=int, default=256)
    p.add_argument('--output-emb-width', type=int, default=256)
    p.add_argument('--down-t', type=int, default=2)
    p.add_argument('--stride-t', type=int, default=2)
    p.add_argument('--width', type=int, default=256)
    p.add_argument('--depth', type=int, default=3)
    p.add_argument('--dilation-growth-rate', type=int, default=3)
    p.add_argument('--vq-act', type=str, default='relu')
    p.add_argument('--vq-norm', type=str, default=None)
    p.add_argument('--quantizer', type=str, default='ema_reset',
                    choices=['ema', 'orig', 'ema_reset', 'reset'])
    p.add_argument('--mu', type=float, default=0.99)
    p.add_argument('--beta', type=float, default=1.0)

    # Logging
    p.add_argument('--print-iter', type=int, default=500)
    p.add_argument('--eval-iter', type=int, default=2000)
    p.add_argument('--save-iter', type=int, default=20000)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    p.add_argument('--early-stop-patience', type=int, default=4)
    p.add_argument('--early-stop-min-delta', type=float, default=0.001)
    p.add_argument('--min-iter-before-early-stop', type=int, default=5000)
    return p.parse_args()


def warmup_lr(opt, it, warm_up, max_lr):
    lr = max_lr * (it + 1) / (warm_up + 1)
    for g in opt.param_groups:
        g['lr'] = lr
    return lr


@torch.no_grad()
def eval_dev_recon(model, loader, device):
    model.eval()
    total = 0.0; total_vq = 0.0; n = 0
    for batch in loader:
        batch = batch.to(device)
        out, vq_loss, _ = model(batch)
        recon = (out - batch).pow(2).mean()
        bs = batch.size(0); n += bs
        total += recon.item() * bs
        total_vq += vq_loss.item() * bs
    model.train()
    return total / max(n, 1), total_vq / max(n, 1)


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'train.log'
    log_f = open(log_path, 'w', encoding='utf-8')
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

    # Optional asymmetric per-stream codebooks
    stream_codes = None
    if any(getattr(args, f'nb_code_{n}', None) is not None for n in ['body', 'hand', 'face']):
        stream_codes = {
            'body': args.nb_code_body or args.nb_code,
            'hand': args.nb_code_hand or args.nb_code,
            'face': args.nb_code_face or args.nb_code,
        }
        log(f'[*] asymmetric stream_codes: {stream_codes}')
    model = MultiStreamVQVAE(
        args, dataset_name=args.dataname,
        nb_code=args.nb_code, code_dim=args.code_dim,
        stream_codes=stream_codes,
        output_emb_width=args.output_emb_width,
        down_t=args.down_t, stride_t=args.stride_t,
        width=args.width, depth=args.depth,
        dilation_growth_rate=args.dilation_growth_rate,
        activation=args.vq_act, norm=args.vq_norm,
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log(f'[*] Multi-Stream VQ-VAE: streams={list(model.stream_dims.keys())}, '
        f'dims={model.stream_dims}, total params={n_params:.2f}M')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99),
                              weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=args.lr_scheduler,
                                                 gamma=args.gamma)
    K_total = model.input_dim // 3
    Loss = losses_lib.ReConsLoss(args.recons_loss, K_total)

    # Pre-compute keypoint slice for hand-weight (in flat-coordinate space)
    from models.vqvae_multistream import KP_SPLITS
    hand_start, hand_end = KP_SPLITS[args.dataname]['hand']
    hand_flat_start, hand_flat_end = hand_start * 3, hand_end * 3

    it = 0
    t0 = time.time()
    best_dev_recon = float('inf'); best_iter = 0; no_improve = 0
    train_iter = iter(train_loader)
    while it < args.total_iter:
        try: batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader); batch = next(train_iter)
        batch = batch.to(args.device)

        cur_lr = warmup_lr(optimizer, it, args.warm_up_iter, args.lr) \
                  if it < args.warm_up_iter else optimizer.param_groups[0]['lr']

        out, vq_loss, ppls = model(batch)
        recon_loss = Loss(out, batch)
        vel_loss = Loss.forward_vel(out, batch)
        # Extra hand weight (concentrated on hand keypoints)
        hand_extra = ((out[..., hand_flat_start:hand_flat_end]
                        - batch[..., hand_flat_start:hand_flat_end])
                        .pow(2).mean()) * (args.hand_weight - 1.0)

        loss = recon_loss + args.commit * vq_loss + args.loss_vel * vel_loss + hand_extra

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if it >= args.warm_up_iter: scheduler.step()

        if it % args.print_iter == 0:
            ppl_str = ' '.join(f'{n}:{p.item():.0f}' for n, p in ppls.items())
            log(f'iter {it:>6d} | lr {cur_lr:.5f} | recon {recon_loss.item():.4f} | '
                f'hand+ {hand_extra.item():.4f} | vel {vel_loss.item():.4f} | '
                f'vq {vq_loss.item():.4f} | ppl {ppl_str} | '
                f'elapsed {time.time()-t0:.0f}s')

        if it > 0 and it % args.eval_iter == 0:
            dev_recon, dev_vq = eval_dev_recon(model, dev_loader, args.device)
            log(f'    >> DEV iter {it} recon={dev_recon:.4f} vq={dev_vq:.4f}')
            improved = dev_recon < best_dev_recon - args.early_stop_min_delta
            if improved:
                best_dev_recon = dev_recon; best_iter = it; no_improve = 0
                torch.save({'model': model.state_dict(), 'iter': it,
                              'args': vars(args), 'dev_recon': dev_recon},
                             out_dir / 'best.pt')
                log(f'    >> SAVED best @ dev recon {best_dev_recon:.4f} (iter {best_iter})')
            else:
                no_improve += 1
                log(f'    >> no improvement ({no_improve}/{args.early_stop_patience}), '
                    f'best still {best_dev_recon:.4f} @ iter {best_iter}')
                if (it >= args.min_iter_before_early_stop and
                        no_improve >= args.early_stop_patience):
                    log(f'    >> EARLY STOP at iter {it}')
                    break

        if it > 0 and it % args.save_iter == 0:
            torch.save({'model': model.state_dict(), 'iter': it, 'args': vars(args)},
                        out_dir / f'iter{it}.pt')

        it += 1

    torch.save({'model': model.state_dict(), 'iter': it, 'args': vars(args)},
                out_dir / 'final.pt')
    log(f'[OK] done in {(time.time()-t0)/60:.1f} min, best dev recon {best_dev_recon:.4f}')
    log_f.close()


if __name__ == '__main__':
    main()
