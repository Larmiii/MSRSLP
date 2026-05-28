"""Stage-2 DISCRETE FLOW MATCHING training (DFM, Gat et al. NeurIPS 2024).

Replaces the AR trans (train_trans_sign_cross.py) with a parallel bidirectional
DFM model (models/t2m_dfm_cross.CrossAttnDFM). Shares text encoder + data
loader with the AR pipeline.

Per-step:
  t ~ LogitNormal(0, 1)
  mask each clean token with prob (1 - t) → MASK_ID
  forward(x_t, t, text_mem) → per-position logits over num_vq classes
  loss = CE on masked positions only
  + auxiliary length-head loss (smooth-L1 on log-length, predict from text)
  text-condition is dropped with prob 0.1 for classifier-free guidance

Usage:
  python train_dfm_sign.py \
    --dataname phix_lift3d \
    --tokens-dir checkpoints/phix/tokens/vq_M1M2_interleaved \
    --vq-ckpt checkpoints/phix/vq/vq_M1M2.pt \
    --exp-name dfm_phix_M1M2 \
    --num-vq 1344 --block-size 320 \
    --num-layers 6 --embed-dim 512 \
    --batch-size 16 --total-iter 30000 \
    --lr 1e-4 --lr-scheduler 15000 25000 --gamma 0.2 \
    --device cuda
"""
from __future__ import annotations
import argparse, os, sys, time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset.dataset_TM_sign import TMSignDataset, collate_tm_sign
from models.t2m_dfm_cross import CrossAttnDFM

LANG_DEFAULTS = {'phix': 'de_DE', 'phix_lift3d': 'de_DE',
                  'csl': 'zh_CN', 'csl_lift3d': 'zh_CN'}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataname', required=True)
    p.add_argument('--tokens-dir', required=True)
    p.add_argument('--vq-ckpt', required=True, help='for sanity check only')
    p.add_argument('--exp-name', default='dfm_debug')
    p.add_argument('--out-dir', default='output_sign/')
    p.add_argument('--mbart-name', default='facebook/mbart-large-50')
    p.add_argument('--lang', default=None)
    p.add_argument('--max-text-len', type=int, default=80)

    p.add_argument('--num-vq', type=int, required=True)
    p.add_argument('--embed-dim', type=int, default=512)
    p.add_argument('--text-dim', type=int, default=1024)
    p.add_argument('--block-size', type=int, default=320)
    p.add_argument('--num-layers', type=int, default=6)
    p.add_argument('--n-head', type=int, default=8)
    p.add_argument('--drop-out-rate', type=float, default=0.1)
    p.add_argument('--fc-rate', type=int, default=4)

    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--total-iter', type=int, default=30000)
    p.add_argument('--warm-up-iter', type=int, default=500)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lr-scheduler', type=int, nargs='+', default=[15000, 25000])
    p.add_argument('--gamma', type=float, default=0.2)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--print-iter', type=int, default=100)
    p.add_argument('--eval-iter', type=int, default=500)
    p.add_argument('--save-iter', type=int, default=2000)
    p.add_argument('--freeze-text', action='store_true', default=True)
    p.add_argument('--cfg-drop-prob', type=float, default=0.1,
                    help='Probability of dropping text condition (replace with empty) for CFG.')
    p.add_argument('--logitnormal-mu', type=float, default=0.0)
    p.add_argument('--logitnormal-sigma', type=float, default=1.0)
    p.add_argument('--lambda-length', type=float, default=0.1)

    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=42)

    p.add_argument('--align-dim', type=int, default=0, help='unused, kept for collate compat')
    p.add_argument('--predict-length', type=int, default=1)
    return p.parse_args()


def warmup_lr(opt, it, warm_up_iter, max_lr):
    lr = max_lr * (it + 1) / (warm_up_iter + 1)
    for g in opt.param_groups:
        g['lr'] = lr
    return lr


@torch.no_grad()
def eval_dev_loss(model, text_enc, loader, device, num_vq, args):
    model.eval()
    total_ce = 0.0; total_len = 0.0; n = 0
    for batch in loader:
        ids = batch['input_ids'].to(device)
        am  = batch['attention_mask'].to(device)
        mt  = batch['motion_tokens'].to(device)            # (B, T) with PAD = num_vq+1
        # Convert AR data convention to DFM:
        # - drop the appended EOS slot at the end if any (AR adds num_vq sentinel)
        # - PAD positions go into pad_mask
        # Original mt has values in [0..num_vq-1] for real tokens, num_vq for BOS/EOS,
        # num_vq+1 for PAD. We treat num_vq and num_vq+1 BOTH as pad for DFM purposes.
        pad_mask = (mt == num_vq + 1) | (mt == num_vq)
        # Replace pad with 0 (any valid id) so embedding works; loss is masked anyway
        clean = torch.where(pad_mask, torch.zeros_like(mt), mt)

        mem = text_enc(input_ids=ids, attention_mask=am).last_hidden_state

        # Sample t and corrupt
        t = model.sample_logit_normal_t(clean.size(0), device,
                                         mu=args.logitnormal_mu, sigma=args.logitnormal_sigma)
        x_t, loss_mask = model.corrupt(clean, t, pad_mask=pad_mask)
        # Forward (no CFG drop at eval)
        logits = model(x_t, t, mem, am, tgt_key_padding_mask=pad_mask)

        # Masked CE
        ce_lin = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                  clean.reshape(-1), reduction='none')
        m = loss_mask.reshape(-1).float()
        if m.sum() > 0:
            ce = (ce_lin * m).sum() / m.sum()
        else:
            ce = torch.tensor(0.0, device=device)
        total_ce += ce.item() * clean.size(0); n += clean.size(0)

        if model.predict_length:
            mem_proj = model.text_norm(model.text_proj(mem))
            mem_kpm = (am == 0)
            log_pred = model.predict_motion_length(mem_proj, mem_kpm)
            gt_len = (batch['motion_len'].to(device).float() - 1.0).clamp(min=1.0)
            log_gt = torch.log(gt_len)
            total_len += F.smooth_l1_loss(log_pred, log_gt).item() * clean.size(0)

    model.train()
    return total_ce / max(n, 1), total_len / max(n, 1)


def main():
    args = get_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.lang is None:
        args.lang = LANG_DEFAULTS[args.dataname]
    out_dir = Path(args.out_dir) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'train.log'
    log_f = open(log_path, 'w', encoding='utf-8')
    def log(m): print(m); log_f.write(m + '\n'); log_f.flush()
    log(f'[*] args: {vars(args)}')

    # mBART text encoder (frozen)
    from transformers import MBart50TokenizerFast, MBartModel
    tokenizer = MBart50TokenizerFast.from_pretrained(args.mbart_name, src_lang=args.lang)
    mbart_full = MBartModel.from_pretrained(args.mbart_name)
    mbart_enc = mbart_full.encoder.to(args.device)
    for p in mbart_enc.parameters():
        p.requires_grad = False
    mbart_enc.eval()
    log('[*] mBART encoder loaded + FROZEN')

    # Datasets: shared with AR training (motion_tokens convention: PAD=num_vq+1)
    tdir = Path(args.tokens_dir)
    train_set = TMSignDataset(tokens_cache_path=str(tdir / 'train_tokens.pt'),
                               tokenizer=tokenizer, num_vq=args.num_vq,
                               max_text_len=args.max_text_len,
                               max_motion_len=args.block_size, lang_code=args.lang)
    dev_set   = TMSignDataset(tokens_cache_path=str(tdir / 'dev_tokens.pt'),
                               tokenizer=tokenizer, num_vq=args.num_vq,
                               max_text_len=args.max_text_len,
                               max_motion_len=args.block_size, lang_code=args.lang)
    collate = partial(collate_tm_sign, motion_pad_id=args.num_vq + 1, gloss_pad_id=None)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, drop_last=True, collate_fn=collate)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate)
    log(f'[*] train: {len(train_set)}, dev: {len(dev_set)}')

    model = CrossAttnDFM(
        num_vq=args.num_vq, text_dim=args.text_dim, embed_dim=args.embed_dim,
        block_size=args.block_size, num_layers=args.num_layers, n_head=args.n_head,
        drop_out_rate=args.drop_out_rate, fc_rate=args.fc_rate,
        predict_length=bool(args.predict_length),
    ).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f'[*] DFM model: {n_params/1e6:.2f}M params')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99),
                              weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler,
                                                 gamma=args.gamma)
    smooth_l1 = nn.SmoothL1Loss()

    # For CFG drop: pre-tokenize an empty/null text
    null_input = tokenizer([""], return_tensors='pt', padding=True,
                            truncation=True, max_length=args.max_text_len)
    null_ids = null_input['input_ids'].to(args.device)
    null_am  = null_input['attention_mask'].to(args.device)
    with torch.no_grad():
        null_mem = mbart_enc(input_ids=null_ids, attention_mask=null_am).last_hidden_state
    log(f'[*] null text memory shape: {null_mem.shape}')

    it = 0; t0 = time.time(); best_dev = float('inf'); best_iter = 0
    train_iter = iter(train_loader)
    while it < args.total_iter:
        try: batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader); batch = next(train_iter)

        ids = batch['input_ids'].to(args.device)
        am  = batch['attention_mask'].to(args.device)
        mt  = batch['motion_tokens'].to(args.device)

        if it < args.warm_up_iter:
            cur_lr = warmup_lr(optimizer, it, args.warm_up_iter, args.lr)
        else:
            cur_lr = optimizer.param_groups[0]['lr']

        with torch.no_grad():
            mem = mbart_enc(input_ids=ids, attention_mask=am).last_hidden_state

        # CFG drop: replace text condition with null with prob cfg_drop_prob
        # Implement at batch level: if drop, use the (B, ?) null memory broadcast.
        # Simpler: per-sample drop via row-wise zeroing of attention mask (so all
        # text tokens are masked out → attention falls back to BOS embedding only).
        # We use null-text replacement per-sample for cleaner semantics.
        B = mt.size(0)
        drop = (torch.rand(B, device=args.device) < args.cfg_drop_prob)
        if drop.any():
            # Pad null_mem and null_am to match mem's T_text
            T_text = mem.size(1)
            T_null = null_mem.size(1)
            if T_null < T_text:
                pad_mem = torch.zeros(1, T_text - T_null, null_mem.size(-1), device=args.device)
                pad_am  = torch.zeros(1, T_text - T_null, dtype=null_am.dtype, device=args.device)
                _null_mem = torch.cat([null_mem, pad_mem], dim=1)
                _null_am  = torch.cat([null_am, pad_am], dim=1)
            else:
                _null_mem = null_mem[:, :T_text]
                _null_am  = null_am[:, :T_text]
            mem = torch.where(drop[:, None, None], _null_mem.expand_as(mem), mem)
            am  = torch.where(drop[:, None], _null_am.expand_as(am), am)

        # Convert to DFM convention (no BOS/PAD in vocab; build pad_mask from sentinels)
        pad_mask = (mt == args.num_vq + 1) | (mt == args.num_vq)
        clean = torch.where(pad_mask, torch.zeros_like(mt), mt)

        # Sample t and corrupt
        t = model.sample_logit_normal_t(B, args.device,
                                         mu=args.logitnormal_mu, sigma=args.logitnormal_sigma)
        x_t, loss_mask = model.corrupt(clean, t, pad_mask=pad_mask)

        # Forward
        logits = model(x_t, t, mem, am, tgt_key_padding_mask=pad_mask)

        # Masked CE
        ce_lin = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                  clean.reshape(-1), reduction='none')
        m = loss_mask.reshape(-1).float()
        if m.sum() > 0:
            ce_loss = (ce_lin * m).sum() / m.sum()
        else:
            ce_loss = torch.tensor(0.0, device=args.device)
        total_loss = ce_loss

        length_val = 0.0
        if model.predict_length:
            mem_proj = model.text_norm(model.text_proj(mem))
            mem_kpm = (am == 0)
            log_pred = model.predict_motion_length(mem_proj, mem_kpm)
            gt_len = (batch['motion_len'].to(args.device).float() - 1.0).clamp(min=1.0)
            log_gt = torch.log(gt_len)
            length_loss = smooth_l1(log_pred, log_gt)
            total_loss = total_loss + args.lambda_length * length_loss
            length_val = length_loss.item()

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if it >= args.warm_up_iter:
            scheduler.step()

        if it % args.print_iter == 0:
            log(f'iter {it:>6d} | lr {cur_lr:.6f} | ce {ce_loss.item():.4f} | '
                f'len {length_val:.4f} | t_mean {t.mean().item():.3f} | elapsed {time.time()-t0:.0f}s')

        if it > 0 and it % args.eval_iter == 0:
            ce_dev, len_dev = eval_dev_loss(model, mbart_enc, dev_loader, args.device,
                                              args.num_vq, args)
            log(f'  >>> EVAL iter {it} | dev_ce {ce_dev:.4f} | dev_len {len_dev:.4f}')
            if ce_dev < best_dev:
                best_dev = ce_dev; best_iter = it
                torch.save({'model': model.state_dict(), 'args': vars(args),
                             'iter': it, 'best_dev_ce': ce_dev},
                            out_dir / 'best.pt')
                log(f'    [*] new best: ce={ce_dev:.4f} (iter {it})')

        if it > 0 and it % args.save_iter == 0:
            torch.save({'model': model.state_dict(), 'args': vars(args), 'iter': it},
                        out_dir / f'ckpt_{it}.pt')

        it += 1

    torch.save({'model': model.state_dict(), 'args': vars(args), 'iter': it},
                out_dir / 'last.pt')
    log(f'[OK] training done. best_dev_ce={best_dev:.4f} at iter {best_iter}')
    log_f.close()


if __name__ == '__main__':
    main()
