"""MotionGPT-style reproduction on sign data, under SLRTP-canonical.

Faithful to MotionGPT's core idea: treat motion as a language and let a
PRETRAINED LLM generate motion tokens from text. We use mT5 (T5 family, as in
MotionGPT; multilingual so it handles German/Chinese natively) fine-tuned
seq2seq: source = spoken text, target = motion-token string. Motion tokens come
from our trained RVQ (6 layers x 512 -> 3072 motion vocab), flattened
frame-major. NOTE: mT5-base (~580M) is far larger than MSR (~50M) — this is the
(iii) "faithful big version", and the parameter gap is reported as a caveat.

    python train_motiongpt_sign.py --dataset phix --vq_name momask_vq_phix --name mgpt_phix
"""
import os, time, argparse
from os.path import join as pjoin
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, '.')
from models.vq.model import RVQVAE
from train_t2m_sign import SignT2MDataset, cycle

NQ, NB = 6, 512                      # quantizer layers, codebook size  -> 3072 motion tokens


def build_targets(code_idx, mlen_tok, base_id, eos_id, device):
    """code_idx (B,T',Q) -> list of 1D LongTensors of mapped mT5 token ids (+EOS)."""
    layer_off = (torch.arange(NQ, device=device) * NB).view(1, 1, NQ)
    gid = code_idx + layer_off                                   # global motion id in [0,3072)
    tgts = []
    for b in range(code_idx.size(0)):
        n = int(mlen_tok[b].item())
        seq = gid[b, :n].reshape(-1) + base_id                  # frame-major flatten -> mT5 ids
        seq = torch.cat([seq, torch.tensor([eos_id], device=device)])
        tgts.append(seq)
    return tgts


def pad_labels(tgts, pad_id, device):
    L = max(t.numel() for t in tgts)
    out = torch.full((len(tgts), L), pad_id, dtype=torch.long, device=device)
    for i, t in enumerate(tgts):
        out[i, :t.numel()] = t
    lab = out.clone(); lab[lab == pad_id] = -100
    return lab


@torch.no_grad()
def validate(model, vq, tok, loader, base_id, device):
    model.eval(); tot, n = 0.0, 0
    for cap, motion, mlen in loader:
        motion = motion.float().to(device); mlen = mlen.long().to(device)
        code_idx, _ = vq.encode(motion)
        tgts = build_targets(code_idx, mlen // 4, base_id, tok.eos_token_id, device)
        labels = pad_labels(tgts, tok.pad_token_id, device)
        enc = tok(list(cap), return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        loss = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=labels).loss
        tot += loss.item() * len(cap); n += len(cap)
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['phix', 'csl'])
    ap.add_argument('--vq_name', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--mt5', default='google/mt5-base')
    ap.add_argument('--max_motion_length', type=int, default=196)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--total_iter', type=int, default=40000)
    ap.add_argument('--warm_up_iter', type=int, default=1000)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--num_workers', type=int, default=6)
    ap.add_argument('--log_every', type=int, default=500)
    ap.add_argument('--val_every', type=int, default=1000)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--min_delta', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=3407)
    opt = ap.parse_args()

    torch.manual_seed(opt.seed); np.random.seed(opt.seed)
    device = torch.device('cuda')
    droot = pjoin('.', 'dataset', f'{opt.dataset}_sign')
    save_dir = pjoin('.', 'sign_ckpt', f'{opt.dataset}_sign', opt.name)
    os.makedirs(save_dir, exist_ok=True)
    mean = np.load(pjoin(droot, 'Mean.npy')); std = np.load(pjoin(droot, 'Std.npy'))

    # frozen RVQ tokenizer
    va = SimpleNamespace(num_quantizers=NQ, shared_codebook=False, quantize_dropout_prob=0.2,
                         quantize_dropout_cutoff_index=0, mu=0.99, nb_code=NB, code_dim=512)
    vq = RVQVAE(va, 534, NB, 512, 512, 2, 2, 512, 3, 3, 'relu', None).to(device)
    vq.load_state_dict(torch.load(pjoin('.', 'sign_ckpt', f'{opt.dataset}_sign', opt.vq_name, 'net_best.tar'),
                                  map_location='cpu')['vq_model'])
    vq.eval()
    for p in vq.parameters():
        p.requires_grad_(False)

    from transformers import AutoTokenizer, MT5ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(opt.mt5)
    base_id = len(tok)
    tok.add_tokens([f'<m{i}>' for i in range(NQ * NB)])
    model = MT5ForConditionalGeneration.from_pretrained(opt.mt5).to(device)
    model.resize_token_embeddings(len(tok))
    print('MotionGPT(mT5) params: %.1fM | motion vocab %d (base_id %d)' %
          (sum(p.numel() for p in model.parameters()) / 1e6, NQ * NB, base_id))

    tr = SignT2MDataset(droot, 'train', mean, std, opt.max_motion_length)
    vd = SignT2MDataset(droot, 'val', mean, std, opt.max_motion_length)
    loader = cycle(DataLoader(tr, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                              drop_last=True, pin_memory=True, persistent_workers=True))
    val_loader = DataLoader(vd, batch_size=opt.batch_size, shuffle=False, num_workers=4, drop_last=False)

    optim = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    best, best_it, bad = float('inf'), 0, 0
    model.train(); t0 = time.time(); run = 0.0
    for it in range(1, opt.total_iter + 1):
        if it <= opt.warm_up_iter:
            for g in optim.param_groups:
                g['lr'] = opt.lr * it / opt.warm_up_iter
        cap, motion, mlen = next(loader)
        motion = motion.float().to(device); mlen = mlen.long().to(device)
        with torch.no_grad():
            code_idx, _ = vq.encode(motion)
        tgts = build_targets(code_idx, mlen // 4, base_id, tok.eos_token_id, device)
        labels = pad_labels(tgts, tok.pad_token_id, device)
        enc = tok(list(cap), return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        loss = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=labels).loss
        optim.zero_grad(); loss.backward(); optim.step()
        run += loss.item()
        if it % opt.log_every == 0:
            print('iter %6d | loss %.4f | lr %.2e | %.0fs'
                  % (it, run / opt.log_every, optim.param_groups[0]['lr'], time.time() - t0), flush=True)
            run = 0.0
        if it % opt.val_every == 0:
            v = validate(model, vq, tok, val_loader, base_id, device)
            torch.save({'iter': it, 'opt': vars(opt)}, pjoin(save_dir, 'meta.tar'))
            if v < best - opt.min_delta:
                best, best_it, bad = v, it, 0
                model.save_pretrained(pjoin(save_dir, 'best')); tok.save_pretrained(pjoin(save_dir, 'best'))
                tag = 'IMPROVED -> best/'
            else:
                bad += 1; tag = f'no-improve {bad}/{opt.patience} (best {best:.4f}@{best_it})'
            print('  [val] iter %d val_ce %.4f | %s' % (it, v, tag), flush=True)
            if bad >= opt.patience:
                print(f'[EARLY STOP] best val_ce {best:.4f} @ {best_it}', flush=True)
                break
    print(f'[DONE] best val_ce {best:.4f} @ {best_it} -> {save_dir}')


if __name__ == '__main__':
    main()
