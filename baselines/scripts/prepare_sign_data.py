"""Convert MSRSLP sign data (.pt, 178-kpt) -> MoMask layout for one dataset.

Usage:
    python prepare_sign_data.py --dataset phix
    python prepare_sign_data.py --dataset csl

Produces under ./dataset/<phix_sign|csl_sign>/:
    new_joint_vecs/<sid>.npy   (T, 534) float32  -- poses_3d flattened
    texts/<sid>.txt            spoken-language sentence (one line)
    {train,val,test}.txt       sid lists (MoMask uses 'val' as dev)
    Mean.npy, Std.npy          (534,) computed over TRAIN frames
"""
import argparse, os
from os.path import join as pjoin
import numpy as np
import torch

SRC = {
    'phix': {
        'dir': r'D:\Graduation\MSRSLP\data\phix',
        'tmpl': 'phix_lift3d.{split}.pt',
    },
    'csl': {
        'dir': r'D:\Graduation\MSRSLP\data\csl',
        'tmpl': 'csl_daily_lift3d.{split}.pt',
    },
}

def load_split(cfg, split):
    p = pjoin(cfg['dir'], cfg['tmpl'].format(split=split))
    return torch.load(p, map_location='cpu', weights_only=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['phix', 'csl'])
    args = ap.parse_args()
    cfg = SRC[args.dataset]
    out_root = pjoin('.', 'dataset', f'{args.dataset}_sign')
    motion_dir = pjoin(out_root, 'new_joint_vecs')
    text_dir = pjoin(out_root, 'texts')
    os.makedirs(motion_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)

    split_map = {'train': 'train', 'dev': 'val', 'test': 'test'}
    train_frames = []
    counts = {}
    for src_split, mm_split in split_map.items():
        d = load_split(cfg, src_split)
        ids = []
        for sid, s in d.items():
            pose = s['poses_3d']
            pose = pose.cpu().numpy() if torch.is_tensor(pose) else np.asarray(pose)
            T = pose.shape[0]
            vec = pose.reshape(T, -1).astype(np.float32)   # (T, 534)
            np.save(pjoin(motion_dir, f'{sid}.npy'), vec)
            text = (s.get('text', '') or '').strip().replace('\n', ' ')
            with open(pjoin(text_dir, f'{sid}.txt'), 'w', encoding='utf-8') as f:
                f.write(text + '\n')
            ids.append(sid)
            if src_split == 'train':
                train_frames.append(vec)
        with open(pjoin(out_root, f'{mm_split}.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(ids) + '\n')
        counts[mm_split] = len(ids)

    allf = np.concatenate(train_frames, axis=0)            # (sumT, 534)
    mean = allf.mean(axis=0).astype(np.float32)
    std = allf.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    np.save(pjoin(out_root, 'Mean.npy'), mean)
    np.save(pjoin(out_root, 'Std.npy'), std)

    print(f'[OK] {args.dataset}: dim=534, splits={counts}, '
          f'train_frames={allf.shape[0]}, mean/std saved to {out_root}')

if __name__ == '__main__':
    main()
