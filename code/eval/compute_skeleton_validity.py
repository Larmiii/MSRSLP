"""Compute bone-length consistency and skeleton validity for generated motion.

For each sample pickle (SLP gen output), compute:
1. Bone-length error (BLE): per-bone std of length over time / |mean bone length over GT|
2. Invalid bone ratio: % of bones whose length deviates > 3σ from GT-train statistics

Output: summary stats over dev or test split.

178-kpt layout for csl_lift3d / phix_lift3d (Ivashechkin body + MediaPipe face + Ivashechkin hand):
  body  0-7   (8 kpts)
  face  8-135 (128 kpts)
  hand  136-177 (42 kpts: 21 LH + 21 RH)

We use a coarse set of skeletal bones from body + LH knuckles, since face is densely
sampled (no bone-tree). Body bones are inferred from MediaPipe Holistic order.
"""
from __future__ import annotations
import argparse, gzip, pickle, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


# Body bones (proxy: adjacent kpts in MediaPipe Pose 8-kpt subset).
# Actual lift3d body indices: 0=nose-ish, then 1-7 form torso/arm chain.
# For robustness, use all adjacent pairs in body region [0,8) → 7 bones.
BODY_BONES = [(i, i + 1) for i in range(7)]                  # 7 body bones

# Hand bones (per hand, 21 kpts each → 20 bones based on MediaPipe Hand topology).
# MediaPipe Hand: 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
HAND_BONE_PAIRS_LOCAL = [
    (0,1),(1,2),(2,3),(3,4),       # thumb
    (0,5),(5,6),(6,7),(7,8),       # index
    (0,9),(9,10),(10,11),(11,12),  # middle
    (0,13),(13,14),(14,15),(15,16),# ring
    (0,17),(17,18),(18,19),(19,20),# pinky
]
LH_OFFSET = 136
RH_OFFSET = 157
LH_BONES = [(LH_OFFSET + a, LH_OFFSET + b) for a, b in HAND_BONE_PAIRS_LOCAL]
RH_BONES = [(RH_OFFSET + a, RH_OFFSET + b) for a, b in HAND_BONE_PAIRS_LOCAL]
ALL_BONES = BODY_BONES + LH_BONES + RH_BONES                  # 47 total


def compute_per_sample_bone_stats(motion):
    """motion: (T, 178, 3) numpy. Returns per-bone (T,) length array dict."""
    if motion.shape[0] < 2:
        return None
    bone_lens = {}
    for (a, b) in ALL_BONES:
        d = motion[:, a, :] - motion[:, b, :]   # (T, 3)
        bone_lens[(a, b)] = np.linalg.norm(d, axis=-1)  # (T,)
    return bone_lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-pickle', required=True,
                     help='SLP-gen pickle file (csl_daily.{split} format)')
    ap.add_argument('--gt-pickle', default=None,
                     help='GT pose pickle (csl_daily_lift3d.{split}.pt). If provided, computes '
                          'deviation from GT bone-length statistics.')
    args = ap.parse_args()

    # Load predictions
    print(f'[*] loading {args.pred_pickle}')
    with gzip.open(args.pred_pickle, 'rb') as f:
        preds = pickle.load(f)
    print(f'    {len(preds)} samples')

    # Aggregate per-bone std (intra-sample temporal stability)
    per_bone_stds = defaultdict(list)            # bone → list of (sample-wise temporal std)
    per_bone_means = defaultdict(list)           # bone → list of per-sample mean length

    for s in preds:
        sign = s['sign']
        if torch.is_tensor(sign): sign = sign.numpy()
        if sign.ndim == 2:                       # flat (T, 534)
            T = sign.shape[0]
            motion = sign.reshape(T, 178, 3)
        else:
            motion = sign                         # (T, 178, 3)
        stats = compute_per_sample_bone_stats(motion)
        if stats is None: continue
        for bone, lens in stats.items():
            per_bone_stds[bone].append(lens.std())
            per_bone_means[bone].append(lens.mean())

    # Aggregate stats across samples per bone
    bone_summary = []
    for bone in ALL_BONES:
        if bone not in per_bone_stds: continue
        stds = np.array(per_bone_stds[bone])
        means = np.array(per_bone_means[bone])
        # Temporal stability: avg std / avg mean (CV-like)
        cv = stds.mean() / max(means.mean(), 1e-6)
        bone_summary.append({'bone': bone, 'mean_len': means.mean(),
                              'mean_std': stds.mean(), 'cv': cv})

    body_cvs = [b['cv'] for b in bone_summary if b['bone'] in BODY_BONES]
    lh_cvs   = [b['cv'] for b in bone_summary if b['bone'] in LH_BONES]
    rh_cvs   = [b['cv'] for b in bone_summary if b['bone'] in RH_BONES]
    all_cvs  = [b['cv'] for b in bone_summary]

    print(f'\n=== Bone-length stability (lower = more rigid) ===')
    print(f'  Body bones (n={len(body_cvs)}):   mean CV = {np.mean(body_cvs):.4f}')
    print(f'  L-Hand bones (n={len(lh_cvs)}):   mean CV = {np.mean(lh_cvs):.4f}')
    print(f'  R-Hand bones (n={len(rh_cvs)}):   mean CV = {np.mean(rh_cvs):.4f}')
    print(f'  ALL bones (n={len(all_cvs)}):     mean CV = {np.mean(all_cvs):.4f}')

    # GT comparison
    if args.gt_pickle:
        print(f'\n[*] loading GT {args.gt_pickle}')
        gt = torch.load(args.gt_pickle, map_location='cpu', weights_only=False)
        gt_bone_means = defaultdict(list)
        for sid, v in gt.items():
            pose = v['poses_3d']
            if torch.is_tensor(pose): pose = pose.numpy()
            stats = compute_per_sample_bone_stats(pose)
            if stats is None: continue
            for bone, lens in stats.items():
                gt_bone_means[bone].append(lens.mean())
        # Per-bone GT mean
        gt_means = {b: np.mean(v) for b, v in gt_bone_means.items()}
        gt_stds = {b: np.std(v) for b, v in gt_bone_means.items()}

        # Invalid bone ratio: pred per-sample mean bone length outside GT mean ± 3σ
        invalid_count = 0; total_count = 0
        for s_i, s in enumerate(preds):
            sign = s['sign']
            if torch.is_tensor(sign): sign = sign.numpy()
            if sign.ndim == 2:
                motion = sign.reshape(-1, 178, 3)
            else:
                motion = sign
            stats = compute_per_sample_bone_stats(motion)
            if stats is None: continue
            for bone, lens in stats.items():
                m_pred = lens.mean()
                if bone not in gt_means: continue
                gt_m, gt_s = gt_means[bone], max(gt_stds[bone], 1e-6)
                total_count += 1
                if abs(m_pred - gt_m) > 3 * gt_s:
                    invalid_count += 1
        print(f'\n=== Skeleton validity (3σ from GT bone-length mean) ===')
        print(f'  Invalid bone ratio: {invalid_count/max(total_count,1)*100:.2f}% '
              f'({invalid_count}/{total_count})')


if __name__ == '__main__':
    main()
