"""Sign-language VQ-VAE training dataset for T2M-GPT framework.

Supports two datasets:
  - PHIX-14T (German Sign Language, SLRTP format)
      .pt file with {sample_id: {'poses_3d': (T, 178, 3), 'text', 'gloss', 'name'}}
      flattened to (T, 534)
  - CSL-Daily (Chinese Sign Language, MSKA HRNet format)
      pickle file with {sample_id: {'keypoint': (T, 133, 3), 'text', 'gloss', 'name', 'num_frames'}}
      flattened to (T, 399)

Returns: (window_size, input_dim) z-normalized motion tensors,
identical convention to T2M-GPT VQMotionDataset.
"""
from __future__ import annotations
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils import data


# Repo root: src/dataset/file.py → release_root/
# Override via env var SLP_DATA_ROOT if your data sits elsewhere.
# dataset_VQ_sign.py at <release>/code/src/dataset/, so parents[3] = release root.
REPO_ROOT = Path(os.environ.get('SLP_DATA_ROOT',
                                  str(Path(__file__).resolve().parents[3])))


class VQSignDataset(data.Dataset):
    """Sign language VQ dataset.

    Args:
      dataset_name: 'phix' or 'csl'
      split: 'train' / 'dev' / 'test'
      window_size: per-sample crop length (T2M-GPT default 64)
      unit_length: codebook stride (kept for API compatibility)
    """

    def __init__(self, dataset_name: str, split: str = 'train',
                 window_size: int = 64, unit_length: int = 4,
                 max_motion_length: int = 200,
                 mean: np.ndarray | None = None,
                 std: np.ndarray | None = None):
        self.dataset_name = dataset_name
        self.split = split
        self.window_size = window_size
        self.unit_length = unit_length
        self.max_motion_length = max_motion_length

        # --- Resolve data files / input dims ---
        if dataset_name == 'phix':
            self.input_dim = 534
            self.joints_num = 178
            data_path = REPO_ROOT / "data/SLRTP-178/data" / f"{split}.pt"  # PHIX-178 Mediapipe — separate download
            self.data, self.ids = self._load_phix(data_path)
        elif dataset_name == 'csl':
            self.input_dim = 399
            self.joints_num = 133
            data_path = REPO_ROOT / "mska_bt/data/CSL-Daily" / f"CSL-Daily.{split}"
            self.data, self.ids = self._load_csl(data_path)
        elif dataset_name == 'phix14t':
            # PHIX-14T with HRNet 133 keypoints (from MSKA paper data).
            # Same loader as CSL — pickle dict of {sid: {keypoint, text, gloss, name, num_frames}}.
            self.input_dim = 399
            self.joints_num = 133
            data_path = REPO_ROOT / "mska_bt/data/Phoenix-2014T" / f"Phoenix-2014T.{split}"
            self.data, self.ids = self._load_csl(data_path)
        elif dataset_name == 'csl_lift3d':
            # CSL-Daily with Ivashechkin 3D lift (178 kpt: 8 body + 128 face + 21+21 hands).
            # SLRTP-format .pt: dict[sid, {name, text, gloss, poses_3d (T,178,3), speaker}]
            self.input_dim = 534
            self.joints_num = 178
            data_path = Path(__file__).resolve().parents[3] / "data" / "csl" / f"csl_daily_lift3d.{split}.pt"
            self.data, self.ids = self._load_phix(data_path)
        elif dataset_name == 'phix_lift3d':
            # PHIX-14T with SLRTP-178 (same 178 kpt layout as csl_lift3d).
            self.input_dim = 534
            self.joints_num = 178
            data_path = Path(__file__).resolve().parents[3] / "data" / "phix" / f"phix_lift3d.{split}.pt"
            self.data, self.ids = self._load_phix(data_path)
        else:
            raise ValueError(f"unknown dataset {dataset_name}")

        # Filter samples shorter than window
        self.data = [m for m in self.data if m.shape[0] >= window_size]
        self.lengths = [max(m.shape[0] - window_size, 1) for m in self.data]

        # Mean/std
        # - For phix (Mediapipe Holistic 178, already normalized in [-1, 1] range):
        #   compute z-score on training data (works fine, std stays small)
        # - For csl / phix14t (HRNet 133, raw pixel coords from MSKA paper data):
        #   z-score is corrupted by outlier frames where some keypoint dims have
        #   pathological values (e.g. CSL kp14_y std ≈ 1.87M). Instead we mirror
        #   MSKA's normalization (datasets.py:96-99): `x/w`, `(h-y)/h` -> not flipped here,
        #   keeping raw orientation: x/(0.5*w) - 1 = (x - 0.5*w) / (0.5*w).
        #   This is bounded, outlier-robust, and round-trips to MSKA's input space cleanly.
        MSKA_WH = {'csl': (512, 512), 'phix14t': (210, 260)}
        if dataset_name in MSKA_WH and (mean is None or std is None):
            w, h = MSKA_WH[dataset_name]
            print(f"[{dataset_name}/{split}] using MSKA-style fixed normalization (w={w}, h={h}, no z-score)")
            self.mean = np.zeros(self.input_dim, dtype=np.float32)
            self.std = np.ones(self.input_dim, dtype=np.float32)
            # x channels (0, 3, 6, ...): (x - 0.5w) / 0.5w
            self.mean[0::3] = 0.5 * w
            self.std[0::3] = 0.5 * w
            # y channels (1, 4, 7, ...): (y - 0.5h) / 0.5h
            self.mean[1::3] = 0.5 * h
            self.std[1::3] = 0.5 * h
            # conf channels (2, 5, 8, ...): identity (mean 0, std 1) → already initialized
        elif mean is None or std is None:
            print(f"[{dataset_name}/{split}] computing mean/std on the fly...")
            all_data = np.concatenate(self.data, axis=0)
            self.mean = all_data.mean(axis=0)
            self.std = all_data.std(axis=0) + 1e-6
        else:
            self.mean = mean
            self.std = std

        print(f"[{dataset_name}/{split}] {len(self.data)} samples loaded "
              f"(input_dim={self.input_dim})")

    def _load_phix(self, path: Path):
        raw = torch.load(path, map_location='cpu', weights_only=False)
        data_list, id_list = [], []
        for sid, sample in raw.items():
            pose = sample['poses_3d']             # (T, 178, 3) torch.Tensor
            if not torch.is_tensor(pose):
                pose = torch.tensor(pose)
            T = pose.shape[0]
            motion = pose.reshape(T, -1).numpy().astype(np.float32)
            data_list.append(motion)
            id_list.append(sid)
        return data_list, id_list

    def _load_csl(self, path: Path):
        # MSKA pickle: dict of {sample_id: {'name', 'text', 'keypoint', 'gloss', 'num_frames'}}
        # IMPORTANT: raw CSL/PHIX-14T HRNet keypoints have sentinel-value outliers
        # (some y values reach -15M, x reach ±1.2M). We clip to image bounds + a small
        # margin before returning; without this z-score / MSKA-norm get poisoned and
        # downstream VQ-VAE cannot reconstruct.
        MSKA_WH = {'csl': (512, 512), 'phix14t': (210, 260)}
        w, h = MSKA_WH.get(self.dataset_name, (None, None))
        with open(path, 'rb') as f:
            raw = pickle.load(f)
        data_list, id_list = [], []
        for sid, sample in raw.items():
            pose = sample['keypoint']             # (T, 133, 3) torch.Tensor
            if torch.is_tensor(pose):
                pose = pose.numpy()
            T = pose.shape[0]
            motion = pose.reshape(T, -1).astype(np.float32)
            if w is not None:
                # Clip channel-wise: x∈[0, w], y∈[0, h], conf∈[0, 1]
                motion[:, 0::3] = np.clip(motion[:, 0::3], 0.0, float(w))
                motion[:, 1::3] = np.clip(motion[:, 1::3], 0.0, float(h))
                motion[:, 2::3] = np.clip(motion[:, 2::3], 0.0, 1.0)
            data_list.append(motion)
            id_list.append(sid)
        return data_list, id_list

    def inv_transform(self, data_np):
        return data_np * self.std + self.mean

    def compute_sampling_prob(self):
        prob = np.array(self.lengths, dtype=np.float32)
        return prob / prob.sum()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        motion = self.data[idx]
        # Random window
        if len(motion) <= self.window_size:
            window = np.tile(motion[:1], (self.window_size, 1))
            window[:len(motion)] = motion
        else:
            start = random.randint(0, len(motion) - self.window_size)
            window = motion[start:start + self.window_size]
        # Z-norm
        window = (window - self.mean) / self.std
        return window.astype(np.float32)


def DATALoader(dataset_name, batch_size, num_workers=4,
                window_size=64, unit_length=4):
    train_set = VQSignDataset(dataset_name, split='train',
                                window_size=window_size, unit_length=unit_length)
    loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    return loader, train_set
