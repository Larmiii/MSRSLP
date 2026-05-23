"""Multi-Stream VQ-VAE for sign language (Module 1).

Splits PHIX-14T's 178 keypoints into 3 streams with separate encoders, codebooks,
and decoders:
  - body  [ 0 :  8] →  8 × 3 =  24 dim
  - hand  [ 8 : 50] → 42 × 3 = 126 dim   (left + right hand)
  - face  [50 :178] →128 × 3 = 384 dim

For CSL-Daily (133 kpts × 3 = 399), we use HRNet WholeBody layout (MSKA convention):
  - body  [ 0 : 17] → 17 × 3 =  51 dim
  - hand  [91 :133] → 42 × 3 = 126 dim
  - face  [23 : 91] → 68 × 3 = 204 dim
  (foot/feet [17:23] 6 kpts grouped with body if needed)

Each stream has its own encoder/decoder/codebook with smaller capacity than the
joint baseline (so the 3-stream sum has similar params).
"""
from __future__ import annotations
import torch
import torch.nn as nn

from models.encdec import Encoder, Decoder
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset


# Keypoint slice tables (start, end) in keypoint index space
KP_SPLITS = {
    'phix': {
        'body': (0, 8),     # 8 kpts
        'hand': (8, 50),    # 42 kpts (both hands)
        'face': (50, 178),  # 128 kpts
    },
    'csl': {
        'body': (0, 23),    # 23 kpts: 17 COCO body + 6 foot (HRNet WholeBody)
        'face': (23, 91),   # 68 kpts (HRNet face)
        'hand': (91, 133),  # 42 kpts (both hands)
        # sum = 23 + 68 + 42 = 133 ✓
    },
    'csl_lift3d': {
        'body': (0, 8),     # 8 kpts (Ivashechkin 3D body)
        'face': (8, 136),   # 128 kpts (MediaPipe face subset)
        'hand': (136, 178), # 42 kpts (Ivashechkin 3D hands)
    },
    'phix_lift3d': {        # same layout as csl_lift3d — SLRTP-178 pipeline
        'body': (0, 8),
        'face': (8, 136),
        'hand': (136, 178),
    },
    'phix14t': {
        'body': (0, 23),
        'face': (23, 91),
        'hand': (91, 133),
    },
}


def get_stream_dims(dataset_name):
    splits = KP_SPLITS[dataset_name]
    return {name: (end - start) * 3 for name, (start, end) in splits.items()}


def split_pose(pose, dataset_name):
    """Split (B, T, K*3) flat pose into 3 stream tensors.

    pose shape: (B, T, K*3) — keypoints are interleaved x,y,z per kpt.
    Returns: dict of stream_name → (B, T, group_kpts*3)
    """
    splits = KP_SPLITS[dataset_name]
    B, T, _ = pose.shape
    K = pose.shape[-1] // 3
    # Reshape back to (B, T, K, 3) for slicing in kpt-index space
    pose_kpt = pose.reshape(B, T, K, 3)
    out = {}
    for name, (start, end) in splits.items():
        out[name] = pose_kpt[:, :, start:end, :].reshape(B, T, -1)
    return out


def merge_streams(streams, dataset_name):
    """Inverse of split_pose: concatenate stream outputs back into (B, T, K*3).

    NOTE: streams will be re-assembled in keypoint order (sorted by stream
    starting index), so 'face' goes BEFORE 'hand' in CSL but AFTER in PHIX.
    """
    splits = KP_SPLITS[dataset_name]
    K = sum((end - start) for start, end in splits.values())
    sorted_names = sorted(splits.keys(), key=lambda n: splits[n][0])
    B = next(iter(streams.values())).shape[0]
    T = next(iter(streams.values())).shape[1]
    out = torch.zeros(B, T, K, 3, device=next(iter(streams.values())).device,
                       dtype=next(iter(streams.values())).dtype)
    for name in sorted_names:
        start, end = splits[name]
        out[:, :, start:end, :] = streams[name].reshape(B, T, end - start, 3)
    return out.reshape(B, T, -1)


class _SingleStreamVQ(nn.Module):
    """One stream's encoder + quantizer + decoder.

    Lightweight wrapper around T2M-GPT's Encoder/Decoder. Width/depth are
    smaller than the joint baseline so the total Multi-Stream params stay close.
    """
    def __init__(self, input_dim, nb_code, code_dim, output_emb_width,
                 down_t, stride_t, width, depth, dilation_growth_rate,
                 activation, norm, args):
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.input_dim = input_dim
        self.encoder = Encoder(input_dim, output_emb_width, down_t, stride_t,
                                width, depth, dilation_growth_rate,
                                activation=activation, norm=norm)
        self.decoder = Decoder(input_dim, output_emb_width, down_t, stride_t,
                                width, depth, dilation_growth_rate,
                                activation=activation, norm=norm)
        if args.quantizer == 'ema_reset':
            self.quantizer = QuantizeEMAReset(nb_code, code_dim, args)
        elif args.quantizer == 'orig':
            self.quantizer = Quantizer(nb_code, code_dim, 1.0)
        elif args.quantizer == 'ema':
            self.quantizer = QuantizeEMA(nb_code, code_dim, args)
        elif args.quantizer == 'reset':
            self.quantizer = QuantizeReset(nb_code, code_dim, args)

    def preprocess(self, x):    # (B, T, D) → (B, D, T)
        return x.permute(0, 2, 1).float()

    def postprocess(self, x):   # (B, D, T) → (B, T, D)
        return x.permute(0, 2, 1)

    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_enc = self.encoder(x_in)
        x_enc = self.postprocess(x_enc).contiguous().view(-1, x_enc.shape[-2])
        idx = self.quantizer.quantize(x_enc).view(N, -1)
        return idx

    def forward(self, x):
        x_in = self.preprocess(x)
        x_enc = self.encoder(x_in)
        x_q, vq_loss, ppl = self.quantizer(x_enc)
        x_dec = self.decoder(x_q)
        return self.postprocess(x_dec), vq_loss, ppl

    def forward_decoder(self, idx):
        x_d = self.quantizer.dequantize(idx)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        return self.postprocess(self.decoder(x_d))


class MultiStreamVQVAE(nn.Module):
    """3-stream VQ-VAE: body + hand + face.

    Each stream is independent. forward returns a single (B, T, K*3) recon
    that's the concatenation of the 3 stream recons in keypoint order.
    """
    def __init__(self, args,
                 dataset_name='phix',
                 nb_code=256, code_dim=256, output_emb_width=256,
                 down_t=2, stride_t=2,
                 width=256, depth=3,
                 dilation_growth_rate=3,
                 activation='relu', norm=None,
                 stream_codes=None):
        """
        stream_codes: optional dict {'body': nb, 'hand': nb, 'face': nb}
                       to use different codebook sizes per stream. If None,
                       all streams share nb_code.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.stream_dims = get_stream_dims(dataset_name)
        self.streams = nn.ModuleDict()
        if stream_codes is None:
            stream_codes = {n: nb_code for n in self.stream_dims}
        self.stream_codes = stream_codes
        for name, d in self.stream_dims.items():
            self.streams[name] = _SingleStreamVQ(
                input_dim=d,
                nb_code=stream_codes[name], code_dim=code_dim,
                output_emb_width=output_emb_width,
                down_t=down_t, stride_t=stride_t,
                width=width, depth=depth,
                dilation_growth_rate=dilation_growth_rate,
                activation=activation, norm=norm, args=args,
            )
        # Total params metadata (for logging)
        self.input_dim = sum(self.stream_dims.values())

    def forward(self, x):
        """Returns reconstruction, summed vq_loss, dict of per-stream ppl."""
        splits = split_pose(x, self.dataset_name)
        recons = {}
        vq_loss = 0.0
        ppls = {}
        for name, sx in splits.items():
            r, vql, ppl = self.streams[name](sx)
            recons[name] = r
            vq_loss = vq_loss + vql
            ppls[name] = ppl
        recon = merge_streams(recons, self.dataset_name)
        return recon, vq_loss, ppls

    def encode(self, x):
        """Returns dict of stream_name → token sequence (B, T_tok)."""
        splits = split_pose(x, self.dataset_name)
        tokens = {}
        for name, sx in splits.items():
            tokens[name] = self.streams[name].encode(sx)
        return tokens

    def forward_decoder(self, tokens_dict):
        """Takes dict of stream_name → token seq, returns (1, T_orig, K*3)."""
        recons = {}
        for name, idx in tokens_dict.items():
            recons[name] = self.streams[name].forward_decoder(idx)
        return merge_streams(recons, self.dataset_name)
