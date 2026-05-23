"""Multi-Stream Residual VQ-VAE (M1 + M2 combined).

Architecture:
  - 3 stream-specific encoders (body/hand/face, asymmetric widths)
  - Per stream: BASE codebook + RESIDUAL codebook (asymmetric per-stream sizes)
  - Concat 3 stream z_q → 1×1 Conv project → shared decoder

Per timestep emits 6 tokens: (body_base, body_res, hand_base, hand_res, face_base, face_res)

Default vocab sizes (matches M1 v2 = 896 codes):
  body  64 base + 64 res = 128
  hand 128 base + 128 res = 256
  face 256 base + 256 res = 512
  Total = 896 (same as M1 v2!)
"""
from __future__ import annotations
import torch
import torch.nn as nn

from models.encdec import Encoder, Decoder
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset
from models.vqvae_multistream import KP_SPLITS, split_pose


DEFAULT_BASE = {'body': 64, 'hand': 128, 'face': 256}
DEFAULT_RES  = {'body': 64, 'hand': 128, 'face': 256}


def _make_quant(name, nb_code, code_dim, args):
    if name == 'ema_reset': return QuantizeEMAReset(nb_code, code_dim, args)
    if name == 'orig':       return Quantizer(nb_code, code_dim, 1.0)
    if name == 'ema':        return QuantizeEMA(nb_code, code_dim, args)
    if name == 'reset':      return QuantizeReset(nb_code, code_dim, args)
    raise ValueError(name)


class _StreamEncoder(nn.Module):
    def __init__(self, input_dim, code_dim, down_t, stride_t, width, depth,
                 dilation_growth_rate, activation, norm):
        super().__init__()
        self.encoder = Encoder(input_dim, code_dim, down_t, stride_t,
                                width, depth, dilation_growth_rate,
                                activation=activation, norm=norm)

    def forward(self, x):
        x_in = x.permute(0, 2, 1).float()
        return self.encoder(x_in)


class MultiStreamResidualVQVAE(nn.Module):
    def __init__(self, args, dataset_name='phix',
                 code_dim=512, output_emb_width=512,
                 down_t=2, stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation='relu', norm=None,
                 stream_codes_base=None, stream_codes_residual=None,
                 enc_widths=None):
        super().__init__()
        self.dataset_name = dataset_name
        self.code_dim = code_dim
        self.stream_dims = {n: (e - s) * 3
                              for n, (s, e) in KP_SPLITS[dataset_name].items()}
        self.input_dim = sum(self.stream_dims.values())

        if stream_codes_base is None: stream_codes_base = dict(DEFAULT_BASE)
        if stream_codes_residual is None: stream_codes_residual = dict(DEFAULT_RES)
        self.stream_codes_base = stream_codes_base
        self.stream_codes_residual = stream_codes_residual

        if enc_widths is None:
            enc_widths = {'body': max(128, width // 4),
                           'hand': max(256, width // 2),
                           'face': width}

        # 3 encoders
        self.encoders = nn.ModuleDict()
        # 3 base + 3 residual quantizers
        self.quantizers_base = nn.ModuleDict()
        self.quantizers_res = nn.ModuleDict()
        for name, d in self.stream_dims.items():
            w = enc_widths[name]
            self.encoders[name] = _StreamEncoder(
                input_dim=d, code_dim=code_dim,
                down_t=down_t, stride_t=stride_t,
                width=w, depth=depth,
                dilation_growth_rate=dilation_growth_rate,
                activation=activation, norm=norm,
            )
            self.quantizers_base[name] = _make_quant(args.quantizer,
                                                       stream_codes_base[name],
                                                       code_dim, args)
            self.quantizers_res[name] = _make_quant(args.quantizer,
                                                      stream_codes_residual[name],
                                                      code_dim, args)

        # Concat 3 streams (each code_dim) → code_dim
        self.proj_in = nn.Conv1d(3 * code_dim, code_dim, kernel_size=1)
        # Shared decoder
        self.decoder = Decoder(self.input_dim, output_emb_width, down_t, stride_t,
                                width, depth, dilation_growth_rate,
                                activation=activation, norm=norm)
        self.quant = args.quantizer

    def _streams_in_order(self):
        return sorted(self.stream_dims.keys(),
                       key=lambda n: KP_SPLITS[self.dataset_name][n][0])

    def forward(self, x):
        splits = split_pose(x, self.dataset_name)
        zq_streams = []   # one final (B, code_dim, T_tok) per stream
        vq_loss = 0.0
        ppls = {}
        for name in self._streams_in_order():
            z = self.encoders[name](splits[name])                 # (B, C, T_tok)
            base_q, loss_b, ppl_b = self.quantizers_base[name](z)
            residual = z - base_q.detach()
            res_q, loss_r, ppl_r = self.quantizers_res[name](residual)
            z_final = base_q + res_q                              # (B, C, T_tok)
            zq_streams.append(z_final)
            vq_loss = vq_loss + loss_b + loss_r
            ppls[f'{name}_base'] = ppl_b
            ppls[f'{name}_res'] = ppl_r
        z_cat = torch.cat(zq_streams, dim=1)                      # (B, 3*C, T_tok)
        z_proj = self.proj_in(z_cat)
        recon = self.decoder(z_proj).permute(0, 2, 1)
        return recon, vq_loss, ppls

    def encode(self, x):
        """Returns dict of 6 keys → (B, T_tok) token indices."""
        splits = split_pose(x, self.dataset_name)
        tokens = {}
        for name in self._streams_in_order():
            z = self.encoders[name](splits[name])
            B, C, T = z.shape
            z_post = z.permute(0, 2, 1).contiguous().view(-1, C)
            base_idx = self.quantizers_base[name].quantize(z_post).view(B, T)
            # Compute residual
            base_q = self.quantizers_base[name].dequantize(base_idx).permute(0, 2, 1)
            residual = z - base_q
            r_post = residual.permute(0, 2, 1).contiguous().view(-1, C)
            res_idx = self.quantizers_res[name].quantize(r_post).view(B, T)
            tokens[f'{name}_base'] = base_idx
            tokens[f'{name}_res'] = res_idx
        return tokens

    def forward_decoder(self, tokens_dict):
        """tokens_dict: 6 keys → (B, T_tok); returns (B, T_orig, K*3)."""
        zq_streams = []
        for name in self._streams_in_order():
            base_idx = tokens_dict[f'{name}_base']
            res_idx = tokens_dict[f'{name}_res']
            base_q = self.quantizers_base[name].dequantize(base_idx)   # (B, T, C)
            res_q = self.quantizers_res[name].dequantize(res_idx)
            zq = (base_q + res_q).permute(0, 2, 1)                     # (B, C, T)
            zq_streams.append(zq)
        z_cat = torch.cat(zq_streams, dim=1)
        z_proj = self.proj_in(z_cat)
        recon = self.decoder(z_proj).permute(0, 2, 1)
        return recon
