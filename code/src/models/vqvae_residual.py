"""Residual VQ-VAE for sign language (Module 2, MoMask-style).

Two-stage quantization on a single-stream pose (PHIX 534-dim or CSL 399-dim):
  encoder → z_continuous
  base_q  = quantize_base(z_continuous)           # discrete base layer
  residual_input = z_continuous - base_q.detach()
  res_q   = quantize_residual(residual_input)     # captures fine detail
  z_q     = base_q + res_q
  decoder(z_q) → reconstruction

Per-step we emit 2 tokens (base_idx, residual_idx).
"""
from __future__ import annotations
import torch
import torch.nn as nn

from models.encdec import Encoder, Decoder
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset


def _make_quant(name, nb_code, code_dim, args):
    if name == 'ema_reset': return QuantizeEMAReset(nb_code, code_dim, args)
    if name == 'orig':       return Quantizer(nb_code, code_dim, 1.0)
    if name == 'ema':        return QuantizeEMA(nb_code, code_dim, args)
    if name == 'reset':      return QuantizeReset(nb_code, code_dim, args)
    raise ValueError(name)


class ResidualVQVAE(nn.Module):
    def __init__(self, args, nb_code=512, code_dim=512, output_emb_width=512,
                 down_t=2, stride_t=2, width=512, depth=3,
                 dilation_growth_rate=3, activation='relu', norm=None,
                 nb_code_residual=None):
        super().__init__()
        # Dataset → input dim mapping (same as VQVAE_251)
        input_dim_map = {'kit': 251, 't2m': 263, 'phix': 534, 'csl': 399, 'phix14t': 399, 'csl_lift3d': 534, 'phix_lift3d': 534}
        name = getattr(args, 'dataname', 't2m')
        input_dim = input_dim_map.get(name, getattr(args, 'input_dim', 263))
        self.input_dim = input_dim
        self.code_dim = code_dim
        self.num_code = nb_code
        self.num_code_residual = nb_code_residual or nb_code
        self.quant = args.quantizer
        self.encoder = Encoder(input_dim, output_emb_width, down_t, stride_t,
                                width, depth, dilation_growth_rate,
                                activation=activation, norm=norm)
        self.decoder = Decoder(input_dim, output_emb_width, down_t, stride_t,
                                width, depth, dilation_growth_rate,
                                activation=activation, norm=norm)
        self.quantizer_base = _make_quant(args.quantizer, nb_code, code_dim, args)
        self.quantizer_res = _make_quant(args.quantizer, self.num_code_residual,
                                            code_dim, args)

    def preprocess(self, x):   # (B, T, D) → (B, D, T)
        return x.permute(0, 2, 1).float()

    def postprocess(self, x):  # (B, D, T) → (B, T, D)
        return x.permute(0, 2, 1)

    def encode(self, x):
        """Returns (base_idx, residual_idx), each (B, T_tok)."""
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        z = self.encoder(x_in)                                # (B, C, T_tok)
        z_post = self.postprocess(z).contiguous().view(-1, z.shape[-2])
        base_idx = self.quantizer_base.quantize(z_post).view(N, -1)
        base_q = self.quantizer_base.dequantize(base_idx)     # (B, T_tok, C)
        base_q = base_q.permute(0, 2, 1).contiguous()         # (B, C, T_tok)
        residual = z - base_q
        r_post = self.postprocess(residual).contiguous().view(-1, residual.shape[-2])
        res_idx = self.quantizer_res.quantize(r_post).view(N, -1)
        return base_idx, res_idx

    def forward(self, x):
        x_in = self.preprocess(x)
        z = self.encoder(x_in)                                # (B, C, T_tok)
        # Base quantization
        base_q, loss_base, ppl_base = self.quantizer_base(z)
        # Residual quantization on (z - base_q.detach())
        residual = z - base_q.detach()
        res_q, loss_res, ppl_res = self.quantizer_res(residual)
        z_q = base_q + res_q
        recon = self.decoder(z_q)
        return self.postprocess(recon), loss_base + loss_res, (ppl_base, ppl_res)

    def forward_decoder(self, base_idx, res_idx):
        """Both base_idx and res_idx are (B, T_tok). Returns (B, T_orig, K*3)."""
        base_q = self.quantizer_base.dequantize(base_idx)     # (B, T_tok, C)
        res_q = self.quantizer_res.dequantize(res_idx)
        z_q = (base_q + res_q).permute(0, 2, 1).contiguous()
        return self.postprocess(self.decoder(z_q))
