"""Cross-attention Text-to-Motion DISCRETE FLOW MATCHING transformer.

Drop-in alternative to t2m_trans_cross.CrossAttnText2MotionTransformer that
replaces autoregressive token generation with discrete flow matching (DFM,
Gat et al. NeurIPS 2024).

Key architectural differences vs AR:
  - Bidirectional self-attention (no causal mask).
  - Time-step `t ∈ [0, 1]` injected via sinusoidal MLP → added to each position.
  - One extra vocab slot for [MASK]; LM head predicts `num_vq` clean-token classes
    (never predicts MASK).
  - Length head (predict_length=True) is mandatory at inference: in DFM we must
    fix sequence length up front (no EOS to terminate generation).

Training step:
  t ~ LogitNormal(0, 1)             # focus capacity on hard middle regime
  for each clean token x1[i]:
      with prob t:        keep x1[i]
      with prob (1 - t):  replace with MASK
  logits = model(x_t, t, text_memory)
  loss = CE(logits[masked positions], x1[masked positions])
  # drop text with p=0.1 for classifier-free guidance

Inference (Euler scheme with re-noising, see Gat et al. eq. 7):
  x_t = full(MASK, T)
  for i in range(N_steps):
      t = i / N_steps;  dt = 1 / N_steps
      p_c = softmax(model(x_t, t, c));  p_u = softmax(model(x_t, t, null))
      p   = p_u * (p_c / p_u) ** w                # CFG, w ∈ [1.5, 3.0]
      x1_hat = sample_categorical(p)              # predicted clean tokens
      keep = bernoulli(t + dt).expand_as(x_t)     # re-mask schedule
      x_t  = where(keep, x1_hat, MASK)
  return x_t

Vocab layout (kept compat with M1+M2 interleaved tokens):
  ids [0 .. num_vq-1]: regular codes (interleaved global ids for MSR)
  id  [num_vq]:        MASK
  (no BOS, no EOS, no PAD — DFM uses fixed-length parallel generation)
"""
from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal time embedding (same as classic diffusion). t in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)   # (B, half)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class CrossAttnDFM(nn.Module):
    """Bidirectional transformer + cross-attn to text, trained with DFM."""

    def __init__(
        self,
        num_vq: int,
        text_dim: int = 1024,
        embed_dim: int = 512,
        block_size: int = 320,
        num_layers: int = 6,
        n_head: int = 8,
        drop_out_rate: float = 0.1,
        fc_rate: int = 4,
        predict_length: bool = True,
    ):
        super().__init__()
        self.num_vq = num_vq
        self.embed_dim = embed_dim
        self.block_size = block_size

        # Vocab: regular codes [0..num_vq-1] + MASK at [num_vq]
        self.MASK_ID = num_vq
        self.token_emb = nn.Embedding(num_vq + 1, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, embed_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Text projection (mBART encoder hidden → decoder dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)
        self.text_norm = nn.LayerNorm(embed_dim)

        # Time embedding: sinusoidal → MLP → broadcast to each position
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self.input_drop = nn.Dropout(drop_out_rate)

        # TransformerDecoder works for us too (cross-attn over text memory).
        # We just won't pass any tgt_mask → bidirectional self-attn.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=embed_dim * fc_rate,
            dropout=drop_out_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers,
                                              norm=nn.LayerNorm(embed_dim))

        # LM head: predicts num_vq clean-token classes (never MASK)
        self.lm_head = nn.Linear(embed_dim, num_vq)

        # Length head: predicts total motion-token length from pooled text memory.
        # Mandatory for DFM inference (no EOS).
        self.predict_length = predict_length
        if predict_length:
            self.length_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 1),
            )
        else:
            self.length_head = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode_text_memory(self, mbart_last_hidden: torch.Tensor,
                            text_attn_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mem = self.text_norm(self.text_proj(mbart_last_hidden))
        mem_key_pad = (text_attn_mask == 0)
        return mem, mem_key_pad

    def predict_motion_length(self, mem: torch.Tensor, mem_key_pad: torch.Tensor) -> torch.Tensor:
        valid = (~mem_key_pad).float().unsqueeze(-1)
        pooled = (mem * valid).sum(1) / valid.sum(1).clamp(min=1)
        return self.length_head(pooled).squeeze(-1)

    def forward(
        self,
        x_t: torch.Tensor,                       # (B, T) noisy tokens (some are MASK)
        t: torch.Tensor,                         # (B,) timesteps in [0, 1]
        mbart_last_hidden: torch.Tensor,         # (B, T_text, text_dim)
        text_attn_mask: torch.Tensor,            # (B, T_text)
        tgt_key_padding_mask: Optional[torch.Tensor] = None,  # (B, T) True = pad position
    ) -> torch.Tensor:
        """Predict per-position logits over clean token vocab.

        Returns:
          logits: (B, T, num_vq)
        """
        B, T = x_t.shape
        assert T <= self.block_size, f"T={T} > block_size={self.block_size}"

        x = self.token_emb(x_t) + self.pos_emb[:, :T]
        # Time embedding: sinusoidal → MLP → (B, embed_dim) → broadcast to T positions
        t_emb = _timestep_embedding(t, self.embed_dim)
        t_emb = self.time_mlp(t_emb).unsqueeze(1)             # (B, 1, embed_dim)
        x = x + t_emb
        x = self.input_drop(x)

        mem, mem_key_pad = self.encode_text_memory(mbart_last_hidden, text_attn_mask)

        # No tgt_mask = bidirectional self-attn (DFM is parallel, not causal)
        out = self.decoder(
            tgt=x,
            memory=mem,
            tgt_mask=None,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=mem_key_pad,
        )                                                     # (B, T, embed_dim)
        logits = self.lm_head(out)                            # (B, T, num_vq)
        return logits

    # --------------------------------------------------------------
    # Training utilities
    # --------------------------------------------------------------

    @staticmethod
    def sample_logit_normal_t(B: int, device, mu: float = 0.0, sigma: float = 1.0) -> torch.Tensor:
        """SD3-style logit-normal t sampling: t = sigmoid(N(mu, sigma^2))."""
        eps = torch.randn(B, device=device) * sigma + mu
        return torch.sigmoid(eps)

    def corrupt(self, x_clean: torch.Tensor, t: torch.Tensor,
                pad_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mask tokens with prob (1 - t) per position. Returns (x_t, mask_pos).
        mask_pos is True where the token was replaced with MASK (these are
        the positions the loss is computed over). pad positions are never masked
        and are excluded from loss.
        """
        B, T = x_clean.shape
        keep_prob = t.view(B, 1).expand(B, T)                  # (B, T)
        rnd = torch.rand_like(keep_prob)
        keep = rnd < keep_prob                                  # True = keep clean
        if pad_mask is not None:
            keep = keep | pad_mask                              # pad always "kept" (no loss)
        x_t = torch.where(keep, x_clean, torch.full_like(x_clean, self.MASK_ID))
        loss_mask = (~keep) & (pad_mask is None or ~pad_mask)
        if pad_mask is not None:
            loss_mask = loss_mask & (~pad_mask)
        return x_t, loss_mask

    # --------------------------------------------------------------
    # Sampling
    # --------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        mbart_last_hidden: torch.Tensor,         # (B, T_text, text_dim)
        text_attn_mask: torch.Tensor,            # (B, T_text)
        lengths: torch.Tensor,                   # (B,) target motion-token lengths
        n_steps: int = 24,
        cfg_scale: float = 2.0,
        null_mbart: Optional[torch.Tensor] = None,
        null_text_attn_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate motion token sequences via Euler DFM sampling with re-noising.

        Returns:
          x_final: (B, T_max) int64 — padded to max(lengths). Valid region per
                   sample is x_final[i, :lengths[i]].
        """
        device = mbart_last_hidden.device
        B = mbart_last_hidden.shape[0]
        T_max = int(lengths.max().item())
        T_max = min(T_max, self.block_size)
        lengths = lengths.clamp(max=T_max).long()

        # pad_mask: True for positions beyond per-sample length
        pad_mask = torch.arange(T_max, device=device)[None, :] >= lengths[:, None]  # (B, T_max)

        x_t = torch.full((B, T_max), self.MASK_ID, dtype=torch.long, device=device)

        # CFG: pre-compute null memory if needed (w > 1.0)
        use_cfg = cfg_scale > 1.0 and null_mbart is not None
        if use_cfg:
            # Expand null to batch size (it's stored as B=1)
            if null_mbart.size(0) == 1 and B > 1:
                null_mbart = null_mbart.expand(B, -1, -1).contiguous()
                null_text_attn_mask = null_text_attn_mask.expand(B, -1).contiguous()

        for i in range(n_steps):
            t = torch.full((B,), i / n_steps, device=device)
            t_next = (i + 1) / n_steps

            logits_c = self.forward(x_t, t, mbart_last_hidden, text_attn_mask,
                                     tgt_key_padding_mask=pad_mask)
            if use_cfg:
                logits_u = self.forward(x_t, t, null_mbart, null_text_attn_mask,
                                         tgt_key_padding_mask=pad_mask)
                # CFG in log-space (equivalent to p_u * (p_c/p_u)^w):
                logits = logits_u + cfg_scale * (logits_c - logits_u)
            else:
                logits = logits_c

            if temperature != 1.0:
                logits = logits / temperature

            probs = F.softmax(logits, dim=-1)                    # (B, T, num_vq)
            # Sample categorical per position
            x1_hat = torch.distributions.Categorical(probs=probs).sample()  # (B, T)

            # Re-mask schedule: at step i→i+1, keep x1_hat with prob t_next, else MASK
            keep = torch.rand_like(x1_hat, dtype=torch.float32) < t_next
            # Final step: keep everything
            if i == n_steps - 1:
                keep = torch.ones_like(keep, dtype=torch.bool)
            # Where already revealed and not re-masked, KEEP it (don't recompute)
            already_clean = (x_t != self.MASK_ID)
            new_x = torch.where(already_clean, x_t,
                                 torch.where(keep, x1_hat,
                                              torch.full_like(x_t, self.MASK_ID)))
            # pad positions stay as MASK (we'll trim later anyway)
            new_x = torch.where(pad_mask, x_t, new_x)
            x_t = new_x

        # Replace any remaining MASKs (shouldn't happen at final step but be safe)
        any_mask = (x_t == self.MASK_ID) & (~pad_mask)
        if any_mask.any():
            # Fall back to argmax for any leftover MASK positions
            with torch.no_grad():
                t_final = torch.full((B,), 1.0, device=device)
                logits_final = self.forward(x_t, t_final, mbart_last_hidden, text_attn_mask,
                                              tgt_key_padding_mask=pad_mask)
                x_argmax = logits_final.argmax(dim=-1)
                x_t = torch.where(any_mask, x_argmax, x_t)

        return x_t
