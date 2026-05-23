# Final Ablation Table — Both Datasets (SLRTP-canonical eval)

All models: strict gloss-free, 8-layer cross-attn AR transformer + multi-stream/residual VQ.

**Eval protocol** (single canonical, used for both datasets):
- BT model: signjoey-format trained on lift3d 178-kpt; PHIX uses `backTranslation_PHIX_model`, CSL uses `backTranslation_CSL_model`.
- Decoding: SLRTP `back_translate()` with fixed beam=3, alpha=-1.
- Metrics: sacrebleu raw_corpus_bleu, sacrebleu CHRF, ROUGE-L, jiwer WER + pose dtw_mje, total_distance, avg_duration.
- PHIX: poses at 25fps → subsample by 2 (signjoey BT trained at 12.5fps).
- CSL: no subsample (BT trained at full rate, `skeleton_subsample=1` in config).
- Chinese (CSL): char-tokenized (space-separated chars) before BLEU/CHRF/ROUGE/WER computation.
- Run with `code/eval/_slrtp_convert_and_run.py` (PHIX) or `code/eval/_slrtp_eval_csl.py` (CSL).

---

## PHIX-14T (German, 7K train, 178-kpt SLRTP lift3D)

### TEST (n=641)

| Variant | B1 | B2 | B3 | **B4** | CHRF | ROUGE | WER↓ |
|---|---|---|---|---|---|---|---|
| Baseline (single VQ 512) | 24.81 | 13.03 | 8.90 | **6.86** | 26.35 | 25.65 | 94.03 |
| + M1 (multi-stream 64/1024/256, hand-priority) | 24.77 | 13.05 | 8.91 | **6.82** | 25.80 | 25.37 | 94.35 |
| + M2 (residual 512+512) | 27.04 | 14.83 | 10.22 | **7.88** | 27.79 | 27.88 | 91.16 |
| **+ M1 + M2 (MSR hand-priority asym 1344, ours)** | **29.19** | **17.00** | **11.83** | **8.97** | **29.51** | **30.25** | **90.14** |

### DEV (n=515)

| Variant | B1 | B2 | B3 | **B4** | CHRF | ROUGE | WER↓ |
|---|---|---|---|---|---|---|---|
| Baseline | 23.39 | 12.00 | 8.07 | **6.15** | 25.72 | 25.44 | 93.65 |
| + M1 | 24.22 | 12.41 | 8.31 | **6.24** | 25.63 | 25.39 | 93.04 |
| + M2 | 23.99 | 12.90 | 8.77 | **6.68** | 25.98 | 26.27 | 91.10 |
| **+ M1 + M2** | **29.58** | **17.35** | **12.08** | **9.28** | **29.43** | **31.24** | **88.72** |

### VQ Ceiling (PHIX TEST, GT pose → encode/decode → BT, v2 canonical VQs)

| Variant | DEV B4 | TEST B4 | TEST B1 | TEST CHRF | TEST ROUGE | TEST WER↓ |
|---|---|---|---|---|---|---|
| Baseline | 7.04 | 8.49 | 27.87 | 28.88 | 27.95 | 93.39 |
| M1 | 8.68 | 9.09 | 29.13 | 29.44 | 28.74 | 92.84 |
| M2 | 9.57 | 9.75 | 30.44 | 30.43 | 30.65 | 89.88 |
| **M1+M2** | **10.65** | **11.62** | **32.94** | **32.65** | **33.59** | **87.63** |

Trans / ceiling ratio (TEST B4): baseline 81%, M1 75%, M2 81%, **M1+M2 77%**.

---

## CSL-Daily (Chinese, 18K train, 178-kpt Ivashechkin lift3D)

### TEST (n=1176)

| Variant | B1 | B2 | B3 | **B4** | CHRF | ROUGE | WER↓ |
|---|---|---|---|---|---|---|---|
| Baseline (single VQ 4096) | 14.47 | 5.14 | 2.18 | **1.02** | 3.00 | 15.40 | 98.10 |
| + M1 (multi-stream 3×1024) | 16.24 | 6.67 | 3.32 | **1.91** | 3.62 | 17.26 | 96.83 |
| + M2 (residual 2048+2048) | 14.20 | 4.96 | 2.12 | **1.06** | 2.91 | 14.62 | 99.33 |
| **+ M1 + M2 (MSR 6×512, ours)** | **18.71** | **9.20** | **5.26** | **3.43** | **4.81** | **20.50** | **93.97** |

### DEV (n=1077)

| Variant | B1 | B2 | B3 | **B4** | CHRF | ROUGE | WER↓ |
|---|---|---|---|---|---|---|---|
| Baseline | 14.62 | 4.89 | 1.93 | **0.91** | 2.96 | 15.24 | 98.97 |
| + M1 | 16.24 | 6.66 | 3.33 | **1.88** | 3.61 | 17.41 | 97.18 |
| + M2 | 13.73 | 4.35 | 1.87 | **0.94** | 2.74 | 14.32 | 100.17 |
| **+ M1 + M2** | **19.41** | **9.36** | **5.24** | **3.35** | **4.84** | **20.92** | **93.99** |

Note: Chinese CHRF is character-level, not directly comparable to German word-level CHRF.

---

## Δ vs baseline (TEST B4)

| Variant | PHIX Δ | CSL Δ |
|---|---|---|
| + M1 | -0.04 (-0.6%) | +0.89 (+87%) |
| + M2 | +1.02 (+15%) | +0.04 (+4%) |
| **+ M1 + M2** | **+2.11 (+31%)** | **+2.41 (+236%)** |

---

## SOTA Comparison (PHIX TEST B4)

| Method | Year | TEST B4 | Type |
|---|---|---|---|
| Progressive Transformer (Saunders) | 2020 | 4.38 | gloss-supervised |
| MDM (Diffusion) | 2023 | 7.55 | gloss-supervised |
| T2M-GPT | 2023 | 8.01 | gloss-free |
| **Ours v2 M1+M2 (gloss-free)** ⭐ | 2026 | **8.97** | **strict gloss-free** |
| Sign-IDD | AAAI 2025 | ~10-13 | gloss-supervised |
| T2S-GPT | NeurIPS 2024 | 11.87 | gloss-free SOTA |

---

## Reproducibility

### Train (PHIX, mBART encoder, hand-priority asym MSR)

```bash
# 1. Train VQ M1+M2 (hand-priority asymmetric MSR)
python train_vq_sign_msr.py --dataname phix_lift3d --exp-name vq_M1M2_v2 \
    --nb-base-body 32 --nb-base-hand 512 --nb-base-face 128 \
    --nb-res-body 32  --nb-res-hand 512  --nb-res-face 128 \
    --code-dim 512 --output-emb-width 512 --down-t 2 --stride-t 2 \
    --width 512 --depth 3 --batch-size 128 --total-iter 100000 \
    --lr 2e-4 --lr-scheduler 50000 80000 --gamma 0.05 \
    --vq-act relu --quantizer ema_reset --recons-loss l2

# 2. Tokenize + interleave (6 substreams: body_base, body_res, hand_base, hand_res, face_base, face_res)
python tokenize_sign_msr.py --dataname phix_lift3d --vq-ckpt vq_M1M2_v2.pt --out-dir tokens/
python interleave_msr_tokens.py --in-dir tokens/ --out-dir tokens_interleaved/ \
    --nb-base-body 32 --nb-base-hand 512 --nb-base-face 128 \
    --nb-res-body 32  --nb-res-hand 512  --nb-res-face 128

# 3. Train trans (mBART encoder, num_vq=1344=32+32+512+512+128+128)
python train_trans_sign_cross.py --dataname phix_lift3d \
    --tokens-dir tokens_interleaved/ --vq-ckpt vq_M1M2_v2.pt \
    --text-encoder mbart --mbart-name facebook/mbart-large-50 --lang de_DE \
    --num-vq 1344 --embed-dim 512 --text-dim 1024 \
    --num-layers 9 --n-head 8 --block-size 320 \
    --batch-size 16 --total-iter 20000 \
    --lr 1e-4 --lr-scheduler 10000 15000 --gamma 0.1 --freeze-text 1
```

### Eval (SLRTP-canonical)

```bash
# 1. Generate SLP poses (with our eval_cross_slt_lift3d.py, decoding T=0.9 k=20 rep=1.5)
python code/eval/eval_cross_slt_lift3d.py --variant msr \
    --vq-ckpt vq_M1M2_v2.pt --trans-ckpt trans_M1M2_v2_mbart.pt \
    --splits dev,test --out results/phix_v2_M1M2/ --dataset phix --lang de_DE \
    --temperature 0.9 --top-k 20 --rep-penalty 1.5 --max-run 4 --rep-streams 6

# 2. SLRTP-canonical BT eval
python code/eval/_slrtp_convert_and_run.py \
    --pred-pickle results/phix_v2_M1M2/test.pickle \
    --gt-pt bt_eval_kit/slrtp_official/data_official/test.pt \
    --bt-model-dir bt_eval_kit/slrtp_official/backTranslation_PHIX_model \
    --slrtp-repo bt_eval_kit/slrtp_official \
    --tag phix_v2_M1M2_test --fps 25 --out-dir results/phix_v2_M1M2/slrtp_official/

# For CSL: use _slrtp_eval_csl.py with --no-subsample and CSL BT model
python code/eval/_slrtp_eval_csl.py \
    --pred results/csl_M1M2/slp_pickles/csl_daily.test \
    --gt-pt data/csl/csl_daily_lift3d.test.pt \
    --bt-model-dir bt_eval_kit/slrtp_official/backTranslation_CSL_model \
    --tag csl_v1_M1M2_test --out-dir results/_slrtp_csl_M1M2/ --no-subsample
```

## Ckpt locations (release)

- **PHIX v2 (mBART, hand-priority MSR)** ← paper-grade
  - `checkpoints/phix/vq_v2/vq_{baseline,M1,M2,M1M2}_v2.pt`
  - `checkpoints/phix/trans_v2/trans_{baseline,M1,M2,M1M2}_v2_mbart.pt`
- **CSL v1 (char encoder, MSR 6×512)** ← paper-grade
  - `checkpoints/csl/vq/vq_{baseline,M1,M2,M1M2}.pt`
  - `checkpoints/csl/trans/trans_{baseline,M1,M2,M1M2}_large.pt`
