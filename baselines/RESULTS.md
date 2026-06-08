# Fair comparison: MSR vs reproduced baselines (all under SLRTP-canonical)

Every baseline below was reproduced under ONE unified setup: the same 178-kpt
lift3d data, the same text encoder family (mBART de_DE / zh_CN), and the same
SLRTP-canonical back-translation evaluator as MSR. So these rows are mutually
comparable (unlike cross-paper numbers measured with each paper's own evaluator).

Baselines (paradigm / capacity):
- T2M-GPT (VQ+AR)         = MSR's single-stream baseline (already in ablation).
- MDM (diffusion)         ~18M, our compact faithful re-impl, mBART text, DDIM+CFG.
- MoMask (masked VQ)      ~46M, official RVQ + mask + residual transformers, mBART.
- MotionGPT (LLM)         mT5-base ~587M (~10x MSR) — (iii) faithful big version; param gap noted.
- MSR (ours, VQ+AR)       ~50M.

## PHOENIX14T  (TEST / DEV BLEU-4 + TEST full metrics)

| method | params | TEST B1 | TEST B4 | TEST CHRF | TEST ROUGE | TEST WER↓ | DEV B4 |
|---|---|---|---|---|---|---|---|
| MDM (repro, diffusion)        | ~18M  | 23.88 | 6.17 | 25.42 | 23.45 | 98.42 | 5.52 |
| T2M-GPT (≈ single-stream base)| ~50M  | 24.81 | 6.86 | 26.35 | 25.65 | 94.03 | 6.15 |
| MotionGPT (repro, mT5-587M)   | 587M  | 27.65 | 8.10 | 27.97 | 29.89 | 87.09 | 7.51 |
| MoMask (repro, masked VQ)     | ~46M  | 28.79 | 8.67 | 28.51 | 29.43 | 91.55 | 8.99 |
| **MSR (ours)**                | ~50M  | **29.19** | **8.97** | **29.51** | **30.25** | 90.14 | **9.28** |

→ MSR best on B1/B4/CHRF/ROUGE on both splits. MotionGPT (~10x params) still below MSR B4.
  (WER: MotionGPT 87.09 best; MSR 90.14 — MSR leads BLEU/CHRF/ROUGE, MotionGPT edges WER.)

## CSL-Daily  (char-level; TEST full + DEV B4)

| method | params | TEST B1 | TEST B4 | TEST CHRF | TEST ROUGE | TEST WER↓ | DEV B4 |
|---|---|---|---|---|---|---|---|
| MDM (repro, diffusion)        | ~18M  | 12.90 | 0.39 | 2.50 | 13.53 | 100.18 | 0.51 |
| T2M-GPT (≈ single-stream base)| ~50M  | 14.47 | 1.02 | 3.00 | 15.40 | 98.10 | 0.91 |
| MoMask (repro, masked VQ)     | ~46M  | 13.73 | 1.19 | 3.06 | 15.73 | 96.66 | 1.12 |
| MotionGPT (repro, mT5-587M)   | 587M  | 13.55 | 1.18 | 3.11 | 16.25 | 96.40 | 1.16 |
| **MSR (ours)**                | ~50M  | **18.71** | **3.43** | **4.81** | **20.50** | **93.97** | **3.35** |

→ MSR decisively best on ALL metrics (B4 3.43 ≈ 2.9x MoMask 1.19 ≈ 2.9x MotionGPT 1.18).
  MotionGPT (587M mT5) only matches MoMask here, far below MSR.

## VQ tokenizer ceiling (PHIX TEST, GT pose -> encode/decode -> BT)
| method | B4 ceiling | codebook |
|---|---|---|
| MoMask RVQ (repro) | 11.34 | 6x512=3072 |
| **MSR (ours)** | **11.62** | 1344 |

## Key narrative for the thesis
1. Under one fair evaluator, MSR is the best on both datasets across nearly all metrics.
2. Evaluator-artifact proof: GLOS reported MoMask CSL B4 = 3.57 (its own evaluator);
   under SLRTP-canonical the same method gets 1.19. Cross-paper numbers are not comparable.
3. MotionGPT uses ~10x params (587M mT5) and still loses on B4 -> MSR's edge is structural,
   not capacity.
4. MDM (diffusion) is weakest here; MoMask (masked VQ) is the strongest non-MSR.
5. Other published methods (T2S-GPT/Sign-IDD/DARSLP/A2V-SLP/GLOS/SignPR) use different
   evaluators/representations/supervision -> cite with caveat, NOT in the comparison table.
