# Baselines — fair reproduction under SLRTP-canonical

This folder reproduces three representative general motion-generation methods —
**MoMask** (masked VQ), **MDM** (diffusion) and **MotionGPT** (pretrained LLM) — on the
sign-language datasets (PHOENIX14T, CSL-Daily) under **the exact same protocol as MSR**:
same 178-keypoint lift3d data, same text encoder (mBART-50 for PHIX, char for CSL), and the
same **SLRTP-canonical** back-translation evaluator. This makes every baseline **strictly
comparable** to MSR — unlike cross-paper numbers measured with each paper's own evaluator.

Results: see [`RESULTS.md`](RESULTS.md). MSR wins on both datasets across nearly all metrics;
MotionGPT (mT5, ~587M, ~10× MSR) still does not beat MSR → the edge is structural, not capacity.

## Attribution / upstream

These are our **adaptation scripts only**. The base MoMask code is **not redistributed here** —
clone it from upstream and apply our patch:

- MoMask — https://github.com/EricGuo5513/momask-codes (RVQ-VAE + mask/residual transformers)
- MDM — https://github.com/GuyTevet/motion-diffusion-model (`train_mdm_sign.py` is a compact
  faithful re-implementation of MDM's x0-prediction diffusion, adapted to sign poses)
- MotionGPT — https://github.com/OpenMotionLab/MotionGPT (we realize the "motion-as-language"
  idea with HuggingFace **mT5**, multilingual so it handles German/Chinese natively)

## Setup

```bash
# 1. clone MoMask (provides models/vq + models/mask_transformer used by our scripts)
git clone https://github.com/EricGuo5513/momask-codes
cd momask-codes

# 2. patch the transformer to accept an mBART / char text encoder instead of CLIP
git apply ../patches/momask_transformer_mbart.patch

# 3. drop our scripts in
cp ../scripts/*.py .

# 4. deps: same venv as the main repo (torch cu128, transformers<5, sentencepiece,
#    protobuf, einops, sacrebleu, jiwer). MotionGPT additionally downloads google/mt5-base.
```

## Data prep (MSRSLP .pt → MoMask format)

```bash
python prepare_sign_data.py --dataset phix     # -> dataset/phix_sign/{new_joint_vecs,texts,*.txt,Mean,Std}
python prepare_sign_data.py --dataset csl
```

## Reproduce each baseline (PHIX shown; for CSL use `--dataset csl` and `--text mbart:zh_CN`)

```bash
# ---- MoMask (RVQ tokenizer + masked transformer + residual transformer) ----
python train_vq_sign.py   --dataset phix --name momask_vq_phix --num_quantizers 6 --nb_code 512
python train_t2m_sign.py  --dataset phix --vq_name momask_vq_phix --name momask_mtrans_phix --text mbart:de_DE
python train_res_sign.py  --dataset phix --vq_name momask_vq_phix --name momask_rtrans_phix --text mbart:de_DE
python gen_sign.py        --dataset phix --vq_name momask_vq_phix --mtrans momask_mtrans_phix --rtrans momask_rtrans_phix --text mbart:de_DE --splits dev,test
python compute_momask_ceiling.py --dataset phix --vq_name momask_vq_phix    # VQ representation ceiling

# ---- MDM (continuous diffusion on raw poses, DDIM + classifier-free guidance) ----
python train_mdm_sign.py  --dataset phix --name mdm_phix --text mbart:de_DE
python gen_mdm_sign.py    --dataset phix --name mdm_phix --text mbart:de_DE --splits dev,test --ddim_steps 50 --guidance 2.5

# ---- MotionGPT (mT5 fine-tuned to emit RVQ motion tokens) ----
python train_motiongpt_sign.py --dataset phix --vq_name momask_vq_phix --name mgpt_phix --batch_size 8
python gen_motiongpt_sign.py   --dataset phix --vq_name momask_vq_phix --name mgpt_phix --splits dev,test
```

Each `gen_*` writes an SLRTP pickle under `sign_results/`. Evaluate it with the main repo's
back-translation scripts (`code/eval/slrtp_eval_phix.py` / `slrtp_eval_csl.py`), e.g.:

```bash
python code/eval/slrtp_eval_phix.py --pred-pickle sign_results/phix_mdm_e2e/test.pickle \
  --gt-pt <...>/data_official/test.pt --bt-model-dir <...>/backTranslation_PHIX_model \
  --slrtp-repo <...>/slrtp_official --tag mdm_phix_test --fps 25 --out-dir sign_results/phix_mdm_e2e
# CSL: slrtp_eval_csl.py --pred <pickle> --gt-pt csl_daily_lift3d.test.pt
#      --bt-model-dir backTranslation_CSL_model --no-subsample
```

## Notes

- Capacity is kept comparable to MSR (RVQ/MDM ~20–46M trainable, frozen mBART not counted).
  MotionGPT is the exception: mT5-base is ~587M (~10× MSR), reported as a caveat.
- All trainings use validation-loss early stopping (`--patience 3`).
- `--num_workers` datasets load lazily (paths only) to avoid per-worker RAM blow-up on Windows.
