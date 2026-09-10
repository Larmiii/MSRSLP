# Frozen-weight adapted references

`generate_reference.py` preserves the evaluated `load_bundle`, `decode_vq` and
`generate_one` algorithms. The public entry replaces machine-specific snapshot
discovery with explicit external resource paths and source hashes. It accepts
ID/text inputs only. It does not obtain a generated sequence's duration from the
test reference poses.

These are adaptations of MoMask, MDM and the MotionGPT idea, with heterogeneous
training and text encoders. They are not full official reproductions or an
equal-training comparison with the main models. Both PHX and CSL MoMask/MDM use
their saved mBART conditions (`de_DE` and `zh_CN`). MotionGPT-style uses mT5 and
the six-layer RVQ vocabulary. Saved models were trained with seed3407 and a
196-frame pose cap. Generation may extrapolate to320 frames.

## External software and checkpoints

Clone [MoMask](https://github.com/EricGuo5513/momask-codes) separately and apply
the repository's `baselines/patches/momask_transformer_mbart.patch`. Preserve
its upstream license. Pass that patched checkout as `--upstream-dir`; its
`models/vq` and `models/mask_transformer` modules are not redistributed here.
The repository's `baselines/scripts` supplies the adaptation-specific text and
MDM definitions. Its upstream references include
[MDM](https://github.com/GuyTevet/motion-diffusion-model) and
[MotionGPT](https://github.com/OpenMotionLab/MotionGPT).

Use matching external adaptation checkpoints, their saved `Mean.npy`/`Std.npy`,
and a local mBART or fine-tuned mT5 folder. Different checkpoint training cannot
be expected to reproduce a reported score. Nothing downloads or supplies those
assets. `transformers<5`, PyTorch, NumPy, sentencepiece and the upstream
dependencies must already be installed.

## Prepare and run

The following example uses MDM on PHX. Run from the repository root:

```bash
python supplementary/20260910/advanced/references/prepare_inputs.py \
  --data data/phix/phix_lift3d.train.pt --dataset phix --split train \
  --out external/phix_length.json
python supplementary/20260910/advanced/references/prepare_inputs.py \
  --data data/phix/phix_lift3d.test.pt --dataset phix --split test \
  --out external/phix_test_text.json
python supplementary/20260910/advanced/references/make_resources.py \
  --dataset phix --method mdm --mean external/Mean.npy --std external/Std.npy \
  --mdm external/mdm_best.tar --mbart-directory external/mbart-large-50 \
  --out external/mdm_resources.json
python supplementary/20260910/advanced/references/generate_reference.py \
  --dataset phix --method mdm --split test \
  --input-manifest external/phix_test_text.json --length-model external/phix_length.json \
  --resources external/mdm_resources.json --upstream-dir external/momask-codes \
  --device cuda --out-dir generated/mdm_phix_test
```

For MoMask, build resources with `--method momask --vq ... --base ...
--residual ... --mbart-directory ...`. For MotionGPT-style use `--method
motiongpt_style --vq ... --mt5-directory ...`; no length-model is used.
Relative resource paths resolve against the resource JSON's directory.
`--dry-run` validates inputs, hashes and length plans without loading a model.
`--check-load` is an explicit model-load check, not a score reproduction.

The fixed decoder rules are:

- MoMask:10 base iterations, guidance4, five residual layers with guidance5,
  temperature1. The native residual `-1` sentinel has zero vector contribution.
  It is retained, not changed to codeword0. The base layer never permits `-1`.
- MDM:1000-step cosine schedule,50 DDIM sampling steps, guidance2.5.
- MotionGPT-style:greedy mT5, six disjoint512-codeword layer ranges, EOS only at
  a nonempty complete-group boundary, capped at480 motion tokens.

The MoMask/MDM length predictor is fixed ordinary least squares on TRAIN text
length and capped TRAIN frame count. Generation uses independent ID-derived
seed20260910, preserves sample coverage and writes gzip SLT predictions,
per-sample metadata, resource hashes and a manifest. Run the published
back-translation tool separately with the correct external evaluator. WER can
exceed100%; do not clamp it. Device-specific output equality must be checked
rather than assumed when comparing CPU and GPU scores.
