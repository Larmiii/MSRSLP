# Local masked parallel generator interface

This is the local DFM-style implementation used for the supplementary PHX
interface experiment. `grouped_mask.py` retains the actual grouped model,
action-only objectives and text-length conversion functions. The underlying
`CrossAttnDFM` is imported from `../../runtime/code/src/models/t2m_dfm_cross.py`.
It is not presented as a complete standard-DFM reproduction.

`config.json` records the actual6-layer,512-dimensional,8-head architecture,
the frozen mBART `de_DE` encoder (maximum80 text tokens), training settings and
fixed50-step sampler. The sampler reveals MASK positions monotonically. Revealed
tokens remain unchanged. Sampling restricts each token position to its declared
codebook range. Complete six-token groups decode through the frozen revised
PAMRVQ, with codebook sizes `[32,32,512,512,128,128]`.

Length comes from the model's text-only head: exponentiate its log prediction,
round to a complete six-token group and clamp to6..480 tokens (4..320 frames).
The position capacity is512. Temperature, length scale and CFG scale are1.
The evaluated AR model has9 layers, a128-token text limit and a different length
mechanism, so this is an interface test rather than a matched-budget ranking.

## Inference with externally supplied weights

Use the released supplement runtime and prepare an ID/text manifest with
`../references/prepare_inputs.py`. No checkpoint, dataset or pretrained text
weight is included in this repository. The mask checkpoint must contain the
training protocol and quantizer/normalization hashes saved by the experiment.
The VQ checkpoint must carry the verified layout and anatomical grouping fields.

```bash
python supplementary/20260910/advanced/dfm/generate_mask.py \
  --checkpoint external/mask_best.pt --vq-checkpoint external/pamrvq/best.pt \
  --text-model external/mbart-large-50 --input-manifest external/phix_test_text.json \
  --split test --device cuda --out generated/masked_phix_test
```

Place the matching `mean.npy` and `std.npy` beside the VQ checkpoint. A local
text-model override relocates the same text snapshot; it does not establish
that arbitrary other weights are equivalent. The original pinned Hugging Face
revision is recorded in the configuration. Only local/cached files are loaded.

`--help` and `--dry-run` do not load a checkpoint or generate poses. Actual CPU
inference temporarily redirects the VQ constructor's hard-coded `.cuda()`
buffer allocation within this process, then strictly loads the saved weights.
CPU/GPU output equality is not presumed. Output manifests record the device,
files, per-ID seed, predicted length and actual generated frame count.

## Training target semantics

For reproducing training, `collate_actions`, `masked_ce` and
`length_target_loss` expose the unchanged numerical objectives. Only corrupted
action-token positions enter the denoising loss. EOS/BOS/PAD are not action
targets. `source_truncated` samples retain prefix action supervision but are
masked from text-length loss. Validation uses fixed corruption and selects the
lowest validation masked-token CE. The detailed settings are in `config.json`.
This directory supplies the model/objective/inference code, not a portable
training scheduler or pretrained checkpoint distribution. Recreating matching
training caches, text weights and quantizer weights is an explicit prerequisite.
