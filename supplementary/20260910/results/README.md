# Supplemental results — 2026-09-10

This directory contains an independent supplemental study. Its configurations and outputs are not the protocol or output set of the earlier release main table. Earlier release directories remain separate.

## Files

- `protocol.json`, `runs.json`: conditions, anatomical mappings and all 22 trained run records.
- `reconstruction.json` / `.csv`: 44 complete dev/test records, including full and base-only decoding where available. Position and velocity metrics retain all anatomical groups.
- `generation.json` / `.csv`: all 26 completed dev/test generation conditions, including the E1 repeats. JSON also retains stop counts, frame lengths and runtime conditions.
- `repeated_comparisons.json` / `.csv`: E1 grouping and E2 storage comparisons, with each seed, mean, sample SD and paired differences.
- `uncertainty.json`: paired sample intervals for the main seed-42 configurations and E1. E1 supplemental intervals were added after some results were known. They use shared whole-sample resampling and do not resample training seeds or concatenate corpora. No multiple-comparison adjustment was applied.
- `skeleton.json` / `.csv`: all ten full-test GT/reconstruction/generation conditions. JSON includes the full aggregate diagnostic scopes, not only the 34 primary arm/finger segments.
- `postprocess.json` / `.csv`: both datasets, both models, both splits, and all three raw/projection/smoothing-plus-projection conditions. JSON includes full aggregate motion and segment statistics.
- `dfm_interface.json` / `.csv`: PHX dev/test masked-generator results; the AR comparator is `phix_pamrvq_s42` in `generation.json`.
- `adapted_references.json` / `.csv`: all 12 frozen-weight adapted-reference conditions and seven language metrics.
- `provenance.json`: SHA256 of original input records, identified by neutral IDs. `checksums.json` separately hashes the public exports. `validation.json` records completed export checks.

JSON preserves original numeric precision. CSV contains convenient flat views. Null is used only for an inapplicable or absent source field with an accompanying note. No raw text, pose sequences, dataset payloads, model weights or thesis files are included.

## Interpretation and reproduction

E1 uses one fixed random partition, not a distribution of random partitions. Both systems use six tokens and the same codebook/nominal-bit budget. Overall and hand reconstruction improve while face reconstruction worsens. The E1 sample interval for the mean BLEU difference includes zero. Training SD is not a confidence interval.

E2 matches stored codeword counts, not token count or nominal bits: PHX uses 1024 versus 512+512 (10 versus 18 bits), CSL uses 4096 versus 2048+2048 (12 versus 22 bits). The CSL single-level comparison is the 4096-code baseline configuration.

Segment CV and adjacent relative length change are dimensionless, averaged equally across segments within each sample and then equally across samples. Mean-length and per-frame 3-sigma statistics have different denominators. Coordinate metrics are not millimeters. CSL exhibits lower generated CV but greater adjacent length change. These diagnostics do not establish perceptual naturalness or comprehensibility.

Postprocessing is offline. Projection targets are TRAIN-derived; the smoothing kernel is fixed at [0.25, 0.5, 0.25]. CSL is a posthoc applicability extension with the same algorithm. Projection can increase acceleration, and CSL baseline backtranslation worsens after treatment. All favorable and unfavorable conditions are retained; raw generation scores are not replaced.

The masked-generator comparison changes layer count, text limit, objective and length prediction as well as generator type. It is an interface experiment, not a parameter-matched architecture ranking. DFM scores are lower than the corresponding AR scores in this study.

Adapted references use original training seed 3407 and heterogeneous original training conditions/text encoders. Their training cap is 196 frames and inference cap is 320, so longer generation is extrapolation. MoMask native residual -1 contributes a zero vector, not codeword zero. MotionGPT-style generation constrains legal layer vocabularies and full-group EOS; cap termination is distinct from natural EOS. These are adapted references, not complete official reproductions. Exact frozen-weight inference requires the corresponding checkpoints; weights are not distributed here. Retraining through the supplied code does not guarantee identical scores.

WER is a percentage and may exceed 100. Language metrics describe the fixed backtranslation evaluator, not human assessment. Use the code and setup instructions in the parent supplementary directory with locally obtained datasets and checkpoints.
