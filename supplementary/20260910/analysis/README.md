# Coordinate-segment diagnostics and fixed offline postprocessing

These CPU tools accept external pose files. No dataset, generated sequence or
checkpoint is included. Install NumPy and PyTorch in the project environment.
Use only trusted `.pt`/pickle files because those formats deserialize Python
objects. JSON/JSONL and non-object NumPy inputs are also supported.

## E4 diagnostics

Run from the repository root, replacing the example paths with your own files:

```bash
python supplementary/20260910/analysis/skeleton_metrics_cpu.py \
  --pred generated/test.pickle --reference data/phix/phix_lift3d.train.pt \
  --reference-split train \
  --config supplementary/20260910/analysis/configurations/phix_segments.json \
  --out analysis_out/phix_raw
```

Use `csl_segments.json` and the CSL training reference for CSL. Inputs must
already be in the released coordinate system. The tool does not invert a
z-score automatically. PHX source order is body 0:8, right hand 8:29, left hand
29:50, face 50:178. CSL source order is body 0:8, face 8:136, left hand 136:157,
right hand 157:178. The configuration maps each to the same canonical order.

The main scope `kind:rigid` contains 4 upper/lower arm segments and 30 finger
segments. Ten wrist-to-palm distances and four trunk-frame distances are
reported separately as references. Face points are excluded from segment
length analysis. The label `rigid` denotes the operational segment set, not a
claim that the estimated input coordinates preserve physiological rigidity.

CV uses population standard deviation over time divided by mean segment
length. Segments are equally weighted within a sample, then samples are
equally weighted. Relative adjacent change is mean absolute first difference
of segment length divided by its temporal mean. Both are dimensionless.
The training bands for temporal-mean lengths and for individual frame lengths
are fitted separately. Their two exceedance rates have different denominators.
Zero-variance reference bands are excluded, with denominator counts retained.

Position speed and acceleration are coordinate units per observed frame step
and per frame step squared. Supply `--fps` only when the actual frame rate at
this input is established. Segment first differences stay per observed frame
step. Motion amplitude and acceleration are not universally better when lower.

## E6 fixed treatments

```bash
python supplementary/20260910/analysis/run_postprocess.py \
  --dataset phix --train data/phix/phix_lift3d.train.pt \
  --pred generated/test.pickle --out analysis_out/phix_post --diagnostics
```

The command writes both `bone_projection.pickle` and
`smooth3_then_projection.pickle`, a TRAIN reference, per-sample fallback events
and an input/output hash manifest. Output directories must be new or empty.
Sequences must be complete and unpadded. Sample IDs and frame counts are retained.

The projection fixes the 34 main segment lengths to the equal-sample TRAIN
means. It preserves the input shoulder roots, follows the original segment
directions, binds duplicate hand roots to body wrists, and retains the ten
pre-projection palm vectors. The second treatment first smooths body and hand
positions with `[0.25, 0.5, 0.25]` and replicated boundaries. Face coordinates
are unchanged in both treatments. CSL is converted to canonical order for
computation and restored to source order before serialization.

Zero directions use the nearest valid direction of the same sample and segment
(ties choose the past), then the training fallback if none exists. Every
fallback offset is recorded. This is offline processing: both smoothing and
fallbacks can use future frames. Reduced length CV is a constructed consequence
of fixed lengths, not independent evidence of natural or understandable motion.
Assess raw and processed outputs under the same evaluator and keep adverse
changes. The CSL application followed the E4 observation and used the same
fixed kernel without tuning. It is an applicability check, not a preregistered
new-model experiment.
