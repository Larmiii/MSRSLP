# Supplementary generator interfaces

- [Local masked generator](dfm/README.md): actual grouped model, action-token
  objectives, text-length conversion and fixed50-step inference for a matching
  external PHX PAMRVQ checkpoint.
- [Adapted references](references/README.md): actual MoMask, MDM and
  MotionGPT-style inference rules, TRAIN-only length-predictor preparation and
  explicit checkpoint/text-resource manifests.

These components extend the supplement's main runtime. They require separately
obtained datasets, trained weights and text-model assets. Their presence does
not make an untrained checkout reproduce checkpoint scores automatically.
The references use heterogeneous training and text conditions. The masked
generator experiment verifies an interface with another generator, whose
architecture and length mechanism differ from AR.

Only source code, configurations and instructions are included here. Keep local
resource manifests, sample texts, generated poses and checkpoints outside the
tracked repository when running the commands.

CPU checks without trained weights:

```bash
python supplementary/20260910/advanced/dfm/test_objectives.py
python supplementary/20260910/advanced/references/test_decoding.py
```
