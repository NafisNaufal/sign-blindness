# Sign Blindness Reproducibility Analysis

This repository contains the lightweight analyses used for the revised ICIIBMS
paper, including the null-model sensitivity sweep, sign-aware phase-error
companion, threshold curve, and adjusted quality regression.

## Reproduce

```bash
python -m pip install -r requirements.txt
python experiments/e7_revisions.py
```

The script writes its outputs to `results/e7_revisions/`. The included tabular
inputs are derived from the held-out predictions used in the paper. Raw
ultrasound data, segmentation masks, and model checkpoints are not included.
When the exact mask cache is unavailable, the null-model sweep automatically
uses the explicitly labelled saved phase-area proxy included in the analysis.

## Contents

- `experiments/e7_revisions.py`: revision-analysis implementation.
- `results/e5_echonet/`: held-out EchoNet phase and patient summaries used as
  inputs.
- `results/e2_ef_all_arms.csv`: CAMUS quality and EF summary used for the
  adjusted regression.
- `results/e7_revisions/`: reported analysis outputs.

The repository is intended to make the analyses and reported CSV results
inspectable without distributing the underlying clinical image data.
