# Sign Blindness and the Limits of Overlap Metrics for Ejection Fraction in Automated Echocardiography

Code, experiments and result tables for a paper accepted at **ICIIBMS 2026** (Okinawa, Japan).

Nafis Naufal Rahman, Dionisius Seraf Saputra, Novanto Yudistira. University of Brawijaya, Malang, Indonesia.

## What the paper shows

Automated left-ventricular segmentation is almost always scored with the Dice coefficient, but the clinical
output derived from it is ejection fraction (EF). Dice is blind to the *sign* of a boundary error, whereas EF,
a normalised difference of two volumes, is not.

- Perturbing expert contours with unbiased boundary noise alone reproduces the Dice/EF decoupling
  (R² 0.27 simulated vs 0.19 measured). End-diastolic and end-systolic errors of opposite sign give 2.0x the EF
  error of same-sign errors while being indistinguishable by Dice.
- Across two institutions and 10,524 patients (CAMUS and EchoNet-Dynamic), 7.5% of patients in the best Dice
  quartile still get an EF error above 10 points. Across four models spanning 4x capacity, Dice varies by 0.47%
  while the rate of such errors varies by 36.6%.
- On CAMUS, expert-graded poor acoustic windows raise the rate of large EF errors 3.3-fold
  (adjusted OR 4.89, 95% CI 2.49 to 9.60; N = 500) while mean Dice falls only 3.5% relative.
- CAMUS cannot power a robustness claim (207 poor-quality patients per arm needed, 47 available), and image-quality
  assessment trained on CAMUS does not transfer to EchoNet-Dynamic.

The paper recommends reporting clinical-endpoint error against a human-variability reference band alongside Dice.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | U-Net baseline, CAMUS dataset code, physics-based acoustic degradation (`acoustic.py`), generic augmentation control, quality grader, metrics, IEEE-style plotting |
| `experiments/` | One script per experiment (table below) |
| `scripts/` | Dataset download and cache builders, correctness tests |
| `results/` | Result tables, figures, logs and held-out CAMUS predictions |
| `paper/` | LaTeX source (`main.tex`), figures, `IEEEtran.cls` |
| `docs/` | Working notes: EF pipeline validation, a negative result, the research proposal |
| `reproducibility/` | Dependencies for the lightweight revision analysis (`e7`) |

### Experiments

| Script | Question |
|---|---|
| `e0_quality_gap.py`, `e0b_ef_gap.py` | Does the quality gap show up in Dice, and does it look worse in EF? |
| `e1_crossval.py` | 5-fold patient-level CV over all 500 CAMUS patients for adequate power |
| `e2_analyse.py` | Catastrophic EF failure rate on the Poor-quality stratum across augmentation arms |
| `e3_quality_net.py`, `e4a_quality_transfer.py` | Train a CAMUS quality grader and test its transfer to EchoNet (it does not transfer; see `docs/quality_transfer_failure.md`) |
| `e5_echonet_seg.py`, `e6_decoupling.py` | Does the Dice/EF decoupling replicate on EchoNet at ~20x the patients? |
| `e7_revisions.py` | Revision analyses: null-model sensitivity, signed errors, threshold curve, adjusted regression |
| `e8_model_sweep.py` | Does a higher-Dice model give lower EF error across models? |
| `make_figures.py` | Paper figures |

## Data

No clinical images are included.

- **CAMUS** (500 patients, expert quality grades): obtain from the CAMUS authors.
- **EchoNet-Dynamic**: obtain through Stanford AIMI under its Research Use Agreement.
  `scripts/download_echonet.py` fetches it from Redivis and needs your own `REDIVIS_API_TOKEN`.

Build the caches first: `scripts/build_cache.py` (CAMUS) and `scripts/build_echonet_cache.py` (EchoNet-Dynamic).
Both write under `data/`, which is git-ignored.

## What is deliberately not in this repository

- **Model checkpoints** (`*.pt`).
- **Per-video EchoNet-Dynamic outputs** (`results/e5_echonet/per_frame.csv`, `per_patient_ef.csv` and two files in
  `results/e7_revisions/`). They are keyed by the dataset's own video IDs. Regenerate them with
  `experiments/e5_echonet_seg.py` once you have the data. `e7_revisions.py` reads them, so it needs that step first.
- Reviewer correspondence and build artefacts.

## Notes on status

- This is the working code as it was run for the paper, copied into a clean repository. It was **not re-run end to
  end** when this repository was assembled; Python files were only syntax-checked.
- `reproducibility/requirements.txt` lists only the lightweight analysis dependencies (numpy, pandas, scipy,
  statsmodels). The training scripts additionally need PyTorch and the usual imaging stack; no pinned environment
  for them is recorded here.
- Server paths in the shipped logs and download script were replaced with `<repo>` / `./data/raw`.

## License

MIT, see [LICENSE](LICENSE). This covers the code and result tables in this repository only. CAMUS and
EchoNet-Dynamic remain under their own terms.

## Related

A companion repository by the second author contains the lightweight revision analysis only:
[yukienjoyer7/sign-blindness-reproducibility](https://github.com/yukienjoyer7/sign-blindness-reproducibility).
