# Research Proposal v2 — Quality-Robust Echocardiography

**Venue:** ICIIBMS 2026, Track 3. IEEE 2-column, max 8 pages, double-blind.

---

## 1. Clinical problem

Automated echocardiographic analysis is **validated on good-quality images but deployed on all
images**. Poor acoustic windows are routine in exactly the patients who most need imaging: obese
patients, COPD, mechanical ventilation, ICU. CAMUS grades **84 of 450 training patients (18.7%)
as poor quality** by an expert cardiologist, which matches clinical experience.

The failure mode that matters is not that accuracy is lower. It is that accuracy is lower
**silently**, on a patient subgroup that is identifiable in advance.

Ejection fraction is the clinical output, and it is derived from segmentations at two frames
(ED and ES). Errors therefore compound and need not track Dice: a model can look acceptable on
overlap metrics while being clinically unusable on EF. We measure both and report the decoupling.

## 2. Load-bearing assumption, tested FIRST

> **A1: segmentation and EF accuracy are materially worse on expert-graded poor-quality
> echocardiograms than on good/medium ones.**

Unlike the previous project's premise, A1 is close to physically guaranteed (poor acoustic
windows produce genuinely ambiguous endocardial borders). But it is still measured before any
method is built, and the size of the gap determines whether there is anything worth closing.

**E0 (gap quantification).** Train a standard segmentation baseline on CAMUS, evaluate
stratified by expert quality grade. Report Dice, Hausdorff, and **EF error / bias** per stratum
with bootstrap CIs.

- If the poor-quality Dice gap is negligible, this framing dies and we pivot immediately.
- Expected: a clear gap. The original CAMUS paper already hints at one.

## 3. Proposed contribution

**Physics-grounded acoustic degradation for quality-robust echo analysis.**

Rather than generic augmentation (brightness, blur, elastic), simulate the *actual* mechanisms
that make echocardiograms poor:

| Mechanism | Physical cause | Simulation |
|---|---|---|
| Acoustic shadowing | ribs, lung, calcification | angular attenuation cones from the probe apex |
| Dropout / signal loss | poor probe contact, obesity | spatially correlated attenuation fields |
| Speckle | coherent interference | multiplicative Rayleigh/Gamma noise |
| Depth attenuation | absorption with depth | range-dependent gain |
| Reverberation / clutter | multipath | low-frequency additive haze near the apex |

Because echo geometry is a sector originating at the probe, these are **directional in polar
coordinates**, not isotropic in image space. That is the distinction from generic augmentation
and is what should make it work.

**Falsifiable test:** does training with simulated degradation on good-quality images improve
performance on **real, held-out, expert-graded poor-quality** patients? Real poor-quality data is
the ground truth the simulator is judged against, and it is never used to fit the simulator.

Secondary: quality-aware uncertainty for the residual failures, i.e. flagging scans a human
should review, evaluated as a selective-prediction curve.

## 4. Experimental design

**Data.** CAMUS, 500 patients, public API (no account needed). 2CH + 4CH, ED + ES, expert
contours for LV endocardium / myocardium / left atrium, EF values, and expert quality grades.
Official 10-fold splits shipped in `database_split`.

**Baselines.** U-Net, nnU-Net-style, a transformer segmentor, plus generic-augmentation controls
(the honest comparison is against strong generic augmentation, not against no augmentation).

**Metrics.** Dice, Hausdorff-95, mean absolute surface distance; **EF MAE, bias, limits of
agreement (Bland-Altman)**; correlation with expert EF. All stratified by quality grade, all
with bootstrap CIs and paired tests.

**Ablations reviewers will ask for.** Each degradation mechanism separately; polar vs Cartesian
application; severity sweep; simulated-only vs mixed training; performance on good quality
(does robustness cost accuracy where it already worked?).

## 5. Why this survives the previous failure mode

The prior project depended on an untested empirical claim about heterogeneity that turned out
false. Here the core measurement (E0) is near-guaranteed, so the paper has a result even in the
worst case: if the degradation simulator does not help, the quality-stratified benchmark plus the
Dice/EF decoupling analysis is still a publishable, clinically useful finding.

## 6. Figure plan

1. **Teaser.** Real good vs real poor CAMUS frames with predicted contours; the failure is visible.
2. **The gap.** Dice and EF error by quality stratum, with CIs.
3. **Simulator.** Real poor-quality alongside our simulated degradation of a good-quality scan.
4. **Main result.** Poor-quality performance, ours vs baselines.
5. **Bland-Altman** for EF, stratified by quality.
6. **Ablation** over degradation mechanisms.
7. **Selective prediction** curve for the flagging component.

## 7. Status

- [x] Direction chosen
- [ ] Data downloaded and verified
- [ ] **E0 gap quantification** ← gates everything
- [ ] Degradation simulator
- [ ] Main experiments

---

## E0/E1 RESULTS (2026-07-27): A1 CONFIRMED, and the story is stronger than expected

5-fold cross-validation, all 500 patients evaluated out-of-fold (2000 frames).

**Segmentation gap is small:**

| Quality | n frames | Dice |
|---|---|---|
| Good | 1010 | 0.8993 |
| Medium | 758 | 0.8899 |
| Poor | 232 | 0.8700 |

**Clinical gap is large:**

| Quality | n patients | EF MAE | 95% CI | catastrophic (>10 EF pts) |
|---|---|---|---|---|
| Good | 288 | 5.02 | [4.49, 5.60] | 11.5% |
| Medium | 165 | 5.23 | [4.49, 6.01] | 12.7% |
| **Poor** | 47 | **7.35** | [5.76, 8.96] | **38.3%** |

Good+Medium 5.10 vs Poor 7.35, ratio **1.44x**, p = 0.0118.

### The headline: overlap metrics hide clinical failure

A **3.5% relative** degradation in Dice corresponds to a **233% relative increase** in
catastrophic EF failure (11.5% -> 38.3%, a 3.3x rate ratio). More than one in three
poor-quality echocardiograms yields a clinically unusable ejection fraction, while the
Dice score that the literature routinely reports barely moves.

This is the Dice/EF decoupling predicted in section 1, and it is much stronger than
anticipated. It reframes the paper: the contribution is not "poor images are harder to
segment" but **"the metric the field reports is nearly blind to the failure that matters."**

### Methodological note, kept as a warning

On the official split alone (n=8 Poor patients with complete 2CH/4CH x ED/ES coverage) the
EF gap read 1.08x, p = 0.50, i.e. **not significant**. That was a false negative from
insufficient power. Cross-validation raised the Poor stratum from 8 to 47 and the same
effect became 1.44x at p = 0.0118. Do not conclude absence of effect from the official
split for any patient-level (as opposed to frame-level) endpoint.
