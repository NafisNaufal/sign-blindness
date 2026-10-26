# Echo quality assessment does not transfer from CAMUS to EchoNet

**Negative result, 2026-07-27.** Reported rather than buried: it is a useful caution, and it
is why EchoNet is used for scale rather than for quality stratification.

## Setup

A quality grader trained on CAMUS expert cardiologist grades reaches **Spearman rho = 0.664**
out-of-fold against those grades (`results/e3_quality/`). For reference, the published
open-source estimator reaches rho = 0.69 and **inter-observer agreement is rho = 0.63**
(arXiv 2408.00591). The grader is therefore about as consistent with the cardiologists as
they are with each other.

Applied to 20,048 EchoNet-Dynamic frames as a 5-fold ensemble, it fails on both checks.

## Failure 1: degenerate score distribution

| | mean | sd | p10 | p50 | p90 |
|---|---|---|---|---|---|
| EchoNet | 0.134 | 0.092 | 0.063 | 0.103 | 0.247 |

CAMUS reference means: **Good 0.322, Medium 0.799, Poor 1.219**. EchoNet's *90th percentile*
(0.247) still scores better than the CAMUS **Good** mean. The grader judges essentially every
EchoNet frame as better than good, which is not credible for 10,024 consecutive clinical
studies.

## Failure 2: the relationship inverts

On CAMUS, poorer quality goes with lower EF (39.5 vs 46.0), because sicker patients have worse
acoustic windows. On EchoNet the predicted-poor decile has *higher* EF (56.7 vs 55.6) and

```
Spearman(quality score, EF) = +0.063     (CAMUS: negative)
```

A sign inversion is not a calibration problem. The score is not measuring what it measured on
CAMUS.

## Ruled out: normalisation

Both caches scale to a comparable range (per-image max 0.995 in both). The distributions do
differ (median intensity 0.172 CAMUS vs 0.020 EchoNet), but an intensity offset would shift
the scores, not invert their correlation with EF.

## Likely cause

CAMUS frames are resampled from ~0.3 mm native pixels; EchoNet ships videos pre-resized to
**112x112**. Much of what makes an echocardiogram "poor" is fine-grained: speckle character,
endocardial border definition, near-field clutter texture. Downsampling to 112x112 removes
exactly those cues, and upsampling back to 256 for the grader restores resolution without
restoring information, yielding smooth frames the grader reads as clean.

## Consequence for the study

EchoNet cannot be stratified by predicted quality, so the powered *method* claim as originally
designed is not available. EchoNet's role changes to what it can support without quality
labels, which is arguably more valuable:

1. replicate the Dice / EF decoupling at 20x scale and a second institution;
2. determine what *does* predict catastrophic EF failure, given that Dice does not;
3. this transfer failure, reported as a finding.

CAMUS retains expert quality grades, biplane EF, and the well-powered E1 result
(catastrophic EF failure 38.3% vs 11.9%, OR 4.59, p = 1.6e-05).

## Caveat kept honest

We did not try histogram matching, CORAL-style feature alignment, or fine-tuning on
pseudo-labels. Those might partially rescue transfer. We report the failure of the direct
approach, not a proof that transfer is impossible. Given the sign inversion rather than a
mere offset, we judged the direct route unpromising enough to redirect effort.
