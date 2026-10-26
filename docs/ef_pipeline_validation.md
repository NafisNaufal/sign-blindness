# EF measurement pipeline: validation and known calibration offset

Ejection fraction is the clinical output of this project, so the instrument that measures it
was validated against expert ground truth **before** being used to judge any model. Validated
on all 500 CAMUS patients using the **expert contours**, so any error here is pipeline error,
not segmentation error.

## Two implementations

| Version | r | MAE | bias | resid. sd |
|---|---|---|---|---|
| v1, long axis assumed vertical | 0.9008 | 4.71 | +2.80 | — |
| v2, anatomical long axis (mitral annulus midpoint to apex) | **0.9765** | 8.21 | +8.20 | **2.61** |

v1 assumed the LV long axis runs down the image y-axis. Apical echo views are routinely
tilted, so disc diameters were measured off-axis. v2 locates the base from the LV-LA interface
(left atrium is label 3 in CAMUS) and the apex as the furthest LV point from it, then measures
diameters perpendicular to that axis, with a PCA fallback when the atrium is absent.

v2 has a much better correlation but a larger raw MAE, which looks contradictory until the
error is decomposed.

## The offset is additive, not noise

```
EF_calc = 0.9975 * EF_ref + 8.312      r = 0.9765
residual sd after linear fit = 2.61 EF points
```

Slope is indistinguishable from 1 and the bias is flat across the whole EF range:

| EF_ref bin | n | bias |
|---|---|---|
| 0-30 | 67 | +8.06 |
| 30-40 | 104 | +8.19 |
| 40-50 | 169 | +8.42 |
| 50-60 | 127 | +8.33 |
| 60-100 | 33 | +6.92 |

Mechanism: our ESV/EDV ratio is **0.474** against the reference-implied **0.556**, i.e. ESV is
systematically underestimated relative to EDV. Since EF = 1 - ESV/EDV, that alone accounts for
the ~8-point offset. Absolute volumes are physiologically plausible (mean EDV 100.2 mL, mean
ESV 49.7 mL).

The most likely residual cause is disc-band quantisation: with a fixed 20 discs, the shorter
end-systolic long axis leaves more bands sparsely populated, truncating ES volume more than ED
volume. The CAMUS reference EF may also come from the original clinical report rather than from
Simpson's method applied to these particular contours.

## Protocol adopted

1. **Model EF error is computed with the same pipeline on both sides**: EF from predicted masks
   versus EF from expert masks. A constant offset cancels exactly, so this measures segmentation
   quality and nothing else. This is the headline EF metric.
2. **Agreement with the clinical reference EF** is reported separately, with the +8.3 offset
   stated explicitly and Bland-Altman limits of agreement given.
3. Never subtract the offset silently to make numbers look better.

With r = 0.977 and residual sd 2.61 EF points, the instrument is precise enough to resolve the
differences this study needs to detect. Precision, not absolute calibration, is what protocol 1
depends on.

Raw data: `results/ef_pipeline_validation.csv`. Reproduce with the validation block in
`src/camus.py` (`simpson_biplane_volume`, `lv_long_axis`, `disc_diameters`).
