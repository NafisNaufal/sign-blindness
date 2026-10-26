"""Revision analyses: null sensitivity, signed errors, thresholds, and adjustment.

The exact null replay uses ``data/cache_echonet`` when available. This workspace retains
the prior exact sigma=1.6 run and the model CSVs, but not the raw cache, so sensitivity
analysis falls back to a concentric-boundary proxy derived from the saved phase areas.
The fallback is labelled in every output and is replaced automatically by exact masks
when the cache is restored.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

R = Path(__file__).resolve().parents[1]
OUT = R / "results" / "e7_revisions"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = R / "data" / "cache_echonet"
SIGMAS = (0.5, 1.0, 1.6, 2.0, 3.0)


# --------------------------------------------------------------------------- #
def disc_volume(mask, n_disks=20):
    pts = np.argwhere(mask > 0).astype(np.float64)
    if len(pts) < 20:
        return np.nan
    c = pts.mean(0)
    u = np.linalg.svd(pts - c, full_matrices=False)[2][0]
    n = np.array([-u[1], u[0]])
    t = (pts - c) @ u
    s = (pts - c) @ n
    t0, t1 = t.min(), t.max()
    L = t1 - t0
    if L < 4:
        return np.nan
    edges = np.linspace(t0, t1, n_disks + 1)
    d = np.zeros(n_disks)
    for i in range(n_disks):
        b = (t >= edges[i]) & (t < edges[i + 1])
        if b.any():
            d[i] = s[b].max() - s[b].min()
    return float(np.pi / 4 * (d ** 2).sum() * (L / n_disks))


def perturb(mask, signed_px, rng):
    """Morphologically grow (positive) or shrink (negative) the mask by ~signed_px."""
    from scipy.ndimage import binary_dilation, binary_erosion
    k = int(round(abs(signed_px)))
    if k == 0:
        return mask.copy()
    return (binary_dilation(mask, iterations=k) if signed_px > 0
            else binary_erosion(mask, iterations=k))


def _null_summary(df, sigma, variant, source):
    from scipy.stats import pearsonr

    r = pearsonr(df.dice, df.err).statistic
    same = df[df.same_sign].err
    opposite = df[~df.same_sign].err
    return {
        "sigma_px": sigma,
        "noise_variant": variant,
        "n_cases": len(df),
        "r_squared": r ** 2,
        "same_sign_fraction": df.same_sign.mean(),
        "opposite_sign_fraction": (~df.same_sign).mean(),
        "mean_same_sign_error": same.mean(),
        "mean_opposite_sign_error": opposite.mean(),
        "same_opposite_error_ratio": opposite.mean() / max(same.mean(), 1e-12),
        "source": source,
    }


def _exact_null_rows(sigma, seed, rho=0.0, n_patients=600):
    """Run the original contour perturbation on exact cached masks."""
    imgs_idx = pd.read_csv(CACHE / "index_112.csv")
    msks = np.load(CACHE / "masks_112.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)
    keys = imgs_idx.key.unique()[:n_patients]
    rows = []
    for key in keys:
        g = imgs_idx[imgs_idx.key == key].sort_values("phase")
        if set(g.phase) != {"ED", "ES"}:
            continue
        deltas = rng.multivariate_normal([0.0, 0.0],
                                          [[1.0, rho], [rho, 1.0]],
                                          size=1)[0] * sigma
        rec = {}
        for delta, (_, row) in zip(deltas, g.iterrows()):
            mask = np.array(msks[int(row.name)]) > 0
            perturbed = perturb(mask, delta, rng)
            v_gt, v_pred = disc_volume(mask), disc_volume(perturbed)
            if not np.isfinite(v_gt) or not np.isfinite(v_pred):
                rec = {}
                break
            inter = (mask & perturbed).sum()
            den = mask.sum() + perturbed.sum()
            rec[row.phase] = {
                "dice": 2 * inter / den,
                "v_gt": v_gt,
                "v_pred": v_pred,
                "delta": delta,
            }
        if len(rec) != 2:
            continue
        ef_gt = (rec["ED"]["v_gt"] - rec["ES"]["v_gt"]) / rec["ED"]["v_gt"] * 100
        ef_pred = (rec["ED"]["v_pred"] - rec["ES"]["v_pred"]) / rec["ED"]["v_pred"] * 100
        rows.append({
            "key": key,
            "dice": (rec["ED"]["dice"] + rec["ES"]["dice"]) / 2,
            "err": abs(ef_pred - ef_gt),
            "same_sign": np.sign(rec["ED"]["delta"]) == np.sign(rec["ES"]["delta"]),
        })
    return pd.DataFrame(rows)


def _proxy_null_rows(sigma, seed, rho=0.0):
    """Use saved phase areas as an explicitly labelled concentric-boundary proxy."""
    f = pd.read_csv(R / "results/e5_echonet/per_frame.csv")
    base = f[["key", "phase", "area_gt", "vol_px"]].pivot(index="key", columns="phase")
    rng = np.random.default_rng(seed)
    noise = rng.multivariate_normal([0.0, 0.0],
                                     [[1.0, rho], [rho, 1.0]],
                                     size=len(base)) * sigma
    rows = []
    for i, (key, row) in enumerate(base.iterrows()):
        areas = np.array([row[("area_gt", "ED")], row[("area_gt", "ES")]])
        volumes = np.array([row[("vol_px", "ED")], row[("vol_px", "ES")]])
        radii = np.sqrt(areas * 112 ** 2 / np.pi)
        scale = ((radii + noise[i]) / radii) ** 2
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            continue
        predicted = volumes * scale
        ef_gt = (volumes[0] - volumes[1]) / volumes[0] * 100
        ef_pred = (predicted[0] - predicted[1]) / predicted[0] * 100
        dice = 2 * np.minimum(1.0, scale) / (1.0 + scale)
        rows.append({
            "key": key,
            "dice": dice.mean(),
            "err": abs(ef_pred - ef_gt),
            "same_sign": np.sign(noise[i, 0]) == np.sign(noise[i, 1]),
        })
    return pd.DataFrame(rows)


def r1_null_model(n_patients=600, sigma=1.6, seed=0):
    """Perturb EXPERT masks with unbiased signed boundary noise; measure Dice vs EF error."""
    if CACHE.exists():
        df = _exact_null_rows(sigma, seed, n_patients=n_patients)
    else:
        path = OUT / "r1_null_model.csv"
        if not path.exists():
            raise FileNotFoundError("exact null cache and prior r1_null_model.csv are both missing")
        df = pd.read_csv(path)
    df.to_csv(OUT / "r1_null_model.csv", index=False)

    from scipy.stats import pearsonr, spearmanr
    r, _ = pearsonr(df.dice, df.err)
    rho, _ = spearmanr(df.dice, df.err)
    print("=" * 74)
    print(f"R1  NULL MODEL: unbiased boundary noise only, n={len(df)}")
    print("=" * 74)
    print(f"  mean Dice {df.dice.mean():.4f}   mean EF error {df.err.mean():.2f} pts")
    print(f"  Dice vs EF error:  R^2 = {r**2:.3f}   Spearman rho = {rho:+.3f}")
    print(f"  observed in the real model: R^2 = 0.194")
    print(f"  --> {'CONSISTENT with pure boundary noise' if abs(r**2-0.194)<0.15 else 'DIFFERS from pure boundary noise'}")
    # the mechanism: same-sign errors partially cancel in the EF ratio, opposite-sign compound
    a = df[df.same_sign].err.mean()
    b = df[~df.same_sign].err.mean()
    print(f"\n  MECHANISM (sign-blindness of Dice):")
    print(f"    ED/ES errors SAME sign      : EF error {a:.2f} pts  (n={int(df.same_sign.sum())})")
    print(f"    ED/ES errors OPPOSITE sign  : EF error {b:.2f} pts  (n={int((~df.same_sign).sum())})")
    print(f"    ratio {b/max(a,1e-9):.2f}x  <- Dice cannot distinguish these cases at all")
    return df


def null_sensitivity():
    """Run independent and empirically correlated null sensitivity analyses."""
    f = pd.read_csv(R / "results/e5_echonet/per_frame.csv")
    phase_error = f.assign(signed_error=f.area_pred / f.area_gt - 1)
    wide = phase_error.pivot(index="key", columns="phase", values="signed_error").dropna()
    rho = float(wide.ED.corr(wide.ES))
    exact = CACHE.exists()
    source = "exact_masks" if exact else "saved_phase_area_proxy"
    rows = []
    for sigma in SIGMAS:
        for variant, corr in (("independent", 0.0), ("empirically_correlated", rho)):
            df = (_exact_null_rows(sigma, int(1000 * sigma), corr)
                  if exact else _proxy_null_rows(sigma, int(1000 * sigma), corr))
            rows.append(_null_summary(df, sigma, variant, source))
    out = pd.DataFrame(rows)
    out["empirical_ed_es_signed_error_correlation"] = rho
    out.to_csv(OUT / "null_model_sensitivity.csv", index=False)
    print("\nNULL SENSITIVITY")
    print(out[["sigma_px", "noise_variant", "n_cases", "r_squared",
               "same_sign_fraction", "source"]].round(3).to_string(index=False))
    return out


def signed_error_analysis():
    """Report a sign-aware phase companion using the saved phase-area errors."""
    f = pd.read_csv(R / "results/e5_echonet/per_frame.csv")
    f["signed_phase_error"] = f.area_pred / f.area_gt - 1
    wide = f.pivot(index="key", columns="phase", values="signed_phase_error").dropna()
    ef = pd.read_csv(R / "results/e5_echonet/per_patient_ef.csv").set_index("key")
    out = wide.join(ef[["err"]]).dropna().rename(columns={"ED": "signed_ed_error",
                                                             "ES": "signed_es_error"})
    out["sign_aware_metric"] = out.signed_ed_error - out.signed_es_error
    out["same_sign"] = np.sign(out.signed_ed_error) == np.sign(out.signed_es_error)
    out.insert(0, "key", out.index)
    out["source"] = "saved_phase_area_proxy"
    out.to_csv(OUT / "signed_error_analysis.csv", index=False)
    from scipy.stats import pearsonr
    r = pearsonr(out.sign_aware_metric, out.err).statistic
    print(f"\nSIGNED COMPANION: n={len(out)}  Pearson r(metric, EF error)={r:+.3f}  R^2={r**2:.3f}")
    print(f"  empirical signed ED/ES correlation={out.signed_ed_error.corr(out.signed_es_error):+.3f}; "
          f"same-sign fraction={out.same_sign.mean():.3f}")
    return out


def threshold_curve():
    """Write the large-EF-error rate for thresholds 0 through 20 EF points."""
    ef = pd.read_csv(R / "results/e5_echonet/per_patient_ef.csv")
    rows = [{"threshold_points": t,
             "n_cases": len(ef),
             "large_ef_error_count": int((ef.err > t).sum()),
             "large_ef_error_rate_percent": 100 * (ef.err > t).mean()}
            for t in range(21)]
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "threshold_curve.csv", index=False)
    print(f"\nTHRESHOLD CURVE: n={len(ef)}; at 10 points "
          f"{out.loc[out.threshold_points == 10, 'large_ef_error_rate_percent'].iat[0]:.1f}%")
    return out


def r3_confound():
    """Adjust the quality-to-failure association for reference EF."""
    ca = pd.read_csv(R / "results/e2_ef_all_arms.csv")
    ca = ca[ca.arm == "none"].copy()
    ca["fail"] = (ca.err > 10).astype(int)
    ca["poor"] = (ca.quality == "Poor").astype(int)
    import statsmodels.api as sm
    X0 = sm.add_constant(ca[["poor"]].astype(float))
    m0 = sm.Logit(ca.fail, X0).fit(disp=0)
    X1 = sm.add_constant(ca[["poor", "ef_gt"]].astype(float))
    m1 = sm.Logit(ca.fail, X1).fit(disp=0)
    print("\n" + "=" * 74)
    print("R3  QUALITY -> FAILURE, ADJUSTED FOR DISEASE SEVERITY (reference EF)")
    print("=" * 74)
    print(f"  unadjusted   OR(poor) = {np.exp(m0.params['poor']):.2f}   "
          f"p = {m0.pvalues['poor']:.4f}")
    print(f"  EF-adjusted  OR(poor) = {np.exp(m1.params['poor']):.2f}   "
          f"p = {m1.pvalues['poor']:.4f}")
    print(f"  EF coefficient        = {m1.params['ef_gt']:+.4f}  "
          f"p = {m1.pvalues['ef_gt']:.4f}")
    surv = m1.pvalues['poor'] < 0.05
    print(f"  --> quality effect {'SURVIVES adjustment' if surv else 'is CONFOUNDED by severity'}")
    ci = m1.conf_int().loc["poor"]
    result = pd.DataFrame([{
        "n_patients": len(ca),
        "outcome": "large EF error (>10 points)",
        "predictors": "poor quality + reference EF",
        "adjusted_or": np.exp(m1.params["poor"]),
        "ci_95_low": np.exp(ci.iloc[0]),
        "ci_95_high": np.exp(ci.iloc[1]),
        "p_value": m1.pvalues["poor"],
        "reference_ef_p_value": m1.pvalues["ef_gt"],
    }])
    result.to_csv(OUT / "r3_confound.csv", index=False)
    with open(OUT / "r3_confound.txt", "w") as f:
        f.write(str(m1.summary()))
    print(f"  95% CI for adjusted OR = [{np.exp(ci.iloc[0]):.2f}, {np.exp(ci.iloc[1]):.2f}]")
    return result


if __name__ == "__main__":
    r1_null_model()
    null_sensitivity()
    signed_error_analysis()
    threshold_curve()
    r3_confound()
