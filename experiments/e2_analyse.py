"""E2 verdict: catastrophic EF failure rate on the Poor stratum, across augmentation arms.

Primary endpoint is EF, not Dice, following the E1 finding that overlap metrics are nearly
blind to clinical failure. Ground-truth EF is identical across arms, so it is computed once
and shared.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.camus import simpson_biplane_volume, ejection_fraction
from src.plotting import bootstrap_ci
from scipy.stats import mannwhitneyu

R = Path(__file__).resolve().parents[1]
ARMS = {"none": "e1_cv_baseline", "generic": "e2_generic", "acoustic": "e2_acoustic"}
idx = pd.read_csv(R/"data/cache/index_256.csv")
labs = np.load(R/"data/cache/labels_256.npy", mmap_mode="r")
key = {(r["patient"], r["view"], r["phase"]): (i, r) for i, r in idx.iterrows()}

def spacing(k):
    _, r = key[k]
    return (r["sp_y"]*r["h"]/256.0, r["sp_x"]*r["w"]/256.0)

def ef_for(pat, get):
    v = {}
    for ph in ("ED", "ES"):
        m2, m4 = get((pat,"2CH",ph)), get((pat,"4CH",ph))
        if m2 is None or m4 is None: return np.nan
        v[ph] = simpson_biplane_volume(m2==1, m4==1, spacing((pat,"2CH",ph)),
                                       spacing((pat,"4CH",ph)),
                                       la_2ch=(m2==3), la_4ch=(m4==3))
    return float(ejection_fraction(v["ED"], v["ES"]))

pats = sorted(idx.patient.unique())
print("computing ground-truth EF once...", flush=True)
gt_ef = {p: ef_for(p, lambda k: np.array(labs[key[k][0]]) if k in key else None) for p in pats}

rows = []
for arm, tag in ARMS.items():
    P = {}
    for f in range(5):
        z = np.load(R/"results"/tag/f"preds_fold{f}.npz")
        for k in z.files:
            a, b, c = k.split("|"); P[(a,b,c)] = z[k]
    print(f"  {arm}: {len(P)} frames", flush=True)
    for p in pats:
        e = ef_for(p, lambda k: P.get(k))
        if not np.isfinite(e) or not np.isfinite(gt_ef[p]): continue
        rows.append({"arm":arm, "patient":p, "quality":key[(p,"4CH","ED")][1]["quality"],
                     "ef_pred":e, "ef_gt":gt_ef[p], "err":abs(e-gt_ef[p])})
df = pd.DataFrame(rows); df.to_csv(R/"results/e2_ef_all_arms.csv", index=False)

print("\n" + "="*78)
print("E2 VERDICT  --  primary endpoint: catastrophic EF failure (>10 pts) on POOR stratum")
print("="*78)
print(f"{'arm':10s} {'stratum':8s} {'n':>4s} {'EF MAE':>8s} {'95% CI':>16s} {'catastrophic':>13s}")
for arm in ARMS:
    for q, sel in [("Poor", lambda d: d.quality=="Poor"),
                   ("Good+Med", lambda d: d.quality.isin(["Good","Medium"]))]:
        s = df[(df.arm==arm) & sel(df)]["err"].values
        if not len(s): continue
        m, lo, hi = bootstrap_ci(s)
        print(f"{arm:10s} {q:8s} {len(s):4d} {m:8.2f}  [{lo:5.2f},{hi:5.2f}]   {100*(s>10).mean():11.1f}%")
    print()

base = df[(df.arm=="none") & (df.quality=="Poor")]["err"].values
for arm in ("generic", "acoustic"):
    a = df[(df.arm==arm) & (df.quality=="Poor")]["err"].values
    _, pv = mannwhitneyu(base, a)
    print(f"POOR: {arm:9s} vs none -> MAE {base.mean():.2f} -> {a.mean():.2f} "
          f"({100*(a.mean()/base.mean()-1):+.1f}%)  catastrophic "
          f"{100*(base>10).mean():.1f}% -> {100*(a>10).mean():.1f}%   p={pv:.3f}")
g = df[(df.arm=="generic") & (df.quality=="Poor")]["err"].values
ac = df[(df.arm=="acoustic") & (df.quality=="Poor")]["err"].values
_, pv = mannwhitneyu(g, ac)
print(f"\nHEAD-TO-HEAD acoustic vs generic on POOR: {g.mean():.2f} vs {ac.mean():.2f}  p={pv:.3f}")
print("  -> the polar-geometry premise stands only if acoustic beats generic here.")
