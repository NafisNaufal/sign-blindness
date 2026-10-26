"""E0b: does the quality gap look worse in the CLINICAL metric (EF) than in Dice?

Dice is a per-frame overlap score. EF is derived from four segmentations (2CH/4CH x ED/ES)
via Simpson's biplane, so errors compound and need not track Dice at all. This decides
whether the clinical story is stronger than the segmentation story.

EF is computed with the SAME pipeline on predictions and on expert contours, so the
characterised +8.3 calibration offset (docs/ef_pipeline_validation.md) cancels exactly.
"""
import sys, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import CamusCached
from src.unet import UNet
from src.camus import simpson_biplane_volume, ejection_fraction
from src.plotting import bootstrap_ci

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="results/e0_baseline/best.pt")
ap.add_argument("--width", type=int, default=32)
ap.add_argument("--tag", default="e0_baseline")
a = ap.parse_args()
R = Path(__file__).resolve().parents[1]
dev = "cuda"

model = UNet(1, 4, width=a.width).to(dev)
model.load_state_dict(torch.load(R / a.ckpt)); model.eval()

# predict every val+test frame, keyed by (patient, view, phase)
pred, gt, meta = {}, {}, {}
for split in ("validation", "testing"):
    ds = CamusCached(split)
    for i in range(len(ds)):
        b = ds[i]
        with torch.no_grad():
            p = model(b["image"][None].to(dev)).argmax(1)[0].cpu().numpy()
        k = (b["patient"], b["view"], b["phase"])
        pred[k] = p; gt[k] = b["label"].numpy()
        # resampling to 256^2 rescales the effective pixel size; volumes need true mm
        h, w = b["orig_shape"].tolist(); sy, sx = b["spacing"].tolist()
        meta[k] = ((sy * h / p.shape[0], sx * w / p.shape[1]),
                   b["quality"], b["ef_ref"], split)

rows = []
for pat in sorted({k[0] for k in pred}):
    need = [(pat, v, ph) for v in ("2CH", "4CH") for ph in ("ED", "ES")]
    if not all(k in pred for k in need): continue
    def ef_of(src):
        vols = {}
        for ph in ("ED", "ES"):
            m2, m4 = src[(pat,"2CH",ph)], src[(pat,"4CH",ph)]
            s2, s4 = meta[(pat,"2CH",ph)][0], meta[(pat,"4CH",ph)][0]
            vols[ph] = simpson_biplane_volume(m2==1, m4==1, s2, s4,
                                              la_2ch=(m2==3), la_4ch=(m4==3))
        return float(ejection_fraction(vols["ED"], vols["ES"])), vols["ED"], vols["ES"]
    ef_p, edv_p, esv_p = ef_of(pred)
    ef_g, edv_g, esv_g = ef_of(gt)
    q = meta[(pat,"4CH","ED")][1]
    rows.append({"patient":pat,"quality":q,"ef_pred":ef_p,"ef_gt":ef_g,
                 "ef_ref":meta[(pat,"4CH","ED")][2],
                 "edv_pred":edv_p,"edv_gt":edv_g,"esv_pred":esv_p,"esv_gt":esv_g,
                 "split":meta[(pat,"4CH","ED")][3]})
df = pd.DataFrame(rows).dropna(subset=["ef_pred","ef_gt"])
df["ef_err"] = (df.ef_pred - df.ef_gt).abs()
df.to_csv(R/"results"/a.tag/"ef_per_patient.csv", index=False)

print("="*72); print(f"E0b  EF ERROR BY QUALITY  (pred vs expert contours, same pipeline)  n={len(df)}"); print("="*72)
for q in ["Good","Medium","Poor"]:
    s = df[df.quality==q]["ef_err"].values
    if not len(s): continue
    m, lo, hi = bootstrap_ci(s)
    print(f"  {q:7s} n={len(s):3d}   EF MAE {m:5.2f}  95% CI [{lo:5.2f}, {hi:5.2f}]   median {np.median(s):5.2f}")
g = df[df.quality.isin(["Good","Medium"])]["ef_err"].values
p = df[df.quality=="Poor"]["ef_err"].values
if len(p) and len(g):
    from scipy.stats import mannwhitneyu
    _, pv = mannwhitneyu(g, p)
    print(f"\n  Good+Medium {g.mean():.2f} vs Poor {p.mean():.2f} EF pts")
    print(f"  ratio {p.mean()/max(g.mean(),1e-9):.2f}x   Mann-Whitney p = {pv:.3e}")
    print(f"\n  Dice gap was +0.032 (3.5% relative). EF gap is {100*(p.mean()/g.mean()-1):.0f}% relative.")
print("\n  catastrophic EF failures (|error| > 10 EF points):")
for q in ["Good","Medium","Poor"]:
    s = df[df.quality==q]["ef_err"].values
    if len(s): print(f"    {q:7s} {100*(s>10).mean():5.1f}%  ({int((s>10).sum())}/{len(s)})")
