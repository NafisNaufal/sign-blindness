"""E6: does the Dice / EF decoupling replicate on EchoNet at 20x scale?

CAMUS (n=500) showed a 3.5% relative Dice drop accompanying a 233% relative increase in
catastrophic EF failure. If that reproduces at n~10,000 and a second institution, the claim
stops being a single-dataset observation.

EF from a mask uses the same monoplane disc construction the expert tracings encode, applied
identically to predicted and ground-truth masks, so the measurement cancels out and only
segmentation quality is being scored.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, Dataset
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.unet import UNet
from src.plotting import bootstrap_ci
from scipy.stats import spearmanr, pearsonr

R = Path(__file__).resolve().parents[1]; dev="cuda"
C = R/"data/cache_echonet"
imgs = np.load(C/"images_112.npy", mmap_mode="r")
msks = np.load(C/"masks_112.npy", mmap_mode="r")
idx  = pd.read_csv(C/"index_112.csv")

def disc_volume(mask, n_disks=20):
    """Monoplane Simpson from a binary mask; PCA long axis (EchoNet has no atrium label)."""
    pts = np.argwhere(mask > 0).astype(np.float64)
    if len(pts) < 20: return np.nan
    c = pts.mean(0)
    u = np.linalg.svd(pts - c, full_matrices=False)[2][0]
    n = np.array([-u[1], u[0]])
    t = (pts - c) @ u; s = (pts - c) @ n
    t0, t1 = t.min(), t.max(); L = t1 - t0
    if L < 4: return np.nan
    edges = np.linspace(t0, t1, n_disks + 1); d = np.zeros(n_disks)
    for i in range(n_disks):
        b = (t >= edges[i]) & (t < edges[i+1])
        if b.any(): d[i] = s[b].max() - s[b].min()
    return float(np.pi/4 * (d**2).sum() * (L/n_disks))

# predict every VAL+TEST frame
model = UNet(1,2,width=32).to(dev)
model.load_state_dict(torch.load(R/"results/e5_echonet/best.pt")); model.eval()
ev = np.asarray(idx.index[idx.split.isin(["VAL","TEST"])])
print(f"[E6] scoring {len(ev)} frames", flush=True)

rows=[]
B=256
with torch.no_grad():
    for i in range(0, len(ev), B):
        j = ev[i:i+B]
        x = torch.from_numpy(np.array(imgs[j], dtype=np.float32))[:,None].to(dev)
        p = model(x).argmax(1).cpu().numpy()
        g = np.array(msks[j])
        for k in range(len(j)):
            inter = ((p[k]==1)&(g[k]==1)).sum(); den = (p[k]==1).sum()+(g[k]==1).sum()
            rows.append({"row": int(j[k]),
                         "dice": 1.0 if den==0 else 2*inter/den,
                         "v_pred": disc_volume(p[k]==1), "v_gt": disc_volume(g[k]==1)})
        if i % 2560 == 0: print(f"  ...{i}/{len(ev)}", flush=True)
f = pd.DataFrame(rows).join(idx, on="row", rsuffix="_i")

# per patient: EF from ED/ES volumes, both sides through the same pipeline
pt=[]
for key, g in f.groupby("key"):
    if len(g) != 2 or g[["v_pred","v_gt"]].isna().any().any(): continue
    ef = lambda col: (g[col].max()-g[col].min())/g[col].max()*100
    pt.append({"key":key, "ef_pred":ef("v_pred"), "ef_gt":ef("v_gt"),
               "ef_ref":g.ef_ref.iloc[0], "dice":g.dice.mean(), "dice_min":g.dice.min()})
d = pd.DataFrame(pt).dropna()
d["err"] = (d.ef_pred - d.ef_gt).abs()
d.to_csv(R/"results/e5_echonet/per_patient_ef.csv", index=False)

print("\n"+"="*72); print(f"E6  DICE / EF DECOUPLING ON ECHONET   n={len(d)} patients"); print("="*72)
m,lo,hi = bootstrap_ci(d.err.values)
print(f"  segmentation Dice : {d.dice.mean():.4f}")
print(f"  EF MAE            : {m:.2f}  95% CI [{lo:.2f},{hi:.2f}]")
print(f"  catastrophic (>10): {100*(d.err>10).mean():.1f}%  ({int((d.err>10).sum())}/{len(d)})")
rho,p1 = spearmanr(d.dice, d.err); r,p2 = pearsonr(d.dice, d.err)
print(f"\n  Spearman(Dice, EF error) = {rho:+.3f}  p={p1:.2e}")
print(f"  Dice explains only R^2 = {r**2:.3f} of EF-error variance")
hi_d = d[d.dice >= d.dice.quantile(0.75)]
print(f"\n  THE DECOUPLING: among the top-quartile-Dice patients (Dice>={d.dice.quantile(0.75):.3f}),")
print(f"    {100*(hi_d.err>10).mean():.1f}% still have catastrophic EF error "
      f"({int((hi_d.err>10).sum())}/{len(hi_d)})")
cat = d[d.err>10]
print(f"  Conversely, catastrophic-EF patients have mean Dice {cat.dice.mean():.4f} "
      f"vs {d[d.err<=10].dice.mean():.4f} for the rest")
print(f"    -> a {d[d.err<=10].dice.mean()-cat.dice.mean():.4f} Dice difference separates "
      f"clinically fine from clinically unusable")
print(f"\n  CAMUS comparison: catastrophic 14.4% overall, Dice barely moved.")
print(f"  --> decoupling {'REPLICATES' if (hi_d.err>10).mean() > 0.03 and r**2 < 0.35 else 'does NOT replicate'}")
