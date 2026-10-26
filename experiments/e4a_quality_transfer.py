"""Apply the CAMUS-trained quality grader to EchoNet. Gating check for E4.

Risk being tested: EchoNet frames are 112x112, far coarser than CAMUS's native ~0.3mm
pixels. Some of what makes a scan "poor" is fine texture that may not survive that
downsampling, in which case the grader's scores collapse and stratifying on them would be
stratifying on noise.

Two things must hold:
  1. The score distribution must be non-degenerate (real spread, not a spike).
  2. It must recover a relationship established independently on CAMUS: poorer quality
     goes with lower EF (sicker patients have worse acoustic windows).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.quality import QualityNet, logits_to_grade

R = Path(__file__).resolve().parents[1]; dev = "cuda"
C = R/"data/cache_echonet"
imgs = np.load(C/"images_112.npy", mmap_mode="r")
idx = pd.read_csv(C/"index_112.csv")

models = []
for f in range(5):
    m = QualityNet(width=24).to(dev)
    m.load_state_dict(torch.load(R/f"results/e3_quality/qnet_fold{f}.pt")); m.eval()
    models.append(m)
print(f"[transfer] {len(idx)} EchoNet frames, ensemble of {len(models)} CAMUS folds", flush=True)

scores = np.zeros(len(idx), dtype=np.float32)
B = 512
with torch.no_grad():
    for i in range(0, len(idx), B):
        x = torch.from_numpy(np.array(imgs[i:i+B], dtype=np.float32))[:, None].to(dev)
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        s = torch.stack([logits_to_grade(m(x).float()) for m in models]).mean(0)
        scores[i:i+B] = s.cpu().numpy()
        if i % 5120 == 0: print(f"  ...{i}/{len(idx)}", flush=True)
idx["q_score"] = scores
idx.to_csv(C/"index_112_quality.csv", index=False)

print("\n" + "="*66); print("QUALITY TRANSFER: CAMUS grader -> EchoNet"); print("="*66)
print(f"  score range [{scores.min():.3f}, {scores.max():.3f}]  mean {scores.mean():.3f}  sd {scores.std():.3f}")
print(f"  CAMUS reference means: Good 0.322 / Medium 0.799 / Poor 1.219")
q = np.percentile(scores, [10,25,50,75,90])
print(f"  percentiles 10/25/50/75/90: {np.round(q,3).tolist()}")
degenerate = scores.std() < 0.10
print(f"  --> distribution {'DEGENERATE (transfer failed)' if degenerate else 'has real spread'}")

# patient-level: worst-quality decile vs rest, does EF track as on CAMUS?
pv = idx.groupby("key").agg(q=("q_score","mean"), ef=("ef_ref","first"), split=("split","first"))
thr = pv.q.quantile(0.90)
poor, rest = pv[pv.q >= thr], pv[pv.q < thr]
from scipy.stats import mannwhitneyu, spearmanr
u, p = mannwhitneyu(rest.ef, poor.ef)
rho, prho = spearmanr(pv.q, pv.ef)
print(f"\n  patients: {len(pv)}   predicted-poor decile n={len(poor)}")
print(f"  EF: rest {rest.ef.mean():.1f}  vs predicted-poor {poor.ef.mean():.1f}  p={p:.2e}")
print(f"  Spearman(quality score, EF) = {rho:+.3f}  p={prho:.2e}")
print(f"  CAMUS showed the same sign: poor quality -> lower EF (39.5 vs 46.0)")
print(f"  --> sanity check {'PASSED' if rho < 0 and p < 0.05 else 'FAILED / inconclusive'}")
print(f"\n  n available for method claim at 10% cut: {len(poor)} patients (needed ~236/arm)")
