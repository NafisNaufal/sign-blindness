"""5-fold CV over all 500 patients: the quality gap with adequate statistical power.

The official split leaves only 8 Poor-quality patients with complete 2CH/4CH x ED/ES
coverage, which cannot support a conclusion about EF in either direction. Cross-validation
evaluates every patient out-of-fold, raising the Poor stratum to ~47 for EF and ~116 for
Dice. Folds are split BY PATIENT so no patient's other view or phase leaks into training.
"""
import sys, argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, Subset
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import CamusCached
from src.unet import UNet, dice_ce_loss, dice_per_class

ap = argparse.ArgumentParser()
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--width", type=int, default=32)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--aug", choices=["none","generic","acoustic"], default="none")
ap.add_argument("--degrade-severity", type=float, default=0.0)
ap.add_argument("--degrade-p", type=float, default=0.0)
ap.add_argument("--tag", default="e1_cv_baseline")
a = ap.parse_args()
R = Path(__file__).resolve().parents[1]; dev="cuda"
out = R/"results"/a.tag; out.mkdir(parents=True, exist_ok=True)
(out/"config.json").write_text(json.dumps(vars(a), indent=2))
torch.manual_seed(a.seed); np.random.seed(a.seed)

idx = pd.read_csv(R/"data/cache/index_256.csv")
pats = np.array(sorted(idx.patient.unique()))
rng = np.random.default_rng(a.seed); rng.shuffle(pats)
folds = np.array_split(pats, a.folds)

degrade = None
if a.aug != "none" and a.degrade_p > 0:
    if a.aug == "acoustic":
        from src.acoustic import AcousticDegradation as _Aug
    else:
        from src.generic_aug import GenericAugmentation as _Aug
    class _D:
        """Severity is resampled per image so the model sees a range, not one setting."""
        def __init__(s): s.d = None; s.sev = a.degrade_severity; s.p = a.degrade_p
        def __call__(s, img, rng_):
            if s.d is None: s.d = _Aug(1.0, 1.0)
            if rng_.random() > s.p: return img
            s.d.severity = float(rng_.uniform(0.15, s.sev))
            return s.d(img, rng_)
    degrade = _D()

all_rows = []
for f, held in enumerate(folds):
    held = set(held)
    full_tr = CamusCached("training", augment=True, seed=a.seed, degrade=degrade)
    full_ev = CamusCached("training")
    # rebuild over ALL splits, then partition by patient
    for ds in (full_tr, full_ev):
        ds.rows = idx.reset_index(drop=True); ds.pos = np.arange(len(idx))
    tr_i = [i for i in range(len(idx)) if idx.patient[i] not in held]
    ev_i = [i for i in range(len(idx)) if idx.patient[i] in held]
    tl = DataLoader(Subset(full_tr, tr_i), batch_size=a.batch_size, shuffle=True,
                    num_workers=6, pin_memory=True, persistent_workers=True)
    el = DataLoader(Subset(full_ev, ev_i), batch_size=32, num_workers=4)

    model = UNet(1,4,width=a.width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    scaler = torch.cuda.amp.GradScaler()
    for ep in range(a.epochs):
        model.train()
        for b in tl:
            x=b["image"].to(dev,non_blocking=True); y=b["label"].to(dev,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(): loss = dice_ce_loss(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sch.step()
    model.eval()
    preds = {}
    with torch.no_grad():
        for b in el:
            p = model(b["image"].to(dev)).argmax(1)
            d = dice_per_class(p, b["label"].to(dev)).cpu().numpy()
            for i in range(len(d)):
                k=(b["patient"][i], b["view"][i], b["phase"][i])
                preds[k]=p[i].cpu().numpy()
                all_rows.append({"fold":f,"patient":k[0],"view":k[1],"phase":k[2],
                                 "quality":b["quality"][i],"dice_lv":d[i,0],
                                 "dice_myo":d[i,1],"dice_la":d[i,2],"dice_mean":float(d[i].mean())})
    np.savez_compressed(out/f"preds_fold{f}.npz",
                        **{f"{k[0]}|{k[1]}|{k[2]}": v.astype(np.uint8) for k,v in preds.items()})
    print(f"  fold {f}: {len(ev_i)} held-out frames, "
          f"Dice {np.mean([r['dice_mean'] for r in all_rows if r['fold']==f]):.4f}", flush=True)

df = pd.DataFrame(all_rows); df.to_csv(out/"per_frame.csv", index=False)
print(f"\n=== {a.tag}: out-of-fold Dice by quality (n={len(df)} frames) ===")
print(df.groupby("quality")["dice_mean"].agg(["count","mean","std"]).round(4).to_string())
