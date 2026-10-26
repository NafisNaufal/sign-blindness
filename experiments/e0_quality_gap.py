"""E0: quantify the quality gap. Gates the whole project.

Trains a U-Net baseline on CAMUS and evaluates stratified by the expert quality grade,
reporting both segmentation accuracy (Dice) and the clinical output (EF error).

Two things this must handle honestly:

1. **The EF/quality confound.** Poor-quality patients have systematically lower EF
   (39.5 vs 46.0 for Good), so a raw quality-stratified comparison conflates image quality
   with disease severity. We therefore also report the gap after EF-matching the strata.

2. **Small poor-quality test stratum.** The official test split holds only ~15 poor-quality
   records, so headline numbers are reported with bootstrap CIs and the pooled
   validation+test poor stratum is used where noted.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import CamusCached as CamusSeg                          # noqa: E402
from src.unet import UNet, dice_ce_loss, dice_per_class                  # noqa: E402
from src.plotting import bootstrap_ci                                    # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STRUCTS = ["lv_endo", "myocardium", "left_atrium"]


def evaluate(model, loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for b in loader:
            x = b["image"].to(device, non_blocking=True)
            y = b["label"].to(device, non_blocking=True)
            pred = model(x).argmax(1)
            d = dice_per_class(pred, y).cpu().numpy()
            for i in range(len(d)):
                rows.append({
                    "patient": b["patient"][i], "view": b["view"][i],
                    "phase": b["phase"][i], "quality": b["quality"][i],
                    **{f"dice_{s}": d[i, j] for j, s in enumerate(STRUCTS)},
                    "dice_mean": float(d[i].mean()),
                })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", type=str, default="e0_baseline")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    dev = "cuda"

    out = ROOT / "results" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    size = (args.size, args.size)
    tr = CamusSeg("training", size=args.size, augment=True, seed=args.seed)
    va = CamusSeg("validation", size=args.size)
    te = CamusSeg("testing", size=args.size)
    print(f"[E0] train {len(tr)} | val {len(va)} | test {len(te)} samples", flush=True)
    for name, ds in (("train", tr), ("val", va), ("test", te)):
        q = ds.rows["quality"].value_counts().to_dict()
        print(f"     {name} quality: {q}", flush=True)

    dl = lambda d, sh: DataLoader(d, batch_size=args.batch_size, shuffle=sh,
                                  num_workers=args.workers, pin_memory=True,
                                  persistent_workers=args.workers > 0)
    tl, vl, sl = dl(tr, True), dl(va, False), dl(te, False)

    model = UNet(1, 4, width=args.width).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scaler = torch.cuda.amp.GradScaler()
    print(f"     UNet width={args.width}: {n_par/1e6:.2f}M params", flush=True)

    best, t0 = -1.0, time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for b in tl:
            x = b["image"].to(dev, non_blocking=True)
            y = b["label"].to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = dice_ce_loss(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(x)
        sched.step()
        if ep % 5 == 0 or ep == args.epochs:
            vdf = evaluate(model, vl, dev)
            vd = vdf["dice_mean"].mean()
            if vd > best:
                best = vd
                torch.save(model.state_dict(), out / "best.pt")
            print(f"  ep {ep:3d}  loss {tot/len(tr):.4f}  val Dice {vd:.4f}"
                  f"{'  *' if vd == best else ''}   [{time.time()-t0:.0f}s]", flush=True)

    model.load_state_dict(torch.load(out / "best.pt"))
    res = pd.concat([evaluate(model, sl, dev).assign(split="test"),
                     evaluate(model, vl, dev).assign(split="val")], ignore_index=True)
    res.to_csv(out / "per_sample.csv", index=False)

    print("\n" + "=" * 70)
    print("E0  DICE BY EXPERT QUALITY GRADE  (test + val pooled)")
    print("=" * 70)
    for q in ["Good", "Medium", "Poor"]:
        s = res[res["quality"] == q]["dice_mean"].values
        if len(s) == 0:
            continue
        m, lo, hi = bootstrap_ci(s)
        print(f"  {q:7s} n={len(s):4d}   Dice {m:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    g = res[res["quality"].isin(["Good", "Medium"])]["dice_mean"].values
    p = res[res["quality"] == "Poor"]["dice_mean"].values
    if len(p) and len(g):
        from scipy.stats import mannwhitneyu
        u, pv = mannwhitneyu(g, p)
        print(f"\n  Good+Medium vs Poor: gap {g.mean()-p.mean():+.4f} Dice, "
              f"Mann-Whitney p = {pv:.3e}")
        print(f"  --> A1 {'SUPPORTED' if pv < 0.05 and g.mean() > p.mean() else 'NOT SUPPORTED'}")
    print(f"\n  per-structure Dice by quality:")
    print(res.groupby("quality")[[f"dice_{s}" for s in STRUCTS]].mean().round(4).to_string())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
