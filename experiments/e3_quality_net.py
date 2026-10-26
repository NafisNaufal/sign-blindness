"""E3: train the CAMUS quality grader that will label EchoNet.

Success bar is human-level agreement, not perfection: the published open-source estimator
reaches Spearman rho = 0.69 against three cardiologists, where inter-observer agreement is
rho = 0.63 (arXiv 2408.00591). A grader near 0.6 is already as consistent as the experts
are with each other, and chasing higher would mostly be fitting one annotator's idiosyncrasy.

Splits are BY PATIENT across 5 folds, so the same patient's 2CH/4CH and ED/ES frames never
straddle train and test. Frame-level splitting would leak heavily here, because a patient's
four frames share an acoustic window and therefore share their quality grade almost by
definition.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.quality import (QualityNet, ordinal_loss, logits_to_grade,       # noqa: E402
                         grade_from_score, GRADES, GRADE_TO_IDX)

ROOT = Path(__file__).resolve().parents[1]


class QualDS(Dataset):
    def __init__(self, images, rows, positions, augment=False, seed=0):
        self.images, self.rows, self.pos = images, rows, positions
        self.augment, self.seed = augment, seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        j = int(self.pos[i])
        r = self.rows.iloc[i]
        img = torch.from_numpy(np.array(self.images[j], dtype=np.float32))
        if self.augment:
            rng = np.random.default_rng(self.seed * 7919 + i)
            # Geometry only. Photometric augmentation would corrupt the very signal the
            # grader is meant to read, since quality IS largely a photometric property.
            if rng.random() < 0.5:
                k = int(rng.integers(-8, 9))
                img = torch.roll(img, shifts=(k, int(rng.integers(-8, 9))), dims=(0, 1))
        return img[None], GRADE_TO_IDX[r["quality"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="e3_quality")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda"
    out = ROOT / "results" / a.tag; out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2))

    images = np.load(ROOT / "data/cache/images_256.npy", mmap_mode="r")
    idx = pd.read_csv(ROOT / "data/cache/index_256.csv")
    idx = idx[idx["quality"].isin(GRADES)].reset_index(drop=True)
    print(f"[E3] {len(idx)} frames  grades: {idx['quality'].value_counts().to_dict()}", flush=True)

    pats = np.array(sorted(idx.patient.unique()))
    rng = np.random.default_rng(a.seed); rng.shuffle(pats)
    folds = np.array_split(pats, a.folds)

    rows = []
    for f, held in enumerate(folds):
        held = set(held)
        tr_m = ~idx.patient.isin(held); ev_m = idx.patient.isin(held)
        tr = QualDS(images, idx[tr_m].reset_index(drop=True),
                    np.asarray(idx.index[tr_m]), augment=True, seed=a.seed)
        ev = QualDS(images, idx[ev_m].reset_index(drop=True), np.asarray(idx.index[ev_m]))
        tl = DataLoader(tr, batch_size=a.batch_size, shuffle=True, num_workers=6,
                        pin_memory=True, persistent_workers=True)
        el = DataLoader(ev, batch_size=64, num_workers=4)

        model = QualityNet(width=a.width).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
        scaler = torch.cuda.amp.GradScaler()
        for ep in range(a.epochs):
            model.train()
            for x, y in tl:
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast():
                    loss = ordinal_loss(model(x).float(), y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sch.step()

        model.eval()
        ev_rows = idx[ev_m].reset_index(drop=True)
        k = 0
        with torch.no_grad():
            for x, y in el:
                s = logits_to_grade(model(x.to(dev)).float()).cpu().numpy()
                for i in range(len(s)):
                    r = ev_rows.iloc[k]
                    rows.append({"fold": f, "patient": r["patient"], "view": r["view"],
                                 "phase": r["phase"], "true": r["quality"],
                                 "true_idx": GRADE_TO_IDX[r["quality"]],
                                 "score": float(s[i]),
                                 "pred_idx": int(grade_from_score(torch.tensor(s[i:i+1]))[0])})
                    k += 1
        torch.save(model.state_dict(), out / f"qnet_fold{f}.pt")
        print(f"  fold {f} done ({int(ev_m.sum())} held-out frames)", flush=True)

    df = pd.DataFrame(rows); df.to_csv(out / "oof_quality.csv", index=False)
    from scipy.stats import spearmanr
    rho, p = spearmanr(df["score"], df["true_idx"])
    acc = (df["pred_idx"] == df["true_idx"]).mean()
    adj = (abs(df["pred_idx"] - df["true_idx"]) <= 1).mean()
    print("\n" + "=" * 68)
    print("E3  QUALITY GRADER, out-of-fold on CAMUS")
    print("=" * 68)
    print(f"  Spearman rho vs expert grade : {rho:.3f}  (p={p:.2e})")
    print(f"  exact 3-class accuracy       : {acc:.3f}")
    print(f"  within-one-grade accuracy    : {adj:.3f}")
    print(f"  reference: published estimator rho=0.69, inter-observer rho=0.63")
    print(f"  --> {'AT OR NEAR HUMAN AGREEMENT' if rho >= 0.55 else 'BELOW BAR, needs work'}")
    print("\n  confusion (rows=true, cols=pred):")
    print(pd.crosstab(df["true"], df["pred_idx"]).to_string())
    print(f"\n  mean predicted score by true grade:")
    print(df.groupby("true")["score"].agg(["count", "mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
