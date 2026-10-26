"""Decode CAMUS once into a memory-mapped cache.

Training re-read and gunzipped ~2000 NIfTI files every epoch, which made the dataloader,
not the A100, the bottleneck. Decoding once to a fixed grid costs ~0.5 GB and makes every
downstream experiment I/O-free, which matters because the study needs many training runs
(seeds, ablations, degradation severities).

Resampling happens here, but the ORIGINAL spacing and shape are stored alongside, because
volumes and EF must be computed back in millimetres.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.camus import CamusIndex, read_split                             # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VIEWS, PHASES = ("2CH", "4CH"), ("ED", "ES")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", default=str(ROOT / "data" / "cache"))
    args = ap.parse_args()

    import nibabel as nib
    idx = CamusIndex()
    size = (args.size, args.size)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    splits = {n: set(read_split(n)) for n in ("training", "validation", "testing")}
    items, imgs, labs = [], [], []

    pats = idx.patients()
    for k, p in enumerate(pats):
        split = next((s for s, v in splits.items() if p in v), "unassigned")
        for view in VIEWS:
            cfg = idx.info(p, view)
            if cfg is None:
                continue
            for ph in PHASES:
                ip, gp = idx.image_path(p, view, ph), idx.image_path(p, view, ph, gt=True)
                if not ip.exists() or not gp.exists():
                    continue
                nii = nib.load(str(ip))
                a = np.asarray(nii.dataobj, dtype=np.float32)
                g = np.asarray(nib.load(str(gp)).dataobj, dtype=np.float32)
                sp = tuple(float(z) for z in nii.header.get_zooms()[:2])
                shape = a.shape

                a = a / max(a.max(), 1e-6)
                at = F.interpolate(torch.from_numpy(a)[None, None], size=size,
                                   mode="bilinear", align_corners=False)[0, 0]
                gt = F.interpolate(torch.from_numpy(g)[None, None], size=size,
                                   mode="nearest")[0, 0]
                imgs.append(at.numpy().astype(np.float32))
                labs.append(gt.numpy().astype(np.uint8))
                items.append({
                    "patient": p, "view": view, "phase": ph, "split": split,
                    "quality": cfg.get("ImageQuality"), "ef_ref": cfg.get("EF"),
                    "sex": cfg.get("Sex"), "age": cfg.get("Age"),
                    "ed": cfg.get("ED"), "es": cfg.get("ES"),
                    "sp_y": sp[0], "sp_x": sp[1], "h": shape[0], "w": shape[1],
                })
        if (k + 1) % 100 == 0:
            print(f"  ...{k+1}/{len(pats)}", flush=True)

    import pandas as pd
    imgs = np.stack(imgs)
    labs = np.stack(labs)
    np.save(outdir / f"images_{args.size}.npy", imgs)
    np.save(outdir / f"labels_{args.size}.npy", labs)
    pd.DataFrame(items).to_csv(outdir / f"index_{args.size}.csv", index=False)

    print(f"\ncached {len(items)} samples at {args.size}^2")
    print(f"  images {imgs.nbytes/1e6:.0f} MB   labels {labs.nbytes/1e6:.0f} MB")
    print(f"  label classes {np.unique(labs).tolist()}")
    print(f"  -> {outdir}")


if __name__ == "__main__":
    main()
