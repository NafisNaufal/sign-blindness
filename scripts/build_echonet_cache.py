"""Decode EchoNet-Dynamic into an ED/ES frame + mask cache.

Only the two expert-traced frames per video matter for this study, so 10,030 videos
collapse to ~20,060 frames. Decoding once avoids re-reading AVIs every epoch, the same
bottleneck that made CAMUS training 25x slower than necessary.

Tracing -> mask
---------------
EchoNet ships the clinical 21-line format: the first row of each traced frame is the LV
long axis, the remaining ~20 rows are chords perpendicular to it. The chord endpoints lie
on the endocardial border, so the LV mask is the polygon through one side's endpoints and
back along the other side. Reconstructing the mask this way (rather than, say, filling a
convex hull) preserves the concave apex and mitral plane.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ECHO = ROOT / "data" / "echonet" / "EchoNet-Dynamic"
OUT = ROOT / "data" / "cache_echonet"


def mask_from_tracing(g, shape, size):
    """Filled LV mask from one frame's chord list."""
    from skimage.draw import polygon
    from skimage.transform import resize as skresize

    ch = g.iloc[1:]                       # row 0 is the long axis, not a chord
    if len(ch) < 3:
        return None
    # walk down one side of the chords and back up the other to close the contour
    xs = np.concatenate([ch.X1.values, ch.X2.values[::-1]])
    ys = np.concatenate([ch.Y1.values, ch.Y2.values[::-1]])
    m = np.zeros(shape, dtype=np.uint8)
    rr, cc = polygon(ys, xs, shape)
    m[rr, cc] = 1
    if shape != (size, size):
        m = (skresize(m.astype(np.float32), (size, size), order=0,
                      preserve_range=True, anti_aliasing=False) > 0.5).astype(np.uint8)
    return m


def process(args):
    key, frames, size = args
    import cv2
    from skimage.transform import resize as skresize

    cap = cv2.VideoCapture(str(ECHO / "Videos" / f"{key}.avi"))
    if not cap.isOpened():
        return None
    want = {int(f): None for f in frames["Frame"].unique()}
    i, got = 0, 0
    while got < len(want):
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            want[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            got += 1
        i += 1
    cap.release()
    if any(v is None for v in want.values()):
        return None

    out = []
    for f, g in frames.groupby("Frame"):
        img = want[int(f)]
        m = mask_from_tracing(g, img.shape, size)
        if m is None:
            continue
        im = img.astype(np.float32) / 255.0
        if img.shape != (size, size):
            im = skresize(im, (size, size), order=1, preserve_range=True,
                          anti_aliasing=True).astype(np.float32)
        out.append((key, int(f), im, m, float(np.pi / 4 * _disc_vol(g))))
    return out


def _disc_vol(g):
    """Volume in pixel^3 by method of discs; scale-free, so EF from it is exact."""
    la = g.iloc[0]
    L = float(np.hypot(la.X2 - la.X1, la.Y2 - la.Y1))
    ch = g.iloc[1:]
    d = np.hypot(ch.X2 - ch.X1, ch.Y2 - ch.Y1).values
    return float((d ** 2).sum() * (L / max(len(d), 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    vt = pd.read_csv(ROOT / "data/echonet_meta/EchoNet-Dynamic/VolumeTracings.csv")
    fl = pd.read_csv(ROOT / "data/echonet_meta/EchoNet-Dynamic/FileList.csv")
    vt["key"] = vt.FileName.str.replace(".avi", "", regex=False)
    keys = sorted(vt.key.unique())
    if a.limit:
        keys = keys[:a.limit]
    meta = fl.set_index(fl.FileName.str.replace(".avi", "", regex=False))

    tasks = [(k, g, a.size) for k, g in vt[vt.key.isin(set(keys))].groupby("key")]
    print(f"[cache] {len(tasks)} videos, {a.workers} workers, size {a.size}", flush=True)

    imgs, masks, rows = [], [], []
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for res in ex.map(process, tasks, chunksize=8):
            done += 1
            if done % 1000 == 0:
                print(f"  ...{done}/{len(tasks)}", flush=True)
            if not res:
                continue
            for key, frame, im, m, vol in res:
                r = meta.loc[key]
                imgs.append(im); masks.append(m)
                rows.append({"key": key, "frame": frame, "vol_px": vol,
                             "ef_ref": float(r.EF), "split": r.Split,
                             "h": int(r.FrameHeight), "w": int(r.FrameWidth),
                             "fps": float(r.FPS), "nframes": int(r.NumberOfFrames)})

    OUT.mkdir(parents=True, exist_ok=True)
    imgs = np.stack(imgs); masks = np.stack(masks)
    df = pd.DataFrame(rows)
    # larger of the two traced volumes is ED, smaller is ES
    df["phase"] = df.groupby("key")["vol_px"].transform(
        lambda s: np.where(s == s.max(), "ED", "ES"))
    np.save(OUT / f"images_{a.size}.npy", imgs)
    np.save(OUT / f"masks_{a.size}.npy", masks)
    df.to_csv(OUT / f"index_{a.size}.csv", index=False)
    print(f"\ncached {len(df)} frames from {df.key.nunique()} videos")
    print(f"  images {imgs.nbytes/1e6:.0f} MB  masks {masks.nbytes/1e6:.0f} MB")
    print(f"  phases: {df.phase.value_counts().to_dict()}")
    print(f"  splits: {df.split.value_counts().to_dict()}")
    print(f"  mean LV area frac: {masks.mean():.4f}")
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
