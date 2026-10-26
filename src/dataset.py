"""PyTorch dataset for CAMUS 2D segmentation.

CAMUS images vary in size (549x389, 433x357, 869x616 ...) at ~0.3 mm isotropic pixel
spacing, so everything is resampled to a common grid. The original spacing and shape are
carried through each sample, because volumes and therefore EF must be computed back in
millimetres, not in resampled pixels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .camus import CamusIndex, read_split

VIEWS = ("2CH", "4CH")
PHASES = ("ED", "ES")


def _resize(img, size, mode):
    t = torch.as_tensor(img, dtype=torch.float32)[None, None]
    kw = {"align_corners": False} if mode in ("bilinear", "bicubic") else {}
    return F.interpolate(t, size=size, mode=mode, **kw)[0, 0]


class CamusSeg(Dataset):
    """One item per (patient, view, phase).

    Parameters
    ----------
    split      : 'training' | 'validation' | 'testing'
    size       : output grid, (H, W)
    augment    : geometric + intensity augmentation (train only)
    degrade    : optional callable(img_tensor, rng) -> img_tensor, the acoustic
                 degradation simulator. Kept as an injected dependency so the baseline
                 and the proposed method differ by exactly this one argument.
    quality    : optional filter, e.g. ('Good','Medium')
    """

    def __init__(self, split="training", size=(256, 256), augment=False,
                 degrade=None, quality=None, index=None, seed=0):
        self.idx = index or CamusIndex()
        self.size = tuple(size)
        self.augment = augment
        self.degrade = degrade
        self.seed = seed

        patients = read_split(split)
        self.items = []
        for p in patients:
            for v in VIEWS:
                cfg = self.idx.info(p, v)
                if cfg is None:
                    continue
                q = cfg.get("ImageQuality")
                if quality is not None and q not in quality:
                    continue
                for ph in PHASES:
                    if not self.idx.image_path(p, v, ph).exists():
                        continue
                    self.items.append({"patient": p, "view": v, "phase": ph,
                                       "quality": q, "ef_ref": cfg.get("EF")})

    def __len__(self):
        return len(self.items)

    def _augment(self, img, lab, rng):
        # Echocardiograms have a fixed anatomical orientation (apex up, sector from the
        # probe), so horizontal flips are anatomically invalid and are NOT used.
        ang = float(rng.uniform(-10, 10)) * np.pi / 180
        sc = float(rng.uniform(0.9, 1.1))
        tx, ty = (float(rng.uniform(-0.05, 0.05)) for _ in range(2))
        cos, sin = np.cos(ang) / sc, np.sin(ang) / sc
        theta = torch.tensor([[cos, -sin, tx], [sin, cos, ty]], dtype=torch.float32)[None]
        grid = F.affine_grid(theta, (1, 1, *self.size), align_corners=False)
        img = F.grid_sample(img[None, None], grid, mode="bilinear",
                            padding_mode="zeros", align_corners=False)[0, 0]
        lab = F.grid_sample(lab[None, None].float(), grid, mode="nearest",
                            padding_mode="zeros", align_corners=False)[0, 0]
        img = (img * float(rng.uniform(0.9, 1.1))).clamp(0, 1)
        return img, lab

    def __getitem__(self, i):
        import nibabel as nib
        it = self.items[i]
        ip = self.idx.image_path(it["patient"], it["view"], it["phase"])
        gp = self.idx.image_path(it["patient"], it["view"], it["phase"], gt=True)
        nii = nib.load(str(ip))
        img = np.asarray(nii.dataobj, dtype=np.float32)
        lab = np.asarray(nib.load(str(gp)).dataobj, dtype=np.float32)
        spacing = tuple(float(z) for z in nii.header.get_zooms()[:2])
        orig_shape = img.shape

        img = img / max(img.max(), 1e-6)
        img = _resize(img, self.size, "bilinear").clamp(0, 1)
        lab = _resize(lab, self.size, "nearest")

        rng = np.random.default_rng(self.seed * 100003 + i)
        if self.augment:
            img, lab = self._augment(img, lab, rng)
        if self.degrade is not None:
            img = self.degrade(img, rng)

        return {
            "image": img[None], "label": lab.long(),
            "patient": it["patient"], "view": it["view"], "phase": it["phase"],
            "quality": it["quality"] or "Unknown",
            "ef_ref": float(it["ef_ref"]) if it["ef_ref"] is not None else float("nan"),
            # needed to convert predicted masks back to millimetres for volumes
            "spacing": torch.tensor(spacing), "orig_shape": torch.tensor(orig_shape),
        }


class CamusCached(Dataset):
    """Same contract as CamusSeg, reading the preprocessed cache.

    Re-gunzipping ~2000 NIfTIs every epoch made the dataloader, not the A100, the
    bottleneck. The cache is small enough to sit in RAM, so training becomes GPU-bound and
    the many runs this study needs (seeds, ablations, degradation sweeps) become practical.
    """

    _store = {}   # class-level so workers and repeated runs share one copy

    def __init__(self, split="training", size=256, augment=False, degrade=None,
                 quality=None, seed=0, cache_dir=None):
        import pandas as pd
        d = Path(cache_dir or Path(__file__).resolve().parents[1] / "data" / "cache")
        key = (str(d), size)
        if key not in CamusCached._store:
            CamusCached._store[key] = (
                np.load(d / f"images_{size}.npy", mmap_mode="r"),
                np.load(d / f"labels_{size}.npy", mmap_mode="r"),
                pd.read_csv(d / f"index_{size}.csv"),
            )
        self.images, self.labels, idx = CamusCached._store[key]

        m = idx["split"] == split
        if quality is not None:
            m &= idx["quality"].isin(quality)
        self.rows = idx[m].reset_index(drop=True)
        self.pos = np.asarray(idx.index[m])
        self.size = (size, size)
        self.augment = augment
        self.degrade = degrade
        self.seed = seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        j = int(self.pos[i])
        r = self.rows.iloc[i]
        img = torch.from_numpy(np.array(self.images[j], dtype=np.float32))
        lab = torch.from_numpy(np.array(self.labels[j], dtype=np.int64))

        rng = np.random.default_rng(self.seed * 100003 + i)
        if self.augment:
            img, lab = CamusSeg._augment(self, img, lab.float(), rng)
            lab = lab.long()
        if self.degrade is not None:
            img = self.degrade(img, rng)

        return {
            "image": img[None], "label": lab,
            "patient": r["patient"], "view": r["view"], "phase": r["phase"],
            "quality": r["quality"] if isinstance(r["quality"], str) else "Unknown",
            "ef_ref": float(r["ef_ref"]) if np.isfinite(r["ef_ref"]) else float("nan"),
            "spacing": torch.tensor([float(r["sp_y"]), float(r["sp_x"])]),
            "orig_shape": torch.tensor([int(r["h"]), int(r["w"])]),
        }
