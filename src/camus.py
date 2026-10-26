"""CAMUS echocardiography dataset.

500 patients from University Hospital of St Etienne. Per patient: 2CH and 4CH apical
views, end-diastole (ED) and end-systole (ES) frames, expert contours for LV endocardium
(1), myocardium (2) and left atrium (3).

The property this project is built on: each patient carries an **expert cardiologist
quality grade** (Good / Medium / Poor). 84 of 450 training patients are graded Poor.
Almost no other public medical imaging dataset ships clinician-assigned quality labels,
and it is what makes a quality-stratified study possible at all.

Per-view `*_info_*.cfg` also gives ED/ES frame indices, LV volumes, and the expert EF,
so ejection fraction (the actual clinical output) can be scored, not just Dice.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "data" / "camus"
NIFTI = ROOT / "database_nifti"
SPLIT = ROOT / "database_split"

LABELS = {"lv_endo": 1, "myocardium": 2, "left_atrium": 3}
QUALITY_ORDER = ["Good", "Medium", "Poor"]


def read_cfg(path):
    """Parse a CAMUS `*_info_*.cfg`: simple `key: value` lines."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        try:
            out[k] = float(v) if re.fullmatch(r"-?\d+\.?\d*(e-?\d+)?", v, re.I) else v
        except ValueError:
            out[k] = v
    return out


def read_split(name):
    """name in {training, validation, testing} -> list of patient ids."""
    f = SPLIT / f"subgroup_{name}.txt"
    return [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]


class CamusIndex:
    """Metadata index over the whole dataset, built without loading pixel data.

    Keeping this separate from image loading means the quality/EF distribution can be
    audited (and the load-bearing assumption checked) before committing to any training.
    """

    def __init__(self, nifti_root=None, split_root=None):
        self.nifti = Path(nifti_root or NIFTI)
        self.split = Path(split_root or SPLIT)

    def patients(self):
        return sorted(p.name for p in self.nifti.iterdir()
                      if p.is_dir() and p.name.startswith("patient"))

    def info(self, patient, view):
        """view in {2CH, 4CH}. Returns the parsed cfg dict, or None if absent."""
        for cand in (self.nifti / patient / f"Info_{view}.cfg",
                     self.nifti / patient / f"{patient}_{view}_info.cfg",
                     self.nifti / patient / f"{patient}_info_{view}.cfg"):
            if cand.exists():
                return read_cfg(cand)
        return None

    def records(self, patients=None):
        """One record per (patient, view) with quality grade, EF and volumes."""
        out = []
        for p in (patients or self.patients()):
            for view in ("2CH", "4CH"):
                cfg = self.info(p, view)
                if cfg is None:
                    continue
                out.append({
                    "patient": p, "view": view,
                    "quality": cfg.get("ImageQuality"),
                    "ef": cfg.get("EF"),
                    "edv": cfg.get("LVedv"), "esv": cfg.get("LVesv"),
                    "ed": cfg.get("ED"), "es": cfg.get("ES"),
                    "nframes": cfg.get("NbFrame"),
                    "sex": cfg.get("Sex"), "age": cfg.get("Age"),
                })
        return out

    def image_path(self, patient, view, phase, gt=False):
        """phase in {ED, ES}. gt=True returns the expert segmentation."""
        stem = f"{patient}_{view}_{phase}"
        return self.nifti / patient / f"{stem}{'_gt' if gt else ''}.nii.gz"


def ejection_fraction(edv, esv):
    """EF = (EDV - ESV) / EDV * 100. The clinical output; errors in ED and ES compound."""
    edv = np.asarray(edv, dtype=float)
    esv = np.asarray(esv, dtype=float)
    return np.where(edv > 0, (edv - esv) / np.maximum(edv, 1e-9) * 100.0, np.nan)


def lv_long_axis(lv_mask, la_mask, spacing):
    """Locate the LV long axis anatomically: mitral annulus midpoint to apex.

    The base is the LV-LA interface (in CAMUS the left atrium is label 3), and the apex is
    the LV point furthest from it. Assuming instead that the long axis is simply vertical
    costs several EF points, because apical views are routinely tilted.

    Returns (base_xy, apex_xy, unit_axis, length_mm) in millimetre coordinates.
    """
    from scipy.ndimage import binary_dilation

    sy, sx = float(spacing[0]), float(spacing[1])
    pts = np.argwhere(lv_mask)
    if len(pts) == 0:
        return None
    mm = pts * np.array([sy, sx])

    base = None
    if la_mask is not None and la_mask.any():
        # LV voxels touching the dilated LA delineate the mitral annulus
        iface = lv_mask & binary_dilation(la_mask, iterations=2)
        if iface.sum() >= 3:
            base = (np.argwhere(iface) * np.array([sy, sx])).mean(0)
    if base is None:
        # fall back to the PCA axis, taking the end nearer the image bottom as the base
        c = mm.mean(0)
        v = np.linalg.svd(mm - c, full_matrices=False)[2][0]
        proj = (mm - c) @ v
        base = c + v * (proj.max() if (v[0] > 0) else proj.min())

    d = np.linalg.norm(mm - base, axis=1)
    apex = mm[int(np.argmax(d))]
    axis = apex - base
    L = float(np.linalg.norm(axis))
    if L < 1e-6:
        return None
    return base, apex, axis / L, L


def disc_diameters(mask, la_mask, spacing, n_disks=20):
    """Diameters perpendicular to the anatomical long axis, for Simpson's method."""
    geo = lv_long_axis(mask, la_mask, spacing)
    if geo is None:
        return None, 0.0
    base, _, u, L = geo
    n = np.array([-u[1], u[0]])                     # in-plane normal to the long axis

    mm = np.argwhere(mask) * np.array([float(spacing[0]), float(spacing[1])])
    t = (mm - base) @ u                             # position along the axis
    s = (mm - base) @ n                             # signed offset across it

    edges = np.linspace(0.0, L, n_disks + 1)
    d = np.zeros(n_disks)
    for i in range(n_disks):
        band = (t >= edges[i]) & (t < edges[i + 1])
        if band.any():
            d[i] = s[band].max() - s[band].min()
    return d, L / n_disks


def simpson_biplane_volume(mask_2ch, mask_4ch, spacing_2ch, spacing_4ch,
                           la_2ch=None, la_4ch=None, n_disks=20):
    """Simpson's biplane method of discs, the clinical standard for LV volume.

    Each of `n_disks` slices is modelled as an ellipse whose two diameters come from the
    2CH and 4CH views: V = pi/4 * sum(d_2CH * d_4CH * h). Using the clinical formula
    rather than a voxel count is what makes EF comparable to a cardiologist's report.
    """
    d2, h2 = disc_diameters(mask_2ch, la_2ch, spacing_2ch, n_disks)
    d4, h4 = disc_diameters(mask_4ch, la_4ch, spacing_4ch, n_disks)
    if d2 is None or d4 is None:
        return np.nan
    h = 0.5 * (h2 + h4)
    return float(np.pi / 4.0 * np.sum(d2 * d4) * h / 1000.0)   # mm^3 -> mL
