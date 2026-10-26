"""Physics-grounded acoustic degradation for echocardiography.

Generic augmentation (brightness, blur, elastic, cutout) perturbs images in ways that have
no acoustic meaning. Echocardiograms degrade through specific, well-understood mechanisms,
and crucially those mechanisms act in the **polar geometry of the beam**, not in image
coordinates: the sector emanates from the transducer, so a rib shadow is an angular wedge
that widens with depth, never an axis-aligned rectangle.

Mechanisms simulated
--------------------
============  ==========================================  ==============================
mechanism     physical cause                              model
============  ==========================================  ==============================
shadowing     ribs / lung / calcification block the beam  angular wedges from the apex
dropout       poor probe contact, obesity                 smooth correlated attenuation
attenuation   absorption, increases with depth            exp(-alpha * r)
speckle       coherent interference of scatterers         multiplicative Rayleigh/Gamma
clutter       multipath, reverberation                    near-field additive haze
blur          reduced lateral resolution with depth       depth-dependent smoothing
============  ==========================================  ==============================

All operate on a normalised image in [0, 1] and return the same. The transducer apex is
detected per image via `detect_apex` (CAMUS stores the sector fanning left-to-right, so a
hard-coded top-centre apex is wrong); `apex` can override. Everything is torch and runs on
the input's device, so degradation happens in the training loop with no CPU round-trip.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def detect_apex(img, thresh=1e-4):
    """Locate the transducer apex from the sector mask.

    Do NOT assume a top-centre apex. CAMUS frames are stored with the sector fanning
    left-to-right, so a hard-coded orientation applies shadow wedges and depth attenuation
    along the wrong axis while still looking superficially plausible. The apex is the
    narrow end of the fan, found by comparing how the mask extent grows from each side.
    """
    m = img > thresh
    if not m.any():
        return (0.0, (img.shape[1] - 1) / 2.0)
    rows = m.sum(1).float()
    cols = m.sum(0).float()
    ys, xs = torch.nonzero(m, as_tuple=True)
    y0, y1, x0, x1 = ys.min().item(), ys.max().item(), xs.min().item(), xs.max().item()

    def taper(profile, lo, hi):
        """How much narrower the near end is than the far end, along one direction."""
        span = max(hi - lo, 1)
        k = max(int(0.12 * span), 1)
        near = profile[lo:lo + k].mean()
        far = profile[hi - k + 1:hi + 1].mean()
        return (far - near).item()

    cand = {
        "top":    taper(rows, y0, y1),
        "bottom": -taper(rows, y0, y1),
        "left":   taper(cols, x0, x1),
        "right":  -taper(cols, x0, x1),
    }
    side = max(cand, key=cand.get)

    if side in ("top", "bottom"):
        edge = y0 if side == "top" else y1
        band = m[edge:edge + 3] if side == "top" else m[max(edge - 2, 0):edge + 1]
        idx = torch.nonzero(band, as_tuple=True)[1]
        return (float(edge), float(idx.float().mean()) if len(idx) else (img.shape[1] - 1) / 2)
    edge = x0 if side == "left" else x1
    band = m[:, edge:edge + 3] if side == "left" else m[:, max(edge - 2, 0):edge + 1]
    idx = torch.nonzero(band, as_tuple=True)[0]
    return (float(idx.float().mean()) if len(idx) else (img.shape[0] - 1) / 2, float(edge))


def polar_coords(h, w, device, apex=None, dtype=torch.float32):
    """Radius (normalised) and angle (radians about the apex) per pixel."""
    ay, ax = apex if apex is not None else (0.0, (w - 1) / 2.0)
    ys = torch.arange(h, device=device, dtype=dtype).view(-1, 1)
    xs = torch.arange(w, device=device, dtype=dtype).view(1, -1)
    dy, dx = ys - ay, xs - ax
    r = torch.sqrt(dy * dy + dx * dx) / max(max(h, w) - 1, 1)
    theta = torch.atan2(dy, dx)          # angle about the apex, orientation-agnostic
    return r, theta


def _smooth_noise(shape, scale, device, gen, dtype=torch.float32):
    """Spatially correlated noise: white noise at low resolution, smoothly upsampled."""
    h, w = shape
    lh, lw = max(int(h / scale), 2), max(int(w / scale), 2)
    n = torch.randn(1, 1, lh, lw, device=device, dtype=dtype, generator=gen)
    return F.interpolate(n, size=(h, w), mode="bicubic", align_corners=False)[0, 0]


class AcousticDegradation:
    """Callable(img[H,W] in [0,1], rng) -> degraded img.

    `severity` in [0,1] scales every mechanism together, so a single knob sweeps from
    pristine to unreadable and the severity ablation is one parameter.
    """

    def __init__(self, severity=1.0, p=1.0, mechanisms=None, apex=None, seed=None):
        self.severity = float(severity)
        self.p = float(p)
        self.mech = set(mechanisms or
                        ["shadow", "dropout", "attenuation", "speckle",
                         "clutter", "blur", "contrast"])
        self.apex = apex
        self.seed = seed
        self._cache = {}

    def _grid(self, img):
        h, w = img.shape
        apex = self.apex if self.apex is not None else detect_apex(img)
        key = (h, w, str(img.device), round(apex[0], 1), round(apex[1], 1))
        if key not in self._cache:
            r, theta = polar_coords(h, w, img.device, apex)
            # centre the angle on the beam axis so wedge offsets mean the same thing
            m = img > 1e-4
            th0 = theta[m].mean() if m.any() else torch.tensor(0.0, device=img.device)
            self._cache[key] = (r, torch.atan2(torch.sin(theta - th0),
                                               torch.cos(theta - th0)))
        return self._cache[key]

    def __call__(self, img, rng=None):
        if img.dim() == 3:
            img = img[0]
        h, w = img.shape
        dev = img.device
        s = self.severity

        # torch generator seeded from the numpy rng keeps runs reproducible
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(rng.integers(0, 2**31 - 1)) if rng is not None
                        else torch.randint(0, 2**31 - 1, (1,)).item())

        def u(a, b):
            return float(torch.rand(1, generator=gen, device=dev).item()) * (b - a) + a

        if self.p < 1.0 and u(0, 1) > self.p:
            return img

        r, theta = self._grid(img)
        out = img.clone()

        # --- acoustic shadowing: angular wedges, the signature echo artefact ----------
        if "shadow" in self.mech:
            for _ in range(int(round(u(0, 2.99 * s)))):
                c = u(-0.85, 0.85)                    # wedge centre angle
                width = u(0.05, 0.28) * (0.5 + s)     # angular half-width
                onset = u(0.15, 0.55)                 # depth at which it begins
                depth = u(0.45, 0.95) * s             # how much signal is lost
                ang = torch.exp(-0.5 * ((theta - c) / max(width, 1e-3)) ** 2)
                rad = torch.sigmoid((r - onset) * 14.0)
                out = out * (1.0 - depth * ang * rad)

        # --- dropout: smooth correlated loss from poor probe contact ------------------
        if "dropout" in self.mech:
            n = _smooth_noise((h, w), u(6, 16), dev, gen)
            field = torch.sigmoid(n * 2.2)
            out = out * (1.0 - u(0.2, 0.6) * s * (1.0 - field))

        # --- depth-dependent attenuation ---------------------------------------------
        if "attenuation" in self.mech:
            out = out * torch.exp(-u(0.4, 1.5) * s * r)

        # --- lateral resolution loss with depth --------------------------------------
        if "blur" in self.mech and s > 0.05:
            k = int(2 * round(u(1, 2 + 2 * s)) + 1)
            sig = u(0.6, 1.8) * s
            ax = torch.arange(k, device=dev, dtype=out.dtype) - k // 2
            g = torch.exp(-0.5 * (ax / max(sig, 1e-3)) ** 2)
            g = (g / g.sum()).view(1, 1, 1, -1)
            b = out[None, None]
            b = F.conv2d(F.pad(b, (k // 2,) * 2 + (0, 0), mode="reflect"), g)
            b = F.conv2d(F.pad(b, (0, 0) + (k // 2,) * 2, mode="reflect"),
                         g.transpose(-1, -2))[0, 0]
            w_depth = torch.clamp(r / 0.9, 0, 1)      # sharper near field, blurrier deep
            out = out * (1 - w_depth) + b * w_depth

        # --- speckle: multiplicative, the defining texture of B-mode ultrasound -------
        if "speckle" in self.mech:
            k = max(1e-3, 1.0 / max(u(0.05, 0.18) * s, 1e-3) ** 2)
            gamma = torch.distributions.Gamma(
                torch.tensor(k, device=dev), torch.tensor(k, device=dev))
            noise = gamma.sample((h, w))
            out = out * noise.clamp(0.55, 1.8)

        # --- near-field clutter / reverberation --------------------------------------
        if "clutter" in self.mech:
            haze = _smooth_noise((h, w), u(3, 9), dev, gen).abs()
            out = out + u(0.06, 0.18) * s * haze * torch.exp(-1.6 * r) * (0.35 + 0.65 * out)

        # --- contrast compression: the dominant visual signature of a poor window ----
        if "contrast" in self.mech:
            g = u(0.55, 0.9) if s > 0 else 1.0
            lo = u(0.0, 0.10) * s
            out = lo + (1.0 - lo) * out.clamp_min(0) ** (1.0 - (1.0 - g) * s)

        # keep the sector mask: pixels outside the beam stay black
        out = torch.where(img > 1e-4, out, img)
        return out.clamp(0, 1)


def make_degradation(severity=1.0, p=0.5, mechanisms=None):
    return AcousticDegradation(severity=severity, p=p, mechanisms=mechanisms)
