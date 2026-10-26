"""Strong generic photometric augmentation: the control arm for E2.

This is deliberately a *strong* baseline, not a strawman. The scientific question is not
"does augmentation help" (obviously it does) but "does augmentation grounded in ultrasound
physics and applied in beam geometry beat equally aggressive generic augmentation". If the
generic arm matches the acoustic arm, the polar-geometry premise is wrong and the paper
must say so.

Mechanisms are chosen to be the standard medical-imaging augmentation set (as in nnU-Net
and MONAI defaults) and the severity knob is scaled so the mean absolute perturbation is
comparable to the acoustic simulator at matched severity.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class GenericAugmentation:
    """Callable(img[H,W] in [0,1], rng) -> perturbed img. Same interface as AcousticDegradation."""

    def __init__(self, severity=1.0, p=1.0, mechanisms=None):
        self.severity = float(severity)
        self.p = float(p)
        self.mech = set(mechanisms or
                        ["brightness", "contrast", "gamma", "noise", "blur", "cutout"])

    def __call__(self, img, rng=None):
        if img.dim() == 3:
            img = img[0]
        h, w = img.shape
        dev = img.device
        s = self.severity

        gen = torch.Generator(device=dev)
        gen.manual_seed(int(rng.integers(0, 2**31 - 1)) if rng is not None
                        else torch.randint(0, 2**31 - 1, (1,)).item())

        def u(a, b):
            return float(torch.rand(1, generator=gen, device=dev).item()) * (b - a) + a

        if self.p < 1.0 and u(0, 1) > self.p:
            return img
        out = img.clone()

        if "brightness" in self.mech:
            out = out + u(-0.25, 0.25) * s
        if "contrast" in self.mech:
            m = out.mean()
            out = (out - m) * (1.0 + u(-0.45, 0.45) * s) + m
        if "gamma" in self.mech:
            out = out.clamp(0, 1) ** (2.0 ** (u(-1.0, 1.0) * s))
        if "noise" in self.mech:
            out = out + torch.randn(h, w, device=dev, generator=gen) * u(0.02, 0.16) * s
        if "blur" in self.mech and s > 0.05:
            sig = u(0.4, 2.2) * s
            k = int(2 * round(2 * sig) + 1)
            if k >= 3:
                ax = torch.arange(k, device=dev, dtype=out.dtype) - k // 2
                g = torch.exp(-0.5 * (ax / max(sig, 1e-3)) ** 2)
                g = (g / g.sum()).view(1, 1, 1, -1)
                b = out[None, None]
                b = F.conv2d(F.pad(b, (k // 2,) * 2 + (0, 0), mode="reflect"), g)
                out = F.conv2d(F.pad(b, (0, 0) + (k // 2,) * 2, mode="reflect"),
                               g.transpose(-1, -2))[0, 0]
        if "cutout" in self.mech:
            for _ in range(int(round(u(0, 2.99 * s)))):
                ch, cw = int(u(0.06, 0.22) * h), int(u(0.06, 0.22) * w)
                y0, x0 = int(u(0, max(h - ch, 1))), int(u(0, max(w - cw, 1)))
                # floor at 0.1 rather than 0: zeroing foreground would punch holes that
                # mimic the sector boundary, confounding the comparison with the
                # acoustic arm, which only ever attenuates.
                out[y0:y0 + ch, x0:x0 + cw] *= u(0.10, 0.5)

        # respect the sector mask, exactly as the acoustic arm does, so the two arms
        # differ only in the nature of the perturbation and not in its support
        out = torch.where(img > 1e-4, out, img)
        return out.clamp(0, 1)
