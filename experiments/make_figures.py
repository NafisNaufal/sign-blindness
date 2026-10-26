"""Publication figures for the ICIIBMS 2026 manuscript."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.plotting import (use_paper_style, save, C, COL_WIDTH,           # noqa: E402
                          FULL_WIDTH, panel_label, bootstrap_ci)
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.lines import Line2D                                      # noqa: E402

R = Path(__file__).resolve().parents[1]
FIG = R / "paper" / "figures"
use_paper_style()

ec = pd.read_csv(R / "results/e5_echonet/per_patient_ef.csv")


# --------------------------------------------------------------------------- #
# Fig 1: two patients with near-identical Dice, opposite clinical outcomes
# --------------------------------------------------------------------------- #
def fig_teaser():
    import torch
    from src.unet import UNet
    imgs = np.load(R / "data/cache_echonet/images_112.npy", mmap_mode="r")
    msks = np.load(R / "data/cache_echonet/masks_112.npy", mmap_mode="r")
    idx = pd.read_csv(R / "data/cache_echonet/index_112.csv")
    model = UNet(1, 2, width=32).cuda()
    model.load_state_dict(torch.load(R / "results/e5_echonet/best.pt"))
    model.eval()

    band = ec[(ec.dice > 0.93) & (ec.dice < 0.95)]
    bad = band.loc[band.err.idxmax()]
    good = band.loc[(band.err - 1.0).abs().idxmin()]

    fig, axes = plt.subplots(2, 2, figsize=(COL_WIDTH, 3.3), layout="constrained")
    for r, (case, tag) in enumerate([(good, "within 10-point threshold"),
                                     (bad, "large EF error")]):
        rows = idx[idx.key == case.key].sort_values("phase", ascending=False)
        for c, (_, rw) in enumerate(rows.iterrows()):
            j = int(rw.name)
            x = torch.from_numpy(np.array(imgs[j], dtype=np.float32))[None, None].cuda()
            with torch.no_grad():
                p = model(x).argmax(1)[0].cpu().numpy()
            ax = axes[r, c]
            ax.imshow(imgs[j], cmap="gray", vmin=0, vmax=1)
            ax.contour(np.array(msks[j]), [0.5], colors="#00E5FF", linewidths=1.0)
            ax.contour(p, [0.5], colors="#FF3B30", linewidths=1.0, linestyles="--")
            ax.set_title(f"{rw.phase}", fontsize=7, pad=2)
            ax.axis("off")
        # horizontal label inside the frame: rotated side-labels get clipped in the
        # single-column IEEE layout and are the first thing a reader gives up on
        col = C["accent"] if r == 0 else C["uniform"]
        axes[r, 0].text(0.03, 0.97,
                        f"Dice {case.dice:.3f}\nEF error {case.err:.1f} pts\n{tag}",
                        transform=axes[r, 0].transAxes, va="top", ha="left",
                        fontsize=6.2, color=col, linespacing=1.35,
                        bbox=dict(facecolor="black", alpha=0.55, pad=1.5, lw=0))
    axes[0, 1].legend(handles=[Line2D([], [], color="#00E5FF", lw=1.2, label="expert"),
                               Line2D([], [], color="#FF3B30", lw=1.2, ls="--",
                                      label="predicted")],
                      loc="upper right", fontsize=5.5, labelcolor="white",
                      handlelength=1.4, borderpad=0.25)
    save(fig, FIG / "fig1_teaser")
    print(f"fig1: good Dice {good.dice:.3f} err {good.err:.1f} | "
          f"bad Dice {bad.dice:.3f} err {bad.err:.1f}")


# --------------------------------------------------------------------------- #
# Fig 2: the decoupling
# --------------------------------------------------------------------------- #
def fig_decoupling():
    fig, axes = plt.subplots(1, 4, figsize=(FULL_WIDTH, 2.08), layout="constrained")
    # Keep panel labels in a dedicated figure-level gutter. The four narrow axes
    # do not have enough room for labels beside their top y-axis tick values.
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.88))

    ax = axes[0]
    ax.scatter(ec.dice, ec.err, s=1.5, alpha=0.18, color=C["ours"], lw=0, rasterized=True)
    ax.axhline(10, color=C["uniform"], lw=0.9, ls="--")
    ax.text(0.60, 11.5, "large EF error", fontsize=6, color=C["uniform"])
    from scipy.stats import pearsonr
    r, _ = pearsonr(ec.dice, ec.err)
    ax.set_xlabel("segmentation Dice"); ax.set_ylabel("EF error (points)")
    ax.set_xlim(0.6, 1.0); ax.set_ylim(0, 40)
    ax.set_title(f"$R^2$ = {r**2:.3f}", fontsize=7)

    ax = axes[1]
    q = pd.qcut(ec.dice, 4, labels=["Q1\n(worst)", "Q2", "Q3", "Q4\n(best)"])
    rate = ec.groupby(q, observed=True).apply(lambda g: 100 * (g.err > 10).mean())
    err = [1.96 * 100 * np.sqrt(p / 100 * (1 - p / 100) / n)
           for p, n in zip(rate, ec.groupby(q, observed=True).size())]
    ax.bar(range(4), rate.values, yerr=err, color=C["ours"], alpha=0.85,
           edgecolor="white", capsize=2, error_kw={"lw": 0.7})
    ax.set_xticks(range(4)); ax.set_xticklabels(rate.index, fontsize=6)
    ax.set_ylabel("large EF error (%)")
    ax.set_title("best quartile still has errors", fontsize=7)

    ax = axes[2]
    for lab, sub, col in [("EF error $\\leq$ 10", ec[ec.err <= 10], C["accent"]),
                          ("EF error > 10", ec[ec.err > 10], C["uniform"])]:
        ax.hist(sub.dice, bins=45, range=(0.6, 1.0), alpha=0.65, color=col,
                label=f"{lab} (n={len(sub)})", density=True, lw=0)
    ax.set_xlabel("segmentation Dice"); ax.set_ylabel("density")
    ax.legend(fontsize=5.5, loc="upper left")
    ax.set_title("distributions overlap heavily", fontsize=7)

    ax = axes[3]
    threshold = pd.read_csv(R / "results/e7_revisions/threshold_curve.csv")
    ax.plot(threshold.threshold_points, threshold.large_ef_error_rate_percent,
            color=C["ours"], lw=1.4)
    ax.scatter([10], threshold.loc[threshold.threshold_points == 10,
                                   "large_ef_error_rate_percent"],
               color=C["uniform"], s=12, zorder=3)
    ax.axvline(10, color=C["uniform"], lw=0.8, ls="--")
    ax.set_xlabel("error threshold (EF points)")
    ax.set_ylabel("large EF error (%)")
    ax.set_title("threshold sensitivity", fontsize=7)
    ax.set_xlim(0, 20); ax.set_ylim(0, 100)
    fig.canvas.draw()
    for label, panel_ax in zip(("(a)", "(b)", "(c)", "(d)"), axes):
        fig.text(panel_ax.get_position().x0, 0.98, label,
                 fontsize=9, fontweight="bold", va="top", ha="left")
    save(fig, FIG / "fig2_decoupling")


# --------------------------------------------------------------------------- #
# Fig 3: quality stratification (CAMUS) + why CAMUS cannot power a method claim
# --------------------------------------------------------------------------- #
def fig_quality():
    ca = pd.read_csv(R / "results/e2_ef_all_arms.csv")
    ca = ca[ca.arm == "none"]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 1.95), layout="constrained")

    ax = axes[0]
    order = ["Good", "Medium", "Poor"]
    dice_cv = pd.read_csv(R / "results/e1_cv_baseline/per_frame.csv")
    vals = [dice_cv[dice_cv.quality == q].dice_mean.values for q in order]
    bp = ax.boxplot(vals, tick_labels=order, widths=0.55, showfliers=False, patch_artist=True)
    for b, col in zip(bp["boxes"], [C["accent"], C["warn"], C["uniform"]]):
        b.set_facecolor(col); b.set_alpha(0.7); b.set_linewidth(0.6)
    for med in bp["medians"]:
        med.set_color("black"); med.set_linewidth(1.0)
    ax.set_ylabel("Dice"); ax.set_title("segmentation: small gap", fontsize=7)
    panel_label(ax, "(a)", dx=-0.26)

    ax = axes[1]
    rates, errs = [], []
    for q in order:
        s = ca[ca.quality == q].err.values
        p = 100 * (s > 10).mean(); rates.append(p)
        errs.append(1.96 * 100 * np.sqrt(p / 100 * (1 - p / 100) / len(s)))
    ax.bar(range(3), rates, yerr=errs, color=[C["accent"], C["warn"], C["uniform"]],
           alpha=0.85, edgecolor="white", capsize=2, error_kw={"lw": 0.7})
    ax.set_xticks(range(3)); ax.set_xticklabels(order)
    ax.set_ylabel("large EF error (%)")
    ax.set_title("clinical: 3.3$\\times$ gap", fontsize=7)
    panel_label(ax, "(b)", dx=-0.28)

    ax = axes[2]
    p1, p2 = 0.383, 0.255                      # observed none vs acoustic on the poor stratum
    from scipy.stats import norm
    za, zb = 1.96, 0.8416
    ns = np.arange(20, 600, 1)
    se = np.sqrt(p1 * (1 - p1) / ns + p2 * (1 - p2) / ns)
    power = 1 - norm.cdf(za - abs(p1 - p2) / se)
    # required n from the standard two-proportion formula, so the figure and the text
    # quote the same number rather than a grid-resolution approximation of it
    pbar = (p1 + p2) / 2
    need = int(np.ceil(((za * np.sqrt(2 * pbar * (1 - pbar))
                         + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2)
                       / (p1 - p2) ** 2))
    ax.plot(ns, 100 * power, color=C["ours"])
    ax.axhline(80, color=C["random"], lw=0.8, ls=":")
    ax.axvline(47, color=C["uniform"], lw=1.0)
    # offset clear of the vertical rule: at x=52 the label sat on top of the line
    ax.annotate("CAMUS\n(n=47)", xy=(47, 26), xytext=(72, 34),
                fontsize=5.5, color=C["uniform"], ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=C["uniform"], lw=0.6,
                                shrinkA=0, shrinkB=1))
    ax.axvline(need, color=C["oracle"], lw=1.0, ls="--")
    ax.text(need + 18, 52, f"needed\n(n={need})", fontsize=5.5)
    ax.set_xlabel("poor-quality patients per arm"); ax.set_ylabel("power (%)")
    ax.set_title("CAMUS cannot power a method claim", fontsize=7)
    panel_label(ax, "(c)", dx=-0.26)
    save(fig, FIG / "fig3_quality_power")



# --------------------------------------------------------------------------- #
# Fig 4: Bland-Altman, the clinical standard for method agreement
# --------------------------------------------------------------------------- #
def fig_bland_altman():
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.08), layout="constrained")
    # Match Figure 2: panel labels occupy a dedicated figure-level gutter,
    # while panel titles remain on their own row below.
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.88))

    ax = axes[0]
    mean_ef = (ec.ef_pred + ec.ef_gt) / 2
    diff = ec.ef_pred - ec.ef_gt
    bias, sd = diff.mean(), diff.std()
    ax.scatter(mean_ef, diff, s=1.5, alpha=0.16, color=C["ours"], lw=0, rasterized=True)
    for y, ls, lab in [(bias, "-", f"bias {bias:+.2f}"),
                       (bias + 1.96 * sd, "--", f"+1.96 SD  {bias+1.96*sd:+.1f}"),
                       (bias - 1.96 * sd, "--", f"-1.96 SD  {bias-1.96*sd:+.1f}")]:
        ax.axhline(y, color=C["uniform"], ls=ls, lw=0.9)
        ax.text(97, y + 0.8, lab, fontsize=5.5, ha="right", color=C["uniform"])
    ax.set_xlabel("mean of predicted and reference EF (%)")
    ax.set_ylabel("predicted $-$ reference EF")
    ax.set_title("EchoNet-Dynamic (n=2564)", fontsize=7)

    # clinical consequence: misclassification across the HF category boundary
    ax = axes[1]
    cuts = [40, 50]
    names = ["reduced\n($<$40)", "mid-range\n(40-50)", "preserved\n($\\geq$50)"]
    gt_c = np.digitize(ec.ef_gt, cuts)
    pr_c = np.digitize(ec.ef_pred, cuts)
    M = np.zeros((3, 3))
    for g, p in zip(gt_c, pr_c):
        M[g, p] += 1
    Mn = M / M.sum(1, keepdims=True) * 100
    im = ax.imshow(Mn, cmap="Blues", vmin=0, vmax=100)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{Mn[i,j]:.0f}%", ha="center", va="center", fontsize=6.5,
                    color="white" if Mn[i, j] > 55 else "black")
    ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=5.5)
    ax.set_yticks(range(3)); ax.set_yticklabels(names, fontsize=5.5)
    ax.set_xlabel("predicted category"); ax.set_ylabel("reference category")
    mis = 100 * (gt_c != pr_c).mean()
    ax.set_title(f"heart-failure misclassification: {mis:.1f}%", fontsize=7)
    fig.canvas.draw()
    for label, panel_ax in zip(("(a)", "(b)"), axes):
        fig.text(panel_ax.get_position().x0, 0.98, label,
                 fontsize=9, fontweight="bold", va="top", ha="left")
    save(fig, FIG / "fig4_bland_altman")
    print(f"fig4: bias {bias:+.2f}, LoA [{bias-1.96*sd:.1f}, {bias+1.96*sd:.1f}], "
          f"HF misclassification {mis:.1f}%")



# --------------------------------------------------------------------------- #
# Fig 5 (revision): the mechanism. Dice is sign-blind; EF is not.
# --------------------------------------------------------------------------- #
def fig_mechanism():
    import pandas as _pd
    nm = _pd.read_csv(R / "results/e7_revisions/r1_null_model.csv")
    sw = _pd.read_csv(R / "results/e8_sweep/sweep.csv")
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 1.95), layout="constrained")

    # (a) null model: unbiased boundary noise alone reproduces the decoupling
    ax = axes[0]
    from scipy.stats import pearsonr
    r, _ = pearsonr(nm.dice, nm.err)
    ax.scatter(nm.dice, nm.err, s=1.6, alpha=0.20, color=C["accent"], lw=0, rasterized=True)
    ax.set_xlabel("Dice (perturbed expert masks)")
    ax.set_ylabel("EF error (points)")
    ax.set_ylim(0, 60)
    ax.set_title(f"null model: $R^2$ = {r**2:.3f}", fontsize=7)
    ax.text(0.03, 0.94, "observed in\nreal model: 0.194", transform=ax.transAxes,
            fontsize=5.5, va="top", color=C["ours"])
    panel_label(ax, "(a)", dx=-0.26)

    # (b) the mechanism: sign concordance drives EF error, invisibly to Dice
    ax = axes[1]
    same = nm[nm.same_sign].err
    opp = nm[~nm.same_sign].err
    bp = ax.boxplot([same, opp], tick_labels=["same\nsign", "opposite\nsign"],
                    widths=0.55, showfliers=False, patch_artist=True)
    for b, col in zip(bp["boxes"], [C["accent"], C["uniform"]]):
        b.set_facecolor(col); b.set_alpha(0.7); b.set_linewidth(0.6)
    for m_ in bp["medians"]:
        m_.set_color("black"); m_.set_linewidth(1.0)
    ax.set_ylabel("EF error (points)")
    ax.set_title(f"{opp.mean()/same.mean():.2f}$\\times$, invisible to Dice", fontsize=7)
    panel_label(ax, "(b)", dx=-0.28)

    # (c) across models: Dice barely moves while clinical failure varies
    ax = axes[2]
    d_rel = 100 * (sw.dice.max() - sw.dice) / sw.dice.max()
    c_rel = 100 * (sw.catastrophic - sw.catastrophic.min()) / sw.catastrophic.min()
    x = np.arange(len(sw))
    ax.bar(x - 0.2, d_rel, 0.4, label="Dice", color=C["ours"], alpha=0.9, lw=0)
    ax.bar(x + 0.2, c_rel, 0.4, label="large EF error", color=C["uniform"],
           alpha=0.9, lw=0)
    ax.set_xticks(x); ax.set_xticklabels(sw.model, fontsize=5, rotation=20, ha="right")
    ax.set_ylabel("% range / best value")
    ax.legend(fontsize=5.5, loc="upper left")
    ax.set_title("across models", fontsize=7)
    panel_label(ax, "(c)", dx=-0.30)
    save(fig, FIG / "fig5_mechanism")
    print(f"fig5: null R2={r**2:.3f}, sign ratio {opp.mean()/same.mean():.2f}x, "
          f"Dice spread {100*(sw.dice.max()-sw.dice.min())/sw.dice.max():.2f}% vs "
          f"large-error spread {100*(sw.catastrophic.max()-sw.catastrophic.min())/sw.catastrophic.min():.1f}%")


if __name__ == "__main__":
    fig_teaser()
    fig_decoupling()
    fig_quality()
    fig_bland_altman()
    fig_mechanism()
    print("figures ->", FIG)
