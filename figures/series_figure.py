"""Figures 5 and 6 -- the two literature series, one figure each.

A  2D scaffold with the series' decorations   B  binding site
C  distance from the ML training set          D  ML parity    E  RBFE parity

A is exported from the interactive scaffold viewer (see README). B is rendered by
render_poses.py. C is the applicability-domain panel: 1 - max Tanimoto (Morgan r=2)
to the nearest of the 4392 ML training compounds, against how far the ML model
missed. Its x axis is fixed 0.5-1.0 in both figures so the two are comparable.

The ML model only ever predicts pEC50, so scoring it against anything else carries a
constant definitional offset that is not model error; both series are in that
position, and C removes it and says so.
"""
import json

import numpy as np
import pandas as pd
from rdkit import RDLogger
from scipy.stats import spearmanr

from common import (DATA, GRID, INK, ML_COLOUR, FEP_COLOUR, OUT, letter, parity,
                    plt, shared_limits, show, trim)

RDLogger.DisableLog("rdApp.*")

DOMAIN_XLIM = (0.5, 1.0)
DOMAIN_FIGSIZE, DOMAIN_DPI = (3.55, 4.3), 200
#: Top margin is deliberate: it matches this panel's plot box to the cinnabar ones,
#: whose own top third is a statistics block. Composed untrimmed for that reason.
DOMAIN_MARGINS = dict(left=0.24, right=0.95, top=0.80, bottom=0.215)


def domain_distances(labels, d):
    """Precomputed distance from each compound to the nearest ML training compound.

    1 - max Tanimoto over Morgan fingerprints (radius 2, 2048 bits). The training
    set itself is not redistributed here, so the distances are shipped rather than
    recomputed; `in_training_set` records the exact-membership check made against
    it at the time.
    """
    t = pd.read_csv(d / "ml_domain.csv").set_index("label")
    return (np.asarray([float(t.loc[n, "train_distance"]) for n in labels]),
            int(sum(bool(t.loc[n, "in_training_set"]) for n in labels)))


def scaffold_2d(path, max_trailing=0.15):
    """The depiction with its stranded 'N compounds' footer dropped.

    Cut at the widest blank band, and only when what sits below it is small enough
    to be a footer rather than part of the drawing.
    """
    img = trim(path)
    rows = (img[:, :, 3] > 0.02).any(axis=1)
    idx = np.where(rows)[0]
    bands, start = [], None
    for i in range(idx[0], idx[-1] + 1):
        if not rows[i]:
            start = i if start is None else start
        elif start is not None:
            bands.append((start, i))
            start = None
    if not bands:
        return img
    cut, resume = max(bands, key=lambda b: b[1] - b[0])
    return img if (idx[-1] - resume) / img.shape[0] > max_trailing else img[:cut]


def domain_panel(dist, err, filename, series, units="log units"):
    fig, ax = plt.subplots(figsize=DOMAIN_FIGSIZE, dpi=DOMAIN_DPI)
    fig.subplots_adjust(**DOMAIN_MARGINS)
    ax.set_box_aspect(1)
    ok = np.isfinite(dist) & np.isfinite(err)
    ax.scatter(dist[ok], err[ok], s=26, color=ML_COLOUR, alpha=0.85,
               edgecolor="white", lw=0.5, zorder=3)
    ax.set_xlim(*DOMAIN_XLIM)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("distance to nearest ML\ntraining compound\n"
                  "(1 $-$ Tanimoto, Morgan r=2)", fontsize=9)
    ax.set_ylabel(f"ML prediction error  ({units})", fontsize=9)
    ax.tick_params(labelsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color="#e4e3dd", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.text(0.97, 0.95, series, transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", color=INK, va="top", ha="right")
    fig.savefig(str(filename), dpi=DOMAIN_DPI, transparent=True)
    plt.close(fig)
    return filename


def build(key, cfg):
    d = DATA / cfg["dir"]
    blob = json.load(open(d / "rbfe_values.json"))[cfg["run"]]
    labels = list(blob["labels"])
    expt = np.asarray(blob["expt"], float)
    pred = np.asarray(blob["pred"], float)
    ml = cfg["ml"](labels, d)

    lim = shared_limits(expt, pred, ml)
    fep_png = parity(expt, pred, OUT / f"_{key}_fep.png", cfg["units"], lim,
                     color=FEP_COLOUR, transparent=True)
    ml_units = cfg.get("ml_units", "log units")
    ml_png = parity(expt, ml, OUT / f"_{key}_ml.png", ml_units, lim,
                    color=ML_COLOUR, transparent=True)

    dist, exact = domain_distances(labels, d)
    raw = ml - expt
    dom_png = domain_panel(dist, np.abs(raw - raw.mean()),
                           OUT / f"_{key}_domain.png", cfg["series"],
                           units=ml_units)

    fig = plt.figure(figsize=(14.4, 10.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.62, 1.0], left=0.02, right=0.99,
                          top=0.96, bottom=0.03, wspace=0.06, hspace=0.05)
    cells = [("A", None, gs[0, :2]), ("B", OUT / f"pose_{key}.png", gs[0, 2]),
             ("C", dom_png, gs[1, 0]), ("D", ml_png, gs[1, 1]),
             ("E", fep_png, gs[1, 2])]
    for tag, path, cell in cells:
        ax = fig.add_subplot(cell)
        if tag == "A":
            ax.imshow(scaffold_2d(d / "scaffold_2d.png"))
            ax.axis("off")
        elif tag == "B":
            img = trim(path)
            ax.imshow(img)
            ax.axis("off")
            # In image coordinates: a square image in a non-square axes does not
            # fill it, so axes fractions would drift into the letterbox.
            ax.text(0.04 * img.shape[1], 0.955 * img.shape[0], cfg["pdb"],
                    fontsize=15, color=INK, va="bottom", ha="left")
        else:
            show(ax, path, do_trim=(tag != "C"))
        letter(ax, tag)

    out = OUT / cfg["out"]
    fig.savefig(out, dpi=170, transparent=True)
    plt.close(fig)
    print(f"wrote {out}   n={len(expt)}  exact ML-training matches: {exact}  "
          f"ML offset removed {raw.mean():+.2f}")
    for name, err in (("RBFE", np.abs(pred - expt)),
                      ("ML", np.abs(raw - raw.mean()))):
        rho, p = spearmanr(dist, err)
        print(f"    {name:<5} |error| vs distance: rho {rho:+.2f} (p {p:.3f})"
              f"   mean |error| {err.mean():.2f}")
    return exact
