"""Figure 7 -- the Werner series on Schrodinger's structure, head to head.

Figure 6 scores 35 Werner compounds on a 6HTY-derived receptor. Schrodinger
published RBFE on 42 of the same compounds, around their own prepared model of
the same structure (Table 8 of their PXR paper). This figure puts the two on one
parity panel against the same experimental pMEC column, in the same units, with
the same centring -- the only direct comparison in the repository between this
project's numbers and a commercial package's on identical compounds.

Panel A is Schrodinger's receptor with Werner-25 reposed into it, which is the
input the predictions in panel B were computed from. It is NOT the 6HTY pose
panel of figure 6: the receptor is their model and the ligand was re-embedded
against a different template (1.86 A from the crystal ligand, moving Werner-25
by 1.9 A).

BOTH SERIES ARE RELATIVE free energies, so each carries an arbitrary additive
offset and each is shifted onto the experimental mean before plotting -- the
same centring the statistics use, so the dots and the numbers are one object.
The shipped `pred` columns are already centred.

RMSE CONVENTION. Schrodinger's published table reports an all-pairs ddG RMSE,
which at n = 42 is sqrt(2n/(n-1)) = 1.43x the centred dG RMSE computed here. The
numbers in this panel are dG RMSE for BOTH series, so they are comparable with
each other and with figures 4-6 -- but the Schrodinger column here is NOT the
number printed in their table. Converted, their published row (R2 0.54, RMSE
1.17 in ddG kcal/mol, slope 0.77) is reproduced by the recomputation to within
0.002 / 0.01 / 0.008.

    python figures/fig7_werner_n42_schrodinger.py
"""
import json

import numpy as np
import pandas as pd
from cinnabar import stats as cinnabar_stats

from common import DATA, FEP_COLOUR, ML_COLOUR, INK, OUT, STATISTICS, plt, trim

SUB = DATA / "werner_schrodinger_n42"
NETWORK = "schro42|LOMAP k=4|ddg_sem<=1.0|pMEC"
POSE = OUT / "pose_schro.png"

#: cinnabar's guideline widths, in the plotted unit.
BANDS = (0.5, 1.0)
NBOOT = 1000


def mets(pred, expt, seed=0):
    """cinnabar's four statistics, each with its 95% bootstrap interval.

    Computed by cinnabar's own `bootstrap_statistic` so the numbers are produced
    exactly as the panels of figures 4-6 produce theirs: compounds resampled with
    replacement, 1000 times, interval from the 2.5/97.5 percentiles. The point
    value is the statistic on the real data ("mle"), not the bootstrap mean.

    cinnabar draws from numpy's global RNG and does not seed it, so its intervals
    move between runs; seeding here makes this panel reproducible.
    """
    np.random.seed(seed)
    out = {}
    for name in STATISTICS:
        r = cinnabar_stats.bootstrap_statistic(
            np.asarray(expt, float), np.asarray(pred, float),
            statistic=name, nbootstrap=NBOOT)
        out[name] = (float(r["mle"]), float(r["low"]), float(r["high"]))
    return out


def main():
    blob = json.load(open(SUB / "rbfe_values.json"))[NETWORK]
    labels = blob["labels"]
    expt = np.asarray(blob["expt"], float)
    ours = np.asarray(blob["pred"], float)

    theirs = pd.read_csv(SUB / "schrodinger_predictions.csv").set_index("label")
    theirs = theirs.loc[labels, "pred_pMEC"].to_numpy(float)
    assert len(expt) == len(ours) == len(theirs) == 42

    ma, mb = mets(ours, expt), mets(theirs, expt)

    # Three columns: the render, the scatter, and the statistics beside it.
    # Putting the statistics to the RIGHT of the plot rather than above it is
    # what lets both panels use the full height of the canvas -- a block above
    # the plot is paid for out of the plot's own height, and with two series it
    # is tall enough to cost about a fifth of it.
    FIG_W, FIG_H = 12.3, 5.9
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=170)
    BOX = dict(bottom=0.150, height=0.780)

    # Panel A gets the same rectangle as panel B's plot box so the two squares
    # sit on the same top and bottom lines; in a taller axes the square image
    # would centre itself and float away from B. Nothing is drawn over the
    # render -- HIS407 is burnt in by PyMOL and is the only label it carries.
    ax_p = fig.add_axes([0.012, BOX["bottom"], 0.374, BOX["height"]])
    ax_p.imshow(trim(POSE))
    ax_p.axis("off")
    fig.text(0.008, 0.990, "A", fontsize=16, fontweight="bold", va="top", color=INK)

    # 0.447 leaves 0.75 in between the two squares, which is what the y label
    # and its tick labels need; any less and "Calculated pMEC" sits on panel A.
    ax = fig.add_axes([0.447, BOX["bottom"], 0.374, BOX["height"]])
    ax.set_box_aspect(1)
    lo = float(min(expt.min(), ours.min(), theirs.min())) - 0.4
    hi = float(max(expt.max(), ours.max(), theirs.max())) + 0.4
    g = np.array([lo, hi])
    for bw in BANDS:
        ax.fill_between(g, g - bw, g + bw, color="grey", alpha=0.2, lw=0, zorder=0)
    ax.plot(g, g, color="black", lw=1.2, ls=":", zorder=1)
    ax.scatter(expt, theirs, s=42, color=ML_COLOUR, alpha=0.9,
               edgecolor="white", linewidth=0.6, zorder=3)
    ax.scatter(expt, ours, s=42, color=FEP_COLOUR, alpha=0.9,
               edgecolor="white", linewidth=0.6, zorder=4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Experimental pMEC", fontsize=14)
    ax.set_ylabel("Calculated pMEC", fontsize=14)
    ax.tick_params(labelsize=12, colors="black")
    for sp in ax.spines.values():
        sp.set_color("black")
        sp.set_linewidth(0.9)
    fig.text(0.432, 0.990, "B", fontsize=16, fontweight="bold", va="top", color=INK)

    # One block per series, stacked rather than one table with two value
    # columns: stacked, each is only as wide as a single cinnabar line and the
    # column fits beside the plot. The pair is centred on the plot's vertical
    # span so it reads as belonging to the axes rather than floating at the top.
    X0, FS, LAB_W = 0.841, 9.5, 7
    PITCH = FS * 1.55 / 72.0 / FIG_H
    GAP = 1.6                                     # blank lines between blocks
    n_lines = 2 * (len(STATISTICS) + 2) + GAP     # header + N + rows, twice
    y = (BOX["bottom"] + BOX["height"] / 2.0) + (n_lines - 1) * PITCH / 2.0

    for lab, m, col in (("this work", ma, FEP_COLOUR),
                        ("Schrödinger", mb, ML_COLOUR)):
        fig.text(X0, y, "●", fontsize=11, color=col, va="center")
        fig.text(X0 + 0.016, y, lab, fontsize=11, color=INK, va="center")
        y -= PITCH
        fig.text(X0, y, f"{'N:':<{LAB_W}s}{len(expt)}", fontsize=FS, color=INK,
                 va="center", family="monospace")
        for name in STATISTICS:
            y -= PITCH
            v = m[name]
            fig.text(X0, y, f"{name + ':':<{LAB_W}s}{v[0]:.2f} [{v[1]:.2f},{v[2]:.2f}]",
                     fontsize=FS, color=INK, va="center", family="monospace")
        y -= PITCH * GAP

    out = OUT / "fig7_werner_n42_schrodinger.png"
    fig.savefig(str(out), dpi=170)
    plt.close(fig)
    print(f"wrote {out}")
    for lab, m in (("this work ", ma), ("Schrodinger", mb)):
        print(f"  {lab} " + "  ".join(
            f"{k} {m[k][0]:.2f} [{m[k][1]:.2f},{m[k][2]:.2f}]" for k in STATISTICS))


if __name__ == "__main__":
    main()
