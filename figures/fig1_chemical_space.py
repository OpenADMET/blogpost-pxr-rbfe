"""Chemical space, coloured by potency and then by each model's error.

One PCA map drawn three times: what the potency actually is, where the ML model gets
it wrong, and where RBFE gets it wrong. The rectangle marks the chemistry the ML model
was trained on; RBFE has no such box, which is the point of the figure.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import pathlib

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

import toy_data as td

HERE = pathlib.Path(__file__).resolve().parent

# --- Design tokens --------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#dedcd5"

#: Sequential ramps: blue is the default magnitude hue, orange the second context.
BLUES = LinearSegmentedColormap.from_list("blues", [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
ORANGES = LinearSegmentedColormap.from_list("oranges", [
    "#fde6d8", "#fac6a6", "#f59f72", "#ee7f48", "#dd5f1c", "#b4440f", "#8a330a"])

#: PC1 carries more variance than PC2; both are scaled to a comparable display range
#: so the map reads as a square and one tick set serves both axes.
PC_DISPLAY_SCALE = 3.2
TICKS = [-10, -5, 0, 5, 10]
#: Padding between the outermost marks and the axes, in display units.
LIMIT_PAD = 0.9

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": GRID, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.titlecolor": INK, "font.size": 10,
})


def training_bounds(pc, train_mask, pad=0.6):
    """(x0, y0, x1, y1) enclosing the training compounds, with breathing room."""
    tr = pc[train_mask]
    x0, y0 = tr.min(0) - pad
    x1, y1 = tr.max(0) + pad
    return x0, y0, x1, y1


def training_box(bounds):
    x0, y0, x1, y1 = bounds
    return Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none",
                     edgecolor=INK, linewidth=1.0, linestyle=(0, (4, 3)), zorder=5)


def main(out=str(HERE / "fig1_chemical_space.png")):
    data = td.build()
    pc = data["scores"] / data["scores"].std(0) * PC_DISPLAY_SCALE
    tr = data["train_mask"]

    err_ml = np.abs(data["ml"] - data["truth"])
    err_rbfe = np.abs(data["fep"] - data["truth"])
    emax = float(np.quantile(np.concatenate([err_ml, err_rbfe]), 0.98))

    # Saturate the top and bottom few percent so the potent cluster separates clearly
    # from the body of the series instead of sitting mid-ramp.
    plo, phi = np.quantile(data["truth"], [0.04, 0.96])
    bounds = training_bounds(pc, tr)
    arrow_tail, arrow_head = (5.0, -5.0), (7.0, 0.0)

    # Crop to the marks themselves rather than a round symmetric window.
    xlim = (min(pc[:, 0].min(), bounds[0]) - LIMIT_PAD,
            max(pc[:, 0].max(), bounds[2], arrow_head[0]) + LIMIT_PAD)
    ylim = (min(pc[:, 1].min(), bounds[1]) - LIMIT_PAD,
            max(pc[:, 1].max(), bounds[3]) + LIMIT_PAD)

    fig = plt.figure(figsize=(12.9, 4.7))
    gs = fig.add_gridspec(1, 9, wspace=0.0,
                          width_ratios=[1, 0.025, 0.05, 0.22, 1, 0.22, 1, 0.025, 0.05])
    ax0 = fig.add_subplot(gs[0])
    axes = [ax0,
            fig.add_subplot(gs[4], sharex=ax0, sharey=ax0),
            fig.add_subplot(gs[6], sharex=ax0, sharey=ax0)]
    caxes = [fig.add_subplot(gs[2]), None, fig.add_subplot(gs[8])]
    panels = [
        (data["truth"], BLUES, plo, phi, "Discovery programs advance toward\nnovel, more potent space",
         True, "pIC$_{50}$"),
        (err_ml, ORANGES, 0.0, emax,
         "ML models excel on in-distribution data\nbut do not extrapolate to novel space",
         True, None),
        (err_rbfe, ORANGES, 0.0, emax,
         "RBFE methods are agnostic to novelty;\nmore potent space is predicted as accurately",
         False, "pIC$_{50}$ prediction error"),
    ]

    for ax, cax, (values, cmap, vmin, vmax, label, show_box, bar) in zip(
            axes, caxes, panels):
        if show_box:
            ax.add_patch(training_box(bounds))
        sc = ax.scatter(pc[:, 0], pc[:, 1], c=values, cmap=cmap, vmin=vmin, vmax=vmax,
                        s=44, edgecolor="white", linewidth=0.5, zorder=3)

        if cax is not None:
            cb = fig.colorbar(sc, cax=cax)
            cb.set_label(bar, fontsize=9, color=INK_2)
            cb.ax.tick_params(labelsize=8.5, color=INK_2)
            cb.outline.set_edgecolor(GRID)

        ax.set_title(label, fontsize=9.5, fontweight="medium", loc="left", pad=9,
                     linespacing=1.4)
        ax.set_xlabel("PC1", fontsize=9.5)
        ax.set_xticks(TICKS)
        ax.set_yticks(TICKS)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color(INK_2)
            ax.spines[side].set_linewidth(0.9)

    for ax in axes:
        ax.tick_params(labelbottom=False, labelleft=False)
    axes[0].set_ylabel("PC2", fontsize=9.5)
    axes[0].text(bounds[0] + 0.45, bounds[3] - 0.45, "Training set", fontsize=8.2,
                 color=INK, ha="left", va="top", zorder=6)

    # The direction a programme actually travels through this space.
    axes[0].annotate("", xy=arrow_head, xytext=arrow_tail,
                     arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6,
                                     shrinkA=0, shrinkB=0), zorder=6)
    fig.subplots_adjust(left=0.05, right=0.925, top=0.83, bottom=0.14)
    # Transparent ground: the figure drops onto whatever the post's background is.
    # The SURFACE rcParams above still set the on-screen axes face, which is what the
    # colourbars and the training-set box are composed against.
    fig.savefig(out, dpi=170, transparent=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
