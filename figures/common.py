"""Shared plotting helpers."""
import pathlib
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import matplotlib as mpl  # noqa: E402

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from cinnabar.plotting import _master_plot  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "figures"

#: kcal/mol per log10 unit at 298 K.
KCAL_PER_LOG = 1.363

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dedcd5"
FEP_COLOUR, ML_COLOUR = "#2a78d6", "#eb6834"
TAUTOMERS = {"HID407": "#2a78d6", "HIE407": "#eb6834", "HIP407": "#1baf7a"}
STATISTICS = ["RMSE", "MUE", "R2", "KTAU"]
ALPHA_FLOOR = 0.02

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "DejaVu Sans",
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10,
})


def trim(path, margin=12):
    """Crop uniform or transparent background off a PNG.

    The alpha floor is just above zero: a PyMOL cartoon drawn with
    cartoon_transparency is content at alpha 0.4, and a half-way threshold
    discards it.
    """
    img = plt.imread(str(path))
    if img.ndim == 3 and img.shape[2] == 4 and img[:, :, 3].min() < 0.5:
        mask = img[:, :, 3] > ALPHA_FLOOR
    else:
        rgb = img[:, :, :3] if img.ndim == 3 else np.dstack([img] * 3)
        bg = np.mean([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]], axis=0)
        mask = np.abs(rgb - bg).max(axis=2) > 6 / 255
    if not mask.any():
        return img
    r, c = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    return img[max(r[0] - margin, 0):r[-1] + margin + 1,
               max(c[0] - margin, 0):c[-1] + margin + 1]


def show(ax, path, do_trim=True):
    ax.imshow(trim(path) if do_trim else plt.imread(str(path)))
    ax.axis("off")


def letter(ax, text, dx=0.0):
    ax.text(dx, 1.0, text, transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="bottom", ha="left")


def _relabel(fig, units):
    """cinnabar hardcodes 'Experimental  (units)'; rebuild without the brackets."""
    ax = fig.axes[0]
    ax.set_xlabel(f"Experimental {units}")
    ax.set_ylabel(f"Calculated {units}")
    return ax


def parity(expt, pred, filename, units, xy_lim, color=FEP_COLOUR, figsize=4.2,
           statistics=None, transparent=False):
    """One cinnabar parity panel with its bootstrapped statistics."""
    fig = _master_plot(
        np.asarray(expt, float), np.asarray(pred, float),
        xlabel="Experimental", ylabel="Calculated", quantity="", units=units,
        statistics=STATISTICS if statistics is None else statistics,
        xy_lim=list(xy_lim), color=color, guidelines=True, origins=False,
        centralizing=True, figsize=figsize, dpi=200, filename=str(filename))
    _relabel(fig, units)
    fig.savefig(str(filename), bbox_inches="tight", dpi=200, transparent=transparent)
    plt.close(fig)
    return filename


def parity_overlay(series, filename, units, xy_lim, figsize=4.2, transparent=False):
    """Several series on one cinnabar panel, no statistics annotation.

    `series` is a list of (expt, pred, colour, label).
    """
    e0, p0, c0, _ = series[0]
    fig = _master_plot(
        np.asarray(e0, float), np.asarray(p0, float),
        xlabel="Experimental", ylabel="Calculated", quantity="", units=units,
        statistics=[], xy_lim=list(xy_lim), color=c0, guidelines=True,
        origins=False, centralizing=True, figsize=figsize, dpi=200)
    ax = _relabel(fig, units)
    ax.set_title("")
    for a in fig.findobj(match=plt.Text):
        if "(N =" in (a.get_text() or ""):
            a.set_visible(False)
    for e, p, c, _ in series[1:]:
        ax.scatter(np.asarray(e, float), np.asarray(p, float), s=20, color=c, zorder=3)
    ax.legend([plt.Line2D([], [], marker="o", ls="", color=c, markersize=5)
               for *_, c, _ in series], [lab for *_, lab in series],
              loc="lower right", fontsize=8, framealpha=0.95)
    fig.savefig(str(filename), bbox_inches="tight", dpi=200, transparent=transparent)
    plt.close(fig)
    return filename


def shared_limits(*series, pad=0.6):
    v = np.concatenate([np.asarray(s, float).ravel() for s in series])
    return [float(np.floor(v.min() - pad)), float(np.ceil(v.max() + pad))]


def style(ax, axis="both"):
    ax.grid(True, axis=axis, color=GRID, lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
