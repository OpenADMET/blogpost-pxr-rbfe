"""Main figure: synthesis wasted on off-target failures, with and without FEP triage.

The filter is a rank-and-cut: score the proposed designs, synthesize the top KEEP.
The y axis is all-in -- the chemistry plus the GPU bill for scoring everything that
was ranked, including what the cut threw away.
"""

import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import simple_model as sm

HERE = pathlib.Path(__file__).resolve().parent

# --- Design tokens (validated default categorical palette, slots 1-2) ----------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
BASELINE = "#2a78d6"   # slot 1 -- assay-only (status quo)
WITH_FEP = "#eb6834"   # slot 2 -- + off-target FEP
GAP_FILL = "#52514e"
GRID = "#dedcd5"

#: Ranking power is an ORDERED magnitude, so it gets a one-hue ordinal ramp in the
#: "+ FEP" hue -- light = weak ranking, dark = strong. Validated with
#: `validate_palette.js "#f0956a,#ea7540,#d85c18,#9c3a0d" --mode light --ordinal`.
#: tau = 0.40 is the best Kendall reported in Rowan's OpenBind RBFE post.
TAU_LEVELS = [0.15, 0.25, 0.40, 0.70]
TAU_RAMP = ["#f0956a", "#ea7540", "#d85c18", "#9c3a0d"]

#: Every point size in the figure is this multiple of the size the original square
#: panel used. The panel is now 11 inches wide instead of 7.1, so at a fixed column
#: width in the post everything is reproduced about a third smaller; scaling the type
#: with the canvas keeps it reading at the size it was drawn to read at. One knob, so
#: the hierarchy between label, tick, legend and callout cannot drift apart.
FONT_SCALE = 1.5


def pt(size):
    return round(size * FONT_SCALE, 1)

mpl.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": GRID,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.titlecolor": INK,
    "font.size": pt(10),
})

CALLOUT = dict(boxstyle="round,pad=0.45", facecolor="white",
               edgecolor=INK, linewidth=0.9, alpha=1.0)


def panel(ax):
    q = np.linspace(0.02, 0.80, 70)
    reactive = sm.cost_without_filter(q)
    curves = [(t, sm.cost_per_clean(q, t)) for t in TAU_LEVELS]

    # The whole recoverable region, stratified by how well the method ranks.
    ax.fill_between(q, curves[-1][1], reactive, color=GAP_FILL, alpha=0.09, linewidth=0)
    ax.plot(q, reactive, color=BASELINE, linewidth=2.0, zorder=4)
    for (tau, y), color in zip(curves, TAU_RAMP):
        ax.plot(q, y, color=color, linewidth=2.0, zorder=3)
        ax.text(0.812, float(y[-1]), f"τ = {tau:.2f}", color=color, fontsize=pt(9),
                fontweight="bold", ha="left", va="center")

    # Ranking power increases down the family of curves. Placed in AXES FRACTION, not
    # data units: the panel is wide and short, and a position that reads well at one
    # aspect lands on a curve at another. Fractions keep the composition when the
    # figure is reshaped.
    ax.text(0.901, 0.97, "Predictive\nmodel with\nincreasing\nranking signal",
            transform=ax.transAxes, fontsize=pt(8.5), color=INK_2, ha="center",
            va="top", linespacing=1.45,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=INK,
                      linewidth=0.8),
            zorder=7)
    ax.annotate("", xy=(0.901, 0.655), xytext=(0.901, 0.715),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=1.2,
                                shrinkA=0, shrinkB=0), zorder=7)

    # Worked example at the best ranking power anyone has published for RBFE, so the
    # number quoted is one a reader could expect to reproduce rather than aspire to.
    # Indexed once: the curve it names and the colour of its marker cannot drift apart.
    q0, example = 0.50, TAU_LEVELS.index(0.40)
    tau0 = TAU_LEVELS[example]
    r0 = float(sm.cost_without_filter(q0))
    p0 = float(sm.cost_per_clean(q0, tau0)[0])
    best = float(sm.cost_per_clean(q0, TAU_LEVELS[-1])[0])
    ax.plot([q0, q0], [best, r0], color=INK_2, linewidth=0.9, linestyle=(0, (3, 2)),
            zorder=5)
    ax.scatter([q0, q0, q0], [r0, p0, best], s=30, zorder=6,
               color=[BASELINE, TAU_RAMP[example], TAU_RAMP[-1]],
               edgecolor=SURFACE, linewidth=1.5)
    # Dollar signs are escaped. Matplotlib parses $...$ as mathtext within a line, so
    # two on one line would silently set everything between them in italic.
    ax.annotate(
        f"At a 50% failure rate, a compound that passes costs \\${r0:,.0f}.\n"
        f"Rank at τ = {tau0:.2f}, make the top {sm.KEEP:.0%}: \\${p0:,.0f}.",
        xy=(q0, r0), xytext=(0.035, 0.775), textcoords="axes fraction",
        va="top", fontsize=pt(9), color=INK, bbox=CALLOUT,
        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1, shrinkB=7,
                        connectionstyle="angle3,angleA=0,angleB=90"))

    ax.set_xlim(0.02, 0.98)
    ax.set_ylim(760, 5600)
    ax.set_xlabel("Fraction of compounds that fail the off-target screen", fontsize=pt(9.5))
    ax.set_ylabel("All-in cost of a single compound that passes\n"
                  "the off-target screen (synthesis + screening)",
                  fontsize=pt(9.5))
    ax.set_yticks([1000, 2000, 3000, 4000, 5000])
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("${x:,.0f}"))
    ax.set_xticks(np.arange(0.1, 0.81, 0.1))
    ax.tick_params(axis="x", labelsize=pt(9))
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(handles=[
        mpl.lines.Line2D([], [], color=BASELINE, lw=2.4, label="Synthesize all, then assay"),
        mpl.lines.Line2D([], [], color=WITH_FEP, lw=2.4,
                         label=f"Rank by predicted off-target margin, "
                               f"synthesize the top {sm.KEEP:.0%}"),
    ], loc="upper left", fontsize=pt(8.5), frameon=True, facecolor="white",
        edgecolor=GRID, framealpha=0.95, borderpad=0.8, handlelength=1.8)

    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main(out=str(HERE / "fig2_screening_economics.png")):
    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    panel(ax)

    fig.tight_layout()
    # Transparent ground, as for the other figures. The callouts and the legend keep
    # their explicit white fill -- they sit over the plotted curves and have to stay
    # opaque to be legible.
    fig.savefig(out, dpi=170, transparent=True)
    print(f"wrote {out}")
    return fig


if __name__ == "__main__":
    main()
