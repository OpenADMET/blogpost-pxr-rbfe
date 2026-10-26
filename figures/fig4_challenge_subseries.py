"""Figure 4 -- the PXR blind-challenge subseries (21 compounds, biaryl sulfonamide).

A  binding site, OADMET-0006557 (the crystallised compound)
B  RBFE vs experimental pEC50, one series per His407 tautomer
C  MUE          } 95% bootstrap over ligand resampling AND the per-ligand FEP
D  |r| and |tau|} uncertainties; the FEP term alone is only 19-41% of each interval

Correlations are absolute values on a fixed 0-1 axis, so the two metrics share a
panel. HID407's Kendall tau is genuinely negative (-0.12) and reads as 0.12 here;
signed values are printed.

    python figures/fig4_challenge_subseries.py
"""
import numpy as np
import pandas as pd
from scipy import stats as sps

from common import (DATA, GRID, INK_2, KCAL_PER_LOG, OUT, TAUTOMERS, letter,
                    parity_overlay, plt, shared_limits, show, style, trim)

BOOTSTRAP_N, SEED, PCT = 5000, 20261023, (2.5, 97.5)
CORR_YLIM = (0.0, 1.0)


def _stats(e, p):
    p = p - p.mean() + e.mean()
    return (abs(float(sps.pearsonr(p, e)[0])), float(np.mean(np.abs(p - e))),
            abs(float(sps.kendalltau(p, e)[0])))


def bootstrap(e, p, sigma, fep_only=False):
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(BOOTSTRAP_N):
        i = np.arange(len(e)) if fep_only else rng.integers(0, len(e), len(e))
        v = _stats(e[i], p[i] + rng.normal(0.0, sigma[i]))
        if all(np.isfinite(x) for x in v):
            draws.append(v)
    lo, hi = np.percentile(np.asarray(draws), PCT, axis=0)
    return {k: (lo[j], hi[j]) for j, k in enumerate(("r", "mue", "tau"))}


def load():
    df = pd.read_csv(DATA / "challenge_subseries/dg_predictions.csv")
    out = {}
    for t in TAUTOMERS:
        sub = df[df[f"DG_{t} (kcal/mol) (FECS)"].notna()]
        e = -sub["DG (kcal/mol) (EXPT)"].to_numpy() / KCAL_PER_LOG
        p = -sub[f"DG_{t} (kcal/mol) (FECS)"].to_numpy() / KCAL_PER_LOG
        p = p - p.mean() + e.mean()          # cinnabar convention
        s = sub[f"uncertainty_{t} (kcal/mol) (FECS)"].to_numpy() / KCAL_PER_LOG
        out[t] = dict(n=len(p), expt=e, pred=p, r=float(sps.pearsonr(p, e)[0]),
                      mue=float(np.mean(np.abs(p - e))),
                      tau=float(sps.kendalltau(p, e)[0]),
                      ci=bootstrap(e, p, s), ci_fep=bootstrap(e, p, s, True))
    return out


def _err(v, ci):
    return [[max(0.0, v - ci[0])], [max(0.0, ci[1] - v)]]


def bars(ax, sweep, key, label, ylim=None, groups=None):
    names = list(sweep)
    if groups is None:
        for i, t in enumerate(names):
            ax.bar(i, sweep[t][key], 0.82, color=TAUTOMERS[t], edgecolor="white",
                   lw=0.8, zorder=2)
            ax.errorbar(i, sweep[t][key], yerr=_err(sweep[t][key], sweep[t]["ci"][key]),
                        fmt="none", ecolor=INK_2, elinewidth=1.3, capsize=4, zorder=3)
        ax.set_xticks([])
        ax.set_ylim(0, max(sweep[t]["ci"][key][1] for t in names) * 1.10)
    else:
        base, w = np.arange(len(groups)), 0.26
        for i, t in enumerate(names):
            x = base + (i - (len(names) - 1) / 2) * w
            vals = [abs(sweep[t][k]) for _, k in groups]
            ax.bar(x, vals, w, color=TAUTOMERS[t], edgecolor="white", lw=0.8, zorder=2)
            for xi, (_, k), v in zip(x, groups, vals):
                ax.errorbar(xi, v, yerr=_err(v, sweep[t]["ci"][k]), fmt="none",
                            ecolor=INK_2, elinewidth=1.3, capsize=3, zorder=3)
        ax.set_xticks(base)
        ax.set_xticklabels([g[0] for g in groups], fontsize=13)
        ax.set_ylim(*ylim)
        ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.tick_params(labelsize=12)
    if label:
        ax.set_ylabel(label, fontsize=13)
    style(ax, axis="y")


def main():
    sweep = load()
    lim = shared_limits(*[v for s in sweep.values() for v in (s["expt"], s["pred"])])
    panel_b = parity_overlay(
        [(s["expt"], s["pred"], TAUTOMERS[t], t) for t, s in sweep.items()],
        OUT / "_fig4_parity.png", "pEC$_{50}$", lim, figsize=6.0, transparent=True)

    fig = plt.figure(figsize=(15.0, 6.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.58],
                          wspace=0.11, hspace=0.16)
    ax_a, ax_b = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1])
    ax_c, ax_d = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])
    show(ax_a, OUT / "pose_challenge.png")
    show(ax_b, panel_b)
    bars(ax_c, sweep, "mue", "MUE  (pEC$_{50}$)")
    bars(ax_d, sweep, None, None, CORR_YLIM,
         [("|Pearson r|", "r"), ("|Kendall τ|", "tau")])
    for t, ax in zip("ABCD", (ax_a, ax_b, ax_c, ax_d)):
        letter(ax, t)

    fig.subplots_adjust(left=0.005, right=0.985, top=0.95, bottom=0.07)
    for ax in (ax_c, ax_d):           # after subplots_adjust, which resets positions
        b = ax.get_position()
        ax.set_position([b.x0 + 0.022, b.y0, b.width - 0.022, b.height])
    out = OUT / "fig4_challenge_subseries.png"
    fig.savefig(out, dpi=170, transparent=True)
    print(f"wrote {out}")
    for t, s in sweep.items():
        print(f"  {t} n={s['n']:3d}  r {s['r']:+.3f}  MUE {s['mue']:.3f}  "
              f"tau {s['tau']:+.3f}  (plotted as absolute values)")


if __name__ == "__main__":
    main()
