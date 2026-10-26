"""What is off-target FEP worth as a pre-synthesis filter?

The question
------------
A campaign is optimizing on-target potency. Off-targets (hERG, PXR) are *constraints*,
not objectives: you need a safety margin, and you get no credit for exceeding it. The
realistic status quo is to synthesize compounds for potency and screen them for the
off-target afterwards. The assay is cheap -- but you cannot assay a molecule you have
not made. So the question is what it is worth to move that information in front of the
synthesis decision.

The rule
--------
Rank the proposed designs by predicted off-target margin and synthesize the top
`KEEP`. That is the entire method: one sort and one cut, no calibration, no tuning.

The two free parameters
-----------------------
`q` -- the **off-target attrition rate**: the fraction of the compounds a potency-driven
process already selects that turn out to carry the liability. This single number absorbs
both the liability base rate in the design space and the adverse coupling between potency
and liability, and because it is defined empirically over a program it also absorbs
whatever SAR learning the team does along the way. A med chemist can estimate it from
their own history; nobody can estimate a latent correlation coefficient.

`tau` -- the **ranking power** of the off-target calculation, as Kendall's tau measured
against experiment on your own benchmark.

Everything is one Gaussian integral. Truth and prediction are bivariate normal with
correlation r = sin(pi tau / 2); "truly clean" is the upper 1 - q of the truth and
"synthesized" is the upper KEEP of the prediction, so the yield of the filter is the
orthant probability of that pair.

Why rank-and-cut rather than a threshold
----------------------------------------
An absolute rule ("synthesize if predicted margin > X kcal/mol") needs the predictions
to sit on the right absolute scale. FEP computes *relative* free energies; the absolute
placement comes from anchoring to experiment, and a constant offset is invisible to
every statistic you would validate with. Ranking needs only the ordering, which is the
thing tau measures and the thing the method actually delivers.

It is also why nothing here depends on the spread of true margins across the series.
The previous version of this model converted tau into an equivalent per-compound error
in kcal/mol, which required assuming that spread -- a guess, and the most load-bearing
number in the file. A quantile rule never leaves the probability scale, so the guess is
gone. (The spread still matters for *interpreting* a quoted tau: ranking power is not a
property of a method alone, and the same calculation looks excellent on a series
spanning 4 kcal/mol and useless on one spanning 1. A tau quoted without the dynamic
range it was measured over is not interpretable.)

Assumptions, in the order they would break
------------------------------------------
1. Errors are Gaussian, so there is no heavy tail. Real filter failures are gross
   (wrong binding mode, wrong protonation state), not mild.
2. Errors are independent per compound. A systematic force-field offset would move a
   whole chemotype together and hurt far more.
3. `q` is constant across the campaign; in reality it falls as the team learns.
4. The off-target is the only failure mode -- potency never enters the arithmetic. It is
   assumed handled upstream by whatever selection defines `q`.
5. Designs are cheap and synthesis is expensive. The filter discards viable compounds,
   which is only acceptable if chemistry can propose more analogues than it can make.

Every one of these flatters FEP, so read the output as an upper bound.
"""

from functools import lru_cache

import numpy as np
from scipy.stats import multivariate_normal, norm

#: Fraction of ranked designs carried forward to synthesis.
KEEP = 0.25

#: Cost of making one compound: synthesis alone, so the axis reads as the cost of the
#: chemistry. Rowan assume $2,000 all-in for synthesis AND assay.
COST_PER_COMPOUND = 1_000.0

#: Cost of one off-target FEP calculation.
COST_PER_FEP = 20.0


def pearson_from_kendall(tau):
    """Pearson r implied by a Kendall tau under a bivariate normal.

    Greiner's relation, r = sin(pi tau / 2).

    Kendall rather than Spearman because tau *is* the excess of concordant over
    discordant pairs -- how often you get the order of two compounds right, which is
    exactly the decision a rank-and-cut filter makes. It also degrades more gracefully
    on the small sets FEP is validated on, and tau-b has a principled tie correction for
    the censored values ("> 10 uM") that are endemic in hERG data.

    For the record, since most FEP benchmarks quote Spearman: under the same bivariate
    normal, rho = (6/pi) arcsin(r/2), and tau -> (2/3) rho in the small-correlation
    limit.
    """
    return np.sin(np.pi * np.asarray(tau, dtype=float) / 2.0)


@lru_cache(maxsize=512)
def _frozen_bvn(rho):
    """Frozen bivariate normal, cached: `rho` depends only on tau, not on q."""
    return multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]])


def precision(q, tau, keep=KEEP):
    """Fraction of the synthesized compounds that truly clear the off-target bar.

    Truly clean is Z1 > Phi^-1(q); kept is Z2 > Phi^-1(1 - keep). Their joint upper
    orthant probability, divided by `keep`, is the hit rate among what gets made.
    Depends on nothing but (q, tau, keep).
    """
    rho = round(float(pearson_from_kendall(tau)), 9)
    a, b = norm.ppf(np.atleast_1d(np.asarray(q, dtype=float))), norm.ppf(1.0 - keep)
    joint = (1.0 - norm.cdf(a) - norm.cdf(b)
             + np.atleast_1d(_frozen_bvn(rho).cdf(
                 np.column_stack([a, np.full_like(a, b)]))))
    return joint / keep


def compounds_per_clean(q, tau, keep=KEEP):
    """Compounds synthesized per compound that passes the off-target screen."""
    return 1.0 / precision(q, tau, keep)


def designs_per_clean(q, tau, keep=KEEP):
    """Designs that must be ranked -- and so FEP-scored -- per compound obtained."""
    return 1.0 / (keep * precision(q, tau, keep))


def cost_per_clean(q, tau, keep=KEEP, screening=True):
    """All-in cost per compound that passes the screen: chemistry plus the GPU bill."""
    cost = COST_PER_COMPOUND * compounds_per_clean(q, tau, keep)
    if screening:
        cost = cost + COST_PER_FEP * designs_per_clean(q, tau, keep)
    return cost


def cost_without_filter(q):
    """Status quo: synthesize everything, assay afterwards, remake what failed."""
    return COST_PER_COMPOUND / (1.0 - np.asarray(q, dtype=float))


if __name__ == "__main__":
    qs = (0.2, 0.4, 0.6, 0.8)
    print(f"Rank by predicted off-target margin, synthesize the top {KEEP:.0%}.\n")
    print("Compounds synthesized per compound that passes the screen:\n")
    print("    tau        " + "   ".join(f"q={q:.0%}" for q in qs))
    print("  assay-only   "
          + "   ".join(f"{1 / (1 - q):5.2f}" for q in qs))
    for tau in (0.15, 0.25, 0.40, 0.70):
        cells = "   ".join(
            f"{float(compounds_per_clean(q, tau)[0]):5.2f}" for q in qs)
        print(f"  {tau:.2f}         {cells}")

    print("\nAll-in cost per compound that passes the screen:\n")
    print("    tau        " + "   ".join(f"q={q:.0%}" for q in qs))
    print("  assay-only   "
          + "   ".join(f"{float(cost_without_filter(q)):6,.0f}" for q in qs))
    for tau in (0.15, 0.25, 0.40, 0.70):
        cells = "   ".join(f"{float(cost_per_clean(q, tau)[0]):6,.0f}" for q in qs)
        print(f"  {tau:.2f}         {cells}")
