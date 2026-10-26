"""Toy dataset behind the interpolation-vs-extrapolation figures.

The claim being illustrated
---------------------------
A trained ML model is an interpolator. It is excellent inside the chemistry it has
seen and degrades as you move away from it, because all it has is the data. A
physics-based method like FEP has no training set to be inside or outside of, so its
error is roughly the same everywhere -- worse than ML on home turf, better once you
leave it.

The honest caveat, which belongs in the post
--------------------------------------------
Modelling FEP error as flat is an idealization. Real FEP degrades too, just for
different reasons -- charge changes, large perturbations, scaffold hops that break the
mapping, protonation ambiguity. The difference is that its failure modes track the
*physics of the perturbation*, not the distance to a training set. This toy keeps the
error flat to make the contrast legible; a real version of this plot would show FEP
error rising modestly with perturbation size.

Construction
------------
A 2D latent "chemical space" is embedded into 24 fingerprint-like descriptors, and the
figure runs PCA back down to two components -- so the map is a real PCA of a real (if
synthetic) descriptor matrix rather than a hand-drawn cartoon.
"""

import numpy as np

SEED = 7

N_TRAIN = 70
N_TEST_IN = 45
N_TEST_BRIDGE = 55
N_TEST_OUT = 45

#: Where the new chemotype sits relative to the training cloud, in latent units.
OOD_CENTRE = np.array([3.0, 1.5])
OOD_SPREAD = 0.55

#: Noise on the measured training labels (log units) -- the assay itself.
EXPERIMENTAL_NOISE = 0.20

#: FEP error, flat everywhere. 0.55 log units ~ 0.75 kcal/mol.
SIGMA_FEP = 0.55

#: Kernel ridge hyperparameters. The RBF length scale is ~the training cloud width, so
#: the model interpolates smoothly inside it and reverts to the mean outside.
KRR_LENGTH_SCALE = 1.15
KRR_RIDGE = 0.05


def true_potency(z):
    """Smooth 'true' pIC50 surface over the latent space.

    It keeps varying outside the training cloud -- which is the whole point. A surface
    that flattened out there would let the ML model extrapolate for free.
    """
    x, y = z[:, 0], z[:, 1]
    return (7.0 + 0.40 * x + 0.26 * y + 0.09 * x * y
            - 0.06 * x**2 + 0.45 * np.sin(1.15 * y))


def _rbf(a, b, length_scale=KRR_LENGTH_SCALE):
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2.0 * length_scale**2))


def build(seed=SEED):
    """Return a dict with latent coords, descriptors, PCA scores, truth and predictions."""
    rng = np.random.default_rng(seed)

    z_train = rng.normal(scale=1.25, size=(N_TRAIN, 2))
    z_in = rng.normal(scale=1.25, size=(N_TEST_IN, 2))
    # Analogues strung out between the training chemotype and the new one, so distance
    # from the training set varies continuously rather than in two clumps.
    steps = rng.uniform(0.25, 1.0, (N_TEST_BRIDGE, 1))
    z_bridge = steps * OOD_CENTRE + 0.55 * rng.normal(size=(N_TEST_BRIDGE, 2))
    z_out = OOD_CENTRE + OOD_SPREAD * rng.normal(size=(N_TEST_OUT, 2))
    z = np.vstack([z_train, z_in, z_bridge, z_out])

    split = np.array(["train"] * N_TRAIN + ["in"] * N_TEST_IN
                     + ["bridge"] * N_TEST_BRIDGE + ["out"] * N_TEST_OUT)
    truth = true_potency(z)

    # One shared random embedding for every compound, then PCA back to 2D.
    weights = rng.normal(size=(2, 24))
    desc = z @ weights + 0.35 * rng.normal(size=(len(z), 24))
    centred = desc - desc.mean(0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    scores = centred @ vt[:2].T

    # --- the ML model: kernel ridge, fit on the training compounds only -------------
    measured = truth + rng.normal(0.0, EXPERIMENTAL_NOISE, len(z))
    tr = split == "train"
    offset = measured[tr].mean()
    k_tr = _rbf(z[tr], z[tr])
    alpha = np.linalg.solve(k_tr + KRR_RIDGE * np.eye(tr.sum()), measured[tr] - offset)
    ml_pred = _rbf(z, z[tr]) @ alpha + offset

    # --- the physics model: truth plus a constant error, wherever you are -----------
    fep_pred = truth + rng.normal(0.0, SIGMA_FEP, len(z))

    # Distance to the nearest training compound, in descriptor space -- the usual
    # applicability-domain yardstick.
    d = np.sqrt(((desc[:, None, :] - desc[None, tr, :]) ** 2).sum(-1))
    np.fill_diagonal(d[:N_TRAIN, :], np.inf)          # don't match a compound to itself
    nearest = d.min(1)

    return dict(latent=z, descriptors=desc, scores=scores, split=split, truth=truth,
                measured=measured, ml=ml_pred, fep=fep_pred, nearest=nearest,
                train_mask=tr)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


if __name__ == "__main__":
    d = build()
    print(f"{'subset':<14}{'n':>5}{'ML RMSE':>10}{'FEP RMSE':>10}{'med. dist':>11}")
    for name, label in (("in", "training chemotype"), ("bridge", "intermediate"),
                        ("out", "new chemotype")):
        m = d["split"] == name
        print(f"{label:<14}{m.sum():>5}{rmse(d['ml'][m], d['truth'][m]):>10.2f}"
              f"{rmse(d['fep'][m], d['truth'][m]):>10.2f}"
              f"{np.median(d['nearest'][m]):>11.1f}")
    m = d["train_mask"]
    print(f"{'(training)':<14}{m.sum():>5}{rmse(d['ml'][m], d['truth'][m]):>10.2f}"
          f"{rmse(d['fep'][m], d['truth'][m]):>10.2f}"
          f"{np.median(d['nearest'][m]):>11.1f}")
