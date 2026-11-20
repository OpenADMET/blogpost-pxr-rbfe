# RBFE on PXR — data and figures

Source data, inputs and plotting code for the OpenADMET blog post *"Alchemical methods
for antitargets: a showcase of first steps in exploring PXR using RBFE"*.

The pregnane X receptor is a promiscuous antitarget: hitting it induces CYP3A4 and
causes drug–drug interactions. ML models predict it well in-distribution and poorly
outside it, which is exactly where discovery programmes go. This repository holds the
first attempt at the physics-based alternative — relative binding free energy (RBFE)
calculations on three PXR congeneric series, run entirely on free, open-source
software: [OpenFE]'s `RelativeHybridTopologyProtocol`, [OpenFF] parameters, and
[alchemiscale] for distribution.

The short version of the result: RBFE gave no signal on the blind-challenge subseries
(cell-based pEC50), a poor-to-modest signal on Li et al. (TR-FRET pIC50), and a
modest-to-strong signal on Werner et al. (transactivation pMEC) that matches
Schrodinger's published numbers on the same structure.

Only the networks whose results appear in a published figure are included.

[OpenFE]: https://openfree.energy/
[OpenFF]: https://openforcefield.org/
[alchemiscale]: https://github.com/openforcefield/alchemiscale

## Setup

```bash
conda env create -f environment.yml
conda activate openadmet-pxr-rbfe
pytest -q
```

`numpy` is pinned below 2.3: cinnabar 0.5.0 assigns a size-1 array into a scalar slot
inside its bootstrap, which numpy removed in 2.3. Figures 5 and 6 raise `ValueError`
above the pin.

## Reproducing the figures

Everything regenerates from a fresh clone. Run from the repository root, in this order
— figures 4–7 need the binding-site renders, which are the one slow step:

```bash
python -m pymol -cq figures/render_poses.py   # pose_{challenge,li,werner}.png, ~1 min
python figures/fig1_chemical_space.py
python figures/fig2_screening_economics.py
python figures/fig3_scaffold_clusters.py
python figures/fig4_challenge_subseries.py
python figures/fig5_li.py
python figures/fig6_werner.py
python figures/fig7_werner_n42_schrodinger.py
```

Each script writes its PNG next to itself and is idempotent. `pytest -q` checks the
shipped data still reproduces the published statistics before you trust any of it.

## Figures

| # | Script | Data | What it shows |
|---|---|---|---|
| 1 | [`fig1_chemical_space.py`](figures/fig1_chemical_space.py) | `toy_data.py` (**mock**) | ML error grows as a programme moves off its training distribution; RBFE does not |
| 2 | [`fig2_screening_economics.py`](figures/fig2_screening_economics.py) | `simple_model.py` (analytic) | Cost per clean compound when you rank by predicted off-target margin and synthesize the top 25% |
| 3 | [`fig3_scaffold_clusters.py`](figures/fig3_scaffold_clusters.py) | [`data/challenge_clusters/`](data/challenge_clusters) | The PXR blind-challenge test set is too chemically fragmented for RBFE; only one subseries (n = 21) is large enough to score |
| 4 | [`fig4_challenge_subseries.py`](figures/fig4_challenge_subseries.py) | [`data/challenge_subseries/`](data/challenge_subseries) | That 21-compound biaryl sulfonamide subseries, across three His407 tautomers — no signal against cell-based pEC50 |
| 5 | [`fig5_li.py`](figures/fig5_li.py) | [`data/li_et_al/`](data/li_et_al) | Li et al. 2021 SPA70 analogues (n = 64), scored on TR-FRET binding pIC50 |
| 6 | [`fig6_werner.py`](figures/fig6_werner.py) | [`data/werner_et_al/`](data/werner_et_al) | Werner et al. 2019 analogues (n = 35), scored on PXR transactivation pMEC |
| 7 | [`fig7_werner_n42_schrodinger.py`](figures/fig7_werner_n42_schrodinger.py) | [`data/werner_schrodinger_n42/`](data/werner_schrodinger_n42) | The same series on Schrodinger's own receptor and their own n = 42 compound set|

Figures 1 and 2 carry no experimental data — figure 1 is explicitly mock, figure 2 is
an analytic cost model. Everything from figure 3 on is real.

Shared code lives in [`figures/common.py`](figures/common.py) (cinnabar panels, image
trimming, styling) and [`figures/series_figure.py`](figures/series_figure.py) (the
five-panel layout shared by figures 5 and 6).

**Figure 3 in the published post differs from the one here.** The post composites this
network panel with an export from the interactive scaffold viewer (the same viewer that
produces panel A of figures 5 and 6, which is not vendored here) and redraws the
central network with dashed edges. This repository regenerates the network panel only.

## RBFE settings

Identical across all three series unless noted. One network per His407 tautomer.

| | |
|---|---|
| Protocol | OpenFE `RelativeHybridTopologyProtocol`, Hamiltonian replica exchange |
| Force fields | Amber ff14SB (protein), OpenFF 2.2.0 (ligand), TIP3P water |
| Partial charges | AM1-BCC (AmberTools) |
| Solvation | dodecahedral box, 1.5 nm padding, 0.15 M NaCl, neutralised |
| Nonbonded | PME, 0.9 nm cutoff |
| Integrator | Langevin, 4 fs timestep, H-mass repartitioning to 3 amu, H-bond constraints, 298.15 K, 1 bar |
| Alchemical | 11 λ windows / 11 replicas, Gapsys soft-core LJ (α = 0.85) |
| Per edge | 5000 steps minimisation, 1 ns equilibration, then production |
| Production | **15 ns** (figure 4) · **5 ns** (figures 5 and 6) |
| Repeats | 1 network (figures 4 and 5) · 3 independent networks, ΔΔG averaged (figure 6) |
| Network planning | LOMAP atom mapper, `default_lomap` scorer, minimal-redundancy planner (redundancy 2) |

Edge ΔΔGs are combined into per-ligand ΔGs with cinnabar's MLE, then shifted onto the
experimental mean before scoring. Run through [alchemiscale] on GPU.

## Data

### `data/challenge_clusters/` — figure 3

The [PXR blind challenge] test set: 513 compounds, 369 Bemis–Murcko scaffolds, the
largest holding 21. `compounds.csv` carries the per-compound table (SMILES, pEC50,
scaffold, RBFE-network assignments at three stringencies); `scaffold_sizes.csv` and
`summary.json` are the clustering results. Upstream source:
[openadmet/pxr-challenge-train-test](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test) (CC-BY).

### `data/challenge_subseries/` — figure 4

The subseries pulled out of that test set: 21 compounds decorating a biaryl sulfonamide
on six vectors. Poses were generated from the UCSF structure of **OADMET-0006557**, the
one member with experimental coordinates.

| File | |
|---|---|
| `receptor_{HID,HIE,HIP}407.pdb` | the three His407 tautomers, each its own RBFE network |
| `posed_ligands.sdf` | 23 posed ligands |
| `dg_predictions.csv` | RBFE ΔG and per-ligand uncertainty per tautomer, with experiment |

All three tautomers appear in the figure, so all three are here.

[PXR blind challenge]: https://openadmet.org/

### `data/li_et_al/` — figure 5

Li et al., *J. Med. Chem.* **2021**, 64, 1733–1761,
[doi:10.1021/acs.jmedchem.0c02201](https://doi.org/10.1021/acs.jmedchem.0c02201) —
SPA70 analogues around [PDB 5X0R](https://www.rcsb.org/structure/5X0R) (Lin et al.,
*Nat. Commun.* 2017, [doi:10.1038/s41467-017-00780-5](https://doi.org/10.1038/s41467-017-00780-5)).

`paper_activities.csv` carries two of the paper's four assay columns:

- `IC50` — hPXR **binding**, TR-FRET competition against T0901317
- `EC50` — cell-based hPXR **agonist** potency, rifampicin reference

Its 64 rows are the subset of the paper's 81 analogues with both values, which is why
the inverse agonists and antagonists are absent: their agonist EC50 is reported `NA`.
**Figure 5 is scored on pIC50**, the binding affinity, because that is the quantity an
RBFE calculation computes; the cellular EC50 sits downstream of permeability and
metabolism. The two correlate at r = 0.68 with a slope of 0.77, so they are related but
not interchangeable.

Only the `benchmark|HIP|pIC50` network reached the figure, so only that one is in
`rbfe_values.json` and only the HIP407 receptor is shipped.

### `data/werner_et_al/` — figure 6

Werner et al., *J. Med. Chem.* **2019**, 62, 11194–11217,
[doi:10.1021/acs.jmedchem.9b01304](https://doi.org/10.1021/acs.jmedchem.9b01304),
around [PDB 6HTY](https://www.rcsb.org/structure/6HTY). These are P2X4 antagonists;
PXR is their off-target CYP3A4-induction liability, so the PXR activities come from
ChEMBL rather than the paper.

`chembl_activities.csv` is the raw ChEMBL pull (assay `CHEMBL4324753`) — endpoint
**MEC**, the lowest concentration at which PXR transactivation clears a fixed threshold
in HepG2 or DPX-2 cells carrying a CYP3A4-promoter luciferase reporter. Werner tabulate
50 compounds; 5 are censored (`standard_relation` `>`) and excluded, and the pose
filter drops a further 14, which is why the figure reports n = 35. Schrodinger report
n = 42 on the same series without stating which 8 they dropped.

Two caveats on the endpoint: a MEC is a threshold crossing, so it depends on efficacy
and on the assay's noise floor, not only on affinity; and the paper reports HepG2 **or**
DPX-2 without saying which per compound, and the two use different thresholds (4× the
DMSO SD vs > 10% of the rifampicin control), so the column may pool two conventions.

### `data/werner_schrodinger_n42/` — figure 7

The same Werner series as figure 6, rerun to be directly comparable with Schrodinger's
published numbers: their receptor, their 42-compound selection, their endpoint.

Schrodinger, *J. Chem. Theory Comput.* **2026**,
[doi:10.1021/acs.jctc.6c01007](https://doi.org/10.1021/acs.jctc.6c01007) — *"Accelerating
Lead Optimization away from AntiTargets: Navigating the Structural Complexity of
Promiscuous ADMET Proteins"*. They do not use the 6HTY crystal directly: the paper builds
PXR models by cross-docking the three most potent ligands into 27 PXR structures
*excluding* 6HTY, varying His407, and keeping the top 10 by GlideWS after a consensus
binding-mode filter.

| File | |
|---|---|
| `posed_ligands.sdf` | 43 records for 42 compounds, posed by us in their receptor |
| `schrodinger_predictions.csv` | their 42 published per-ligand predictions, with the experimental MEC |
| `rbfe_values.json` | our predictions for the same 42, network `schro42\|LOMAP k=4\|ddg_sem<=1.0\|pMEC` |


### ML applicability domain — figures 5C and 6C

`data/{li,werner}_et_al/ml_domain.csv` gives, per compound, the distance to the nearest
compound in the training set of the OpenADMET production PXR model
([openadmet/pxr-chemeleon-v1](https://huggingface.co/openadmet/pxr-chemeleon-v1)) —
1 minus the maximum Tanimoto over Morgan fingerprints (radius 2, 2048 bits) — and
whether it was found in that training set at all.

**The training set itself is not redistributed here**; get it from the model repository
above. These distances are the values recorded when the check was run.

The model card states it was trained on 100% of the combined PXR data with no held-out
split, so the benchmark compounds were checked for membership rather than assumed
absent. **Neither series overlaps it**: zero exact matches, and nearest neighbour
0.57–0.71 (Werner) and 0.64–0.71 (Li).

### Scaffold depictions

`data/{li,werner}_et_al/scaffold_2d.png` are panel A of figures 5 and 6, exported from
an interactive scaffold viewer that is not vendored here. They are transparent PNGs;
`scaffold_2d()` in `series_figure.py` crops the trailing "N compounds" footer.

## Caveats worth carrying into any reuse

**The RBFE panels cannot detect an absolute-scale error.** FEP computes relative free
energies, and the absolute placement comes from the experimental data twice over —
cinnabar's MLE takes the experimental values as anchors, and the scoring then shifts
predictions onto the experimental mean. A constant offset in the data, in the
predictions, or between them gives byte-identical statistics. Only the ML panels, which
predict on an absolute scale, constrain it.

**Compare against the right baseline.** The Li experimental spread is narrow (sd 0.38 in
pIC50, MAD 0.28), and predicting the experimental mean for every compound gives MUE 0.30
against RBFE's 1.07. Werner, with roughly twice the spread (sd 0.81), is the only one of
the two where RBFE beats that baseline. Range alone does not capture this: the two sets
span nearly the same 2.5 log units, but Werner's interquartile range is 2.9× wider.

**The ML model always predicts pEC50.** Scored against Werner's pMEC or Li's pIC50 it
carries a constant definitional offset (−1.11 and −0.33 log units) that is not model
error. Panels 5C and 6C remove it and say so on the axis.

**Figure 2 is a model, not a measurement.** Gaussian errors, independent per compound,
a constant attrition rate, and designs cheap enough to discard four for every one made.
Every assumption flatters FEP, so read it as an upper bound.

## Licence

Code and derived data: MIT, see [LICENSE](LICENSE). Upstream experimental data belongs
to the cited publications, to ChEMBL, and to the OpenADMET challenge dataset (CC-BY);
see those sources for their terms.

**One exception.** `data/werner_schrodinger_n42/schrodinger_predictions.csv` reproduces
42 predicted values from the tables of
[doi:10.1021/acs.jctc.6c01007](https://doi.org/10.1021/acs.jctc.6c01007), for comparison;
those values are theirs. Their structure files are not redistributed here at all — take
them from the publisher.
