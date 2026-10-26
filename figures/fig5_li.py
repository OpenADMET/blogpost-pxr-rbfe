"""Figure 5 -- Li et al. 2021, SPA70 analogues (doi 10.1021/acs.jmedchem.0c02201).

Scored on pIC50, the TR-FRET binding affinity, not the cell-based agonist pEC50
that sits beside it in the paper's tables: binding is what RBFE computes. The 64
rows are the subset of the paper's 81 analogues carrying both values.

    python figures/fig5_li.py
"""
import numpy as np
import pandas as pd
from rdkit import Chem

from common import DATA
from series_figure import build


def _names(d):
    """The paper rows, named as the benchmark builder named them: file order."""
    t = pd.read_csv(d / "paper_activities.csv")
    t["name"] = [f"PXR-Li-{i:03d}" for i in range(1, len(t) + 1)]
    return t


def ml(labels, d):
    t = _names(d)
    pred = pd.read_csv(d / "ml_predictions.csv")
    t["ml"] = t["SMILES"].map(dict(zip(pred["SMILES"],
                                       pred["OADMET_PRED_chemprop-chembl_pEC50"])))
    look = dict(zip(t["name"], t["ml"]))
    return np.asarray([float(look[n]) for n in labels])



CFG = dict(dir="li_et_al", run="benchmark|HIP|pIC50", units="pIC$_{50}$",
           ml_units="pIC$_{50}$",
           series="Li et al., 2021", pdb="PDB 5X0R", out="fig5_li.png",
           ml=ml)

if __name__ == "__main__":
    build("li", CFG)
