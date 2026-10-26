"""Figure 6 -- Werner et al. 2019 (doi 10.1021/acs.jmedchem.9b01304).

Scored on PXR pMEC from ChEMBL: a cell-based transactivation minimum effective
concentration (HepG2/DPX2, CYP3A4 promoter, One-Glo luciferase). Censored records
(standard_relation '>') are excluded, which is why n = 35 and not 36.

    python figures/fig6_werner.py
"""
import numpy as np
import pandas as pd

from common import DATA
from series_figure import build

COL = "OADMET_PRED_chemprop-chembl_pEC50"


def _table(d):
    return pd.read_csv(d / "ml_predictions.csv").set_index("ligand_name")


def ml(labels, d):
    t = _table(d)
    return np.asarray([float(t.loc[n, COL]) for n in labels])



CFG = dict(dir="werner_et_al", run="HID|PXR pMEC", units="pMEC",
           ml_units="pMEC",
           series="Werner et al., 2019", pdb="PDB 6HTY", out="fig6_werner.png",
           ml=ml)

if __name__ == "__main__":
    build("werner", CFG)
