"""Checks that the shipped data reproduces the published numbers.

    pytest -q

Does not re-render the PyMOL panels (slow, needs a GL stack); test_poses_present
asserts they are in the repo and test_no_private_paths covers the whole tree.
"""
import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem, RDLogger
from scipy import stats as sps

RDLogger.DisableLog("rdApp.*")

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA, FIGURES = ROOT / "data", ROOT / "figures"
sys.path.insert(0, str(FIGURES))

KCAL_PER_LOG = 1.363
TAUTOMERS = ("HID407", "HIE407", "HIP407")


def scored(expt, pred):
    """The project's scoring: predictions shifted onto the experimental mean."""
    pred = pred - pred.mean() + expt.mean()
    return dict(n=len(expt), mue=float(np.mean(np.abs(pred - expt))),
                rmse=float(np.sqrt(np.mean((pred - expt) ** 2))),
                r=float(sps.pearsonr(pred, expt)[0]),
                tau=float(sps.kendalltau(pred, expt)[0]))


# --- figure 4 ---------------------------------------------------------------
@pytest.mark.parametrize("taut,n,mue,tau", [
    ("HID407", 14, 2.433, -0.121),
    ("HIE407", 23, 1.630, +0.119),
    ("HIP407", 22, 1.211, +0.078),
])
def test_challenge_subseries(taut, n, mue, tau):
    df = pd.read_csv(DATA / "challenge_subseries/dg_predictions.csv")
    sub = df[df[f"DG_{taut} (kcal/mol) (FECS)"].notna()]
    e = -sub["DG (kcal/mol) (EXPT)"].to_numpy() / KCAL_PER_LOG
    p = -sub[f"DG_{taut} (kcal/mol) (FECS)"].to_numpy() / KCAL_PER_LOG
    s = scored(e, p)
    assert s["n"] == n
    assert s["mue"] == pytest.approx(mue, abs=5e-3)
    assert s["tau"] == pytest.approx(tau, abs=5e-3)


def test_challenge_uncertainties_present():
    """The figure 4 whiskers need a per-ligand FEP uncertainty for every point."""
    df = pd.read_csv(DATA / "challenge_subseries/dg_predictions.csv")
    for t in TAUTOMERS:
        sub = df[df[f"DG_{t} (kcal/mol) (FECS)"].notna()]
        u = sub[f"uncertainty_{t} (kcal/mol) (FECS)"]
        assert u.notna().all() and (u > 0).all()


# --- figures 5 and 6 --------------------------------------------------------
@pytest.mark.parametrize("sub,run,n,mue,tau", [
    ("li_et_al", "benchmark|HIP|pIC50", 64, 1.065, +0.491),
    ("werner_et_al", "HID|PXR pMEC", 35, 0.515, +0.537),
])
def test_series_rbfe(sub, run, n, mue, tau):
    blob = json.load(open(DATA / sub / "rbfe_values.json"))[run]
    s = scored(np.asarray(blob["expt"], float), np.asarray(blob["pred"], float))
    assert s["n"] == n
    assert s["mue"] == pytest.approx(mue, abs=5e-3)
    assert s["tau"] == pytest.approx(tau, abs=5e-3)


def test_rbfe_values_only_published_runs():
    """Only the runs that reached a figure are shipped."""
    assert list(json.load(open(DATA / "li_et_al/rbfe_values.json"))) == \
        ["benchmark|HIP|pIC50"]
    assert list(json.load(open(DATA / "werner_et_al/rbfe_values.json"))) == \
        ["HID|PXR pMEC"]


def test_li_p_columns_derive_from_their_own_linear_column():
    """IC50 is the TR-FRET binding assay, EC50 the cell-based agonist assay.

    Guards the endpoint mix-up: pIC50 must come from IC50, not from EC50.
    """
    d = pd.read_csv(DATA / "li_et_al/paper_activities.csv")
    assert np.allclose(d.pIC50, 9 - np.log10(d.IC50))
    assert np.allclose(d.pEC50, 9 - np.log10(d.EC50))
    assert not np.allclose(d.pIC50, 9 - np.log10(d.EC50))


def test_werner_scored_rows_are_uncensored():
    """n = 35 not 36: ChEMBL MEC records with relation '>' are dropped."""
    act = pd.read_csv(DATA / "werner_et_al/chembl_activities.csv")
    assert set(act.standard_type) == {"MEC"}
    assert (act.standard_relation == ">").sum() > 0
    labels = json.load(open(DATA / "werner_et_al/rbfe_values.json"))[
        "HID|PXR pMEC"]["labels"]
    ml = pd.read_csv(DATA / "werner_et_al/ml_predictions.csv").set_index("ligand_name")
    assert not ml.loc[labels, "pxr_mec_censored"].any()


# --- ML applicability domain ------------------------------------------------
@pytest.mark.parametrize("sub,n,lo,hi", [("li_et_al", 64, 0.643, 0.712),
                                         ("werner_et_al", 35, 0.565, 0.714)])
def test_ml_domain_distances(sub, n, lo, hi):
    """The shipped applicability-domain distances, and no training-set overlap.

    The training set is not redistributed, so these are the values recorded when
    the check was run against it rather than a live recomputation.
    """
    labels = json.load(open(DATA / sub / "rbfe_values.json"))
    labels = labels[list(labels)[0]]["labels"]
    t = pd.read_csv(DATA / sub / "ml_domain.csv")
    assert len(t) == n and set(t.label) == set(labels)
    assert not t.in_training_set.any()
    assert t.train_distance.min() == pytest.approx(lo, abs=0.01)
    assert t.train_distance.max() == pytest.approx(hi, abs=0.01)


# --- inputs -----------------------------------------------------------------
@pytest.mark.parametrize("sub,name,core", [
    ("challenge_subseries", "OADMET-0006557", "O=S(=O)(Nc1ccccc1)c1ccccc1"),
    ("li_et_al", "PXR-Li-024", "O=S(=O)(c1ccccc1)c1cn(-c2ccccc2)nn1"),
    ("werner_et_al", "Werner-25", "O=C(Cc1ccccc1)Nc1ccc(O)cc1"),
])
def test_representative_ligand_and_scaffold(sub, name, core):
    """Each pose panel's ligand is present and matches its scaffold exactly once."""
    mols = {m.GetProp("_Name"): m for m in Chem.SDMolSupplier(
        str(DATA / sub / "posed_ligands.sdf"), removeHs=False) if m}
    assert name in mols
    match = mols[name].GetSubstructMatches(Chem.MolFromSmiles(core))
    assert len(match) == 1
    heavy = [a.GetIdx() for a in mols[name].GetAtoms() if a.GetAtomicNum() > 1]
    # PyMOL selects the scaffold by rank, which only survives hydrogen removal
    # while every scaffold atom sits ahead of the first hydrogen.
    assert max(match[0]) <= max(heavy)


@pytest.mark.parametrize("sub,receptor,chain", [
    ("challenge_subseries", "receptor_HID407.pdb", "B"),
    ("li_et_al", "receptor_HIP407.pdb", "A"),
    ("werner_et_al", "receptor_HID407.pdb", "B"),
])
def test_receptor_has_pocket_and_his407(sub, receptor, chain):
    resi = set()
    for ln in open(DATA / sub / receptor):
        if ln.startswith(("ATOM", "HETATM")) and ln[21] == chain:
            resi.add(int(ln[22:26]))
    assert {209, 288, 299, 407} <= resi


def test_ligands_sit_in_their_receptor():
    """Poses must be in their receptor's frame, or every contact is meaningless."""
    from scipy.spatial import cKDTree
    for sub, rec in (("challenge_subseries", "receptor_HID407.pdb"),
                     ("li_et_al", "receptor_HIP407.pdb"),
                     ("werner_et_al", "receptor_HID407.pdb")):
        xyz = np.array([[float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]
                        for ln in open(DATA / sub / rec)
                        if ln.startswith("ATOM") and ln[76:78].strip() != "H"])
        lig = np.vstack([m.GetConformer().GetPositions() for m in
                         Chem.SDMolSupplier(str(DATA / sub / "posed_ligands.sdf"),
                                            removeHs=False) if m])
        d, _ = cKDTree(xyz).query(lig, k=1)
        assert np.median(d) < 8.0, f"{sub}: ligands are not in the receptor frame"


def test_ml_training_set_absent():
    """The ML training set is not redistributed; get it from the model repository."""
    assert not (DATA / "ml_model").exists()


def test_challenge_test_set_present():
    """Figure 3 needs the challenge test set, so it ships -- 513 compounds.

    Guards the columns the figure reads rather than just the file: `scaffold` is what
    groups the clusters and `canonical_smiles` is what the inset is drawn from.
    """
    df = pd.read_csv(DATA / "challenge_clusters/compounds.csv")
    assert len(df) == 513
    assert {"canonical_smiles", "scaffold"} <= set(df.columns)
    summary = json.load(open(DATA / "challenge_clusters/summary.json"))
    assert summary["n_compounds"] == len(df)
    assert df["scaffold"].nunique() == summary["n_scaffolds"]
    assert df["scaffold"].value_counts().iloc[0] == summary["largest_scaffold"]["n"]


def test_poses_present():
    for k in ("challenge", "li", "werner"):
        assert (FIGURES / f"pose_{k}.png").exists()


# --- public-repo hygiene ----------------------------------------------------
def test_no_private_paths():
    """No home directories, usernames, hosts or credentials anywhere in the repo."""
    bad = ("/home/", "/Users/", "jenke", "choderaj", "/data1/", "password",
           "api_key", "secret", "BEGIN RSA", "BEGIN OPENSSH")
    hits = []
    skip = {".git", "__pycache__"}
    for p in ROOT.rglob("*"):
        if (not p.is_file() or skip & set(p.parts) or p.suffix in (".png", ".pyc")
                or p.resolve() == pathlib.Path(__file__).resolve()):
            continue
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        for b in bad:
            if b in text:
                hits.append(f"{p.relative_to(ROOT)}: {b!r}")
    assert not hits, "private strings found:\n" + "\n".join(hits)


def test_figure_scripts_compile():
    for p in sorted(FIGURES.glob("*.py")):
        subprocess.run([sys.executable, "-m", "py_compile", str(p)], check=True)
