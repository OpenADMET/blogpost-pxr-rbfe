"""Render the binding-site panels for figures 4, 5, 6 and 7.

One representative ligand per series -- the compound crystallised in that structure,
where there is one -- with its scaffold ball-and-stick and its R groups as plain
sticks. Every receptor but the challenge one is superposed onto the challenge receptor
so all four panels share one camera.

    python -m pymol -cq figures/render_poses.py
"""
import pathlib

import numpy as np
from pymol import cmd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = pathlib.Path("data")
OUT = pathlib.Path("figures")

VIEW = (-0.250618249, 0.512070417, -0.821545184,
        -0.170818254, 0.811921358, 0.558180809,
        0.952879250, 0.280228674, -0.116017684,
        0.000543685, -0.000175030, -52.616828918,
        8.470845222, 5.657642841, -14.073610306,
        37.557807922, 67.706779480, 20.000000000)

HIDDEN, POCKET, HIS = "161-177+234-256", "209+288+299", 407
ALIGN_RESI = "142-434"
SCAFFOLD_C, RGROUP_C = [0.55, 0.27, 0.62], [0.80, 0.70, 0.85]

SYSTEMS = {
    "challenge": dict(dir="challenge_subseries", receptor="receptor_HID407.pdb",
                      chain="B", name="OADMET-0006557", zoom=None, pdb=None,
                      core="O=S(=O)(Nc1ccccc1)c1ccccc1"),
    "li": dict(dir="li_et_al", receptor="receptor_HIP407.pdb", chain="A",
               name="PXR-Li-024", zoom=4.0, pdb="PDB 5X0R",
               core="O=S(=O)(c1ccccc1)c1cn(-c2ccccc2)nn1"),
    "werner": dict(dir="werner_et_al", receptor="receptor_HID407.pdb", chain="B",
                   name="Werner-25", zoom=4.0, pdb="PDB 6HTY",
                   core="O=C(Cc1ccccc1)Nc1ccc(O)cc1"),
    # Figure 7's receptor is Schrodinger's predicted model rather than the crystal, so it
    # is its own system even though the series is the same as "werner". That receptor is
    # THEIRS and is not redistributed here, so this entry is skipped unless you have put
    # the file in place yourself -- see the README. figures/pose_schro.png ships, so
    # figure 7 reproduces either way. Two things that are easy to get wrong if you do
    # rebuild it: the model is a SINGLE chain and that chain is A, not B; and the
    # representative cannot be Werner-25, which the n=42 pose set carries only as the
    # stereo pair Werner-25_R / Werner-25_S. Werner-20 is used instead -- it is the pose
    # parent the rest of the set was rebuilt from (see the `pose_rebuilt_from` tag).
    "schro": dict(dir="werner_schrodinger_n42", receptor="receptor_schro.pdb", chain="A",
                  name="Werner-20", zoom=4.0, pdb="6HTY model 2",
                  core="O=C(Cc1ccccc1)Nc1ccc(O)cc1"),
}
REFERENCE = SYSTEMS["challenge"]


def representative(cfg):
    """Write the chosen ligand to a temp SDF; return its scaffold atom ranks.

    PyMOL's `rank` is the atom's position in the file and survives
    `remove hydrogens`, so these still name the same atoms after the H's go.
    """
    src = ROOT / cfg["dir"] / "posed_ligands.sdf"
    mol = next(m for m in Chem.SDMolSupplier(str(src), removeHs=False)
               if m and m.GetProp("_Name") == cfg["name"])
    match = mol.GetSubstructMatches(Chem.MolFromSmiles(cfg["core"]))
    assert len(match) == 1, f"{cfg['name']}: {len(match)} scaffold matches"
    out = OUT / f"_lig_{cfg['name']}.sdf"
    w = Chem.SDWriter(str(out))
    w.write(mol)
    w.close()
    return out, sorted(match[0])


def label_his(sel, away_from):
    """Place HIS407 on whichever side of the residue the ligand is not."""
    rot = np.array(cmd.get_view()[:9]).reshape(3, 3)
    right, up = rot[:, 0] / np.linalg.norm(rot[:, 0]), rot[:, 1] / np.linalg.norm(rot[:, 1])
    xyz = np.array(cmd.get_coords(f"{sel} and not elem H"))
    d = xyz.mean(0) - np.array(cmd.get_coords(f"{away_from} and not elem H")).mean(0)
    s = np.array([d @ right, d @ up])
    s = s / np.linalg.norm(s) if np.linalg.norm(s) > 1e-6 else np.array([1.0, 0.0])
    direction = s[0] * right + s[1] * up
    cmd.delete("his_label")
    cmd.pseudoatom("his_label", pos=(xyz[np.argmax(xyz @ direction)]
                                     + direction * 2.7).tolist(), label="HIS407")
    cmd.set("label_color", "tv_orange", "his_label")
    cmd.set("label_size", 34, "his_label")
    cmd.set("label_position", (0, 0, 0), "his_label")
    cmd.hide("everything", "his_label")
    cmd.show("label", "his_label")


def render(key, cfg):
    cmd.delete("all")
    cmd.set_color("scaffold_c", SCAFFOLD_C)
    cmd.set_color("rgroup_c", RGROUP_C)

    lig_sdf, ranks = representative(cfg)
    cmd.load(str(ROOT / cfg["dir"] / cfg["receptor"]), "rec")
    cmd.load(str(lig_sdf), "lig")

    if key != "challenge":
        cmd.load(str(ROOT / REFERENCE["dir"] / REFERENCE["receptor"]), "ref")
        rms = cmd.align(f"rec and chain {cfg['chain']} and polymer and "
                        f"resi {ALIGN_RESI} and name CA",
                        f"ref and chain {REFERENCE['chain']} and polymer and "
                        f"resi {ALIGN_RESI} and name CA")
        cmd.matrix_copy("rec", "lig")   # the ligand is its own object
        cmd.delete("ref")
        print(f"  aligned onto challenge: RMSD {rms[0]:.2f} A over {rms[1]} atoms")

    cmd.hide("everything")
    cmd.remove("hydrogens")
    ch = cfg["chain"]
    cmd.show("cartoon", f"rec and chain {ch} and polymer")
    cmd.color("grey80", "rec")
    cmd.set("cartoon_transparency", 0.6, "rec")
    cmd.hide("everything", f"rec and resi {HIDDEN}")

    cmd.select("pocket", f"rec and chain {ch} and resi {POCKET}")
    cmd.show("sticks", "pocket")
    cmd.color("grey50", "pocket and elem C")
    cmd.select("his407", f"rec and chain {ch} and resi {HIS}")
    cmd.show("sticks", "his407")
    cmd.color("tv_orange", "his407 and elem C")

    cmd.select("scaffold", "lig and rank " + "+".join(map(str, ranks)))
    cmd.select("rgroups", "lig and not scaffold")
    assert cmd.count_atoms("scaffold") == len(ranks), "scaffold ranks did not survive"
    cmd.show("sticks", "rgroups")
    cmd.set("stick_radius", 0.11, "lig")
    cmd.color("rgroup_c", "rgroups and elem C")
    cmd.show("sticks", "scaffold")
    cmd.show("spheres", "scaffold")
    cmd.set("sphere_scale", 0.22, "scaffold")
    cmd.color("scaffold_c", "scaffold and elem C")
    for s in ("pocket", "his407", "lig"):
        cmd.util.cnc(f"{s} and not elem C")

    cmd.set_view(VIEW)
    if cfg["zoom"]:
        cmd.zoom("lig or pocket or his407", cfg["zoom"])
    label_his("his407", "lig")

    cmd.set("ray_opaque_background", 0)
    cmd.ray(2000, 2000)
    out = OUT / f"pose_{key}.png"
    cmd.png(str(out), dpi=300)
    lig_sdf.unlink()
    print(f"wrote {out}  ({cfg['name']}, {len(ranks)} scaffold atoms)")


for _k, _c in SYSTEMS.items():
    if not (ROOT / _c["dir"] / _c["receptor"]).exists():
        print(f"skipping {_k}: {_c['dir']}/{_c['receptor']} is not in this repository "
              f"(see README); figures/pose_{_k}.png ships pre-rendered")
        continue
    render(_k, _c)
