"""Bemis-Murcko scaffold groups of the PXR challenge set, drawn as separate clusters.

Each group with four or more compounds is drawn as its own small network: one node per
compound, wired into a random spanning tree. The tree is illustrative -- it says "these
belong together", not that any particular pair is connected. Groups of three or fewer
are too numerous to draw, so they appear as schematics in the corners with their counts.

The inset structure is the medoid of the larger half of the biggest group -- among
compounds at or above the 85th percentile for size, the one with the highest mean
similarity to the group as a whole. Its mean Tanimoto to that group is only ~0.16,
which is the point: these compounds share a Murcko scaffold (benzene) and essentially
nothing else.

    python -I make_figure.py
"""

import io
import json
import pathlib

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D

HERE = pathlib.Path(__file__).resolve().parent

RDLogger.DisableLog("rdApp.*")

RESULTS = HERE.parent / "data/challenge_clusters"

#: Groups smaller than this are counted, not drawn.
MIN_DRAWN = 4

#: Cluster radius per compound, and the breathing room between clusters.
RADIUS_SCALE = 0.52
CLUSTER_GAP = 0.15

#: Node marker area, in points^2. Radii scale as sqrt(n), so node density is the same
#: in every cluster and one size is safe for all of them.
NODE_SIZE = 62

#: Schematic glyph geometry, in the same data units as the clusters.
GLYPH_SIDE = 1.7
LABEL_LIFT = 0.55

#: Only compounds at or above this heavy-atom quantile are eligible to be the inset,
#: so the structure shown is a substantial molecule rather than a small fragment.
SIZE_QUANTILE = 0.50

#: Structure inset: render size in pixels, and its scale on the canvas.
MOL_PIXELS = (1000, 950)
MOL_ZOOM = 0.105

#: Where the depiction sits, as a fraction of the canvas (x from left, y from bottom).
#: Lower x moves it left, higher y moves it up.
MOL_ANCHOR = (0.50, 0.12)

#: Arrow endpoints, as canvas fractions. Set either to override the automatic
#: placement, e.g. ARROW_TAIL = (0.46, 0.46), ARROW_HEAD = (0.52, 0.56).
ARROW_TAIL = None
ARROW_HEAD = None

#: Rotation of the depiction, in degrees -- a diagonal molecule packs better into
#: the gap between the central group and the ring below it.
MOL_ROTATION = 45

SEED = 11


def random_tree(n, rng):
    """A random spanning tree on `n` nodes: every node joins one already placed."""
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(1, n):
        g.add_edge(i, int(rng.integers(0, i)))
    return g


def pack(radii, gap=CLUSTER_GAP):
    """Greedy spiral packing: each circle takes the innermost spot that is free."""
    placed = []
    for r in radii:
        if not placed:
            placed.append((0.0, 0.0, r))
            continue
        for t in np.linspace(0.0, 90 * np.pi, 30000):
            x, y = 0.11 * t * np.cos(t), 0.11 * t * np.sin(t)
            if all((x - px) ** 2 + (y - py) ** 2 >= (r + pr + gap) ** 2
                   for px, py, pr in placed):
                placed.append((x, y, r))
                break
    return placed


def cluster_positions(sizes, rng):
    """Node coordinates and edges for every drawn group, packed onto the canvas."""
    radii = [RADIUS_SCALE * np.sqrt(n) for n in sizes]
    centres = pack(radii)
    clusters = []
    for n, (cx, cy, r) in zip(sizes, centres):
        g = random_tree(n, rng)
        pos = nx.kamada_kawai_layout(g)
        p = np.array([pos[i] for i in range(n)])
        p -= p.mean(0)
        # Scale on the furthest node so every cluster really fits the circle the
        # packer reserved for it, and no arm reaches into a neighbour.
        span = float(np.linalg.norm(p, axis=1).max())
        if span > 0:
            p = p / span * r
        p += (cx, cy)
        clusters.append((p, list(g.edges())))
    return clusters


def medoid_of_largest_group(size_quantile=SIZE_QUANTILE):
    """The most typical of the larger compounds in the biggest scaffold group."""
    df = pd.read_csv(RESULTS / "compounds.csv")
    top = df["scaffold"].value_counts().index[0]
    group = df[df["scaffold"] == top].copy()
    mols = [Chem.MolFromSmiles(s) for s in group["canonical_smiles"]]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    group_fps = [gen.GetFingerprint(m) for m in mols]
    group["heavy_atoms"] = [m.GetNumHeavyAtoms() for m in mols]

    eligible = group["heavy_atoms"] >= group["heavy_atoms"].quantile(size_quantile)
    idx = np.flatnonzero(eligible.to_numpy())
    mean_sim = [float(np.mean(DataStructs.BulkTanimotoSimilarity(group_fps[i],
                                                                 group_fps)))
                for i in idx]
    best = idx[int(np.argmax(mean_sim))]
    return (group["canonical_smiles"].iloc[best], max(mean_sim),
            int(group["heavy_atoms"].iloc[best]))


def molecule_image(smiles, size=MOL_PIXELS):
    """High-resolution 2D depiction, atoms coloured by element, transparent ground."""
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 4
    opts.multipleBondOffset = 0.14
    opts.padding = 0.06
    opts.rotate = MOL_ROTATION
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return plt.imread(io.BytesIO(drawer.GetDrawingText()), format="png")


def draw_glyph(ax, kind, x, y, label):
    """One schematic group, with its count sitting directly above it."""
    grey = "0.45"
    h = GLYPH_SIDE * np.sqrt(3) / 2
    shapes = {
        "triangle": (np.array([(-GLYPH_SIDE / 2, h / 2), (GLYPH_SIDE / 2, h / 2),
                               (0.0, -h / 2)]), [(0, 1), (1, 2), (2, 0)]),
        "pair": (np.array([(-GLYPH_SIDE / 2, 0.0), (GLYPH_SIDE / 2, 0.0)]), [(0, 1)]),
        "single": (np.array([(0.0, 0.0)]), []),
    }
    pts, edges = shapes[kind]
    pts = pts + np.array([x, y])
    for a, b in edges:
        ax.plot(*zip(pts[a], pts[b]), color=grey, linewidth=1.1, zorder=3)
    ax.scatter(pts[:, 0], pts[:, 1], s=NODE_SIZE, color=grey, edgecolor="white",
               linewidth=0.7, zorder=4)
    ax.text(x, pts[:, 1].max() + LABEL_LIFT, label, ha="center", va="bottom",
            fontsize=11)


def main(out=str(HERE / "fig3_scaffold_clusters.png")):
    sns.set_theme()
    summary = json.loads((RESULTS / "summary.json").read_text())
    sizes_all = np.sort(
        pd.read_csv(RESULTS / "scaffold_sizes.csv")["n_compounds"].to_numpy())[::-1]
    drawn = [int(s) for s in sizes_all if s >= MIN_DRAWN]
    small = sizes_all[sizes_all < MIN_DRAWN]
    counts = pd.Series(small).value_counts().to_dict()

    rng = np.random.default_rng(SEED)
    clusters = cluster_positions(drawn, rng)
    # One hue per group. Identity is not meant to be looked up -- the colour is only
    # there to say "this tree is a separate group from that one".
    palette = sns.color_palette("husl", len(clusters))

    allp = np.vstack([p for p, _ in clusters])
    pad = 2.4
    xmin, xmax = allp[:, 0].min() - pad, allp[:, 0].max() + pad
    ymin, ymax = allp[:, 1].min() - pad, allp[:, 1].max() + pad

    fig, ax = plt.subplots(figsize=(8.4, 8.4 * (ymax - ymin) / (xmax - xmin)))
    for (pts, edges), colour in zip(clusters, palette):
        for a, b in edges:
            ax.plot(*zip(pts[a], pts[b]), color=colour, linewidth=1.1, alpha=0.75,
                    zorder=1)
        ax.scatter(pts[:, 0], pts[:, 1], s=NODE_SIZE, color=colour,
                   edgecolor="white", linewidth=0.7, zorder=2)

    # Counts go in the corners the packed disc leaves empty, label above glyph.
    inset = 1.5
    draw_glyph(ax, "triangle", xmin + inset, ymax - inset - 1.1,
               f"n = {int(counts.get(3, 0)):,}")
    draw_glyph(ax, "pair", xmax - inset, ymax - inset - 1.1,
               f"n = {int(counts.get(2, 0)):,}")
    draw_glyph(ax, "single", xmin + inset, ymin + inset,
               f"n = {int(counts.get(1, 0)):,}")

    # A real structure from the big central group, in the remaining corner.
    smiles, mean_sim, heavy = medoid_of_largest_group()
    # Just under the big central group, above the ring of smaller ones.
    anchor = (xmin + MOL_ANCHOR[0] * (xmax - xmin),
              ymin + MOL_ANCHOR[1] * (ymax - ymin))
    ax.add_artist(AnnotationBbox(
        OffsetImage(molecule_image(smiles), zoom=MOL_ZOOM), anchor,
        frameon=False, box_alignment=(0.5, 0.5), zorder=6))

    # Start the arrow just off the top edge of the depiction, in data units.
    data_per_point = (xmax - xmin) / (fig.get_figwidth() * 72.0)
    mol_half_height = MOL_PIXELS[1] * MOL_ZOOM / 2 * data_per_point
    mol_half_width = MOL_PIXELS[0] * MOL_ZOOM / 2 * data_per_point
    # Leave from the right flank of the structure, not its top edge -- the top edge
    # sits against the cluster and leaves no room for an arc.
    tail = (anchor[0] + mol_half_width * 0.86, anchor[1] + mol_half_height * 0.25)
    # Aim at a node along the bottom edge of the big group. Which node exactly does
    # not matter -- the tree is illustrative; it only has to read as pointing into
    # that cluster from below.
    centre = clusters[0][0]
    bottom = centre[centre[:, 1] <= np.quantile(centre[:, 1], 0.15)]
    above = bottom[bottom[:, 1] > tail[1] + 0.5]
    pool = above if len(above) else bottom
    target = pool[np.argmin(np.linalg.norm(pool - np.array(tail), axis=1))]

    def to_data(frac):
        return (xmin + frac[0] * (xmax - xmin), ymin + frac[1] * (ymax - ymin))

    def to_frac(pt):
        return ((pt[0] - xmin) / (xmax - xmin), (pt[1] - ymin) / (ymax - ymin))

    if ARROW_TAIL is not None:
        tail = to_data(ARROW_TAIL)
    if ARROW_HEAD is not None:
        target = np.array(to_data(ARROW_HEAD))

    ax.annotate("", xy=tuple(target), xytext=tail,
                arrowprops=dict(arrowstyle="-|>", color="0.25", lw=1.6,
                                connectionstyle="arc3,rad=0.30", shrinkA=2,
                                shrinkB=8),
                zorder=7)

    # What is actually left for RBFE, in the remaining corner.
    typical = summary["stringencies"]["typical"]
    networks = typical["n_components"] - typical["n_isolated"]
    ax.text(0.985, 0.025,
            f"{networks:,} RBFE networks containing\na total of "
            f"{typical['n_compounds_with_a_partner']:,} compounds",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=11,
            linespacing=1.5, multialignment="center",
            bbox=dict(boxstyle="round,pad=0.55", facecolor="white",
                      edgecolor="0.45", linewidth=1.0))

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")
    print(f"inset structure: {smiles}  ({heavy} heavy atoms, mean Tanimoto to "
          f"its group {mean_sim:.2f})")
    print(f"  anchor {MOL_ANCHOR} of canvas = data ({anchor[0]:.2f}, {anchor[1]:.2f}); "
          f"canvas spans x {xmax - xmin:.1f}, y {ymax - ymin:.1f} data units, "
          f"so 0.01 of canvas = {0.01 * (xmax - xmin):.2f} x / "
          f"{0.01 * (ymax - ymin):.2f} y")
    tf, hf = to_frac(tail), to_frac(target)
    print(f"  arrow tail: canvas ({tf[0]:.3f}, {tf[1]:.3f}) = data "
          f"({tail[0]:.2f}, {tail[1]:.2f})")
    print(f"  arrow head: canvas ({hf[0]:.3f}, {hf[1]:.3f}) = data "
          f"({target[0]:.2f}, {target[1]:.2f})   curvature rad = 0.30")
    print(f"drawn:     {len(drawn)} groups, {sum(drawn):,} compounds "
          f"({sum(drawn) / summary['n_compounds']:.1%})")
    print(f"not drawn: {len(small):,} groups, {int(small.sum()):,} compounds "
          f"({small.sum() / summary['n_compounds']:.1%})")


if __name__ == "__main__":
    main()
