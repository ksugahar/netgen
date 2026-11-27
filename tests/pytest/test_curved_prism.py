import pytest

ngs = pytest.importorskip("ngsolve")
occ = pytest.importorskip("netgen.occ")
from netgen.meshing import BoundaryLayerParameters


def _curved_prism_mesh(order):
    cyl = occ.Cylinder(occ.Pnt(0, 0, 0), occ.Z, r=0.5, h=1.0)
    cyl.faces.name = "side"
    cyl.faces.Min(occ.Z).name = "bot"
    cyl.faces.Max(occ.Z).name = "top"
    ngmesh = occ.OCCGeometry(cyl).GenerateMesh(
        maxh=0.25,
        boundary_layers=[BoundaryLayerParameters(boundary="side", thickness=[0.05])])
    mesh = ngs.Mesh(ngmesh)
    mesh.Curve(order)
    return mesh


@pytest.mark.parametrize("order", [2, 3])
def test_curved_prisms_agree_on_shared_edges(order):
    """A curved quad face of a prism must not move the edge opposite to it:
    every element sharing an edge maps its points to the same place."""
    mesh = _curved_prism_mesh(order)
    ref = [(1, 0, 0), (0, 1, 0), (0, 0, 0), (1, 0, 1), (0, 1, 1), (0, 0, 1)]
    edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)]
    ts = [0.25, 0.5, 0.75]
    pts = [tuple((1 - t) * ref[a][k] + t * ref[b][k] for k in range(3))
           for a, b in edges for t in ts]
    ir = ngs.IntegrationRule(pts, [0] * len(pts))
    mapped = ngs.CF((ngs.x, ngs.y, ngs.z))(
        mesh.MapToAllElements({ngs.ET.PRISM: ir}, ngs.VOL))
    prisms = [el for el in mesh.Elements(ngs.VOL) if el.type == ngs.ET.PRISM]
    assert prisms and len(mapped) == len(prisms) * len(pts)

    seen = {}
    worst = 0.0
    for e, el in enumerate(prisms):
        v = [p.nr for p in el.vertices]
        for i, (a, b) in enumerate(edges):
            key = (v[a], v[b]) if v[a] < v[b] else (v[b], v[a])
            row = [mapped[e * len(pts) + i * len(ts) + j] for j in range(len(ts))]
            if v[a] > v[b]:
                row = row[::-1]
            if key in seen:
                worst = max(worst, max(sum((p - q) ** 2 for p, q in zip(r, s)) ** 0.5
                                       for r, s in zip(row, seen[key])))
            else:
                seen[key] = row
    assert worst < 1e-12, worst
