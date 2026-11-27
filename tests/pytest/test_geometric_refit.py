"""Mesh.GeometricRefit: element type gate and exception safety."""
import math
import os

import pytest
from netgen.csg import CSGeometry, Pnt, Sphere
from netgen.meshing import NetgenGeometry, Vec

_KEEP = []   # the mesh holds the C++ geometry; Python must keep the subclass alive


def _unit(p):
    r = math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2)
    return [p[0] / r, p[1] / r, p[2] / r]


class UnitSphere(NetgenGeometry):
    """Unit sphere in Python; raises on callback number `fail_at`."""

    def __init__(self, fail_at=None):
        super().__init__()
        self.fail_at = fail_at
        self.calls = 0

    def _tick(self):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("injected projection failure")

    def ProjectPointGI(self, surf, p, gi):
        self._tick()
        p[0], p[1], p[2] = _unit([p[0], p[1], p[2]])
        return True

    def GetNormal(self, surf, p, gi):
        self._tick()
        return Vec(*_unit([p[0], p[1], p[2]]))

    def PointBetween(self, p1, p2, s, surf, gi1, gi2, newp, newgi):
        q = _unit([p1[i] + s * (p2[i] - p1[i]) for i in range(3)])
        for i in range(3):
            newp[i] = q[i]


def _set(mesh, geo):
    _KEEP.append(geo)
    mesh.SetGeometry(geo)
    return geo


def _curved_sphere():
    g = CSGeometry()
    g.Add(Sphere(Pnt(0, 0, 0), 1))
    m = g.GenerateMesh(maxh=0.6)
    _set(m, UnitSphere())
    m.Curve(3)
    return m


def _lines(mesh, tmp_path, name):
    f = os.path.join(str(tmp_path), name)
    mesh.Save(f)
    with open(f) as fh:
        return fh.read().splitlines()


def test_refit_rejects_second_order_elements_before_changing_coefficients(tmp_path):
    g = CSGeometry()
    g.Add(Sphere(Pnt(0, 0, 0), 1))
    m = g.GenerateMesh(maxh=0.6)
    m.SecondOrder()
    m.Curve(3)
    before = _lines(m, tmp_path, "before.vol")
    with pytest.raises(Exception, match="unsupported surface element type"):
        m.GeometricRefit()
    assert _lines(m, tmp_path, "after.vol") == before


def test_refit_exception_leaves_only_original_or_accepted_coefficients(tmp_path):
    original = _lines(_curved_sphere(), tmp_path, "orig.vol")
    full = _curved_sphere()
    geo = _set(full, UnitSphere())
    stats = full.GeometricRefit()
    assert stats["edges_accepted"] > 0 and stats["faces_accepted"] > 0, stats
    accepted = _lines(full, tmp_path, "full.vol")
    # Early, middle and late failures: inside an edge trial, then a face trial.
    for k in (45, 47, geo.calls // 3, 2 * geo.calls // 3):
        m = _curved_sphere()
        _set(m, UnitSphere(fail_at=k))
        with pytest.raises(Exception, match="injected"):
            m.GeometricRefit()
        got = _lines(m, tmp_path, "k%d.vol" % k)
        foreign = [i for i, (a, b, c) in enumerate(zip(got, original, accepted)) if a != b and a != c]
        assert not foreign, "k=%d: %d lines match neither the original nor the accepted value" % (k, len(foreign))
