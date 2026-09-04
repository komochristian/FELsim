"""xsuite dipole/edge mapping vs the FELsim native models.

Checks, for every distinct DPW parameter set in the UH FEL lattice and for
both dipole families, that the xtrack elements emitted by XsuiteAdapter
reproduce FELsim's linear maps:

  * DPW -> xt.DipoleEdge(model='linear'): r21 == M[1,0] = tan(eta)/R and
    r43 == M[3,2] = -tan(eta - phi)/R with the triangle fringe phi
    (fint = le/(12 hgap) equivalence).
  * DPH -> xt.Bend (radians, body only): (x,x') block matches FELsim's
    sector matrix; vertical block stays a pure drift (edges deactivated).
"""
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("xtrack")
pytest.importorskip("xpart")

_BACKEND = Path(__file__).resolve().parent.parent
for _p in (str(_BACKEND), str(_BACKEND / "test")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import xtrack as xt
import xpart as xp

from beamline import dipole, dipole_wedge
from simulatorBase import BeamlineElement
from xsuiteAdapter import XsuiteAdapter

E_MEV = 40.0
_ME_MEV = 0.51099895

# (wedge_angle_deg, dipole_angle_deg, dipole_length_m, pole_gap_m): all
# distinct DPW parameter sets appearing in Beamline_elements.xlsx.
DPW_SETS = [
    (0.0, 1.5, 0.0889, 0.014478),
    (0.75, 1.5, 0.0889, 0.014478),
    (1.5, 1.5, 0.0889, 0.014478),
    (2.018, -4.0, 0.04064, 0.014478),
    (2.536, 5.0, 0.04064, 0.014478),
    (0.0, 11.25, 0.037389, 0.0127),
    (11.25, 11.25, 0.037389, 0.0127),
    (11.25, -11.25, 0.037389, 0.0127),
]
DPH_SETS = [(1.5, 0.0889), (-4.0, 0.04064), (5.0, 0.04064),
            (11.25, 0.037389), (-11.25, 0.037389)]


def _xt_linear_map(elements, d=1e-7):
    """6x6 linear map of an xtrack element list in (x, x', y, y', ...) via FD."""
    line = xt.Line(elements=list(elements))
    line.particle_ref = xp.Particles(mass0=_ME_MEV * 1e6, q0=-1,
                                     kinetic_energy0=E_MEV * 1e6)
    line.build_tracker()
    R = np.zeros((4, 4))
    for j, coord in enumerate(("x", "px", "y", "py")):
        pp = line.build_particles(**{coord: +d})
        pm = line.build_particles(**{coord: -d})
        line.track(pp); line.track(pm)
        out_p = [pp.x[0], pp.px[0], pp.y[0], pp.py[0]]
        out_m = [pm.x[0], pm.px[0], pm.y[0], pm.py[0]]
        R[:, j] = (np.array(out_p) - np.array(out_m)) / (2 * d)
    return R


def _adapter_dpw_element(wedge_deg, d_ang, d_len, gap, le=0.010):
    elem = BeamlineElement("DIPOLE_WEDGE", le, angle=wedge_deg,
                           dipole_angle=d_ang, dipole_length=d_len,
                           pole_gap=gap)
    return XsuiteAdapter._dpw_edge(types.SimpleNamespace(), elem)


@pytest.mark.parametrize("wedge_deg,d_ang,d_len,gap", DPW_SETS)
def test_dpw_edge_matches_felsim(wedge_deg, d_ang, d_len, gap):
    fel = dipole_wedge(0.010, angle=wedge_deg, dipole_length=d_len,
                       dipole_angle=d_ang, pole_gap=gap)
    fel.setE(E_MEV)
    M = np.asarray(fel._compute_numeric_matrix())
    R = _xt_linear_map([_adapter_dpw_element(wedge_deg, d_ang, d_len, gap)])
    assert R[1, 0] == pytest.approx(M[1, 0], rel=1e-9, abs=1e-12), \
        f"r21 mismatch: xt {R[1,0]:.9e} vs FELsim {M[1,0]:.9e}"
    assert R[3, 2] == pytest.approx(M[3, 2], rel=1e-9, abs=1e-12), \
        f"r43 mismatch: xt {R[3,2]:.9e} vs FELsim {M[3,2]:.9e}"
    # thin lens: no drift terms
    assert R[0, 1] == pytest.approx(0.0, abs=1e-9)
    assert R[2, 3] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("ang_deg,length", DPH_SETS)
def test_dph_bend_matches_felsim_sector(ang_deg, length):
    fel = dipole(length, ang_deg)
    fel.setE(E_MEV)
    M = np.asarray(fel._compute_numeric_matrix())
    elem = BeamlineElement("DIPOLE", length, angle=ang_deg)
    adapter = XsuiteAdapter.__new__(XsuiteAdapter)
    xel = XsuiteAdapter._xsuite_elements_for(adapter, elem, length, None)
    R = _xt_linear_map(xel)
    for (i, j) in ((0, 0), (0, 1), (1, 0), (1, 1)):
        assert R[i, j] == pytest.approx(M[i, j], rel=2e-6, abs=1e-9), \
            f"R{i+1}{j+1}: xt {R[i,j]:.8e} vs FELsim {M[i,j]:.8e}"
    # vertical block must be a pure drift (edges deactivated, FELsim y-drift)
    assert R[3, 2] == pytest.approx(0.0, abs=1e-10)
    assert R[2, 3] == pytest.approx(length, rel=1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
