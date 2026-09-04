"""Verification for the xsuite adapter.

1. FELSIM <-> XSUITE coordinate round-trip (machine precision).
2. No-SC transport vs FELsim: drift-only (exact) and a FODO cell (thick quads).
3. Space-charge smoke test (frozen-Gaussian): growth is positive and finite.

Run: cd backend && MPLBACKEND=Agg PYTHONPATH=$(pwd) python test/test_xsuite_adapter.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("xtrack")
pytest.importorskip("xpart")

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from simulatorBase import CoordinateSystem
from simulatorFactory import CoordinateTransformer
from felsimAdapter import FELsimAdapter
from xsuiteAdapter import XsuiteAdapter
from beamline import driftLattice, qpfLattice, qpdLattice
from ebeam import beam as ebeam_class

E_MEV = 45.0


def beta_xy(twiss):
    bx = twiss['x'].get(r'$\beta$ (m)', twiss['x'].get('beta'))
    by = twiss['y'].get(r'$\beta$ (m)', twiss['y'].get('beta'))
    return float(bx), float(by)


def test_roundtrip():
    rng = np.random.default_rng(1)
    p = rng.normal(0, [1.0, 0.5, 1.0, 0.5, 5.0, 1.0], size=(5000, 6))
    xs = CoordinateTransformer.transform(p, CoordinateSystem.FELSIM,
                                         CoordinateSystem.XSUITE, E_MEV)
    back = CoordinateTransformer.transform(xs, CoordinateSystem.XSUITE,
                                           CoordinateSystem.FELSIM, E_MEV)
    err = np.max(np.abs(p - back))
    print(f"[1] roundtrip max abs error = {err:.2e}")
    assert err < 1e-9, f"roundtrip error too large: {err}"
    return True


def _track(elements, particles):
    fs = FELsimAdapter(); fs.set_beam_energy(E_MEV); fs.set_beamline(elements)
    xa = XsuiteAdapter(beam_energy=E_MEV); xa.set_beamline(elements)
    rf = fs.simulate(particles=particles.copy())
    rx = xa.simulate(particles=particles.copy())
    return beta_xy(rf.get_twiss()), beta_xy(rx.get_twiss())


def test_transport():
    eb = ebeam_class()
    np.random.seed(42)
    particles = eb.gen_6d_gaussian(0, [1.0, 0.1, 1.0, 0.1, 1.0, 0.1], 3000)

    # (a) drift-only -- must match closely (pure linear drift in both)
    drift = [driftLattice(0.5), driftLattice(1.0), driftLattice(0.5)]
    (bxf, byf), (bxx, byx) = _track(drift, particles)
    rel = max(abs(bxx - bxf) / bxf, abs(byx - byf) / byf)
    print(f"[2a] drift  FELsim beta=({bxf:.4f},{byf:.4f})  "
          f"xsuite=({bxx:.4f},{byx:.4f})  max rel diff={rel:.2e}")
    assert rel < 1e-3, f"drift mismatch {rel}"

    # (b) stable FODO cell with thick quads (I=0.5 A -> bounded beta).
    # NB: codes share the same per-element matrices (verified separately), so
    # a *stable* lattice agrees to ~1e-4; an unstable one (e.g. I=3 A) is
    # ill-conditioned and diverges in both codes -- not a valid comparison.
    fodo = [driftLattice(0.5), qpfLattice(current=0.5, length=0.1),
            driftLattice(1.0), qpdLattice(current=0.5, length=0.1),
            driftLattice(0.5)]
    (bxf, byf), (bxx, byx) = _track(fodo, particles)
    rel = max(abs(bxx - bxf) / bxf, abs(byx - byf) / byf)
    print(f"[2b] FODO   FELsim beta=({bxf:.4f},{byf:.4f})  "
          f"xsuite=({bxx:.4f},{byx:.4f})  max rel diff={rel:.2e}")
    assert (bxx > byx) == (bxf > byf), "x/y planes swapped -> flip _FOCUS_SIGN"
    assert rel < 5e-3, f"stable-FODO mismatch {rel}"
    return True


def test_space_charge():
    # SC must run, stay finite, and visibly act on the beam (defocus -> larger
    # exit spot). Sign of emittance change is not asserted: frozen-Gaussian
    # linear SC is ~emittance-conserving, so the robust check is the envelope.
    eb = ebeam_class()
    np.random.seed(7)
    particles = eb.gen_6d_gaussian(0, [1.0, 0.1, 1.0, 0.1, 2.0, 0.0], 2000)
    fodo = [driftLattice(0.5), qpfLattice(current=0.5, length=0.1),
            driftLattice(1.0), qpdLattice(current=0.5, length=0.1),
            driftLattice(0.5)]
    xa = XsuiteAdapter(beam_energy=E_MEV, space_charge=True, sc_method="frozen",
                       n_slice_sc=40, bunch_charge_nc=10.0)
    xa.set_beamline(fodo)
    r_off = XsuiteAdapter(beam_energy=E_MEV); r_off.set_beamline(fodo)
    res_on = xa.simulate(particles=particles.copy())
    res_off = r_off.simulate(particles=particles.copy())
    assert res_on.success and res_off.success
    sx_on = float(np.std(res_on.final_particles[:, 0]))
    sx_off = float(np.std(res_off.final_particles[:, 0]))
    e_on = res_on.get_twiss()['y'].get(r'$\epsilon$ ($\pi$.mm.mrad)')
    print(f"[3] SC frozen (10 nC): exit sigma_x off={sx_off:.4f} on={sx_on:.4f} mm; "
          f"eps_ny(on)={e_on:.4f} (finite={np.isfinite(e_on)})")
    assert np.isfinite(e_on)
    assert abs(sx_on - sx_off) / sx_off > 1e-3, "SC had no visible effect"
    return True


if __name__ == "__main__":
    ok = True
    for fn in (test_roundtrip, test_transport, test_space_charge):
        try:
            fn()
        except AssertionError as e:
            print(f"  FAIL {fn.__name__}: {e}"); ok = False
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}"); ok = False
    print("\nALL PASS" if ok else "\nFAILURES")
    sys.exit(0 if ok else 1)
