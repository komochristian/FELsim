#!/usr/bin/env python3
"""Validate the XsuiteAdapter RF_CAVITY multi-cell TW model + energy-aware k1.

The adapter previously treated RF_CAVITY as a drift; it now builds a multi-cell
travelling-wave chain (Cavity + ReferenceEnergyIncrease) so the beam actually
accelerates, and computes quad k1 at the local (post-acceleration) energy.

Author: Eremey Valetov
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("xtrack")
pytest.importorskip("xpart")

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
REPO = BACKEND.parent

from xsuiteAdapter import XsuiteAdapter
from simulatorBase import BeamlineElement
from physicalConstants import PhysicalConstants

MC2 = PhysicalConstants.E0_electron   # MeV


def test_linac_acceleration():
    """slac_linac.json: the RF_CAVITY should accelerate 1 MeV -> ~41 MeV."""
    sim = XsuiteAdapter(lattice_path=str(REPO / "var" / "slac_linac.json"),
                        beam_energy=1.0)
    line = sim._build_line(sc_on=False)
    p = line.build_particles(x=[0.0], px=[0.0], y=[0.0], py=[0.0],
                             zeta=[0.0], delta=[0.0])
    line.track(p)
    K_out = (p.energy[0] - MC2 * 1e6) / 1e6
    print(f"[linac] adapter RF_CAVITY: 1 MeV -> {K_out:.3f} MeV "
          f"(delta={p.delta[0]:.1e})")
    assert 39.0 < K_out < 42.0, f"unexpected K_out {K_out}"
    assert abs(p.delta[0]) < 1e-6, f"reference not tracking: delta={p.delta[0]}"
    return K_out


def test_energy_aware_k1():
    """A quad after the linac must use the post-acceleration energy (k1 ~ 1/p)."""
    sim = XsuiteAdapter(beam_energy=1.0)
    bg1, bg41 = sim._betagamma_at(1.0), sim._betagamma_at(41.0)
    k1_1 = sim._current_to_k1(5.0, True, betagamma=bg1)
    k1_41 = sim._current_to_k1(5.0, True, betagamma=bg41)
    print(f"[k1] same magnet: k1(1 MeV)={k1_1:.4f}, k1(41 MeV)={k1_41:.4f}, "
          f"ratio {k1_1/k1_41:.1f} (expected {bg41/bg1:.1f})")
    assert abs((k1_1 / k1_41) - (bg41 / bg1)) < 1e-9

    # in a built line, the quad after the cavity must NOT use the 1 MeV value
    cav = BeamlineElement('RF_CAVITY', 3.048, frequency_hz=2856e6,
                          gradient_mv_per_m=13.3, phase_advance_deg=120.0)
    quad = BeamlineElement('QUAD_F', 0.1, current=5.0)
    sim.beamline = [cav, quad]
    line = sim._build_line(sc_on=False)
    k1_used = [e.k1 for e in line.elements if hasattr(e, 'k1') and e.k1 != 0][0]
    print(f"[k1] quad after cavity uses k1={k1_used:.4f} "
          f"(1 MeV would be {k1_1:.4f})")
    assert abs(k1_used) < abs(k1_1), "quad k1 not reduced to post-linac energy"


def test_transport_unchanged():
    """No RF_CAVITY -> behaviour is the original fixed-energy build."""
    sim = XsuiteAdapter(beam_energy=45.0)
    sim.beamline = [BeamlineElement('DRIFT', 0.2),
                    BeamlineElement('QUAD_F', 0.1, current=5.0),
                    BeamlineElement('DRIFT', 0.2)]
    line = sim._build_line(sc_on=False)
    k1 = [e.k1 for e in line.elements if hasattr(e, 'k1') and e.k1 != 0][0]
    bg45 = sim._betagamma_at(45.0)
    k1_ref = sim._current_to_k1(5.0, True, betagamma=bg45)
    print(f"[transport] quad k1={k1:.5f} vs fixed-energy {k1_ref:.5f}")
    assert abs(k1 - k1_ref) < 1e-9


def test_length_and_guard():
    """The TW chain spans the element length exactly; a cavity missing
    frequency_hz falls back to a drift."""
    sim = XsuiteAdapter(lattice_path=str(REPO / "var" / "slac_linac.json"),
                        beam_energy=1.0)
    line = sim._build_line(sc_on=False)
    total_L = sum(getattr(e, 'length', 0.0) for e in line.elements)
    print(f"[length] modeled line = {total_L:.6f} m (cavity nominal 3.048)")
    assert abs(total_L - 3.048) < 1e-9, f"length mismatch {total_L}"

    sim2 = XsuiteAdapter(beam_energy=1.0)
    el, dK, _ = sim2._build_tw_cavity(
        BeamlineElement('RF_CAVITY', 1.0, gradient_mv_per_m=13.3), 1.0)
    print(f"[guard] no-frequency cavity -> {len(el)} elem(s), dK={dK}")
    assert dK == 0.0 and len(el) == 1


if __name__ == "__main__":
    test_linac_acceleration()
    test_energy_aware_k1()
    test_transport_unchanged()
    test_length_and_guard()
    print("\nALL CHECKS PASSED")
