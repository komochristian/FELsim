"""Alpha magnet element: closed-form map, lattice wiring, COSY adapter path.

The FELsim element must reproduce the first-order map of the stock COSY
INFINITY element in test/transport/alpha_element.fox:

    x  = -x0 - (s/2)*a0,  a = -a0          s = S_COEFF/sqrt(k), k = g/(B*rho)
    y  =  CC*y0 + UU*s*b0
    b  = (VV/s)*y0 + CC*b0
    R56 = s*(1/gamma^2 - 1/2)              in momentum coordinates

with s_alpha = 257.984 mm at 0.75 MeV / 12 A, symplecticity ~1e-13, and
agreement with an independent lab-frame integration at the 1e-7 level.

The COSY tests run the generated deck and compare COSY's own transfer map to
the element; they skip when no COSY INFINITY binary with a prebuilt COSY.bin
is available.

Author: Eremey Valetov
"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from beamline import driftLattice, alphaMagnetLattice
from cosyAdapter import COSYAdapter
from cosySimulator import COSYSimulator
from latticeLoaderBase import LatticeLoaderBase
from simulatorBase import BeamlineElement
from latticeLoaderBase import TrackedDict

KE_NOMINAL = 0.75    # MeV
I_NOMINAL = 12.0     # A
S_ANCHOR_MM = 257.987  # stock-COSY path length at 0.75 MeV / 12 A
S_ELEMENT_MM = 257.984  # COSY element at the same operating point
S_COEFF_NIELS = 0.19165  # s_alpha = 0.19165*sqrt(beta*gamma/g), beamUtility.py

_TRANSPORT = _BACKEND / "test" / "transport"


def make_alpha(current=I_NOMINAL, energy=KE_NOMINAL, gradient_per_amp=None):
    a = alphaMagnetLattice(current, gradient_per_amp=gradient_per_amp)
    a.setE(energy)
    return a


def _load_check_module():
    """Load the independent ODE arm."""
    path = _TRANSPORT / "alpha_element_check.py"
    if not path.exists():
        pytest.skip(f"{path} not available")
    spec = importlib.util.spec_from_file_location("alpha_element_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_cosy_dir():
    """Directory holding cosy, cosy.fox and a prebuilt COSY.bin, or None."""
    root = _BACKEND.parent
    candidates = [root / "results", _BACKEND / "results",
                  _BACKEND / "test" / "results", Path("/usr/local/bin")]
    for d in candidates:
        if all((d / f).exists() for f in ("cosy", "cosy.fox", "COSY.bin")):
            return d
    return None


def _cosy_to_felsim_r56(element, r56_cosy):
    """COSY (l, dK) R56 in metres to the FELsim (dToF/T_RF, dK/K) convention."""
    return -r56_cosy * element.f * (1 + element.gamma) / (element.beta * element.C * element.gamma)


def _minimal_lattice(elements, format_version=3):
    return {
        "beamline": {
            "metadata": {
                "name": "alpha", "version": "1.0",
                "format_version": format_version,
                "reference_energy_mev": KE_NOMINAL,
                "particle_type": "electron",
                "description": "", "author": "test", "date": "2026-09-02",
            },
            "beam_parameters": {
                "particle": {"type": "electron",
                             "kinetic_energy_mev": KE_NOMINAL,
                             "mass_mev": 0.51099895, "charge_e": -1},
                "rf_frequency_hz": 2856e6,
            },
            "elements": elements,
        }
    }


# ── Closed-form map ──────────────────────────────────────────────────────

class TestClosedFormMap:

    def test_path_length_anchor(self):
        """s_alpha at 0.75 MeV / 12 A matches the stock COSY element."""
        s_mm = make_alpha().length * 1e3
        assert round(s_mm, 3) == S_ELEMENT_MM
        assert abs(s_mm / S_ANCHOR_MM - 1) < 2e-5

    def test_scaling_coefficient(self):
        """S_COEFF reproduces s_alpha = 0.19165*sqrt(beta*gamma/g)."""
        for energy in (0.4307, 0.61, 0.75, 1.0, 1.1):
            for current in (11.0, 12.0, 13.5):
                a = make_alpha(current, energy)
                bg = a.beta * a.gamma
                s_niels = S_COEFF_NIELS * math.sqrt(bg / a.gradient)
                assert abs(a.length / s_niels - 1) < 1e-4

    def test_horizontal_block_is_minus_identity_and_drift(self):
        a = make_alpha()
        M = a._compute_numeric_matrix()
        assert M[0, 0] == -1.0
        assert M[1, 1] == -1.0
        assert M[1, 0] == 0.0
        assert M[0, 1] == pytest.approx(-a.length / 2, rel=1e-15)

    def test_achromatic_to_first_order(self):
        M = make_alpha()._compute_numeric_matrix()
        assert M[0, 5] == 0.0
        assert M[1, 5] == 0.0

    def test_vertical_block_constants(self):
        a = make_alpha()
        M = a._compute_numeric_matrix()
        s = a.length
        assert M[2, 2] == pytest.approx(alphaMagnetLattice.CC, rel=1e-15)
        assert M[3, 3] == pytest.approx(alphaMagnetLattice.CC, rel=1e-15)
        assert M[2, 3] == pytest.approx(alphaMagnetLattice.UU * s, rel=1e-15)
        assert M[3, 2] == pytest.approx(alphaMagnetLattice.VV / s, rel=1e-15)

    def test_symplecticity(self):
        """Both transverse blocks have unit determinant (3.7e-13 on the COSY element)."""
        M = make_alpha()._compute_numeric_matrix()
        det_x = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        det_y = M[2, 2] * M[3, 3] - M[2, 3] * M[3, 2]
        assert abs(det_x - 1.0) < 1e-15
        assert abs(det_y - 1.0) < 1e-12

    def test_r56_is_positive_and_closed_form(self):
        """Higher-energy electrons run a longer path, so they arrive later."""
        a = make_alpha()
        M = a._compute_numeric_matrix()
        r56_momentum = -a.length * (1 / a.gamma ** 2 - 0.5)
        expected = a.f * r56_momentum * a.gamma / ((a.gamma + 1) * a.beta * a.C)
        assert M[4, 5] == pytest.approx(expected, rel=1e-15)
        assert M[4, 5] > 0

    def test_path_length_scaling(self):
        """s ~ sqrt(beta*gamma) and s ~ 1/sqrt(I)."""
        a12 = make_alpha(12.0)
        a13 = make_alpha(13.5)
        assert a13.length / a12.length == pytest.approx(math.sqrt(12.0 / 13.5), rel=1e-14)

        a_low = make_alpha(12.0, 0.61)
        ratio = math.sqrt((a12.beta * a12.gamma) / (a_low.beta * a_low.gamma))
        assert a12.length / a_low.length == pytest.approx(ratio, rel=1e-14)

    def test_length_tracks_energy_changes(self):
        a = alphaMagnetLattice(I_NOMINAL)
        s45 = a.length
        a.setE(KE_NOMINAL)
        assert a.length < s45
        assert a.length == pytest.approx(a.path_length(), rel=1e-15)

    def test_numeric_matches_symbolic(self):
        a = make_alpha()
        num = a._compute_numeric_matrix()
        sym = np.array(a._compute_symbolic_matrix().evalf(), dtype=np.float64)
        np.testing.assert_allclose(sym, num, rtol=1e-12, atol=1e-14)

    def test_zero_current_rejected(self):
        with pytest.raises(ValueError):
            alphaMagnetLattice(0.0)


# ── Independent lab-frame integration ─────────────────────────────────────

class TestIndependentIntegration:

    def test_transverse_block_matches_ode(self):
        """Element vs the independent orbit integration (1.4e-07 on the COSY element)."""
        check = _load_check_module()
        a = make_alpha()
        s_ode, M_ode, _gamma, _k = check.ode_arm(KE_NOMINAL, a.gradient)
        M = a._compute_numeric_matrix()

        assert a.length == pytest.approx(s_ode, rel=1e-9)
        for i, j in ((0, 0), (0, 1), (1, 1), (2, 2), (2, 3), (3, 2), (3, 3)):
            assert M[i, j] == pytest.approx(M_ode[i, j], rel=1e-7)

    def test_r56_matches_time_of_flight(self):
        """R56 against a finite-difference time of flight from the ODE arm."""
        check = _load_check_module()
        a = make_alpha()
        h = 1e-3

        def time_of_flight(energy):
            s, _M, gamma, _k = check.ode_arm(energy, a.gradient)
            beta = math.sqrt(1 - 1 / gamma ** 2)
            return s / (beta * a.C)

        dtof = time_of_flight(KE_NOMINAL * (1 + h)) - time_of_flight(KE_NOMINAL * (1 - h))
        expected = a.f * dtof / (2 * h)
        assert a._compute_numeric_matrix()[4, 5] == pytest.approx(expected, rel=1e-6)


# ── Lattice wiring ───────────────────────────────────────────────────────

class TestLatticeWiring:

    @pytest.mark.parametrize("type_name", ["ALPHA_MAGNET", "AMG"])
    def test_loader_creates_element(self, type_name):
        data = _minimal_lattice([{
            "name": "AM1", "type": type_name,
            "s_start_m": 0.5, "s_end_m": 0.5, "length_m": 0.0,
            "parameters": {"current_a": I_NOMINAL},
        }])
        line = LatticeLoaderBase(TrackedDict(data)).create_beamline()
        assert [type(e).__name__ for e in line] == ["driftLattice", "alphaMagnetLattice"]
        alpha = line[1]
        assert alpha.name == "AM1"
        assert alpha.current == I_NOMINAL
        assert alpha.gradient_per_amp == alphaMagnetLattice.G_PER_AMP

    def test_loader_honours_gradient_calibration(self):
        data = _minimal_lattice([{
            "name": "AM1", "type": "ALPHA_MAGNET",
            "s_start_m": 0.0, "s_end_m": 0.0, "length_m": 0.0,
            "parameters": {"current_a": 13.0, "gradient_t_per_m_per_a": 0.11},
        }])
        alpha = LatticeLoaderBase(TrackedDict(data)).create_beamline()[0]
        assert alpha.gradient == pytest.approx(0.11 * 13.0)

    def test_parse_beamline_dict(self):
        data = _minimal_lattice([{
            "name": "AM1", "type": "ALPHA_MAGNET",
            "s_start_m": 0.0, "s_end_m": 0.0, "length_m": 0.0,
            "parameters": {"current_a": I_NOMINAL},
        }])
        elem = LatticeLoaderBase(TrackedDict(data)).parse_beamline()[0]
        assert elem["type"] == "AMG"
        assert elem["current"] == I_NOMINAL
        assert elem["gradient_per_amp"] == alphaMagnetLattice.G_PER_AMP

    def test_schema_accepts_alpha_magnet(self):
        jsonschema = pytest.importorskip("jsonschema")
        import json
        schema_path = _BACKEND.parent / "var" / "lattice_schema_v3.json"
        with open(schema_path) as f:
            schema = json.load(f)
        data = _minimal_lattice([{
            "name": "AM1", "type": "ALPHA_MAGNET",
            "s_start_m": 0.0, "s_end_m": 0.0, "length_m": 0.0,
            "parameters": {"current_a": I_NOMINAL},
        }])
        jsonschema.validate(data, schema)


# ── COSY adapter path ────────────────────────────────────────────────────

class TestCOSYAdapterPath:

    def test_generic_element_to_cosy_dict(self):
        elem = BeamlineElement("ALPHA_MAGNET", 0.2579836, current=I_NOMINAL,
                               gradient_per_amp=alphaMagnetLattice.G_PER_AMP)
        d = COSYAdapter._beamline_element_to_dict(elem)
        assert d["type"] == "AMG"
        assert d["current"] == I_NOMINAL
        assert d["gradient_per_amp"] == alphaMagnetLattice.G_PER_AMP

    def test_deck_emits_element(self, tmp_path):
        sim = COSYSimulator(excel_path=None,
                            config_dict={"simulation": {"KE": KE_NOMINAL, "order": 2,
                                                        "dimensions": 3}},
                            transfer_matrix_order=1)
        sim.beamline = [{"type": "AMG", "length": 0.2579836, "current": I_NOMINAL,
                         "gradient_per_amp": alphaMagnetLattice.G_PER_AMP}]
        deck = Path(sim.generate_input(output_dir=str(tmp_path))).read_text()

        assert "PROCEDURE ALPHAMAG AMGG ;" in deck
        assert f"ALPHAMAG {alphaMagnetLattice.G_PER_AMP * I_NOMINAL} ;" in deck
        assert deck.index("PROCEDURE ALPHAMAG") < deck.index("PROCEDURE LATTICE")

    @pytest.mark.cosy
    def test_cosy_map_matches_element(self, tmp_path):
        """Run a lattice with an alpha magnet through COSY and compare maps."""
        cosy_dir = _find_cosy_dir()
        if cosy_dir is None:
            pytest.skip("no COSY INFINITY installation with a prebuilt COSY.bin")

        sim = COSYSimulator(excel_path=None,
                            config_dict={"simulation": {"KE": KE_NOMINAL, "order": 2,
                                                        "dimensions": 3}},
                            transfer_matrix_order=1)
        sim.search_dirs.insert(0, str(cosy_dir))
        sim.beamline = [
            {"type": "DRIFT", "length": 0.1},
            {"type": "AMG", "length": 0.2579836, "current": I_NOMINAL,
             "gradient_per_amp": alphaMagnetLattice.G_PER_AMP},
            {"type": "DRIFT", "length": 0.1},
        ]
        sim.run_simulation(output_dir=str(tmp_path))
        M_cosy = sim.analyze_results(str(tmp_path)).read_linear_transfer_map()

        line = [driftLattice(0.1), alphaMagnetLattice(I_NOMINAL), driftLattice(0.1)]
        M = np.eye(6)
        for element in line:
            element.setE(KE_NOMINAL)
            M = element._compute_numeric_matrix() @ M

        np.testing.assert_allclose(M[:4, :4], M_cosy[:4, :4], rtol=1e-6, atol=1e-12)
        assert M[4, 5] == pytest.approx(
            _cosy_to_felsim_r56(line[1], M_cosy[4, 5]), rel=1e-6)
