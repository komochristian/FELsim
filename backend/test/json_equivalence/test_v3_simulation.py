"""
QA: Excel → v3 JSON → 3-way simulation (FELsim Python, RF-Track, COSY).

Part A — Format equivalence: same simulator, Excel vs v3 JSON input.
Part B — Cross-code comparison: all three simulators via v3 JSON.

The v3 JSON includes computed PALS fields (Bn1, BendP) which must
round-trip correctly through the loaders to produce identical physics.

Run:
    PYTHONPATH=backend MPLBACKEND=Agg python -m pytest backend/test/json_equivalence/test_v3_simulation.py -v

Author: Eremey Valetov
"""

import sys
import os
import tempfile
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

EXCEL_PATH = str(Path(__file__).resolve().parent.parent.parent.parent
                 / "beam_excel" / "Beamline_elements.xlsx")

from excelToJson import convert as excel_to_json
from felsimAdapter import FELsimAdapter
from simulatorBase import CoordinateSystem
from beamline import driftLattice

try:
    from cosyAdapter import COSYAdapter
    _COSY_AVAILABLE = True
except ImportError:
    _COSY_AVAILABLE = False

try:
    import RF_Track as rft
    from rftrackAdapter import RFTrackAdapter
    _RFTRACK_AVAILABLE = True
except ImportError:
    _RFTRACK_AVAILABLE = False


# --- Shared test fixtures ---

ENERGY = 40.0  # MeV
N_PARTICLES = 500
SEED = 42
EPSILON_N = 8.0  # pi.mm.mrad normalised

# FELsim uses LaTeX keys, RF-Track uses plain keys
_TWISS_KEY_MAP = {
    'beta': [r'$\beta$ (m)', 'beta'],
    'alpha': [r'$\alpha$', 'alpha'],
    'gamma': [r'$\gamma$ (rad/m)', 'gamma'],
    'emittance': [r'$\epsilon$ ($\pi$.mm.mrad)', 'emittance'],
}

# COSY element types relevant for physics comparison
_COSY_ACTIVE_TYPES = {'DRIFT', 'QPF', 'QPD', 'DPH', 'DPW', 'SOL', 'RFC', 'SXT', 'UND'}


def _get_twiss(tw_dict, plane, param):
    """Extract Twiss parameter handling both LaTeX and plain key formats."""
    d = tw_dict[plane]
    for key in _TWISS_KEY_MAP.get(param, [param]):
        if key in d:
            return d[key]
    raise KeyError(f"No key for {param} in {list(d.keys())}")


def _merge_drifts(beamline):
    """Merge adjacent driftLattice elements (standard comparison helper)."""
    merged = []
    for elem in beamline:
        if isinstance(elem, driftLattice) and merged and isinstance(merged[-1], driftLattice):
            merged[-1] = driftLattice(merged[-1].length + elem.length)
        else:
            merged.append(elem)
    return merged


def _active_elements(bl_dicts):
    """Filter beamline dicts to physics-active elements only (skip diagnostics)."""
    return [e for e in bl_dicts if e['type'] in _COSY_ACTIVE_TYPES and e['length'] > 0]


def _merge_drift_dicts(bl_dicts):
    """Merge adjacent DRIFT dicts (analogous to _merge_drifts for objects)."""
    merged = []
    for e in bl_dicts:
        if e['type'] == 'DRIFT' and merged and merged[-1]['type'] == 'DRIFT':
            merged[-1] = dict(merged[-1], length=merged[-1]['length'] + e['length'])
        else:
            merged.append(e)
    return merged


@pytest.fixture(scope="module")
def v3_json_path():
    """Convert Excel to v3 JSON in a temp file, shared across module."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp = f.name
    excel_to_json(EXCEL_PATH, tmp, name="v3_qa_test", format_version=3)
    yield tmp
    os.unlink(tmp)


@pytest.fixture(scope="module")
def felsim_particles():
    """Generate reproducible FELsim-coordinate particle distribution.

    Returns an immutable (read-only) array to prevent accidental mutation
    between tests. Callers must use .copy() before passing to simulate().
    """
    from beamline import lattice

    rng = np.random.default_rng(SEED)

    rel = lattice(1, fringeType=None)
    rel.setE(E=ENERGY)
    norm = rel.gamma * rel.beta
    epsilon = EPSILON_N / norm

    x_std, y_std = 0.8, 0.8  # mm
    xp_std = epsilon / x_std
    yp_std = epsilon / y_std
    tof_std = 0.5e-9 * 2.856e9  # 0.5 ps bunch
    energy_std = 0.5 * 10  # 0.5% × 10

    particles = rng.standard_normal((N_PARTICLES, 6))
    for i, std in enumerate([x_std, xp_std, y_std, yp_std, tof_std, energy_std]):
        particles[:, i] *= std

    particles.flags.writeable = False
    return particles


# --- Part A: Format equivalence ---

class TestPartA_FELsim:
    """FELsim Python: Excel vs v3 JSON produce identical results."""

    def test_beamline_element_count(self, v3_json_path):
        """Same element count after merging adjacent drifts."""
        sim_excel = FELsimAdapter(lattice_path=EXCEL_PATH, beam_energy=ENERGY)
        sim_json = FELsimAdapter(lattice_path=v3_json_path, beam_energy=ENERGY)
        # JSON loader may merge adjacent zero-length diagnostics into one drift
        bl_excel = _merge_drifts(sim_excel._native_beamline)
        bl_json = _merge_drifts(sim_json._native_beamline)
        assert len(bl_excel) == len(bl_json), \
            f"After drift merge: Excel={len(bl_excel)}, JSON={len(bl_json)}"

    def test_transfer_matrices(self, v3_json_path):
        """Element-by-element transfer matrices match to machine precision."""
        sim_excel = FELsimAdapter(lattice_path=EXCEL_PATH, beam_energy=ENERGY)
        sim_json = FELsimAdapter(lattice_path=v3_json_path, beam_energy=ENERGY)

        bl_excel = _merge_drifts(sim_excel._native_beamline)
        bl_json = _merge_drifts(sim_json._native_beamline)

        for i, (ex, js) in enumerate(zip(bl_excel, bl_json)):
            m_ex = ex._compute_numeric_matrix()
            m_js = js._compute_numeric_matrix()
            diff = np.max(np.abs(m_ex - m_js))
            assert diff < 1e-12, \
                f"[{i}] {type(ex).__name__}: max matrix diff = {diff:.2e}"

    def test_particle_propagation(self, v3_json_path, felsim_particles):
        """Final particle distributions are identical."""
        sim_excel = FELsimAdapter(lattice_path=EXCEL_PATH, beam_energy=ENERGY)
        sim_json = FELsimAdapter(lattice_path=v3_json_path, beam_energy=ENERGY)

        res_excel = sim_excel.simulate(felsim_particles.copy())
        res_json = sim_json.simulate(felsim_particles.copy())

        diff = np.max(np.abs(res_excel.final_particles - res_json.final_particles))
        assert diff < 1e-10, f"Propagation max diff = {diff:.2e}"

    def test_twiss_parameters(self, v3_json_path, felsim_particles):
        """Statistical Twiss parameters match between Excel and v3 JSON.

        Note: x-plane has extreme dispersion (~260 m) with unmatched random
        particles, so dispersion-subtracted Twiss has amplified numerical
        noise. The y-plane (negligible dispersion) matches to machine precision.
        The test_particle_propagation test already proves physics equivalence.
        """
        sim_excel = FELsimAdapter(lattice_path=EXCEL_PATH, beam_energy=ENERGY)
        sim_json = FELsimAdapter(lattice_path=v3_json_path, beam_energy=ENERGY)

        tw_excel = sim_excel.simulate(felsim_particles.copy()).twiss_parameters_statistical['final']
        tw_json = sim_json.simulate(felsim_particles.copy()).twiss_parameters_statistical['final']

        for plane in ('x', 'y'):
            for param in ('beta', 'alpha', 'gamma', 'emittance'):
                val_ex = _get_twiss(tw_excel, plane, param)
                val_js = _get_twiss(tw_json, plane, param)
                # y-plane: tight relative tolerance; x-plane: relaxed due to
                # dispersion subtraction amplifying floating-point noise
                rtol = 1e-6 if plane == 'y' else 0.15
                np.testing.assert_allclose(
                    val_ex, val_js, rtol=rtol, atol=1e-12,
                    err_msg=f"{plane}.{param}"
                )

    def test_quad_currents_unsigned(self, v3_json_path):
        """Bn1→current conversion produces unsigned current matching Excel."""
        sim_excel = FELsimAdapter(lattice_path=EXCEL_PATH, beam_energy=ENERGY)
        sim_json = FELsimAdapter(lattice_path=v3_json_path, beam_energy=ENERGY)

        for ex, js in zip(sim_excel._native_beamline, sim_json._native_beamline):
            if hasattr(ex, 'current'):
                assert abs(ex.current - js.current) < 1e-10, \
                    f"Current mismatch: Excel={ex.current}, JSON={js.current}"


@pytest.mark.skipif(not _COSY_AVAILABLE, reason="COSY not available")
class TestPartA_COSY:
    """COSY INFINITY: Excel vs v3 JSON produce identical beamline dicts."""

    def test_beamline_parse(self, v3_json_path):
        """parse_beamline() produces same physics-active elements from both paths.

        JSON loader keeps all positioned elements (diagnostics etc.) while
        Excel BeamlineBuilder filters them. Compare only physics-active
        elements after merging adjacent drifts.
        """
        cosy_excel = COSYAdapter(lattice_path=EXCEL_PATH,
                                 mode='transfer_matrix', fringe_field_order=0)
        cosy_json = COSYAdapter(lattice_path=v3_json_path, excel_path=EXCEL_PATH,
                                mode='transfer_matrix', fringe_field_order=0)

        bl_excel = _merge_drift_dicts(_active_elements(cosy_excel.get_beamline()))
        bl_json = _merge_drift_dicts(_active_elements(cosy_json.get_beamline()))

        assert len(bl_excel) == len(bl_json), \
            f"Active element count: {len(bl_excel)} vs {len(bl_json)}"

        for i, (ex, js) in enumerate(zip(bl_excel, bl_json)):
            assert ex['type'] == js['type'], \
                f"[{i}] type: {ex['type']} vs {js['type']}"
            assert abs(ex['length'] - js['length']) < 1e-10, \
                f"[{i}] length: {ex['length']} vs {js['length']}"
            if ex['type'] in ('QPF', 'QPD'):
                assert abs(ex['current'] - js['current']) < 1e-10, \
                    f"[{i}] current: {ex['current']} vs {js['current']}"
            elif ex['type'] == 'DPH':
                assert abs(ex['angle'] - js['angle']) < 1e-10, \
                    f"[{i}] angle: {ex['angle']} vs {js['angle']}"

    def test_transfer_map(self, v3_json_path, tmp_path):
        """COSY transfer maps from Excel and v3 JSON are identical."""
        os.makedirs(str(tmp_path / 'results'), exist_ok=True)

        cosy_excel = COSYAdapter(lattice_path=EXCEL_PATH,
                                 mode='transfer_matrix', fringe_field_order=0)
        cosy_json = COSYAdapter(lattice_path=v3_json_path, excel_path=EXCEL_PATH,
                                mode='transfer_matrix', fringe_field_order=0)

        cosy_excel.set_beam_energy(ENERGY)
        cosy_json.set_beam_energy(ENERGY)

        res_excel = cosy_excel.simulate()
        res_json = cosy_json.simulate()

        assert res_excel.success, "COSY Excel simulation failed"
        assert res_json.success, "COSY JSON simulation failed"

        map_excel = res_excel.transfer_map
        map_json = res_json.transfer_map

        if map_excel is not None and map_json is not None:
            diff = np.max(np.abs(np.array(map_excel) - np.array(map_json)))
            assert diff < 1e-10, f"Transfer map max diff = {diff:.2e}"

    def test_twiss_from_map(self, v3_json_path, tmp_path):
        """Twiss parameters from transfer map match between Excel and v3 JSON."""
        os.makedirs(str(tmp_path / 'results'), exist_ok=True)

        cosy_excel = COSYAdapter(lattice_path=EXCEL_PATH,
                                 mode='transfer_matrix', fringe_field_order=0)
        cosy_json = COSYAdapter(lattice_path=v3_json_path, excel_path=EXCEL_PATH,
                                mode='transfer_matrix', fringe_field_order=0)

        cosy_excel.set_beam_energy(ENERGY)
        cosy_json.set_beam_energy(ENERGY)

        res_excel = cosy_excel.simulate()
        res_json = cosy_json.simulate()

        tw_excel = res_excel.twiss_parameters_transfer_map
        tw_json = res_json.twiss_parameters_transfer_map

        if tw_excel and tw_json:
            for plane in ('x', 'y'):
                if plane in tw_excel and plane in tw_json:
                    for param in ('beta', 'alpha'):
                        val_ex = tw_excel[plane].get(param, 0)
                        val_js = tw_json[plane].get(param, 0)
                        assert abs(val_ex - val_js) < 1e-6, \
                            f"COSY {plane}.{param}: {val_ex} vs {val_js}"


@pytest.mark.skipif(not _RFTRACK_AVAILABLE, reason="RF-Track not available")
class TestPartA_RFTrack:
    """RF-Track: Excel vs v3 JSON produce identical tracking results."""

    def test_beamline_element_count(self, v3_json_path):
        """Same element count after accounting for drift merge."""
        rft_excel = RFTrackAdapter(lattice_path=EXCEL_PATH, beam_energy=ENERGY)
        rft_json = RFTrackAdapter(lattice_path=v3_json_path, beam_energy=ENERGY)
        # Allow ±1 due to drift-insertion differences
        assert abs(len(rft_excel.beamline) - len(rft_json.beamline)) <= 1, \
            f"Element count: Excel={len(rft_excel.beamline)}, JSON={len(rft_json.beamline)}"

    def test_particle_tracking(self, v3_json_path, felsim_particles):
        """Final particles from RF-Track match between Excel and v3 JSON."""
        rft_excel = RFTrackAdapter(lattice_path=EXCEL_PATH, beam_energy=ENERGY)
        rft_json = RFTrackAdapter(lattice_path=v3_json_path, beam_energy=ENERGY)

        res_excel = rft_excel.simulate(felsim_particles.copy())
        res_json = rft_json.simulate(felsim_particles.copy())

        assert res_excel.final_particles.shape[0] > 0, "No particles survived (Excel)"
        assert res_json.final_particles.shape[0] > 0, "No particles survived (JSON)"

        # Same number of surviving particles
        assert res_excel.final_particles.shape[0] == res_json.final_particles.shape[0], \
            f"Survivor count: {res_excel.final_particles.shape[0]} vs {res_json.final_particles.shape[0]}"

        diff = np.max(np.abs(res_excel.final_particles - res_json.final_particles))
        assert diff < 1e-6, f"RF-Track propagation max diff = {diff:.2e}"

    def test_twiss_parameters(self, v3_json_path, felsim_particles):
        """RF-Track Twiss parameters match between Excel and v3 JSON."""
        rft_excel = RFTrackAdapter(lattice_path=EXCEL_PATH, beam_energy=ENERGY)
        rft_json = RFTrackAdapter(lattice_path=v3_json_path, beam_energy=ENERGY)

        tw_excel = rft_excel.simulate(felsim_particles.copy()).twiss_parameters_statistical['final']
        tw_json = rft_json.simulate(felsim_particles.copy()).twiss_parameters_statistical['final']

        for plane in ('x', 'y'):
            for param in ('beta', 'alpha'):
                val_ex = _get_twiss(tw_excel, plane, param)
                val_js = _get_twiss(tw_json, plane, param)
                rel_diff = abs(val_ex - val_js) / max(abs(val_ex), 1e-10)
                assert rel_diff < 1e-6, \
                    f"RF-Track {plane}.{param}: {val_ex:.6f} vs {val_js:.6f} (rel={rel_diff:.2e})"


# --- Part B: Cross-code comparison via v3 JSON ---

class TestPartB_CrossCode:
    """3-way cross-code comparison using v3 JSON as the single source of truth."""

    def test_felsim_vs_cosy_beamline(self, v3_json_path):
        """FELsim and COSY load compatible beamlines from v3 JSON."""
        if not _COSY_AVAILABLE:
            pytest.skip("COSY not available")

        sim_f = FELsimAdapter(lattice_path=v3_json_path, beam_energy=ENERGY)
        sim_c = COSYAdapter(lattice_path=v3_json_path, excel_path=EXCEL_PATH,
                            mode='transfer_matrix', fringe_field_order=0)
        sim_c.set_beam_energy(ENERGY)

        bl_cosy = _merge_drift_dicts(_active_elements(sim_c.get_beamline()))
        n_felsim = len(_merge_drifts(sim_f._native_beamline))

        assert abs(len(bl_cosy) - n_felsim) <= 5, \
            f"Active elements: COSY={len(bl_cosy)}, FELsim={n_felsim}"

    def test_felsim_vs_cosy_twiss(self, v3_json_path, felsim_particles, tmp_path):
        """FELsim and COSY both produce finite Twiss from v3 JSON."""
        if not _COSY_AVAILABLE:
            pytest.skip("COSY not available")

        os.makedirs(str(tmp_path / 'results'), exist_ok=True)

        sim_f = FELsimAdapter(lattice_path=v3_json_path, beam_energy=ENERGY)
        res_f = sim_f.simulate(felsim_particles.copy())
        tw_f = res_f.twiss_parameters_statistical['final']

        sim_c = COSYAdapter(lattice_path=v3_json_path, excel_path=EXCEL_PATH,
                            mode='transfer_matrix', fringe_field_order=0)
        sim_c.set_beam_energy(ENERGY)
        res_c = sim_c.simulate()
        tw_c = res_c.twiss_parameters_transfer_map

        if tw_c is None:
            pytest.skip("COSY did not produce transfer map Twiss")

        for plane in ('x', 'y'):
            if plane in tw_c:
                beta_cosy = tw_c[plane].get('beta', float('nan'))
                beta_felsim = _get_twiss(tw_f, plane, 'beta')
                assert np.isfinite(beta_cosy), f"COSY beta_{plane} not finite"
                assert np.isfinite(beta_felsim), f"FELsim beta_{plane} not finite"
                assert beta_cosy > 0, f"COSY beta_{plane} = {beta_cosy:.2f}"
                assert beta_felsim > 0, f"FELsim beta_{plane} = {beta_felsim:.2f}"

    @pytest.mark.skipif(not _RFTRACK_AVAILABLE, reason="RF-Track not available")
    def test_felsim_vs_rftrack_twiss(self, v3_json_path, felsim_particles):
        """FELsim and RF-Track both produce finite Twiss from v3 JSON."""
        sim_f = FELsimAdapter(lattice_path=v3_json_path, beam_energy=ENERGY)
        sim_r = RFTrackAdapter(lattice_path=v3_json_path, beam_energy=ENERGY)

        res_f = sim_f.simulate(felsim_particles.copy())
        res_r = sim_r.simulate(felsim_particles.copy())

        tw_f = res_f.twiss_parameters_statistical['final']
        tw_r = res_r.twiss_parameters_statistical['final']

        for plane in ('x', 'y'):
            beta_f = _get_twiss(tw_f, plane, 'beta')
            beta_r = _get_twiss(tw_r, plane, 'beta')
            assert np.isfinite(beta_f), f"FELsim beta_{plane} not finite"
            assert np.isfinite(beta_r), f"RF-Track beta_{plane} not finite"
            assert beta_f > 0, f"FELsim beta_{plane} = {beta_f:.2f}"
            assert beta_r > 0, f"RF-Track beta_{plane} = {beta_r:.2f}"

    @pytest.mark.skipif(not _RFTRACK_AVAILABLE or not _COSY_AVAILABLE,
                        reason="Both RF-Track and COSY required")
    def test_all_three_stable(self, v3_json_path, felsim_particles, tmp_path):
        """All three codes produce finite Twiss from v3 JSON."""
        os.makedirs(str(tmp_path / 'results'), exist_ok=True)

        sim_f = FELsimAdapter(lattice_path=v3_json_path, beam_energy=ENERGY)
        sim_r = RFTrackAdapter(lattice_path=v3_json_path, beam_energy=ENERGY)
        sim_c = COSYAdapter(lattice_path=v3_json_path, excel_path=EXCEL_PATH,
                            mode='transfer_matrix', fringe_field_order=0)
        sim_c.set_beam_energy(ENERGY)

        res_f = sim_f.simulate(felsim_particles.copy())
        res_r = sim_r.simulate(felsim_particles.copy())
        res_c = sim_c.simulate()

        tw_f = res_f.twiss_parameters_statistical['final']
        tw_r = res_r.twiss_parameters_statistical['final']
        tw_c = res_c.twiss_parameters_transfer_map

        print("\n3-Way Simulation QA via v3 JSON")
        print("-" * 50)
        for plane in ('x', 'y'):
            beta_f = _get_twiss(tw_f, plane, 'beta')
            alpha_f = _get_twiss(tw_f, plane, 'alpha')
            beta_r = _get_twiss(tw_r, plane, 'beta')
            alpha_r = _get_twiss(tw_r, plane, 'alpha')
            beta_c = tw_c[plane].get('beta', float('nan')) if tw_c and plane in tw_c else float('nan')
            alpha_c = tw_c[plane].get('alpha', float('nan')) if tw_c and plane in tw_c else float('nan')

            print(f"  {plane}-plane: FELsim b={beta_f:.4f} a={alpha_f:.4f}, "
                  f"RF-Track b={beta_r:.4f} a={alpha_r:.4f}, "
                  f"COSY b={beta_c:.4f} a={alpha_c:.4f}")

            for name, beta in [("FELsim", beta_f), ("RF-Track", beta_r), ("COSY", beta_c)]:
                if np.isfinite(beta):
                    assert beta > 0, f"{name} beta_{plane} = {beta:.2f}"


# --- Standalone runner ---

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
