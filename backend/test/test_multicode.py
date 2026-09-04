"""Multi-code simulation framework tests.

Validates that MultiCodeSimulator correctly chains simulator sections
and converts beamline elements across adapters.

Author: Eremey Valetov
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _BACKEND.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

JSON_PATH = _PROJECT_ROOT / "var" / "UH_FEL_beamline.json"

from multiCodeSimulator import MultiCodeSimulator, SimSection, _felsim_to_generic
from simulatorBase import CoordinateSystem, SimulationResult, BeamlineElement
from simulatorFactory import SimulatorFactory, CoordinateTransformer

try:
    import RF_Track
    _RFTRACK_AVAILABLE = True
except ImportError:
    _RFTRACK_AVAILABLE = False

try:
    from cosyAdapter import COSYAdapter
    _COSY_AVAILABLE = True
except ImportError:
    _COSY_AVAILABLE = False


# ── Helpers ──────────────────────────────────────────────────────────────

def load_beamline(path):
    import latticeLoader
    return latticeLoader.create_beamline(str(path))


def propagate_single_pass(beamline, particles, energy_mev=40.0):
    """Single-pass FELsim tracking through full beamline."""
    state = particles.copy()
    for elem in beamline:
        elem.setE(energy_mev)
        state = np.array(elem.useMatrice(state))
    return state


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def beamline():
    if not JSON_PATH.exists():
        pytest.skip("UH_FEL_beamline.json not found")
    return load_beamline(JSON_PATH)


@pytest.fixture
def particles():
    np.random.seed(42)
    return np.random.randn(100, 6) * [1e-3, 1e-4, 1e-3, 1e-4, 1e-3, 0.005]


# ── Unit tests ───────────────────────────────────────────────────────────

class TestSimSection:
    """SimSection dataclass basics."""

    def test_creation(self):
        s = SimSection("prefix", "felsim", (0, 60))
        assert s.name == "prefix"
        assert s.simulator_key == "felsim"
        assert s.element_range == (0, 60)
        assert s.config == {}

    def test_with_config(self):
        s = SimSection("suffix", "cosy", (60, 137), config={"mode": "particle_tracking"})
        assert s.config["mode"] == "particle_tracking"


class TestMultiCodeInit:
    """MultiCodeSimulator initialisation."""

    def test_no_sections_raises(self):
        mc = MultiCodeSimulator(sections=[])
        with pytest.raises(ValueError, match="No sections"):
            mc.simulate(np.zeros((10, 6)))

    def test_no_particles_raises(self):
        mc = MultiCodeSimulator(sections=[
            SimSection("a", "felsim", (0, 10))
        ])
        with pytest.raises(ValueError, match="particles required"):
            mc.simulate()

    def test_from_config(self):
        config = {
            "beam_energy_mev": 40.0,
            "sections": [
                {"name": "first", "simulator": "felsim", "elements": [0, 60]},
                {"name": "second", "simulator": "felsim", "elements": [60, 137]},
            ]
        }
        mc = MultiCodeSimulator.from_config(config)
        assert len(mc.sections) == 2
        assert mc.beam_energy == 40.0


class TestCoordinateTransformerRoundtrips:
    """Round-trip tests for all 6 pairwise coordinate transforms."""

    @pytest.fixture
    def felsim_particles(self):
        np.random.seed(99)
        return np.random.randn(50, 6) * [1.0, 0.1, 1.0, 0.1, 5.0, 1.0]

    @pytest.mark.parametrize("from_sys,to_sys", [
        (CoordinateSystem.FELSIM, CoordinateSystem.COSY),
        (CoordinateSystem.FELSIM, CoordinateSystem.RFTRACK),
        (CoordinateSystem.COSY, CoordinateSystem.RFTRACK),
    ])
    def test_roundtrip(self, felsim_particles, from_sys, to_sys):
        energy = 45.0

        if from_sys == CoordinateSystem.FELSIM:
            original = felsim_particles
        elif from_sys == CoordinateSystem.COSY:
            original = CoordinateTransformer.transform(
                felsim_particles, CoordinateSystem.FELSIM, CoordinateSystem.COSY, energy
            )
        else:
            original = CoordinateTransformer.transform(
                felsim_particles, CoordinateSystem.FELSIM, from_sys, energy
            )

        intermediate = CoordinateTransformer.transform(original, from_sys, to_sys, energy)
        recovered = CoordinateTransformer.transform(intermediate, to_sys, from_sys, energy)

        np.testing.assert_allclose(recovered, original, atol=1e-10, rtol=1e-10)

    def test_identity_transform(self, felsim_particles):
        result = CoordinateTransformer.transform(
            felsim_particles, CoordinateSystem.FELSIM, CoordinateSystem.FELSIM, 45.0
        )
        np.testing.assert_array_equal(result, felsim_particles)


class TestFELsimSplitEquivalence:
    """Core MVP test: splitting FELsim beamline into 2 sections
    via MultiCodeSimulator must produce identical results to single-pass."""

    SPLIT_POINT = 60
    ENERGY = 40.0

    def test_two_section_matches_single_pass(self, beamline, particles):
        """MultiCode(felsim+felsim) == single FELsim pass."""
        n_elem = len(beamline)

        # Single-pass reference
        ref = propagate_single_pass(beamline, particles, self.ENERGY)

        # Multi-code: two FELsim sections
        mc = MultiCodeSimulator(
            sections=[
                SimSection("prefix", "felsim", (0, self.SPLIT_POINT)),
                SimSection("suffix", "felsim", (self.SPLIT_POINT, n_elem)),
            ],
            lattice_path=str(JSON_PATH),
            beam_energy=self.ENERGY,
        )

        result = mc.simulate(particles=particles)
        assert result.success
        assert result.final_particles is not None

        np.testing.assert_allclose(
            result.final_particles, ref, atol=1e-12,
            err_msg="MultiCode 2-section FELsim != single-pass FELsim"
        )

    def test_three_section_matches_single_pass(self, beamline, particles):
        """MultiCode(felsim+felsim+felsim) == single FELsim pass."""
        n_elem = len(beamline)
        sp1, sp2 = 40, 90

        ref = propagate_single_pass(beamline, particles, self.ENERGY)

        mc = MultiCodeSimulator(
            sections=[
                SimSection("sec1", "felsim", (0, sp1)),
                SimSection("sec2", "felsim", (sp1, sp2)),
                SimSection("sec3", "felsim", (sp2, n_elem)),
            ],
            lattice_path=str(JSON_PATH),
            beam_energy=self.ENERGY,
        )

        result = mc.simulate(particles=particles)
        assert result.success
        np.testing.assert_allclose(
            result.final_particles, ref, atol=1e-12,
            err_msg="MultiCode 3-section FELsim != single-pass FELsim"
        )

    def test_metadata_sections(self, beamline, particles):
        """Result metadata should record section info."""
        n_elem = len(beamline)

        mc = MultiCodeSimulator(
            sections=[
                SimSection("prefix", "felsim", (0, self.SPLIT_POINT)),
                SimSection("suffix", "felsim", (self.SPLIT_POINT, n_elem)),
            ],
            lattice_path=str(JSON_PATH),
            beam_energy=self.ENERGY,
        )

        result = mc.simulate(particles=particles)
        assert result.metadata['num_sections'] == 2
        assert len(result.metadata['sections']) == 2
        assert result.metadata['sections'][0]['name'] == 'prefix'
        assert result.metadata['sections'][1]['name'] == 'suffix'

    def test_particle_count_preserved(self, beamline, particles):
        """All particles should survive through the pipeline."""
        n_elem = len(beamline)

        mc = MultiCodeSimulator(
            sections=[
                SimSection("prefix", "felsim", (0, self.SPLIT_POINT)),
                SimSection("suffix", "felsim", (self.SPLIT_POINT, n_elem)),
            ],
            lattice_path=str(JSON_PATH),
            beam_energy=self.ENERGY,
        )

        result = mc.simulate(particles=particles)
        assert result.final_particles.shape == particles.shape


class TestElementConversion:
    """FELsim native → generic BeamlineElement conversion."""

    def test_drift_conversion(self, beamline):
        from beamline import driftLattice
        drifts = [e for e in beamline if isinstance(e, driftLattice)]
        assert len(drifts) > 0
        generic = _felsim_to_generic(drifts[0])
        assert isinstance(generic, BeamlineElement)
        assert generic.element_type == 'DRIFT'
        assert generic.length == drifts[0].length

    def test_quad_conversion(self, beamline):
        from beamline import qpfLattice
        quads = [e for e in beamline if isinstance(e, qpfLattice)]
        assert len(quads) > 0
        generic = _felsim_to_generic(quads[0])
        assert generic.element_type == 'QUAD_F'
        assert generic.parameters['current'] == quads[0].current

    def test_dipole_wedge_conversion(self, beamline):
        from beamline import dipole_wedge
        dpws = [e for e in beamline if isinstance(e, dipole_wedge)]
        assert len(dpws) > 0
        generic = _felsim_to_generic(dpws[0])
        assert generic.element_type == 'DIPOLE_WEDGE'
        for key in ('angle', 'dipole_length', 'dipole_angle', 'pole_gap'):
            assert key in generic.parameters, f"Missing '{key}' in DPW conversion"
            assert generic.parameters[key] == getattr(dpws[0], key)

    def test_all_elements_convertible(self, beamline):
        """Every element in the beamline must convert without error."""
        for i, elem in enumerate(beamline):
            generic = _felsim_to_generic(elem)
            assert isinstance(generic, BeamlineElement), f"Element {i} conversion failed"
            assert generic.length >= 0


class TestFactoryRegistration:
    """MultiCodeSimulator accessible via SimulatorFactory."""

    def test_multicode_in_available(self):
        available = SimulatorFactory.get_available_simulators()
        assert 'multicode' in available

    def test_create_multicode(self):
        mc = SimulatorFactory.create('multicode', sections=[])
        assert isinstance(mc, MultiCodeSimulator)


@pytest.mark.skipif(not _RFTRACK_AVAILABLE, reason="RF-Track not installed")
class TestHybridFELsimRFTrack:
    """FELsim→RF-Track hybrid simulation via MultiCodeSimulator."""

    SPLIT = 87  # Stage 11 boundary
    ENERGY = 40.0

    def test_hybrid_runs_successfully(self, beamline, particles):
        """MultiCode(felsim:0-87 + rftrack:87-end) completes without error."""
        n_elem = len(beamline)

        mc = MultiCodeSimulator(
            sections=[
                SimSection("prefix", "felsim", (0, self.SPLIT)),
                SimSection("stage11", "rftrack", (self.SPLIT, n_elem),
                           config={'beam_energy': self.ENERGY}),
            ],
            lattice_path=str(JSON_PATH),
            beam_energy=self.ENERGY,
        )

        result = mc.simulate(particles=particles)
        assert result.success, f"Hybrid simulation failed: {result.metadata}"
        assert result.final_particles is not None
        assert result.final_particles.shape[1] == 6
        assert result.metadata['num_sections'] == 2

    def test_hybrid_vs_full_rftrack(self, beamline, particles):
        """MultiCode(felsim+rftrack) vs full RF-Track: same physics, different models.

        FELsim uses linear transfer matrices; RF-Track uses analytical
        sector-bend corrections. Results should be qualitatively similar
        (same order of magnitude) but not identical.
        """
        n_elem = len(beamline)

        # Hybrid: FELsim prefix + RF-Track suffix
        mc = MultiCodeSimulator(
            sections=[
                SimSection("prefix", "felsim", (0, self.SPLIT)),
                SimSection("stage11", "rftrack", (self.SPLIT, n_elem),
                           config={'beam_energy': self.ENERGY}),
            ],
            lattice_path=str(JSON_PATH),
            beam_energy=self.ENERGY,
        )
        hybrid_result = mc.simulate(particles=particles)
        assert hybrid_result.success

        # Full RF-Track reference
        from rftrackAdapter import RFTrackAdapter
        rt = RFTrackAdapter(beam_energy=self.ENERGY)
        generic_bl = [_felsim_to_generic(e) for e in beamline]
        rt.set_beamline(generic_bl)
        rt_result = rt.simulate(particles=particles)
        assert rt_result.success

        # Both should produce finite, non-degenerate output
        h = hybrid_result.final_particles
        r = rt_result.final_particles
        assert np.all(np.isfinite(h))
        assert np.all(np.isfinite(r))

        # Transverse coordinates should be same order of magnitude
        # (not a tight match: different dipole models)
        for col in [0, 2]:  # x, y
            h_rms = np.std(h[:, col])
            r_rms = np.std(r[:, col])
            ratio = h_rms / r_rms if r_rms > 0 else float('inf')
            assert 0.1 < ratio < 10, (
                f"Column {col}: hybrid RMS={h_rms:.4g} vs RF-Track RMS={r_rms:.4g}"
            )

    def test_hybrid_element_conversion_preserves_dpw(self, beamline):
        """DPW parameters (pole_gap, dipole_angle) survive conversion for RF-Track."""
        from beamline import dipole_wedge
        dpws = [e for e in beamline if isinstance(e, dipole_wedge)]
        if not dpws:
            pytest.skip("No DPW elements in beamline")

        generic = _felsim_to_generic(dpws[0])
        assert generic.parameters.get('pole_gap', 0) > 0
        assert generic.parameters.get('dipole_angle', 0) != 0

    def test_physical_apertures_config(self, beamline, particles):
        """Per-section config toggles physical apertures on RF-Track."""
        n_elem = len(beamline)

        # With physical apertures: may lose particles
        mc_ap = MultiCodeSimulator(
            sections=[
                SimSection("prefix", "felsim", (0, self.SPLIT)),
                SimSection("stage11", "rftrack", (self.SPLIT, n_elem),
                           config={'physical_apertures': True}),
            ],
            lattice_path=str(JSON_PATH),
            beam_energy=self.ENERGY,
        )
        result_ap = mc_ap.simulate(particles=particles)
        assert result_ap.success

        # Without physical apertures: all particles survive
        mc_no = MultiCodeSimulator(
            sections=[
                SimSection("prefix", "felsim", (0, self.SPLIT)),
                SimSection("stage11", "rftrack", (self.SPLIT, n_elem),
                           config={'physical_apertures': False}),
            ],
            lattice_path=str(JSON_PATH),
            beam_energy=self.ENERGY,
        )
        result_no = mc_no.simulate(particles=particles)
        assert result_no.success

        # Without apertures, all particles survive; with apertures, possibly fewer
        assert result_no.final_particles.shape[0] >= result_ap.final_particles.shape[0]


@pytest.mark.skipif(not _RFTRACK_AVAILABLE, reason="RF-Track not installed")
class TestPerSectionConfig:
    """Per-section config passthrough tests."""

    ENERGY = 40.0

    def test_runtime_keys_not_passed_to_constructor(self):
        """Runtime config keys should not be passed to SimulatorFactory.create()."""
        mc = MultiCodeSimulator(
            sections=[
                SimSection("s1", "rftrack", (0, 10),
                           config={'space_charge': False, 'physical_apertures': True}),
            ],
            lattice_path=str(JSON_PATH),
            beam_energy=self.ENERGY,
        )
        # Should have created an rftrack simulator without error
        # (space_charge/physical_apertures are not constructor kwargs)
        assert len(mc._simulators) == 1

    def test_same_key_different_creation_config_separate_instances(self):
        """Two sections with same key but different creation config → separate instances."""
        mc = MultiCodeSimulator(
            sections=[
                SimSection("s1", "rftrack", (0, 60),
                           config={'G_quad': 5.0}),
                SimSection("s2", "rftrack", (60, 137),
                           config={'G_quad': 10.0}),
            ],
            lattice_path=str(JSON_PATH),
            beam_energy=self.ENERGY,
        )
        assert len(mc._simulators) == 2

    def test_same_key_same_creation_config_shared_instance(self):
        """Two sections with same key and same creation config → shared instance."""
        mc = MultiCodeSimulator(
            sections=[
                SimSection("s1", "rftrack", (0, 60),
                           config={'physical_apertures': True}),
                SimSection("s2", "rftrack", (60, 137),
                           config={'physical_apertures': False}),
            ],
            lattice_path=str(JSON_PATH),
            beam_energy=self.ENERGY,
        )
        # Both have only runtime config, no creation-time differences
        assert len(mc._simulators) == 1


# ── COSY adapter integration tests ──────────────────────────────────────

class TestCOSYAdapterSetBeamline:
    """COSYAdapter.set_beamline() for partial beamline tracking."""

    @pytest.mark.skipif(not _COSY_AVAILABLE, reason="COSY not available")
    def test_set_beamline_from_generic_elements(self):
        """set_beamline accepts generic BeamlineElement objects."""
        adapter = COSYAdapter(mode='transfer_matrix')
        elements = [
            BeamlineElement(element_type='DRIFT', length=0.1),
            BeamlineElement(element_type='QUAD_F', length=0.089,
                            current=2.5),
            BeamlineElement(element_type='DRIFT', length=0.2),
        ]
        adapter.set_beamline(elements)
        bl = adapter.get_native_simulator().beamline
        assert len(bl) == 3
        assert bl[0]['type'] == 'DRIFT'
        assert bl[0]['length'] == 0.1
        assert bl[1]['type'] == 'QPF'
        assert bl[1]['current'] == 2.5
        assert bl[2]['type'] == 'DRIFT'

    @pytest.mark.skipif(not _COSY_AVAILABLE, reason="COSY not available")
    def test_set_beamline_from_dicts(self):
        """set_beamline accepts native COSY dict elements."""
        adapter = COSYAdapter(mode='transfer_matrix')
        elements = [
            {'type': 'DRIFT', 'length': 0.5},
            {'type': 'QPD', 'length': 0.089, 'current': 3.0},
        ]
        adapter.set_beamline(elements)
        bl = adapter.get_native_simulator().beamline
        assert len(bl) == 2
        assert bl[1]['current'] == 3.0

    @pytest.mark.skipif(not _COSY_AVAILABLE, reason="COSY not available")
    def test_set_beamline_dpw_conversion(self):
        """DPW elements preserve pole_gap and dipole parameters."""
        adapter = COSYAdapter(mode='transfer_matrix')
        elements = [
            BeamlineElement(element_type='DIPOLE_WEDGE', length=0.0,
                            angle=8.5, pole_gap=0.014478,
                            dipole_length=0.0889, dipole_angle=17.0),
        ]
        adapter.set_beamline(elements)
        bl = adapter.get_native_simulator().beamline
        assert bl[0]['type'] == 'DPW'
        assert bl[0]['pole_gap'] == 0.014478
        assert bl[0]['dipole_angle'] == 17.0

    @pytest.mark.skipif(not _COSY_AVAILABLE, reason="COSY not available")
    def test_set_beamline_marks_parsed(self):
        """After set_beamline, _beamline_parsed should be True."""
        adapter = COSYAdapter(mode='transfer_matrix')
        assert not adapter._beamline_parsed
        adapter.set_beamline([
            BeamlineElement(element_type='DRIFT', length=1.0),
        ])
        assert adapter._beamline_parsed

    @pytest.mark.skipif(not _COSY_AVAILABLE, reason="COSY not available")
    def test_set_beamline_from_felsim_slice(self, beamline):
        """set_beamline works with elements converted from FELsim native via _felsim_to_generic."""
        adapter = COSYAdapter(mode='transfer_matrix')
        # Take a small slice and convert
        felsim_slice = beamline[:10]
        generic_slice = [_felsim_to_generic(e) for e in felsim_slice]
        adapter.set_beamline(generic_slice)
        bl = adapter.get_native_simulator().beamline
        assert len(bl) == 10

    @pytest.mark.skipif(not _COSY_AVAILABLE, reason="COSY not available")
    def test_transfer_matrix_with_set_beamline(self):
        """Transfer matrix simulation works with a set_beamline partial beamline."""
        adapter = COSYAdapter(mode='transfer_matrix')
        # Simple FODO-like cell
        adapter.set_beamline([
            BeamlineElement(element_type='DRIFT', length=0.5),
            BeamlineElement(element_type='QUAD_F', length=0.089, current=2.0),
            BeamlineElement(element_type='DRIFT', length=1.0),
            BeamlineElement(element_type='QUAD_D', length=0.089, current=2.0),
            BeamlineElement(element_type='DRIFT', length=0.5),
        ])
        result = adapter.simulate()
        assert result.success
        assert result.transfer_map is not None
        M = result.transfer_map
        assert M.shape == (6, 6)
        # Drift-quad-drift system: diagonal elements should be finite
        assert np.all(np.isfinite(M))


@pytest.mark.skipif(not _COSY_AVAILABLE, reason="COSY not available")
class TestHybridFELsimCOSY:
    """FELsim→COSY hybrid simulation via MultiCodeSimulator."""

    SPLIT = 10  # Small prefix for fast test
    ENERGY = 40.0

    def test_cosy_section_via_multicode(self, beamline, particles):
        """MultiCode(felsim:0-10 + cosy:10-20) runs to completion."""
        mc = MultiCodeSimulator(
            sections=[
                SimSection("prefix", "felsim", (0, self.SPLIT)),
                SimSection("suffix", "cosy", (self.SPLIT, 20),
                           config={'mode': 'particle_tracking'}),
            ],
            lattice_path=str(JSON_PATH),
            beam_energy=self.ENERGY,
        )
        result = mc.simulate(particles=particles)
        assert result.success, f"Hybrid FELsim+COSY failed: {result.metadata}"
        assert result.final_particles is not None
        assert result.final_particles.shape[1] == 6
        assert result.metadata['num_sections'] == 2


@pytest.mark.skipif(not (_COSY_AVAILABLE and _RFTRACK_AVAILABLE),
                    reason="COSY and RF-Track both required")
class TestHybridCOSYRFTrack:
    """COSY→RF-Track hybrid simulation via MultiCodeSimulator."""

    COSY_END = 60
    ENERGY = 40.0

    def test_cosy_rftrack_chain(self, beamline, particles):
        """MultiCode(cosy:0-60 + rftrack:60-end) runs to completion."""
        n_elem = len(beamline)
        mc = MultiCodeSimulator(
            sections=[
                SimSection("cosy_part", "cosy", (0, self.COSY_END),
                           config={'mode': 'particle_tracking'}),
                SimSection("rftrack_part", "rftrack",
                           (self.COSY_END, n_elem),
                           config={'beam_energy': self.ENERGY}),
            ],
            lattice_path=str(JSON_PATH),
            beam_energy=self.ENERGY,
        )
        result = mc.simulate(particles=particles)
        assert result.success, f"COSY→RF-Track failed: {result.metadata}"
        assert result.final_particles is not None
        assert result.final_particles.shape[1] == 6
        assert result.metadata['num_sections'] == 2

    def test_cosy_rftrack_vs_full_rftrack(self, beamline, particles):
        """COSY→RF-Track hybrid vs full RF-Track: qualitatively similar."""
        n_elem = len(beamline)

        # Hybrid: COSY prefix + RF-Track suffix
        mc = MultiCodeSimulator(
            sections=[
                SimSection("cosy_part", "cosy", (0, self.COSY_END),
                           config={'mode': 'particle_tracking'}),
                SimSection("rftrack_part", "rftrack",
                           (self.COSY_END, n_elem),
                           config={'beam_energy': self.ENERGY}),
            ],
            lattice_path=str(JSON_PATH),
            beam_energy=self.ENERGY,
        )
        hybrid = mc.simulate(particles=particles)
        assert hybrid.success

        # Full RF-Track reference
        from rftrackAdapter import RFTrackAdapter
        rt = RFTrackAdapter(beam_energy=self.ENERGY)
        generic_bl = [_felsim_to_generic(e) for e in beamline]
        rt.set_beamline(generic_bl)
        ref = rt.simulate(particles=particles)
        assert ref.success

        h = hybrid.final_particles
        r = ref.final_particles
        assert np.all(np.isfinite(h))
        assert np.all(np.isfinite(r))

        # Transverse RMS within order of magnitude
        for col in [0, 2]:
            h_rms = np.std(h[:, col])
            r_rms = np.std(r[:, col])
            ratio = h_rms / r_rms if r_rms > 0 else float('inf')
            assert 0.1 < ratio < 10, (
                f"Col {col}: hybrid RMS={h_rms:.4g} vs RF-Track RMS={r_rms:.4g}"
            )


@pytest.mark.skipif(not (_COSY_AVAILABLE and _RFTRACK_AVAILABLE),
                    reason="COSY and RF-Track both required")
class TestThreeWayHybrid:
    """3-way FELsim→COSY→RF-Track hybrid simulation."""

    FELSIM_END = 10
    COSY_END = 60
    ENERGY = 40.0

    def test_felsim_cosy_rftrack_chain(self, beamline, particles):
        """MultiCode(felsim:0-10 + cosy:10-60 + rftrack:60-end) runs to completion."""
        n_elem = len(beamline)
        mc = MultiCodeSimulator(
            sections=[
                SimSection("felsim_part", "felsim", (0, self.FELSIM_END)),
                SimSection("cosy_part", "cosy",
                           (self.FELSIM_END, self.COSY_END),
                           config={'mode': 'particle_tracking'}),
                SimSection("rftrack_part", "rftrack",
                           (self.COSY_END, n_elem),
                           config={'beam_energy': self.ENERGY}),
            ],
            lattice_path=str(JSON_PATH),
            beam_energy=self.ENERGY,
        )
        result = mc.simulate(particles=particles)
        assert result.success, f"3-way hybrid failed: {result.metadata}"
        assert result.final_particles is not None
        assert result.final_particles.shape[1] == 6
        assert result.metadata['num_sections'] == 3

    def test_three_way_metadata(self, beamline, particles):
        """3-way metadata records all three sections correctly."""
        n_elem = len(beamline)
        mc = MultiCodeSimulator(
            sections=[
                SimSection("felsim_part", "felsim", (0, self.FELSIM_END)),
                SimSection("cosy_part", "cosy",
                           (self.FELSIM_END, self.COSY_END),
                           config={'mode': 'particle_tracking'}),
                SimSection("rftrack_part", "rftrack",
                           (self.COSY_END, n_elem),
                           config={'beam_energy': self.ENERGY}),
            ],
            lattice_path=str(JSON_PATH),
            beam_energy=self.ENERGY,
        )
        result = mc.simulate(particles=particles)
        assert result.success
        meta = result.metadata
        assert meta['num_sections'] == 3
        assert meta['sections'][0]['simulator'] == 'felsim'
        assert meta['sections'][1]['simulator'] == 'cosy'
        assert meta['sections'][2]['simulator'] == 'rftrack'
        # Particle count should be preserved or decrease (no creation)
        for s in meta['sections']:
            assert s['num_particles_out'] > 0
            assert s['num_particles_out'] <= s['num_particles_in']

    def test_three_way_vs_full_rftrack(self, beamline, particles):
        """3-way hybrid vs full RF-Track: qualitatively similar."""
        n_elem = len(beamline)

        mc = MultiCodeSimulator(
            sections=[
                SimSection("felsim_part", "felsim", (0, self.FELSIM_END)),
                SimSection("cosy_part", "cosy",
                           (self.FELSIM_END, self.COSY_END),
                           config={'mode': 'particle_tracking'}),
                SimSection("rftrack_part", "rftrack",
                           (self.COSY_END, n_elem),
                           config={'beam_energy': self.ENERGY}),
            ],
            lattice_path=str(JSON_PATH),
            beam_energy=self.ENERGY,
        )
        hybrid = mc.simulate(particles=particles)
        assert hybrid.success

        # Full RF-Track reference
        from rftrackAdapter import RFTrackAdapter
        rt = RFTrackAdapter(beam_energy=self.ENERGY)
        generic_bl = [_felsim_to_generic(e) for e in beamline]
        rt.set_beamline(generic_bl)
        ref = rt.simulate(particles=particles)
        assert ref.success

        h = hybrid.final_particles
        r = ref.final_particles
        assert np.all(np.isfinite(h))
        assert np.all(np.isfinite(r))

        for col in [0, 2]:
            h_rms = np.std(h[:, col])
            r_rms = np.std(r[:, col])
            ratio = h_rms / r_rms if r_rms > 0 else float('inf')
            assert 0.1 < ratio < 10, (
                f"Col {col}: 3-way RMS={h_rms:.4g} vs RF-Track RMS={r_rms:.4g}"
            )
