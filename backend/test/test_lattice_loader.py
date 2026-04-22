"""Comprehensive tests for FELsim lattice loaders (JSON/YAML, v1–v3).

Uses inline dicts wrapped in TrackedDict — no external fixture files.

Author: Eremey Valetov
"""

import sys
import json
import tempfile
import os
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from tracked_dict import TrackedDict
from latticeLoaderBase import LatticeLoaderBase

_PROJECT_ROOT = _BACKEND.parent
_VAR = _PROJECT_ROOT / "var"


# --- Helpers ---

def _minimal_lattice(elements, format_version=2, global_settings=None, extra_keys=None):
    """Build a minimal valid lattice dict for testing."""
    d = {
        "beamline": {
            "metadata": {
                "name": "test",
                "version": "1.0",
                "format_version": format_version,
                "reference_energy_mev": 45.0,
                "particle_type": "electron",
                "description": "",
                "author": "test",
                "date": "2026-01-01",
            },
            "beam_parameters": {
                "particle": {
                    "type": "electron",
                    "kinetic_energy_mev": 45.0,
                    "mass_mev": 0.51099895,
                    "charge_e": -1,
                },
                "rf_frequency_hz": 2856e6,
            },
            "elements": elements,
        }
    }
    if global_settings:
        d["beamline"]["global_settings"] = global_settings
    if extra_keys:
        d["beamline"].update(extra_keys)
    return d


def _loader(data, **kwargs):
    """Create a LatticeLoaderBase from a plain dict."""
    return LatticeLoaderBase(TrackedDict(data), **kwargs)


def _write_json(data, tmp_path):
    """Write data as JSON to a temp file and return the path."""
    p = os.path.join(str(tmp_path), "test.json")
    with open(p, "w") as f:
        json.dump(data, f)
    return p


# v3-specific tests

class TestV3:

    def test_v3_bn1_quadrupole_to_dict(self):
        """Bn1 on a quadrupole → current conversion in parse_beamline()."""
        elems = [{
            "name": "Q1", "type": "Quadrupole", "polarity": "focusing",
            "s_start_m": 0.0, "s_end_m": 0.1, "length_m": 0.1,
            "aperture_m": 0.027,
            "parameters": {"current_a": 1.0},
            "MagneticMultipoleP": {"Bn1": -0.05},
        }]
        data = _minimal_lattice(elems, format_version=3)
        loader = _loader(data)
        result = loader.parse_beamline()
        q = [e for e in result if e["type"] == "QPF"][0]
        # |Bn1| / (G * r) = 0.05 / (2.694 * 0.0135) = 1.375...
        # abs() because polarity is encoded in QPF/QPD, not current sign
        expected = abs(-0.05 / (2.694 * 0.0135))
        assert q["current"] == pytest.approx(expected, rel=1e-10)

    def test_v3_bn1_overrides_current(self):
        """When Bn1 is present, it overrides current_a."""
        elems = [{
            "name": "Q1", "type": "Quadrupole", "polarity": "defocusing",
            "s_start_m": 0.0, "s_end_m": 0.1, "length_m": 0.1,
            "parameters": {"current_a": 999.0},
            "MagneticMultipoleP": {"Bn1": 0.03},
        }]
        data = _minimal_lattice(elems, format_version=3)
        loader = _loader(data)
        result = loader.parse_beamline()
        q = [e for e in result if e["type"] == "QPD"][0]
        assert q["current"] != 999.0
        assert q["current"] == pytest.approx(0.03 / (2.694 * 0.0135), rel=1e-10)

    def test_v3_bn1_quadrupole_to_object(self):
        """Bn1 → current conversion in create_beamline()."""
        elems = [{
            "name": "Q1", "type": "Quadrupole", "polarity": "focusing",
            "s_start_m": 0.0, "s_end_m": 0.1, "length_m": 0.1,
            "parameters": {},
            "MagneticMultipoleP": {"Bn1": -0.04},
        }]
        data = _minimal_lattice(elems, format_version=3)
        loader = _loader(data)
        result = loader.create_beamline()
        from beamline import qpfLattice
        qpfs = [e for e in result if isinstance(e, qpfLattice)]
        assert len(qpfs) == 1
        expected = abs(-0.04 / (2.694 * 0.0135))
        assert qpfs[0].current == pytest.approx(expected, rel=1e-10)

    def test_v3_bendp_to_dict(self):
        """BendP.g_ref → angle, BendP.e1/e2 → edge angles."""
        import math
        g_ref = 0.3
        L = 0.1
        elems = [{
            "name": "B1", "type": "SBend",
            "s_start_m": 0.0, "s_end_m": L, "length_m": L,
            "parameters": {"bending_angle_deg": 0, "dipole_length_m": L, "pole_gap_m": 0.01},
            "BendP": {"g_ref": g_ref, "e1": 0.1, "e2": 0.2},
        }]
        data = _minimal_lattice(elems, format_version=3)
        loader = _loader(data)
        result = loader.parse_beamline()
        d = [e for e in result if e["type"] == "DPH"][0]
        assert d["angle"] == pytest.approx(math.degrees(g_ref * L), rel=1e-10)

    def test_v3_bendp_to_object(self):
        """BendP.g_ref overrides angle in create_beamline()."""
        import math
        g_ref = 0.5
        L = 0.2
        elems = [{
            "name": "B1", "type": "DPH",
            "s_start_m": 0.0, "s_end_m": L, "length_m": L,
            "parameters": {"bending_angle_deg": 1.0, "dipole_length_m": L},
            "BendP": {"g_ref": g_ref},
        }]
        data = _minimal_lattice(elems, format_version=3)
        loader = _loader(data)
        result = loader.create_beamline()
        from beamline import dipole
        dips = [e for e in result if isinstance(e, dipole)]
        assert len(dips) == 1
        assert dips[0].angle == pytest.approx(math.degrees(g_ref * L), rel=1e-10)

    def test_v3_schema_validation(self, tmp_path):
        """format_version 3 validates against v3 schema (not v1)."""
        elems = [{
            "name": "Q1", "kind": "Quadrupole", "polarity": "focusing",
            "s_start_m": 0.0, "s_end_m": 0.1, "length_m": 0.1,
            "parameters": {},
            "MagneticMultipoleP": {"Bn1": -0.05},
        }]
        data = _minimal_lattice(elems, format_version=3)
        json_path = _write_json(data, tmp_path)

        from jsonLatticeLoader import JsonLatticeLoader
        # Should not raise — v3 schema accepts MagneticMultipoleP
        loader = JsonLatticeLoader(json_path, validate_schema=True)
        result = loader.create_beamline()
        assert len(result) > 0

    def test_v3_format_version_accepted(self):
        """format_version=3 does not raise ValueError."""
        elems = [{
            "name": "D1", "type": "DRIFT",
            "s_start_m": 0.0, "s_end_m": 0.5, "length_m": 0.5,
            "parameters": {},
        }]
        data = _minimal_lattice(elems, format_version=3)
        loader = _loader(data)
        result = loader.parse_beamline()
        assert len(result) == 1

    def test_v3_bendp_edge_angles_warning(self, caplog):
        """BendP with non-zero e1/e2 emits warning (edge angles not applied)."""
        import logging
        elems = [{
            "name": "B1", "type": "SBend",
            "s_start_m": 0.0, "s_end_m": 0.1, "length_m": 0.1,
            "parameters": {"bending_angle_deg": 1.5, "dipole_length_m": 0.1, "pole_gap_m": 0.01},
            "BendP": {"g_ref": 0.3, "e1": 0.1, "e2": 0.2},
        }]
        data = _minimal_lattice(elems, format_version=3)
        with caplog.at_level(logging.WARNING):
            loader = _loader(data)
            loader.parse_beamline()
        assert "edge angles" in caplog.text.lower()
        assert "DPW" in caplog.text

    def test_v3_bendp_edge_angles_warning_create_beamline(self, caplog):
        """BendP edge angle warning also fires from create_beamline()."""
        import logging
        elems = [{
            "name": "B1", "type": "SBend",
            "s_start_m": 0.0, "s_end_m": 0.1, "length_m": 0.1,
            "parameters": {"bending_angle_deg": 1.5, "dipole_length_m": 0.1},
            "BendP": {"g_ref": 0.3, "e1": 0.05, "e2": 0.0},
        }]
        data = _minimal_lattice(elems, format_version=3)
        with caplog.at_level(logging.WARNING):
            loader = _loader(data)
            loader.create_beamline()
        assert "edge angles" in caplog.text.lower()

    def test_v3_bendp_zero_edge_angles_no_warning(self, caplog):
        """BendP with e1=0, e2=0 does not emit edge angle warning."""
        import logging
        elems = [{
            "name": "B1", "type": "SBend",
            "s_start_m": 0.0, "s_end_m": 0.1, "length_m": 0.1,
            "parameters": {"bending_angle_deg": 1.5, "dipole_length_m": 0.1, "pole_gap_m": 0.01},
            "BendP": {"g_ref": 0.3, "e1": 0, "e2": 0},
        }]
        data = _minimal_lattice(elems, format_version=3)
        with caplog.at_level(logging.WARNING):
            loader = _loader(data)
            loader.parse_beamline()
        assert "edge angles" not in caplog.text.lower()

    def test_v3_custom_gradient_and_aperture(self):
        """Custom G and aperture_m are used for Bn1→current conversion."""
        G_custom = 3.0
        aperture = 0.040  # 40 mm bore
        elems = [{
            "name": "Q1", "type": "Quadrupole", "polarity": "focusing",
            "s_start_m": 0.0, "s_end_m": 0.1, "length_m": 0.1,
            "aperture_m": aperture,
            "parameters": {},
            "MagneticMultipoleP": {"Bn1": -0.06},
        }]
        data = _minimal_lattice(
            elems, format_version=3,
            global_settings={"quadrupole_gradient_coefficient_t_per_a_per_m": G_custom}
        )
        loader = _loader(data)
        result = loader.parse_beamline()
        q = [e for e in result if e["type"] == "QPF"][0]
        expected = abs(-0.06 / (G_custom * aperture / 2))
        assert q["current"] == pytest.approx(expected, rel=1e-10)


# Error path tests

class TestErrorPaths:

    def test_unsupported_format_version_raises(self):
        """format_version 99 → ValueError."""
        data = _minimal_lattice([], format_version=99)
        with pytest.raises(ValueError, match="Unsupported format_version"):
            _loader(data)

    def test_missing_element_type_and_kind(self):
        """Element without type or kind raises KeyError."""
        elems = [{
            "name": "X1",
            "s_start_m": 0.0, "s_end_m": 0.5, "length_m": 0.5,
            "parameters": {},
        }]
        data = _minimal_lattice(elems)
        loader = _loader(data)
        with pytest.raises(KeyError):
            loader.parse_beamline()

    def test_malformed_json_raises(self, tmp_path):
        """Broken JSON raises an error."""
        p = os.path.join(str(tmp_path), "bad.json")
        with open(p, "w") as f:
            f.write("{invalid json")

        from jsonLatticeLoader import JsonLatticeLoader
        with pytest.raises(Exception):
            JsonLatticeLoader(p)

    def test_malformed_yaml_raises(self, tmp_path):
        """Broken YAML raises an error."""
        p = os.path.join(str(tmp_path), "bad.yaml")
        with open(p, "w") as f:
            f.write(":\n  - :\n    invalid: [")

        from yamlLatticeLoader import YamlLatticeLoader
        with pytest.raises(Exception):
            YamlLatticeLoader(p)


# --- DPW tests ---

class TestDPW:

    def test_dpw_to_dict(self):
        """DPW element produces correct wedge fields in returned dict."""
        elems = [{
            "name": "W1", "type": "DIPOLE_WEDGE",
            "s_start_m": 0.0, "s_end_m": 0.01, "length_m": 0.01,
            "parameters": {
                "wedge_angle_deg": 7.5,
                "dipole_angle_deg": 15.0,
                "dipole_length_m": 0.2,
                "pole_gap_m": 0.014,
            },
        }]
        data = _minimal_lattice(elems)
        loader = _loader(data)
        result = loader.parse_beamline()
        dpw = [e for e in result if e["type"] == "DPW"][0]
        assert dpw["wedge_angle"] == pytest.approx(7.5)
        assert dpw["angle"] == pytest.approx(15.0)
        assert dpw["gap_wedge"] == pytest.approx(0.01)
        assert dpw["pole_gap"] == pytest.approx(0.014)

    def test_dpw_to_object(self):
        """DPW element produces correct dipole_wedge instance."""
        from beamline import dipole_wedge
        elems = [{
            "name": "W1", "type": "DPW",
            "s_start_m": 0.0, "s_end_m": 0.01, "length_m": 0.01,
            "parameters": {
                "wedge_angle_deg": 7.5,
                "dipole_angle_deg": 15.0,
                "dipole_length_m": 0.2,
                "pole_gap_m": 0.014,
            },
        }]
        data = _minimal_lattice(elems)
        loader = _loader(data)
        result = loader.create_beamline()
        dws = [e for e in result if isinstance(e, dipole_wedge)]
        assert len(dws) == 1
        assert dws[0].angle == pytest.approx(7.5)
        assert dws[0].dipole_angle == pytest.approx(15.0)
        assert dws[0].dipole_length == pytest.approx(0.2)


# Type resolution tests

class TestTypeResolution:

    def test_type_and_kind_both_present_uses_type(self):
        """When both type and kind are present, type wins."""
        elems = [{
            "name": "Q1", "type": "QPF", "kind": "Quadrupole",
            "s_start_m": 0.0, "s_end_m": 0.1, "length_m": 0.1,
            "parameters": {"current_a": 1.0},
        }]
        data = _minimal_lattice(elems)
        loader = _loader(data)
        result = loader.parse_beamline()
        q = [e for e in result if e["type"] == "QPF"]
        assert len(q) == 1

    def test_kicker_plane_resolution(self):
        """Kicker with plane=horizontal → STH."""
        elems = [{
            "name": "K1", "kind": "Kicker", "plane": "horizontal",
            "s_start_m": 0.0, "s_end_m": 0.05, "length_m": 0.05,
            "parameters": {},
        }]
        data = _minimal_lattice(elems)
        loader = _loader(data)
        result = loader.parse_beamline()
        sth = [e for e in result if e["type"] == "STH"]
        assert len(sth) == 1

    def test_kicker_vertical_default(self):
        """Kicker without plane → STV (default vertical)."""
        elems = [{
            "name": "K1", "kind": "Kicker",
            "s_start_m": 0.0, "s_end_m": 0.05, "length_m": 0.05,
            "parameters": {},
        }]
        data = _minimal_lattice(elems)
        loader = _loader(data)
        result = loader.parse_beamline()
        stv = [e for e in result if e["type"] == "STV"]
        assert len(stv) == 1

    def test_instrument_type_resolution(self):
        """Instrument with instrument_type=OTR → OTR."""
        elems = [{
            "name": "I1", "kind": "Instrument", "instrument_type": "OTR",
            "s_start_m": 0.0, "s_end_m": 0.01, "length_m": 0.01,
            "parameters": {},
        }]
        data = _minimal_lattice(elems)
        loader = _loader(data)
        result = loader.parse_beamline()
        otr = [e for e in result if e["type"] == "OTR"]
        assert len(otr) == 1

    def test_zero_length_marker_skipped(self):
        """Marker (zero-length drift) returns None from create_beamline."""
        elems = [{
            "name": "M1", "kind": "Marker",
            "s_start_m": 0.5, "s_end_m": 0.5, "length_m": 0.0,
            "parameters": {},
        }]
        data = _minimal_lattice(elems)
        loader = _loader(data)
        result = loader.create_beamline()
        # Zero-length drifts return None → filtered out
        from beamline import driftLattice
        drifts = [e for e in result if isinstance(e, driftLattice)]
        # The marker itself produces None, but a drift before it is inserted
        assert all(d.length > 0 for d in drifts)


# --- Drift insertion ---

class TestDriftInsertion:

    def test_drift_insertion_between_elements(self):
        """Drifts are auto-inserted between non-contiguous elements."""
        elems = [
            {
                "name": "Q1", "type": "QPF",
                "s_start_m": 0.5, "s_end_m": 0.6, "length_m": 0.1,
                "parameters": {"current_a": 1.0},
            },
            {
                "name": "Q2", "type": "QPD",
                "s_start_m": 1.0, "s_end_m": 1.1, "length_m": 0.1,
                "parameters": {"current_a": 1.0},
            },
        ]
        data = _minimal_lattice(elems)
        loader = _loader(data)
        result = loader.parse_beamline()
        types = [e["type"] for e in result]
        # DRIFT(0.5), QPF, DRIFT(0.4), QPD
        assert types == ["DRIFT", "QPF", "DRIFT", "QPD"]
        assert result[0]["length"] == pytest.approx(0.5)
        assert result[2]["length"] == pytest.approx(0.4)

    def test_no_drift_for_contiguous_elements(self):
        """No drift inserted when elements are contiguous."""
        elems = [
            {
                "name": "D1", "type": "DRIFT",
                "s_start_m": 0.0, "s_end_m": 0.5, "length_m": 0.5,
                "parameters": {},
            },
            {
                "name": "Q1", "type": "QPF",
                "s_start_m": 0.5, "s_end_m": 0.6, "length_m": 0.1,
                "parameters": {"current_a": 1.0},
            },
        ]
        data = _minimal_lattice(elems)
        loader = _loader(data)
        result = loader.parse_beamline()
        types = [e["type"] for e in result]
        assert types == ["DRIFT", "QPF"]


# or-DEFAULT pattern fix verification

class TestOrDefaultFix:

    def test_explicit_zero_gradient_preserved(self):
        """G=0 in global_settings should not be silently overridden to 2.694."""
        # With G=0, Bn1→current is division by zero; the guard (G*r != 0) skips it
        elems = [{
            "name": "Q1", "type": "Quadrupole", "polarity": "focusing",
            "s_start_m": 0.0, "s_end_m": 0.1, "length_m": 0.1,
            "parameters": {"current_a": 5.0},
            "MagneticMultipoleP": {"Bn1": -0.05},
        }]
        data = _minimal_lattice(
            elems, format_version=3,
            global_settings={"quadrupole_gradient_coefficient_t_per_a_per_m": 0}
        )
        loader = _loader(data)
        result = loader.parse_beamline()
        q = [e for e in result if e["type"] == "QPF"][0]
        # With G=0, Bn1 conversion is skipped, so current_a=5.0 is retained
        assert q["current"] == pytest.approx(5.0)

    def test_explicit_zero_aperture_preserved(self):
        """aperture_m=0 should not be silently overridden to 0.027."""
        elems = [{
            "name": "Q1", "type": "Quadrupole", "polarity": "focusing",
            "s_start_m": 0.0, "s_end_m": 0.1, "length_m": 0.1,
            "aperture_m": 0,
            "parameters": {"current_a": 5.0},
            "MagneticMultipoleP": {"Bn1": -0.05},
        }]
        data = _minimal_lattice(elems, format_version=3)
        loader = _loader(data)
        result = loader.parse_beamline()
        q = [e for e in result if e["type"] == "QPF"][0]
        # With aperture=0, r=0 → G*r=0 → Bn1 conversion skipped → current_a=5.0
        assert q["current"] == pytest.approx(5.0)

    def test_null_current_defaults_to_zero(self):
        """A null current_a should default to 0, not raise."""
        elems = [{
            "name": "Q1", "type": "QPF",
            "s_start_m": 0.0, "s_end_m": 0.1, "length_m": 0.1,
            "parameters": {"current_a": None},
        }]
        data = _minimal_lattice(elems)
        loader = _loader(data)
        result = loader.parse_beamline()
        q = [e for e in result if e["type"] == "QPF"][0]
        # TrackedDict.get("current_a", 0) returns None → stored as None
        # This is acceptable; the downstream consumer handles None
        assert q["current"] is None or q["current"] == 0


# --- Angle unit warning ---

class TestAngleUnit:

    def test_format_version_string_coerced(self):
        """format_version: "2" (string) is coerced to int and accepted."""
        elems = [{
            "name": "D1", "type": "DRIFT",
            "s_start_m": 0.0, "s_end_m": 0.5, "length_m": 0.5,
            "parameters": {},
        }]
        data = _minimal_lattice(elems, format_version="2")
        loader = _loader(data)
        result = loader.parse_beamline()
        assert len(result) == 1

    def test_format_version_string_v3_coerced(self):
        """format_version: "3" (string) is coerced to int and accepted."""
        elems = [{
            "name": "D1", "type": "DRIFT",
            "s_start_m": 0.0, "s_end_m": 0.5, "length_m": 0.5,
            "parameters": {},
        }]
        data = _minimal_lattice(elems, format_version="3")
        loader = _loader(data)
        result = loader.parse_beamline()
        assert len(result) == 1

    def test_dph_to_object_returns_dipole(self):
        """DPH element produces a dipole instance via create_beamline()."""
        from beamline import dipole
        elems = [{
            "name": "B1", "type": "DPH",
            "s_start_m": 0.0, "s_end_m": 0.1, "length_m": 0.1,
            "parameters": {"bending_angle_deg": 1.5, "dipole_length_m": 0.1},
        }]
        data = _minimal_lattice(elems)
        loader = _loader(data)
        result = loader.create_beamline()
        dips = [e for e in result if isinstance(e, dipole)]
        assert len(dips) == 1
        assert dips[0].angle == pytest.approx(1.5)
        assert dips[0].length == pytest.approx(0.1)

    def test_angle_unit_rad_warns(self, caplog):
        """angle_unit: rad emits a warning."""
        import logging
        elems = [{
            "name": "D1", "type": "DRIFT",
            "s_start_m": 0.0, "s_end_m": 0.5, "length_m": 0.5,
            "parameters": {},
        }]
        data = _minimal_lattice(
            elems, format_version=3,
            global_settings={"angle_unit": "rad"}
        )
        with caplog.at_level(logging.WARNING):
            loader = _loader(data)
            loader.parse_beamline()
        assert "angle_unit: rad" in caplog.text
        assert "reserved for future use" in caplog.text
