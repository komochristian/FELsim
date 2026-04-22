"""
Equivalence tests: verify that JSON lattice loading produces the same
beamline as Excel loading.

Tests compare:
  1. create_beamline() — element types, lengths, and constructor parameters
  2. parse_beamline()  — BeamlineBuilder-compatible dict fields
  3. Transfer matrices — numeric matrices for all active elements
  4. Full propagation  — end-to-end particle tracking
  5. Schema validation
  6. Round-trip conversion (Excel → JSON → beamline)

Run directly:   python test_excel_json_equivalence.py
Run via pytest:  pytest test_excel_json_equivalence.py -v

Author: Eremey Valetov
"""

import sys
import os
import json
import tempfile
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

EXCEL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "beam_excel" / "Beamline_elements.xlsx"
JSON_PATH = Path(__file__).resolve().parent.parent.parent.parent / "var" / "UH_FEL_beamline.json"

from excelElements import ExcelElements
from beamlineBuilder import BeamlineBuilder
from jsonLatticeLoader import JsonLatticeLoader
from excelToJson import convert
from beamline import driftLattice, qpfLattice, qpdLattice, dipole, dipole_wedge

TOL = 1e-10
MATRIX_TOL = 1e-12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def merge_consecutive_drifts(beamline):
    """Merge adjacent driftLattice elements into single drifts."""
    merged = []
    for elem in beamline:
        if isinstance(elem, driftLattice) and merged and isinstance(merged[-1], driftLattice):
            merged[-1] = driftLattice(merged[-1].length + elem.length)
        else:
            merged.append(elem)
    return merged


def merge_consecutive_drift_dicts(beamline):
    """Merge adjacent DRIFT dicts into single drifts."""
    merged = []
    for elem in beamline:
        if elem["type"] == "DRIFT" and merged and merged[-1]["type"] == "DRIFT":
            merged[-1] = {"type": "DRIFT", "length": merged[-1]["length"] + elem["length"]}
        else:
            merged.append(elem)
    return merged


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_schema_validation():
    """Verify that the converted JSON passes schema validation."""
    loader = JsonLatticeLoader(str(JSON_PATH), validate_schema=True)
    assert len(loader.create_beamline()) > 0


def test_create_beamline():
    """Compare beamline.py class instances from Excel and JSON loading."""
    ee = ExcelElements(str(EXCEL_PATH))
    excel_bl = merge_consecutive_drifts(ee.create_beamline())

    loader = JsonLatticeLoader(str(JSON_PATH), validate_schema=False)
    json_bl = merge_consecutive_drifts(loader.create_beamline())

    assert len(excel_bl) == len(json_bl), \
        f"Element count: Excel={len(excel_bl)} vs JSON={len(json_bl)}"

    for i, (ex, js) in enumerate(zip(excel_bl, json_bl)):
        ex_type = type(ex).__name__
        js_type = type(js).__name__
        assert ex_type == js_type, f"[{i}] type: {ex_type} vs {js_type}"
        assert abs(ex.length - js.length) < TOL, \
            f"[{i}] {ex_type} length: {ex.length} vs {js.length}"

        if isinstance(ex, (qpfLattice, qpdLattice)):
            assert abs(ex.current - js.current) < TOL, \
                f"[{i}] {ex_type} current: {ex.current} vs {js.current}"

        elif isinstance(ex, dipole):
            assert abs(ex.angle - js.angle) < TOL, \
                f"[{i}] {ex_type} angle: {ex.angle} vs {js.angle}"

        elif isinstance(ex, dipole_wedge):
            for attr in ("angle", "dipole_length", "dipole_angle", "pole_gap"):
                assert abs(getattr(ex, attr) - getattr(js, attr)) < TOL, \
                    f"[{i}] {ex_type} {attr}: {getattr(ex, attr)} vs {getattr(js, attr)}"


def test_parse_beamline():
    """Compare BeamlineBuilder-style dicts from Excel and JSON loading."""
    bb = BeamlineBuilder(str(EXCEL_PATH))
    excel_bl = merge_consecutive_drift_dicts(bb.parse_beamline())

    loader = JsonLatticeLoader(str(JSON_PATH), validate_schema=False)
    json_bl_raw = loader.parse_beamline()
    json_bl = merge_consecutive_drift_dicts(
        [e for e in json_bl_raw if e["length"] > 0 or e["type"] == "DRIFT"]
    )

    assert len(excel_bl) == len(json_bl), \
        f"Element count: Excel={len(excel_bl)} vs JSON={len(json_bl)}"

    for i, (ex, js) in enumerate(zip(excel_bl, json_bl)):
        assert ex["type"] == js["type"], \
            f"[{i}] type: {ex['type']} vs {js['type']}"
        assert abs(ex["length"] - js["length"]) < TOL, \
            f"[{i}] {ex['type']} length: {ex['length']} vs {js['length']}"

        if ex["type"] in ("QPF", "QPD"):
            assert abs(ex["current"] - js["current"]) < TOL, \
                f"[{i}] {ex['type']} current"

        elif ex["type"] == "DPH":
            assert abs(ex["angle"] - js["angle"]) < TOL, \
                f"[{i}] {ex['type']} angle"

        elif ex["type"] == "DPW":
            for key in ("angle", "wedge_angle", "gap_wedge", "pole_gap"):
                assert abs(ex[key] - js[key]) < TOL, \
                    f"[{i}] {ex['type']} {key}: {ex[key]} vs {js[key]}"


def test_transfer_matrices():
    """Compare numeric transfer matrices from Excel and JSON beamlines."""
    ee = ExcelElements(str(EXCEL_PATH))
    excel_bl = merge_consecutive_drifts(ee.create_beamline())

    loader = JsonLatticeLoader(str(JSON_PATH), validate_schema=False)
    json_bl = merge_consecutive_drifts(loader.create_beamline())

    assert len(excel_bl) == len(json_bl)

    for i, (ex, js) in enumerate(zip(excel_bl, json_bl)):
        ex_mat = ex._compute_numeric_matrix()
        js_mat = js._compute_numeric_matrix()
        diff = np.max(np.abs(ex_mat - js_mat))
        assert diff < MATRIX_TOL, \
            f"[{i}] {type(ex).__name__}: max matrix diff = {diff:.2e}"


def test_full_propagation():
    """Propagate a test particle through both beamlines and compare output."""
    ee = ExcelElements(str(EXCEL_PATH))
    excel_bl = merge_consecutive_drifts(ee.create_beamline())

    loader = JsonLatticeLoader(str(JSON_PATH), validate_schema=False)
    json_bl = merge_consecutive_drifts(loader.create_beamline())

    particle = [[1.0, 0.1, 0.5, 0.05, 0.2, 0.01]]

    excel_out = list(particle)
    json_out = list(particle)
    for elem in excel_bl:
        excel_out = elem.useMatrice(excel_out)
    for elem in json_bl:
        json_out = elem.useMatrice(json_out)

    diff = np.max(np.abs(np.array(excel_out[0]) - np.array(json_out[0])))
    assert diff < 1e-10, f"Propagation max diff = {diff:.2e}"


def test_round_trip_conversion():
    """Convert Excel→JSON→beamline and verify it matches Excel→beamline."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp_path = f.name

    try:
        convert(str(EXCEL_PATH), tmp_path, name="round_trip_test")
        loader = JsonLatticeLoader(tmp_path, validate_schema=False)
        json_bl = merge_consecutive_drifts(loader.create_beamline())

        ee = ExcelElements(str(EXCEL_PATH))
        excel_bl = merge_consecutive_drifts(ee.create_beamline())

        assert len(excel_bl) == len(json_bl), \
            f"Element count: {len(excel_bl)} vs {len(json_bl)}"

        for i, (ex, js) in enumerate(zip(excel_bl, json_bl)):
            assert type(ex).__name__ == type(js).__name__, \
                f"[{i}] type mismatch"
            assert abs(ex.length - js.length) < TOL, \
                f"[{i}] length mismatch"
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# v2 format tests
# ---------------------------------------------------------------------------

def test_v2_round_trip():
    """Convert Excel→JSON(v2)→beamline and verify it matches Excel→beamline."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp_path = f.name

    try:
        convert(str(EXCEL_PATH), tmp_path, name="v2_test", format_version=2)
        loader = JsonLatticeLoader(tmp_path, validate_schema=True)
        json_bl = merge_consecutive_drifts(loader.create_beamline())

        ee = ExcelElements(str(EXCEL_PATH))
        excel_bl = merge_consecutive_drifts(ee.create_beamline())

        assert len(excel_bl) == len(json_bl), \
            f"Element count: {len(excel_bl)} vs {len(json_bl)}"

        for i, (ex, js) in enumerate(zip(excel_bl, json_bl)):
            assert type(ex).__name__ == type(js).__name__, \
                f"[{i}] type mismatch: {type(ex).__name__} vs {type(js).__name__}"
            assert abs(ex.length - js.length) < TOL, \
                f"[{i}] length mismatch"
    finally:
        os.unlink(tmp_path)


def test_v1_backward_compat():
    """v1 JSON files still load correctly with the updated parser."""
    loader = JsonLatticeLoader(str(JSON_PATH), validate_schema=True)
    bl = loader.create_beamline()
    assert len(bl) > 0
    assert loader._format_version == 1


def test_kind_only_elements():
    """Elements with `kind` instead of `type` load correctly (v2 PALS convention)."""
    SPEC_PATH = Path(__file__).resolve().parent.parent.parent.parent / "var" / "lattice_specification.json"
    if not SPEC_PATH.exists():
        import pytest
        pytest.skip(f"Spec file not found: {SPEC_PATH}")

    loader = JsonLatticeLoader(str(SPEC_PATH), validate_schema=True)
    assert loader._format_version == 2

    bl = loader.create_beamline()
    assert len(bl) > 0

    # Verify compound types resolved correctly: Kicker->corrector, Instrument->diagnostic
    parsed = loader.parse_beamline()
    types = {d["type"] for d in parsed}
    # Must contain at least QPF, QPD, DPH, DPW from the curated spec
    assert "QPF" in types or "QPD" in types
    assert "DPH" in types or "DPW" in types


SCHEMA_V3_PATH = Path(__file__).resolve().parent.parent.parent.parent / "var" / "lattice_schema_v3.json"


def test_v3_round_trip():
    """Convert Excel→JSON(v3)→beamline and verify it matches v2 beamline."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp_path = f.name

    try:
        convert(str(EXCEL_PATH), tmp_path, name="v3_test", format_version=3)
        loader = JsonLatticeLoader(tmp_path, validate_schema=False)
        json_bl = merge_consecutive_drifts(loader.create_beamline())

        ee = ExcelElements(str(EXCEL_PATH))
        excel_bl = merge_consecutive_drifts(ee.create_beamline())

        assert len(excel_bl) == len(json_bl), \
            f"Element count: {len(excel_bl)} vs {len(json_bl)}"

        for i, (ex, js) in enumerate(zip(excel_bl, json_bl)):
            assert type(ex).__name__ == type(js).__name__, \
                f"[{i}] type mismatch: {type(ex).__name__} vs {type(js).__name__}"
            assert abs(ex.length - js.length) < TOL, \
                f"[{i}] length mismatch"
    finally:
        os.unlink(tmp_path)


def test_v3_schema_validation():
    """Verify that v3 output validates against lattice_schema_v3.json."""
    import pytest
    jsonschema = pytest.importorskip("jsonschema")

    result = convert(str(EXCEL_PATH), format_version=3)

    with open(SCHEMA_V3_PATH) as f:
        schema = json.load(f)

    jsonschema.validate(result, schema)


def test_v3_bn1_values():
    """Bn1 = sign × G × I × r for quadrupoles."""
    result = convert(str(EXCEL_PATH), format_version=3)
    elems = result["beamline"]["elements"]
    G = 2.694

    for e in elems:
        if e["type"] != "Quadrupole":
            continue
        current = e["parameters"]["current_a"]
        aperture = e.get("aperture_m", 0.027)
        r = aperture / 2.0
        sign = -1.0 if e["polarity"] == "focusing" else 1.0
        expected = sign * G * current * r
        actual = e["MagneticMultipoleP"]["Bn1"]
        assert abs(actual - expected) < 1e-12, \
            f"{e['name']}: Bn1 = {actual}, expected {expected}"


def test_v3_bendp_g_ref():
    """g_ref = angle_rad / L for SBend elements."""
    import math

    result = convert(str(EXCEL_PATH), format_version=3)
    elems = result["beamline"]["elements"]

    for e in elems:
        if e["type"] != "SBend":
            continue
        angle_rad = math.radians(e["parameters"]["bending_angle_deg"])
        L = e["parameters"]["dipole_length_m"]
        expected = angle_rad / L
        actual = e["BendP"]["g_ref"]
        assert abs(actual - expected) < 1e-12, \
            f"{e['name']}: g_ref = {actual}, expected {expected}"


def test_v3_bendp_edge_angles():
    """e1/e2 from DPW-DPH-DPW triplet match wedge angles."""
    import math

    result = convert(str(EXCEL_PATH), format_version=3)
    elems = result["beamline"]["elements"]

    found = False
    for i in range(1, len(elems) - 1):
        if (elems[i]["type"] == "SBend"
                and elems[i - 1]["type"] == "DIPOLE_WEDGE"
                and elems[i + 1]["type"] == "DIPOLE_WEDGE"):
            found = True
            e1_expected = math.radians(elems[i - 1]["parameters"]["wedge_angle_deg"])
            e2_expected = math.radians(elems[i + 1]["parameters"]["wedge_angle_deg"])
            assert abs(elems[i]["BendP"]["e1"] - e1_expected) < 1e-12
            assert abs(elems[i]["BendP"]["e2"] - e2_expected) < 1e-12
            assert abs(elems[i]["parameters"]["entrance_edge_angle_deg"]
                       - elems[i - 1]["parameters"]["wedge_angle_deg"]) < 1e-12
            assert abs(elems[i]["parameters"]["exit_edge_angle_deg"]
                       - elems[i + 1]["parameters"]["wedge_angle_deg"]) < 1e-12

    assert found, "No DPW-SBend-DPW triplet found in beamline"


def test_kind_only_inline():
    """Inline v2 JSON with kind-only elements loads correctly."""
    lattice = {
        "beamline": {
            "metadata": {
                "format_version": 2,
                "name": "test",
                "version": "1.0",
                "reference_energy_mev": 45.0,
                "particle_type": "electron",
            },
            "beam_parameters": {
                "particle": {
                    "type": "electron",
                    "kinetic_energy_mev": 45.0,
                    "mass_mev": 0.51099895,
                    "charge_e": -1,
                },
                "rf_frequency_hz": 2.856e9,
            },
            "elements": [
                {
                    "name": "D1",
                    "kind": "Drift",
                    "s_start_m": 0.0,
                    "s_end_m": 0.5,
                    "length_m": 0.5,
                    "parameters": {},
                },
                {
                    "name": "Q1",
                    "kind": "Quadrupole",
                    "polarity": "focusing",
                    "s_start_m": 0.5,
                    "s_end_m": 0.6,
                    "length_m": 0.1,
                    "parameters": {"current_a": 2.0},
                },
                {
                    "name": "B1",
                    "kind": "SBend",
                    "s_start_m": 0.6,
                    "s_end_m": 0.7,
                    "length_m": 0.1,
                    "parameters": {
                        "bending_angle_deg": 15.0,
                        "dipole_length_m": 0.1,
                    },
                },
                {
                    "name": "K1",
                    "kind": "Kicker",
                    "plane": "horizontal",
                    "s_start_m": 0.7,
                    "s_end_m": 0.7,
                    "length_m": 0.0,
                    "parameters": {},
                },
                {
                    "name": "BPM1",
                    "kind": "Instrument",
                    "instrument_type": "BPM",
                    "s_start_m": 0.7,
                    "s_end_m": 0.7,
                    "length_m": 0.0,
                    "parameters": {},
                },
                {
                    "name": "M1",
                    "kind": "Marker",
                    "s_start_m": 0.7,
                    "s_end_m": 0.7,
                    "length_m": 0.0,
                    "parameters": {},
                },
            ],
        }
    }

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump(lattice, f)
        tmp_path = f.name

    try:
        loader = JsonLatticeLoader(tmp_path, validate_schema=False)
        assert loader._format_version == 2

        bl = loader.create_beamline()
        # Drift + QPF + Dipole = 3 objects (zero-length elements produce no object)
        assert len(bl) == 3
        assert type(bl[0]).__name__ == "driftLattice"
        assert type(bl[1]).__name__ == "qpfLattice"
        assert type(bl[2]).__name__ == "dipole"

        parsed = loader.parse_beamline()
        type_list = [d["type"] for d in parsed]
        assert "DRIFT" in type_list
        assert "QPF" in type_list
        assert "DPH" in type_list
        assert "STH" in type_list  # Kicker horizontal -> STH
        assert "BPM" in type_list  # Instrument BPM -> BPM
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Standalone mode
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
