"""
Convert an Excel beamline lattice file to the FELsim JSON lattice format.

Element type names in format_version 2+ follow PALS conventions.
See manuals/lattice_specification.md for the mapping.

format_version 3 extends v2 by computing PALS-aligned physics fields:
  - MagneticMultipoleP.Bn1 for quadrupoles (pole-tip field in Tesla)
  - BendP.g_ref for dipoles (reference bend strength in 1/m)
  - BendP.e1/e2 for dipoles bracketed by wedge elements (edge angles in rad)

Uses only the data already parsed by ExcelElements (no new columns or
transformations). The output can be loaded by JsonLatticeLoader and
should produce equivalent beamline representations.

Usage:
    python excelToJson.py [input.xlsx] [output.json] [--v2|--v3]

Author: Eremey Valetov
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from excelElements import ExcelElements

# Excel short names -> canonical JSON type names (v1)
_TYPE_MAP_V1 = {
    "QPF": ("QUADRUPOLE", "focusing"),
    "QPD": ("QUADRUPOLE", "defocusing"),
    "DPH": "DIPOLE",
    "DPW": "DIPOLE_WEDGE",
    "BPM": "BPM",
    "OTR": "OTR",
    "STV": "CORRECTOR_V",
    "STH": "CORRECTOR_H",
    "SPC": "SPECTROMETER",
    "UND": "UNDULATOR",
    "XRS": "XRS",
    "BSW": "BSW",
}

# PALS-aligned type names (v2)
_TYPE_MAP_V2 = {
    "QPF": ("Quadrupole", "focusing"),
    "QPD": ("Quadrupole", "defocusing"),
    "DPH": "SBend",
    "DPW": "DIPOLE_WEDGE",  # no PALS equivalent
    "BPM": "BPM",
    "OTR": "OTR",
    "STV": "CORRECTOR_V",
    "STH": "CORRECTOR_H",
    "SPC": "SPECTROMETER",
    "UND": "Wiggler",
    "XRS": "XRS",
    "BSW": "BSW",
}


def convert(excel_path, output_path=None, name=None, description=None,
            reference_energy_mev=45.0, particle_type="electron",
            format_version=1):
    """Convert an Excel lattice file to JSON.

    Parameters
    ----------
    excel_path : str or Path
        Path to the Excel beamline file.
    output_path : str or Path, optional
        If given, write JSON to this file.
    name : str, optional
        Lattice name for metadata. Defaults to stem of Excel filename.
    description : str, optional
        Lattice description for metadata.
    reference_energy_mev : float
        Reference kinetic energy in MeV.
    particle_type : str
        Particle species.
    format_version : int
        Output format version (1, 2, or 3). Default 1 for backward compatibility.
        v3 adds MagneticMultipoleP.Bn1 for quads and BendP for dipoles.
        Note: excelToYaml defaults to format_version=2 since YAML is a v2-era format.

    Returns
    -------
    dict
        The complete JSON lattice structure.
    """
    excel_path = Path(excel_path)
    ee = ExcelElements(str(excel_path))
    df = ee.get_dataframe()

    if name is None:
        name = excel_path.stem

    elements, sectors = _build_elements(df, format_version)

    if format_version >= 3:
        G = 2.694  # matches global_settings default
        _add_v3_fields(elements, G=G, default_aperture=0.027)

    lattice = {
        "beamline": {
            "metadata": {
                "format_version": format_version,
                "name": name,
                "version": "1.0",
                "description": description or f"Converted from {excel_path.name}",
                "reference_energy_mev": reference_energy_mev,
                "particle_type": particle_type,
            },
            "beam_parameters": {
                "particle": {
                    "type": particle_type,
                    "kinetic_energy_mev": reference_energy_mev,
                    "mass_mev": 0.51099895,
                    "charge_e": -1,
                },
                "rf_frequency_hz": 2.856e9,
            },
            "elements": elements,
            "lattice_structure": {"sectors": sectors},
            "global_settings": {
                "coordinate_system": "felsim",
                "length_unit": "m",
                "angle_unit": "deg",
                "field_unit": "T",
                "current_unit": "A",
                "energy_unit": "MeV",
                "quadrupole_gradient_coefficient_t_per_a_per_m": 2.694,
            },
        }
    }

    if output_path is not None:
        output_path = Path(output_path)
        with open(output_path, "w") as f:
            json.dump(lattice, f, indent=2)
            f.write("\n")

    return lattice


def _safe(val):
    """Convert NaN / numpy types to JSON-safe Python types."""
    if isinstance(val, float) and np.isnan(val):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


def _add_v3_fields(elements, G, default_aperture):
    """Enrich elements in-place with PALS v3 fields (Bn1, BendP).

    Parameters
    ----------
    elements : list[dict]
        Element dicts produced by _build_elements() with format_version >= 2.
    G : float
        Quadrupole gradient coefficient (T/A/m).
    default_aperture : float
        Default full aperture (m) used when aperture_m is not set on a quad.
    """
    for elem in elements:
        if elem["type"] == "Quadrupole":
            current = elem["parameters"].get("current_a", 0.0)
            aperture = elem.get("aperture_m", default_aperture)
            r = aperture / 2.0
            sign = -1.0 if elem.get("polarity") == "focusing" else 1.0
            elem["MagneticMultipoleP"] = {"Bn1": sign * G * current * r}

        elif elem["type"] == "SBend":
            angle_deg = elem["parameters"].get("bending_angle_deg", 0.0)
            L = elem["parameters"].get("dipole_length_m", elem["length_m"])
            angle_rad = math.radians(angle_deg)
            g_ref = angle_rad / L if L != 0 else 0.0
            elem["BendP"] = {"g_ref": g_ref}

    # Propagate DPW edge angles into adjacent SBend elements
    for i in range(1, len(elements) - 1):
        if (elements[i]["type"] == "SBend"
                and elements[i - 1]["type"] == "DIPOLE_WEDGE"
                and elements[i + 1]["type"] == "DIPOLE_WEDGE"):
            e1 = math.radians(elements[i - 1]["parameters"].get("wedge_angle_deg", 0.0))
            e2 = math.radians(elements[i + 1]["parameters"].get("wedge_angle_deg", 0.0))
            elements[i].setdefault("BendP", {}).update({"e1": e1, "e2": e2})
            elements[i]["parameters"]["entrance_edge_angle_deg"] = elements[i - 1]["parameters"].get("wedge_angle_deg", 0.0)
            elements[i]["parameters"]["exit_edge_angle_deg"] = elements[i + 1]["parameters"].get("wedge_angle_deg", 0.0)


def _build_elements(df, format_version=1):
    """Build JSON element list and sector grouping from DataFrame."""
    elements = []
    sector_map = {}
    used_names = set()

    for _, row in df.iterrows():
        etype = row["Element"]
        if pd.isna(etype):
            continue

        z_start = row["z_start"]
        z_end = row["z_end"]
        if pd.isna(z_start) or pd.isna(z_end):
            continue

        elem_name = _make_name(row, used_names)
        used_names.add(elem_name)

        length = float(z_end - z_start)

        json_type, polarity = _map_type(etype, format_version)
        params = _build_parameters(etype, row)
        fringe = _build_fringe(row)
        aperture = _build_aperture(etype, row)
        meta = _build_metadata(row)

        elem = {
            "name": elem_name,
            "type": json_type,
            "s_start_m": float(z_start),
            "s_end_m": float(z_end),
            "length_m": length,
            "parameters": params,
        }
        if polarity:
            elem["polarity"] = polarity
        if fringe:
            elem["fringe_fields"] = fringe
        if aperture is not None:
            elem["aperture_m"] = aperture
        if meta:
            elem["metadata"] = meta

        elements.append(elem)

        sector = _safe(row.get("Sector"))
        if sector:
            sector_map.setdefault(sector, []).append(elem_name)

    sectors = [{"name": s, "element_names": names} for s, names in sector_map.items()]
    return elements, sectors


def _make_name(row, used_names):
    """Generate a unique element name from the row data."""
    nomenclature = row.get("Nomenclature")
    label = row.get("Label")
    etype = row["Element"]
    sector = row.get("Sector", "")

    if pd.notna(label) and str(label).strip():
        candidate = str(label).strip()
    elif pd.notna(nomenclature) and str(nomenclature).strip():
        candidate = str(nomenclature).strip().replace(".", "_")
    else:
        candidate = f"{sector}_{etype}" if pd.notna(sector) else etype

    name = candidate
    i = 2
    while name in used_names:
        name = f"{candidate}_{i}"
        i += 1
    return name


def _map_type(excel_type, format_version=1):
    """Map Excel element type to (JSON type, polarity or None)."""
    type_map = _TYPE_MAP_V2 if format_version >= 2 else _TYPE_MAP_V1
    mapped = type_map.get(excel_type, excel_type)
    if isinstance(mapped, tuple):
        return mapped
    return mapped, None


def _build_parameters(etype, row):
    """Build the parameters dict for an element."""
    params = {}
    if etype in ("QPF", "QPD"):
        current = row.get("Current (A)")
        if pd.notna(current):
            params["current_a"] = float(current)

    elif etype == "DPH":
        angle = row.get("Dipole Angle (deg)")
        if pd.notna(angle):
            params["bending_angle_deg"] = float(angle)
        dlen = row.get("Dipole length (m)")
        if pd.notna(dlen):
            params["dipole_length_m"] = float(dlen)
        pgap = row.get("Pole gap (m)")
        if pd.notna(pgap):
            params["pole_gap_m"] = float(pgap)

    elif etype == "DPW":
        wedge_angle = row.get("Dipole wedge (deg)")
        if pd.notna(wedge_angle):
            params["wedge_angle_deg"] = float(wedge_angle)
        angle = row.get("Dipole Angle (deg)")
        if pd.notna(angle):
            params["dipole_angle_deg"] = float(angle)
        dlen = row.get("Dipole length (m)")
        if pd.notna(dlen):
            params["dipole_length_m"] = float(dlen)
        pgap = row.get("Pole gap (m)")
        if pd.notna(pgap):
            params["pole_gap_m"] = float(pgap)

    return params


def _build_fringe(row):
    """Build fringe_fields dict if Enge coefficients are present."""
    enge_raw = row.get("Fringe Field Enge coefficients")
    if pd.isna(enge_raw):
        return None
    if isinstance(enge_raw, str) and enge_raw.strip():
        coeffs = [float(v.strip()) for v in enge_raw.split(",") if v.strip()]
        return {"enge_coefficients": coeffs}
    if isinstance(enge_raw, (int, float)):
        return {"enge_coefficients": [float(enge_raw)]}
    return None


def _build_aperture(etype, row):
    """Extract aperture (Pole gap used as aperture for quads)."""
    if etype in ("QPF", "QPD"):
        pgap = row.get("Pole gap (m)")
        if pd.notna(pgap):
            return float(pgap)
    return None


def _build_metadata(row):
    """Build the metadata dict from auxiliary Excel columns."""
    meta = {}
    nomenclature = row.get("Nomenclature")
    if pd.notna(nomenclature) and str(nomenclature).strip():
        meta["nomenclature"] = str(nomenclature).strip()
    elem_name = row.get("Element name")
    if pd.notna(elem_name) and str(elem_name).strip():
        meta["element_name"] = str(elem_name).strip()
    channel = row.get("Channel")
    if pd.notna(channel):
        meta["channel"] = int(channel)
    label = row.get("Label")
    if pd.notna(label) and str(label).strip():
        meta["label"] = str(label).strip()
    sector = row.get("Sector")
    if pd.notna(sector) and str(sector).strip():
        meta["sector"] = str(sector).strip()
    return meta


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <input.xlsx> [output.json] [--v2|--v3]")
        sys.exit(1)

    fv = 3 if "--v3" in sys.argv else (2 if "--v2" in sys.argv else 1)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    excel_file = args[0]
    json_file = args[1] if len(args) > 1 else None

    result = convert(excel_file, json_file, name="UH_FEL_Beamline",
                     description="University of Hawaii FEL beamline",
                     format_version=fv)

    n = len(result["beamline"]["elements"])
    if json_file:
        print(f"Converted {n} elements -> {json_file} (format_version {fv})")
    else:
        print(json.dumps(result, indent=2))
