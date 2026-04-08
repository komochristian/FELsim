"""
Base class for loading beamline lattice configurations from structured data.

Element kind names and conventions follow the PALS (Particle Accelerator
Lattice Standard) where practical. The _TYPE_ALIASES mapping includes both
FELsim-native names and PALS-compatible names. When adding new element types,
prefer PALS naming conventions.
See: https://pals-project.readthedocs.io/

Produces two output formats:
  - parse_beamline()  -> list of dicts (BeamlineBuilder-compatible)
  - create_beamline() -> list of beamline.py class instances

Author: Eremey Valetov
"""

import math

from tracked_dict import TrackedDict
from beamline import driftLattice, qpfLattice, qpdLattice, dipole, dipole_wedge
from loggingConfig import get_logger_with_fallback

SUPPORTED_FORMAT_VERSIONS = [1, 2, 3]

# Map element type/kind names to internal short names.
# Includes FELsim-native names and PALS CamelCase aliases.
_TYPE_ALIASES = {
    # FELsim-native names
    "QUADRUPOLE": None,  # resolved via polarity
    "QPF": "QPF",
    "QPD": "QPD",
    "DIPOLE": "DPH",
    "DPH": "DPH",
    "DIPOLE_WEDGE": "DPW",
    "DPW": "DPW",
    "SOLENOID": "SOL",
    "SOL": "SOL",
    "RF_CAVITY": "RFC",
    "RFC": "RFC",
    "SEXTUPOLE": "SXT",
    "SXT": "SXT",
    "UNDULATOR": "UND",
    "UND": "UND",
    "BPM": "BPM",
    "OTR": "OTR",
    "CORRECTOR_V": "STV",
    "STV": "STV",
    "CORRECTOR_H": "STH",
    "STH": "STH",
    "SPECTROMETER": "SPC",
    "SPC": "SPC",
    "XRS": "XRS",
    "BSW": "BSW",
    "DRIFT": "DRIFT",
    # PALS CamelCase aliases (format_version 2)
    "Drift": "DRIFT",
    "Quadrupole": None,  # resolved via polarity
    "SBend": "DPH",
    "RBend": "DPH",
    "Wiggler": "UND",
    "Solenoid": "SOL",
    "RFCavity": "RFC",
    "Sextupole": "SXT",
    "Kicker": None,  # resolved via plane
    "Instrument": None,  # resolved via instrument_type
    "Marker": "DRIFT",  # zero-length drift
}


class LatticeLoaderBase:
    """Format-independent lattice loader.

    Subclasses provide file I/O (JSON, YAML, etc.) and pass a pre-parsed
    dict wrapped in TrackedDict to this base class.
    """

    def __init__(self, tracked_dict, file_path=None, debug=None):
        self.file_path = str(file_path) if file_path else "<in-memory>"
        self.logger, self.debug = get_logger_with_fallback(__name__, debug)

        self._tracked = tracked_dict
        self._beamline = self._tracked["beamline"]

        fv = self._beamline["metadata"]["format_version"]
        try:
            fv = int(fv)
        except (TypeError, ValueError):
            pass
        if fv not in SUPPORTED_FORMAT_VERSIONS:
            raise ValueError(
                f"Unsupported format_version {fv} (expected one of {SUPPORTED_FORMAT_VERSIONS})"
            )
        self._format_version = fv
        self._resolved_types = {}  # id(elem) -> resolved type string

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_beamline(self):
        """Return list of dicts compatible with BeamlineBuilder / adapters.

        Dict keys: type, length, current, angle, wedge_angle, gap_wedge,
        pole_gap, enge_fct, z_start, z_end.
        Drifts are auto-inserted between elements.
        """
        elements = self._positioned_elements()
        result = []
        prev_z_end = 0.0

        for elem in elements:
            z_start = elem["s_start_m"]
            z_end = elem["s_end_m"]

            if z_start > prev_z_end:
                result.append({"type": "DRIFT", "length": z_start - prev_z_end})

            result.append(self._element_to_dict(elem))
            prev_z_end = z_end

        self._report_unaccessed()
        return result

    def create_beamline(self):
        """Return list of beamline.py class instances (driftLattice, qpfLattice, etc.).

        Mirrors the output of ExcelElements.create_beamline().
        """
        elements = self._positioned_elements()
        result = []
        prev_z_end = 0.0

        for elem in elements:
            z_start = elem["s_start_m"]
            z_end = elem["s_end_m"]

            if z_start > prev_z_end:
                result.append(driftLattice(z_start - prev_z_end))

            obj = self._element_to_object(elem)
            if obj is not None:
                result.append(obj)
            prev_z_end = z_end

        self._report_unaccessed()
        return result

    @property
    def metadata(self):
        return self._beamline["metadata"]

    @property
    def beam_parameters(self):
        return self._beamline["beam_parameters"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _positioned_elements(self):
        """Return elements with valid s_start/s_end, sorted by s_start."""
        meta = self._beamline["metadata"]
        _ = meta["name"], meta["version"], meta["reference_energy_mev"], meta["particle_type"]
        meta.mark_accessed("description", "author", "date")

        bp = self._beamline["beam_parameters"]
        _ = bp["particle"]["type"], bp["particle"]["kinetic_energy_mev"]
        _ = bp["particle"]["mass_mev"], bp["particle"]["charge_e"]
        _ = bp["rf_frequency_hz"]

        self._beamline.mark_accessed(
            "lattice_structure", "global_settings", "simulator_specific"
        )

        gs = self._beamline.get("global_settings", {})
        if hasattr(gs, 'get') and gs.get("angle_unit") == "rad":
            self.logger.warning(
                "angle_unit: rad is reserved for future use and currently has no effect. "
                "All angle parameters are interpreted as degrees."
            )

        raw_elements = self._beamline["elements"]
        positioned = []
        for elem in raw_elements:
            # Normalize kind -> type for v2
            self._normalize_element_kind(elem)

            s_start = elem["s_start_m"]
            s_end = elem["s_end_m"]
            if s_start is not None and s_end is not None and s_end > s_start:
                positioned.append(elem)
            elif s_start is not None and s_end is not None and s_start == s_end:
                positioned.append(elem)
            else:
                self.logger.debug(f"Skipping element {elem['name']!r} (no valid position)")
                elem.mark_all_accessed()
        positioned.sort(key=lambda e: e["s_start_m"].raw if hasattr(e["s_start_m"], 'raw') else e["s_start_m"])
        return positioned

    def _normalize_element_kind(self, elem):
        """Resolve v2 conventions: kind->type, Kicker->corrector, Instrument->diagnostic, Marker->drift."""
        has_kind = "kind" in elem
        has_type = "type" in elem

        if has_kind:
            _ = elem["kind"]  # mark accessed
            if has_type:
                # Both present; type takes precedence
                self.logger.debug(
                    f"Element {elem.get('name', '<unnamed>')!r}: both 'type' ({elem['type']}) "
                    f"and 'kind' ({elem['kind']}) present; using 'type'"
                )
                self._resolved_types[id(elem)] = elem["type"]
                return
            raw_type = _
        elif has_type:
            raw_type = elem["type"]
        else:
            return

        # Resolve PALS compound types
        if raw_type == "Kicker":
            plane = elem.get("plane", "vertical")
            elem.mark_accessed("plane")
            self._resolved_types[id(elem)] = "CORRECTOR_V" if plane == "vertical" else "CORRECTOR_H"
        elif raw_type == "Instrument":
            inst_type = elem.get("instrument_type", "BPM")
            elem.mark_accessed("instrument_type")
            self._resolved_types[id(elem)] = inst_type
        elif raw_type == "Marker":
            self._resolved_types[id(elem)] = "DRIFT"
        else:
            self._resolved_types[id(elem)] = raw_type

    def _resolve_type(self, elem):
        """Resolve element type + polarity to internal short name."""
        raw_type = self._resolved_types.get(id(elem))
        if raw_type is None:
            raw_type = elem["type"]

        if raw_type in ("QUADRUPOLE", "Quadrupole"):
            polarity = elem["polarity"]
            return "QPF" if polarity == "focusing" else "QPD"

        short = _TYPE_ALIASES.get(raw_type, raw_type)
        if short is None:
            # Fallback for unresolved compound types
            return raw_type
        return short

    def _resolve_quad_current(self, elem, params):
        """Resolve quadrupole current from current_a or MagneticMultipoleP.Bn1."""
        current = params.get("current_a", 0)
        mmp = elem.get("MagneticMultipoleP")
        if mmp is not None:
            bn1 = mmp.get("Bn1")
            if bn1 is not None:
                gs = self._beamline.get("global_settings", {})
                G = gs.get("quadrupole_gradient_coefficient_t_per_a_per_m")
                if G is None:
                    G = 2.694
                r = elem.get("aperture_m")
                if r is None:
                    r = 0.027
                r /= 2  # bore diameter → radius
                if G * r != 0:
                    # Bn1 sign encodes polarity (negative=focusing), but
                    # FELsim uses unsigned current with QPF/QPD class choice.
                    current = abs(bn1 / (G * r))
            mmp.mark_all_accessed()
        elem.mark_accessed("MagneticMultipoleP")
        return current

    def _resolve_dipole_angle(self, elem, params, length):
        """Resolve dipole angle from bending_angle_deg or BendP.g_ref."""
        angle = params.get("bending_angle_deg", 0)
        bend_p = elem.get("BendP")
        if bend_p is not None:
            g_ref = bend_p.get("g_ref")
            if g_ref is not None:
                angle = math.degrees(g_ref * length)
            bend_p.mark_all_accessed()
        elem.mark_accessed("BendP")
        return angle

    def _element_to_dict(self, elem):
        """Convert a tracked element to a BeamlineBuilder-compatible dict."""
        internal_type = self._resolve_type(elem)
        length = elem["length_m"]
        z_start = elem["s_start_m"]
        z_end = elem["s_end_m"]
        params = elem["parameters"]

        current = self._resolve_quad_current(elem, params)
        angle = 0.0
        wedge_angle = 0.0
        gap_wedge = 0.0
        pole_gap = 0.0
        enge_fct = ""

        if internal_type == "DPH":
            angle = self._resolve_dipole_angle(elem, params, length)
            pole_gap = params.get("pole_gap_m", 0)
            params.mark_accessed("dipole_length_m")

            # BendP edge angles cannot be applied in dict/object output —
            # FELsim requires DPW-DPH-DPW triplets for edge kicks.
            bend_p = elem.get("BendP")
            if bend_p is not None:
                e1 = bend_p.get("e1")
                e2 = bend_p.get("e2")
                if e1 is not None:
                    params.mark_accessed("entrance_edge_angle_deg")
                if e2 is not None:
                    params.mark_accessed("exit_edge_angle_deg")
                if (e1 is not None and e1 != 0) or (e2 is not None and e2 != 0):
                    self.logger.warning(
                        f"Element {elem.get('name')!r}: BendP edge angles "
                        f"(e1={e1}, e2={e2}) are not applied in FELsim output. "
                        f"Edge kicks require DPW-DPH-DPW triplet representation."
                    )
                bend_p.mark_all_accessed()
            elem.mark_accessed("BendP")

        elif internal_type == "DPW":
            angle = params.get("dipole_angle_deg", 0)
            wedge_angle = params.get("wedge_angle_deg", 0)
            gap_wedge = length
            pole_gap = params.get("pole_gap_m", 0)
            params.mark_accessed("dipole_length_m")
            enge_fct = self._get_enge(elem)

        name = elem.get("name")
        elem.mark_accessed("name", "aperture_m", "optimization", "fringe_fields", "metadata")
        params.mark_all_accessed()

        return {
            "type": internal_type,
            "name": name,
            "length": length,
            "current": current,
            "angle": angle,
            "wedge_angle": wedge_angle,
            "gap_wedge": gap_wedge,
            "pole_gap": pole_gap,
            "enge_fct": enge_fct,
            "z_start": z_start,
            "z_end": z_end,
        }

    def _element_to_object(self, elem):
        """Convert a tracked element to a beamline.py class instance."""
        internal_type = self._resolve_type(elem)
        length = elem["length_m"]
        z_start = elem["s_start_m"]
        z_end = elem["s_end_m"]
        params = elem["parameters"]
        name = elem.get("name")

        elem.mark_accessed("name", "aperture_m", "optimization", "fringe_fields", "metadata")

        if internal_type == "DRIFT":
            params.mark_all_accessed()
            if length > 0:
                return driftLattice(length, name=name)
            return None

        elif internal_type in ("QPF", "QPD"):
            current = self._resolve_quad_current(elem, params)
            params.mark_all_accessed()
            cls = qpfLattice if internal_type == "QPF" else qpdLattice
            return cls(current=current, length=length, name=name)

        elif internal_type == "DPH":
            dipole_length = params.get("dipole_length_m", length) or length
            angle = self._resolve_dipole_angle(elem, params, dipole_length)
            # Warn about BendP edge angles that can't be applied
            bend_p = elem.get("BendP")
            if bend_p is not None:
                e1, e2 = bend_p.get("e1"), bend_p.get("e2")
                if (e1 is not None and e1 != 0) or (e2 is not None and e2 != 0):
                    self.logger.warning(
                        f"Element {name!r}: BendP edge angles "
                        f"(e1={e1}, e2={e2}) are not applied in FELsim output. "
                        f"Edge kicks require DPW-DPH-DPW triplet representation."
                    )
                bend_p.mark_all_accessed()
            elem.mark_accessed("BendP")
            params.mark_all_accessed()
            return dipole(length=dipole_length, angle=angle, name=name)

        elif internal_type == "DPW":
            wedge_angle = params.get("wedge_angle_deg", 0)
            dipole_angle = params.get("dipole_angle_deg", 0)
            dipole_length = params.get("dipole_length_m", 0)
            pole_gap = params.get("pole_gap_m", 0)
            params.mark_all_accessed()
            enge_fct = self._get_enge(elem)
            return dipole_wedge(
                length=length, angle=wedge_angle,
                dipole_length=dipole_length, dipole_angle=dipole_angle,
                pole_gap=pole_gap, enge_fct=enge_fct, name=name,
            )

        else:
            # Diagnostic / passive elements
            params.mark_all_accessed()
            if length > 0:
                return driftLattice(length, name=name)
            return None

    def _get_enge(self, elem):
        """Extract Enge coefficients from a tracked element."""
        if "fringe_fields" not in elem:
            return []
        ff = elem["fringe_fields"]
        coeffs = ff.get("enge_coefficients")
        ff.mark_all_accessed()
        if coeffs is None:
            return []
        if hasattr(coeffs, "raw"):
            return list(coeffs.raw)
        return list(coeffs) if coeffs else []

    def _report_unaccessed(self):
        """Log any unconsumed data paths."""
        unaccessed = self._tracked.unaccessed()
        if unaccessed:
            self.logger.info(f"Lattice: {len(unaccessed)} unhandled field(s):")
            for path in unaccessed:
                self.logger.info(f"  {path}")
