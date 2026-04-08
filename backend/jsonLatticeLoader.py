"""
Load beamline lattice configurations from JSON files.

Thin wrapper around LatticeLoaderBase that handles JSON I/O and schema
validation. Supports format_version 1, 2, and 3.

Author: Eremey Valetov
"""

import json
from pathlib import Path

from tracked_dict import TrackedDict
from latticeLoaderBase import LatticeLoaderBase, SUPPORTED_FORMAT_VERSIONS

_SCHEMA_FILES = {
    1: "lattice_schema_v1.json",
    2: "lattice_schema_v2.json",
    3: "lattice_schema_v3.json",
}


class JsonLatticeLoader(LatticeLoaderBase):
    """Load a FELsim JSON lattice file into beamline representations."""

    def __init__(self, file_path, validate_schema=True, debug=None):
        file_path = str(file_path)

        with open(file_path) as f:
            raw = json.load(f)

        if validate_schema:
            self._validate_schema(raw, file_path)

        tracked = TrackedDict(raw)
        super().__init__(tracked, file_path=file_path, debug=debug)

    @staticmethod
    def _validate_schema(raw, file_path):
        """Validate raw JSON against the lattice schema (if jsonschema is installed)."""
        try:
            import jsonschema
        except ImportError:
            return

        fv = raw.get("beamline", {}).get("metadata", {}).get("format_version", 1)
        schema_name = _SCHEMA_FILES.get(fv, _SCHEMA_FILES[1])

        schema_path = Path(file_path).resolve().parent.parent / "var" / schema_name
        if not schema_path.exists():
            schema_path = Path(__file__).resolve().parent.parent / "var" / schema_name
        if not schema_path.exists():
            return

        with open(schema_path) as f:
            schema = json.load(f)

        try:
            jsonschema.validate(raw, schema)
        except jsonschema.ValidationError as e:
            raise ValueError(f"JSON schema validation failed: {e.message}") from e
