"""
Unified lattice loading from any supported format (Excel, JSON, YAML).

Auto-detects format from file extension and delegates to the appropriate
loader. Provides two output modes:

  - create_beamline(path) -> list of beamline.py class instances
  - parse_beamline(path)  -> list of dicts (BeamlineBuilder-compatible)

Author: Eremey Valetov
"""

from pathlib import Path

_EXCEL_EXTENSIONS = {".xlsx", ".xls"}
_JSON_EXTENSIONS = {".json"}
_YAML_EXTENSIONS = {".yaml", ".yml"}

_ALL_EXTENSIONS = _EXCEL_EXTENSIONS | _JSON_EXTENSIONS | _YAML_EXTENSIONS


def create_beamline(lattice_path):
    """Load a lattice file and return beamline.py class instances.

    Parameters
    ----------
    lattice_path : str or Path
        Path to an Excel (.xlsx), JSON (.json), or YAML (.yaml/.yml) lattice file.

    Returns
    -------
    list
        beamline.py objects (driftLattice, qpfLattice, etc.)
    """
    ext = _check_extension(lattice_path)

    if ext in _EXCEL_EXTENSIONS:
        from excelElements import ExcelElements
        return ExcelElements(str(lattice_path)).create_beamline()

    if ext in _JSON_EXTENSIONS:
        from jsonLatticeLoader import JsonLatticeLoader
        return JsonLatticeLoader(str(lattice_path)).create_beamline()

    from yamlLatticeLoader import YamlLatticeLoader
    return YamlLatticeLoader(str(lattice_path)).create_beamline()


def parse_beamline(lattice_path):
    """Load a lattice file and return BeamlineBuilder-compatible dict list.

    Parameters
    ----------
    lattice_path : str or Path
        Path to an Excel (.xlsx), JSON (.json), or YAML (.yaml/.yml) lattice file.

    Returns
    -------
    list[dict]
        Element dicts with keys: type, length, current, angle, etc.
    """
    ext = _check_extension(lattice_path)

    if ext in _EXCEL_EXTENSIONS:
        from beamlineBuilder import BeamlineBuilder
        bb = BeamlineBuilder(str(lattice_path))
        return bb.parse_beamline()

    if ext in _JSON_EXTENSIONS:
        from jsonLatticeLoader import JsonLatticeLoader
        return JsonLatticeLoader(str(lattice_path)).parse_beamline()

    from yamlLatticeLoader import YamlLatticeLoader
    return YamlLatticeLoader(str(lattice_path)).parse_beamline()


def detect_format(lattice_path):
    """Return the format name for a lattice file path.

    Returns
    -------
    str
        One of 'excel', 'json', 'yaml'.
    """
    ext = _check_extension(lattice_path)
    if ext in _EXCEL_EXTENSIONS:
        return "excel"
    if ext in _JSON_EXTENSIONS:
        return "json"
    return "yaml"


def _check_extension(lattice_path):
    """Validate file extension and return it lowercased."""
    ext = Path(lattice_path).suffix.lower()
    if ext not in _ALL_EXTENSIONS:
        raise ValueError(
            f"Unsupported lattice file format '{ext}'. "
            f"Expected one of: {', '.join(sorted(_ALL_EXTENSIONS))}"
        )
    return ext
