"""Data locations resolved from the package, independent of the working directory.

Package data (ESTAR tables) ships inside felsim/. Machine inputs (lattices and schemas
in var/, field maps, reference parameters, the Excel element table) and run outputs
live in the repository checkout that holds the package, as with `pip install -e`.
"""
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

ESTAR_DIR = PACKAGE_DIR / "data" / "estar"
LATTICE_DIR = REPO_ROOT / "var"
FIELDS_DIR = REPO_ROOT / "fields"
PARAMETERS_DIR = REPO_ROOT / "parameters"
EXCEL_DIR = REPO_ROOT / "beam_excel"
OUTPUT_DIR = REPO_ROOT / "outputs"
