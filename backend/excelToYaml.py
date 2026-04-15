"""
Convert an Excel beamline lattice file to YAML format.

Element type names in format_version 2 follow PALS conventions.
See manuals/lattice_specification.md for the mapping.

Delegates to excelToJson.convert() for the data transformation, then
serialises to YAML.

Usage:
    python excelToYaml.py <input.xlsx> [output.yaml]

Author: Eremey Valetov
"""

import sys
from pathlib import Path

import yaml
from excelToJson import convert as json_convert


def convert(excel_path, output_path=None, name=None, description=None,
            reference_energy_mev=45.0, particle_type="electron",
            format_version=2):  # v2 default; use 3 for computed Bn1/BendP fields
    """Convert an Excel lattice file to YAML.

    Parameters
    ----------
    excel_path : str or Path
        Path to the Excel beamline file.
    output_path : str or Path, optional
        If given, write YAML to this file.
    name : str, optional
        Lattice name for metadata.
    description : str, optional
        Lattice description for metadata.
    reference_energy_mev : float
        Reference kinetic energy in MeV.
    particle_type : str
        Particle species.
    format_version : int
        Output format version. Default 2 (PALS-aligned). Use 3 for
        computed MagneticMultipoleP.Bn1 and BendP fields.

    Returns
    -------
    dict
        The complete lattice structure (same dict as JSON variant).
    """
    lattice = json_convert(
        excel_path, output_path=None,
        name=name, description=description,
        reference_energy_mev=reference_energy_mev,
        particle_type=particle_type,
        format_version=format_version,
    )

    if output_path is not None:
        output_path = Path(output_path)
        with open(output_path, "w") as f:
            yaml.dump(lattice, f, sort_keys=False, default_flow_style=False,
                      allow_unicode=True)

    return lattice


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <input.xlsx> [output.yaml]")
        sys.exit(1)

    excel_file = sys.argv[1]
    yaml_file = sys.argv[2] if len(sys.argv) > 2 else None

    result = convert(excel_file, yaml_file, name="UH_FEL_Beamline",
                     description="University of Hawaii FEL beamline")

    n = len(result["beamline"]["elements"])
    if yaml_file:
        print(f"Converted {n} elements -> {yaml_file}")
    else:
        print(yaml.dump(result, sort_keys=False, default_flow_style=False))
