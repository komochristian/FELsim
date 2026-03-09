# FELsim Lattice Specification

Version 2.0

> See also: the v3 format specification for format_version 3
> extensions (MagneticMultipoleP, BendP).

## Overview

The lattice format provides a human-readable, version-controllable beamline specification supporting both JSON and YAML serialisation. It contains all information needed to define a beamline: element sequence, physical parameters, beam properties, and simulator-specific settings.

Reference examples:
- JSON (v2, curated PALS-aligned): `var/lattice_specification.json`
- JSON (v1, converted from Excel): `var/UH_FEL_beamline.json`
- YAML (v2, converted from Excel): `var/UH_FEL_beamline.yaml`


## File Formats

| Format | Loader | Extension |
|--------|--------|-----------|
| JSON | `JsonLatticeLoader` | `.json` |
| YAML | `YamlLatticeLoader` | `.yaml`, `.yml` |

Both formats produce identical in-memory representations and share the same JSON Schema for validation.


## Loading a Lattice

The `latticeLoader` module provides format-agnostic loading that auto-detects by file extension:

```python
import latticeLoader

# Returns beamline.py transfer matrix objects (driftLattice, qpfLattice, etc.)
beamline = latticeLoader.create_beamline("var/UH_FEL_beamline.yaml")

# Returns BeamlineBuilder-compatible dict list (for COSY adapter, etc.)
elements = latticeLoader.parse_beamline("var/UH_FEL_beamline.json")

# Works with Excel too
beamline = latticeLoader.create_beamline("beam_excel/Beamline_elements.xlsx")
```

All simulator adapters accept a `lattice_path` parameter that delegates to `latticeLoader`:

```python
from felsimAdapter import FELsimAdapter

adapter = FELsimAdapter(lattice_path="var/UH_FEL_beamline.yaml")
```


## Top-Level Structure

```yaml
beamline:
  metadata: {}
  beam_parameters: {}
  elements: []
  lattice_structure: {}
  global_settings: {}
  simulator_specific: {}
```

All data lives under the top-level `beamline` key.


## Versioning

| Field            | Type    | Meaning                                                   |
|------------------|---------|-----------------------------------------------------------|
| `format_version` | integer | Schema version. Incremented when the structure changes.   |
| `version`        | string  | User-managed label for the lattice data. Purely informational. |

### Format versions

| Version | Schema file | Key changes |
|---------|-------------|-------------|
| 1 | `var/lattice_schema_v1.json` | Initial format. FELsim-native type names only. |
| 2 | `var/lattice_schema_v2.json` | PALS-aligned element kind names. `kind` accepted as alias for `type`. YAML support. |

### Compatibility rules

- A parser supporting format_version *N* must accept any file with `format_version <= N`.
- Additions of new **optional** fields do not increment `format_version`.
- Additions of new **required** fields, renaming of existing fields, or structural changes increment `format_version`.

### Validation

Structural validation uses JSON Schema (works for both JSON and YAML since the in-memory representation is identical):

```python
import json, jsonschema, yaml

with open('var/UH_FEL_beamline.yaml') as f:
    data = yaml.safe_load(f)
with open('var/lattice_schema_v2.json') as f:
    schema = json.load(f)

jsonschema.validate(data, schema)
```

The schema uses `additionalProperties: true` everywhere — extra fields are allowed.

### Tracking unconsumed data

The parser wraps loaded data in a `TrackedDict` that records which keys are actually read. After parsing, `unaccessed()` returns every path that was never touched. This catches both parser omissions and user extras without maintaining an explicit expected-key list.


## PALS Alignment

The [PALS (Particle Accelerator Lattice Standard)](https://github.com/pals-project/pals) project is establishing a community standard for describing particle accelerator lattices. Format version 2 aligns FELsim with PALS where practical.

### What we adopt from PALS

- **CamelCase element kind names** as aliases (`Quadrupole`, `SBend`, `Wiggler`, etc.)
- **`kind` as alias for `type`** — the core PALS convention for element classification
- **Compound types** `Kicker` (resolved via `plane`) and `Instrument` (resolved via `instrument_type`)
- **`Marker`** — treated as zero-length drift

### Where we diverge and why

| Area | FELsim | PALS | Reason |
|------|--------|------|--------|
| Dipole wedges | Separate `DIPOLE_WEDGE` element | `e1`/`e2` fields on the bend | FELsim's transfer matrix pipeline treats wedges as distinct elements |
| Quadrupole excitation | Current in Amperes (`current_a`) | Field gradient (T/m) | FELsim uses measured gradient coefficients to convert current to gradient |
| Angle units | Degrees | Radians | FELsim convention throughout the codebase |
| Parameter structure | Flat `parameters` dict | Varies | Keeps FELsim's existing pipeline intact |

### FELsim ↔ PALS element kind mapping

| PALS name | FELsim equivalent | Internal code | Notes |
|-----------|-------------------|---------------|-------|
| `Drift` | `DRIFT` | `DRIFT` | |
| `Quadrupole` | `QUADRUPOLE` | `QPF`/`QPD` | Resolved via `polarity` |
| `SBend` | `DIPOLE` | `DPH` | Sector bend |
| `RBend` | `DIPOLE` | `DPH` | Rectangular bend (same internal treatment) |
| `Wiggler` | `UNDULATOR` | `UND` | |
| `Solenoid` | `SOLENOID` | `SOL` | |
| `RFCavity` | `RF_CAVITY` | `RFC` | |
| `Sextupole` | `SEXTUPOLE` | `SXT` | |
| `Kicker` | `CORRECTOR_V`/`CORRECTOR_H` | `STV`/`STH` | Resolved via `plane` field |
| `Instrument` | `BPM`/`OTR`/`SPECTROMETER` | `BPM`/`OTR`/`SPC` | Resolved via `instrument_type` field |
| `Marker` | — | `DRIFT` | Zero-length drift |
| — | `DIPOLE_WEDGE`/`DPW` | `DPW` | FELsim-specific, no PALS equivalent |


## Sections

### `metadata`

| Field                  | Type    | Required | Description                               |
|------------------------|---------|----------|-------------------------------------------|
| `format_version`       | integer | yes      | Schema version (`1` or `2`)               |
| `name`                 | string  | yes      | Beamline identifier                       |
| `version`              | string  | yes      | Lattice data version (user-managed)       |
| `description`          | string  | no       | Human-readable description                |
| `author`               | string  | no       | Author or group                           |
| `date`                 | string  | no       | Date of last modification (ISO 8601)      |
| `reference_energy_mev` | float   | yes      | Reference kinetic energy in MeV           |
| `particle_type`        | string  | yes      | Particle species                          |


### `beam_parameters`

Defines the reference particle and beam RF structure.

```yaml
beam_parameters:
  particle:
    type: electron
    kinetic_energy_mev: 45.0
    mass_mev: 0.51099895
    charge_e: -1
  rf_frequency_hz: 2.856e9
```

| Field              | Type   | Required | Description                                    |
|--------------------|--------|----------|------------------------------------------------|
| `particle.type`    | string | yes      | Particle species                               |
| `particle.kinetic_energy_mev` | float | yes | Kinetic energy in MeV                   |
| `particle.mass_mev`| float  | yes      | Rest mass in MeV/c²                           |
| `particle.charge_e`| int    | yes      | Charge in units of elementary charge           |
| `rf_frequency_hz`  | float  | yes      | RF frequency in Hz (used in M56 matrix terms)  |


### `elements`

Ordered array of beamline elements. Each element has:

| Field           | Type   | Required | Description                                |
|-----------------|--------|----------|--------------------------------------------|
| `name`          | string | yes      | Unique element identifier                  |
| `type`          | string | yes*     | Element type (see table below)             |
| `kind`          | string | yes*     | Element kind — PALS alias for `type` (v2)  |
| `s_start_m`     | float  | yes**    | Longitudinal start position in metres      |
| `s_end_m`       | float  | yes**    | Longitudinal end position in metres        |
| `length_m`      | float  | yes      | Physical length in metres                  |
| `parameters`    | object | yes      | Type-specific parameters (see below)       |
| `polarity`      | string | cond.    | `"focusing"` or `"defocusing"` (required for `QUADRUPOLE`/`Quadrupole`) |
| `plane`         | string | cond.    | `"horizontal"` or `"vertical"` (required for `Kicker`) |
| `instrument_type` | string | cond.  | `"BPM"`, `"OTR"`, or `"SPECTROMETER"` (required for `Instrument`) |
| `aperture_m`    | float  | no       | Element aperture in metres                 |
| `fringe_fields` | object | no       | Fringe field specification                 |
| `optimization`  | object | no       | Optimization variable definition           |
| `metadata`      | object | no       | Descriptive metadata                       |

\* At least one of `type` or `kind` must be present. If both are given, `type` takes precedence.

\** `s_start_m` and `s_end_m` may be `null` for placeholder elements not placed in the lattice.


## Element Types

### Type names and aliases

Each element type has a canonical name, optional short aliases, and (in v2) PALS CamelCase aliases.

| Canonical Name  | Short alias | PALS alias (v2) | Description                      |
|-----------------|-------------|------------------|----------------------------------|
| `DRIFT`         | —           | `Drift`          | Empty drift space                |
| `QUADRUPOLE`    | `QPF`, `QPD`| `Quadrupole`     | Quadrupole magnet                |
| `DIPOLE`        | `DPH`       | `SBend`, `RBend` | Horizontal bending dipole        |
| `DIPOLE_WEDGE`  | `DPW`       | —                | Dipole wedge (FELsim-specific)   |
| `SOLENOID`      | `SOL`       | `Solenoid`       | Solenoid magnet                  |
| `RF_CAVITY`     | `RFC`       | `RFCavity`       | RF accelerating cavity           |
| `SEXTUPOLE`     | `SXT`       | `Sextupole`      | Sextupole magnet                 |
| `UNDULATOR`     | `UND`       | `Wiggler`        | Undulator / wiggler              |
| `BPM`           | —           | via `Instrument`  | Beam position monitor           |
| `OTR`           | —           | via `Instrument`  | OTR screen                      |
| `CORRECTOR_V`   | `STV`       | via `Kicker`      | Vertical corrector              |
| `CORRECTOR_H`   | `STH`       | via `Kicker`      | Horizontal corrector            |
| `SPECTROMETER`  | `SPC`       | via `Instrument`  | Spectrometer                    |
| —               | —           | `Marker`          | Zero-length marker (→ drift)    |

When using the short aliases `QPF` or `QPD`, the `polarity` field is not required. When using `QUADRUPOLE` or `Quadrupole`, the `polarity` field is required.


### Type-specific parameters

#### DRIFT

No parameters required. The `parameters` object should be empty.

#### QUADRUPOLE

| Parameter    | Type  | Required | Description                                     |
|--------------|-------|----------|-------------------------------------------------|
| `current_a`  | float | yes      | Excitation current in Amperes                   |

The field gradient is: `G × current`, where G is the gradient coefficient from `global_settings.quadrupole_gradient_coefficient_t_per_a_per_m` (default 2.694 T/A/m).

#### DIPOLE

| Parameter          | Type  | Required | Description                                     |
|--------------------|-------|----------|-------------------------------------------------|
| `bending_angle_deg`| float | yes      | Bending angle in degrees                        |
| `dipole_length_m`  | float | yes      | Effective magnetic length in metres              |
| `pole_gap_m`       | float | no       | Pole gap in metres                               |

#### DIPOLE_WEDGE

| Parameter          | Type  | Required | Description                                     |
|--------------------|-------|----------|-------------------------------------------------|
| `wedge_angle_deg`  | float | yes      | Wedge (pole face rotation) angle in degrees      |
| `dipole_angle_deg` | float | yes      | Bending angle of the associated dipole           |
| `dipole_length_m`  | float | yes      | Magnetic length of the associated dipole         |
| `pole_gap_m`       | float | yes      | Pole gap in metres                               |

A dipole with wedge pole faces is represented as: DIPOLE_WEDGE (entrance) → DIPOLE → DIPOLE_WEDGE (exit).

#### SOLENOID

| Parameter  | Type  | Required | Description                  |
|------------|-------|----------|------------------------------|
| `field_t`  | float | yes      | Axial magnetic field Bz in Tesla |

#### RF_CAVITY

| Parameter      | Type  | Required | Description                    |
|----------------|-------|----------|--------------------------------|
| `voltage_mv`   | float | yes      | Peak voltage in MV             |
| `frequency_hz` | float | yes      | RF frequency in Hz             |
| `phase_deg`    | float | yes      | RF phase in degrees            |

#### SEXTUPOLE

| Parameter  | Type  | Required | Description                       |
|------------|-------|----------|-----------------------------------|
| `strength` | float | yes      | Integrated sextupole strength     |

#### UNDULATOR

| Parameter      | Type  | Required | Description                       |
|----------------|-------|----------|-----------------------------------|
| `period_m`     | float | no       | Undulator period in metres        |
| `num_periods`  | int   | no       | Number of periods                 |
| `K_parameter`  | float | no       | Undulator K parameter             |
| `peak_field_t` | float | no       | Peak magnetic field in Tesla      |

The undulator is treated as a drift space for beam transport.


### Fringe fields

```yaml
fringe_fields:
  enge_coefficients: [56.49, -50.79, 19.32, -3.621, 0.3315, -0.01193]
```

Set to `null` or omit entirely when no Enge coefficients are available.

In the UH FEL beamline, only the MkIII chicane (FC1/FC2) dipoles carry measured Enge coefficients.


### Optimization

```yaml
optimization:
  variable: I_QF1
  bounds: [0.0, 10.0]
```


### Element metadata

```yaml
metadata:
  nomenclature: DC1.QPF.021
  element_name: Chromacity quad
  channel: 28
  label: DPHQ
  sector: DC1
```

All metadata fields are optional.


### `lattice_structure`

Groups elements into named sectors for organisational purposes.

```yaml
lattice_structure:
  sectors:
    - name: LIN
      description: Linac section
      element_names: [D_LIN_01, LQ1, BPM_LIN, LQ2, VC1]
```


### `global_settings`

```yaml
global_settings:
  coordinate_system: felsim
  length_unit: m
  angle_unit: deg
  field_unit: T
  current_unit: A
  energy_unit: MeV
  quadrupole_gradient_coefficient_t_per_a_per_m: 2.694
```


### `simulator_specific`

Optional per-simulator configuration.


## Mapping to Internal Representations

### Type mapping

| type/kind       | polarity         | Internal type | beamline.py class |
|-----------------|------------------|---------------|-------------------|
| `DRIFT` / `Drift` | —             | `DRIFT`       | `driftLattice`    |
| `QUADRUPOLE` / `Quadrupole` | `focusing` | `QPF` | `qpfLattice`  |
| `QUADRUPOLE` / `Quadrupole` | `defocusing` | `QPD` | `qpdLattice` |
| `QPF`           | —                | `QPF`         | `qpfLattice`      |
| `QPD`           | —                | `QPD`         | `qpdLattice`      |
| `DIPOLE` / `SBend` / `RBend` | — | `DPH`         | `dipole`          |
| `DPH`           | —                | `DPH`         | `dipole`          |
| `DIPOLE_WEDGE` / `DPW` | —       | `DPW`         | `dipole_wedge`    |
| `UNDULATOR` / `Wiggler` | —      | `UND`         | `driftLattice`*   |
| `Marker`        | —                | `DRIFT`       | (zero-length)     |

\* Undulators are treated as drift spaces for beam transport.


## Migration Guide: v1 → v2

v1 files load unchanged in the updated parser — no migration is required. To create v2 files:

1. Change `format_version` from `1` to `2`
2. Optionally replace `type` with `kind` using PALS names
3. Optionally use PALS element kind names (e.g. `Quadrupole` instead of `QUADRUPOLE`)

The converters support this:
```bash
# Generate v1 JSON (excelToJson default, backward-compatible)
python excelToJson.py input.xlsx output.json

# Generate v2 JSON with PALS names
python excelToJson.py input.xlsx output.json --v2

# Generate v2 YAML (excelToYaml default — YAML is a v2-era format)
python excelToYaml.py input.xlsx output.yaml
```

Note: `excelToJson` defaults to `format_version=1` for backward compatibility with existing JSON consumers. `excelToYaml` defaults to `format_version=2` since YAML support was introduced alongside v2.


## YAML Format

YAML is a first-class alternative to JSON. The structure is identical; only the serialisation differs. YAML files use the same JSON Schema for validation.

Example element in YAML:
```yaml
- name: LQ1
  type: Quadrupole
  polarity: focusing
  s_start_m: 0.358775
  s_end_m: 0.447675
  length_m: 0.0889
  parameters:
    current_a: 0.886
  aperture_m: 0.027
```


## Coordinate Systems

| System    | Coordinates                                                            |
|-----------|------------------------------------------------------------------------|
| `felsim`  | [x(mm), x'(mrad), y(mm), y'(mrad), ΔToF/T×10³, δW/W×10³]           |
| `cosy`    | [x(m), a, y(m), b, l(m), δK]                                         |
| `rftrack` | [x(mm), x'(mrad), y(mm), y'(mrad), t(mm/c), P(MeV/c)]              |
| `elegant` | [x(m), x'(rad), y(m), y'(rad), t(s), δ]                              |

The lattice file always uses SI units (metres, degrees, Tesla, Amperes). Coordinate system conversions are handled by simulator adapters at runtime.


## Design Notes

**Why separate `length_m` and `dipole_length_m`?**
For a DIPOLE, `length_m` is the physical extent along the beamline. `dipole_length_m` is the effective magnetic length for transfer matrices. They are typically equal but stated explicitly to avoid ambiguity.

**Why does DIPOLE_WEDGE carry dipole parameters?**
A wedge element's transfer matrix depends on the bending radius of the associated dipole. Carrying these parameters directly avoids fragile cross-references.

**Drift insertion:**
Drifts between elements are inferred from position gaps. Explicit DRIFT elements are optional but recommended for clarity.

**Zero-length elements:**
Diagnostic and corrector elements have zero physical length and do not contribute to beam transport but are preserved for control system integration.
