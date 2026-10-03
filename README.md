# FELsim

Start-to-end simulation framework for the University of Hawai'i FEL beamline:
thermionic RF gun, alpha magnet, S-band linac, transport line and undulator.
The native engine is a transfer-matrix model of the lattice (`beamline.py`,
`ebeam.py`); the same beamline description drives, and is benchmarked
against, COSY INFINITY, RF-Track and Xsuite through a common adapter
interface. Different codes can be chained on different sections of the line,
and quadrupole settings are optimized against Twiss, dispersion and envelope
targets. A FastAPI service and a React front end expose the model as a web
application.

## Repository layout

| Path | Contents |
|------|----------|
| `backend/` | Python modules (flat, imported with `PYTHONPATH=backend`) |
| `backend/beamline.py`, `ebeam.py` | Lattice elements, transfer matrices, beam moments and Twiss |
| `backend/simulatorBase.py`, `simulatorFactory.py` | Common simulator interface and registry |
| `backend/felsimAdapter.py`, `cosy*.py`, `rftrackAdapter.py`, `xsuiteAdapter.py` | Adapters to the four codes |
| `backend/multiCodeSimulator.py` | Chains codes section by section in FELsim coordinates |
| `backend/beamOptimizer.py`, `AlgebraicOptimization.py` | Quadrupole optimization |
| `backend/beamUtility.py`, `backend/data/estar/` | Energy deposition: beam-pipe heating and FLASH dose from NIST ESTAR tables |
| `backend/felAPI.py` | FastAPI service used by the front end |
| `backend/examples/` | Stand-alone example and the energy deposition notebook |
| `backend/test/beamline_optimization.ipynb` | Sequential optimization of the UH beamline |
| `beam_excel/Beamline_elements.xlsx` | Element table of the UH beamline (source of the lattice files) |
| `var/`, `manuals/` | Lattice files, JSON schemas and the lattice specification |
| `fields/` | Dipole fringe-field data used by COSY |
| `fel-app/` | React + Vite front end |

## Installation

Python 3.12 (`.python-version`). Create the environment with
[uv](https://docs.astral.sh/uv/) from the repository root:

```bash
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
uv pip install -r backend/requirements-local.txt
python -m ipykernel install --sys-prefix --name felsim --display-name "Python (FELsim .venv)"
```

`backend/requirements.txt` pins the simulation and API dependencies
(numpy 2.3.2, scipy 1.16.1, pandas 2.3.2, fastapi 0.116.1, ...).
`backend/requirements-local.txt` adds the tracking codes and notebook tools,
pinned to the versions verified on macOS arm64:

```bash
pip install xsuite==0.62.2 RF-Track==2.7.2
```

RF-Track is imported as `RF_Track`. Xsuite compiles its tracking kernels and
needs a C compiler (Apple Command Line Tools on macOS).

Modules are imported as flat files, so scripts and notebooks run with
`PYTHONPATH=backend` or insert `backend/` into `sys.path`.

### COSY INFINITY

COSY INFINITY 10.2 is a separate licensed download from
[cosyinfinity.org](https://www.cosyinfinity.org). `COSYSimulator` looks for
`cosy`, `cosy.fox`, `COSY.bin` and `SYSCA.DAT` in `backend/`, the working
directory, `fields/`, `/usr/local/bin` and `~/COSY/10.2/UNIX`; another
location is passed as `COSYSimulator(..., cosy_dist_dir=...)`.

```bash
mkdir -p ~/COSY/10.2/UNIX && cd ~/COSY/10.2/UNIX
tar xzf maccosy10_2_M3.tgz        # arm64 build, also for M4 and later; Intel: maccosy10_2_I.tgz
cp /path/to/cosy.fox /path/to/SYSCA.DAT .
./cosy cosy                       # compiles cosy.fox once into COSY.bin
```

The macOS binary needs the gfortran runtime (`brew install gcc`).
`SYSCA.DAT` is only required for `fringe_field_order > 0`.

## Running a simulation

Each code is created through the factory and driven with the same calls:

```python
from simulatorFactory import SimulatorFactory
from simulatorBase import BeamlineElement

sim = SimulatorFactory.create("felsim")       # or "cosy", "rftrack", "xsuite"
sim.set_beam_energy(45.0)                     # MeV
sim.set_beamline([BeamlineElement("DRIFT", length=0.3),
                  BeamlineElement("QUAD_F", length=0.1, current=0.5)])
evolution = sim.collect_evolution(particles, interval=0.02)   # particles: (N, 6) FELsim coordinates
```

FELsim coordinates are `x [mm], x' [mrad], y [mm], y' [mrad],
1000 dt/T_RF, 1000 dE/E`. Adapters convert to and from the native
coordinates of each code.

- `backend/examples/simulate_without_cosy.py`: 45 MeV beam through a
  drift-quadrupole line with the native engine; writes the envelope plot and
  CSV to `outputs/tmp/without_cosy/`. Add `--show` for an interactive window.
- `backend/test/beamline_optimization.ipynb`: loads the UH lattice from
  `beam_excel/Beamline_elements.xlsx` and optimizes the quadrupole currents
  section by section (doublets, triplets, chromaticity quadrupoles).
- `backend/examples/energy_deposition.ipynb`: stopping power, range and
  radiation yield from ESTAR; beam-pipe heating of a stainless-steel wall and
  FLASH dose in water or skin for the UH macropulse parameters.

### Lattice specification

Beamlines are described in JSON or YAML following
`manuals/lattice_specification.md`, validated against
`var/lattice_schema_v{1,2,3}.json`. The UH beamline is
`var/UH_FEL_beamline.json` / `.yaml` (converted from the Excel table with
`excelToJson.py` and `excelToYaml.py`); `var/lattice_specification.json` is
the curated PALS-aligned reference. `latticeLoader.create_beamline(path)`
selects the loader from the extension.

### Output conventions

Run outputs go under `outputs/`, which is not versioned: `outputs/tmp/` for
scratch runs that may be overwritten, `outputs/ref/` for reference runs kept
locally for code-to-code comparisons. COSY writes its `results/` directory
relative to the working directory; it is ignored as well.

## Web application

With Docker, from the repository root (needs a `.env` with
`BACKEND_API_IP=0.0.0.0`, `BACKEND_API_PORT=8000`, `FRONTEND_IP=0.0.0.0`,
`FRONTEND_PORT=5173`, `VERSION=<image tag>`):

```bash
docker compose pull
docker compose up --build
```

Without Docker, start the API from `backend/` so it finds `../.env`, then the
front end:

```bash
cd backend && uvicorn felAPI:app --host 127.0.0.1 --port 8000 --reload
cd fel-app && npm ci && npm run dev
```

The application is served at http://localhost:5173/.

## Documentation

Project notes and the physics description of the beamline:
https://docs.google.com/document/d/1dWrnrAh5NK7wX0UzH4LToEC9UzlSGCQBZpvEENmjYh4/edit?tab=t.0
