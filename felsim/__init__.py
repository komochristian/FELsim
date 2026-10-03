"""FELsim: transfer-matrix model of the UH FEL beamline with adapters to COSY INFINITY, RF-Track and Xsuite.

Submodules load on first access, so ``import felsim`` stays light and the optional
tracking codes are imported only when their adapter is used.
"""
from importlib import import_module
from importlib.util import find_spec

__version__ = "0.1.0"

_EXPORTS = {
    "beam": "ebeam",
    "lattice": "beamline",
    "PhysicalConstants": "physicalConstants",
    "BeamlineElement": "simulatorBase",
    "SimulatorBase": "simulatorBase",
    "SimulationResult": "simulatorBase",
    "SimulationMode": "simulatorBase",
    "CoordinateSystem": "simulatorBase",
    "BeamEvolution": "beamEvolution",
    "SimulatorFactory": "simulatorFactory",
    "CoordinateTransformer": "simulatorFactory",
    "create_simulator": "simulatorFactory",
    "MultiCodeSimulator": "multiCodeSimulator",
    "SimSection": "multiCodeSimulator",
    "FELsimAdapter": "felsimAdapter",
    "COSYAdapter": "cosyAdapter",
    "RFTrackAdapter": "rftrackAdapter",
    "XsuiteAdapter": "xsuiteAdapter",
    "create_beamline": "latticeLoader",
    "parse_beamline": "latticeLoader",
    "load_reference_parameters": "parameters",
    "chicane_inputs": "parameters",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    elif not name.startswith("_") and find_spec(f"{__name__}.{name}") is not None:
        value = import_module(f"{__name__}.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
