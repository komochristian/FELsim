"""COSY INFINITY optimisation stage translation helper.

Translates FELsim beamOptimizer stage definitions into COSY FIT blocks,
handling element index mapping, variable prefixing, and mirror symmetry.

Author: Eremey Valetov
"""

import pandas as pd
import numpy as np
from loggingConfig import get_logger_with_fallback

logger, _ = get_logger_with_fallback(__name__)

_PHYSICS_TYPES = {'QPF', 'QPD', 'DPH', 'DPW'}


def parse_beamline_felsim_indexed(excel_path):
    """Parse Excel beamline with FELsim-compatible element indexing.

    ExcelElements.create_beamline() and BeamlineBuilder.parse_beamline() differ
    in their handling of zero-extent diagnostic elements (BPM, OTR, STV, etc.):
    create_beamline() inserts gap drifts before them, splitting what
    parse_beamline() would produce as a single longer drift. This gives
    create_beamline() more elements and different indices.

    This function produces COSY-compatible dicts but follows create_beamline()'s
    logic, ensuring that optimisation stage indices (defined against FELsim's
    beamline) work correctly with the COSY simulator.

    Parameters
    ----------
    excel_path : str or Path
        Path to the Excel beamline file.

    Returns
    -------
    list of dict
        Beamline elements in COSY dict format with FELsim-compatible indexing.
    """
    from excelElements import ExcelElements

    excel = ExcelElements(str(excel_path))
    df = excel.get_dataframe()

    beamline = []
    prev_z_end = 0.0

    for _, row in df.iterrows():
        z_sta = row['z_start']
        z_end = row['z_end']
        element = row['Element']

        if pd.isna(z_sta) or pd.isna(z_end):
            continue

        # Gap drift (same condition as create_beamline)
        if z_sta > prev_z_end:
            beamline.append({"type": "DRIFT", "length": z_sta - prev_z_end})

        if element in _PHYSICS_TYPES:
            current = float(row['Current (A)']) if pd.notna(row['Current (A)']) else 0.0
            angle = float(row['Dipole Angle (deg)']) if pd.notna(row['Dipole Angle (deg)']) else 0.0
            wedge_angle = float(row['Dipole wedge (deg)']) if pd.notna(row['Dipole wedge (deg)']) else 0.0
            gap_wedge = float(row['Gap wedge (m)']) if pd.notna(row['Gap wedge (m)']) else 0.0
            pole_gap = float(row['Pole gap (m)']) if pd.notna(row['Pole gap (m)']) else 0.0
            enge_raw = row['Fringe Field Enge coefficients']
            enge_fct = enge_raw if pd.notna(enge_raw) else ""

            beamline.append({
                "type": element,
                "length": z_end - z_sta,
                "current": current,
                "angle": angle,
                "wedge_angle": wedge_angle,
                "gap_wedge": gap_wedge,
                "pole_gap": pole_gap,
                "enge_fct": enge_fct,
                "z_start": z_sta,
                "z_end": z_end,
            })
        elif (z_end - z_sta != 0) and not np.isnan(z_sta) and not np.isnan(z_end):
            # Non-physics element with extent → drift (matches create_beamline)
            beamline.append({"type": "DRIFT", "length": z_end - z_sta})
        # else: zero-extent diagnostic → no element added (matches create_beamline)

        if not np.isnan(z_end):
            prev_z_end = z_end

    return beamline


def add_stage(sim, stage, stage_num, index_map):
    """Add one optimisation stage to COSYSimulator as a FIT block.

    Parameters
    ----------
    sim : COSYSimulator
        COSY simulator with beamline loaded.
    stage : dict
        Stage definition with keys:
            'variables': {felsim_idx: var_name}
            'start_point': {var_name: {"start": val, "bounds": (lo, hi)}}
            'objectives': {felsim_idx: [{"measure": [...], "goal": ..., "weight": ...}]}
            'mirror': {target_felsim_idx: source_felsim_idx} (optional)
    stage_num : int
        Stage number (1-based), used to prefix variable names for uniqueness.
    index_map : dict
        Output of sim.get_element_index_mapping()['orig_to_consolidated'].
    """
    variables = stage['variables']
    start_point = stage['start_point']
    objectives = stage['objectives']
    mirror = stage.get('mirror', {})

    prefix = f"S{stage_num}_"
    var_name_map = {}  # original name → prefixed name

    # Build prefixed start point
    prefixed_start = {}
    for felsim_idx, var_name in variables.items():
        prefixed = prefix + var_name
        var_name_map[var_name] = prefixed
        if var_name in start_point:
            prefixed_start[prefixed] = start_point[var_name]

    # Apply symbolic variable names to beamline elements
    var_mapping = {}
    for felsim_idx, var_name in variables.items():
        var_mapping[felsim_idx] = {"current": var_name_map[var_name]}

    # Mirror pairs: same variable on mirrored element
    for target_idx, source_idx in mirror.items():
        if source_idx in variables:
            var_mapping[target_idx] = {"current": var_name_map[variables[source_idx]]}

    sim.apply_variable_mapping(var_mapping, validation=False)

    # Add variable initializations
    sim.set_optimization_initial_point(prefixed_start, reset=False)

    # Map objective element indices: FELsim 0-based → COSY MAP_ARR 1-based
    cosy_objectives = {}
    for felsim_idx, obj_list in objectives.items():
        if felsim_idx not in index_map:
            raise KeyError(
                f"FELsim element {felsim_idx} not found in index mapping "
                f"(max index: {max(index_map.keys())})"
            )
        map_arr_idx = index_map[felsim_idx] + 1  # 0-based consolidated → 1-based MAP_ARR
        cosy_objectives[str(map_arr_idx)] = obj_list

    sim.set_optimization_objectives(cosy_objectives, reset=False)

    logger.info(
        f"Stage {stage_num}: {len(variables)} variable(s), "
        f"{sum(len(v) for v in objectives.values())} objective(s)"
        + (f", {len(mirror)} mirror pair(s)" if mirror else "")
    )


def add_stages(sim, stages, index_map=None):
    """Add multiple optimisation stages to COSYSimulator.

    Parameters
    ----------
    sim : COSYSimulator
        COSY simulator with beamline loaded.
    stages : list of dict
        Stage definitions (see add_stage for format).
    index_map : dict, optional
        Pre-computed orig_to_consolidated mapping. Computed if None.

    Returns
    -------
    dict
        'index_map': the mapping used
        'n_stages': number of stages added
    """
    if index_map is None:
        mapping = sim.get_element_index_mapping()
        index_map = mapping['orig_to_consolidated']
        logger.info(
            f"Element mapping: {mapping['n_original']} original → "
            f"{mapping['n_consolidated']} consolidated"
        )

    sim.set_optimization_enabled(True)

    for i, stage in enumerate(stages, 1):
        add_stage(sim, stage, i, index_map)

    return {'index_map': index_map, 'n_stages': len(stages)}


def get_optimized_currents(reader, stages):
    """Extract optimized quad currents from COSY results.

    Parameters
    ----------
    reader : COSYResultsReader
        Results reader from completed simulation.
    stages : list of dict
        Stage definitions (same as passed to add_stages).

    Returns
    -------
    dict
        {felsim_idx: optimized_current_value}
    """
    variables = reader.get_variables()
    currents = {}

    for stage_num, stage in enumerate(stages, 1):
        prefix = f"S{stage_num}_"
        for felsim_idx, var_name in stage['variables'].items():
            prefixed = prefix + var_name
            if prefixed in variables:
                currents[felsim_idx] = variables[prefixed]

        # Mirror pairs get the same value as their source
        for target_idx, source_idx in stage.get('mirror', {}).items():
            if source_idx in currents:
                currents[target_idx] = currents[source_idx]

    return currents
