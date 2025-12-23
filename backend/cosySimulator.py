import subprocess
import os
import shutil
import json
from beamlineBuilder import BeamlineBuilder
from cosyResultsReader import COSYResultsReader
from physicalConstants import PhysicalConstants
from loggingConfig import get_logger_with_fallback


class COSYSimulator(BeamlineBuilder):
    def __init__(self, excel_path, json_config_path=None, config_dict=None, debug=None, use_enge_coeffs=True,
                 use_mge_for_dipoles=False):
        """Initialize COSY simulator with beamline specification and configuration."""
        super().__init__(excel_path, json_config_path)

        self.config = self._load_config(json_config_path, config_dict)
        sim_config = self.config.get('simulation', {})
        self.KE = sim_config.get('KE', 45.0)
        self.order = sim_config.get('order', 3)
        self.dimensions = sim_config.get('dimensions', 2)

        # Optimization state
        self.optimization_init_code = ""
        self.optimization_initial_point = {}
        self.optimization_objectives = []
        self.optimization_fit_blocks = []
        self.variables_used_in_fit = set()
        self.variables_available_for_next_fit = {}

        self.fit_eps = 1E-8
        self.fit_nmax = 1000
        self.fit_nalgorithm = 3
        self.optimization_enabled = False

        # Apply config if provided
        if self.config.get('optimization_initial_point'):
            self._add_optimization_initial_point(self.config['optimization_initial_point'], reset=True)
        if self.config.get('optimization_objectives'):
            self._add_optimization_objectives(self.config['optimization_objectives'], reset=True)
            self.optimization_enabled = True

        # Twiss parameter mapping
        self.MEASURE_MAP = {
            ("x", "alpha"): "A0(1)", ("y", "alpha"): "A0(2)",
            ("x", "beta"): "B0(1)", ("y", "beta"): "B0(2)",
            ("x", "gamma"): "G0(1)", ("y", "gamma"): "G0(2)"
        }

        # Particle tracking
        self.particle_tracking_mode = False
        self.particle_input_unit = 200
        self.particle_checkpoint_base_unit = 10000
        self.particle_checkpoint_elements = None
        self.particle_checkpoint_count = 0

        self.logger, self.debug = get_logger_with_fallback(__name__, debug)
        self.use_enge_coeffs = use_enge_coeffs
        self.use_mge_for_dipoles = use_mge_for_dipoles
        self.quad_aperture = 0.027
        self.dipole_aperture = 0.0127

        # Physical constants
        self.E0 = PhysicalConstants.E0_electron
        self.Q = PhysicalConstants.Q
        self.M = PhysicalConstants.M_e
        self.C = PhysicalConstants.C
        self.G = PhysicalConstants.G_quad_default

        # Energy-dependent parameters
        self.E = self.KE + self.E0
        self.P = PhysicalConstants.momentum(self.KE, self.E0)
        self.P_45 = PhysicalConstants.momentum(45, self.E0)
        self.gamma, self.beta = PhysicalConstants.relativistic_parameters(self.KE, self.E0)

        # File search paths
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.current_dir = os.getcwd()
        self.default_cosy_dir = os.path.expanduser('/usr/local/bin')
        self.search_dirs = [self.script_dir, self.current_dir, self.default_cosy_dir]

        # MGE parameters
        self.mge_array_name = "MkIIIChicaneDipoleField"
        self.mge_fieldmap_filename = "chicane_dipole_fieldmap.dat"
        self.mge_s = 1.4
        self.mge_scaling = self.P / self.P_45
        self.symbolic_variables = set()

        self.mge_initialization = """    VARIABLE {array_name} 1 1 300 ;
    VARIABLE VEC 300 1 ;
    VARIABLE MGE_FIELD 300 ;
    VARIABLE COL 1 1 ;
    VARIABLE J 1 ;
    VARIABLE NP 1 ;
    VARIABLE NS 1 ;
    VARIABLE DELTAS 1 ;
    COL(1) := 1 ;
    FILE2VE '{fieldmap_file}' 1 COL VEC ;
    NP := VEC(1)|1 ;
    NS := VEC(1)|2 ;
    DELTAS := VEC(1)|3 ;
    MGE_FIELD := VEC(1)|(4&LENGTH(VEC(1))) ;
    MGE_FIELD := MGE_FIELD * {mge_scaling} ;
    LOOP J 1 NS ;
        {array_name}(1,J) := MGE_FIELD|J ;
    ENDLOOP ;
"""

    def _build_boilerplate(self):
        """Generate COSY input boilerplate with current configuration."""
        return f"""
INCLUDE 'COSY' ;
PROCEDURE RUN ;
    VARIABLE A0 100 2 ; VARIABLE B0 100 2 ; VARIABLE G0 100 2 ;
    VARIABLE R0 100 2 ; VARIABLE MU0 100 2 ; VARIABLE F0 100 6 ;
{{Initialization}}
    PROCEDURE LATTICE ;
        UM ; CR ;
{{Elements}}
    ENDPROCEDURE ;

    OV {self.order} {self.dimensions} 0 ;
    RPE {self.KE} ;
    LATTICE ;
{{Optimization}}
    CO 1 ; PM 99 ;
    GT MAP F0 MU0 A0 B0 G0 R0 ;
    OPENF 51 'result.txt' 'UNKNOWN';
        WRITE 51 '{{';
        WRITE 51 '"spos": '&S(SPOS)&',';
        WRITE 51 '"optimization_enabled": '&S({1 if self.optimization_enabled else 0})&',' ;
        WRITE 51 '"twiss": {{' ;
           WRITE 51 '  "beta_x": "'&S(CONS(B0(1)))&'",' ;
           WRITE 51 '  "beta_y": "'&S(CONS(B0(2)))&'",' ;
           WRITE 51 '  "alpha_x": "'&S(CONS(A0(1)))&'",' ;
           WRITE 51 '  "alpha_y": "'&S(CONS(A0(2)))&'",' ;
           WRITE 51 '  "gamma_x": "'&S(CONS(G0(1)))&'",' ;
           WRITE 51 '  "gamma_y": "'&S(CONS(G0(2)))&'",' ;
           WRITE 51 '  "mu_x": "'&S(CONS(MU0(1)))&'",' ;
           WRITE 51 '  "mu_y": "'&S(CONS(MU0(2)))&'"' ;
        WRITE 51 '}}' ;
{{Variables}}
        WRITE 51 '}}';
    CLOSEF 51 ;
ENDPROCEDURE ;
RUN ;
END ;
    """

    def analyze_results(self, output_dir=None):
        """Return COSYResultsReader for parsing simulation output."""
        return COSYResultsReader(output_dir or 'results')

    def _format_cosy_value(self, value):
        return value if isinstance(value, str) else str(value)

    def _multiply_expression(self, coefficient, variable):
        """Multiply coefficient by variable, handling symbolic expressions."""
        if isinstance(variable, str):
            if coefficient == 1.0:
                return variable
            elif coefficient == -1.0:
                return f"-{variable}"
            else:
                return f"{coefficient}*{variable}"
        return coefficient * variable

    def _load_config(self, json_path, config_dict):
        if config_dict is not None:
            return config_dict
        elif json_path is not None:
            with open(json_path, 'r') as f:
                return json.load(f)
        return {}

    def _get_all_cosy_variables(self):
        """Union of variables from beamline elements and optimization config."""
        all_vars = set(self.symbolic_variables)
        all_vars.update(self.optimization_initial_point.keys())
        return all_vars

    def _extract_symbolic_variables(self, expression):
        """Extract variable names from symbolic expression."""
        if not isinstance(expression, str):
            return set()

        import re
        variables = re.findall(r'[A-Za-z][A-Za-z0-9_]*', expression)
        excluded = {'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'E', 'PI'}
        variables = [v for v in variables if v not in excluded]

        if variables and self.debug:
            self.logger.debug(f"Extracted variables from '{expression}': {variables}")

        return set(variables)

    def _generate_variable_declarations(self):
        """Generate COSY VARIABLE declarations for tracked variables."""
        all_vars = self._get_all_cosy_variables()

        if not all_vars:
            declarations = ""
        else:
            declarations = "".join(f"    VARIABLE {var} 1 ;\n" for var in sorted(all_vars))

        if self.optimization_enabled:
            declarations += """    VARIABLE MAP_ARR 6 6 200 ;
    VARIABLE MAP_IDX 1 ;
    VARIABLE IDX 1 ;
    VARIABLE NOM 1 ;
    VARIABLE MAP_TMP 6 6 ;
    VARIABLE OBJ 1 100 ;
"""
            if self.debug:
                self.logger.debug("Added optimization variables: MAP_ARR, MAP_IDX, IDX, NOM, MAP_TMP, OBJ")

        if self.debug:
            self.logger.debug(f"Generated declarations for {len(all_vars)} variable(s)")

        return declarations

    def _validate_variable_consistency(self):
        """Check for variables used in beamline but not initialized in optimization."""
        beamline_vars = self.symbolic_variables
        opt_vars = set(self.optimization_initial_point.keys())

        unused_opt = opt_vars - beamline_vars
        if unused_opt:
            self.logger.warning(f"Optimization variables not used in beamline: {sorted(unused_opt)}")

        uninitialized = beamline_vars - opt_vars
        if uninitialized:
            self.logger.warning(
                f"Symbolic variables without initialization: {sorted(uninitialized)}. "
                f"Consider adding to optimization_initial_point."
            )

    def _check_duplicate_variables(self, cosy_code):
        """Verify no duplicate VARIABLE declarations in generated code."""
        import re
        from collections import Counter

        matches = re.findall(r'\s*VARIABLE\s+([A-Za-z][A-Za-z0-9_]*)\s+', cosy_code, re.IGNORECASE)
        if not matches:
            return

        duplicates = [v for v, cnt in Counter(matches).items() if cnt > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate VARIABLE declarations: {', '.join(sorted(duplicates))}. "
                f"Each variable can only be declared once."
            )

        if self.debug:
            self.logger.debug(f"Variable check passed: {len(matches)} unique variables")

    def _find_file(self, filename):
        """Search for file in configured directories."""
        for directory in self.search_dirs:
            filepath = os.path.join(directory, filename)
            if os.path.exists(filepath):
                if self.debug:
                    self.logger.debug(f"Found {filename} in {directory}")
                return filepath
        return None

    def _reset_optimization(self):
        self.optimization_initial_point = {}
        self.variables_available_for_next_fit = {}
        if self.debug:
            self.logger.debug("Reset optimization initialization")

    def _add_optimization_initial_point(self, initial_point_config, reset=False):
        """
        Add variable initialization for optimization.

        Note: COSY doesn't support bounds directly, but we store them for external optimizers.
        Variables are assigned immediately before their FIT block.
        """
        if reset:
            self._reset_optimization()

        for var_name, config in initial_point_config.items():
            if not isinstance(var_name, str) or not var_name.strip():
                raise ValueError(f"Invalid variable name: {var_name}")

            start = config.get('start', 0)
            bounds = config.get('bounds', None)

            self.optimization_initial_point[var_name] = {'start': start, 'bounds': bounds}
            self.variables_available_for_next_fit[var_name] = f"    {var_name} := {start} ;\n"

            if self.debug:
                bounds_str = f" (bounds: {bounds})" if bounds else ""
                self.logger.debug(f"Added optimization variable: {var_name} = {start}{bounds_str}")

    def set_optimization_enabled(self, enabled):
        """Enable/disable optimization code generation."""
        old = self.optimization_enabled
        self.optimization_enabled = enabled

        status = "enabled" if enabled else "disabled"
        self.logger.info(f"Optimization {status}")

        if not enabled and (self.optimization_objectives or self.optimization_initial_point):
            self.logger.warning("Optimization configuration preserved but will not be used in generated code")

        return {
            "optimization_enabled": enabled,
            "previous_state": old,
            "has_objectives": len(self.optimization_objectives) > 0,
            "has_initial_point": len(self.optimization_initial_point) > 0
        }

    def is_optimization_enabled(self):
        return self.optimization_enabled

    def set_optimization_initial_point(self, initial_point_config, reset=True):
        """
        Configure optimization initial point and bounds.

        Format: {var_name: {"start": value, "bounds": (min, max)}}
        """
        if not isinstance(initial_point_config, dict):
            raise TypeError("initial_point_config must be a dictionary")

        if not initial_point_config:
            self.logger.warning("No optimization variables provided")
            return {"variables_added": [], "reset": reset, "total_variables": 0}

        self._add_optimization_initial_point(initial_point_config, reset=reset)

        if 'optimization_initial_point' not in self.config:
            self.config['optimization_initial_point'] = {}

        if reset:
            self.config['optimization_initial_point'] = dict(self.optimization_initial_point)
        else:
            self.config['optimization_initial_point'].update(self.optimization_initial_point)

        var_names = list(initial_point_config.keys())
        self.logger.info(
            f"{'Set' if reset else 'Added'} optimization initial point for "
            f"{len(var_names)} variable(s): {', '.join(var_names)}"
        )

        return {
            "variables_added": var_names,
            "reset": reset,
            "total_variables": len(self.optimization_initial_point)
        }

    def get_optimization_initial_point(self):
        return dict(self.optimization_initial_point)

    def _reset_optimization_objectives(self):
        self.optimization_objectives = []
        self.optimization_fit_blocks = []
        self.variables_used_in_fit = set()
        if self.debug:
            self.logger.debug("Reset optimization objectives and FIT blocks")

    def _validate_measure(self, measure):
        """Validate and normalize measure specification like ["x", "alpha"]."""
        if not isinstance(measure, (list, tuple)) or len(measure) != 2:
            raise ValueError(f"Invalid measure format: {measure}. Expected [axis, parameter]")

        axis, param = measure[0].lower(), measure[1].lower()
        if (axis, param) not in self.MEASURE_MAP:
            raise ValueError(f"Invalid measure ({axis}, {param}). Valid: {list(self.MEASURE_MAP.keys())}")

        return axis, param

    def _generate_objective_expression(self, objective):
        """Generate COSY objective expression from objective dict."""
        measure = self._validate_measure(objective["measure"])
        goal = objective["goal"]
        weight = objective.get("weight", 1)
        cosy_var = self.MEASURE_MAP[measure]
        return f"{weight}*ABS(CONS({cosy_var}) - {goal})"

    def _add_optimization_objectives(self, objectives_config, reset=False):
        """
        Add optimization objectives and generate FIT block.

        Multiple calls with reset=False create multiple FIT blocks (sequential optimization).
        Each FIT block uses variables not yet assigned to previous blocks.
        """
        if reset:
            self._reset_optimization_objectives()

        # Extract optimizer settings
        settings = objectives_config.get("optimizer_settings", {})
        eps = settings.get("eps", self.fit_eps)
        nmax = settings.get("Nmax", self.fit_nmax)
        nalg = settings.get("Nalgorithm", self.fit_nalgorithm)

        if settings:
            self.fit_eps = eps
            self.fit_nmax = nmax
            self.fit_nalgorithm = nalg

            if self.debug:
                self.logger.debug(
                    f"FIT optimizer settings: eps={eps}, Nmax={nmax}, Nalgorithm={nalg}"
                )

        # Group objectives by element, excluding special keys
        objectives_by_elem = {}
        for elem_num, obj_list in objectives_config.items():
            if elem_num == "optimizer_settings":
                continue
            if not isinstance(obj_list, list):
                raise ValueError(f"Objectives for element {elem_num} must be list")

            elem_int = int(elem_num)
            objectives_by_elem.setdefault(elem_int, []).extend(obj_list)

        if not objectives_by_elem:
            if self.debug:
                self.logger.debug("No objectives provided")
            return

        # Store objectives for this FIT block
        fit_objectives = []
        for elem_num, obj_list in objectives_by_elem.items():
            for obj in obj_list:
                fit_objectives.append({
                    "element": elem_num,
                    "measure": obj["measure"],
                    "goal": obj["goal"],
                    "weight": obj.get("weight", 1)
                })
            self.optimization_objectives.extend(obj_list)

        # Variables for this FIT block
        fit_vars = set(self.variables_available_for_next_fit.keys())
        if not fit_vars:
            self.logger.warning("No new variables available for FIT block")
            return

        # Build init code
        init_code = "".join(self.variables_available_for_next_fit[v] for v in sorted(fit_vars))
        self.variables_used_in_fit.update(fit_vars)
        self.variables_available_for_next_fit.clear()

        # Generate FIT block
        fit_vars_str = " ".join(sorted(fit_vars))
        fit_body = ""
        obj_refs = []
        obj_idx = 1

        for elem_num in sorted(objectives_by_elem.keys()):
            fit_body += f"        LOOP IDX 1 6 ; MAP_TMP(IDX) := MAP_ARR(IDX, {elem_num}) ; ENDLOOP ;\n"
            fit_body += f"        GT MAP_TMP F0 MU0 A0 B0 G0 R0 ;\n"

            for obj in objectives_by_elem[elem_num]:
                expr = self._generate_objective_expression(obj)
                fit_body += f"        OBJ({obj_idx}) := {expr} ;\n"
                obj_refs.append(f"OBJ({obj_idx})")

                if self.debug:
                    measure_str = f"{obj['measure'][0]}.{obj['measure'][1]}"
                    self.logger.debug(
                        f"Added objective {obj_idx}: {measure_str} → {obj['goal']} "
                        f"(weight={obj.get('weight', 1)}) at element {elem_num}"
                    )

                obj_idx += 1

        obj_list_str = " ".join(obj_refs)
        fit_code = f"""    FIT {fit_vars_str} ;
            LATTICE ;
    {fit_body}    ENDFIT {eps} {nmax} {nalg} {obj_list_str} ;
    """

        self.optimization_fit_blocks.append({
            "init_code": init_code,
            "fit_code": fit_code,
            "variables": sorted(fit_vars),
            "objectives": fit_objectives,
            "optimizer_settings": {"eps": eps, "Nmax": nmax, "Nalgorithm": nalg}
        })

        self.optimization_enabled = True

        if self.debug:
            self.logger.debug(
                f"Created FIT block #{len(self.optimization_fit_blocks)} with "
                f"{len(fit_vars)} variable(s) and {len(obj_refs)} objective(s)"
            )
            if len(self.optimization_fit_blocks) == 1:
                self.logger.debug("Optimization automatically enabled")

    def set_optimization_objectives(self, objectives_config, reset=True):
        """
        Configure optimization objectives.

        Format: {
            element_num: [{"measure": [axis, param], "goal": value, "weight": value}],
            "optimizer_settings": {"eps": float, "Nmax": int, "Nalgorithm": int}  # optional
        }

        Settings: eps (tolerance), Nmax (max evaluations, 0=no optimization), Nalgorithm (algorithm)
        """
        if not isinstance(objectives_config, dict):
            raise TypeError("objectives_config must be a dictionary")

        if not objectives_config:
            self.logger.debug("No optimization objectives provided")
            return {"objectives_added": 0, "num_fit_blocks": 0, "reset": reset}

        self._add_optimization_objectives(objectives_config, reset=reset)

        if 'optimization_objectives' not in self.config:
            self.config['optimization_objectives'] = {}

        if reset:
            self.config['optimization_objectives'] = dict(objectives_config)
        else:
            for elem_num, obj_list in objectives_config.items():
                if elem_num == "optimizer_settings":
                    self.config['optimization_objectives']["optimizer_settings"] = obj_list
                elif elem_num in self.config['optimization_objectives']:
                    self.config['optimization_objectives'][elem_num].extend(obj_list)
                else:
                    self.config['optimization_objectives'][elem_num] = obj_list

        total_objectives = len(self.optimization_objectives)
        num_blocks = len(self.optimization_fit_blocks)

        self.logger.info(
            f"{'Set' if reset else 'Added'} {total_objectives} optimization objective(s) "
            f"across {num_blocks} FIT block(s)"
        )

        return {
            "objectives_added": total_objectives,
            "num_fit_blocks": num_blocks,
            "reset": reset,
            "variables_in_fit": sorted(self.variables_used_in_fit),
            "fit_blocks": [
                {
                    "block_number": i + 1,
                    "variables": block["variables"],
                    "optimizer_settings": block["optimizer_settings"]
                }
                for i, block in enumerate(self.optimization_fit_blocks)
            ]
        }

    def get_optimization_objectives(self):
        fit_blocks_info = [
            {
                "block_number": i + 1,
                "variables": block["variables"],
                "objectives": block["objectives"],
                "optimizer_settings": block["optimizer_settings"]
            }
            for i, block in enumerate(self.optimization_fit_blocks)
        ]

        return {
            "objectives": list(self.optimization_objectives),
            "num_fit_blocks": len(self.optimization_fit_blocks),
            "fit_blocks": fit_blocks_info,
            "variables_in_fit": sorted(self.variables_used_in_fit),
            "default_optimizer_settings": {
                "eps": self.fit_eps,
                "Nmax": self.fit_nmax,
                "Nalgorithm": self.fit_nalgorithm
            }
        }

    def enable_particle_tracking(self, input_unit=200, checkpoint_base_unit=10000,
                                 checkpoint_elements=None):
        """
        Enable particle tracking with RRAY/WRAY commands.

        Reads initial distribution from fort.{input_unit} and saves checkpoints
        to fort.{checkpoint_base_unit + element_index}.

        checkpoint_elements: List of 1-based indices where WRAY is called, or None for all elements.
        """
        self.particle_tracking_mode = True
        self.particle_input_unit = input_unit
        self.particle_input_file = f'fort.{input_unit}'
        self.particle_checkpoint_base_unit = checkpoint_base_unit
        self.particle_checkpoint_elements = checkpoint_elements
        self.particle_checkpoint_count = 0

        if self.debug:
            mode = "ALL elements" if checkpoint_elements is None else f"{len(checkpoint_elements)} elements"
            self.logger.debug(
                f"Particle tracking enabled: fort.{input_unit} → fort.{checkpoint_base_unit}+N ({mode})"
            )

        return {
            'particle_tracking_mode': True,
            'input_unit': input_unit,
            'checkpoint_base_unit': checkpoint_base_unit,
            'checkpoint_elements': checkpoint_elements,
            'checkpoint_mode': 'all' if checkpoint_elements is None else 'selective'
        }

    def disable_particle_tracking(self):
        old_state = self.particle_tracking_mode
        self.particle_tracking_mode = False
        self.particle_checkpoint_elements = None

        if self.debug:
            self.logger.debug("Disabled particle tracking mode")

        return {
            'particle_tracking_mode': False,
            'previous_state': old_state
        }

    def get_particle_tracking_config(self):
        """Return current particle tracking configuration."""
        config = {
            'particle_tracking_mode': self.particle_tracking_mode,
            'input_unit': self.particle_input_unit,
            'checkpoint_base_unit': self.particle_checkpoint_base_unit,
            'checkpoint_elements': self.particle_checkpoint_elements,
            'checkpoint_mode': 'all' if self.particle_checkpoint_elements is None else 'selective',
            'checkpoints_written': self.particle_checkpoint_count,
            'input_file': f'fort.{self.particle_input_unit}',
            'checkpoint_files': []
        }

        if self.particle_tracking_mode and self.particle_checkpoint_count > 0:
            if self.particle_checkpoint_elements is None:
                config['checkpoint_files'] = [
                    f'fort.{self.particle_checkpoint_base_unit + i}'
                    for i in range(1, self.particle_checkpoint_count + 1)
                ]
            else:
                config['checkpoint_files'] = [
                    f'fort.{self.particle_checkpoint_base_unit + i}'
                    for i in self.particle_checkpoint_elements
                ]

        return config

    def _add_particle_checkpoint(self, elements_str, element_idx):
        """Add WRAY command after element if tracking enabled and element selected."""
        if not self.particle_tracking_mode:
            return elements_str

        should_save = (self.particle_checkpoint_elements is None or
                       element_idx in self.particle_checkpoint_elements)

        if should_save:
            unit = self.particle_checkpoint_base_unit + element_idx
            elements_str += f"    WRAY {unit} ;\n"
            self.particle_checkpoint_count += 1

            if self.debug and self.particle_checkpoint_count <= 5:
                self.logger.debug(f"Adding checkpoint: WRAY {unit} after element {element_idx}")

        return elements_str

    def _parse_enge_coefficients(self, enge_data):
        """Parse Enge coefficients from string/list/single value."""
        if isinstance(enge_data, list):
            return enge_data
        elif isinstance(enge_data, str) and enge_data.strip():
            try:
                coeffs = [float(c.strip()) for c in enge_data.split(',') if c.strip()]
                return coeffs if coeffs else None
            except ValueError:
                return None
        return None

    def _format_enge_coefficients(self, coeffs):
        """Ensure exactly 6 Enge coefficients, padding/truncating as needed."""
        if not isinstance(coeffs, list):
            return [0.0] * 6
        return (coeffs + [0.0] * 6)[:6]

    def _get_fieldmap_parameters(self, output_dir):
        """Read NS and DELTAS from fieldmap file to calculate length."""
        fieldmap_path = os.path.join(output_dir, self.mge_fieldmap_filename)

        if not os.path.exists(fieldmap_path):
            found = self._find_file(self.mge_fieldmap_filename)
            if not found:
                if self.debug:
                    self.logger.debug(
                        f"Fieldmap file {self.mge_fieldmap_filename} not found in any search directory"
                    )
                return None, None, None
            fieldmap_path = found

        try:
            with open(fieldmap_path, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 3:
                    np_val = float(lines[0].strip())
                    ns_val = int(lines[1].strip())
                    deltas_val = float(lines[2].strip())
                    fieldmap_length = ns_val * deltas_val

                    if self.debug:
                        self.logger.debug(
                            f"Fieldmap parameters: NS={ns_val}, DELTAS={deltas_val}, Length={fieldmap_length}"
                        )

                    return ns_val, deltas_val, fieldmap_length
                else:
                    self.logger.error(f"Fieldmap file has insufficient data")
                    return None, None, None
        except Exception as e:
            self.logger.error(f"Error reading fieldmap: {e}")
            return None, None, None

    def _get_enge_sign(self, enge_str):
        """Return sign of first Enge coefficient: 1=entrance, -1=exit, 0=none."""
        if isinstance(enge_str, (int, float)):
            return 1 if enge_str > 0 else -1 if enge_str < 0 else 0
        elif isinstance(enge_str, list):
            return 1 if enge_str and enge_str[0] > 0 else -1 if enge_str and enge_str[0] < 0 else 0
        elif isinstance(enge_str, str) and enge_str.strip():
            try:
                coeffs = [float(c.strip()) for c in enge_str.split(',') if c.strip()]
                return 1 if coeffs and coeffs[0] > 0 else -1 if coeffs and coeffs[0] < 0 else 0
            except ValueError:
                pass
        return 0

    def _detect_dipole_triplets(self):
        """Detect and consolidate DPW-DPH-DPW triplets into single dipole units."""
        grouped = []
        i = 0

        while i < len(self.beamline):
            elem = self.beamline[i]

            # Check for triplet pattern
            if (elem['type'] == 'DPW' and i + 2 < len(self.beamline) and
                    self.beamline[i + 1]['type'] == 'DPH' and
                    self.beamline[i + 2]['type'] == 'DPW'):

                entrance, main, exit_wedge = self.beamline[i:i + 3]

                # Parse Enge coefficients by sign
                entrance_enge = exit_enge = None

                for wedge in [entrance, exit_wedge]:
                    raw = wedge.get('enge_fct', 0)
                    parsed = self._parse_enge_coefficients(raw)
                    if parsed:
                        if self.debug:
                            self.logger.debug(f"Parsed DPW Enge coefficients: {raw} → {parsed}")
                        sign = self._get_enge_sign(parsed)
                        if sign > 0:
                            entrance_enge = parsed
                        elif sign < 0:
                            exit_enge = parsed

                consolidated = {
                    'type': 'DIPOLE_CONSOLIDATED',
                    'length': main.get('length', 0.0889),
                    'angle': main.get('angle', 0.0),
                    'entrance_angle': entrance.get('wedge_angle', 0) if self._get_enge_sign(
                        entrance.get('enge_fct', 0)) > 0 else 0,
                    'exit_angle': exit_wedge.get('wedge_angle', 0) if self._get_enge_sign(
                        exit_wedge.get('enge_fct', 0)) < 0 else 0,
                    'pole_gap': main.get('pole_gap', self.dipole_aperture),
                    'entrance_enge_coeffs': entrance_enge,
                    'exit_enge_coeffs': exit_enge,
                    'original_elements': [entrance, main, exit_wedge]
                }

                grouped.append(consolidated)
                i += 3
            else:
                grouped.append(elem)
                i += 1

        return grouped

    def _check_for_mge_dipoles(self, grouped_elements):
        """Check if any dipoles will use MGE mode."""
        if not (self.use_mge_for_dipoles and self.use_enge_coeffs):
            return False

        return any(
            elem['type'] == "DIPOLE_CONSOLIDATED" and
            (elem.get('entrance_enge_coeffs') or elem.get('exit_enge_coeffs'))
            for elem in grouped_elements
        )

    def update_simulation_config(self, **kwargs):
        """Update simulation parameters (KE, order, dimensions)."""
        valid = {'KE', 'order', 'dimensions'}
        invalid = set(kwargs.keys()) - valid
        if invalid:
            raise ValueError(f"Invalid parameters: {invalid}. Valid: {valid}")

        changes = {}
        for key, new_val in kwargs.items():
            old_val = getattr(self, key)
            if old_val != new_val:
                changes[key] = {'old': old_val, 'new': new_val}
                setattr(self, key, new_val)
                self.config.setdefault('simulation', {})[key] = new_val

        if 'KE' in changes:
            self._recalculate_energy_dependent_parameters()
            if self.debug:
                self.logger.debug(f"Recalculated for KE={self.KE} MeV: γ={self.gamma:.6f}, β={self.beta:.6f}")

        if changes:
            self.logger.info(f"Updated simulation config: {list(changes.keys())}")

        return changes

    def _recalculate_energy_dependent_parameters(self):
        """Recalculate derived quantities after energy change."""
        self.E = self.KE + self.E0
        self.P = PhysicalConstants.momentum(self.KE, self.E0)
        self.gamma, self.beta = PhysicalConstants.relativistic_parameters(self.KE, self.E0)
        self.mge_scaling = self.P / self.P_45

    # FELsim beamline.py compatibility
    def setE(self, kinetic_energy):
        self.update_simulation_config(KE=kinetic_energy)
        return self

    def setMQE(self, mass, charge, rest_energy):
        self.M, self.Q, self.E0 = mass, charge, rest_energy
        self._recalculate_energy_dependent_parameters()
        return self

    def changeBeamType(self, particle_type, kinetic_energy):
        particle_props = PhysicalConstants.parse_particle_specification(particle_type)
        self.setMQE(particle_props["mass"], particle_props["charge"], particle_props["rest_energy"])
        self.setE(kinetic_energy)
        return self

    def update_config(self, config_dict=None, simulation=None, variable_mapping=None,
                      optimization_initial_point=None, optimization_objectives=None):
        """Update simulation configuration, variable mappings, and/or optimization settings."""

        results = {
            "simulation_changes": {},
            "variable_mapping_results": None,
            "optimization_initial_point_results": None,
            "optimization_objectives_results": None,
            "total_changes": 0
        }

        # Parse inputs
        if config_dict:
            sim_params = config_dict.get('simulation', {})
            var_map = config_dict.get('variable_mapping', {})
            opt_init = config_dict.get('optimization_initial_point', {})
            opt_obj = config_dict.get('optimization_objectives', {})
        else:
            sim_params = simulation or {}
            var_map = variable_mapping or {}
            opt_init = optimization_initial_point or {}
            opt_obj = optimization_objectives or {}

        if sim_params:
            sim_changes = self.update_simulation_config(**sim_params)
            results["simulation_changes"] = sim_changes
            results["total_changes"] += len(sim_changes)

        if var_map:
            var_map_int = {int(k) if isinstance(k, str) else k: v for k, v in var_map.items()}
            var_results = self.apply_variable_mapping(var_map_int)
            results["variable_mapping_results"] = var_results
            results["total_changes"] += var_results["summary"]["successful"]

        if opt_init:
            opt_init_results = self.set_optimization_initial_point(opt_init, reset=True)
            results["optimization_initial_point_results"] = opt_init_results
            results["total_changes"] += opt_init_results["total_variables"]

        if opt_obj:
            opt_obj_results = self.set_optimization_objectives(opt_obj, reset=True)
            results["optimization_objectives_results"] = opt_obj_results
            results["total_changes"] += opt_obj_results["objectives_added"]

        # Update internal config
        if sim_params:
            self.config.setdefault('simulation', {}).update(sim_params)
        if var_map:
            self.config.setdefault('variable_mapping', {}).update(var_map)

        if results["total_changes"] > 0:
            self.logger.info(f"Configuration update: {results['total_changes']} change(s)")

        return results

    def get_full_config(self):
        """Return complete current configuration including computed values."""
        fit_blocks = [
            {
                "block_number": i + 1,
                "variables": b["variables"],
                "num_objectives": len(b["objectives"]),
                "elements": sorted(set(o["element"] for o in b["objectives"])),
                "optimizer_settings": b["optimizer_settings"]
            }
            for i, b in enumerate(self.optimization_fit_blocks)
        ]

        return {
            'simulation': {'KE': self.KE, 'order': self.order, 'dimensions': self.dimensions},
            'variable_mapping': self.config.get('variable_mapping', {}),
            'optimization_initial_point': dict(self.optimization_initial_point),
            'optimization_objectives': self.config.get('optimization_objectives', {}),
            'optimization_enabled': self.optimization_enabled,
            'computed': {
                'gamma': self.gamma,
                'beta': self.beta,
                'momentum': self.P,
                'mge_scaling': self.mge_scaling,
                'num_fit_blocks': len(self.optimization_fit_blocks),
                'fit_blocks': fit_blocks,
                'all_variables_in_fit': sorted(self.variables_used_in_fit),
                'default_optimizer_settings': {
                    'eps': self.fit_eps, 'Nmax': self.fit_nmax, 'Nalgorithm': self.fit_nalgorithm
                }
            }
        }

    def _generate_variable_output_code(self):
        """Generate COSY WRITE statements for variable values in JSON output."""
        all_vars = self._get_all_cosy_variables()
        if not all_vars:
            return ""

        code = "        WRITE 51 ',' ;\n        WRITE 51 '\"variables\": {' ;\n"
        sorted_vars = sorted(all_vars)
        for i, var in enumerate(sorted_vars):
            comma = ',' if i < len(sorted_vars) - 1 else ''
            code += f"        WRITE 51 '  \"{var}\": \"'&S(CONS({var}))&'\"{comma}' ;\n"
        code += "        WRITE 51 '}' ;\n"

        if self.debug:
            self.logger.debug(f"Generated variable output for {len(all_vars)} variable(s)")

        return code

    def _add_map_tracking_code(self, elements_str):
        """Add map tracking after element if optimization enabled."""
        if self.optimization_enabled:
            elements_str += "    MAP_IDX := MAP_IDX + 1 ; NOM := NOC ; CO 1 ;\n"
            elements_str += "    LOOP IDX 1 6 ; MAP_ARR(IDX, MAP_IDX) := MAP(IDX) ; ENDLOOP ;\n"
            elements_str += "    CO NOM ;\n"
        return elements_str

    def generate_input(self, output_dir='results'):
        """Generate COSY input file from beamline specification."""
        os.makedirs(output_dir, exist_ok=True)

        # Apply variable mapping
        var_mapping = self.config.get('variable_mapping', {})
        if var_mapping:
            var_mapping = {int(k): v for k, v in var_mapping.items()}
            self.apply_variable_mapping(var_mapping)

        grouped_elements = self._detect_dipole_triplets()
        needs_mge = self._check_for_mge_dipoles(grouped_elements)
        fieldmap_params = self._get_fieldmap_parameters(output_dir) if needs_mge else None

        elements_str = "    MAP_IDX := 0 ;\n" if self.optimization_enabled else ""

        if self.particle_tracking_mode:
            elements_str += f"    RRAY {self.particle_input_unit} ;\n"
            if self.debug:
                mode = "all" if self.particle_checkpoint_elements is None else f"{len(self.particle_checkpoint_elements)}"
                self.logger.debug(
                    f"Particle tracking: fort.{self.particle_input_unit} → "
                    f"fort.{self.particle_checkpoint_base_unit}+N ({mode} elements)"
                )

        element_idx = 0
        self.particle_checkpoint_count = 0

        for elem in grouped_elements:
            element_idx += 1

            if elem['type'] == "DRIFT":
                elements_str += f"    DL {elem['length']} ;\n"
                elements_str = self._add_particle_checkpoint(elements_str, element_idx)
                elements_str = self._add_map_tracking_code(elements_str)

            elif elem['type'] in ["QPF", "QPD"]:
                sign = -1 if elem['type'] == "QPF" else 1
                current = elem['current']
                gradient = self._multiply_expression(self.G, current)
                radius = self.quad_aperture / 2

                if isinstance(gradient, str):
                    b_pole = self._multiply_expression(-radius if sign == -1 else radius, gradient)
                    self.symbolic_variables.update(self._extract_symbolic_variables(b_pole))
                else:
                    b_pole = sign * gradient * radius

                b_pole_str = self._format_cosy_value(b_pole)
                elements_str += f"    MQ {elem['length']} {b_pole_str} {radius} ;\n"
                elements_str = self._add_particle_checkpoint(elements_str, element_idx)
                elements_str = self._add_map_tracking_code(elements_str)

            elif elem['type'] == "DIPOLE_CONSOLIDATED":
                d_half = elem['pole_gap'] / 2
                has_enge = elem.get('entrance_enge_coeffs') or elem.get('exit_enge_coeffs')

                if self.use_mge_for_dipoles and has_enge and self.use_enge_coeffs:
                    # MGE mode
                    if self.debug:
                        self.logger.debug("Using MGE command for dipole with Enge coefficients")

                    use_cb = elem['angle'] < 0
                    if use_cb:
                        elements_str += "    CB ;\n"

                    # Handle length mismatch
                    if fieldmap_params and fieldmap_params[2]:
                        _, _, fieldmap_len = fieldmap_params
                        mismatch = elem['length'] - fieldmap_len
                        if abs(mismatch) > 1e-6:
                            drift = mismatch / 2
                            if self.debug:
                                self.logger.debug(f"Adding compensating drifts: {drift} m")
                            elements_str += f"    DL {drift} ;\n"

                    elements_str += f"    MGE NP {self.mge_array_name} NS DELTAS {self.mge_s} {d_half} ;\n"

                    if fieldmap_params and fieldmap_params[2] and abs(mismatch) > 1e-6:
                        elements_str += f"    DL {drift} ;\n"

                    if use_cb:
                        elements_str += "    CB ;\n"

                else:
                    # Standard FC/DIL mode
                    e1, e2 = elem['entrance_angle'], elem['exit_angle']
                    has_fc = False

                    if self.use_enge_coeffs and not self.use_mge_for_dipoles:
                        if elem.get('entrance_enge_coeffs'):
                            coeffs = self._format_enge_coefficients(elem['entrance_enge_coeffs'])
                            if self.debug:
                                self.logger.debug(f"Adding entrance Enge coefficients: {coeffs}")
                            elements_str += f"    FC 1 1 1 {' '.join(map(str, coeffs))} ;\n"
                            has_fc = True
                        if elem.get('exit_enge_coeffs'):
                            coeffs = self._format_enge_coefficients(elem['exit_enge_coeffs'])
                            if self.debug:
                                self.logger.debug(f"Adding exit Enge coefficients: {coeffs}")
                            elements_str += f"    FC 1 2 1 {' '.join(map(str, coeffs))} ;\n"
                            has_fc = True
                    elif self.debug and (elem.get('entrance_enge_coeffs') or elem.get('exit_enge_coeffs')):
                        self.logger.debug("Enge coefficients available but not using FC commands")

                    use_cb = elem['angle'] < 0
                    if use_cb:
                        elements_str += "    CB ;\n"

                    elements_str += f"    DIL {elem['length']} {abs(elem['angle'])} {d_half} {e1} 0 {e2} 0 ;\n"

                    if use_cb:
                        elements_str += "    CB ;\n"

                    if has_fc:
                        if self.debug:
                            self.logger.debug("Resetting fringe field coefficients (FD)")
                        elements_str += "    FD ;\n"

                elements_str = self._add_particle_checkpoint(elements_str, element_idx)
                elements_str = self._add_map_tracking_code(elements_str)

        self._validate_variable_consistency()
        var_declarations = self._generate_variable_declarations()
        var_output = self._generate_variable_output_code()

        if needs_mge:
            mge_init = self.mge_initialization.format(
                array_name=self.mge_array_name,
                fieldmap_file=self.mge_fieldmap_filename,
                mge_scaling=self.mge_scaling
            )
            full_init = var_declarations + mge_init
        else:
            full_init = var_declarations

        full_input = self._build_boilerplate().replace("{Initialization}", full_init)
        full_input = full_input.replace("{Elements}", elements_str.rstrip())

        # Optimization code
        opt_code = ""
        if self.optimization_enabled:
            for i, block in enumerate(self.optimization_fit_blocks):
                if i > 0:
                    opt_code += "\n"
                opt_code += block["init_code"].rstrip() + "\n" + block["fit_code"].rstrip()

            if self.debug:
                self.logger.debug(
                    f"Optimization enabled: {len(self.optimization_fit_blocks)} FIT block(s)"
                )
        else:
            if self.debug:
                self.logger.debug("Optimization disabled")

        full_input = full_input.replace("{Optimization}", opt_code)
        full_input = full_input.replace("{Variables}", var_output.rstrip())

        self._check_duplicate_variables(full_input)

        input_file = os.path.join(output_dir, 'input.fox')
        with open(input_file, 'w') as f:
            f.write(full_input)

        if self.debug:
            self.logger.debug(f"Generated input with {len(self.symbolic_variables)} symbolic variable(s)")

        return input_file

    def _parse_cosy_errors(self, output_text):
        """Parse COSY output for error messages (handles both 'OCCURED' typo and 'OCCURRED')."""
        errors = []
        lines = output_text.split('\n')

        for i, line in enumerate(lines):
            if "ERROR OCCURRED IN .LIS LINE" in line or "ERROR OCCURED IN .LIS LINE" in line:
                try:
                    parts = line.split("LINE")
                    line_num = int(parts[1].strip()) if len(parts) > 1 else None
                except (ValueError, IndexError):
                    line_num = None

                # Look backwards for error message
                error_msg = line.strip()
                for j in range(max(0, i - 3), i):
                    prev = lines[j].strip()
                    if prev.startswith("$$$"):
                        error_msg = prev
                        break

                errors.append({'type': 'EXECUTION_ERROR', 'line': line_num, 'message': error_msg})

            elif line.strip().startswith("$$$ ERROR"):
                errors.append({'type': 'GENERAL_ERROR', 'line': None, 'message': line.strip()})

        return errors

    def run_simulation(self, output_dir='results'):
        """Generate and execute COSY simulation."""

        # Copy required files
        files_to_copy = ['cosy', 'cosy.fox']
        if self.use_mge_for_dipoles:
            files_to_copy.append(self.mge_fieldmap_filename)

        files_not_found = []
        for fname in files_to_copy:
            src = self._find_file(fname)
            if src:
                shutil.copy(src, os.path.join(output_dir, fname))
            elif fname == self.mge_fieldmap_filename and self.use_mge_for_dipoles:
                grouped = self._detect_dipole_triplets()
                if self._check_for_mge_dipoles(grouped):
                    raise FileNotFoundError(
                        f"MGE fieldmap '{fname}' required but not found in: {', '.join(self.search_dirs)}"
                    )
            elif fname in ['cosy', 'cosy.fox']:
                files_not_found.append(fname)

        if files_not_found:
            self.logger.warning(f"COSY files not found: {', '.join(files_not_found)}")
            self.logger.warning(f"Searched: {', '.join(self.search_dirs)}")

        # Copy COSY.bin if available
        src_bin = self._find_file('COSY.bin')
        if src_bin:
            shutil.copy(src_bin, os.path.join(output_dir, 'COSY.bin'))

        input_file = self.generate_input(output_dir)

        # Compile if needed
        dst_bin = os.path.join(output_dir, 'COSY.bin')
        if not os.path.exists(dst_bin):
            compile_result = subprocess.run(['./cosy', 'cosy.fox'], cwd=output_dir,
                                            capture_output=True, text=True)
            if compile_result.returncode != 0:
                raise RuntimeError(f"COSY compilation failed: {compile_result.stderr}")

            compile_errors = self._parse_cosy_errors(compile_result.stdout + compile_result.stderr)
            if compile_errors:
                error_lines = [f"  Line {e['line']}: {e['message']}" if e['line']
                               else f"  {e['message']}" for e in compile_errors]
                raise RuntimeError("COSY compilation errors:\n" + "\n".join(error_lines))

        # Run simulation
        result = subprocess.run(['./cosy', 'input.fox'], cwd=output_dir,
                                capture_output=True, text=True)

        execution_errors = self._parse_cosy_errors(result.stdout + result.stderr)
        if execution_errors:
            error_lines = [f"  Line {e['line']}: {e['message']}" if e['line']
                           else f"  {e['message']}" for e in execution_errors]
            raise RuntimeError(
                f"COSY execution failed ({len(execution_errors)} error(s)):\n" +
                "\n".join(error_lines) +
                f"\n\nCheck {os.path.join(output_dir, 'input.fox')}"
            )

        if result.returncode != 0:
            raise RuntimeError(f"COSY failed (code {result.returncode}): {result.stderr}")

        return {"status": "success", "log": result.stdout, "errors": []}