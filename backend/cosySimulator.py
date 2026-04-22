"""
COSY INFINITY simulator interface.

Author: Eremey Valetov
"""

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
                 use_mge_for_dipoles=False, transfer_matrix_order=None, fringe_field_order=0,
                 quad_aperture=0.027, dipole_aperture=0.0127,
                 cosy_dist_dir=None):
        super().__init__(excel_path, json_config_path)

        self.config = self._load_config(json_config_path, config_dict)
        sim_config = self.config.get('simulation', {})
        self.KE = sim_config.get('KE', 45.0)
        self.order = sim_config.get('order', 3)
        self.dimensions = sim_config.get('dimensions', 2)

        # Transfer matrix order for PM command - defaults to 1 (linear), cannot exceed computation order
        if transfer_matrix_order is None:
            self.transfer_matrix_order = sim_config.get('transfer_matrix_order', 1)
        else:
            self.transfer_matrix_order = transfer_matrix_order

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
        self.fit_combined_mse = True  # Single combined MSE objective (vs individual objectives)
        self.optimization_enabled = False

        # Apply config if provided
        if self.config.get('optimization_initial_point'):
            self._add_optimization_initial_point(self.config['optimization_initial_point'], reset=True)
        if self.config.get('optimization_objectives'):
            self._add_optimization_objectives(self.config['optimization_objectives'], reset=True)
            self.optimization_enabled = True

        # Twiss parameter mapping: (axis, param) → COSY expression
        # None values require special handling in _generate_objective_expression
        #
        # Transverse objectives (α, β, γ, dispersion, envelope) suffice when the
        # transport line has no RF and chirp is zero — longitudinal phase space
        # decouples from transverse matching in that regime.
        # Longitudinal objectives (R56, T566) are needed for non-zero chirp or
        # bunch compression studies where the optimizer must target specific
        # longitudinal transfer matrix elements.
        self.MEASURE_MAP = {
            ("x", "alpha"): "A0(1)", ("y", "alpha"): "A0(2)",
            ("x", "beta"): "B0(1)", ("y", "beta"): "B0(2)",
            ("x", "gamma"): "G0(1)", ("y", "gamma"): "G0(2)",
            ("x", "dispersion"): "ME(1,6)", ("y", "dispersion"): "ME(3,6)",
            ("x", "envelope"): None, ("y", "envelope"): None,
            # Longitudinal transfer matrix elements (requires dimensions=3)
            ("l", "r51"): "ME(5,1)", ("l", "r52"): "ME(5,2)",
            ("l", "r56"): "ME(5,6)",  # R56: path length dependence on energy
            ("l", "t566"): "ME(5,66)",  # T566: 2nd-order R56; requires order >= 2
        }

        # Geometric emittance in pi.mm.mrad — required for envelope objectives
        self.geometric_emittance = None

        # Initial Twiss parameters for objective computation (default: round beam, unit beta)
        self.initial_twiss_x = {'beta': 1.0, 'alpha': 0.0}
        self.initial_twiss_y = {'beta': 1.0, 'alpha': 0.0}

        # Particle tracking
        self.aperture_cuts_enabled = False
        self.dipole_half_width = 0.050  # m, horizontal half-aperture for dipoles
        self._aperture_warning_logged = False
        self.particle_tracking_mode = False
        self.particle_input_unit = 200
        self.particle_checkpoint_base_unit = 10000
        self.particle_checkpoint_elements = None
        self.particle_checkpoint_count = 0

        self.logger, self.debug = get_logger_with_fallback(__name__, debug)

        # Validate transfer_matrix_order
        if self.transfer_matrix_order > self.order:
            self.logger.warning(
                f"transfer_matrix_order={self.transfer_matrix_order} exceeds "
                f"computation order={self.order}, clamping to {self.order}"
            )
            self.transfer_matrix_order = self.order
        elif self.transfer_matrix_order < 1:
            self.logger.warning(
                f"transfer_matrix_order={self.transfer_matrix_order} is invalid, setting to 1"
            )
            self.transfer_matrix_order = 1

        self.use_enge_coeffs = use_enge_coeffs
        self.use_mge_for_dipoles = use_mge_for_dipoles
        self.fringe_field_order = fringe_field_order
        self.quad_aperture = quad_aperture    # m, bore radius for MQ command
        self.dipole_aperture = dipole_aperture  # m, pole gap for DIL command

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
        self.project_fields_dir = os.path.join(os.path.dirname(self.script_dir), 'fields')
        self.default_cosy_dir = os.path.expanduser('/usr/local/bin')
        # Path to COSY INFINITY distribution; override to match local installation
        self.cosy_dist_dir = cosy_dist_dir or os.path.expanduser('~/COSY/10.2/UNIX')
        self.search_dirs = [self.script_dir, self.current_dir, self.project_fields_dir,
                            self.default_cosy_dir, self.cosy_dist_dir]

        # MGE parameters
        self.mge_array_name = "MkIIIChicaneDipoleField"
        self.mge_fieldmap_filename = "chicane_dipole_fieldmap.dat"
        self.mge_s = 1.4
        self.mge_scaling = self.P / self.P_45
        self.symbolic_variables = set()

        # MGE declarations go in {Initialization} (before PROCEDURE LATTICE)
        # Only the field array + NP/NS/DELTAS scalars — no intermediate arrays
        self.mge_declarations = """    VARIABLE {array_name} 1 1 {ns} ;
    VARIABLE NP 1 ;
    VARIABLE NS 1 ;
    VARIABLE DELTAS 1 ;
"""
        # MGE executable code goes in {PreLattice} (after ENDPROCEDURE, before OV)
        # Field values written directly to avoid VEC/MGE_FIELD intermediate arrays
        self.mge_setup_template = "    NP := {np} ;\n    NS := {ns} ;\n    DELTAS := {deltas} ;\n"

    def _build_boilerplate(self):
        return f"""
INCLUDE 'COSY' ;
PROCEDURE RUN ;
    VARIABLE A0 100 {self.dimensions} ; VARIABLE B0 100 {self.dimensions} ; VARIABLE G0 100 {self.dimensions} ;
    VARIABLE R0 100 {self.dimensions} ; VARIABLE MU0 100 {self.dimensions} ; VARIABLE F0 100 {2*self.dimensions} ;
{{Initialization}}
    PROCEDURE LATTICE ;
        UM ; CR ;
{{Elements}}
    ENDPROCEDURE ;

{{PreLattice}}
    OV {self.order} {self.dimensions} 0 ;
    RPE {self.KE} ;
    FR {self.fringe_field_order} ;
    LATTICE ;
{{Optimization}}
    CO {self.transfer_matrix_order} ; PM 99 ;

    OPENF 51 'result.txt' 'UNKNOWN';
        WRITE 51 '{{';
        WRITE 51 '"spos": '&S(SPOS)&',';
        WRITE 51 '"optimization_enabled": {1 if self.optimization_enabled else 0}' ;
{{Variables}}
        WRITE 51 '}}';
    CLOSEF 51 ;

    GT MAP F0 MU0 A0 B0 G0 R0 ;

    OPENF 51 'result.txt' 'UNKNOWN';
        WRITE 51 '{{';
        WRITE 51 '"spos": '&S(SPOS)&',';
        WRITE 51 '"optimization_enabled": {1 if self.optimization_enabled else 0},' ;
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
    VARIABLE OBJ 1 100 ;
"""
            if self.debug:
                self.logger.debug("Added optimization variables: MAP_ARR, MAP_IDX, IDX, NOM, OBJ")

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
            valid = sorted(f"{a}.{p}" for a, p in self.MEASURE_MAP.keys())
            raise ValueError(f"Invalid measure ({axis}, {param}). Valid: {valid}")

        if axis == "l" and self.dimensions < 3:
            raise ValueError(
                f"Longitudinal objective ({axis}, {param}) requires dimensions=3. "
                f"Current dimensions={self.dimensions}."
            )

        return axis, param

    def _generate_twiss_fox(self, indent="        "):
        """Generate FOX code for Twiss computation from transfer matrix.

        Uses initial_twiss_x/y: β(s) = R₁₁²β₀ - 2R₁₁R₁₂α₀ + R₁₂²γ₀, etc.
        Assumes MAP(IDX) has already been loaded from MAP_ARR.
        """
        bx0 = self.initial_twiss_x['beta']
        ax0 = self.initial_twiss_x['alpha']
        gx0 = (1 + ax0**2) / bx0
        by0 = self.initial_twiss_y['beta']
        ay0 = self.initial_twiss_y['alpha']
        gy0 = (1 + ay0**2) / by0

        lines = []
        # β₀=1, α₀=0, γ₀=1 is the default — use simplified expressions
        if abs(ax0) < 1e-12 and abs(bx0 - 1) < 1e-12:
            lines.append(f"{indent}B0(1) := ME(1,1)*ME(1,1) + ME(1,2)*ME(1,2) ;")
            lines.append(f"{indent}A0(1) := -(ME(1,1)*ME(2,1) + ME(1,2)*ME(2,2)) ;")
        elif abs(ax0) < 1e-12:
            lines.append(f"{indent}B0(1) := ME(1,1)*ME(1,1)*{bx0} + ME(1,2)*ME(1,2)*{gx0} ;")
            lines.append(f"{indent}A0(1) := -(ME(1,1)*ME(2,1)*{bx0} + ME(1,2)*ME(2,2)*{gx0}) ;")
        else:
            lines.append(f"{indent}B0(1) := ME(1,1)*ME(1,1)*{bx0} - 2*ME(1,1)*ME(1,2)*{ax0} + ME(1,2)*ME(1,2)*{gx0} ;")
            lines.append(f"{indent}A0(1) := -(ME(1,1)*ME(2,1)*{bx0} - (ME(1,1)*ME(2,2)+ME(1,2)*ME(2,1))*{ax0} + ME(1,2)*ME(2,2)*{gx0}) ;")

        if abs(ay0) < 1e-12 and abs(by0 - 1) < 1e-12:
            lines.append(f"{indent}B0(2) := ME(3,3)*ME(3,3) + ME(3,4)*ME(3,4) ;")
            lines.append(f"{indent}A0(2) := -(ME(3,3)*ME(4,3) + ME(3,4)*ME(4,4)) ;")
        elif abs(ay0) < 1e-12:
            lines.append(f"{indent}B0(2) := ME(3,3)*ME(3,3)*{by0} + ME(3,4)*ME(3,4)*{gy0} ;")
            lines.append(f"{indent}A0(2) := -(ME(3,3)*ME(4,3)*{by0} + ME(3,4)*ME(4,4)*{gy0}) ;")
        else:
            lines.append(f"{indent}B0(2) := ME(3,3)*ME(3,3)*{by0} - 2*ME(3,3)*ME(3,4)*{ay0} + ME(3,4)*ME(3,4)*{gy0} ;")
            lines.append(f"{indent}A0(2) := -(ME(3,3)*ME(4,3)*{by0} - (ME(3,3)*ME(4,4)+ME(3,4)*ME(4,3))*{ay0} + ME(3,4)*ME(4,4)*{gy0}) ;")

        return "\n".join(lines) + "\n"

    def _generate_objective_expression(self, objective):
        """Generate COSY objective expression from objective dict.

        When fit_combined_mse is True, returns a squared weighted term without
        CONS() — preserves DA derivatives for gradient-based FIT (algorithm 1).
        When False, returns ABS(CONS(...)) terms for individual-objective FIT.
        """
        measure = self._validate_measure(objective["measure"])
        goal = objective["goal"]
        weight = objective.get("weight", 1)
        axis, param = measure
        cosy_var = self.MEASURE_MAP[measure]

        if cosy_var is None and param == "envelope":
            if self.geometric_emittance is None:
                raise ValueError(
                    "Set geometric_emittance (pi.mm.mrad) before using envelope objectives. "
                    "Call sim.set_geometric_emittance(epsilon)."
                )
            plane = 1 if axis == "x" else 2
            eps_si = self.geometric_emittance * 1e-6  # pi.mm.mrad → m·rad
            expr = f"1000*SQRT({eps_si}*B0({plane}))"
            if self.fit_combined_mse:
                return f"{weight}*({expr} - {goal})*({expr} - {goal})"
            return f"{weight}*ABS(1000*SQRT({eps_si}*CONS(B0({plane}))) - {goal})"

        if param == "t566":
            if self.order < 2:
                raise ValueError(
                    "T566 objective requires computation order >= 2 "
                    f"(current order={self.order})"
                )
            cosy_var = "ME(5,66)"
            if self.fit_combined_mse:
                return f"{weight}*({cosy_var} - {goal})*({cosy_var} - {goal})"
            return f"{weight}*ABS(CONS({cosy_var}) - {goal})"

        if self.fit_combined_mse:
            return f"{weight}*({cosy_var} - {goal})*({cosy_var} - {goal})"
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
        combined_mse = self.fit_combined_mse
        obj_refs = []
        obj_idx = 1
        n_goals = 0

        if combined_mse:
            fit_body += "        OBJ(1) := 0 ;\n"

        twiss_fox = self._generate_twiss_fox(indent="        ")
        for elem_num in sorted(objectives_by_elem.keys()):
            fit_body += f"        LOOP IDX 1 6 ; MAP(IDX) := MAP_ARR(IDX, {elem_num}) ; ENDLOOP ;\n"
            fit_body += twiss_fox

            for obj in objectives_by_elem[elem_num]:
                expr = self._generate_objective_expression(obj)

                if combined_mse:
                    # Accumulate squared terms into OBJ(1) before B0/A0 are overwritten
                    fit_body += f"        OBJ(1) := OBJ(1) + {expr} ;\n"
                    n_goals += 1
                else:
                    fit_body += f"        OBJ({obj_idx}) := {expr} ;\n"
                    obj_refs.append(f"OBJ({obj_idx})")
                    obj_idx += 1

                if self.debug:
                    measure_str = f"{obj['measure'][0]}.{obj['measure'][1]}"
                    label = "MSE term" if combined_mse else f"objective {obj_idx - 1}"
                    self.logger.debug(
                        f"Added {label}: {measure_str} → {obj['goal']} "
                        f"(weight={obj.get('weight', 1)}) at element {elem_num}"
                    )

        if combined_mse:
            fit_body += f"        OBJ(1) := OBJ(1) / {n_goals} ;\n"
            obj_list_str = "OBJ(1)"
        else:
            obj_list_str = " ".join(obj_refs)

        # Post-FIT diagnostic: re-run lattice with optimized variables and print Twiss at objectives
        diag_twiss = self._generate_twiss_fox(indent="    ")
        diag = "\n    { Post-FIT verification }\n    LATTICE ;\n"
        for elem_num in sorted(objectives_by_elem.keys()):
            diag += f"    LOOP IDX 1 6 ; MAP(IDX) := MAP_ARR(IDX, {elem_num}) ; ENDLOOP ;\n"
            diag += diag_twiss
            diag += (
                f"    WRITE 6 'POST-FIT elem {elem_num}: "
                f"ax='&SF(CONS(A0(1)),'(F12.6)')&"
                f"' ay='&SF(CONS(A0(2)),'(F12.6)')&"
                f"' bx='&SF(CONS(B0(1)),'(F12.6)')&"
                f"' by='&SF(CONS(B0(2)),'(F12.6)')&"
                f"' Dx='&SF(CONS(ME(1,6)),'(F12.6)') ;\n"
            )
        for v in sorted(fit_vars):
            diag += f"    WRITE 6 'POST-FIT {v}='&SF(CONS({v}),'(F12.6)') ;\n"

        fit_code = f"""    FIT {fit_vars_str} ;
            LATTICE ;
    {fit_body}    ENDFIT {eps} {nmax} {nalg} {obj_list_str} ;
    {diag}"""

        self.optimization_fit_blocks.append({
            "init_code": init_code,
            "fit_code": fit_code,
            "variables": sorted(fit_vars),
            "objectives": fit_objectives,
            "optimizer_settings": {"eps": eps, "Nmax": nmax, "Nalgorithm": nalg}
        })

        self.optimization_enabled = True

        n_obj_str = f"{n_goals} MSE terms" if combined_mse else f"{len(obj_refs)} objective(s)"
        if self.debug:
            self.logger.debug(
                f"Created FIT block #{len(self.optimization_fit_blocks)} with "
                f"{len(fit_vars)} variable(s) and {n_obj_str}"
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

    def enable_aperture_cuts(self, dipole_half_width=None):
        """Enable AP commands after elements in particle tracking mode.

        Quads: circular aperture AP r r 1 with r = quad_aperture / 2.
        Dipoles: rectangular aperture AP w h 2 with h = pole_gap / 2 and
        w = dipole_half_width.

        Parameters
        ----------
        dipole_half_width : float, optional
            Horizontal half-aperture for dipoles [m]. Default 0.050 m (conservative placeholder).
        """
        self.aperture_cuts_enabled = True
        if dipole_half_width is not None:
            self.dipole_half_width = dipole_half_width
        if not self._aperture_warning_logged:
            self.logger.warning(
                f"Aperture cuts enabled. Dipole horizontal half-width = "
                f"{self.dipole_half_width} m."
            )
            self._aperture_warning_logged = True
        return {'aperture_cuts_enabled': True, 'dipole_half_width': self.dipole_half_width}

    def disable_aperture_cuts(self):
        self.aperture_cuts_enabled = False
        return {'aperture_cuts_enabled': False}

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

    def _add_aperture_cut(self, elements_str, elem):
        """Add COSY AP command after element if aperture cuts are enabled.

        AP X Y I: X = horizontal half-aperture, Y = vertical half-aperture.
        I=1: elliptic (x²/X² + y²/Y² ≤ 1).  I=2: rectangular (|x|≤X, |y|≤Y).
        """
        if not (self.particle_tracking_mode and self.aperture_cuts_enabled):
            return elements_str

        etype = elem['type']
        if etype in ('QPF', 'QPD'):
            r = self.quad_aperture / 2
            elements_str += f"    AP {r} {r} 1 ;\n"
        elif etype == 'DIPOLE_CONSOLIDATED':
            h = elem.get('pole_gap', self.dipole_aperture) / 2
            w = self.dipole_half_width
            elements_str += f"    AP {w} {h} 2 ;\n"

        return elements_str

    def _parse_enge_coefficients(self, enge_data):
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
                    'entrance_angle': entrance.get('wedge_angle', 0),
                    'exit_angle': exit_wedge.get('wedge_angle', 0),
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

    def set_geometric_emittance(self, emittance):
        """Set geometric emittance for envelope objective calculations.

        Parameters
        ----------
        emittance : float
            Geometric emittance in pi.mm.mrad (same unit as FELsim's ebeam.epsilon).
        """
        self.geometric_emittance = emittance

    def set_initial_twiss(self, beta_x, alpha_x, beta_y, alpha_y):
        """Set initial Twiss parameters for objective computation in FIT blocks.

        FELsim's beamOptimizer computes Twiss from particle statistics, so the
        objectives depend on the initial beam distribution. This method sets the
        equivalent initial Twiss so that transfer-matrix-based Twiss in COSY
        matches FELsim's particle-based values.

        Parameters
        ----------
        beta_x, alpha_x : float
            Initial horizontal Twiss parameters at beamline entrance [m, rad].
        beta_y, alpha_y : float
            Initial vertical Twiss parameters at beamline entrance [m, rad].
        """
        self.initial_twiss_x = {'beta': beta_x, 'alpha': alpha_x}
        self.initial_twiss_y = {'beta': beta_y, 'alpha': alpha_y}
        if self.debug:
            self.logger.debug(
                f"Initial Twiss set: βx={beta_x:.4f}, αx={alpha_x:.4f}, "
                f"βy={beta_y:.4f}, αy={alpha_y:.4f}"
            )

    def get_element_index_mapping(self):
        """Build mapping from original beamline indices to COSY consolidated indices.

        COSY's _detect_dipole_triplets() consolidates DPW-DPH-DPW sequences into
        single dipole elements, shifting all downstream indices. This mapping is
        required to translate FELsim element indices to COSY MAP_ARR indices.

        Consolidated indices are 0-based. For MAP_ARR references in FIT blocks,
        add 1 (MAP_ARR uses 1-based indexing).

        Returns
        -------
        dict
            'orig_to_consolidated': {orig_idx: consolidated_idx}
            'consolidated_to_orig': {consolidated_idx: [orig_indices]}
            'n_original': total original elements
            'n_consolidated': total consolidated elements
        """
        grouped = self._detect_dipole_triplets()

        orig_to_cons = {}
        cons_to_orig = {}
        orig_idx = 0

        for cons_idx, elem in enumerate(grouped):
            if elem['type'] == 'DIPOLE_CONSOLIDATED':
                indices = [orig_idx, orig_idx + 1, orig_idx + 2]
                for oi in indices:
                    orig_to_cons[oi] = cons_idx
                cons_to_orig[cons_idx] = indices
                orig_idx += 3
            else:
                orig_to_cons[orig_idx] = cons_idx
                cons_to_orig[cons_idx] = [orig_idx]
                orig_idx += 1

        if self.debug:
            self.logger.debug(
                f"Element index mapping: {orig_idx} original → {len(grouped)} consolidated "
                f"({orig_idx - len(grouped)} elements absorbed by triplet consolidation)"
            )

        return {
            'orig_to_consolidated': orig_to_cons,
            'consolidated_to_orig': cons_to_orig,
            'n_original': orig_idx,
            'n_consolidated': len(grouped),
        }

    def _check_for_mge_dipoles(self, grouped_elements):
        if not (self.use_mge_for_dipoles and self.use_enge_coeffs):
            return False

        return any(
            elem['type'] == "DIPOLE_CONSOLIDATED" and
            (elem.get('entrance_enge_coeffs') or elem.get('exit_enge_coeffs'))
            for elem in grouped_elements
        )

    def update_simulation_config(self, **kwargs):
        """Update simulation parameters (KE, order, dimensions, transfer_matrix_order)."""
        valid = {'KE', 'order', 'dimensions', 'transfer_matrix_order'}
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

        # Validate transfer_matrix_order if it was changed
        if 'transfer_matrix_order' in changes or 'order' in changes:
            if self.transfer_matrix_order > self.order:
                old_tm_order = self.transfer_matrix_order
                self.transfer_matrix_order = self.order
                self.logger.warning(
                    f"transfer_matrix_order={old_tm_order} exceeds computation order={self.order}, "
                    f"clamped to {self.order}"
                )
                if 'transfer_matrix_order' not in changes:
                    changes['transfer_matrix_order'] = {'old': old_tm_order, 'new': self.order}
                else:
                    changes['transfer_matrix_order']['new'] = self.order
            elif self.transfer_matrix_order < 1:
                old_tm_order = self.transfer_matrix_order
                self.transfer_matrix_order = 1
                self.logger.warning(f"transfer_matrix_order={old_tm_order} is invalid, set to 1")
                if 'transfer_matrix_order' not in changes:
                    changes['transfer_matrix_order'] = {'old': old_tm_order, 'new': 1}
                else:
                    changes['transfer_matrix_order']['new'] = 1

        if 'KE' in changes:
            self._recalculate_energy_dependent_parameters()
            if self.debug:
                self.logger.debug(f"Recalculated for KE={self.KE} MeV: γ={self.gamma:.6f}, β={self.beta:.6f}")

        if changes:
            self.logger.info(f"Updated simulation config: {list(changes.keys())}")

        return changes

    def _recalculate_energy_dependent_parameters(self):
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
            'simulation': {
                'KE': self.KE,
                'order': self.order,
                'dimensions': self.dimensions,
                'transfer_matrix_order': self.transfer_matrix_order
            },
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

        # Aperture cuts: when enabled, AP commands are inserted after each
        # element via _add_aperture_cut() — see enable_aperture_cuts().
        element_idx = 0
        self.particle_checkpoint_count = 0

        for elem in grouped_elements:
            element_idx += 1

            if elem['type'] == "DRIFT":
                elements_str += f"    DL {elem['length']} ;\n"
                elements_str = self._add_aperture_cut(elements_str, elem)
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
                elements_str = self._add_aperture_cut(elements_str, elem)
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

                elements_str = self._add_aperture_cut(elements_str, elem)
                elements_str = self._add_particle_checkpoint(elements_str, element_idx)
                elements_str = self._add_map_tracking_code(elements_str)

        self._validate_variable_consistency()
        var_declarations = self._generate_variable_declarations()
        var_output = self._generate_variable_output_code()

        if needs_mge:
            # Read fieldmap and resample to ≤30 points (COSY MFD limit)
            import numpy as np_lib
            fm_path = self._find_file(self.mge_fieldmap_filename)
            if not fm_path:
                raise FileNotFoundError(
                    f"MGE fieldmap '{self.mge_fieldmap_filename}' not found in: "
                    f"{', '.join(self.search_dirs)}")
            with open(fm_path) as f:
                fm_lines = [l.strip() for l in f if l.strip()]
            np_val = int(float(fm_lines[0]))
            ns_orig = int(fm_lines[1])
            deltas_orig = float(fm_lines[2])
            field_orig = np_lib.array([float(l) for l in fm_lines[3:3 + ns_orig]])

            # COSY 10.2 MFD array limits NS to 30
            mge_ns_max = 30
            if ns_orig > mge_ns_max:
                z_orig = np_lib.arange(ns_orig) * deltas_orig
                z_new = np_lib.linspace(z_orig[0], z_orig[-1], mge_ns_max)
                field_values = list(np_lib.interp(z_new, z_orig, field_orig))
                ns_val = mge_ns_max
                deltas_val = float(z_new[1] - z_new[0])
                if self.debug:
                    self.logger.debug(
                        f"Resampled fieldmap: {ns_orig}→{ns_val} points, "
                        f"DELTAS={deltas_orig}→{deltas_val:.6f}")
            else:
                field_values = list(field_orig)
                ns_val = ns_orig
                deltas_val = deltas_orig

            fmt_args = dict(array_name=self.mge_array_name, ns=ns_val)
            full_init = var_declarations + self.mge_declarations.format(**fmt_args)

            # Build direct-assignment code for field values
            pre_lattice = self.mge_setup_template.format(
                np=np_val, ns=ns_val, deltas=deltas_val)
            for j, val in enumerate(field_values, 1):
                scaled = val * self.mge_scaling
                pre_lattice += f"    {self.mge_array_name}(1,{j}) := {scaled} ;\n"
        else:
            full_init = var_declarations
            pre_lattice = ""

        full_input = self._build_boilerplate().replace("{Initialization}", full_init)
        full_input = full_input.replace("{PreLattice}", pre_lattice)
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

        # Copy SYSCA.DAT if fringe fields are enabled (required by COSY FR >= 1)
        if self.fringe_field_order > 0:
            src_sysca = self._find_file('SYSCA.DAT')
            if src_sysca:
                shutil.copy(src_sysca, os.path.join(output_dir, 'SYSCA.DAT'))
            else:
                self.logger.warning("SYSCA.DAT not found — required for FR >= 1")

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