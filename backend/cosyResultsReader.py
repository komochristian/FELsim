import os
import json
import re
import numpy as np
from loggingConfig import get_logger_with_fallback


class COSYResultsReader:
    def __init__(self, results_dir='results', debug=None):
        """Initialize COSY INFINITY output file parser."""
        self.results_dir = results_dir
        self._json_data = None
        self.logger, self.debug = get_logger_with_fallback(__name__, debug)

    def _load_json_if_needed(self):
        """Lazy load JSON data."""
        if self._json_data is None:
            self._json_data = self.read_json_results()
        return self._json_data

    @property
    def optimization_enabled(self):
        """Check if optimization was enabled."""
        json_data = self._load_json_if_needed()
        return bool(json_data.get('optimization_enabled', 0))

    def get_twiss_from_transfer_map(self,
                                    initial_twiss_x=None,
                                    initial_twiss_y=None,
                                    initial_emittance_x=1.0,
                                    initial_emittance_y=1.0,
                                    include_dispersion=True):
        """
        Compute Twiss parameters by propagating initial values through transfer map.

        Uses standard linear optics propagation:
            β₁ = M₁₁²β₀ - 2M₁₁M₁₂α₀ + M₁₂²γ₀
            α₁ = -M₁₁M₂₁β₀ + (M₁₁M₂₂ + M₁₂M₂₁)α₀ - M₁₂M₂₂γ₀
            γ₁ = M₂₁²β₀ - 2M₂₁M₂₂α₀ + M₂₂²γ₀

        Dispersion (if include_dispersion=True and 6D simulation):
            η₁ = M₁₁η₀ + M₁₂η₀' + M₁₆
            η₁' = M₂₁η₀ + M₂₂η₀' + M₂₆

        Parameters
        ----------
        initial_twiss_x : dict, optional
            Initial x-plane Twiss: {'beta': [m], 'alpha': [], 'eta': [m], 'etap': []}
            Default: β=1.0 m, α=0 (parallel beam at waist), η=η'=0
        initial_twiss_y : dict, optional
            Initial y-plane Twiss (same structure)
        initial_emittance_x : float, optional
            Geometric emittance [π·mm·mrad], default 1.0
        initial_emittance_y : float, optional
            Geometric emittance [π·mm·mrad], default 1.0
        include_dispersion : bool, optional
            Propagate dispersion (requires 6D simulation), default True

        Returns
        -------
        dict
            Twiss parameters: beta_x, alpha_x, gamma_x, emittance_x, mu_x (if stable),
            eta_x, etap_x (if dispersion), and same for y-plane.

        Notes
        -----
        - Emittance is conserved in linear transport
        - Twiss relation βγ - α² = 1 preserved by symplectic transport
        - Phase advance μ calculated from cos(μ) = (M₁₁ + M₂₂)/2
        - Dispersion propagation requires dimensions=3 in COSY config
        """
        # Set defaults for initial conditions
        if initial_twiss_x is None:
            initial_twiss_x = {'beta': 1.0, 'alpha': 0.0, 'eta': 0.0, 'etap': 0.0}
        if initial_twiss_y is None:
            initial_twiss_y = {'beta': 1.0, 'alpha': 0.0, 'eta': 0.0, 'etap': 0.0}

        # Extract initial parameters
        beta_x0 = initial_twiss_x['beta']
        alpha_x0 = initial_twiss_x['alpha']
        gamma_x0 = (1 + alpha_x0 ** 2) / beta_x0
        eta_x0 = initial_twiss_x.get('eta', 0.0)
        etap_x0 = initial_twiss_x.get('etap', 0.0)

        beta_y0 = initial_twiss_y['beta']
        alpha_y0 = initial_twiss_y['alpha']
        gamma_y0 = (1 + alpha_y0 ** 2) / beta_y0
        eta_y0 = initial_twiss_y.get('eta', 0.0)
        etap_y0 = initial_twiss_y.get('etap', 0.0)

        M = self.read_linear_transfer_map()

        # Extract 2×2 blocks for each plane
        M11_x, M12_x = M[0, 0], M[0, 1]
        M21_x, M22_x = M[1, 0], M[1, 1]
        M11_y, M12_y = M[2, 2], M[2, 3]
        M21_y, M22_y = M[3, 2], M[3, 3]

        # Dispersion coupling elements
        M16_x, M26_x = M[0, 5], M[1, 5]
        M36_y, M46_y = M[2, 5], M[3, 5]

        has_dispersion_elements = (abs(M16_x) > 1e-10 or abs(M26_x) > 1e-10 or
                                   abs(M36_y) > 1e-10 or abs(M46_y) > 1e-10)

        # Propagate Twiss parameters
        beta_x = M11_x ** 2 * beta_x0 - 2 * M11_x * M12_x * alpha_x0 + M12_x ** 2 * gamma_x0
        alpha_x = -M11_x * M21_x * beta_x0 + (M11_x * M22_x + M12_x * M21_x) * alpha_x0 - M12_x * M22_x * gamma_x0
        gamma_x = M21_x ** 2 * beta_x0 - 2 * M21_x * M22_x * alpha_x0 + M22_x ** 2 * gamma_x0

        beta_y = M11_y ** 2 * beta_y0 - 2 * M11_y * M12_y * alpha_y0 + M12_y ** 2 * gamma_y0
        alpha_y = -M11_y * M21_y * beta_y0 + (M11_y * M22_y + M12_y * M21_y) * alpha_y0 - M12_y * M22_y * gamma_y0
        gamma_y = M21_y ** 2 * beta_y0 - 2 * M21_y * M22_y * alpha_y0 + M22_y ** 2 * gamma_y0

        # Phase advance (check stability)
        cos_mu_x = (M11_x + M22_x) / 2.0
        cos_mu_y = (M11_y + M22_y) / 2.0

        stable_x = abs(cos_mu_x) <= 1.0
        stable_y = abs(cos_mu_y) <= 1.0

        mu_x = None
        if stable_x:
            mu_x = np.arccos(np.clip(cos_mu_x, -1.0, 1.0))
            if M12_x < 0:
                mu_x = 2 * np.pi - mu_x
        else:
            self.logger.warning(f"X-plane unstable: |Tr(Mx)/2| = {abs(cos_mu_x):.3f} > 1")

        mu_y = None
        if stable_y:
            mu_y = np.arccos(np.clip(cos_mu_y, -1.0, 1.0))
            if M12_y < 0:
                mu_y = 2 * np.pi - mu_y
        else:
            self.logger.warning(f"Y-plane unstable: |Tr(My)/2| = {abs(cos_mu_y):.3f} > 1")

        # Check Twiss relation (should be 1.0)
        twiss_check_x = beta_x * gamma_x - alpha_x ** 2
        twiss_check_y = beta_y * gamma_y - alpha_y ** 2

        if abs(twiss_check_x - 1.0) > 1e-6:
            self.logger.warning(f"X-plane Twiss relation: βγ - α² = {twiss_check_x:.9f}")
        if abs(twiss_check_y - 1.0) > 1e-6:
            self.logger.warning(f"Y-plane Twiss relation: βγ - α² = {twiss_check_y:.9f}")

        result = {
            'beta_x': float(beta_x),
            'alpha_x': float(alpha_x),
            'gamma_x': float(gamma_x),
            'emittance_x': initial_emittance_x,
            'beta_y': float(beta_y),
            'alpha_y': float(alpha_y),
            'gamma_y': float(gamma_y),
            'emittance_y': initial_emittance_y,
        }

        if mu_x is not None:
            result['mu_x'] = float(mu_x)
        if mu_y is not None:
            result['mu_y'] = float(mu_y)

        # Propagate dispersion
        if include_dispersion:
            if has_dispersion_elements or eta_x0 != 0.0 or etap_x0 != 0.0 or eta_y0 != 0.0 or etap_y0 != 0.0:
                eta_x = M11_x * eta_x0 + M12_x * etap_x0 + M16_x
                etap_x = M21_x * eta_x0 + M22_x * etap_x0 + M26_x
                eta_y = M11_y * eta_y0 + M12_y * etap_y0 + M36_y
                etap_y = M21_y * eta_y0 + M22_y * etap_y0 + M46_y

                result.update({
                    'eta_x': float(eta_x),
                    'etap_x': float(etap_x),
                    'eta_y': float(eta_y),
                    'etap_y': float(etap_y)
                })

                if self.debug:
                    self.logger.debug("Dispersion propagation:")
                    self.logger.debug(f"  X: η₀={eta_x0:.6f} m, η₀'={etap_x0:.6f} → "
                                      f"η={eta_x:.6f} m, η'={etap_x:.6f}")
                    self.logger.debug(f"     M₁₆={M16_x:.6f} m, M₂₆={M26_x:.6f}")
                    self.logger.debug(f"  Y: η₀={eta_y0:.6f} m, η₀'={etap_y0:.6f} → "
                                      f"η={eta_y:.6f} m, η'={etap_y:.6f}")
                    self.logger.debug(f"     M₃₆={M36_y:.6f} m, M₄₆={M46_y:.6f}")
            else:
                result.update({'eta_x': 0.0, 'etap_x': 0.0, 'eta_y': 0.0, 'etap_y': 0.0})
                if self.debug:
                    self.logger.debug("No dispersion (2D simulation)")

        if self.debug:
            self.logger.debug("\n" + "=" * 60)
            self.logger.debug("Twiss Propagation via Transfer Map")
            self.logger.debug("=" * 60)
            self.logger.debug(f"Initial: βx₀={beta_x0:.6f} m, αx₀={alpha_x0:.6f}, γx₀={gamma_x0:.6f} m⁻¹")
            self.logger.debug(f"         βy₀={beta_y0:.6f} m, αy₀={alpha_y0:.6f}, γy₀={gamma_y0:.6f} m⁻¹")
            self.logger.debug(f"         εx₀={initial_emittance_x:.6f}, εy₀={initial_emittance_y:.6f} π·mm·mrad")
            self.logger.debug(f"\nFinal:   βx={beta_x:.6f} m, αx={alpha_x:.6f}, γx={gamma_x:.6f} m⁻¹")
            self.logger.debug(f"         βy={beta_y:.6f} m, αy={alpha_y:.6f}, γy={gamma_y:.6f} m⁻¹")

            if mu_x is not None:
                self.logger.debug(f"\nPhase:   μx={mu_x:.6f} rad ({np.rad2deg(mu_x):.3f}°)")
            else:
                self.logger.debug("\nPhase:   X-plane UNSTABLE")

            if mu_y is not None:
                self.logger.debug(f"         μy={mu_y:.6f} rad ({np.rad2deg(mu_y):.3f}°)")
            else:
                self.logger.debug("         Y-plane UNSTABLE")

            self.logger.debug("=" * 60 + "\n")

        return result

    def get_variables(self):
        """Get optimization variable values from results."""
        json_data = self._load_json_if_needed()

        if 'variables' not in json_data:
            return {}

        variables = {}
        for var_name, value in json_data['variables'].items():
            complex_value = self.convert_complex_pair(value)
            variables[var_name] = complex_value.real

            if abs(complex_value.imag) > 1e-10:
                import warnings
                warnings.warn(
                    f"Variable '{var_name}' has imaginary component: "
                    f"{complex_value.imag:.3e}, using real part only"
                )

        return variables

    def get_beam_position(self):
        """Get final beam s-coordinate from results."""
        json_data = self._load_json_if_needed()

        if 'spos' not in json_data:
            raise ValueError("No beam position (spos) in results")

        spos_value = self.convert_complex_pair(json_data['spos'])
        return spos_value.real

    def get_full_results(self):
        """Get all results including Twiss, variables, and metadata."""
        json_data = self._load_json_if_needed()

        return {
            'twiss': self.get_twiss_from_transfer_map(),
            'variables': self.get_variables(),
            'spos': self.get_beam_position(),
            'optimization_enabled': self.optimization_enabled,
            'raw': json_data
        }

    def read_linear_transfer_map(self, filename='fort.99'):
        """
        Read 6×6 linear transfer matrix from COSY output.

        Extracts the complete linear map: M[i,j] = ∂(coordinate_i)/∂(coordinate_j)
        where coordinates are [x, x', y, y', l, δK].

        In 2D transverse-only simulations, longitudinal elements default to identity.

        Parameters
        ----------
        filename : str
            COSY output file name

        Returns
        -------
        numpy.ndarray (6, 6)
            Full transfer matrix. Key elements:
            - M[0,5], M[1,5]: x, x' dispersion (M₁₆, M₂₆)
            - M[2,5], M[3,5]: y, y' dispersion (M₃₆, M₄₆)
            - M[:4,:4]: transverse 4×4 block
            - M[4:,4:]: longitudinal 2×2 block (may be identity in 2D)

        Notes
        -----
        COSY uses 6-digit indices where each digit is the power of an initial coordinate.
        Format: [x, a, y, b, l, δK] where a=x', b=y'
        - 100000 → ∂/∂x₀    (column 0)
        - 010000 → ∂/∂x'₀   (column 1)
        - 001000 → ∂/∂y₀    (column 2)
        - 000100 → ∂/∂y'₀   (column 3)
        - 000010 → ∂/∂l₀    (column 4)
        - 000001 → ∂/∂δK₀   (column 5)
        """
        filepath = os.path.join(self.results_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"COSY output not found: {filepath}")

        coefficients = {}

        with open(filepath, 'r') as f:
            for line in f:
                line_stripped = line.strip()

                if not line_stripped or line_stripped.startswith('--') or len(line_stripped) < 10:
                    continue

                parts = line_stripped.split()
                if len(parts) < 2:
                    continue

                index = parts[-1]

                # Check for valid 6-digit index
                if not (len(index) == 6 and index.isdigit()):
                    continue

                target_indices = ['100000', '010000', '001000', '000100', '000010', '000001']

                if index in target_indices:
                    try:
                        coeff_values = []

                        for i in range(min(6, len(parts) - 1)):
                            coeff_str = parts[i]

                            # Handle concatenated scientific notation
                            if coeff_str.count('E') > 1 or coeff_str.count('e') > 1:
                                numbers = re.findall(r'[+-]?\d+\.?\d*[Ee][+-]?\d+', coeff_str)
                                if numbers:
                                    coeff_values.append(float(numbers[0]))
                                    continue

                            coeff_values.append(float(coeff_str))

                        while len(coeff_values) < 6:
                            coeff_values.append(0.0)

                        coefficients[index] = coeff_values[:6]

                    except (ValueError, IndexError) as e:
                        self.logger.error(f"Parse error on line: {line_stripped}")
                        self.logger.error(f"  {e}")
                        continue

        # Verify we have required transverse indices
        required_transverse = ['100000', '010000', '001000', '000100']
        missing_transverse = [idx for idx in required_transverse if idx not in coefficients]

        if missing_transverse:
            found = [idx for idx in required_transverse if idx in coefficients]
            raise ValueError(
                f"Missing transverse indices in {filename}: {missing_transverse}\n"
                f"Found: {found}\n"
                f"Try increasing 'order' parameter in COSY config (e.g., order=3)"
            )

        # Check for longitudinal indices
        longitudinal_indices = ['000010', '000001']
        has_longitudinal = all(idx in coefficients for idx in longitudinal_indices)

        if not has_longitudinal:
            missing = [idx for idx in longitudinal_indices if idx not in coefficients]
            self.logger.debug(
                f"Missing longitudinal indices {missing} - assuming 2D simulation"
            )

        # Build 6×6 matrix
        transfer_matrix = np.zeros((6, 6))

        # Transverse block (always present)
        transfer_matrix[:, 0] = coefficients['100000']
        transfer_matrix[:, 1] = coefficients['010000']
        transfer_matrix[:, 2] = coefficients['001000']
        transfer_matrix[:, 3] = coefficients['000100']

        # Longitudinal block (identity if missing)
        if '000010' in coefficients:
            transfer_matrix[:, 4] = coefficients['000010']
        else:
            transfer_matrix[4, 4] = 1.0

        if '000001' in coefficients:
            transfer_matrix[:, 5] = coefficients['000001']
        else:
            transfer_matrix[5, 5] = 1.0

        if self.debug:
            self.logger.debug("\n" + "=" * 60)
            self.logger.debug("Transfer Matrix Extraction")
            self.logger.debug("=" * 60)
            self.logger.debug(f"Longitudinal dynamics: {'Yes' if has_longitudinal else 'No (2D)'}")

            # Check coupling
            x_y_coupling = np.any(transfer_matrix[:2, 2:4] != 0) or np.any(transfer_matrix[2:4, :2] != 0)
            trans_long_coupling = (np.any(transfer_matrix[:4, 4:6] != 0) or
                                   np.any(transfer_matrix[4:6, :4] != 0))

            self.logger.debug(f"X-Y coupling: {'Yes' if x_y_coupling else 'No'}")
            self.logger.debug(f"Transverse-Longitudinal coupling: {'Yes' if trans_long_coupling else 'No'}")

            # Show matrix structure
            coord_labels = ['x', "x'", 'y', "y'", 'l', 'δK']
            self.logger.debug("\nMatrix structure:")
            header = "        " + "".join(f"{lbl:>8s} " for lbl in coord_labels)
            self.logger.debug(header)

            for i, row_label in enumerate(coord_labels):
                row_str = f"  {row_label:>4s}  "
                for j in range(6):
                    val = transfer_matrix[i, j]
                    if abs(val) < 1e-10:
                        row_str += "    ·    "
                    else:
                        row_str += f"{val:>8.4f} "
                self.logger.debug(row_str)

            # Show dispersion if present
            if has_longitudinal and np.any(transfer_matrix[:4, 5] != 0):
                self.logger.debug(f"\nDispersion elements:")
                self.logger.debug(f"  M₁₆ (x|δ)  = {transfer_matrix[0, 5]:>10.6f} m")
                self.logger.debug(f"  M₂₆ (x'|δ) = {transfer_matrix[1, 5]:>10.6f}")
                self.logger.debug(f"  M₃₆ (y|δ)  = {transfer_matrix[2, 5]:>10.6f} m")
                self.logger.debug(f"  M₄₆ (y'|δ) = {transfer_matrix[3, 5]:>10.6f}")

            self.logger.debug("=" * 60 + "\n")

        return transfer_matrix

    def read_json_results(self, filename='result.txt'):
        """Read JSON-formatted results from COSY output."""
        filepath = os.path.join(self.results_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"JSON results not found: {filepath}")

        try:
            with open(filepath, 'r') as f:
                content = f.read().strip()

                if not content:
                    raise ValueError(f"{filename} is empty")

                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Try cleaning common issues
                    cleaned = self._clean_json(content)
                    return json.loads(cleaned)

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {filename}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error reading {filename}: {e}")

    def _clean_json(self, json_str):
        """Remove trailing commas from JSON string."""
        return re.sub(r',\s*([}\]])', r'\1', json_str)

    def convert_complex_pair(self, value_pair):
        """
        Convert COSY complex format (real, imag) to numpy complex number.

        Accepts: complex, numeric, string "(real, imag)", tuple/list [real, imag]
        """
        if isinstance(value_pair, (complex, np.complexfloating)):
            return np.complex128(value_pair)

        elif isinstance(value_pair, (int, float, np.number)):
            return np.complex128(value_pair)

        elif isinstance(value_pair, str):
            try:
                cleaned = value_pair.strip().strip('()').strip()
                if ',' in cleaned:
                    parts = cleaned.split(',')
                    if len(parts) != 2:
                        raise ValueError(f"Expected 'real, imag': {value_pair}")

                    real_part = float(parts[0].strip())
                    imag_part = float(parts[1].strip())
                    return np.complex128(real_part + 1j * imag_part)
                else:
                    return np.complex128(float(cleaned))

            except (ValueError, TypeError) as e:
                raise ValueError(f"Cannot parse '{value_pair}' as complex: {e}")

        elif isinstance(value_pair, (tuple, list)):
            if len(value_pair) != 2:
                raise ValueError(f"Complex pair needs 2 elements, got {len(value_pair)}")

            try:
                real_part = float(value_pair[0])
                imag_part = float(value_pair[1])
                return np.complex128(real_part + 1j * imag_part)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Cannot convert {value_pair} to complex: {e}")

        else:
            raise TypeError(f"Unsupported type for complex conversion: {type(value_pair)}")

    def __str__(self):
        return f"COSYResultsReader(results_dir='{self.results_dir}')"

    def __repr__(self):
        return f"COSYResultsReader(results_dir='{self.results_dir}')"