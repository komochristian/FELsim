import os
import json
import re
import numpy as np
from loggingConfig import get_logger_with_fallback


class COSYResultsReader:
    def __init__(self, results_dir='results', debug=None):
        self.results_dir = results_dir
        self._json_data = None
        self.logger, self.debug = get_logger_with_fallback(__name__, debug)

    def _load_json_if_needed(self):
        if self._json_data is None:
            self._json_data = self.read_json_results()
        return self._json_data

    @property
    def optimization_enabled(self):
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

    def read_transfer_map_full(self, filename='fort.99', max_order=None):
        """
        Read complete transfer map including higher-order coefficients.

        Parses fort.99 to extract all aberration coefficients up to the specified order.
        Each line contains 6 coefficients (for x, x', y, y', l, δK outputs) and a 6-digit
        index indicating which partial derivative.

        Parameters
        ----------
        filename : str
            COSY output file (default: 'fort.99')
        max_order : int, optional
            Maximum order to read (1-5). If None, reads all available orders.

        Returns
        -------
        dict
            Dictionary with keys 1, 2, 3, ... for each order:
            {
                1: (6, 6) array - linear transfer matrix,
                2: dict - 2nd order coefficients {index: [6 coeffs]},
                3: dict - 3rd order coefficients {index: [6 coeffs]},
                ...
            }

        Notes
        -----
        Index format: 6 digits [i, j, k, l, m, n] representing powers of [x, x', y, y', l, δK]
        - Order = i+j+k+l+m+n
        - Example: '110000' = ∂²/∂x∂x' (2nd order)
        - Example: '200100' = ∂³/∂x²∂y' (3rd order)

        Each coefficient line gives all 6 output components:
        coeff_x  coeff_x'  coeff_y  coeff_y'  coeff_l  coeff_δK  index
        """
        filepath = os.path.join(self.results_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"COSY output not found: {filepath}")

        # Storage for each order
        coefficients_by_order = {}

        with open(filepath, 'r') as f:
            for line in f:
                line_stripped = line.strip()

                # Skip empty lines and separators
                if not line_stripped or line_stripped.startswith('--') or len(line_stripped) < 10:
                    continue

                parts = line_stripped.split()
                if len(parts) < 2:
                    continue

                index = parts[-1]

                # Check for valid 6-digit index
                if not (len(index) == 6 and index.isdigit()):
                    continue

                # Calculate order from sum of digits
                order = sum(int(d) for d in index)

                # Skip if beyond max_order
                if max_order is not None and order > max_order:
                    continue

                # Skip zero-order (000000) - just constants
                if order == 0:
                    continue

                try:
                    coeff_values = []

                    # Parse up to 6 coefficient values
                    for i in range(min(6, len(parts) - 1)):
                        coeff_str = parts[i]

                        # Handle concatenated scientific notation (e.g., "1.23E-04-5.67E-08")
                        if coeff_str.count('E') > 1 or coeff_str.count('e') > 1:
                            numbers = re.findall(r'[+-]?\d+\.?\d*[Ee][+-]?\d+', coeff_str)
                            if numbers:
                                coeff_values.append(float(numbers[0]))
                                continue

                        coeff_values.append(float(coeff_str))

                    # Pad with zeros if needed
                    while len(coeff_values) < 6:
                        coeff_values.append(0.0)

                    # Store by order
                    if order not in coefficients_by_order:
                        coefficients_by_order[order] = {}

                    coefficients_by_order[order][index] = coeff_values[:6]

                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Parse error on line: {line_stripped}")
                    self.logger.warning(f"  {e}")
                    continue

        # Convert order 1 to matrix form
        if 1 in coefficients_by_order:
            linear_dict = coefficients_by_order[1]
            transfer_matrix = np.zeros((6, 6))

            # Map indices to columns
            index_to_col = {
                '100000': 0,  # ∂/∂x
                '010000': 1,  # ∂/∂x'
                '001000': 2,  # ∂/∂y
                '000100': 3,  # ∂/∂y'
                '000010': 4,  # ∂/∂l
                '000001': 5  # ∂/∂δK
            }

            for index, col in index_to_col.items():
                if index in linear_dict:
                    transfer_matrix[:, col] = linear_dict[index]
                else:
                    # Identity for missing longitudinal
                    if col >= 4:
                        transfer_matrix[col, col] = 1.0

            coefficients_by_order[1] = transfer_matrix

        if self.debug:
            self.logger.debug(f"\n{'=' * 60}")
            self.logger.debug(f"Transfer Map Extraction (Full)")
            self.logger.debug(f"{'=' * 60}")
            for order in sorted(coefficients_by_order.keys()):
                if order == 1:
                    self.logger.debug(f"  Order {order}: 6×6 matrix")
                else:
                    n_terms = len(coefficients_by_order[order])
                    self.logger.debug(f"  Order {order}: {n_terms} coefficient terms")
            self.logger.debug(f"{'=' * 60}\n")

        return coefficients_by_order

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
        # Use the full reader to get just order 1
        full_map = self.read_transfer_map_full(filename=filename, max_order=1)

        if 1 not in full_map:
            raise ValueError("No linear (order 1) coefficients found in transfer map")

        return full_map[1]

    def read_transfer_map_all_orders(self, filename='fort.99', max_order=None):
        """
        Read transfer map coefficients of all orders from COSY output.

        Parameters
        ----------
        filename : str
            COSY output file name
        max_order : int, optional
            Maximum order to read (default: read all available)

        Returns
        -------
        dict
            Dictionary mapping order → coefficients
            {
                1: 6×6 linear map,
                2: dict of 2nd order coefficients,
                3: dict of 3rd order coefficients,
                ...
            }

        Notes
        -----
        Second and third order coefficients are stored as dicts with index tuples as keys:
        - 2nd order: {(target, src1, src2): value}
        - 3rd order: {(target, src1, src2, src3): value}

        Coordinate indices: 0=x, 1=x', 2=y, 3=y', 4=l, 5=δK

        Examples
        --------
        >>> reader = COSYResultsReader()
        >>> maps = reader.read_transfer_map_all_orders(max_order=3)
        >>> linear_map = maps[1]  # 6×6 matrix
        >>> # Get x coefficient from x²: maps[2][(0, 0, 0)]
        >>> # Get x coefficient from x³: maps[3][(0, 0, 0, 0)]
        """
        filepath = os.path.join(self.results_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"COSY output not found: {filepath}")

        # Store all coefficients by index
        all_coefficients = {}

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

                try:
                    # Parse coefficient values for all 6 target coordinates
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

                    all_coefficients[index] = coeff_values[:6]

                except (ValueError, IndexError) as e:
                    self.logger.error(f"Parse error on line: {line_stripped}")
                    self.logger.error(f"  {e}")
                    continue

        # Organize by order
        coeffs_by_order = {}

        for index, values in all_coefficients.items():
            # Calculate order (sum of digits in index)
            order = sum(int(d) for d in index)

            if max_order is not None and order > max_order:
                continue

            if order not in coeffs_by_order:
                coeffs_by_order[order] = {}

            coeffs_by_order[order][index] = values

        # Convert to appropriate data structures
        result = {}

        # Order 0 (constant term, usually zero)
        if 0 in coeffs_by_order:
            result[0] = np.array(coeffs_by_order[0]['000000'])

        # Order 1 (linear map as 6×6 matrix)
        if 1 in coeffs_by_order:
            linear_map = np.zeros((6, 6))
            index_to_col = {
                '100000': 0, '010000': 1, '001000': 2,
                '000100': 3, '000010': 4, '000001': 5
            }

            for index, col in index_to_col.items():
                if index in coeffs_by_order[1]:
                    linear_map[:, col] = coeffs_by_order[1][index]
                elif col == 4 or col == 5:
                    # Default longitudinal to identity
                    linear_map[col, col] = 1.0

            result[1] = linear_map

        # Order 2 and higher (as indexed dictionaries)
        for order in range(2, max(coeffs_by_order.keys()) + 1 if coeffs_by_order else 2):
            if order not in coeffs_by_order:
                continue

            order_dict = {}

            for index, values in coeffs_by_order[order].items():
                # Convert index to source coordinate tuple
                source_coords = self._index_to_coords(index)

                # Store coefficients for each target coordinate
                for target_coord in range(6):
                    if abs(values[target_coord]) > 1e-15:  # Skip negligible terms
                        key = (target_coord,) + source_coords
                        order_dict[key] = values[target_coord]

            result[order] = order_dict

        if self.debug:
            self.logger.debug(f"\n{'=' * 60}")
            self.logger.debug("Transfer Map - All Orders")
            self.logger.debug(f"{'=' * 60}")
            for order in sorted(result.keys()):
                if order == 1:
                    self.logger.debug(
                        f"Order {order}: 6×6 matrix with {np.count_nonzero(result[order])} non-zero elements")
                elif order == 0:
                    self.logger.debug(f"Order {order}: constant term")
                else:
                    self.logger.debug(f"Order {order}: {len(result[order])} non-zero coefficients")
            self.logger.debug(f"{'=' * 60}\n")

        return result

    def _index_to_coords(self, index_str):
        """
        Convert 6-digit COSY index to source coordinate tuple.

        Example: '200000' → (0, 0) for x²
                 '110000' → (0, 1) for xx'
                 '300000' → (0, 0, 0) for x³

        Returns tuple of coordinate indices where each coordinate appears
        according to its power.
        """
        powers = [int(d) for d in index_str]
        coords = []

        for coord_idx, power in enumerate(powers):
            coords.extend([coord_idx] * power)

        return tuple(coords)

    def _coords_to_index(self, coord_tuple):
        """
        Convert coordinate tuple to 6-digit COSY index.

        Example: (0, 0) → '200000' for x²
                 (0, 1) → '110000' for xx'
                 (0, 0, 0) → '300000' for x³
        """
        powers = [0] * 6
        for coord in coord_tuple:
            powers[coord] += 1
        return ''.join(str(p) for p in powers)

    def get_aberration_coefficient(self, target_coord, source_coords, order=None):
        """
        Get specific aberration coefficient from transfer map.

        Parameters
        ----------
        target_coord : int or str
            Target coordinate (0-5 or 'x', 'a'/'xp', 'y', 'b'/'yp', 'l', 'delta')
        source_coords : tuple of int or str
            Source coordinates as tuple
            Example: (0, 0) for x² coefficient
            Example: (0, 1) for xa coefficient
            Example: (0, 0, 0) for x³ coefficient
        order : int, optional
            Expected order (for validation)

        Returns
        -------
        float
            Aberration coefficient value

        Notes
        -----
        COSY coordinates: [x, a, y, b, l, δK] where:
        - a = px/p0 (normalized transverse momentum, ≈ x' for paraxial beams)
        - b = py/p0 (normalized transverse momentum, ≈ y' for paraxial beams)
        - Aliases 'xp' and 'yp' are accepted for compatibility but map to a and b

        Examples
        --------
        >>> reader = COSYResultsReader()
        >>> # Get T_200000: x coefficient from x²
        >>> coeff = reader.get_aberration_coefficient('x', (0, 0))
        >>> # Get T_110000: x coefficient from xa (or x×x' in paraxial approx)
        >>> coeff = reader.get_aberration_coefficient('x', (0, 1))
        >>> coeff = reader.get_aberration_coefficient('x', ('x', 'a'))
        >>> # Get T_300000: x coefficient from x³
        >>> coeff = reader.get_aberration_coefficient('x', (0, 0, 0))
        """
        # Map coordinate names (accept both COSY notation and common aliases)
        coord_map = {
            'x': 0,
            'a': 1, 'xp': 1,  # a = px/p0, xp accepted as alias
            'y': 2,
            'b': 3, 'yp': 3,  # b = py/p0, yp accepted as alias
            'l': 4,
            'delta': 5, 'dk': 5
        }

        if isinstance(target_coord, str):
            if target_coord not in coord_map:
                raise ValueError(f"Unknown coordinate: {target_coord}")
            target_idx = coord_map[target_coord]
        else:
            target_idx = int(target_coord)

        # Convert source coord names to indices
        source_tuple = tuple(
            coord_map[c] if isinstance(c, str) else int(c)
            for c in source_coords
        )

        # Determine order
        calc_order = len(source_tuple)
        if order is not None and calc_order != order:
            raise ValueError(f"Calculated order {calc_order} != specified order {order}")

        # Read map
        maps = self.read_transfer_map_all_orders(max_order=calc_order)

        if calc_order not in maps:
            raise ValueError(f"Order {calc_order} not found in transfer map")

        # Look up coefficient
        if calc_order == 1:
            # Linear case: extract from matrix
            return float(maps[1][target_idx, source_tuple[0]])
        else:
            # Higher order: look up in dict
            key = (target_idx,) + source_tuple
            if key not in maps[calc_order]:
                return 0.0  # Coefficient not present (zero)
            return float(maps[calc_order][key])

    def get_aberration_from_powers(self, target_coord, power_list):
        """
        Get aberration coefficient using power notation.

        Convenience method that accepts powers [i, j, k, l, m, n] representing
        coordinate powers and converts to source coordinate tuple format.

        Parameters
        ----------
        target_coord : int or str
            Target coordinate (0-5 or 'x', 'a'/'xp', 'y', 'b'/'yp', 'l', 'delta')
        power_list : list or tuple of 6 ints
            Powers of [x, a, y, b, l, δK] coordinates
            Example: [2, 0, 0, 0, 0, 0] for x²
            Example: [1, 1, 0, 0, 0, 0] for xa (x×a, or x×x' in paraxial approx)
            Example: [3, 0, 0, 0, 0, 0] for x³

        Returns
        -------
        float
            Aberration coefficient value

        Notes
        -----
        COSY uses [x, a, y, b, l, δK] where a = px/p0 and b = py/p0.
        For paraxial beams, a ≈ x' and b ≈ y', but the distinction matters
        for large angles or higher-order momentum-dependent aberrations.

        Examples
        --------
        >>> reader = COSYResultsReader()
        >>> # Get T_200000: x from x²
        >>> coeff = reader.get_aberration_from_powers('x', [2, 0, 0, 0, 0, 0])
        >>> # Get T_110000: x from xa
        >>> coeff = reader.get_aberration_from_powers('x', [1, 1, 0, 0, 0, 0])
        >>> # Get T_300000: x from x³
        >>> coeff = reader.get_aberration_from_powers('x', [3, 0, 0, 0, 0, 0])
        """
        if len(power_list) != 6:
            raise ValueError(f"power_list must have 6 elements, got {len(power_list)}")

        # Convert powers to source coordinate tuple
        # [2, 0, 0, 0, 0, 0] → (0, 0) for x²
        # [1, 1, 0, 0, 0, 0] → (0, 1) for xa
        # [3, 0, 0, 0, 0, 0] → (0, 0, 0) for x³
        source_coords = []
        for coord_idx, power in enumerate(power_list):
            source_coords.extend([coord_idx] * int(power))

        return self.get_aberration_coefficient(target_coord, tuple(source_coords))

    def read_json_results(self, filename='result.txt'):
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