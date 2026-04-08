import json
import pandas as pd
import re
import os
from functools import wraps
from collections.abc import Iterable
from excelElements import ExcelElements
from loggingConfig import get_logger_with_fallback


class BeamlineBuilder:
    def __init__(self, excel_path, json_config_path=None, debug=None):
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Excel beamline file not found: {excel_path}")

        self.excel_path = excel_path
        self.json_config_path = json_config_path
        self.beamline = []  # List of dicts
        self.logger, self.debug = get_logger_with_fallback(__name__, debug)

    def parse_beamline(self):
        excel = ExcelElements(self.excel_path)
        df = excel.get_dataframe()
        prev_z_end = 0
        for _, row in df.iterrows():
            z_start = row['z_start']
            z_end = row['z_end']
            if pd.notna(z_start) and pd.notna(z_end) and z_start < z_end:
                if z_start > prev_z_end:
                    self.beamline.append({"type": "DRIFT", "length": z_start - prev_z_end})

                # Element name: prefer Label, fall back to Nomenclature
                label = row.get('Label')
                if pd.isna(label) or (isinstance(label, str) and not label.strip()):
                    label = row.get('Nomenclature')
                if pd.isna(label) or (isinstance(label, str) and not label.strip()):
                    label = None
                else:
                    label = str(label).strip()

                element = {
                    "type": row['Element'],
                    "name": label,
                    "length": z_end - z_start,
                    "current": row.get('Current (A)', 0),
                    "angle": row.get('Dipole Angle (deg)', 0),
                    "wedge_angle": row.get('Dipole wedge (deg)', 0),
                    "gap_wedge": row.get('Gap wedge (m)', 0),
                    "pole_gap": row.get('Pole gap (m)', 0),
                    "enge_fct": row.get('Fringe Field Enge coefficients', ""),
                    "z_start": z_start,
                    "z_end": z_end
                }
                self.beamline.append(element)
                prev_z_end = z_end
            else:
                pass
        self._apply_overrides()
        return self.beamline

    def _apply_overrides(self):
        if self.json_config_path:
            with open(self.json_config_path, 'r') as f:
                overrides = json.load(f).get('overrides', {})
            for idx, mods in overrides.items():
                if 0 <= int(idx) < len(self.beamline):
                    self.beamline[int(idx)].update(mods)
                else:
                    self.logger.warning(f"Override index {idx} out of range")

    def handle_multiple_indices(func):
        """
        Decorator allowing functions to accept either a single index or list of indices.
        Applies the function to each index and aggregates results/errors.
        """

        @wraps(func)
        def wrapper(self, index, *args, **kwargs):
            if isinstance(index, Iterable) and not isinstance(index, str):
                results = []
                errors = []

                for idx in index:
                    try:
                        result = func(self, idx, *args, **kwargs)
                        results.append(result)
                    except Exception as e:
                        error_msg = f"Error at index {idx}: {e}"
                        errors.append(error_msg)
                        self.logger.warning(f"{error_msg}")

                if errors and len(errors) == len(index):
                    raise ValueError(f"All operations failed. Errors: {'; '.join(errors)}")

                return results
            else:
                return func(self, index, *args, **kwargs)

        return wrapper

    @handle_multiple_indices
    def modify_element(self, index, **kwargs):
        """
        Modify parameters of beamline element(s).

        Parameters:
            index: int or list[int] - Element index/indices
            **kwargs: Parameters to modify (numeric or symbolic string values)

        Returns:
            Modified element dict or list of dicts
        """
        if not isinstance(index, int):
            raise TypeError(f"Index must be an integer, got {type(index)}")

        if not (0 <= index < len(self.beamline)):
            raise IndexError(
                f"Element index {index} out of range. Beamline has {len(self.beamline)} elements (indices 0-{len(self.beamline) - 1})")

        if not kwargs:
            raise ValueError("No parameters provided to modify")

        element = self.beamline[index]
        original_values = {}

        for key in kwargs:
            if key in element:
                original_values[key] = element[key]
            else:
                original_values[key] = None

        for key, value in kwargs.items():
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"String parameter '{key}' cannot be empty or whitespace only")

        # Apply modifications
        for key, value in kwargs.items():
            element[key] = value

        # Log changes
        changes = []
        for key, new_value in kwargs.items():
            old_value = original_values[key]
            if old_value != new_value:
                if isinstance(new_value, str):
                    changes.append(f"{key}: {old_value} → '{new_value}' (symbolic)")
                else:
                    changes.append(f"{key}: {old_value} → {new_value}")

        if changes:
            self.logger.debug(f"Modified element {index} ({element.get('type', 'UNKNOWN')}): {', '.join(changes)}")

        return element

    @handle_multiple_indices
    def set_current(self, index, current):
        """
        Set current for quadrupole element(s).

        Parameters:
            index: int or list[int] - Quadrupole element index/indices
            current: float or str - Current in Amperes or symbolic variable name

        Returns:
            Modified element dict or list of dicts
        """
        if not isinstance(index, int):
            raise TypeError(f"Index must be an integer, got {type(index)}")

        if not (0 <= index < len(self.beamline)):
            raise IndexError(
                f"Element index {index} out of range. Beamline has {len(self.beamline)} elements (indices 0-{len(self.beamline) - 1})")

        if not isinstance(current, (int, float, str)):
            raise TypeError(f"Current must be a number or string, got {type(current)}")

        if isinstance(current, str):
            if not current.strip():
                raise ValueError("String current cannot be empty or whitespace only")

        element = self.beamline[index]
        element_type = element.get('type', 'UNKNOWN')

        if element_type not in ['QPF', 'QPD']:
            raise ValueError(f"Cannot set current for element type '{element_type}' at index {index}. "
                             f"Current can only be set for quadrupole elements (QPF, QPD)")

        return self.modify_element(index, current=current)

    def apply_variable_mapping(self, xVar, validation=True):
        """
        Apply variable mappings to beamline elements from xVar-style dictionary.

        Parameters:
            xVar: dict - Maps element indices to parameter modifications
                  Format: {index: {parameter_name: value, ...}, ...}
            validation: bool - Whether to validate currents are only set on quadrupoles
        """
        if not isinstance(xVar, dict):
            raise TypeError(f"xVar must be a dictionary, got {type(xVar)}")

        for index, modifications in xVar.items():
            if not isinstance(index, int):
                raise TypeError(f"Index {index} must be an integer")
            if not isinstance(modifications, dict) or not modifications:
                raise ValueError(f"Invalid modifications for index {index}")

            if validation and "current" in modifications:
                element = self.beamline[index]
                element_type = element.get('type', 'UNKNOWN')
                if element_type not in ['QPF', 'QPD']:
                    raise ValueError(f"Cannot set current for element type '{element_type}' at index {index}")

            self.modify_element(index, **modifications)

    def find_elements(self, element_type=None, **criteria):
        """
        Find beamline element indices matching specified criteria.

        Parameters:
            element_type: str, optional - Element type (e.g., 'QPF', 'QPD', 'DRIFT')
            **criteria: Additional matching criteria (e.g., current=2.5, length=0.1)

        Returns:
            list[int] - Matching element indices
        """
        matching_indices = []

        for idx, element in enumerate(self.beamline):
            if element_type is not None and element.get('type') != element_type:
                continue

            match = True
            for key, value in criteria.items():
                if element.get(key) != value:
                    match = False
                    break

            if match:
                matching_indices.append(idx)

        return matching_indices

    def generate_input(self):
        raise NotImplementedError("Subclasses must implement input generation")

    def print_beamline(self):
        """
        Print beamline elements as a colour-coded table.
        """
        if not self.beamline:
            print("No beamline elements loaded. Call parse_beamline() first.")
            return

        headers = ["Index", "Type", "Length", "Current", "Angle", "Z Start", "Z End"]
        col_widths = [6, 10, 10, 10, 10, 10, 10]

        type_colors = {
            "DRIFT": "\033[96m",  # Cyan
            "QPF": "\033[92m",  # Green
            "QPD": "\033[93m",  # Yellow
            "DPH": "\033[91m",  # Red
            "DPW": "\033[95m",  # Magenta
        }
        reset_color = "\033[0m"

        # Calculate column widths
        data_rows = []
        for idx, elem in enumerate(self.beamline):
            row = [
                str(idx),
                elem.get("type", "N/A"),
                f"{elem.get('length', 'N/A'):.4f}" if isinstance(elem.get('length'), float) else str(
                    elem.get('length', 'N/A')),
                f"{elem.get('current', 'N/A'):.4f}" if isinstance(elem.get('current'), float) else str(
                    elem.get('current', 'N/A')),
                f"{elem.get('angle', 'N/A'):.4f}" if isinstance(elem.get('angle'), float) else str(
                    elem.get('angle', 'N/A')),
                f"{elem.get('z_start', 'N/A'):.4f}" if isinstance(elem.get('z_start'), float) else str(
                    elem.get('z_start', 'N/A')),
                f"{elem.get('z_end', 'N/A'):.4f}" if isinstance(elem.get('z_end'), float) else str(
                    elem.get('z_end', 'N/A'))
            ]
            data_rows.append(row)
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(val))

        def strip_ansi(s):
            return re.sub(r'\033\[[0-9;]*m', '', s)

        def visible_len(s):
            return len(strip_ansi(s))

        def _center_with_ansi(val, width):
            visible = visible_len(val)
            if visible >= width:
                return val
            padding = width - visible
            left_pad = padding // 2
            right_pad = padding - left_pad
            return ' ' * left_pad + val + ' ' * right_pad

        # Print header
        print("Beamline Elements:")
        sep_length = sum(col_widths) + 3 * (len(headers) - 1)
        print("-" * sep_length)
        header_str = " | ".join(_center_with_ansi(header, width) for header, width in zip(headers, col_widths))
        print(header_str)
        print("-" * sep_length)

        # Print rows with coloured types
        for row in data_rows:
            colored_type = type_colors.get(row[1], "") + row[1] + reset_color
            colored_row = row.copy()
            colored_row[1] = colored_type
            row_str = " | ".join(_center_with_ansi(val, width) for val, width in zip(colored_row, col_widths))
            print(row_str)

        print("-" * sep_length)