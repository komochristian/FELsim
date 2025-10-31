import pandas as pd
import numpy as np
import os
from typing import Optional, List
from beamline import (driftLattice, qpfLattice, qpdLattice, 
                     dipole, dipole_wedge)


class ExcelElements:
    def __init__(self, file_path):
        """
        Initialise the ExcelElements class and load the beamline data.
        
        :param file_path: Path to Excel file or dictionary containing beamline information.
        """
        self.df = pd.DataFrame()
        self.file_path = file_path
        
        # Define standard column names
        self.COLUMNS = ['Nomenclature', 'z_start', 'z_mid', 'z_end', 'Current (A)', 
                       'Dipole Angle (deg)', 'Dipole length (m)', 'Dipole wedge (deg)', 
                       'Gap wedge (m)', 'Pole gap (m)', 'Fringe Field Enge coefficients',
                       'Element name', 'Channel', 'Label', 'Sector', 'Element']
        
        # Legacy column names for backward compatibility
        OLDCOLUMNS = ['Nomenclature', 'z start (m)', 'z mid (m)', 'z end (m)', 'Current A)',
                     'Dipole Angle (deg)', 'Dipole length (m)', 'Dipole wedge (deg)',
                     'Gap wedge (m)', 'Pole gap (m)', 'Fringe Field Enge coefficients', 
                     'Element name', 'Channel #', 'Label', 'Sector', 'Element']
        
        # Column name mapping
        self.columnReplaceHandler = {}
        for i in range(len(OLDCOLUMNS)):
            self.columnReplaceHandler.update({OLDCOLUMNS[i]: self.COLUMNS[i]})
        
        # Try loading as Excel first, fall back to dictionary format
        try:
            self.load_excel_lattice(file_path)
        except:
            self.load_dictionary_lattice(file_path)
    
    def load_dictionary_lattice(self, beamlineJson):
        """
        Load beamline from dictionary or JSON format.
        
        :param beamlineJson: Dictionary containing beamline data.
        """
        self.df = pd.DataFrame(beamlineJson)
        self.df.rename(columns=self.columnReplaceHandler, inplace=True)
        self.df['Channel'] = pd.to_numeric(self.df['Channel'], errors='coerce')
    
    def load_excel_lattice(self, file_path: str):
        """
        Load the lattice from an Excel file and store it in a DataFrame.
        
        :param file_path: Path to the Excel file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Excel file does not exist: {file_path}")
        
        try:
            df = pd.read_excel(file_path, header=None, skiprows=1)
        except Exception as e:
            raise ValueError(f"Failed to load Excel file '{file_path}': {str(e)}") from e
        
        df.columns = self.COLUMNS
        df['Channel'] = pd.to_numeric(df['Channel'], errors='coerce')
        self.df = df
    
    def create_beamline(self) -> List:
        """
        Create the beamline using Python-based lattice elements.
        
        :return: List of beamline elements (drifts, quadrupoles, dipoles).
        """
        beamline = []
        prev_z_end = 0.0
        
        for index, row in self.df.iterrows():
            element = row['Element']
            z_sta = row['z_start']
            z_end = row['z_end']
            
            # Extract parameters
            current = float(row['Current (A)']) if pd.notna(row['Current (A)']) else 0.0
            angle = float(row['Dipole Angle (deg)']) if pd.notna(row['Dipole Angle (deg)']) else 0.0
            curvature = float(row['Dipole length (m)']) if pd.notna(row['Dipole length (m)']) else 0.0
            angle_wedge = float(row['Dipole wedge (deg)']) if pd.notna(row['Dipole wedge (deg)']) else 0.0
            gap_wedge = float(row['Gap wedge (m)']) if pd.notna(row['Gap wedge (m)']) else 0.0
            pole_gap = float(row['Pole gap (m)']) if pd.notna(row['Pole gap (m)']) else 0.0
            
            if pd.notna(row['Fringe Field Enge coefficients']) and row['Fringe Field Enge coefficients'].strip():
                enge_fct = [float(val.strip()) for val in row['Fringe Field Enge coefficients'].split(',') 
                           if val.strip()]
            else:
                enge_fct = []
            
            # Add drift if gap exists
            if z_sta > prev_z_end:
                drift_length = z_sta - prev_z_end
                beamline.append(driftLattice(drift_length))
            
            # Add beamline element
            if element == "QPF":
                beamline.append(qpfLattice(current=current, length=(z_end - z_sta)))
            elif element == "QPD":
                beamline.append(qpdLattice(current=current, length=(z_end - z_sta)))
            elif element == "DPH":
                beamline.append(dipole(length=curvature, angle=angle))
            elif element == "DPW":
                beamline.append(dipole_wedge(length=gap_wedge, angle=angle_wedge, 
                                            dipole_length=curvature, dipole_angle=angle, 
                                            pole_gap=pole_gap, enge_fct=enge_fct))
            else:
                # Generic drift for undefined elements
                if (not z_end - z_sta == 0) and (not np.isnan(z_sta)) and (not np.isnan(z_end)):
                    beamline.append(driftLattice(z_end - z_sta))
            
            if not np.isnan(z_end):
                prev_z_end = z_end
        
        return beamline
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Return the DataFrame containing the beamline elements.
        
        :return: DataFrame with the loaded Excel data.
        """
        return self.df
    
    def find_element_by_position(self, z: float) -> Optional[str]:
        """
        Find the beamline element at a given z position.
        
        :param z: Longitudinal position to query.
        :return: Element type at position z, or None if out of range.
        """
        for index, row in self.df.iterrows():
            z_start = row['z_start']
            z_end = row['z_end']
            
            if pd.notna(z_start) and pd.notna(z_end):
                if z_start <= z <= z_end:
                    return row['Element']
        
        return None
    
    def __str__(self):
        return f"Beamline: {len(self.df)} elements"