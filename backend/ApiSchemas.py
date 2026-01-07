from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

class AxisTwiss(BaseModel):
    alpha: float
    beta: float
    phi: float
    epsilon: float

class TwissParameters(BaseModel):
    x: AxisTwiss
    y: AxisTwiss
    z: AxisTwiss

class BeamlineInfo(BaseModel):
    #__root__: Dict[str, Dict[str, Any]]
    segmentName: str
    parameters: Dict[str, Any]

class PlottingParameters(BaseModel):
    beamlineData: list[BeamlineInfo]
    beamType: str = 'electron'
    num_particles: int
    kineticE: int = 45
    interval: float = 1
    defineLim: bool = True
    saveData: bool = False
    matchScaling: bool = True
    scatter: bool = True
    beam_setup: str = 'twiss'
    twiss: TwissParameters = None
    #  I THINK WE NEED SAVE FIG AND SHAPE

class LineAxObject(BaseModel):
    axis: str # temporary placeholder axes
    twiss: str
    x_axis: list[float]
    beamsegment: list

class AxesPNGData(BaseModel):
    images: Dict[float, Any]
    line_graph: LineAxObject

class GraphParameters(BaseModel):
    beam_index: int
    target_parameter: str
    target_s_pos: float
    beamline_data: list[BeamlineInfo]
    min: int | float = 0
    max: int | float = 10
    custom_step: int | float = 1

class GraphPlotPointResponse(BaseModel):
    x: float | None
    y: float | None
    z: float | None
    twiss_parameter: str

class GraphPlotData(BaseModel):
    parameter_value: float
    data: List[GraphPlotPointResponse]

class ExcelBeamlineElement(BaseModel):
    Nomenclature: Optional[str] = Field(None, alias=' Nomenclature', description="Element nomenclature")
    z_start_m: Optional[float] = Field(None, alias='z start (m)', description="Start position in meters")
    z_mid_m: Optional[float] = Field(None, alias='z mid (m)', description="Mid position in meters")
    z_end_m: Optional[float] = Field(None, alias='z end (m)', description="End position in meters")
    Current_A: Optional[float] = Field(None, alias='Current A)', description="Current in Amperes")
    dipole_angle_deg: Optional[float] = Field(None, alias='Dipole Angle (deg)', description="Dipole angle in degrees")
    dipole_length_m: Optional[float] = Field(None, alias='Dipole length (m)', description="Dipole length in meters")
    dipole_wedge_deg: Optional[float] = Field(None, alias='Dipole wedge (deg)', description="Dipole wedge angle in degrees")
    gap_wedge_m: Optional[float] = Field(None, alias='Gap wedge (m)', description="Gap wedge length in n")
    Pole_gap_m: Optional[float] = Field(None, alias='Pole gap (m)', description="Pole gap in meters")
    fringe_field_coefficient: Optional[str] = Field(None, alias='Fringe Field Enge coefficients', description="Fringe field coefficient")
    Element_name: Optional[str] = Field(None, alias='Element name', description="Name of the element")
    Channel_num: Optional[int] = Field(None, alias='Channel #', description="Channel number")
    label: Optional[str] = Field(None, alias='Label', description="Element label")
    Sector: Optional[str] = Field(None, alias='Sector', description="Sector name")
    Element: Optional[str] = Field(None, alias='Element', description="Element type")


    class Config:
        validate_by_name = True
        populate_by_name = True
        json_schema_extra = {
            "example": {
                " Nomenclature": "LIN.QPF.004",
                "z start (m)": 0.358775,
                "z mid (m)": 0.403225,
                "z end (m)": 0.447675,
                "Current A)": 0.8857,
                "Pole gap (m)": 0.027,
                "Element name": "Quad",
                "Channel #": 20,
                "Sector": "LIN",
                "Element": "QPF"
            }
        }