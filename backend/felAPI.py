from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from ebeam import beam
from beamline import *
from schematic import *
from fastapi.middleware.cors import CORSMiddleware
import inspect
import importlib
import io
import base64
from excelElements import ExcelElements
import uvicorn
import os
from dotenv import load_dotenv
import copy
import math
import numpy as np

load_dotenv('../.env')  # Only during dev testing when not using Dockerfile...
FRONTEND_PORT = os.getenv('FRONTEND_PORT')

ORIGINS = [f'http://localhost:{FRONTEND_PORT}', f"localhost:{FRONTEND_PORT}"]
#ORIGINS = ["http://localhost:5173", "localhost:5173"]
moduleName = 'beamline'

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
    domain_range: int | float = 10
    custom_step: int | float = 1

class GraphPlotPointResponse(BaseModel):
    x: float | None
    y: float | None
    z: float | None
    twiss_parameter: str

class GraphPlotData(BaseModel):
    parameter_value: float
    data: List[GraphPlotPointResponse]

app = FastAPI()
ebeam = beam()
# Allow requests from your frontend (CORS!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,  # In production, use your frontend's exact origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def getPngObjFromBeamList(beamlist, plotParams: PlottingParameters):
    beam_dist = ebeam.gen_6d_gaussian(0,[1,1,1,1,0.1,100], plotParams.num_particles) 
    schem = draw_beamline()
    axList, lineAxObj = schem.plotBeamPositionTransform(beam_dist, beamlist, plot=False, apiCall=True, scatter=True, interval=plotParams.interval)
    fig = lineAxObj['axis'].figure
    buf = io.BytesIO()
    fig.savefig(buf, format="png",bbox_inches="tight")
    buf.seek(0)
    lineAx_img = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()

    images = {}
    for index, axes in axList.items():

        fig = axes.figure
        buf = io.BytesIO()
        fig.savefig(buf, format="png",bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()

        images.update({index: img_base64})
     
    lineAxObj['axis'] = lineAx_img
    print(lineAxObj['twiss'])
    lineAxObj['twiss'] = lineAxObj['twiss'].to_json()
    beamsegmentJson = []
    #for segment in lineAxObj['beamsegment']:
    #    beamsegmentJson.append(segment.__dict__)
    lineAxObj['beamsegment'] = beamsegmentJson
    #print(beamsegmentJson)

    lineAxObj = LineAxObject(**lineAxObj)
    pngObject = AxesPNGData(**{'images': images, 'line_graph': lineAxObj})
    return pngObject

def beamlineToJson():
    pass

@app.get("/")
def root():
    return {"Hello" : "World!"}

@app.post("/get-dist")
def gen_beam(particle_num : int): 
    beam_dist = ebeam.gen_6d_gaussian(0,[1,1,1,1,0.1,100], particle_num).tolist()
    return beam_dist

@app.post("/excel-to-beamline")
def excelToBeamline(excelJson: list[Dict[str, Any]]) -> list[dict[str, dict[str, Any]]]:
    try:
        excelHandler = ExcelElements(excelJson)
        beamlist = excelHandler.create_beamline()

        jsonBeamlist = []

        for segment in beamlist:
            clas = segment.__class__
            className = clas.__name__
            classSig= inspect.signature(clas.__init__)

            paramsDict = {}
            for name, param in classSig.parameters.items():
                if name == "self":
                    continue
                paramVal = getattr(segment, name, None)
                paramsDict.update({name: paramVal})
                    
            jsonBeamlist.append({className: paramsDict})

        return jsonBeamlist
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/axes")
def loadAxes(plotParams: PlottingParameters) -> AxesPNGData:
    try:
        beamline = importlib.import_module("beamline")
        beamlist = []
        beamlineData = plotParams.beamlineData
        for segment in beamlineData:
            if hasattr(beamline, segment.segmentName):
                segmentClass = getattr(beamline, segment.segmentName)
                beamlist.append(segmentClass(**segment.parameters))
    
        latObj = lattice(1)
        beamlist = latObj.changeBeamType(plotParams.beamType, plotParams.kineticE, beamlist)
    
        pngObject = getPngObjFromBeamList(beamlist, plotParams)
        return pngObject
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/beamsegmentinfo")
def getBeamSegmentInfo():
    module = importlib.import_module(moduleName)
    classes = inspect.getmembers(module, inspect.isclass)
    classes_in_module = [cls for name, cls in classes if cls.__module__ == moduleName and cls.__name__ not in ["Beamline", "lattice"]]
    beamSegInfo = {}

    for cls in classes_in_module:
        sig = inspect.signature(cls.__init__)
        params_info = {}
    
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            default = (
                param.default
                if param.default is not inspect.Parameter.empty
                else 1  # or some other marker
            )
            params_info[name] = default

        params_info['color'] = cls.color  # Manually add class info about beam's color
    
        beamSegInfo[cls.__name__] = params_info

    return beamSegInfo

@app.post("/plot-parameters")
def plot_parameters(graphParams: GraphParameters) -> List[GraphPlotData]:
    LABELMAPPING = {
        r'$\epsilon$ ($\pi$.mm.mrad)': 'emittance',
        r'$\alpha$': 'alpha',
        r'$\beta$ (m)': 'beta',
        r'$\gamma$ (rad/m)': 'gamma',
        r'$D$ (mm)': 'dispersion',
        r'$D^{\prime}$ (mrad)': 'dispersion_prime',
        r'$\phi$ (deg)': 'angle',
        r'Envelope $E$ (mm)': 'envelope'
    }
    try:
        beamline = importlib.import_module(moduleName)
        beamlist = []
        beamlineData = graphParams.beamline_data
        for segment in beamlineData:
            if hasattr(beamline, segment.segmentName):
                segmentClass = getattr(beamline, segment.segmentName)
                beamlist.append(segmentClass(**segment.parameters))

        cleanedBeamlist = beamlist[:graphParams.beam_index]

        schem = draw_beamline()
        ebeam = beam()
        beam_dist = ebeam.gen_6d_gaussian(0,[1,1,1,1,0.1,100], 1000)

        # print("Plotting initial beamline up to segment", cleanedBeamlist)

        #  100 chosen as a large number to speed up initial calculation
        schem.plotBeamPositionTransform(beam_dist, cleanedBeamlist, plot=False, interval=100, rendering=False)
        beam_dist = schem.matrixVariables

        beamObj = Beamline(beamlist)
        indexOfSSegment = beamObj.findSegmentAtPos(graphParams.target_s_pos)

        newSegment = copy.deepcopy(beamObj.beamline[indexOfSSegment])
        newSegment.length = graphParams.target_s_pos - beamObj.beamline[indexOfSSegment - 1].endPos
        optimized_beamlist = beamObj.beamline[graphParams.beam_index:indexOfSSegment]
        optimized_beamlist.append(newSegment)

        # for i in optimized_beamlist:
        #     print("Printing segment:", i) 

        plotInfo = [] 
        domain_range = np.arange(0, graphParams.domain_range, graphParams.custom_step).tolist()
        if graphParams.domain_range not in domain_range: domain_range.append(graphParams.domain_range)

        for i in domain_range:
            setattr(optimized_beamlist[0], graphParams.target_parameter, i)
            twiss = schem.plotBeamPositionTransform(beam_dist, optimized_beamlist, plot=False, interval=100, rendering=False)
            # col = LABELMAPPING.get(graphParams.twiss_target, graphParams.twiss_target)
            plotDict = {f'parameter_value': i, 
                         'data': [
                                    {
                                        **{name: None if axis[-1] is None or math.isnan(axis[-1]) or math.isinf(axis[-1]) else axis[-1]
                                        for name, axis in twiss[col].items()},
                                        'twiss_parameter': LABELMAPPING.get(col, col)
                                    }
                                    for col in twiss.columns
                                 ]
                        }
            # print(plotDict)
            plotInfo.append(plotDict)

        # RETURN ALL TWISS, MAKE USER SELECT WHICH ONE
        return plotInfo
    
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=str(e))
    
# Don't use, doesn't check for changes and server reloads
#if __name__ == "__main__":
    #uvicorn.run(app, host="127.0.0.1", port=8000)
